#!/usr/bin/env python3
"""
Stage 1: Classify FineWeb-Edu documents with topic + complexity models.

Reads documents from local parquet files (preferred) or streams from
HuggingFace, runs both classifiers, and writes parquet shards with the
full 17-dim sigmoid topic vector and complexity score.

Features:
  - Local parquet reading with pyarrow predicate pushdown for CC dump filtering
  - Fallback HF streaming mode (--dataset / --config flags, no --local-dir)
  - Parquet shard output (1M docs per shard by default)
  - Shard-based checkpointing and resume
  - Length-sorted dynamic padding for minimal padding waste
  - AMP (bf16/fp16) + optional torch.compile
  - CUDA stream double-buffering for async H2D transfers
  - Background tokenization prefetching
  - Throughput logging with ETA

Usage (local — preferred):
  python pipeline_classify.py --local-dir /workspace/fineweb-curation/raw_data --compile
  python pipeline_classify.py --local-dir /workspace/fineweb-curation/raw_data --cc-dumps CC-MAIN-2024-10

Usage (streaming fallback):
  python pipeline_classify.py --output-dir scored_shards/ --compile

Resume from interruption (automatic):
  python pipeline_classify.py --local-dir /workspace/fineweb-curation/raw_data --compile
  (skips already-completed shards)
"""

import argparse
import contextlib
import json
import logging
import time
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPLEXITY_MODEL,
    DEFAULT_MAX_LENGTH,
    DEFAULT_OUTPUT_BASE,
    DEFAULT_SHARD_SIZE,
    DEFAULT_TOPIC_MODEL,
    NUM_LABELS,
    RECOMMENDED_CC_DUMPS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parquet schema for output shards
# ---------------------------------------------------------------------------

SHARD_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("url", pa.string()),
    pa.field("token_count", pa.int64()),
    pa.field("dump", pa.string()),
    pa.field("topic_scores", pa.list_(pa.float32(), NUM_LABELS)),
    pa.field("complexity", pa.float32()),
])

# ---------------------------------------------------------------------------
# Tokenization + batch preparation
# (adapted from categorize_fineweb_edu_bert.py)
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 250000  # 117GB RAM: ~5GB per chunk of tokenized tensors
DEFAULT_PREFETCH_CHUNKS = 2
DEFAULT_READ_BATCH_SIZE = 8192
DEFAULT_WRITE_QUEUE_SIZE = 2
DEFAULT_COLLECT_QUEUE_SIZE = 1


@dataclass
class Stage1Metrics:
    docs: int = 0
    batches: int = 0
    parquet_read_s: float = 0.0
    parquet_convert_s: float = 0.0
    tokenization_s: float = 0.0
    h2d_s: float = 0.0
    topic_forward_s: float = 0.0
    complexity_forward_s: float = 0.0
    d2h_s: float = 0.0
    write_block_s: float = 0.0

    def merge(self, other: "Stage1Metrics"):
        self.docs += other.docs
        self.batches += other.batches
        self.parquet_read_s += other.parquet_read_s
        self.parquet_convert_s += other.parquet_convert_s
        self.tokenization_s += other.tokenization_s
        self.h2d_s += other.h2d_s
        self.topic_forward_s += other.topic_forward_s
        self.complexity_forward_s += other.complexity_forward_s
        self.d2h_s += other.d2h_s
        self.write_block_s += other.write_block_s

    def to_dict(self):
        base = asdict(self)
        measured_total = (
            self.parquet_read_s
            + self.parquet_convert_s
            + self.tokenization_s
            + self.h2d_s
            + self.topic_forward_s
            + self.complexity_forward_s
            + self.d2h_s
            + self.write_block_s
        )
        if measured_total > 0:
            base["pct"] = {
                "parquet_read": round(self.parquet_read_s * 100.0 / measured_total, 2),
                "parquet_convert": round(self.parquet_convert_s * 100.0 / measured_total, 2),
                "tokenization": round(self.tokenization_s * 100.0 / measured_total, 2),
                "h2d": round(self.h2d_s * 100.0 / measured_total, 2),
                "topic_forward": round(self.topic_forward_s * 100.0 / measured_total, 2),
                "complexity_forward": round(self.complexity_forward_s * 100.0 / measured_total, 2),
                "d2h": round(self.d2h_s * 100.0 / measured_total, 2),
                "write_block": round(self.write_block_s * 100.0 / measured_total, 2),
            }
        return base


def prepare_batches(
    texts,
    tokenizer,
    max_length,
    batch_size,
    pin_memory=False,
    static_padding=False,
):
    """Tokenize texts, sort by length, return padded batch tensors.

    Returns list of (batch_positions, input_ids, attention_mask) tuples,
    where batch_positions maps back to the original indices in `texts`.
    """
    t0 = time.perf_counter()
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_length=True,
    )
    ids_list = encodings["input_ids"]
    lengths = encodings["length"]

    order = sorted(range(len(ids_list)), key=lambda i: lengths[i])

    batches = []
    for start in range(0, len(order), batch_size):
        batch_order = order[start : start + batch_size]
        batch_ids = [ids_list[i] for i in batch_order]
        if static_padding:
            padded = tokenizer.pad(
                {"input_ids": batch_ids},
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            padded = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(torch.long)
        attention_mask = padded["attention_mask"].to(torch.long)
        if pin_memory:
            input_ids = input_ids.pin_memory()
            attention_mask = attention_mask.pin_memory()
        batches.append((batch_order, input_ids, attention_mask))

    return batches, time.perf_counter() - t0


def iter_chunks_with_prefetch(texts, tokenizer, max_length, batch_size,
                              chunk_size, pin_memory, prefetch, static_padding=False):
    """Yield (chunk_idx, num_chunks, batches) with background tokenization."""
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size

    if prefetch <= 0:
        for ci in range(num_chunks):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(texts))
            batches, tokenize_s = prepare_batches(
                texts[s:e],
                tokenizer,
                max_length,
                batch_size,
                pin_memory,
                static_padding,
            )
            yield ci, num_chunks, batches, s, tokenize_s
        return

    queue: Queue = Queue(maxsize=max(1, prefetch))
    sentinel = object()

    def _producer():
        try:
            for ci in range(num_chunks):
                s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(texts))
                batches, tokenize_s = prepare_batches(
                    texts[s:e],
                    tokenizer,
                    max_length,
                    batch_size,
                    pin_memory,
                    static_padding,
                )
                queue.put((ci, num_chunks, batches, s, tokenize_s))
        except Exception as exc:
            queue.put(exc)
        finally:
            queue.put(sentinel)

    Thread(target=_producer, daemon=True).start()

    while True:
        item = queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ---------------------------------------------------------------------------
# Model loading + AMP
# ---------------------------------------------------------------------------

def load_models(args, device):
    """Load topic + complexity models and tokenizer."""
    print(f"Loading topic model: {args.topic_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.topic_model)
    topic_model = AutoModelForSequenceClassification.from_pretrained(args.topic_model)
    topic_model.to(device).eval()

    print(f"Loading complexity model: {args.complexity_model}")
    complexity_model = AutoModelForSequenceClassification.from_pretrained(args.complexity_model)
    complexity_model.to(device).eval()

    if args.compile and hasattr(torch, "compile"):
        cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
        mode = "max-autotune" if cc >= (9, 0) else "reduce-overhead"
        print(f"Compiling models (mode={mode})...")
        topic_model = torch.compile(topic_model, mode=mode)
        complexity_model = torch.compile(complexity_model, mode=mode)

    return tokenizer, topic_model, complexity_model


def get_amp_context(device, force_bf16=False):
    if device.type == "cuda":
        if force_bf16 and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"AMP enabled (dtype={dtype})")
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


def warmup_models(topic_model, complexity_model, device, amp_ctx, max_length, batch_size, steps=8):
    if device.type != "cuda" or steps <= 0:
        return
    seq_len = min(max_length, 256)
    input_ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(steps):
            with amp_ctx:
                _ = topic_model(input_ids=input_ids, attention_mask=attention_mask).logits
                _ = complexity_model(input_ids=input_ids, attention_mask=attention_mask).logits
    torch.cuda.synchronize(device)


# ---------------------------------------------------------------------------
# Shard writing
# ---------------------------------------------------------------------------

def shard_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / f"shard_{shard_idx:04d}.parquet"


def count_existing_shards(output_dir: Path) -> int:
    """Count contiguous completed shards for resume."""
    idx = 0
    while shard_path(output_dir, idx).exists():
        idx += 1
    return idx


def numpy_to_fixed_list_array(arr: np.ndarray, list_size: int) -> pa.FixedSizeListArray:
    """Convert (N, list_size) float32 numpy array to pyarrow FixedSizeListArray.

    Avoids the massive overhead of creating N*list_size Python float objects
    that the naive list-of-lists approach incurs (~17M objects per 1M-doc shard).
    """
    flat = pa.array(arr.ravel(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def write_shard(output_dir: Path, shard_idx: int, records: dict):
    """Write a shard to parquet."""
    table = pa.table(records, schema=SHARD_SCHEMA)
    pq.write_table(table, shard_path(output_dir, shard_idx))


def validate_shard(output_dir: Path, shard_idx: int, expected_rows: int) -> bool:
    """Quick validation: check row count matches."""
    path = shard_path(output_dir, shard_idx)
    if not path.exists():
        return False
    try:
        meta = pq.read_metadata(path)
        return meta.num_rows == expected_rows
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Local parquet reading (preferred over HF streaming)
# ---------------------------------------------------------------------------

def iter_local_parquets(local_dir, cc_dumps=None, batch_size=DEFAULT_READ_BATCH_SIZE, metrics=None):
    """Yield batches from local parquet files with predicate pushdown.

    When cc_dumps is provided, uses parquet predicate pushdown to skip
    entire row groups that don't contain the target dumps — orders of
    magnitude faster than the HF streaming .filter() approach.
    """
    files = sorted(Path(local_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {local_dir}")

    print(f"Found {len(files)} local parquet files in {local_dir}")
    if cc_dumps:
        print(f"Filtering to CC dumps: {cc_dumps} (predicate pushdown)")

    filters = [("dump", "in", cc_dumps)] if cc_dumps else None
    cols = ["text", "url", "token_count", "dump"]
    for f in files:
        t0_read = time.perf_counter()
        table = pq.read_table(f, columns=cols, filters=filters)
        if metrics is not None:
            metrics.parquet_read_s += time.perf_counter() - t0_read
        if table.num_rows == 0:
            continue
        for rb in table.to_batches(max_chunksize=batch_size):
            t0_convert = time.perf_counter()
            texts = rb.column(0)
            urls = rb.column(1)
            token_counts = rb.column(2).to_numpy(zero_copy_only=False)
            dumps = rb.column(3)
            if metrics is not None:
                metrics.parquet_convert_s += time.perf_counter() - t0_convert
            yield {
                "__batch__": {
                    "text": texts,
                    "url": urls,
                    "token_count": token_counts,
                    "dump": dumps,
                }
            }
        del table


# ---------------------------------------------------------------------------
# Data collection (works with both local and streaming iterators)
# ---------------------------------------------------------------------------

def collect_shard_buffer(dataset_iter, shard_size, pending_batch=None, skip_errors=True):
    """Collect shard_size documents from the streaming dataset iterator.

    Returns a dict of lists (columnar format) and the number collected.
    Returns None if iterator is exhausted.
    """
    buf = {
        "text": [],
        "url": [],
        "token_count": [],
        "dump": [],
    }
    collected = 0
    exhausted = False
    pending = pending_batch

    while collected < shard_size:
        if pending is not None:
            example = {"__batch__": pending}
            pending = None
        else:
            try:
                example = next(dataset_iter)
            except StopIteration:
                exhausted = True
                break

        try:
            if isinstance(example, dict) and "__batch__" in example:
                batch = example["__batch__"]
                texts = batch["text"]
                urls = batch["url"]
                token_counts = batch["token_count"]
                dumps = batch["dump"]
                n_batch = len(texts)
                if n_batch == 0:
                    continue

                take = min(shard_size - collected, n_batch)
                if isinstance(texts, pa.Array):
                    buf["text"].extend(texts.slice(0, take).to_pylist())
                else:
                    buf["text"].extend(texts[:take])
                if isinstance(urls, pa.Array):
                    buf["url"].extend(urls.slice(0, take).to_pylist())
                else:
                    buf["url"].extend(urls[:take])
                if isinstance(token_counts, np.ndarray):
                    buf["token_count"].extend(token_counts[:take].tolist())
                else:
                    buf["token_count"].extend(token_counts[:take])
                if isinstance(dumps, pa.Array):
                    buf["dump"].extend(dumps.slice(0, take).to_pylist())
                else:
                    buf["dump"].extend(dumps[:take])
                collected += take

                if take < n_batch:
                    pending = {
                        "text": texts.slice(take) if isinstance(texts, pa.Array) else texts[take:],
                        "url": urls.slice(take) if isinstance(urls, pa.Array) else urls[take:],
                        "token_count": token_counts[take:],
                        "dump": dumps.slice(take) if isinstance(dumps, pa.Array) else dumps[take:],
                    }
            else:
                text = example.get("text", "")
                if not text:
                    continue
                buf["text"].append(text)
                buf["url"].append(example.get("url", ""))
                buf["token_count"].append(example.get("token_count", len(text.split())))
                buf["dump"].append(example.get("dump", ""))
                collected += 1
        except Exception as e:
            if skip_errors:
                logger.warning("Skipping document due to error: %s", e)
                continue
            raise
    else:
        exhausted = True

    if collected == 0:
        return None, 0, True, pending

    return buf, collected, exhausted, pending


def _skip_documents(dataset_iter, num_to_skip):
    """Skip exactly num_to_skip docs, efficiently handling batched iterators."""
    if num_to_skip <= 0:
        return dataset_iter

    remaining = num_to_skip
    skipped = 0
    progress_step = 1_000_000
    next_progress = progress_step

    while remaining > 0:
        try:
            item = next(dataset_iter)
        except StopIteration:
            print(f"Skip stopped early at {skipped:,} documents (dataset exhausted).")
            return iter(())

        if isinstance(item, dict) and "__batch__" in item:
            batch = item["__batch__"]
            batch_size = len(batch["text"])
            if batch_size <= remaining:
                remaining -= batch_size
                skipped += batch_size
                if skipped >= next_progress:
                    print(f"  skipped {skipped:,} documents...")
                    next_progress += progress_step
                continue

            # Partially consume this batch and prepend remainder.
            cut = remaining
            skipped += cut
            remaining = 0
            bt = batch["text"]
            bu = batch["url"]
            bd = batch["dump"]
            remainder = {
                "text": bt.slice(cut) if isinstance(bt, pa.Array) else bt[cut:],
                "url": bu.slice(cut) if isinstance(bu, pa.Array) else bu[cut:],
                "token_count": batch["token_count"][cut:],
                "dump": bd.slice(cut) if isinstance(bd, pa.Array) else bd[cut:],
            }
            if skipped >= next_progress:
                print(f"  skipped {skipped:,} documents...")
            return chain(({"__batch__": remainder},), dataset_iter)

        remaining -= 1
        skipped += 1
        if skipped >= next_progress:
            print(f"  skipped {skipped:,} documents...")
            next_progress += progress_step

    return dataset_iter


# ---------------------------------------------------------------------------
# Classification loop for a shard buffer
# ---------------------------------------------------------------------------

def classify_buffer(
    texts,
    tokenizer,
    topic_model,
    complexity_model,
    device,
    amp_ctx,
    batch_size,
    max_length,
    chunk_size,
    prefetch_chunks,
    static_padding=False,
    approximate_complexity=False,
    use_cuda_graphs=False,
):
    """Run both classifiers on a list of texts.

    Returns (topic_scores, complexity_scores) as numpy arrays.
    topic_scores: (N, 17) float32
    complexity_scores: (N,) float32
    """
    n = len(texts)
    metrics = Stage1Metrics(docs=n)
    all_topic = np.empty((n, NUM_LABELS), dtype=np.float32)
    all_complexity = np.empty(n, dtype=np.float32)
    pbar = tqdm(total=n, desc="Classifying shard", unit="doc", leave=False)

    use_streams = device.type == "cuda"
    if use_streams:
        compute_stream = torch.cuda.default_stream(device)
        h2d_stream = torch.cuda.Stream(device)
        d2h_stream = torch.cuda.Stream(device)
    else:
        h2d_stream = None
        d2h_stream = None

    def _to_device(batch_tuple):
        positions, ids, mask = batch_tuple
        t0 = time.perf_counter()
        stream = h2d_stream if use_streams else None
        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            ids_d = ids.to(device, non_blocking=True)
            mask_d = mask.to(device, non_blocking=True)
        metrics.h2d_s += time.perf_counter() - t0
        return positions, ids_d, mask_d

    graph_state = None
    graph_failed = False

    for chunk_idx, num_chunks, batches, chunk_offset, tokenize_s in iter_chunks_with_prefetch(
        texts, tokenizer, max_length, batch_size,
        chunk_size, pin_memory=use_streams, prefetch=prefetch_chunks,
        static_padding=static_padding,
    ):
        metrics.tokenization_s += tokenize_s
        if not batches:
            continue

        batch_idx = 0
        next_pos, next_ids, next_mask = _to_device(batches[batch_idx])
        pending_transfer = None

        with torch.no_grad():
            while batch_idx < len(batches):
                if pending_transfer is not None:
                    prev_event, prev_positions, prev_topic_cpu, prev_complexity_cpu, prev_chunk_offset = pending_transfer
                    if prev_event is not None:
                        prev_event.synchronize()
                    global_positions = prev_chunk_offset + np.asarray(prev_positions, dtype=np.int64)
                    all_topic[global_positions] = prev_topic_cpu.numpy()
                    all_complexity[global_positions] = prev_complexity_cpu.numpy()
                    pbar.update(len(prev_positions))
                    pending_transfer = None

                positions = next_pos
                input_ids = next_ids
                attention_mask = next_mask

                if use_streams:
                    compute_stream.wait_stream(h2d_stream)

                batch_idx += 1
                if batch_idx < len(batches):
                    next_pos, next_ids, next_mask = _to_device(batches[batch_idx])

                used_graph = False
                t0_topic = time.perf_counter()
                with amp_ctx:
                    if (
                        use_cuda_graphs
                        and use_streams
                        and not graph_failed
                        and input_ids.shape[0] == batch_size
                        and input_ids.shape[1] == max_length
                    ):
                        if graph_state is None:
                            try:
                                static_ids = torch.empty_like(input_ids)
                                static_mask = torch.empty_like(attention_mask)
                                static_ids.copy_(input_ids)
                                static_mask.copy_(attention_mask)
                                graph_topic = torch.cuda.CUDAGraph()
                                with torch.cuda.graph(graph_topic):
                                    static_topic_logits = topic_model(
                                        input_ids=static_ids, attention_mask=static_mask
                                    ).logits
                                graph_complexity = None
                                static_complexity_logits = None
                                if not approximate_complexity:
                                    graph_complexity = torch.cuda.CUDAGraph()
                                    with torch.cuda.graph(graph_complexity):
                                        static_complexity_logits = complexity_model(
                                            input_ids=static_ids, attention_mask=static_mask
                                        ).logits.squeeze(-1)
                                graph_state = (
                                    static_ids,
                                    static_mask,
                                    graph_topic,
                                    static_topic_logits,
                                    graph_complexity,
                                    static_complexity_logits,
                                )
                            except Exception as graph_exc:
                                graph_failed = True
                                logger.warning("CUDA graph capture disabled due to error: %s", graph_exc)

                        if graph_state is not None and not graph_failed:
                            used_graph = True
                            (
                                static_ids,
                                static_mask,
                                graph_topic,
                                static_topic_logits,
                                graph_complexity,
                                static_complexity_logits,
                            ) = graph_state
                            static_ids.copy_(input_ids)
                            static_mask.copy_(attention_mask)
                            graph_topic.replay()
                            t_logits = static_topic_logits
                            if approximate_complexity:
                                c_logits = t_logits.sigmoid().amax(dim=1) * 4.0
                            else:
                                graph_complexity.replay()
                                c_logits = static_complexity_logits
                        else:
                            t_logits = topic_model(input_ids=input_ids, attention_mask=attention_mask).logits
                            if approximate_complexity:
                                c_logits = t_logits.sigmoid().amax(dim=1) * 4.0
                            else:
                                t0_complexity = time.perf_counter()
                                c_logits = complexity_model(
                                    input_ids=input_ids, attention_mask=attention_mask
                                ).logits.squeeze(-1)
                                metrics.complexity_forward_s += time.perf_counter() - t0_complexity
                    else:
                        t_logits = topic_model(input_ids=input_ids, attention_mask=attention_mask).logits
                        if approximate_complexity:
                            c_logits = t_logits.sigmoid().amax(dim=1) * 4.0
                        else:
                            t0_complexity = time.perf_counter()
                            c_logits = complexity_model(
                                input_ids=input_ids, attention_mask=attention_mask
                            ).logits.squeeze(-1)
                            metrics.complexity_forward_s += time.perf_counter() - t0_complexity
                topic_elapsed = time.perf_counter() - t0_topic
                metrics.topic_forward_s += topic_elapsed
                if used_graph and not approximate_complexity:
                    # Graph replay runs both forwards in one block; split evenly for reporting.
                    half = topic_elapsed * 0.5
                    metrics.topic_forward_s -= half
                    metrics.complexity_forward_s += half

                t0_d2h = time.perf_counter()
                topic_cpu = torch.empty(
                    (len(positions), NUM_LABELS),
                    dtype=torch.float32,
                    pin_memory=use_streams,
                )
                complexity_cpu = torch.empty(
                    (len(positions),),
                    dtype=torch.float32,
                    pin_memory=use_streams,
                )
                with torch.cuda.stream(d2h_stream) if use_streams else contextlib.nullcontext():
                    if use_streams:
                        d2h_stream.wait_stream(compute_stream)
                    topic_cpu.copy_(torch.sigmoid(t_logits).float(), non_blocking=use_streams)
                    complexity_cpu.copy_(c_logits.float(), non_blocking=use_streams)
                metrics.d2h_s += time.perf_counter() - t0_d2h
                transfer_event = torch.cuda.Event() if use_streams else None
                if use_streams:
                    transfer_event.record(d2h_stream)
                pending_transfer = (transfer_event, positions, topic_cpu, complexity_cpu, chunk_offset)
                metrics.batches += 1

            if pending_transfer is not None:
                prev_event, prev_positions, prev_topic_cpu, prev_complexity_cpu, prev_chunk_offset = pending_transfer
                if prev_event is not None:
                    prev_event.synchronize()
                global_positions = prev_chunk_offset + np.asarray(prev_positions, dtype=np.int64)
                all_topic[global_positions] = prev_topic_cpu.numpy()
                all_complexity[global_positions] = prev_complexity_cpu.numpy()
                pbar.update(len(prev_positions))
        del batches

    pbar.close()
    return all_topic, all_complexity, metrics


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    is_a100 = False
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        is_a100 = "A100" in gpu_name.upper()
        print(f"GPU: {gpu_name}")
        if is_a100:
            print("Detected A100 accelerator")

    if args.perf_mode:
        if not args.compile:
            args.compile = True
            print("Perf mode enabled: forcing --compile")
        if args.prefetch_chunks is None:
            args.prefetch_chunks = 4
        if args.tokenize_chunk_size is None:
            args.tokenize_chunk_size = 500000 if is_a100 else 250000
        if args.read_batch_size is None:
            args.read_batch_size = 32768
        if args.collect_queue_size is None:
            args.collect_queue_size = 3
        if args.write_queue_size is None:
            args.write_queue_size = 8
        if is_a100 and not args.force_bf16:
            args.force_bf16 = True
    else:
        args.prefetch_chunks = args.prefetch_chunks or DEFAULT_PREFETCH_CHUNKS
        args.tokenize_chunk_size = args.tokenize_chunk_size or DEFAULT_CHUNK_SIZE
        args.read_batch_size = args.read_batch_size or DEFAULT_READ_BATCH_SIZE
        args.collect_queue_size = args.collect_queue_size or DEFAULT_COLLECT_QUEUE_SIZE
        args.write_queue_size = args.write_queue_size or DEFAULT_WRITE_QUEUE_SIZE

    if args.cuda_graphs and not args.static_padding:
        args.static_padding = True
        print("Enabled --static-padding because --cuda-graphs is set")

    print(
        "Runtime config: "
        f"batch_size={args.batch_size}, max_length={args.max_length}, "
        f"chunk_size={args.tokenize_chunk_size}, prefetch={args.prefetch_chunks}, "
        f"read_batch={args.read_batch_size}, collect_q={args.collect_queue_size}, "
        f"write_q={args.write_queue_size}, static_padding={args.static_padding}, "
        f"cuda_graphs={args.cuda_graphs}, approximate_complexity={args.approximate_complexity}"
    )

    tokenizer, topic_model, complexity_model = load_models(args, device)
    amp_ctx = get_amp_context(device, force_bf16=args.force_bf16)
    warmup_models(
        topic_model,
        complexity_model,
        device,
        amp_ctx,
        args.max_length,
        args.batch_size,
        steps=args.warmup_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip completed shards
    resume_shard = count_existing_shards(output_dir)
    skip_docs = resume_shard * args.shard_size
    if resume_shard > 0:
        print(f"Resuming: found {resume_shard} completed shards, skipping {skip_docs:,} docs")

    # Load data source
    cc_dumps = args.cc_dumps
    io_metrics = Stage1Metrics()
    if args.local_dir:
        print(f"\nReading local parquets from: {args.local_dir}")
        ds_iter = iter_local_parquets(
            args.local_dir,
            cc_dumps,
            batch_size=args.read_batch_size,
            metrics=io_metrics,
        )
    else:
        print(f"\nLoading dataset: {args.dataset} ({args.config})")
        if cc_dumps:
            print(f"Filtering to CC dumps: {cc_dumps}")
        ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
        if cc_dumps:
            dump_set = set(cc_dumps)
            ds = ds.filter(lambda x: x.get("dump", "") in dump_set)
        ds_iter = iter(ds)

    # Skip docs from completed shards
    if skip_docs > 0:
        print(f"Skipping {skip_docs:,} docs for resume...")
        ds_iter = _skip_documents(ds_iter, skip_docs)
        print("Skip complete.")

    # I/O pipelining: collect next shard's docs in a background thread while
    # the GPU classifies the current shard, and write the previous shard
    # in another background thread. This keeps the GPU fed continuously.
    shard_idx = resume_shard
    processed_shards = 0
    total_docs = 0
    total_tokens = 0
    t_start = time.time()

    progress_log = output_dir / "progress.jsonl"
    metrics_log = output_dir / "perf_metrics.jsonl"
    total_metrics = Stage1Metrics()

    # Background writer: writes shards to disk without blocking the GPU
    write_queue: Queue = Queue(maxsize=args.write_queue_size)
    write_sentinel = object()

    def _shard_writer():
        while True:
            item = write_queue.get()
            if item is write_sentinel:
                break
            w_idx, w_records, w_expected = item
            try:
                write_shard(output_dir, w_idx, w_records)
                if not validate_shard(output_dir, w_idx, w_expected):
                    logger.error("Shard %04d failed validation!", w_idx)
            except Exception as e:
                logger.error("Shard %04d write failed: %s", w_idx, e)

    writer_thread = Thread(target=_shard_writer, daemon=True)
    writer_thread.start()

    # Background collector: fetches next shard's docs while GPU is busy
    collect_queue: Queue = Queue(maxsize=args.collect_queue_size)

    def _shard_collector():
        pending_batch = None
        while True:
            result = collect_shard_buffer(ds_iter, args.shard_size, pending_batch=pending_batch)
            collect_queue.put(result)
            buf, collected, exhausted, pending_batch = result
            if buf is None or exhausted:
                break

    collector_thread = Thread(target=_shard_collector, daemon=True)
    collector_thread.start()

    while True:
        buf, collected, exhausted, _ = collect_queue.get()
        if buf is None:
            break

        print(f"\nShard {shard_idx:04d}: classifying {collected:,} documents...")
        shard_t0 = time.time()

        # Run classification (GPU-bound)
        topic_scores, complexity_scores, shard_metrics = classify_buffer(
            buf["text"], tokenizer, topic_model, complexity_model,
            device, amp_ctx, args.batch_size, args.max_length,
            chunk_size=args.tokenize_chunk_size,
            prefetch_chunks=args.prefetch_chunks,
            static_padding=args.static_padding,
            approximate_complexity=args.approximate_complexity,
            use_cuda_graphs=args.cuda_graphs,
        )

        # Efficient numpy -> pyarrow (no Python list-of-lists intermediate)
        buf["topic_scores"] = numpy_to_fixed_list_array(topic_scores, NUM_LABELS)
        buf["complexity"] = pa.array(complexity_scores, type=pa.float32())

        shard_elapsed = time.time() - shard_t0
        shard_rate = collected / shard_elapsed if shard_elapsed > 0 else 0
        total_docs += collected
        shard_tokens = sum(buf["token_count"])
        total_tokens += shard_tokens

        # Queue shard for background writing (non-blocking unless queue full)
        t0_write_block = time.perf_counter()
        write_queue.put((shard_idx, buf, collected))
        shard_metrics.write_block_s += time.perf_counter() - t0_write_block
        total_metrics.merge(shard_metrics)
        total_metrics.parquet_read_s = io_metrics.parquet_read_s
        total_metrics.parquet_convert_s = io_metrics.parquet_convert_s

        # Log progress
        elapsed = time.time() - t_start
        overall_rate = total_docs / elapsed if elapsed > 0 else 0
        print(f"  Shard {shard_idx:04d} classified: {collected:,} docs, "
              f"{shard_tokens:,} tokens, {shard_rate:,.0f} docs/sec")
        print(f"  Total: {total_docs:,} docs, {total_tokens:,} tokens, "
              f"{overall_rate:,.0f} docs/sec overall, {elapsed:.0f}s elapsed")

        log_entry = {
            "shard_idx": shard_idx,
            "docs_in_shard": collected,
            "tokens_in_shard": shard_tokens,
            "total_docs": total_docs,
            "total_tokens": total_tokens,
            "shard_rate": round(shard_rate, 1),
            "overall_rate": round(overall_rate, 1),
            "elapsed_sec": round(elapsed, 1),
            "approximate_complexity": args.approximate_complexity,
            "perf_mode": args.perf_mode,
        }
        with open(progress_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
        with open(metrics_log, "a") as f:
            f.write(json.dumps({
                "shard_idx": shard_idx,
                "docs": collected,
                "metrics": shard_metrics.to_dict(),
                "totals": total_metrics.to_dict(),
            }) + "\n")

        shard_idx += 1
        processed_shards += 1

        if args.max_shards is not None and processed_shards >= args.max_shards:
            print(f"Reached max_shards={args.max_shards}; stopping early for benchmark/tuning.")
            break

        if exhausted:
            break

    # Drain the writer
    write_queue.put(write_sentinel)
    writer_thread.join()

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Classification complete!")
    print(f"  Shards: {shard_idx}")
    print(f"  Documents: {total_docs:,}")
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Rate: {total_docs/elapsed:,.0f} docs/sec")
    print(f"  Perf metrics log: {metrics_log}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Classify FineWeb-Edu documents with topic + complexity models",
    )

    parser.add_argument("--output-dir",
                        default=f"{DEFAULT_OUTPUT_BASE}/scored_shards",
                        help="Output directory for parquet shards")
    parser.add_argument("--topic-model", default=DEFAULT_TOPIC_MODEL)
    parser.add_argument("--complexity-model", default=DEFAULT_COMPLEXITY_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
                        help=f"Documents per shard (default: {DEFAULT_SHARD_SIZE:,})")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for models")
    parser.add_argument("--perf-mode", action="store_true",
                        help="A100-oriented throughput mode with tuned defaults")
    parser.add_argument("--force-bf16", action="store_true",
                        help="Force bfloat16 autocast on CUDA when supported")
    parser.add_argument("--warmup-steps", type=int, default=8,
                        help="Warmup forward passes (helps compiled models)")
    parser.add_argument("--tokenize-chunk-size", type=int, default=None,
                        help="Docs per tokenization chunk (default depends on mode)")
    parser.add_argument("--prefetch-chunks", type=int, default=None,
                        help="How many tokenized chunks to prefetch")
    parser.add_argument("--read-batch-size", type=int, default=None,
                        help="Parquet reader batch size for local files")
    parser.add_argument("--collect-queue-size", type=int, default=None,
                        help="Background collector queue depth")
    parser.add_argument("--write-queue-size", type=int, default=None,
                        help="Background writer queue depth")
    parser.add_argument("--static-padding", action="store_true",
                        help="Pad all batches to max-length for stable kernels")
    parser.add_argument("--cuda-graphs", action="store_true",
                        help="Enable optional CUDA graph replay for full fixed-size batches")
    parser.add_argument("--approximate-complexity", action="store_true",
                        help="Skip complexity model and estimate complexity from topic logits")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--config", default="sample-100BT",
                        help="Dataset config/subset")
    parser.add_argument("--cc-dumps", nargs="*", default=RECOMMENDED_CC_DUMPS,
                        help="CC dumps to include (default: recommended set). "
                             "Pass --cc-dumps with no args to use all dumps.")
    parser.add_argument("--local-dir",
                        help="Local directory with downloaded parquet files. "
                             "When set, reads from disk with pyarrow predicate "
                             "pushdown instead of HF streaming.")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Process at most N shards then exit (useful for tuning runs)")

    args = parser.parse_args()
    if args.cc_dumps is not None and len(args.cc_dumps) == 0:
        args.cc_dumps = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    run(args)


if __name__ == "__main__":
    main()
