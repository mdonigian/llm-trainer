#!/usr/bin/env python3
"""
Stage 1: Classify FineWeb-Edu documents with topic + complexity models.

Rewritten for predictable A100 throughput:
- Fast-tokenizer guard (fails if slow tokenizer unless explicitly allowed)
- Tokenization subchunks so classification starts immediately
- Length-sorted dynamic padding to reduce wasted compute
- Optional approximate complexity mode for maximum docs/sec
- Background writer + optional collector overlap
- Per-shard performance metrics JSONL
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
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

SHARD_SCHEMA = pa.schema([
    pa.field("text", pa.string()),
    pa.field("url", pa.string()),
    pa.field("token_count", pa.int64()),
    pa.field("dump", pa.string()),
    pa.field("topic_scores", pa.list_(pa.float32(), NUM_LABELS)),
    pa.field("complexity", pa.float32()),
])

DEFAULT_TOKENIZE_CHUNK_SIZE = 250_000
DEFAULT_TOKENIZE_SUBCHUNK_SIZE = 16_384
DEFAULT_READ_BATCH_SIZE = 32_768
DEFAULT_WRITE_QUEUE_SIZE = 8
DEFAULT_COLLECT_QUEUE_SIZE = 3


@dataclass
class PerfMetrics:
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

    def merge(self, other: "PerfMetrics"):
        for key in asdict(self):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def to_dict(self):
        out = asdict(self)
        measured = (
            self.parquet_read_s
            + self.parquet_convert_s
            + self.tokenization_s
            + self.h2d_s
            + self.topic_forward_s
            + self.complexity_forward_s
            + self.d2h_s
            + self.write_block_s
        )
        if measured > 0:
            out["pct"] = {
                "parquet_read": round(self.parquet_read_s * 100.0 / measured, 2),
                "parquet_convert": round(self.parquet_convert_s * 100.0 / measured, 2),
                "tokenization": round(self.tokenization_s * 100.0 / measured, 2),
                "h2d": round(self.h2d_s * 100.0 / measured, 2),
                "topic_forward": round(self.topic_forward_s * 100.0 / measured, 2),
                "complexity_forward": round(self.complexity_forward_s * 100.0 / measured, 2),
                "d2h": round(self.d2h_s * 100.0 / measured, 2),
                "write_block": round(self.write_block_s * 100.0 / measured, 2),
            }
        return out


def shard_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / f"shard_{shard_idx:04d}.parquet"


def count_existing_shards(output_dir: Path) -> int:
    i = 0
    while shard_path(output_dir, i).exists():
        i += 1
    return i


def numpy_to_fixed_list_array(arr: np.ndarray, list_size: int) -> pa.FixedSizeListArray:
    flat = pa.array(arr.ravel(), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size)


def write_shard(output_dir: Path, shard_idx: int, records: dict):
    table = pa.table(records, schema=SHARD_SCHEMA)
    pq.write_table(table, shard_path(output_dir, shard_idx))


def validate_shard(output_dir: Path, shard_idx: int, expected_rows: int) -> bool:
    path = shard_path(output_dir, shard_idx)
    if not path.exists():
        return False
    try:
        return pq.read_metadata(path).num_rows == expected_rows
    except Exception:
        return False


def get_amp_context(device, force_bf16=False):
    if device.type != "cuda":
        return contextlib.nullcontext()
    if force_bf16 and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"AMP enabled (dtype={dtype})")
    return torch.amp.autocast("cuda", dtype=dtype)


def warmup_models(topic_model, complexity_model, device, amp_ctx, max_length, batch_size, steps):
    if device.type != "cuda" or steps <= 0:
        return
    seq_len = min(256, max_length)
    ids = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(steps):
            with amp_ctx:
                _ = topic_model(input_ids=ids, attention_mask=mask).logits
                _ = complexity_model(input_ids=ids, attention_mask=mask).logits
    torch.cuda.synchronize(device)


def load_models(args, device):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    if args.tokenizer_workers > 0:
        os.environ["RAYON_NUM_THREADS"] = str(args.tokenizer_workers)

    print(f"Loading topic model: {args.topic_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.topic_model, use_fast=True)
    print(f"Tokenizer: {tokenizer.__class__.__name__} (is_fast={tokenizer.is_fast})")
    if not tokenizer.is_fast and not args.allow_slow_tokenizer:
        raise RuntimeError(
            "Slow tokenizer detected. Use a fast tokenizer model or pass --allow-slow-tokenizer."
        )

    topic_model = AutoModelForSequenceClassification.from_pretrained(args.topic_model).to(device).eval()
    print(f"Loading complexity model: {args.complexity_model}")
    complexity_model = AutoModelForSequenceClassification.from_pretrained(args.complexity_model).to(device).eval()

    if args.compile and hasattr(torch, "compile"):
        cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
        mode = "max-autotune" if cc >= (9, 0) else "reduce-overhead"
        print(f"Compiling models (mode={mode})")
        topic_model = torch.compile(topic_model, mode=mode)
        if not args.approximate_complexity:
            complexity_model = torch.compile(complexity_model, mode=mode)

    return tokenizer, topic_model, complexity_model


def iter_local_parquets(local_dir, cc_dumps=None, batch_size=DEFAULT_READ_BATCH_SIZE, metrics=None):
    files = sorted(Path(local_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {local_dir}")

    print(f"Found {len(files)} local parquet files in {local_dir}")
    if cc_dumps:
        print(f"Filtering to CC dumps: {cc_dumps}")
    filters = [("dump", "in", cc_dumps)] if cc_dumps else None
    cols = ["text", "url", "token_count", "dump"]

    for path in files:
        t0 = time.perf_counter()
        table = pq.read_table(path, columns=cols, filters=filters)
        if metrics is not None:
            metrics.parquet_read_s += time.perf_counter() - t0
        if table.num_rows == 0:
            continue
        for rb in table.to_batches(max_chunksize=batch_size):
            t1 = time.perf_counter()
            payload = {
                "text": rb.column(0),
                "url": rb.column(1),
                "token_count": rb.column(2).fill_null(0).to_numpy(zero_copy_only=False),
                "dump": rb.column(3),
            }
            if metrics is not None:
                metrics.parquet_convert_s += time.perf_counter() - t1
            yield {"__batch__": payload}
        del table


def collect_shard_buffer(dataset_iter, shard_size, pending_batch=None):
    buf = {"text": [], "url": [], "token_count": [], "dump": []}
    collected = 0
    pending = pending_batch
    exhausted = False

    while collected < shard_size:
        if pending is not None:
            item = {"__batch__": pending}
            pending = None
        else:
            try:
                item = next(dataset_iter)
            except StopIteration:
                exhausted = True
                break

        if isinstance(item, dict) and "__batch__" in item:
            b = item["__batch__"]
            texts = b["text"]
            urls = b["url"]
            toks = b["token_count"]
            dumps = b["dump"]
            n = len(texts)
            if n == 0:
                continue
            take = min(shard_size - collected, n)
            buf["text"].extend(texts.slice(0, take).to_pylist() if isinstance(texts, pa.Array) else texts[:take])
            buf["url"].extend(urls.slice(0, take).to_pylist() if isinstance(urls, pa.Array) else urls[:take])
            if isinstance(toks, np.ndarray):
                buf["token_count"].extend(toks[:take].tolist())
            else:
                buf["token_count"].extend(toks[:take])
            buf["dump"].extend(dumps.slice(0, take).to_pylist() if isinstance(dumps, pa.Array) else dumps[:take])
            collected += take
            if take < n:
                pending = {
                    "text": texts.slice(take) if isinstance(texts, pa.Array) else texts[take:],
                    "url": urls.slice(take) if isinstance(urls, pa.Array) else urls[take:],
                    "token_count": toks[take:],
                    "dump": dumps.slice(take) if isinstance(dumps, pa.Array) else dumps[take:],
                }
        else:
            txt = item.get("text", "")
            if not txt:
                continue
            buf["text"].append(txt)
            buf["url"].append(item.get("url", ""))
            buf["token_count"].append(item.get("token_count", len(txt.split())))
            buf["dump"].append(item.get("dump", ""))
            collected += 1

    if collected == 0:
        return None, 0, exhausted, pending
    return buf, collected, exhausted, pending


def _skip_documents(dataset_iter, num_to_skip):
    if num_to_skip <= 0:
        return dataset_iter
    remaining = num_to_skip
    while remaining > 0:
        try:
            item = next(dataset_iter)
        except StopIteration:
            return iter(())
        if isinstance(item, dict) and "__batch__" in item:
            b = item["__batch__"]
            n = len(b["text"])
            if n <= remaining:
                remaining -= n
                continue
            cut = remaining
            remaining = 0
            remainder = {
                "text": b["text"].slice(cut) if isinstance(b["text"], pa.Array) else b["text"][cut:],
                "url": b["url"].slice(cut) if isinstance(b["url"], pa.Array) else b["url"][cut:],
                "token_count": b["token_count"][cut:],
                "dump": b["dump"].slice(cut) if isinstance(b["dump"], pa.Array) else b["dump"][cut:],
            }
            return chain(({"__batch__": remainder},), dataset_iter)
        remaining -= 1
    return dataset_iter


def prepare_chunk_batches(
    indices,
    texts,
    tokenizer,
    max_length,
    batch_size,
    pin_memory=False,
    static_padding=False,
    max_chars_per_doc=0,
):
    if max_chars_per_doc and max_chars_per_doc > 0:
        texts = [t if len(t) <= max_chars_per_doc else t[:max_chars_per_doc] for t in texts]

    t0_tok = time.perf_counter()
    enc = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_length=True,
    )
    tokenization_s = time.perf_counter() - t0_tok
    ids_list = enc["input_ids"]
    order = np.argsort(np.asarray(enc["length"]))

    batches = []
    for bs in range(0, len(order), batch_size):
        idx = order[bs : bs + batch_size]
        batch_indices = [indices[i] for i in idx]
        batch_ids = [ids_list[i] for i in idx]
        padded = tokenizer.pad(
            {"input_ids": batch_ids},
            padding="max_length" if static_padding else True,
            max_length=max_length if static_padding else None,
            return_tensors="pt",
        )
        input_ids = padded["input_ids"].to(torch.long)
        attention_mask = padded["attention_mask"].to(torch.long)
        if pin_memory:
            input_ids = input_ids.pin_memory()
            attention_mask = attention_mask.pin_memory()
        batches.append((batch_indices, input_ids, attention_mask))

    return batches, tokenization_s


def iter_prepared_chunks(
    texts,
    tokenizer,
    max_length,
    batch_size,
    chunk_size,
    pin_memory,
    prefetch_chunks,
    static_padding,
    max_chars_per_doc,
):
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size
    prefetch_chunks = max(0, prefetch_chunks)

    if prefetch_chunks == 0:
        for chunk_idx in range(num_chunks):
            s = chunk_idx * chunk_size
            e = min(s + chunk_size, len(texts))
            indices = list(range(s, e))
            chunk_texts = texts[s:e]
            batches, tok_s = prepare_chunk_batches(
                indices=indices,
                texts=chunk_texts,
                tokenizer=tokenizer,
                max_length=max_length,
                batch_size=batch_size,
                pin_memory=pin_memory,
                static_padding=static_padding,
                max_chars_per_doc=max_chars_per_doc,
            )
            yield chunk_idx, num_chunks, batches, tok_s
        return

    queue = Queue(maxsize=max(1, prefetch_chunks))
    sentinel = object()

    def _producer():
        try:
            for chunk_idx in range(num_chunks):
                s = chunk_idx * chunk_size
                e = min(s + chunk_size, len(texts))
                indices = list(range(s, e))
                chunk_texts = texts[s:e]
                batches, tok_s = prepare_chunk_batches(
                    indices=indices,
                    texts=chunk_texts,
                    tokenizer=tokenizer,
                    max_length=max_length,
                    batch_size=batch_size,
                    pin_memory=pin_memory,
                    static_padding=static_padding,
                    max_chars_per_doc=max_chars_per_doc,
                )
                queue.put((chunk_idx, num_chunks, batches, tok_s))
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


def classify_buffer(texts, tokenizer, topic_model, complexity_model, device, amp_ctx, args):
    n = len(texts)
    all_topic = np.empty((n, NUM_LABELS), dtype=np.float32)
    all_complexity = np.empty(n, dtype=np.float32)
    metrics = PerfMetrics(docs=n)
    pbar = tqdm(total=n, desc="Classifying shard", unit="doc", leave=False)

    step = max(1, min(args.tokenize_chunk_size, args.tokenize_subchunk_size))
    use_streams = device.type == "cuda"
    if use_streams:
        compute_stream = torch.cuda.default_stream(device)
        transfer_stream = torch.cuda.Stream(device)

    def _to_device(batch_tuple):
        batch_indices, b_ids, b_mask = batch_tuple
        t0_h2d = time.perf_counter()
        stream = transfer_stream if use_streams else None
        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            ids = b_ids.to(device, non_blocking=True)
            mask = b_mask.to(device, non_blocking=True)
        metrics.h2d_s += time.perf_counter() - t0_h2d
        return ids, mask, batch_indices

    with torch.no_grad():
        for _, _, batches, tok_s in iter_prepared_chunks(
            texts=texts,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            chunk_size=step,
            pin_memory=use_streams,
            prefetch_chunks=args.prefetch_chunks,
            static_padding=args.static_padding,
            max_chars_per_doc=args.max_chars_per_doc,
        ):
            metrics.tokenization_s += tok_s
            if not batches:
                continue

            batch_idx = 0
            next_ids, next_mask, next_indices = _to_device(batches[batch_idx])

            while batch_idx < len(batches):
                input_ids = next_ids
                attention_mask = next_mask
                batch_indices = next_indices

                if use_streams:
                    compute_stream.wait_stream(transfer_stream)

                batch_idx += 1
                if batch_idx < len(batches):
                    next_ids, next_mask, next_indices = _to_device(batches[batch_idx])

                t0_topic = time.perf_counter()
                with amp_ctx:
                    t_logits = topic_model(input_ids=input_ids, attention_mask=attention_mask).logits
                metrics.topic_forward_s += time.perf_counter() - t0_topic

                if args.approximate_complexity:
                    c_logits = t_logits.sigmoid().amax(dim=1) * 4.0
                else:
                    t0_c = time.perf_counter()
                    with amp_ctx:
                        c_logits = complexity_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                        ).logits.squeeze(-1)
                    metrics.complexity_forward_s += time.perf_counter() - t0_c

                t0_d2h = time.perf_counter()
                t_scores = torch.sigmoid(t_logits).float().cpu().numpy()
                c_scores = c_logits.float().cpu().numpy()
                metrics.d2h_s += time.perf_counter() - t0_d2h

                all_topic[np.asarray(batch_indices, dtype=np.int64)] = t_scores
                all_complexity[np.asarray(batch_indices, dtype=np.int64)] = c_scores
                metrics.batches += 1
                pbar.update(len(batch_indices))

    pbar.close()
    return all_topic, all_complexity, metrics


def configure_runtime(args, device):
    is_a100 = device.type == "cuda" and "A100" in torch.cuda.get_device_name(device).upper()
    if args.perf_mode:
        args.compile = True
        if args.force_bf16 is None:
            args.force_bf16 = is_a100
        args.tokenize_chunk_size = args.tokenize_chunk_size or 500_000
        args.tokenize_subchunk_size = args.tokenize_subchunk_size or DEFAULT_TOKENIZE_SUBCHUNK_SIZE
        args.prefetch_chunks = args.prefetch_chunks or 1
        args.read_batch_size = args.read_batch_size or DEFAULT_READ_BATCH_SIZE
        args.collect_queue_size = args.collect_queue_size or DEFAULT_COLLECT_QUEUE_SIZE
        args.write_queue_size = args.write_queue_size or DEFAULT_WRITE_QUEUE_SIZE
        args.max_chars_per_doc = args.max_chars_per_doc or 8192
    else:
        args.force_bf16 = bool(args.force_bf16)
        args.tokenize_chunk_size = args.tokenize_chunk_size or DEFAULT_TOKENIZE_CHUNK_SIZE
        args.tokenize_subchunk_size = args.tokenize_subchunk_size or DEFAULT_TOKENIZE_SUBCHUNK_SIZE
        args.prefetch_chunks = args.prefetch_chunks or 0
        args.read_batch_size = args.read_batch_size or 8192
        args.collect_queue_size = args.collect_queue_size or 1
        args.write_queue_size = args.write_queue_size or 2
        args.max_chars_per_doc = args.max_chars_per_doc or 0


def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    configure_runtime(args, device)
    print(
        "Runtime config: "
        f"batch={args.batch_size}, max_len={args.max_length}, tok_chunk={args.tokenize_chunk_size}, "
        f"tok_subchunk={args.tokenize_subchunk_size}, max_chars={args.max_chars_per_doc}, "
        f"prefetch={args.prefetch_chunks}, compile={args.compile}, "
        f"perf_mode={args.perf_mode}, approx_complexity={args.approximate_complexity}"
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
        args.warmup_steps,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "progress.jsonl"
    perf_log = output_dir / "perf_metrics.jsonl"

    resume_shard = count_existing_shards(output_dir)
    skip_docs = resume_shard * args.shard_size
    if resume_shard > 0:
        print(f"Resuming: {resume_shard} completed shards, skipping ~{skip_docs:,} docs")

    io_metrics = PerfMetrics()
    if args.local_dir:
        ds_iter = iter_local_parquets(
            args.local_dir,
            cc_dumps=args.cc_dumps,
            batch_size=args.read_batch_size,
            metrics=io_metrics,
        )
    else:
        ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
        if args.cc_dumps:
            dump_set = set(args.cc_dumps)
            ds = ds.filter(lambda x: x.get("dump", "") in dump_set)
        ds_iter = iter(ds)

    if skip_docs > 0:
        ds_iter = _skip_documents(ds_iter, skip_docs)

    write_queue: Queue = Queue(maxsize=args.write_queue_size)
    write_sentinel = object()

    def writer_thread_fn():
        while True:
            item = write_queue.get()
            if item is write_sentinel:
                break
            shard_idx, records, expected = item
            try:
                write_shard(output_dir, shard_idx, records)
                if not validate_shard(output_dir, shard_idx, expected):
                    logger.error("Shard %04d failed validation", shard_idx)
            except Exception as exc:
                logger.error("Writer error on shard %04d: %s", shard_idx, exc)

    wt = Thread(target=writer_thread_fn, daemon=True)
    wt.start()

    collect_queue: Queue = Queue(maxsize=args.collect_queue_size)

    def collector_thread_fn():
        pending = None
        while True:
            item = collect_shard_buffer(ds_iter, args.shard_size, pending)
            collect_queue.put(item)
            buf, _, exhausted, pending = item
            if buf is None or exhausted:
                break

    ct = Thread(target=collector_thread_fn, daemon=True)
    ct.start()

    shard_idx = resume_shard
    processed_shards = 0
    total_docs = 0
    total_tokens = 0
    total_metrics = PerfMetrics()
    t_start = time.time()

    while True:
        buf, collected, exhausted, _ = collect_queue.get()
        if buf is None:
            break

        print(f"\nShard {shard_idx:04d}: classifying {collected:,} documents...")
        shard_t0 = time.time()
        topic_scores, complexity_scores, shard_metrics = classify_buffer(
            buf["text"], tokenizer, topic_model, complexity_model, device, amp_ctx, args
        )
        buf["topic_scores"] = numpy_to_fixed_list_array(topic_scores, NUM_LABELS)
        buf["complexity"] = pa.array(complexity_scores, type=pa.float32())

        shard_elapsed = time.time() - shard_t0
        shard_rate = collected / shard_elapsed if shard_elapsed > 0 else 0
        shard_tokens = int(np.sum(np.asarray(buf["token_count"], dtype=np.int64)))
        total_docs += collected
        total_tokens += shard_tokens

        t0_block = time.perf_counter()
        write_queue.put((shard_idx, buf, collected))
        shard_metrics.write_block_s += time.perf_counter() - t0_block
        total_metrics.merge(shard_metrics)
        total_metrics.parquet_read_s = io_metrics.parquet_read_s
        total_metrics.parquet_convert_s = io_metrics.parquet_convert_s

        elapsed = time.time() - t_start
        overall_rate = total_docs / elapsed if elapsed > 0 else 0
        print(
            f"  shard_rate={shard_rate:,.0f} docs/s | overall={overall_rate:,.0f} docs/s | "
            f"total_docs={total_docs:,}"
        )

        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "shard_idx": shard_idx,
                "docs_in_shard": collected,
                "tokens_in_shard": shard_tokens,
                "total_docs": total_docs,
                "total_tokens": total_tokens,
                "shard_rate": round(shard_rate, 1),
                "overall_rate": round(overall_rate, 1),
                "elapsed_sec": round(elapsed, 1),
            }) + "\n")
        with open(perf_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "shard_idx": shard_idx,
                "docs": collected,
                "metrics": shard_metrics.to_dict(),
                "totals": total_metrics.to_dict(),
            }) + "\n")

        shard_idx += 1
        processed_shards += 1
        if args.max_shards is not None and processed_shards >= args.max_shards:
            print(f"Reached max_shards={args.max_shards}, stopping early")
            break
        if exhausted:
            break

    write_queue.put(write_sentinel)
    wt.join()

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("Classification complete")
    print(f"Shards written: {shard_idx - resume_shard}")
    print(f"Documents: {total_docs:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Elapsed: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"Throughput: {total_docs / elapsed:,.0f} docs/s")
    print(f"Progress log: {progress_log}")
    print(f"Perf log: {perf_log}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: classify FineWeb-Edu")
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_BASE}/scored_shards")
    parser.add_argument("--topic-model", default=DEFAULT_TOPIC_MODEL)
    parser.add_argument("--complexity-model", default=DEFAULT_COMPLEXITY_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--perf-mode", action="store_true")
    parser.add_argument("--force-bf16", action="store_true", default=None)
    parser.add_argument("--warmup-steps", type=int, default=4)
    parser.add_argument("--tokenize-chunk-size", type=int, default=None)
    parser.add_argument("--tokenize-subchunk-size", type=int, default=None)
    parser.add_argument("--prefetch-chunks", type=int, default=None)
    parser.add_argument("--read-batch-size", type=int, default=None)
    parser.add_argument("--collect-queue-size", type=int, default=None)
    parser.add_argument("--write-queue-size", type=int, default=None)
    parser.add_argument("--max-chars-per-doc", type=int, default=None)
    parser.add_argument("--tokenizer-workers", type=int, default=0)
    parser.add_argument("--allow-slow-tokenizer", action="store_true")
    parser.add_argument("--static-padding", action="store_true")
    parser.add_argument("--approximate-complexity", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-100BT")
    parser.add_argument("--cc-dumps", nargs="*", default=RECOMMENDED_CC_DUMPS)
    parser.add_argument("--local-dir")
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

