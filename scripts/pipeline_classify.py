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

CHUNK_SIZE = 250000  # 117GB RAM: ~5GB per chunk of tokenized tensors, well within budget
PREFETCH_CHUNKS = 2   # overlap 2 chunks of CPU tokenization with GPU inference


def prepare_batches(texts, tokenizer, max_length, batch_size, pin_memory=False):
    """Tokenize texts, sort by length, return padded batch tensors.

    Returns list of (batch_positions, input_ids, attention_mask) tuples,
    where batch_positions maps back to the original indices in `texts`.
    """
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
        padded = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(torch.long)
        attention_mask = padded["attention_mask"].to(torch.long)
        if pin_memory:
            input_ids = input_ids.pin_memory()
            attention_mask = attention_mask.pin_memory()
        batches.append((batch_order, input_ids, attention_mask))

    return batches


def iter_chunks_with_prefetch(texts, tokenizer, max_length, batch_size,
                              chunk_size, pin_memory, prefetch):
    """Yield (chunk_idx, num_chunks, batches) with background tokenization."""
    num_chunks = (len(texts) + chunk_size - 1) // chunk_size

    if prefetch <= 0:
        for ci in range(num_chunks):
            s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(texts))
            batches = prepare_batches(texts[s:e], tokenizer, max_length, batch_size, pin_memory)
            yield ci, num_chunks, batches, s
        return

    queue: Queue = Queue(maxsize=max(1, prefetch))
    sentinel = object()

    def _producer():
        try:
            for ci in range(num_chunks):
                s, e = ci * chunk_size, min((ci + 1) * chunk_size, len(texts))
                batches = prepare_batches(texts[s:e], tokenizer, max_length, batch_size, pin_memory)
                queue.put((ci, num_chunks, batches, s))
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


def get_amp_context(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"AMP enabled (dtype={dtype})")
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


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

def iter_local_parquets(local_dir, cc_dumps=None):
    """Yield docs from local parquet files with pyarrow predicate pushdown.

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
        table = pq.read_table(f, columns=cols, filters=filters)
        if table.num_rows == 0:
            continue
        texts = table.column("text").to_pylist()
        urls = table.column("url").to_pylist()
        token_counts = table.column("token_count").to_pylist()
        dumps = table.column("dump").to_pylist()
        del table
        for i in range(len(texts)):
            yield {"text": texts[i], "url": urls[i],
                   "token_count": token_counts[i], "dump": dumps[i]}


# ---------------------------------------------------------------------------
# Data collection (works with both local and streaming iterators)
# ---------------------------------------------------------------------------

def collect_shard_buffer(dataset_iter, shard_size, skip_errors=True):
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

    for example in dataset_iter:
        try:
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

        if collected >= shard_size:
            break
    else:
        exhausted = True

    if collected == 0:
        return None, 0, True

    return buf, collected, exhausted


# ---------------------------------------------------------------------------
# Classification loop for a shard buffer
# ---------------------------------------------------------------------------

def classify_buffer(texts, tokenizer, topic_model, complexity_model,
                    device, amp_ctx, batch_size, max_length):
    """Run both classifiers on a list of texts.

    Returns (topic_scores, complexity_scores) as numpy arrays.
    topic_scores: (N, 17) float32
    complexity_scores: (N,) float32
    """
    n = len(texts)
    all_topic = np.empty((n, NUM_LABELS), dtype=np.float32)
    all_complexity = np.empty(n, dtype=np.float32)

    use_streams = device.type == "cuda"
    if use_streams:
        compute_stream = torch.cuda.default_stream(device)
        transfer_stream = torch.cuda.Stream(device)

    def _to_device(batch_tuple):
        positions, ids, mask = batch_tuple
        stream = transfer_stream if use_streams else None
        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            ids_d = ids.to(device, non_blocking=True)
            mask_d = mask.to(device, non_blocking=True)
        return positions, ids_d, mask_d

    for chunk_idx, num_chunks, batches, chunk_offset in iter_chunks_with_prefetch(
        texts, tokenizer, max_length, batch_size,
        CHUNK_SIZE, pin_memory=use_streams, prefetch=PREFETCH_CHUNKS,
    ):
        if not batches:
            continue

        batch_idx = 0
        next_pos, next_ids, next_mask = _to_device(batches[batch_idx])

        with torch.no_grad():
            while batch_idx < len(batches):
                positions = next_pos
                input_ids = next_ids
                attention_mask = next_mask

                if use_streams:
                    compute_stream.wait_stream(transfer_stream)

                batch_idx += 1
                if batch_idx < len(batches):
                    next_pos, next_ids, next_mask = _to_device(batches[batch_idx])

                with amp_ctx:
                    t_logits = topic_model(input_ids=input_ids, attention_mask=attention_mask).logits.clone()
                    c_logits = complexity_model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

                t_scores = torch.sigmoid(t_logits).float().cpu().numpy()
                c_scores = c_logits.float().cpu().numpy()

                global_positions = [chunk_offset + p for p in positions]
                all_topic[global_positions] = t_scores
                all_complexity[global_positions] = c_scores

        del batches

    return all_topic, all_complexity


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

    tokenizer, topic_model, complexity_model = load_models(args, device)
    amp_ctx = get_amp_context(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip completed shards
    resume_shard = count_existing_shards(output_dir)
    skip_docs = resume_shard * args.shard_size
    if resume_shard > 0:
        print(f"Resuming: found {resume_shard} completed shards, skipping {skip_docs:,} docs")

    # Load data source
    cc_dumps = args.cc_dumps
    if args.local_dir:
        print(f"\nReading local parquets from: {args.local_dir}")
        ds_iter = iter_local_parquets(args.local_dir, cc_dumps)
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
        for i, _ in enumerate(ds_iter):
            if i + 1 >= skip_docs:
                break
        print("Skip complete.")

    # I/O pipelining: collect next shard's docs in a background thread while
    # the GPU classifies the current shard, and write the previous shard
    # in another background thread. This keeps the GPU fed continuously.
    shard_idx = resume_shard
    total_docs = 0
    total_tokens = 0
    t_start = time.time()

    progress_log = output_dir / "progress.jsonl"

    # Background writer: writes shards to disk without blocking the GPU
    write_queue: Queue = Queue(maxsize=2)
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
    collect_queue: Queue = Queue(maxsize=1)

    def _shard_collector():
        while True:
            result = collect_shard_buffer(ds_iter, args.shard_size)
            collect_queue.put(result)
            buf, collected, exhausted = result
            if buf is None or exhausted:
                break

    collector_thread = Thread(target=_shard_collector, daemon=True)
    collector_thread.start()

    while True:
        buf, collected, exhausted = collect_queue.get()
        if buf is None:
            break

        print(f"\nShard {shard_idx:04d}: classifying {collected:,} documents...")
        shard_t0 = time.time()

        # Run classification (GPU-bound)
        topic_scores, complexity_scores = classify_buffer(
            buf["text"], tokenizer, topic_model, complexity_model,
            device, amp_ctx, args.batch_size, args.max_length,
        )

        # Efficient numpy -> pyarrow (no Python list-of-lists intermediate)
        buf["topic_scores"] = numpy_to_fixed_list_array(topic_scores, NUM_LABELS)
        buf["complexity"] = complexity_scores.tolist()

        shard_elapsed = time.time() - shard_t0
        shard_rate = collected / shard_elapsed if shard_elapsed > 0 else 0
        total_docs += collected
        shard_tokens = sum(buf["token_count"])
        total_tokens += shard_tokens

        # Queue shard for background writing (non-blocking unless queue full)
        write_queue.put((shard_idx, buf, collected))

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
        }
        with open(progress_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        shard_idx += 1

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
