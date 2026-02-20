#!/usr/bin/env python3
"""
Stage 1: Classify StarCoder code files with the multi-task code classifier.

Uses a single UniXcoder-base model with three heads:
  - Code Quality (1-5): regression
  - Structured Data Relevance (0-3): regression
  - Content Type (9 classes): classification

Key design decisions from the project:
  - zlib compression ratio pre-filter (<0.10) runs BEFORE BERT inference
    to skip repetitive boilerplate and save GPU time
  - The model is loaded via the custom CodeClassifierModel class
    (shared encoder + three linear heads), NOT AutoModelForSequenceClassification
  - Stores all three scores per document for downstream filtering flexibility

Optimized for RTX 5090 (32GB GDDR7, 1.79 TB/s, Blackwell sm_120):
  - bf16 AMP via Tensor Cores (680 5th-gen)
  - torch.compile with max-autotune for CUDA graph capture
  - Dual CUDA streams: overlap H2D transfers with compute, async D2H
  - Background tokenization prefetch (CPU prepares next chunk while GPU runs)
  - Multiprocess zlib compression pre-filter
  - Batch size 4096 default (125M param model leaves ~30GB for activations)
  - Bucket padding to reduce wasted Tensor Core cycles
  - Warmup all bucket shapes for torch.compile graph capture
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from pipeline_config import (
    COMPRESSION_RATIO_FLOOR,
    CONTENT_TYPES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_DATASET,
    DEFAULT_MAX_LENGTH,
    DEFAULT_OUTPUT_BASE,
    DEFAULT_SHARD_SIZE,
    NUM_CONTENT_TYPES,
    RECOMMENDED_LANGUAGES,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model definition (mirrors train_starcoderdata.py)
# ---------------------------------------------------------------------------

class CodeClassifierModel(nn.Module):
    """Multi-task model: shared encoder with three task-specific heads."""

    def __init__(self, encoder_name_or_path: str, num_content_types: int = NUM_CONTENT_TYPES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        drop_rate = getattr(self.encoder.config, "classifier_dropout", None)
        if drop_rate is None:
            drop_rate = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_rate)
        self.quality_head = nn.Linear(hidden, 1)
        self.structured_data_head = nn.Linear(hidden, 1)
        self.content_type_head = nn.Linear(hidden, num_content_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0])
        return (
            self.quality_head(cls).squeeze(-1),
            self.structured_data_head(cls).squeeze(-1),
            self.content_type_head(cls),
        )


def load_classifier(model_path: str, device: torch.device, compile_model=False):
    """Load the multi-task classifier from a saved checkpoint."""
    model_dir = Path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)

    model = CodeClassifierModel(str(model_dir))
    heads_path = model_dir / "classifier_heads.pt"
    if heads_path.exists():
        heads = torch.load(heads_path, map_location=device, weights_only=True)
        model.quality_head.load_state_dict(heads["quality_head"])
        model.structured_data_head.load_state_dict(heads["structured_data_head"])
        model.content_type_head.load_state_dict(heads["content_type_head"])
    else:
        logger.warning("No classifier_heads.pt found — using randomly initialized heads")

    model.to(device).eval()

    if compile_model and hasattr(torch, "compile"):
        cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
        # Blackwell (RTX 5090) is cc 10.x / 12.x — use max-autotune for CUDA graphs.
        # Ampere (A100) is cc 8.0 — also benefits from max-autotune.
        # Anything >= 8.0 gets max-autotune; older falls back to reduce-overhead.
        mode = "max-autotune" if cc >= (8, 0) else "reduce-overhead"
        print(f"Compiling model (mode={mode}, cc={cc[0]}.{cc[1]})...")
        model = torch.compile(model, mode=mode, dynamic=True)

    return tokenizer, model


# ---------------------------------------------------------------------------
# Schema and constants
# ---------------------------------------------------------------------------

SHARD_SCHEMA = pa.schema([
    pa.field("content", pa.string()),
    pa.field("lang", pa.string()),
    pa.field("size", pa.int64()),
    pa.field("token_count", pa.int64()),
    pa.field("quality", pa.float32()),
    pa.field("structured_data", pa.float32()),
    pa.field("content_type", pa.string()),
    pa.field("content_type_idx", pa.int32()),
])

DEFAULT_TOKENIZE_CHUNK_SIZE = 250_000
DEFAULT_TOKENIZE_SUBCHUNK_SIZE = 16_384
DEFAULT_READ_BATCH_SIZE = 32_768
DEFAULT_WRITE_QUEUE_SIZE = 8
DEFAULT_COLLECT_QUEUE_SIZE = 3


# ---------------------------------------------------------------------------
# Compression ratio pre-filter
# ---------------------------------------------------------------------------

def compression_ratio(text: str) -> float:
    """Compute zlib compression ratio. Low ratio = highly repetitive."""
    if not text:
        return 0.0
    raw = text.encode("utf-8", errors="replace")
    if len(raw) == 0:
        return 0.0
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def _compression_ratio_chunk(texts: list[str]) -> list[float]:
    """Process a chunk of texts — used by ProcessPoolExecutor."""
    return [compression_ratio(t) for t in texts]


# Global pool for compression filtering, lazily initialized.
# zlib is CPU-bound and releases the GIL, so multiprocessing helps
# saturate all cores while GPU waits for the next shard.
_compression_pool: ProcessPoolExecutor | None = None
_COMPRESSION_WORKERS = min(os.cpu_count() or 4, 16)
_COMPRESSION_CHUNK = 8192


def _get_compression_pool() -> ProcessPoolExecutor:
    global _compression_pool
    if _compression_pool is None:
        _compression_pool = ProcessPoolExecutor(max_workers=_COMPRESSION_WORKERS)
    return _compression_pool


def compression_ratio_batch(texts: list[str], parallel: bool = True) -> np.ndarray:
    """Compute compression ratios, optionally using multiprocessing."""
    n = len(texts)
    if not parallel or n < _COMPRESSION_CHUNK * 2:
        ratios = np.empty(n, dtype=np.float32)
        for i, t in enumerate(texts):
            ratios[i] = compression_ratio(t)
        return ratios

    pool = _get_compression_pool()
    chunks = [texts[i : i + _COMPRESSION_CHUNK]
              for i in range(0, n, _COMPRESSION_CHUNK)]
    results = list(pool.map(_compression_ratio_chunk, chunks))
    return np.array([r for chunk in results for r in chunk], dtype=np.float32)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

@dataclass
class PerfMetrics:
    docs: int = 0
    batches: int = 0
    parquet_read_s: float = 0.0
    parquet_convert_s: float = 0.0
    compression_filter_s: float = 0.0
    tokenization_s: float = 0.0
    h2d_s: float = 0.0
    forward_s: float = 0.0
    d2h_s: float = 0.0
    write_block_s: float = 0.0

    def merge(self, other: "PerfMetrics"):
        for key in asdict(self):
            setattr(self, key, getattr(self, key) + getattr(other, key))

    def to_dict(self):
        out = asdict(self)
        measured = sum(v for k, v in out.items() if k.endswith("_s"))
        if measured > 0:
            out["pct"] = {
                k.replace("_s", ""): round(v * 100.0 / measured, 2)
                for k, v in out.items() if k.endswith("_s")
            }
        return out


# ---------------------------------------------------------------------------
# Shard I/O
# ---------------------------------------------------------------------------

def shard_path(output_dir: Path, shard_idx: int) -> Path:
    return output_dir / f"shard_{shard_idx:04d}.parquet"


def count_existing_shards(output_dir: Path) -> int:
    i = 0
    while shard_path(output_dir, i).exists():
        i += 1
    return i


def get_completed_languages(output_dir: Path) -> set[str]:
    manifest = output_dir / "completed_languages.json"
    if manifest.exists():
        with open(manifest, encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("languages", []))

    langs: set[str] = set()
    i = 0
    while True:
        p = shard_path(output_dir, i)
        if not p.exists():
            break
        try:
            table = pq.read_table(p, columns=["lang"])
            langs.update(table.column("lang").to_pylist())
            del table
        except Exception:
            pass
        i += 1
    return langs


def save_completed_languages(output_dir: Path, langs: set[str]):
    manifest = output_dir / "completed_languages.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"languages": sorted(langs)}, f)


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


# ---------------------------------------------------------------------------
# AMP — bf16 is optimal for 5090's 5th-gen Tensor Cores
# ---------------------------------------------------------------------------

def get_amp_context(device, force_bf16=False):
    if device.type != "cuda":
        return contextlib.nullcontext()
    # RTX 5090 (Blackwell) and A100+ all support bf16 natively.
    # bf16 avoids the loss-scaling complexity of fp16 and matches Tensor Core
    # throughput on Blackwell. Always prefer bf16 when available.
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    print(f"AMP enabled (dtype={dtype})")
    return torch.amp.autocast("cuda", dtype=dtype)


# Pad to multiples of 64 for Tensor Core alignment on Blackwell.
# Fewer, rounder buckets reduce torch.compile recompilations.
_PAD_BUCKETS = (64, 128, 192, 256, 384, 512)


def warmup_model(model, device, amp_ctx, max_length, batch_size, steps):
    """Warmup all bucket shapes to trigger torch.compile graph capture.

    On RTX 5090 with max-autotune, the first pass through each shape triggers
    CUDA graph capture + autotuning. We run extra steps to ensure the graphs
    are fully compiled before the real workload starts.
    """
    if device.type != "cuda" or steps <= 0:
        return
    buckets = [b for b in _PAD_BUCKETS if b <= max_length]
    if not buckets or buckets[-1] < max_length:
        buckets.append(max_length)

    # Clamp warmup batch size to avoid OOM on the max_length bucket
    warmup_bs = min(batch_size, 2048)

    print(f"Warming up {len(buckets)} bucket shapes (bs={warmup_bs}): {buckets}")
    with torch.no_grad():
        for seq_len in buckets:
            ids = torch.ones((warmup_bs, seq_len), dtype=torch.long, device=device)
            mask = torch.ones((warmup_bs, seq_len), dtype=torch.long, device=device)
            for _ in range(steps):
                with amp_ctx:
                    _ = model(input_ids=ids, attention_mask=mask)
    torch.cuda.synchronize(device)
    # Clear warmup allocations so the real workload starts from a clean state
    torch.cuda.empty_cache()
    print("Warmup complete")


# ---------------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------------

def _build_per_lang_file_lists(local_dir, languages=None):
    """Group parquet files by language directory."""
    base = Path(local_dir)
    per_lang: dict[str, list[Path]] = {}

    for f in sorted(base.rglob("*.parquet")):
        lang_dir = f.parent.name
        if languages and lang_dir not in languages:
            continue
        per_lang.setdefault(lang_dir, []).append(f)

    return per_lang


# How many rows to yield from one language before rotating to the next.
# This ensures each shard (1M docs) gets ~equal representation per language.
_ROWS_PER_LANG_TURN = 50_000


def iter_local_parquets(local_dir, languages=None, batch_size=DEFAULT_READ_BATCH_SIZE, metrics=None):
    """Yield batches from local parquets, interleaved by language.

    Yields up to _ROWS_PER_LANG_TURN rows from one language, then rotates
    to the next. This ensures partial classifier runs get proportional
    coverage across all languages rather than exhausting one language first.
    """
    per_lang = _build_per_lang_file_lists(local_dir, languages)
    if not per_lang:
        raise FileNotFoundError(f"No parquet files found in {local_dir}")

    total_files = sum(len(v) for v in per_lang.values())
    lang_order = sorted(per_lang.keys())

    print(f"Found {total_files} local parquet files across {len(lang_order)} languages (interleaved)")
    for lang in lang_order:
        print(f"  {lang}: {len(per_lang[lang])} files")
    if languages:
        print(f"Filtering to languages: {languages}")

    # Per-language state: file iterator + leftover table rows from previous turn
    class LangState:
        def __init__(self, files):
            self.file_iter = iter(files)
            self.current_table = None
            self.table_offset = 0
            self.exhausted = False

    states = {lang: LangState(per_lang[lang]) for lang in lang_order}

    def _load_next_table(state, lang):
        """Load the next parquet file for this language. Returns False if exhausted."""
        while True:
            try:
                path = next(state.file_iter)
            except StopIteration:
                state.exhausted = True
                return False

            t0 = time.perf_counter()
            try:
                schema = pq.read_schema(path)
                available_cols = schema.names
            except Exception:
                continue

            cols = ["content", "size"]
            read_cols = [c for c in cols if c in available_cols]
            if "lang" in available_cols:
                read_cols.append("lang")
            elif "language" in available_cols:
                read_cols.append("language")

            if "content" not in available_cols:
                continue

            table = pq.read_table(path, columns=read_cols)
            if metrics is not None:
                metrics.parquet_read_s += time.perf_counter() - t0
            if table.num_rows == 0:
                continue

            state.current_table = table
            state.table_offset = 0
            return True

    def _take_rows(state, lang, max_rows):
        """Yield up to max_rows from this language's current position."""
        yielded = 0
        while yielded < max_rows:
            if state.current_table is None or state.table_offset >= state.current_table.num_rows:
                if not _load_next_table(state, lang):
                    return
                if state.current_table is None:
                    return

            table = state.current_table
            remaining_in_table = table.num_rows - state.table_offset
            remaining_in_turn = max_rows - yielded
            take = min(remaining_in_table, remaining_in_turn, batch_size)

            chunk = table.slice(state.table_offset, take)
            state.table_offset += take
            yielded += take
            yield chunk

    active_langs = list(lang_order)
    while active_langs:
        exhausted_this_round = []
        for lang in active_langs:
            state = states[lang]
            if state.exhausted:
                exhausted_this_round.append(lang)
                continue

            rows_this_turn = 0
            for chunk in _take_rows(state, lang, _ROWS_PER_LANG_TURN):
                # Normalize column names and yield in the expected format
                rb = chunk
                content_col = rb.column("content")

                lang_col = None
                if "lang" in rb.column_names:
                    lang_col = "lang"
                elif "language" in rb.column_names:
                    lang_col = "language"

                lang_values = None
                if lang_col:
                    lang_values = rb.column(lang_col)
                else:
                    lang_values = pa.array([lang] * rb.num_rows)

                size_values = None
                if "size" in rb.column_names:
                    size_values = rb.column("size").fill_null(0).to_numpy(zero_copy_only=False)
                else:
                    size_values = np.array([len(str(t)) for t in content_col.to_pylist()], dtype=np.int64)

                token_counts = (size_values // 4).astype(np.int64)

                payload = {
                    "content": content_col,
                    "lang": lang_values,
                    "size": size_values,
                    "token_count": token_counts,
                }
                yield {"__batch__": payload}
                rows_this_turn += rb.num_rows

            if state.exhausted and rows_this_turn == 0:
                exhausted_this_round.append(lang)

        for lang in exhausted_this_round:
            active_langs.remove(lang)


def collect_shard_buffer(dataset_iter, shard_size, compression_floor, pending_batch=None):
    """Collect documents into a shard buffer, applying compression ratio pre-filter."""
    buf = {"content": [], "lang": [], "size": [], "token_count": []}
    collected = 0
    dropped_compression = 0
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
            contents = b["content"]
            langs = b["lang"]
            sizes = b["size"]
            toks = b["token_count"]

            if isinstance(contents, (pa.Array, pa.ChunkedArray)):
                text_list = contents.to_pylist()
            else:
                text_list = list(contents)

            n = len(text_list)
            if n == 0:
                continue

            # Compression ratio pre-filter
            if compression_floor > 0:
                ratios = compression_ratio_batch(text_list)
                keep_mask = ratios >= compression_floor
                dropped_compression += int((~keep_mask).sum())
                keep_indices = np.where(keep_mask)[0]
                if len(keep_indices) == 0:
                    continue
            else:
                keep_indices = np.arange(n)

            take = min(shard_size - collected, len(keep_indices))
            indices_to_take = keep_indices[:take]

            for i in indices_to_take:
                buf["content"].append(text_list[i])
                if isinstance(langs, (pa.Array, pa.ChunkedArray)):
                    buf["lang"].append(langs[i].as_py() if isinstance(langs[i], pa.Scalar) else str(langs[i]))
                else:
                    buf["lang"].append(str(langs[i]) if i < len(langs) else "unknown")
                if isinstance(sizes, np.ndarray):
                    buf["size"].append(int(sizes[i]))
                elif isinstance(sizes, (pa.Array, pa.ChunkedArray)):
                    buf["size"].append(int(sizes[i].as_py()))
                else:
                    buf["size"].append(int(sizes[i]))
                if isinstance(toks, np.ndarray):
                    buf["token_count"].append(int(toks[i]))
                else:
                    buf["token_count"].append(int(toks[i]))

            collected += take

            if take < len(keep_indices):
                remaining = keep_indices[take:]
                pending = {
                    "content": [text_list[i] for i in remaining],
                    "lang": [langs[i].as_py() if isinstance(langs, pa.Array) and isinstance(langs[i], pa.Scalar) else str(langs[i]) for i in remaining],
                    "size": [int(sizes[i]) if isinstance(sizes, np.ndarray) else int(sizes[i].as_py()) if isinstance(sizes, pa.Array) else int(sizes[i]) for i in remaining],
                    "token_count": [int(toks[i]) if isinstance(toks, np.ndarray) else int(toks[i]) for i in remaining],
                }
        else:
            txt = item.get("content", "") or item.get("text", "")
            if not txt:
                continue
            if compression_floor > 0 and compression_ratio(txt) < compression_floor:
                dropped_compression += 1
                continue
            buf["content"].append(txt)
            buf["lang"].append(item.get("lang", item.get("language", "unknown")))
            buf["size"].append(item.get("size", len(txt)))
            buf["token_count"].append(item.get("token_count", len(txt.split())))
            collected += 1

    if collected == 0:
        return None, 0, exhausted, pending, dropped_compression
    return buf, collected, exhausted, pending, dropped_compression


# ---------------------------------------------------------------------------
# Batching and inference
# ---------------------------------------------------------------------------

def _bucket_pad_length(seq_len: int, max_length: int) -> int:
    for bucket in _PAD_BUCKETS:
        if seq_len <= bucket <= max_length:
            return bucket
    return max_length


def prepare_chunk_batches(indices, texts, tokenizer, max_length, batch_size,
                          pin_memory=False, max_chars_per_doc=0):
    if max_chars_per_doc and max_chars_per_doc > 0:
        texts = [t if len(t) <= max_chars_per_doc else t[:max_chars_per_doc] for t in texts]

    t0_tok = time.perf_counter()
    enc = tokenizer(
        texts, max_length=max_length, truncation=True,
        padding=False, return_attention_mask=False, return_length=True,
    )
    tokenization_s = time.perf_counter() - t0_tok
    ids_list = enc["input_ids"]
    order = np.argsort(np.asarray(enc["length"]))

    batches = []
    for bs in range(0, len(order), batch_size):
        idx = order[bs : bs + batch_size]
        batch_indices = [indices[i] for i in idx]
        batch_ids = [ids_list[i] for i in idx]
        longest = max(len(batch_ids[j]) for j in range(len(batch_ids)))
        pad_to = _bucket_pad_length(longest, max_length)
        padded = tokenizer.pad(
            {"input_ids": batch_ids}, padding="max_length",
            max_length=pad_to, return_tensors="pt",
        )
        input_ids = padded["input_ids"].to(torch.long)
        attention_mask = padded["attention_mask"].to(torch.long)
        if pin_memory:
            input_ids = input_ids.pin_memory()
            attention_mask = attention_mask.pin_memory()
        batches.append((batch_indices, input_ids, attention_mask))

    return batches, tokenization_s


# ---------------------------------------------------------------------------
# Prefetch tokenization — CPU prepares next chunk while GPU runs inference
# ---------------------------------------------------------------------------

def iter_prepared_chunks(texts, tokenizer, max_length, batch_size,
                         chunk_size, pin_memory, prefetch_chunks,
                         max_chars_per_doc):
    """Yield (chunk_idx, num_chunks, batches, tok_s) with optional prefetch."""
    n = len(texts)
    num_chunks = (n + chunk_size - 1) // chunk_size

    def _prepare(chunk_idx):
        s = chunk_idx * chunk_size
        e = min(s + chunk_size, n)
        indices = list(range(s, e))
        chunk_texts = texts[s:e]
        batches, tok_s = prepare_chunk_batches(
            indices, chunk_texts, tokenizer, max_length, batch_size,
            pin_memory=pin_memory, max_chars_per_doc=max_chars_per_doc,
        )
        return chunk_idx, num_chunks, batches, tok_s

    if prefetch_chunks <= 0:
        for ci in range(num_chunks):
            yield _prepare(ci)
        return

    queue: Queue = Queue(maxsize=max(1, prefetch_chunks))
    sentinel = object()

    def _producer():
        try:
            for ci in range(num_chunks):
                queue.put(_prepare(ci))
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
# Scatter results from D2H into output arrays
# ---------------------------------------------------------------------------

def _scatter_results(all_quality, all_sd, all_ct, batch_indices,
                     q_cpu, sd_cpu, ct_cpu):
    idx = np.asarray(batch_indices, dtype=np.int64)
    all_quality[idx] = q_cpu.numpy() if hasattr(q_cpu, 'numpy') else q_cpu
    all_sd[idx] = sd_cpu.numpy() if hasattr(sd_cpu, 'numpy') else sd_cpu
    ct_vals = ct_cpu.numpy().argmax(axis=1) if ct_cpu.ndim == 2 else ct_cpu
    all_ct[idx] = ct_vals


def classify_buffer(texts, tokenizer, model, device, amp_ctx, args):
    """Run the multi-task classifier on a buffer of texts.

    Optimized for RTX 5090 with dual CUDA streams:
      - Transfer stream: overlaps H2D copies with compute
      - Compute stream: runs model forward passes
      - Async D2H: non-blocking copy back with event synchronization
      - Prefetch: CPU tokenizes next chunk while GPU processes current one

    Returns (quality, structured_data, content_type_idx) arrays.
    """
    n = len(texts)
    all_quality = np.empty(n, dtype=np.float32)
    all_sd = np.empty(n, dtype=np.float32)
    all_ct = np.empty(n, dtype=np.int32)
    metrics = PerfMetrics(docs=n)
    pbar = tqdm(total=n, desc="Classifying shard", unit="doc", leave=False)

    chunk_size = max(1, min(
        getattr(args, 'tokenize_chunk_size', DEFAULT_TOKENIZE_CHUNK_SIZE),
        getattr(args, 'tokenize_subchunk_size', DEFAULT_TOKENIZE_SUBCHUNK_SIZE),
    ))
    use_streams = device.type == "cuda"
    max_chars = getattr(args, 'max_chars_per_doc', 0)
    prefetch = getattr(args, 'prefetch_chunks', 1) if use_streams else 0

    # Dual CUDA streams for overlapping transfers with compute
    if use_streams:
        compute_stream = torch.cuda.default_stream(device)
        transfer_stream = torch.cuda.Stream(device)

    def _to_device(batch_tuple):
        batch_indices, b_ids, b_mask = batch_tuple
        t0_h2d = time.perf_counter()
        if use_streams:
            with torch.cuda.stream(transfer_stream):
                ids = b_ids.to(device, non_blocking=True)
                mask = b_mask.to(device, non_blocking=True)
        else:
            ids = b_ids.to(device, non_blocking=True)
            mask = b_mask.to(device, non_blocking=True)
        metrics.h2d_s += time.perf_counter() - t0_h2d
        return ids, mask, batch_indices

    def _start_d2h(q_pred, sd_pred, ct_logits):
        """Kick off non-blocking D2H copies on the transfer stream."""
        t0_d2h = time.perf_counter()
        if use_streams:
            with torch.cuda.stream(transfer_stream):
                transfer_stream.wait_stream(compute_stream)
                q_cpu = q_pred.float().to("cpu", non_blocking=True)
                sd_cpu = sd_pred.float().to("cpu", non_blocking=True)
                ct_cpu = ct_logits.float().to("cpu", non_blocking=True)
            d2h_event = transfer_stream.record_event()
        else:
            q_cpu = q_pred.float().cpu()
            sd_cpu = sd_pred.float().cpu()
            ct_cpu = ct_logits.float().cpu()
            d2h_event = None
        metrics.d2h_s += time.perf_counter() - t0_d2h
        return q_cpu, sd_cpu, ct_cpu, d2h_event

    # Pipeline: process previous batch's D2H while current batch runs forward
    pending_d2h = None  # (batch_indices, q_cpu, sd_cpu, ct_cpu, d2h_event, count)

    def _flush_pending():
        nonlocal pending_d2h
        if pending_d2h is None:
            return
        p_indices, q_cpu, sd_cpu, ct_cpu, d2h_event, count = pending_d2h
        if d2h_event is not None:
            d2h_event.synchronize()
        _scatter_results(all_quality, all_sd, all_ct, p_indices,
                         q_cpu, sd_cpu, ct_cpu)
        metrics.batches += 1
        pbar.update(count)
        pending_d2h = None

    with torch.no_grad():
        for _, _, batches, tok_s in iter_prepared_chunks(
            texts=texts,
            tokenizer=tokenizer,
            max_length=args.max_length,
            batch_size=args.batch_size,
            chunk_size=chunk_size,
            pin_memory=use_streams,
            prefetch_chunks=prefetch,
            max_chars_per_doc=max_chars,
        ):
            metrics.tokenization_s += tok_s
            if not batches:
                continue

            # Pipeline: prefetch first batch H2D
            batch_idx = 0
            next_ids, next_mask, next_indices = _to_device(batches[batch_idx])

            while batch_idx < len(batches):
                input_ids = next_ids
                attention_mask = next_mask
                batch_indices = next_indices

                if use_streams:
                    compute_stream.wait_stream(transfer_stream)

                # Start next batch H2D while current batch runs forward
                batch_idx += 1
                if batch_idx < len(batches):
                    next_ids, next_mask, next_indices = _to_device(batches[batch_idx])

                # Flush previous batch's D2H results (overlapped with this H2D)
                _flush_pending()

                t0_fwd = time.perf_counter()
                with amp_ctx:
                    q_pred, sd_pred, ct_logits = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                    )
                metrics.forward_s += time.perf_counter() - t0_fwd

                # Start async D2H
                q_cpu, sd_cpu, ct_cpu, d2h_event = _start_d2h(
                    q_pred, sd_pred, ct_logits,
                )
                pending_d2h = (batch_indices, q_cpu, sd_cpu, ct_cpu,
                               d2h_event, len(batch_indices))

    # Flush final pending D2H
    _flush_pending()

    pbar.close()
    return all_quality, all_sd, all_ct, metrics


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        cc = torch.cuda.get_device_capability(device)
        vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
        print(f"GPU: {gpu_name} (cc {cc[0]}.{cc[1]}, {vram_gb:.1f} GB)")

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        # GDDR7 on 5090 benefits from expandable memory segments to avoid
        # fragmentation during dynamic-shape inference with torch.compile.
        try:
            os.environ.setdefault(
                "PYTORCH_CUDA_ALLOC_CONF",
                "expandable_segments:True",
            )
        except Exception:
            pass

        # Enable cudnn benchmark for conv-heavy layers (if any in encoder)
        torch.backends.cudnn.benchmark = True

    # Defaults
    args.tokenize_chunk_size = getattr(args, 'tokenize_chunk_size', None) or DEFAULT_TOKENIZE_CHUNK_SIZE
    args.tokenize_subchunk_size = getattr(args, 'tokenize_subchunk_size', None) or DEFAULT_TOKENIZE_SUBCHUNK_SIZE
    args.max_chars_per_doc = getattr(args, 'max_chars_per_doc', None) or 8192
    args.write_queue_size = getattr(args, 'write_queue_size', None) or DEFAULT_WRITE_QUEUE_SIZE
    args.prefetch_chunks = getattr(args, 'prefetch_chunks', None)
    if args.prefetch_chunks is None:
        args.prefetch_chunks = 2 if device.type == "cuda" else 0

    print(f"Runtime config: batch={args.batch_size}, max_len={args.max_length}, "
          f"compression_floor={COMPRESSION_RATIO_FLOOR}, compile={args.compile}, "
          f"prefetch={args.prefetch_chunks}")

    tokenizer, model = load_classifier(args.classifier_model, device, args.compile)
    amp_ctx = get_amp_context(device)
    warmup_model(model, device, amp_ctx, args.max_length, args.batch_size, args.warmup_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_log = output_dir / "progress.jsonl"
    perf_log = output_dir / "perf_metrics.jsonl"

    resume_shard = count_existing_shards(output_dir)
    already_done = get_completed_languages(output_dir) if resume_shard > 0 else set()

    effective_langs = args.languages
    if already_done and effective_langs:
        skipped = set(effective_langs) & already_done
        effective_langs = [l for l in effective_langs if l not in already_done]
        if skipped:
            print(f"Skipping already-processed languages: {sorted(skipped)}")

    if effective_langs is not None and len(effective_langs) == 0:
        print("All requested languages have already been processed. Nothing to do.")
        return

    if resume_shard > 0:
        print(f"Resuming: {resume_shard} existing shards")

    io_metrics = PerfMetrics()
    if args.local_dir:
        ds_iter = iter_local_parquets(
            args.local_dir, languages=effective_langs,
            batch_size=DEFAULT_READ_BATCH_SIZE, metrics=io_metrics,
        )
    else:
        lang = effective_langs[0] if effective_langs else "python"
        ds = load_dataset(args.dataset, lang, split="train", streaming=True)
        ds_iter = ({"content": ex.get("content", ""), "lang": lang,
                     "size": len(ex.get("content", "")),
                     "token_count": len(ex.get("content", "").split())}
                    for ex in ds)

    # Writer thread
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

    shard_idx = resume_shard
    processed_shards = 0
    total_docs = 0
    total_tokens = 0
    total_dropped_compression = 0
    run_langs: set[str] = set()
    total_metrics = PerfMetrics()
    t_start = time.time()

    pending = None
    exhausted = False
    while not exhausted:
        buf, collected, exhausted, pending, dropped = collect_shard_buffer(
            ds_iter, args.shard_size, COMPRESSION_RATIO_FLOOR, pending,
        )
        total_dropped_compression += dropped
        if buf is None:
            break

        print(f"\nShard {shard_idx:04d}: classifying {collected:,} code files "
              f"(dropped {dropped:,} by compression filter)...")
        shard_t0 = time.time()
        quality, structured_data, content_type_idx, shard_metrics = classify_buffer(
            buf["content"], tokenizer, model, device, amp_ctx, args,
        )

        content_type_names = [CONTENT_TYPES[i] for i in content_type_idx]

        buf["quality"] = pa.array(quality, type=pa.float32())
        buf["structured_data"] = pa.array(structured_data, type=pa.float32())
        buf["content_type"] = pa.array(content_type_names, type=pa.string())
        buf["content_type_idx"] = pa.array(content_type_idx, type=pa.int32())

        shard_elapsed = time.time() - shard_t0
        shard_rate = collected / shard_elapsed if shard_elapsed > 0 else 0
        shard_tokens = int(np.sum(np.asarray(buf["token_count"], dtype=np.int64)))
        total_docs += collected
        total_tokens += shard_tokens

        shard_langs = set(buf["lang"]) - {None, ""}
        run_langs.update(shard_langs)

        t0_block = time.perf_counter()
        write_queue.put((shard_idx, buf, collected))
        shard_metrics.write_block_s += time.perf_counter() - t0_block
        total_metrics.merge(shard_metrics)
        total_metrics.parquet_read_s = io_metrics.parquet_read_s
        total_metrics.parquet_convert_s = io_metrics.parquet_convert_s

        elapsed = time.time() - t_start
        overall_rate = total_docs / elapsed if elapsed > 0 else 0
        print(f"  shard_rate={shard_rate:,.0f} docs/s | overall={overall_rate:,.0f} docs/s | "
              f"total_docs={total_docs:,}")

        with open(progress_log, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "shard_idx": shard_idx,
                "docs_in_shard": collected,
                "tokens_in_shard": shard_tokens,
                "total_docs": total_docs,
                "total_tokens": total_tokens,
                "total_dropped_compression": total_dropped_compression,
                "shard_rate": round(shard_rate, 1),
                "overall_rate": round(overall_rate, 1),
                "elapsed_sec": round(elapsed, 1),
                "languages_in_shard": sorted(shard_langs),
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

    write_queue.put(write_sentinel)
    wt.join()

    all_completed = already_done | run_langs
    save_completed_languages(output_dir, all_completed)

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print("Classification complete")
    print(f"Shards written: {shard_idx - resume_shard}")
    print(f"Code files: {total_docs:,}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Dropped (compression filter): {total_dropped_compression:,}")
    print(f"Elapsed: {elapsed:.1f}s")
    if elapsed > 0:
        print(f"Throughput: {total_docs / elapsed:,.0f} docs/s")
    print(f"Languages processed: {sorted(run_langs)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stage 1: classify StarCoder code files")
    parser.add_argument("--output-dir", default=f"{DEFAULT_OUTPUT_BASE}/scored_shards")
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile (max-autotune on 5090/Blackwell)")
    parser.add_argument("--warmup-steps", type=int, default=6,
                        help="Warmup steps per bucket shape (default: 6 for torch.compile)")
    parser.add_argument("--prefetch-chunks", type=int, default=None,
                        help="Tokenization prefetch depth (default: 2 on CUDA)")
    parser.add_argument("--tokenize-chunk-size", type=int, default=None)
    parser.add_argument("--tokenize-subchunk-size", type=int, default=None)
    parser.add_argument("--max-chars-per-doc", type=int, default=None)
    parser.add_argument("--write-queue-size", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--languages", nargs="*", default=RECOMMENDED_LANGUAGES)
    parser.add_argument("--local-dir")
    args = parser.parse_args()

    if args.languages is not None and len(args.languages) == 0:
        args.languages = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    run(args)


if __name__ == "__main__":
    main()
