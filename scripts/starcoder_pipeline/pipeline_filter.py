#!/usr/bin/env python3
"""
Stage 2: Filter scored code files using per-language-slice token budgets.

Two data sources:
  - Scored shards (--input-dir): classifier output for relevance_filter languages
    (python, typescript, rust, go, java)
  - Raw parquets (--raw-dir): original download for non-classified slices
    (schema languages, jupyter, github issues)

Each language slice has its own filtering strategy:
  - schema_languages (JSON/YAML/SQL/protocol-buffer/thrift): token budget, random sample
  - typescript (relevance ≥ 2): relevance classifier
  - python (relevance ≥ 2): relevance classifier
  - rust_go_java (relevance ≥ 2): relevance classifier
  - jupyter: passthrough with token budget
  - github_issues: keyword filter for structured data topics

Total target: ~3.5B tokens across all slices.

Usage:
  python pipeline_filter.py \
      --input-dir scored_shards/ \
      --raw-dir /workspace/starcoder-curation/raw_data \
      --output-dir filtered_shards/
"""

import argparse
import json
import logging
import time
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from pipeline_config import (
    CLASSIFIER_LANGUAGES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    GITHUB_ISSUES_KEYWORDS,
    LANGUAGE_SLICES,
    LANGUAGE_SLICE_MAP,
    QUALITY_HARD_FLOOR,
    RANDOM_SEED,
    LanguageSlice,
    compute_relevance_score_batch,
    resolve_language_to_slice,
    text_matches_keywords,
)

logger = logging.getLogger(__name__)

# Output schema for filtered parquets (unified across both sources)
OUTPUT_COLUMNS = [
    "content", "lang", "size", "token_count",
    "quality", "structured_data", "content_type",
    "language_slice", "relevance_score",
]


def _cast_content_large_string(table: pa.Table) -> pa.Table:
    """Cast the 'content' column to large_string to avoid 2GB offset overflow on take()."""
    if "content" in table.column_names:
        i = table.schema.get_field_index("content")
        if table.schema.field(i).type == pa.string():
            table = table.set_column(i, "content",
                                     table.column(i).cast(pa.large_string()))
    return table


# ---------------------------------------------------------------------------
# Scan scored shards (classifier output — relevance_filter slices only)
# ---------------------------------------------------------------------------

def scan_scored_shards(input_dir: Path, min_tokens: int, max_tokens: int):
    """Scan classifier output shards for relevance_filter languages."""
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        print(f"  No scored shards found in {input_dir}")
        return {}, []

    print(f"  Found {len(shard_files)} scored shards")

    slice_data: dict[str, dict] = {}
    for s in LANGUAGE_SLICES:
        if s.strategy == "relevance_filter":
            slice_data[s.name] = _empty_slice_accum()

    shard_map = []
    global_offset = 0
    total_raw = 0
    stats = {"dropped_quality": 0, "dropped_short": 0, "dropped_long": 0}

    for shard_path in tqdm(shard_files, desc="Scanning scored shards"):
        table = pq.read_table(shard_path,
                              columns=["quality", "structured_data", "content_type",
                                       "token_count", "lang"])
        n_rows = table.num_rows
        total_raw += n_rows

        quality = table.column("quality").to_numpy().astype(np.float32)
        sd = table.column("structured_data").to_numpy().astype(np.float32)
        ct = table.column("content_type").to_pylist()
        token_counts = table.column("token_count").to_numpy().astype(np.int64)
        langs = table.column("lang").to_pylist()
        del table

        short_mask = token_counts < min_tokens
        long_mask = token_counts > max_tokens
        q_mask = quality <= QUALITY_HARD_FLOOR

        stats["dropped_quality"] += int(q_mask.sum())
        stats["dropped_short"] += int(short_mask.sum())
        stats["dropped_long"] += int(long_mask.sum())

        keep = ~q_mask & ~short_mask & ~long_mask
        kept_indices = np.where(keep)[0]
        if len(kept_indices) == 0:
            continue

        shard_map.append((shard_path, kept_indices))

        for pos, idx in enumerate(kept_indices):
            lang = langs[idx] or "unknown"
            slice_name = resolve_language_to_slice(lang)
            if slice_name is None or slice_name not in slice_data:
                continue

            sd_val = slice_data[slice_name]
            sd_val["quality"].append(float(quality[idx]))
            sd_val["structured_data"].append(float(sd[idx]))
            sd_val["content_types"].append(ct[idx])
            sd_val["token_counts"].append(int(token_counts[idx]))
            sd_val["langs"].append(lang)
            sd_val["global_indices"].append(global_offset + pos)

        global_offset += len(kept_indices)

    _finalize_slice_data(slice_data)

    print(f"  Scored shards: {total_raw:,} total rows")
    print(f"    Dropped quality: {stats['dropped_quality']:,}, "
          f"short: {stats['dropped_short']:,}, long: {stats['dropped_long']:,}")
    for sname, sd_val in slice_data.items():
        tok = sd_val["token_counts"].sum() if sd_val["n_docs"] > 0 else 0
        s = LANGUAGE_SLICE_MAP[sname]
        print(f"    {sname:<25s}: {sd_val['n_docs']:>10,} files, "
              f"{tok:>15,} tokens (target: {s.target_tokens:,})")

    return slice_data, shard_map


# ---------------------------------------------------------------------------
# Scan raw parquets (non-classified slices)
# ---------------------------------------------------------------------------

def scan_raw_parquets(raw_dir: Path, min_tokens: int, max_tokens: int):
    """Scan raw download parquets for non-classified language slices."""
    non_classified = [s for s in LANGUAGE_SLICES if s.strategy != "relevance_filter"]
    if not non_classified:
        return {}, []

    # Build language -> slice mapping for non-classified only
    lang_to_slice: dict[str, str] = {}
    for s in non_classified:
        for lang in s.languages:
            lang_to_slice[lang.lower()] = s.name

    slice_data: dict[str, dict] = {}
    for s in non_classified:
        slice_data[s.name] = _empty_slice_accum(include_texts=s.strategy == "keyword_filter")

    raw_files = []
    for s in non_classified:
        for lang in s.languages:
            lang_dir = raw_dir / lang
            if lang_dir.is_dir():
                found = sorted(lang_dir.glob("*.parquet"))
                raw_files.extend(found)
                print(f"    {lang}: {len(found)} parquet files")
            else:
                print(f"    {lang}: directory not found at {lang_dir}")

    if not raw_files:
        print("  No raw parquet files found for non-classified slices")
        return slice_data, []

    print(f"  Found {len(raw_files)} raw parquet files for non-classified slices")

    # Slices that need text content during scan (e.g. keyword_filter)
    slices_needing_text = {s.name for s in non_classified if s.strategy == "keyword_filter"}

    file_map = []  # (file_path, kept_indices_array)
    global_offset = 0
    total_raw = 0

    for fpath in tqdm(raw_files, desc="Scanning raw parquets"):
        schema = pq.read_schema(fpath)
        available_cols = set(schema.names)

        # Determine which slice this file belongs to
        inferred_lang = fpath.parent.name.lower()
        slice_name = lang_to_slice.get(inferred_lang)
        needs_text = slice_name in slices_needing_text

        read_cols = []
        has_size = "size" in available_cols
        has_content = "content" in available_cols

        if has_size:
            read_cols.append("size")
        if needs_text and has_content:
            read_cols.append("content")
        elif not has_size and has_content:
            read_cols.append("content")

        lang_col = None
        if "lang" in available_cols:
            lang_col = "lang"
            read_cols.append("lang")
        elif "language" in available_cols:
            lang_col = "language"
            read_cols.append("language")

        if not has_size and not has_content:
            continue

        table = pq.read_table(fpath, columns=read_cols)
        n_rows = table.num_rows
        total_raw += n_rows

        texts = None
        if "content" in table.column_names:
            if needs_text:
                texts = table.column("content").to_pylist()

        if "size" in table.column_names:
            sizes = table.column("size").to_numpy(zero_copy_only=False).astype(np.int64)
        elif "content" in table.column_names:
            content_col = table.column("content")
            if texts is not None:
                sizes = np.array([len(t.encode("utf-8", errors="replace")) if t else 0
                                  for t in texts], dtype=np.int64)
            else:
                sizes = np.array([len(s.as_py().encode("utf-8", errors="replace")) if s.as_py() else 0
                                  for s in content_col], dtype=np.int64)
        else:
            del table
            continue

        token_counts = sizes // 4

        if lang_col and lang_col in table.column_names:
            langs = table.column(lang_col).to_pylist()
        else:
            langs = [fpath.parent.name] * n_rows

        del table

        short_mask = token_counts < min_tokens
        long_mask = token_counts > max_tokens
        keep = ~short_mask & ~long_mask
        kept_indices = np.where(keep)[0]
        if len(kept_indices) == 0:
            continue

        file_map.append((fpath, kept_indices))

        for pos, idx in enumerate(kept_indices):
            lang = (langs[idx] or fpath.parent.name).lower()
            sname = lang_to_slice.get(lang)
            if sname is None:
                continue

            sd_val = slice_data[sname]
            sd_val["token_counts"].append(int(token_counts[idx]))
            sd_val["langs"].append(lang)
            sd_val["global_indices"].append(global_offset + pos)
            if "texts" in sd_val and texts is not None:
                sd_val["texts"].append(texts[idx] or "")

        global_offset += len(kept_indices)

    _finalize_slice_data(slice_data)

    print(f"  Raw parquets: {total_raw:,} total rows")
    for sname, sd_val in slice_data.items():
        tok = sd_val["token_counts"].sum() if sd_val["n_docs"] > 0 else 0
        s = LANGUAGE_SLICE_MAP[sname]
        print(f"    {sname:<25s}: {sd_val['n_docs']:>10,} files, "
              f"{tok:>15,} tokens (target: {s.target_tokens:,})")

    return slice_data, file_map


# ---------------------------------------------------------------------------
# Helpers for slice data accumulators
# ---------------------------------------------------------------------------

def _empty_slice_accum(include_texts=False):
    d = {
        "quality": [],
        "structured_data": [],
        "content_types": [],
        "token_counts": [],
        "langs": [],
        "global_indices": [],
    }
    if include_texts:
        d["texts"] = []
    return d


def _finalize_slice_data(slice_data):
    for sd_val in slice_data.values():
        sd_val["quality"] = np.array(sd_val["quality"], dtype=np.float32) if sd_val["quality"] else np.array([], dtype=np.float32)
        sd_val["structured_data"] = np.array(sd_val["structured_data"], dtype=np.float32) if sd_val["structured_data"] else np.array([], dtype=np.float32)
        sd_val["token_counts"] = np.array(sd_val["token_counts"], dtype=np.int64) if sd_val["token_counts"] else np.array([], dtype=np.int64)
        sd_val["global_indices"] = np.array(sd_val["global_indices"], dtype=np.int64) if sd_val["global_indices"] else np.array([], dtype=np.int64)
        sd_val["n_docs"] = len(sd_val["token_counts"])


# ---------------------------------------------------------------------------
# Per-slice filtering
# ---------------------------------------------------------------------------

def filter_slice(slice_def: LanguageSlice, slice_meta: dict, rng) -> np.ndarray:
    """Apply the slice's filtering strategy and return selected global indices."""
    n = slice_meta["n_docs"]
    if n == 0:
        return np.array([], dtype=np.int64)

    token_counts = slice_meta["token_counts"]
    global_indices = slice_meta["global_indices"]
    target = slice_def.target_tokens

    if slice_def.strategy == "relevance_filter":
        quality = slice_meta["quality"]
        sd = slice_meta["structured_data"]
        return _filter_by_relevance(quality, sd, token_counts, global_indices,
                                    target, slice_def.min_relevance,
                                    slice_def.min_quality, rng)

    elif slice_def.strategy == "light_filter":
        return _filter_to_budget_random(token_counts, global_indices, target, rng)

    elif slice_def.strategy == "passthrough":
        return _filter_to_budget_random(token_counts, global_indices, target, rng)

    elif slice_def.strategy == "keyword_filter":
        texts = slice_meta.get("texts", [])
        return _filter_by_keywords(token_counts, global_indices,
                                   texts, target, rng)

    else:
        logger.warning(f"Unknown strategy '{slice_def.strategy}' for {slice_def.name}")
        return _filter_to_budget_random(token_counts, global_indices, target, rng)


def _filter_by_relevance(quality, sd, token_counts, global_indices,
                         target_tokens, min_relevance, min_quality, rng):
    """Relevance filter: structured_data >= threshold, rank by relevance, sample to budget."""
    mask = (quality > min_quality) & (sd >= min_relevance)
    candidates = np.where(mask)[0]

    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    relevance = compute_relevance_score_batch(sd[candidates], quality[candidates])
    order = np.argsort(-relevance)
    sorted_cands = candidates[order]

    available_tokens = token_counts[sorted_cands].sum()
    if available_tokens <= target_tokens:
        return global_indices[sorted_cands]

    cumtok = np.cumsum(token_counts[sorted_cands])
    n_take = int(np.searchsorted(cumtok, target_tokens, side="right")) + 1
    n_take = min(n_take, len(sorted_cands))
    return global_indices[sorted_cands[:n_take]]


def _filter_to_budget_random(token_counts, global_indices, target_tokens, rng):
    """Random sample to token budget (for non-classified slices)."""
    available_tokens = token_counts.sum()
    if available_tokens <= target_tokens:
        return global_indices.copy()

    # Shuffle and take until budget
    order = rng.permutation(len(token_counts))
    cumtok = np.cumsum(token_counts[order])
    n_take = int(np.searchsorted(cumtok, target_tokens, side="right")) + 1
    n_take = min(n_take, len(order))
    return global_indices[order[:n_take]]


def _filter_by_keywords(token_counts, global_indices, texts, target_tokens, rng):
    """Keyword filter: match text against structured data keywords, then sample."""
    pattern = re.compile("|".join(re.escape(kw) for kw in GITHUB_ISSUES_KEYWORDS), re.IGNORECASE)
    keyword_mask = np.fromiter(
        (pattern.search(t) is not None for t in tqdm(texts, desc="    Keyword matching", unit="doc")),
        dtype=bool, count=len(texts),
    )
    candidates = np.where(keyword_mask)[0]

    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    available_tokens = token_counts[candidates].sum()
    if available_tokens <= target_tokens:
        return global_indices[candidates]

    order = rng.permutation(len(candidates))
    sorted_cands = candidates[order]
    cumtok = np.cumsum(token_counts[sorted_cands])
    n_take = int(np.searchsorted(cumtok, target_tokens, side="right")) + 1
    n_take = min(n_take, len(sorted_cands))
    return global_indices[sorted_cands[:n_take]]


# ---------------------------------------------------------------------------
# Pass 2: Emit filtered parquets
# ---------------------------------------------------------------------------

def emit_scored_filtered(shard_map, selected_globals: set, global_to_slice: dict,
                         writer_state: dict):
    """Write selected rows from scored shards."""
    global_offset = 0

    for shard_path, kept_indices in tqdm(shard_map, desc="Writing from scored shards"):
        n_kept = len(kept_indices)
        emit_positions = []
        emit_slices = []

        for pos in range(n_kept):
            gidx = global_offset + pos
            if gidx in selected_globals:
                emit_positions.append(pos)
                emit_slices.append(global_to_slice.get(gidx, "unknown"))

        global_offset += n_kept
        if not emit_positions:
            continue

        original_rows = kept_indices[np.array(emit_positions)]
        table = _cast_content_large_string(pq.read_table(shard_path))
        emit_table = table.take(original_rows)
        del table

        emit_table = emit_table.append_column(
            "language_slice", pa.array(emit_slices, type=pa.string()),
        )

        sd_col = emit_table.column("structured_data").to_numpy().astype(np.float32)
        q_col = emit_table.column("quality").to_numpy().astype(np.float32)
        rel_scores = compute_relevance_score_batch(sd_col, q_col)
        emit_table = emit_table.append_column(
            "relevance_score", pa.array(rel_scores.tolist(), type=pa.float32()),
        )

        _write_chunk(emit_table, writer_state)
        del emit_table


def emit_raw_filtered(file_map, selected_globals: set, global_to_slice: dict,
                      writer_state: dict):
    """Write selected rows from raw parquet files (non-classified slices)."""
    global_offset = 0

    for fpath, kept_indices in tqdm(file_map, desc="Writing from raw parquets"):
        n_kept = len(kept_indices)
        emit_positions = []
        emit_slices = []

        for pos in range(n_kept):
            gidx = global_offset + pos
            if gidx in selected_globals:
                emit_positions.append(pos)
                emit_slices.append(global_to_slice.get(gidx, "unknown"))

        global_offset += n_kept
        if not emit_positions:
            continue

        original_rows = kept_indices[np.array(emit_positions)]
        table = _cast_content_large_string(pq.read_table(fpath))
        emit_table = table.take(original_rows)
        del table

        n_emit = emit_table.num_rows

        # Normalize column names
        col_names = set(emit_table.column_names)

        if "language" in col_names and "lang" not in col_names:
            lang_arr = emit_table.column("language")
            emit_table = emit_table.drop("language")
            emit_table = emit_table.append_column("lang", lang_arr)

        if "lang" not in emit_table.column_names:
            inferred = fpath.parent.name
            emit_table = emit_table.append_column(
                "lang", pa.array([inferred] * n_emit, type=pa.string()))

        if "size" not in emit_table.column_names:
            if "content" in emit_table.column_names:
                sizes = [len((t or "").encode("utf-8", errors="replace"))
                         for t in emit_table.column("content").to_pylist()]
                emit_table = emit_table.append_column(
                    "size", pa.array(sizes, type=pa.int64()))
            else:
                emit_table = emit_table.append_column(
                    "size", pa.array([0] * n_emit, type=pa.int64()))

        if "token_count" not in emit_table.column_names:
            sizes = emit_table.column("size").to_numpy()
            emit_table = emit_table.append_column(
                "token_count", pa.array((sizes // 4).tolist(), type=pa.int64()))

        # Non-classified slices don't have classifier scores — fill with defaults
        if "quality" not in emit_table.column_names:
            emit_table = emit_table.append_column(
                "quality", pa.array([0.0] * n_emit, type=pa.float32()))
        if "structured_data" not in emit_table.column_names:
            emit_table = emit_table.append_column(
                "structured_data", pa.array([0.0] * n_emit, type=pa.float32()))
        if "content_type" not in emit_table.column_names:
            emit_table = emit_table.append_column(
                "content_type", pa.array(["unclassified"] * n_emit, type=pa.string()))

        emit_table = emit_table.append_column(
            "language_slice", pa.array(emit_slices, type=pa.string()),
        )
        emit_table = emit_table.append_column(
            "relevance_score", pa.array([0.0] * n_emit, type=pa.float32()),
        )

        # Keep only output columns (drop extras like hex_digest, max_stars, etc.)
        keep_cols = [c for c in OUTPUT_COLUMNS if c in emit_table.column_names]
        emit_table = emit_table.select(keep_cols)

        _write_chunk(emit_table, writer_state)
        del emit_table


def _write_chunk(table, state):
    """Write a table chunk to the current output shard, rolling over as needed."""
    rows_per_shard = state["rows_per_shard"]
    n_rows = table.num_rows
    row_start = 0

    while row_start < n_rows:
        if state["writer"] is None:
            out_path = state["output_dir"] / f"filtered_{state['shard_idx']:04d}.parquet"
            state["writer"] = pq.ParquetWriter(out_path, table.schema)
            state["current_rows"] = 0
            if state["schema"] is None:
                state["schema"] = table.schema

        space = rows_per_shard - state["current_rows"]
        chunk_end = min(row_start + space, n_rows)
        state["writer"].write_table(table.slice(row_start, chunk_end - row_start))
        state["current_rows"] += chunk_end - row_start
        state["total_written"] += chunk_end - row_start
        row_start = chunk_end

        if state["current_rows"] >= rows_per_shard:
            state["writer"].close()
            state["writer"] = None
            state["shard_idx"] += 1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(all_slice_data, slice_results):
    report = {
        "total_docs_selected": 0,
        "total_tokens_selected": 0,
        "slices": {},
    }

    for s in LANGUAGE_SLICES:
        sm = all_slice_data.get(s.name)
        selected_indices = slice_results.get(s.name, np.array([], dtype=np.int64))

        if sm is None or sm["n_docs"] == 0:
            report["slices"][s.name] = {
                "description": s.description,
                "strategy": s.strategy,
                "target_tokens": s.target_tokens,
                "available_docs": 0,
                "available_tokens": 0,
                "selected_docs": 0,
                "selected_tokens": 0,
                "status": "empty",
            }
            continue

        selected_globals = set(selected_indices.tolist())
        local_selected = np.isin(sm["global_indices"], list(selected_globals))
        selected_tokens = int(sm["token_counts"][local_selected].sum())
        selected_docs = int(local_selected.sum())
        report["total_tokens_selected"] += selected_tokens
        report["total_docs_selected"] += selected_docs

        pct_filled = selected_tokens / s.target_tokens * 100 if s.target_tokens > 0 else 0
        status = "ok" if pct_filled >= 90 else ("shortfall" if pct_filled >= 50 else "low")

        slice_report = {
            "description": s.description,
            "strategy": s.strategy,
            "languages": s.languages,
            "min_relevance": s.min_relevance,
            "target_tokens": s.target_tokens,
            "available_docs": int(sm["n_docs"]),
            "available_tokens": int(sm["token_counts"].sum()),
            "selected_docs": selected_docs,
            "selected_tokens": selected_tokens,
            "pct_of_target": round(pct_filled, 1),
            "status": status,
        }

        if s.strategy == "relevance_filter" and sm["n_docs"] > 0 and local_selected.sum() > 0:
            sel_sd = sm["structured_data"][local_selected]
            sel_q = sm["quality"][local_selected]
            slice_report["quality_mean"] = round(float(sel_q.mean()), 2)
            slice_report["sd_mean"] = round(float(sel_sd.mean()), 2)

        report["slices"][s.name] = slice_report

    return report


def print_report(report):
    print(f"\n{'='*85}")
    print("FILTERING REPORT (StarCoder — Per-Language Slice)")
    print(f"{'='*85}")
    print(f"Total selected: {report['total_docs_selected']:,} files, "
          f"{report['total_tokens_selected']:,} tokens")

    print(f"\n{'Slice':<25s} {'Strategy':<20s} {'Target':>10s} {'Selected':>12s} {'%':>6s} {'Status':>10s}")
    print(f"{'-'*85}")
    for sname, sdata in report["slices"].items():
        target_m = sdata["target_tokens"] / 1e6
        selected_m = sdata["selected_tokens"] / 1e6
        pct = sdata.get("pct_of_target", 0)
        print(f"{sname:<25s} {sdata['strategy']:<20s} {target_m:>9.0f}M {selected_m:>11.0f}M "
              f"{pct:>5.1f}% {sdata['status']:>10s}")

    total_target = sum(s.target_tokens for s in LANGUAGE_SLICES)
    total_selected = report["total_tokens_selected"]
    pct = total_selected / total_target * 100 if total_target > 0 else 0
    print(f"\n{'TOTAL':<25s} {'':20s} {total_target/1e6:>9.0f}M {total_selected/1e6:>11.0f}M "
          f"{pct:>5.1f}%")
    print(f"{'='*85}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Scored shards: {input_dir}")
    print(f"Raw data:      {raw_dir or '(not provided — non-classified slices will be empty)'}")
    print(f"Output:        {output_dir}")
    print(f"\nLanguage slices:")
    for s in LANGUAGE_SLICES:
        source = "scored shards" if s.strategy == "relevance_filter" else "raw parquets"
        print(f"  {s.name:<25s}: {s.target_tokens/1e6:>6.0f}M tokens | "
              f"{s.strategy} | [{source}] | {', '.join(s.languages)}")
    total_target = sum(s.target_tokens for s in LANGUAGE_SLICES)
    print(f"  {'TOTAL':<25s}: {total_target/1e6:>6.0f}M tokens")

    t0 = time.time()

    # --- Scan both sources ---
    print("\n--- Scanning scored shards (classified languages) ---")
    scored_slice_data, scored_shard_map = scan_scored_shards(
        input_dir, args.min_tokens, args.max_tokens)

    raw_slice_data, raw_file_map = {}, []
    if raw_dir:
        print("\n--- Scanning raw parquets (non-classified slices) ---")
        raw_slice_data, raw_file_map = scan_raw_parquets(
            raw_dir, args.min_tokens, args.max_tokens)

    # Merge slice data from both sources
    all_slice_data = {}
    all_slice_data.update(scored_slice_data)
    all_slice_data.update(raw_slice_data)

    # --- Filter per slice ---
    print("\n--- Filtering per slice ---")
    slice_results: dict[str, np.ndarray] = {}
    scored_selected: set[int] = set()
    raw_selected: set[int] = set()
    scored_to_slice: dict[int, str] = {}
    raw_to_slice: dict[int, str] = {}

    for s in LANGUAGE_SLICES:
        sm = all_slice_data.get(s.name)
        if sm is None or sm["n_docs"] == 0:
            print(f"\n  {s.name} ({s.strategy}): no data")
            slice_results[s.name] = np.array([], dtype=np.int64)
            continue

        print(f"\n  {s.name} ({s.strategy}):")
        print(f"    Available: {sm['n_docs']:,} files, {sm['token_counts'].sum():,} tokens")
        print(f"    Target:    {s.target_tokens:,} tokens")

        selected_globals = filter_slice(s, sm, rng)
        slice_results[s.name] = selected_globals

        is_scored = s.strategy == "relevance_filter"
        target_set = scored_selected if is_scored else raw_selected
        target_map = scored_to_slice if is_scored else raw_to_slice

        for gidx in selected_globals:
            target_set.add(int(gidx))
            target_map[int(gidx)] = s.name

        local_selected = np.isin(sm["global_indices"], selected_globals)
        selected_tokens = sm["token_counts"][local_selected].sum()
        print(f"    Selected:  {int(local_selected.sum()):,} files, {int(selected_tokens):,} tokens "
              f"({selected_tokens/s.target_tokens*100:.1f}%)")

    # --- Write output ---
    print(f"\n--- Writing filtered shards ---")
    writer_state = {
        "output_dir": output_dir,
        "writer": None,
        "schema": None,
        "shard_idx": 0,
        "current_rows": 0,
        "total_written": 0,
        "rows_per_shard": 1_000_000,
    }

    if scored_shard_map and scored_selected:
        emit_scored_filtered(scored_shard_map, scored_selected, scored_to_slice, writer_state)

    if raw_file_map and raw_selected:
        emit_raw_filtered(raw_file_map, raw_selected, raw_to_slice, writer_state)

    if writer_state["writer"] is not None:
        writer_state["writer"].close()

    total_written = writer_state["total_written"]
    n_shards = writer_state["shard_idx"] + (1 if writer_state["current_rows"] > 0 else 0)
    print(f"  Written {total_written:,} files across {n_shards} shards")

    # --- Report ---
    report = build_report(all_slice_data, slice_results)
    print_report(report)

    config = {
        "quality_floor": QUALITY_HARD_FLOOR,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "random_seed": RANDOM_SEED,
        "input_dir": str(input_dir),
        "raw_dir": str(raw_dir) if raw_dir else None,
        "slices": {
            s.name: {
                "languages": s.languages,
                "target_tokens": s.target_tokens,
                "strategy": s.strategy,
                "min_relevance": s.min_relevance,
            }
            for s in LANGUAGE_SLICES
        },
    }
    with open(output_dir / "filter_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(output_dir / "distribution_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\nFiltering complete in {elapsed:.0f}s")
    print(f"  Output: {output_dir}")
    print(f"  Code files: {total_written:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Filter by per-language-slice budgets (StarCoder)",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory with scored shards from pipeline_classify.py")
    parser.add_argument("--raw-dir",
                        help="Directory with raw downloaded parquets (for non-classified slices)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
