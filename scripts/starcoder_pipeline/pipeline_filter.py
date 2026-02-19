#!/usr/bin/env python3
"""
Stage 2: Filter scored code files using per-language-slice token budgets.

Each language slice has its own filtering strategy:
  - schema_languages (JSON/YAML/SQL/protobuf/GraphQL): light filter (quality floor only)
  - typescript (relevance ≥ 2): relevance classifier
  - python (relevance ≥ 2): relevance classifier
  - rust_go_java (relevance ≥ 2): relevance classifier
  - jupyter: passthrough with quality floor
  - github_issues: keyword filter for structured data topics

Total target: ~3.5B tokens across all slices.

Usage:
  python pipeline_filter.py --input-dir scored_shards/ --output-dir filtered_shards/
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from pipeline_config import (
    CONTENT_GROUP_DISPLAY,
    CONTENT_GROUP_MAP,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_TOTAL_TARGET_TOKENS,
    GITHUB_ISSUES_KEYWORDS,
    LANGUAGE_SLICES,
    LANGUAGE_SLICE_MAP,
    QUALITY_HARD_FLOOR,
    RANDOM_SEED,
    SD_BINS,
    SD_TARGET_PCT,
    LanguageSlice,
    assign_content_group_vec,
    compute_relevance_score_batch,
    resolve_language_to_slice,
    sd_bin_vec,
    text_matches_keywords,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1: Scan shards, build metadata arrays partitioned by language slice
# ---------------------------------------------------------------------------

def scan_shards(input_dir: Path, min_tokens: int, max_tokens: int):
    """Scan all shards and partition documents by language slice.

    Returns per-slice metadata dicts and the shard_map for pass 2.
    """
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {input_dir}")

    print(f"Found {len(shard_files)} shards")

    # Per-slice accumulators
    slice_data: dict[str, dict] = {}
    for s in LANGUAGE_SLICES:
        slice_data[s.name] = {
            "quality": [],
            "structured_data": [],
            "content_types": [],
            "token_counts": [],
            "langs": [],
            "texts": [],         # only for keyword_filter strategy
            "global_indices": [], # position in the flat concatenated array
        }

    # Also track docs that don't match any slice
    unmatched_count = 0
    unmatched_tokens = 0

    shard_map = []    # (shard_path, kept_indices, kept_langs)
    global_offset = 0
    total_raw = 0
    dropped_quality = 0
    dropped_short = 0
    dropped_long = 0

    need_content = any(s.strategy == "keyword_filter" for s in LANGUAGE_SLICES)

    for shard_path in tqdm(shard_files, desc="Scanning shards"):
        read_cols = ["quality", "structured_data", "content_type", "token_count", "lang"]
        if need_content:
            read_cols.append("content")

        table = pq.read_table(shard_path, columns=read_cols)
        n_rows = table.num_rows
        total_raw += n_rows

        quality = table.column("quality").to_numpy().astype(np.float32)
        sd = table.column("structured_data").to_numpy().astype(np.float32)
        ct = table.column("content_type").to_pylist()
        token_counts = table.column("token_count").to_numpy().astype(np.int64)
        langs = table.column("lang").to_pylist()
        texts = table.column("content").to_pylist() if need_content and "content" in table.column_names else [None] * n_rows

        del table

        # Basic filters (applied to ALL slices uniformly)
        q_mask = quality <= QUALITY_HARD_FLOOR
        short_mask = token_counts < min_tokens
        long_mask = token_counts > max_tokens

        dropped_quality += int(q_mask.sum())
        dropped_short += int(short_mask.sum())
        dropped_long += int(long_mask.sum())

        keep = ~q_mask & ~short_mask & ~long_mask
        kept_indices = np.where(keep)[0]
        if len(kept_indices) == 0:
            continue

        kept_langs = [langs[i] for i in kept_indices]
        shard_map.append((shard_path, kept_indices, kept_langs))

        for pos, idx in enumerate(kept_indices):
            lang = langs[idx] or "unknown"
            slice_name = resolve_language_to_slice(lang)

            if slice_name is None:
                unmatched_count += 1
                unmatched_tokens += int(token_counts[idx])
                continue

            sd_val = slice_data[slice_name]
            sd_val["quality"].append(float(quality[idx]))
            sd_val["structured_data"].append(float(sd[idx]))
            sd_val["content_types"].append(ct[idx])
            sd_val["token_counts"].append(int(token_counts[idx]))
            sd_val["langs"].append(lang)
            sd_val["global_indices"].append(global_offset + pos)
            if texts[idx] is not None:
                sd_val["texts"].append(texts[idx])
            else:
                sd_val["texts"].append("")

        global_offset += len(kept_indices)

    # Convert to numpy
    for sname, sd_val in slice_data.items():
        sd_val["quality"] = np.array(sd_val["quality"], dtype=np.float32)
        sd_val["structured_data"] = np.array(sd_val["structured_data"], dtype=np.float32)
        sd_val["token_counts"] = np.array(sd_val["token_counts"], dtype=np.int64)
        sd_val["global_indices"] = np.array(sd_val["global_indices"], dtype=np.int64)
        sd_val["n_docs"] = len(sd_val["quality"])

    n_kept = sum(sd_val["n_docs"] for sd_val in slice_data.values()) + unmatched_count

    print(f"\nScan complete:")
    print(f"  Total raw code files: {total_raw:,}")
    print(f"  Dropped (quality <= {QUALITY_HARD_FLOOR}): {dropped_quality:,}")
    print(f"  Dropped (tokens < {min_tokens}): {dropped_short:,}")
    print(f"  Dropped (tokens > {max_tokens:,}): {dropped_long:,}")
    print(f"  Unmatched (not in any slice): {unmatched_count:,} ({unmatched_tokens:,} tokens)")
    print()
    for s in LANGUAGE_SLICES:
        sd_val = slice_data[s.name]
        total_tok = sd_val["token_counts"].sum() if sd_val["n_docs"] > 0 else 0
        print(f"  {s.name:<25s}: {sd_val['n_docs']:>10,} files, "
              f"{total_tok:>15,} tokens available "
              f"(target: {s.target_tokens:,})")

    return slice_data, shard_map, global_offset


# ---------------------------------------------------------------------------
# Per-slice filtering
# ---------------------------------------------------------------------------

def filter_slice(slice_def: LanguageSlice, slice_meta: dict, rng) -> np.ndarray:
    """Apply the slice's filtering strategy and return selected global indices."""
    n = slice_meta["n_docs"]
    if n == 0:
        return np.array([], dtype=np.int64)

    quality = slice_meta["quality"]
    sd = slice_meta["structured_data"]
    token_counts = slice_meta["token_counts"]
    global_indices = slice_meta["global_indices"]
    target = slice_def.target_tokens

    if slice_def.strategy == "light_filter":
        return _filter_light(quality, sd, token_counts, global_indices, target,
                             slice_def.min_quality, rng)

    elif slice_def.strategy == "relevance_filter":
        return _filter_by_relevance(quality, sd, token_counts, global_indices,
                                    target, slice_def.min_relevance,
                                    slice_def.min_quality, rng)

    elif slice_def.strategy == "passthrough":
        return _filter_passthrough(quality, token_counts, global_indices, target,
                                   slice_def.min_quality, rng)

    elif slice_def.strategy == "keyword_filter":
        texts = slice_meta["texts"]
        return _filter_by_keywords(quality, sd, token_counts, global_indices,
                                   texts, target, slice_def.min_quality, rng)

    else:
        logger.warning(f"Unknown strategy '{slice_def.strategy}' for {slice_def.name}, using light_filter")
        return _filter_light(quality, sd, token_counts, global_indices, target,
                             slice_def.min_quality, rng)


def _filter_light(quality, sd, token_counts, global_indices, target_tokens,
                  min_quality, rng):
    """Light filter: quality floor only, then sample to budget."""
    mask = quality > min_quality
    candidates = np.where(mask)[0]
    return _sample_to_budget(candidates, sd, quality, token_counts,
                             global_indices, target_tokens, rng)


def _filter_by_relevance(quality, sd, token_counts, global_indices,
                         target_tokens, min_relevance, min_quality, rng):
    """Relevance filter: structured_data >= threshold, then sample to budget."""
    mask = (quality > min_quality) & (sd >= min_relevance)
    candidates = np.where(mask)[0]
    return _sample_to_budget(candidates, sd, quality, token_counts,
                             global_indices, target_tokens, rng)


def _filter_passthrough(quality, token_counts, global_indices, target_tokens,
                        min_quality, rng):
    """Passthrough: quality floor only, no relevance filter."""
    mask = quality > min_quality
    candidates = np.where(mask)[0]

    if candidates.size == 0:
        return np.array([], dtype=np.int64)

    available_tokens = token_counts[candidates].sum()
    if available_tokens <= target_tokens:
        return global_indices[candidates]

    # Random subsample to budget
    cumtok = np.cumsum(token_counts[candidates])
    n_take = int(np.searchsorted(cumtok, target_tokens, side="right")) + 1
    n_take = min(n_take, len(candidates))
    chosen = rng.choice(candidates, size=n_take, replace=False)
    return global_indices[chosen]


def _filter_by_keywords(quality, sd, token_counts, global_indices,
                        texts, target_tokens, min_quality, rng):
    """Keyword filter: match against structured data keywords, then sample."""
    mask = quality > min_quality
    keyword_mask = np.array([text_matches_keywords(t) for t in texts])
    combined = mask & keyword_mask
    candidates = np.where(combined)[0]
    return _sample_to_budget(candidates, sd, quality, token_counts,
                             global_indices, target_tokens, rng)


def _sample_to_budget(candidates, sd, quality, token_counts, global_indices,
                      target_tokens, rng):
    """From candidates, select up to target_tokens by relevance ranking."""
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


# ---------------------------------------------------------------------------
# Pass 2: Emit filtered shards
# ---------------------------------------------------------------------------

def emit_filtered(shard_map, selected_global_indices: set, global_to_slice: dict,
                  output_dir: Path):
    """Re-read shards and write only selected documents."""
    output_dir.mkdir(parents=True, exist_ok=True)

    out_shard_idx = 0
    total_written = 0
    rows_per_output_shard = 1_000_000
    writer = None
    current_rows = 0
    output_schema = None
    global_offset = 0

    for shard_path, kept_indices, kept_langs in tqdm(shard_map, desc="Writing filtered shards"):
        n_kept = len(kept_indices)
        emit_positions = []
        emit_slices = []

        for pos in range(n_kept):
            gidx = global_offset + pos
            if gidx in selected_global_indices:
                emit_positions.append(pos)
                emit_slices.append(global_to_slice.get(gidx, "unknown"))

        global_offset += n_kept

        if not emit_positions:
            continue

        original_rows = kept_indices[np.array(emit_positions)]
        table = pq.read_table(shard_path)
        emit_table = table.take(original_rows)
        del table

        emit_table = emit_table.append_column(
            "language_slice", pa.array(emit_slices, type=pa.string()),
        )

        # Compute and add relevance_score
        sd_col = emit_table.column("structured_data").to_numpy().astype(np.float32)
        q_col = emit_table.column("quality").to_numpy().astype(np.float32)
        rel_scores = compute_relevance_score_batch(sd_col, q_col)
        emit_table = emit_table.append_column(
            "relevance_score", pa.array(rel_scores.tolist(), type=pa.float32()),
        )

        if output_schema is None:
            output_schema = emit_table.schema

        n_emit = emit_table.num_rows
        row_start = 0
        while row_start < n_emit:
            if writer is None:
                out_path = output_dir / f"filtered_{out_shard_idx:04d}.parquet"
                writer = pq.ParquetWriter(out_path, output_schema)
                current_rows = 0

            space = rows_per_output_shard - current_rows
            chunk_end = min(row_start + space, n_emit)
            writer.write_table(emit_table.slice(row_start, chunk_end - row_start))
            current_rows += chunk_end - row_start
            total_written += chunk_end - row_start
            row_start = chunk_end

            if current_rows >= rows_per_output_shard:
                writer.close()
                writer = None
                out_shard_idx += 1

        del emit_table

    if writer is not None:
        writer.close()

    print(f"  Written {total_written:,} code files across {out_shard_idx + 1} shards")
    return total_written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_report(slice_data, slice_results, total_written):
    report = {
        "total_docs_selected": total_written,
        "total_tokens_selected": 0,
        "slices": {},
    }

    for s in LANGUAGE_SLICES:
        sm = slice_data[s.name]
        selected_indices = slice_results.get(s.name, np.array([], dtype=np.int64))

        if sm["n_docs"] == 0:
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

        # Find which local indices were selected
        selected_globals = set(selected_indices.tolist())
        local_selected = np.isin(sm["global_indices"], list(selected_globals))
        selected_tokens = int(sm["token_counts"][local_selected].sum())
        report["total_tokens_selected"] += selected_tokens

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
            "selected_docs": int(local_selected.sum()),
            "selected_tokens": selected_tokens,
            "pct_of_target": round(pct_filled, 1),
            "status": status,
        }

        if sm["n_docs"] > 0 and local_selected.sum() > 0:
            sel_sd = sm["structured_data"][local_selected]
            sel_q = sm["quality"][local_selected]
            slice_report["quality_mean"] = round(float(sel_q.mean()), 2)
            slice_report["sd_mean"] = round(float(sel_sd.mean()), 2)

        report["slices"][s.name] = slice_report

    return report


def print_report(report):
    print(f"\n{'='*80}")
    print("FILTERING REPORT (StarCoder — Per-Language Slice)")
    print(f"{'='*80}")
    print(f"Total selected: {report['total_docs_selected']:,} code files, "
          f"{report['total_tokens_selected']:,} tokens")

    print(f"\n{'Slice':<25s} {'Strategy':<20s} {'Target':>10s} {'Selected':>12s} {'%':>6s} {'Status':>10s}")
    print(f"{'-'*83}")
    for sname, sdata in report["slices"].items():
        target_m = sdata["target_tokens"] / 1e6
        selected_m = sdata["selected_tokens"] / 1e6
        pct = sdata.get("pct_of_target", 0)
        print(f"{sname:<25s} {sdata['strategy']:<20s} {target_m:>9.0f}M {selected_m:>11.0f}M "
              f"{pct:>5.1f}% {sdata['status']:>10s}")

    total_target = sum(s.target_tokens for s in LANGUAGE_SLICES)
    total_selected = report["total_tokens_selected"]
    print(f"\n{'TOTAL':<25s} {'':20s} {total_target/1e6:>9.0f}M {total_selected/1e6:>11.0f}M "
          f"{total_selected/total_target*100:>5.1f}%")
    print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nLanguage slices:")
    for s in LANGUAGE_SLICES:
        print(f"  {s.name:<25s}: {s.target_tokens/1e6:>6.0f}M tokens | "
              f"{s.strategy} | {', '.join(s.languages)}")
    total_target = sum(s.target_tokens for s in LANGUAGE_SLICES)
    print(f"  {'TOTAL':<25s}: {total_target/1e6:>6.0f}M tokens")

    t0 = time.time()

    print("\n--- Pass 1: Scanning shards ---")
    slice_data, shard_map, total_kept = scan_shards(
        input_dir, args.min_tokens, args.max_tokens,
    )

    print("\n--- Filtering per slice ---")
    slice_results: dict[str, np.ndarray] = {}
    all_selected_globals: set[int] = set()
    global_to_slice: dict[int, str] = {}

    for s in LANGUAGE_SLICES:
        sm = slice_data[s.name]
        print(f"\n  {s.name} ({s.strategy}):")
        print(f"    Available: {sm['n_docs']:,} files, {sm['token_counts'].sum():,} tokens")
        print(f"    Target:    {s.target_tokens:,} tokens")

        selected_globals = filter_slice(s, sm, rng)
        slice_results[s.name] = selected_globals

        for gidx in selected_globals:
            all_selected_globals.add(int(gidx))
            global_to_slice[int(gidx)] = s.name

        # Find selected tokens for this slice
        local_selected = np.isin(sm["global_indices"], selected_globals)
        selected_tokens = sm["token_counts"][local_selected].sum()
        print(f"    Selected:  {int(local_selected.sum()):,} files, {int(selected_tokens):,} tokens "
              f"({selected_tokens/s.target_tokens*100:.1f}%)")

    print(f"\n--- Pass 2: Writing filtered shards ---")
    total_written = emit_filtered(shard_map, all_selected_globals, global_to_slice, output_dir)

    report = build_report(slice_data, slice_results, total_written)
    print_report(report)

    config = {
        "quality_floor": QUALITY_HARD_FLOOR,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "random_seed": RANDOM_SEED,
        "input_dir": str(input_dir),
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
        description="Stage 2: Filter scored code files by per-language-slice budgets (StarCoder)",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
