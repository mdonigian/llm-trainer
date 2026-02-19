#!/usr/bin/env python3
"""
Stage 2: Filter scored code files by quality, structured data relevance,
and content type distribution targets.

Filtering strategy (from project decisions):
  1. Hard floor: drop quality <= 1.5 (broken/gibberish)
  2. Content type grouping: library/application/script/test/low_value
  3. Within each group, prioritize by structured data relevance
  4. Structured data distribution targets (boost SD2/SD3 heavily)
  5. Quality >= 4 gets a soft relevance boost

Usage:
  python pipeline_filter.py --input-dir scored_shards/ --output-dir filtered_shards/
  python pipeline_filter.py --input-dir scored_shards/ --output-dir filtered_shards/ \
      --target-tokens 11000000000
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
    CONTENT_GROUP_PRIORITY,
    CONTENT_GROUP_TARGET_PCT,
    CONTENT_TYPES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_TOTAL_TARGET_TOKENS,
    NUM_CONTENT_TYPES,
    QUALITY_HARD_FLOOR,
    RANDOM_SEED,
    SD_BINS,
    SD_TARGET_PCT,
    STRUCTURED_DATA_NAMES,
    assign_content_group_vec,
    compute_relevance_score_batch,
    sd_bin_vec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1: Scan shards, build metadata arrays
# ---------------------------------------------------------------------------

def scan_shards(input_dir: Path, quality_floor: float, min_tokens: int, max_tokens: int):
    shard_files = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No shard files found in {input_dir}")

    print(f"Found {len(shard_files)} shards")

    all_quality = []
    all_sd = []
    all_ct = []
    all_token_counts = []
    shard_map = []
    total_raw = 0
    dropped_quality = 0
    dropped_short = 0
    dropped_long = 0

    for shard_path in tqdm(shard_files, desc="Scanning shards"):
        table = pq.read_table(shard_path, columns=["quality", "structured_data",
                                                      "content_type", "token_count"])
        n_rows = table.num_rows
        total_raw += n_rows

        quality = table.column("quality").to_numpy().astype(np.float32)
        sd = table.column("structured_data").to_numpy().astype(np.float32)
        ct = table.column("content_type").to_pylist()
        token_counts = table.column("token_count").to_numpy().astype(np.int64)
        del table

        q_mask = quality <= quality_floor
        short_mask = token_counts < min_tokens
        long_mask = token_counts > max_tokens

        dropped_quality += int(q_mask.sum())
        dropped_short += int(short_mask.sum())
        dropped_long += int(long_mask.sum())

        keep = ~q_mask & ~short_mask & ~long_mask
        kept_indices = np.where(keep)[0]
        if len(kept_indices) == 0:
            continue

        all_quality.append(quality[kept_indices])
        all_sd.append(sd[kept_indices])
        all_ct.extend([ct[i] for i in kept_indices])
        all_token_counts.append(token_counts[kept_indices])
        shard_map.append((shard_path, kept_indices))

    quality = np.concatenate(all_quality)
    sd = np.concatenate(all_sd)
    token_counts = np.concatenate(all_token_counts)
    n_kept = len(quality)

    print(f"\nQuality filtering:")
    print(f"  Total raw code files: {total_raw:,}")
    print(f"  Dropped (quality <= {quality_floor}): {dropped_quality:,}")
    print(f"  Dropped (tokens < {min_tokens}): {dropped_short:,}")
    print(f"  Dropped (tokens > {max_tokens:,}): {dropped_long:,}")
    print(f"  Kept: {n_kept:,} ({n_kept/total_raw*100:.1f}%)")

    return {
        "quality": quality,
        "structured_data": sd,
        "content_types": all_ct,
        "token_counts": token_counts,
        "n_docs": n_kept,
    }, shard_map


# ---------------------------------------------------------------------------
# Group assignment and sampling
# ---------------------------------------------------------------------------

def assign_all_groups(content_types):
    """For each document, determine its content group."""
    groups = assign_content_group_vec(content_types)
    group_members = defaultdict(list)
    for i, g in enumerate(groups):
        group_members[g].append(i)
    return {k: np.array(v, dtype=np.int64) for k, v in group_members.items()}, groups


def priority_sample(meta, group_members, target_tokens, rng):
    n = meta["n_docs"]
    quality = meta["quality"]
    sd = meta["structured_data"]
    token_counts = meta["token_counts"]

    relevance = compute_relevance_score_batch(sd, quality)
    sd_bins = sd_bin_vec(sd)

    selected = np.zeros(n, dtype=bool)
    group_report = {}

    sorted_groups = sorted(CONTENT_GROUP_MAP.keys(),
                           key=lambda g: CONTENT_GROUP_PRIORITY[g])

    for group_name in sorted_groups:
        target_pct = CONTENT_GROUP_TARGET_PCT[group_name]
        members = group_members.get(group_name, np.array([], dtype=np.int64))

        if len(members) == 0:
            group_report[group_name] = {
                "target_tokens": int(target_pct * target_tokens),
                "available_tokens": 0,
                "selected_tokens": 0,
                "status": "empty",
            }
            continue

        if group_name == "low_value":
            already_selected_tokens = token_counts[selected].sum()
            group_target = max(0, target_tokens - already_selected_tokens)
        else:
            group_target = target_pct * target_tokens

        already_in = selected[members]
        already_tokens = token_counts[members[already_in]].sum()
        remaining_needed = max(0, group_target - already_tokens)

        unselected_mask = ~selected[members]
        candidates = members[unselected_mask]

        if remaining_needed <= 0 or len(candidates) == 0:
            group_report[group_name] = {
                "target_tokens": int(group_target),
                "available_tokens": int(token_counts[members].sum()),
                "selected_tokens": int(already_tokens),
                "newly_selected": 0,
                "status": "filled_by_overlap" if remaining_needed <= 0 else "no_candidates",
            }
            continue

        newly_selected = _sample_with_sd_targets(
            candidates, relevance, sd_bins, token_counts, remaining_needed, rng,
        )
        selected[newly_selected] = True

        final_tokens = token_counts[members[selected[members]]].sum()
        group_report[group_name] = {
            "target_tokens": int(group_target),
            "available_tokens": int(token_counts[members].sum()),
            "selected_tokens": int(final_tokens),
            "newly_selected": len(newly_selected),
            "status": "ok" if final_tokens >= group_target * 0.9 else "shortfall",
        }

    return selected, relevance, group_report


def _sample_with_sd_targets(candidates, relevance, sd_bins, token_counts,
                            target_tokens, rng):
    selected_indices = []
    for bin_label, _, _ in SD_BINS:
        bin_target = SD_TARGET_PCT[bin_label] * target_tokens
        bin_candidates = candidates[sd_bins[candidates] == bin_label]
        if len(bin_candidates) == 0:
            continue
        order = np.argsort(-relevance[bin_candidates])
        sorted_cands = bin_candidates[order]
        cumulative_tokens = np.cumsum(token_counts[sorted_cands])
        n_take = np.searchsorted(cumulative_tokens, bin_target, side="right") + 1
        n_take = min(n_take, len(sorted_cands))
        selected_indices.extend(sorted_cands[:n_take].tolist())

    return np.array(selected_indices, dtype=np.int64) if selected_indices else np.array([], dtype=np.int64)


# ---------------------------------------------------------------------------
# Pass 2: Emit filtered shards
# ---------------------------------------------------------------------------

def emit_filtered(shard_map, selected, relevance, content_groups, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    offset = 0
    out_shard_idx = 0
    total_written = 0
    rows_per_output_shard = 1_000_000
    writer = None
    current_rows = 0
    output_schema = None

    for shard_path, kept_indices in tqdm(shard_map, desc="Writing filtered shards"):
        n_kept = len(kept_indices)
        shard_selected = selected[offset : offset + n_kept]
        shard_relevance = relevance[offset : offset + n_kept]
        shard_groups = content_groups[offset : offset + n_kept]
        offset += n_kept

        emit_positions = np.where(shard_selected)[0]
        if len(emit_positions) == 0:
            continue

        original_rows = kept_indices[emit_positions]
        table = pq.read_table(shard_path)
        emit_table = table.take(original_rows)
        del table

        emit_groups = [shard_groups[i] for i in emit_positions]
        emit_table = emit_table.append_column(
            "content_group", pa.array(emit_groups, type=pa.string()),
        )
        emit_table = emit_table.append_column(
            "relevance_score",
            pa.array(shard_relevance[emit_positions].tolist(), type=pa.float32()),
        )

        if output_schema is None:
            output_schema = emit_table.schema

        rows_to_write = emit_table.num_rows
        row_start = 0
        while row_start < rows_to_write:
            if writer is None:
                out_path = output_dir / f"filtered_{out_shard_idx:04d}.parquet"
                writer = pq.ParquetWriter(out_path, output_schema)
                current_rows = 0

            space = rows_per_output_shard - current_rows
            chunk_end = min(row_start + space, rows_to_write)
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

def build_report(meta, selected, relevance, group_members, content_groups,
                 group_report, target_tokens):
    quality = meta["quality"]
    sd = meta["structured_data"]
    token_counts = meta["token_counts"]
    sel_mask = selected

    report = {
        "total_docs_selected": int(sel_mask.sum()),
        "total_tokens_selected": int(token_counts[sel_mask].sum()),
        "target_tokens": int(target_tokens),
    }

    report["group_distribution"] = {}
    for gname in CONTENT_GROUP_MAP:
        members = group_members.get(gname, np.array([], dtype=np.int64))
        if len(members) == 0:
            report["group_distribution"][gname] = {
                "display_name": CONTENT_GROUP_DISPLAY[gname],
                "target_pct": CONTENT_GROUP_TARGET_PCT[gname],
                "actual_pct": 0, "tokens": 0, "docs": 0,
            }
            continue
        sel_in = sel_mask[members]
        tokens_in = token_counts[members[sel_in]].sum()
        total_sel_tokens = token_counts[sel_mask].sum()
        report["group_distribution"][gname] = {
            "display_name": CONTENT_GROUP_DISPLAY[gname],
            "target_pct": CONTENT_GROUP_TARGET_PCT[gname],
            "actual_pct": float(tokens_in / total_sel_tokens) if total_sel_tokens > 0 else 0,
            "tokens": int(tokens_in),
            "docs": int(sel_in.sum()),
            **group_report.get(gname, {}),
        }

    sd_bins = sd_bin_vec(sd[sel_mask])
    sel_tokens = token_counts[sel_mask]
    report["sd_distribution"] = {}
    for bin_label, lo, hi in SD_BINS:
        mask = sd_bins == bin_label
        report["sd_distribution"][bin_label] = {
            "range": f"[{lo}, {hi})",
            "target_pct": SD_TARGET_PCT[bin_label],
            "actual_pct": float(mask.mean()) if len(sd_bins) > 0 else 0,
            "docs": int(mask.sum()),
            "tokens": int(sel_tokens[mask].sum()),
        }

    sel_quality = quality[sel_mask]
    report["quality_stats"] = {
        "mean": float(sel_quality.mean()),
        "median": float(np.median(sel_quality)),
    }

    return report


def print_report(report):
    print(f"\n{'='*70}")
    print("FILTERING REPORT (StarCoder)")
    print(f"{'='*70}")
    print(f"Selected: {report['total_docs_selected']:,} code files, "
          f"{report['total_tokens_selected']:,} tokens "
          f"(target: {report['target_tokens']:,})")

    print(f"\nContent group distribution:")
    print(f"  {'Group':<30s} {'Target%':>8s} {'Actual%':>8s} {'Tokens':>15s} {'Files':>10s}")
    print(f"  {'-'*71}")
    for gname, gdata in report["group_distribution"].items():
        print(f"  {CONTENT_GROUP_DISPLAY.get(gname, gname):<30s} "
              f"{gdata['target_pct']*100:>7.1f}% "
              f"{gdata['actual_pct']*100:>7.1f}% "
              f"{gdata['tokens']:>15,} "
              f"{gdata['docs']:>10,}")

    print(f"\nStructured data distribution:")
    for bl, bdata in report["sd_distribution"].items():
        print(f"  {bl} {bdata['range']}: target={bdata['target_pct']*100:.0f}% "
              f"actual={bdata['actual_pct']*100:.1f}% "
              f"({bdata['docs']:,} files)")

    print(f"\nQuality: mean={report['quality_stats']['mean']:.2f}, "
          f"median={report['quality_stats']['median']:.2f}")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    target_tokens = args.target_tokens

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target tokens: {target_tokens:,}")

    t0 = time.time()

    print("\n--- Pass 1: Scanning shards ---")
    meta, shard_map = scan_shards(
        input_dir, QUALITY_HARD_FLOOR, args.min_tokens, args.max_tokens,
    )

    print("\nAssigning content groups...")
    group_members, content_groups = assign_all_groups(meta["content_types"])
    content_groups = np.array(content_groups)
    for gname, members in group_members.items():
        print(f"  {CONTENT_GROUP_DISPLAY.get(gname, gname):<30s}: {len(members):>10,} files, "
              f"{meta['token_counts'][members].sum():>15,} tokens")

    print("\n--- Sampling ---")
    selected, relevance, group_report = priority_sample(meta, group_members, target_tokens, rng)

    report = build_report(meta, selected, relevance, group_members, content_groups,
                          group_report, target_tokens)
    print_report(report)

    print("\n--- Pass 2: Writing filtered shards ---")
    total_written = emit_filtered(shard_map, selected, relevance, content_groups, output_dir)

    config = {
        "quality_floor": QUALITY_HARD_FLOOR,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "target_tokens": target_tokens,
        "random_seed": RANDOM_SEED,
        "input_dir": str(input_dir),
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
        description="Stage 2: Filter scored code files (StarCoder)",
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TOTAL_TARGET_TOKENS)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
