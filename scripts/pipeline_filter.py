#!/usr/bin/env python3
"""
Stage 2: Filter scored documents by topic/complexity distribution targets.

Implements multi-label-aware, priority-based sampling:
  1. Quality floor (ambiguity, token length)
  2. Compute group assignments and relevance scores
  3. Priority-based per-group sampling (STEM-Core first, General last)
  4. Complexity distribution targets within each group
  5. Multi-label documents count toward multiple group quotas

Usage:
  python pipeline_filter.py --input-dir scored_shards/ --output-dir filtered_shards/
  python pipeline_filter.py --input-dir scored_shards/ --output-dir filtered_shards/ \
      --topic-threshold 0.3 --ambiguity-floor 0.3 --target-tokens 23500000000
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
    COMPLEXITY_BINS,
    COMPLEXITY_TARGET_PCT,
    DEFAULT_AMBIGUITY_FLOOR,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_TOKENS,
    DEFAULT_TOPIC_THRESHOLD,
    DEFAULT_TOTAL_TARGET_TOKENS,
    GROUP_DISPLAY_NAMES,
    GROUP_MAP,
    GROUP_PRIORITY,
    GROUP_TARGET_PCT,
    LABEL_DISPLAY_NAMES,
    NUM_LABELS,
    RANDOM_SEED,
    complexity_bin_vec,
    compute_relevance_score_batch,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pass 1: Scan shards, build metadata arrays
# ---------------------------------------------------------------------------

def scan_shards(input_dir: Path, topic_threshold: float, ambiguity_floor: float,
                min_tokens: int, max_tokens: int):
    """Read all shards and build in-memory metadata for filtering decisions.

    Returns:
        meta: dict with arrays for all quality-passing documents
        shard_map: list of (shard_path, kept_row_indices) for pass 2
    """
    shard_files_raw = sorted(input_dir.glob("shard_*.parquet"))
    if not shard_files_raw:
        raise FileNotFoundError(f"No shard files found in {input_dir}")

    print(f"Found {len(shard_files_raw)} shards")

    all_topic_scores = []
    all_complexity = []
    all_token_counts = []
    shard_map = []
    total_raw = 0
    dropped_ambiguity = 0
    dropped_short = 0
    dropped_long = 0

    for shard_path in tqdm(shard_files_raw, desc="Scanning shards"):
        table = pq.read_table(shard_path, columns=["topic_scores", "complexity", "token_count"])
        n_rows = table.num_rows
        total_raw += n_rows

        # Efficient deserialization: flatten the list column into a 1-D array
        # and reshape, avoiding the creation of N*17 Python float objects that
        # to_pylist() would produce (~17M objects per 1M-doc shard).
        ts_col = table.column("topic_scores")
        flat_values = ts_col.combine_chunks().values.to_numpy(zero_copy_only=False)
        topic_scores_shard = flat_values.astype(np.float32).reshape(n_rows, NUM_LABELS)

        complexity_shard = table.column("complexity").to_numpy().astype(np.float32)
        token_counts_shard = table.column("token_count").to_numpy().astype(np.int64)
        del table

        max_sigmoid = topic_scores_shard.max(axis=1)
        amb_mask = max_sigmoid < ambiguity_floor
        short_mask = token_counts_shard < min_tokens
        long_mask = token_counts_shard > max_tokens

        dropped_ambiguity += int(amb_mask.sum())
        dropped_short += int(short_mask.sum())
        dropped_long += int(long_mask.sum())

        keep = ~amb_mask & ~short_mask & ~long_mask
        kept_indices = np.where(keep)[0]
        if len(kept_indices) == 0:
            continue

        all_topic_scores.append(topic_scores_shard[kept_indices])
        all_complexity.append(complexity_shard[kept_indices])
        all_token_counts.append(token_counts_shard[kept_indices])
        shard_map.append((shard_path, kept_indices))

    topic_scores = np.concatenate(all_topic_scores, axis=0)
    complexity = np.concatenate(all_complexity, axis=0)
    token_counts = np.concatenate(all_token_counts, axis=0)
    n_kept = len(topic_scores)

    print(f"\nQuality filtering:")
    print(f"  Total raw documents: {total_raw:,}")
    print(f"  Dropped (ambiguity < {ambiguity_floor}): {dropped_ambiguity:,}")
    print(f"  Dropped (tokens < {min_tokens}): {dropped_short:,}")
    print(f"  Dropped (tokens > {max_tokens:,}): {dropped_long:,}")
    print(f"  Kept: {n_kept:,} ({n_kept/total_raw*100:.1f}%)")
    print(f"  Total tokens available: {token_counts.sum():,}")

    return {
        "topic_scores": topic_scores,
        "complexity": complexity,
        "token_counts": token_counts,
        "n_docs": n_kept,
    }, shard_map


# ---------------------------------------------------------------------------
# Group assignment and sampling
# ---------------------------------------------------------------------------

def assign_all_groups(topic_scores, threshold):
    """For each document, determine which groups it belongs to.

    Returns dict: group_name -> np.array of document indices
    """
    n = len(topic_scores)
    group_members = {}

    for group_name, label_indices in GROUP_MAP.items():
        mask = np.zeros(n, dtype=bool)
        for li in label_indices:
            mask |= (topic_scores[:, li] >= threshold)
        group_members[group_name] = np.where(mask)[0]

    return group_members


def priority_sample(meta, group_members, target_tokens, rng):
    """Priority-based per-group sampling with complexity distribution targets.

    Returns a boolean array indicating which documents are selected.
    """
    n = meta["n_docs"]
    topic_scores = meta["topic_scores"]
    complexity = meta["complexity"]
    token_counts = meta["token_counts"]

    relevance = compute_relevance_score_batch(topic_scores, complexity)
    cbins = complexity_bin_vec(complexity)

    selected = np.zeros(n, dtype=bool)
    group_report = {}

    # Sort groups by priority (lower number = higher priority, "general" is last)
    sorted_groups = sorted(GROUP_MAP.keys(), key=lambda g: GROUP_PRIORITY[g])

    for group_name in sorted_groups:
        target_pct = GROUP_TARGET_PCT[group_name]
        members = group_members[group_name]

        if len(members) == 0:
            group_report[group_name] = {
                "target_tokens": int(target_pct * target_tokens),
                "available_tokens": 0,
                "selected_tokens": 0,
                "status": "empty",
            }
            continue

        if group_name == "general":
            # General fills whatever total budget remains
            already_selected_tokens = token_counts[selected].sum()
            group_target = max(0, target_tokens - already_selected_tokens)
        else:
            group_target = target_pct * target_tokens

        # Tokens from already-selected docs that are also in this group
        already_in = selected[members]
        already_tokens = token_counts[members[already_in]].sum()
        remaining_needed = max(0, group_target - already_tokens)

        # Unselected candidates in this group
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

        # Apply complexity distribution within this group's candidates
        newly_selected = _sample_with_complexity_targets(
            candidates, relevance, cbins, token_counts,
            remaining_needed, rng,
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


def _sample_with_complexity_targets(candidates, relevance, cbins, token_counts,
                                    target_tokens, rng):
    """Select documents from candidates respecting complexity distribution targets.

    Within each complexity bin, sort by relevance (descending) and fill until
    the bin's token quota is met.
    """
    selected_indices = []

    for bin_label, _, _ in COMPLEXITY_BINS:
        bin_target = COMPLEXITY_TARGET_PCT[bin_label] * target_tokens
        bin_candidates = candidates[cbins[candidates] == bin_label]

        if len(bin_candidates) == 0:
            continue

        # Sort by relevance, descending
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

def emit_filtered(shard_map, selected, relevance, topic_scores, threshold,
                  output_dir: Path):
    """Re-read source shards and write only selected documents to output."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build per-shard selection masks
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
        shard_topic_scores = topic_scores[offset : offset + n_kept]
        offset += n_kept

        emit_positions = np.where(shard_selected)[0]
        if len(emit_positions) == 0:
            continue

        original_rows = kept_indices[emit_positions]
        table = pq.read_table(shard_path)

        # Select rows and add new columns
        emit_table = table.take(original_rows)
        del table

        # Compute assigned_groups for emitted docs
        emit_topic = shard_topic_scores[emit_positions]
        groups_list = []
        for i in range(len(emit_positions)):
            doc_groups = []
            for gname, label_indices in GROUP_MAP.items():
                if any(emit_topic[i, li] >= threshold for li in label_indices):
                    doc_groups.append(gname)
            groups_list.append(doc_groups)

        emit_table = emit_table.append_column(
            "assigned_groups", pa.array(groups_list, type=pa.list_(pa.string()))
        )
        emit_table = emit_table.append_column(
            "relevance_score",
            pa.array(shard_relevance[emit_positions].tolist(), type=pa.float32()),
        )

        if output_schema is None:
            output_schema = emit_table.schema

        # Write in chunks respecting output shard size
        rows_to_write = emit_table.num_rows
        row_start = 0

        while row_start < rows_to_write:
            if writer is None:
                out_path = output_dir / f"filtered_{out_shard_idx:04d}.parquet"
                writer = pq.ParquetWriter(out_path, output_schema)
                current_rows = 0

            space = rows_per_output_shard - current_rows
            chunk_end = min(row_start + space, rows_to_write)
            chunk = emit_table.slice(row_start, chunk_end - row_start)
            writer.write_table(chunk)
            current_rows += chunk.num_rows
            total_written += chunk.num_rows
            row_start = chunk_end

            if current_rows >= rows_per_output_shard:
                writer.close()
                writer = None
                out_shard_idx += 1

        del emit_table

    if writer is not None:
        writer.close()

    print(f"  Written {total_written:,} documents across {out_shard_idx + 1} shards")
    return total_written


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_distribution_report(meta, selected, relevance, group_members, group_report,
                              threshold, target_tokens):
    """Build detailed distribution report for logging."""
    topic_scores = meta["topic_scores"]
    complexity = meta["complexity"]
    token_counts = meta["token_counts"]

    sel_mask = selected
    sel_tokens = token_counts[sel_mask]
    sel_complexity = complexity[sel_mask]
    sel_topic = topic_scores[sel_mask]

    report = {
        "total_docs_selected": int(sel_mask.sum()),
        "total_tokens_selected": int(sel_tokens.sum()),
        "target_tokens": int(target_tokens),
        "topic_threshold": threshold,
    }

    # Per-group distribution
    report["group_distribution"] = {}
    for gname in GROUP_MAP:
        members = group_members[gname]
        sel_in_group = sel_mask[members]
        tokens_in_group = token_counts[members[sel_in_group]].sum()
        report["group_distribution"][gname] = {
            "display_name": GROUP_DISPLAY_NAMES[gname],
            "target_pct": GROUP_TARGET_PCT[gname],
            "actual_pct": float(tokens_in_group / sel_tokens.sum()) if sel_tokens.sum() > 0 else 0,
            "tokens": int(tokens_in_group),
            "docs": int(sel_in_group.sum()),
            **group_report.get(gname, {}),
        }

    # Per-label distribution (all 17 labels)
    report["label_distribution"] = {}
    for i, lname in enumerate(LABEL_DISPLAY_NAMES):
        mask = sel_topic[:, i] >= threshold
        report["label_distribution"][lname] = {
            "docs": int(mask.sum()),
            "pct": float(mask.mean() * 100) if len(sel_topic) > 0 else 0,
        }

    # Complexity distribution
    cbins = complexity_bin_vec(sel_complexity)
    report["complexity_distribution"] = {}
    for bin_label, lo, hi in COMPLEXITY_BINS:
        mask = cbins == bin_label
        report["complexity_distribution"][bin_label] = {
            "range": f"[{lo}, {hi})",
            "target_pct": COMPLEXITY_TARGET_PCT[bin_label],
            "actual_pct": float(mask.mean()) if len(cbins) > 0 else 0,
            "docs": int(mask.sum()),
            "tokens": int(sel_tokens[mask].sum()),
        }

    # Multi-label overlap stats
    multi_counts = (sel_topic >= threshold).sum(axis=1)
    report["multi_label_stats"] = {
        "mean_labels": float(multi_counts.mean()) if len(multi_counts) > 0 else 0,
        "pct_1_label": float((multi_counts == 1).mean() * 100) if len(multi_counts) > 0 else 0,
        "pct_2plus": float((multi_counts >= 2).mean() * 100) if len(multi_counts) > 0 else 0,
        "pct_3plus": float((multi_counts >= 3).mean() * 100) if len(multi_counts) > 0 else 0,
    }

    return report


def print_report(report):
    """Print a summary of the filtering results."""
    print(f"\n{'='*70}")
    print("FILTERING REPORT")
    print(f"{'='*70}")
    print(f"Selected: {report['total_docs_selected']:,} docs, "
          f"{report['total_tokens_selected']:,} tokens "
          f"(target: {report['target_tokens']:,})")

    print(f"\nGroup distribution:")
    print(f"  {'Group':<25s} {'Target%':>8s} {'Actual%':>8s} {'Tokens':>15s} {'Docs':>10s} {'Status':>12s}")
    print(f"  {'-'*78}")
    for gname, gdata in report["group_distribution"].items():
        print(f"  {GROUP_DISPLAY_NAMES[gname]:<25s} "
              f"{gdata['target_pct']*100:>7.1f}% "
              f"{gdata['actual_pct']*100:>7.1f}% "
              f"{gdata['tokens']:>15,} "
              f"{gdata['docs']:>10,} "
              f"{gdata.get('status', ''):>12s}")

    print(f"\nComplexity distribution:")
    for bin_label, bdata in report["complexity_distribution"].items():
        print(f"  {bin_label} {bdata['range']}: "
              f"target={bdata['target_pct']*100:.0f}% "
              f"actual={bdata['actual_pct']*100:.1f}% "
              f"({bdata['docs']:,} docs, {bdata['tokens']:,} tokens)")

    ml = report["multi_label_stats"]
    print(f"\nMulti-label: mean={ml['mean_labels']:.1f} labels/doc, "
          f"2+: {ml['pct_2plus']:.1f}%, 3+: {ml['pct_3plus']:.1f}%")
    print(f"{'='*70}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    topic_threshold = args.topic_threshold
    ambiguity_floor = args.ambiguity_floor
    target_tokens = args.target_tokens

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target tokens: {target_tokens:,}")
    print(f"Topic threshold: {topic_threshold}")
    print(f"Ambiguity floor: {ambiguity_floor}")

    t0 = time.time()

    # Pass 1: scan and build index
    print("\n--- Pass 1: Scanning shards ---")
    meta, shard_map = scan_shards(
        input_dir, topic_threshold, ambiguity_floor,
        args.min_tokens, args.max_tokens,
    )

    # Assign groups
    print("\nAssigning groups...")
    group_members = assign_all_groups(meta["topic_scores"], topic_threshold)
    for gname, members in group_members.items():
        print(f"  {GROUP_DISPLAY_NAMES[gname]:<25s}: {len(members):>10,} docs, "
              f"{meta['token_counts'][members].sum():>15,} tokens")

    # Priority sampling
    print("\n--- Sampling ---")
    selected, relevance, group_report = priority_sample(
        meta, group_members, target_tokens, rng,
    )

    # Report
    report = build_distribution_report(
        meta, selected, relevance, group_members, group_report,
        topic_threshold, target_tokens,
    )
    print_report(report)

    # Pass 2: emit filtered shards
    print("\n--- Pass 2: Writing filtered shards ---")
    total_written = emit_filtered(
        shard_map, selected, relevance, meta["topic_scores"],
        topic_threshold, output_dir,
    )

    # Save config and report
    config = {
        "topic_threshold": topic_threshold,
        "ambiguity_floor": ambiguity_floor,
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
    print(f"  Documents: {total_written:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: Filter scored documents by topic/complexity targets",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory of scored parquet shards from Stage 1")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for filtered shards")
    parser.add_argument("--topic-threshold", type=float, default=DEFAULT_TOPIC_THRESHOLD,
                        help=f"Sigmoid threshold for topic assignment (default: {DEFAULT_TOPIC_THRESHOLD})")
    parser.add_argument("--ambiguity-floor", type=float, default=DEFAULT_AMBIGUITY_FLOOR,
                        help=f"Minimum max-sigmoid to keep doc (default: {DEFAULT_AMBIGUITY_FLOOR})")
    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TOTAL_TARGET_TOKENS,
                        help=f"Total target tokens (default: {DEFAULT_TOTAL_TARGET_TOKENS:,})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
