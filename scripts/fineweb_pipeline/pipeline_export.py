#!/usr/bin/env python3
"""
Stage 4: Export — merge shards, compute stats, generate dataset card, upload.

Merges deduped parquet shards into consistently-sized output files (~500MB each),
computes comprehensive statistics, generates a HuggingFace dataset card, and
optionally uploads to the Hub.

Usage:
  python pipeline_export.py --input-dir deduped_shards/ --output-dir final_output/
  python pipeline_export.py --input-dir deduped_shards/ --output-dir final_output/ \
      --upload --repo-id myuser/fineweb-edu-curated
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from tqdm.auto import tqdm

from pipeline_config import (
    COMPLEXITY_BINS,
    COMPLEXITY_TARGET_PCT,
    DEFAULT_TOPIC_THRESHOLD,
    GROUP_DISPLAY_NAMES,
    GROUP_MAP,
    GROUP_TARGET_PCT,
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    NUM_LABELS,
    complexity_bin_vec,
)

logger = logging.getLogger(__name__)

TARGET_SHARD_BYTES = 500 * 1024 * 1024  # 500MB


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------

def merge_shards(input_dir: Path, output_dir: Path):
    """Merge deduped shards into consistently-sized parquet files."""
    shard_files = sorted(input_dir.glob("deduped_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No deduped shard files in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(shard_files)} shards into ~500MB output files...")

    # Drop dedup_cluster_id from final output (internal metadata)
    out_idx = 0
    writer = None
    current_bytes = 0
    output_schema = None

    for shard_path in tqdm(shard_files, desc="Merging"):
        table = pq.read_table(shard_path)

        if "dedup_cluster_id" in table.column_names:
            table = table.drop("dedup_cluster_id")

        if output_schema is None:
            output_schema = table.schema

        table_bytes = table.nbytes

        if writer is not None and current_bytes + table_bytes > TARGET_SHARD_BYTES:
            writer.close()
            writer = None
            out_idx += 1
            current_bytes = 0

        if writer is None:
            out_path = output_dir / f"data_{out_idx:04d}.parquet"
            writer = pq.ParquetWriter(out_path, output_schema)
            current_bytes = 0

        writer.write_table(table)
        current_bytes += table_bytes
        del table

    if writer is not None:
        writer.close()
        out_idx += 1

    print(f"  Written {out_idx} output files")
    return out_idx


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(output_dir: Path, topic_threshold: float):
    """Compute comprehensive dataset statistics."""
    data_files = sorted(output_dir.glob("data_*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files in {output_dir}")

    total_docs = 0
    total_tokens = 0

    all_topic_scores = []
    all_complexity = []
    all_token_counts = []
    all_groups = []
    domain_tokens = defaultdict(int)

    for fp in tqdm(data_files, desc="Computing stats"):
        table = pq.read_table(fp)
        n = table.num_rows
        total_docs += n

        tc = table.column("token_count").to_numpy()
        total_tokens += tc.sum()
        all_token_counts.append(tc)

        ts_col = table.column("topic_scores")
        flat = ts_col.combine_chunks().values.to_numpy(zero_copy_only=False)
        ts = flat.astype(np.float32).reshape(n, NUM_LABELS)
        all_topic_scores.append(ts)

        cx = table.column("complexity").to_numpy().astype(np.float32)
        all_complexity.append(cx)

        if "assigned_groups" in table.column_names:
            all_groups.extend(table.column("assigned_groups").to_pylist())

        urls = table.column("url").to_pylist()
        for url, tokens in zip(urls, tc):
            try:
                domain = urlparse(str(url)).netloc
                if domain:
                    domain_tokens[domain] += int(tokens)
            except Exception:
                pass

        del table

    topic_scores = np.concatenate(all_topic_scores, axis=0)
    complexity = np.concatenate(all_complexity, axis=0)
    token_counts = np.concatenate(all_token_counts, axis=0)

    stats = {
        "total_documents": int(total_docs),
        "total_tokens": int(total_tokens),
        "num_output_files": len(data_files),
    }

    # Per-group distribution
    stats["group_distribution"] = {}
    for gname, label_indices in GROUP_MAP.items():
        mask = np.zeros(total_docs, dtype=bool)
        for li in label_indices:
            mask |= (topic_scores[:, li] >= topic_threshold)
        group_tokens = int(token_counts[mask].sum())
        stats["group_distribution"][gname] = {
            "display_name": GROUP_DISPLAY_NAMES[gname],
            "docs": int(mask.sum()),
            "tokens": group_tokens,
            "pct_tokens": round(group_tokens / total_tokens * 100, 2) if total_tokens > 0 else 0,
            "target_pct": GROUP_TARGET_PCT[gname] * 100,
        }

    # Per-label distribution (all 17)
    stats["label_distribution"] = {}
    for i, lname in enumerate(LABEL_NAMES):
        mask = topic_scores[:, i] >= topic_threshold
        stats["label_distribution"][lname] = {
            "display_name": LABEL_DISPLAY_NAMES[i],
            "docs": int(mask.sum()),
            "pct": round(mask.mean() * 100, 2) if total_docs > 0 else 0,
        }

    # Complexity distribution
    cbins = complexity_bin_vec(complexity)
    stats["complexity_distribution"] = {}
    for bin_label, lo, hi in COMPLEXITY_BINS:
        mask = cbins == bin_label
        stats["complexity_distribution"][bin_label] = {
            "range": f"[{lo}, {hi})",
            "docs": int(mask.sum()),
            "pct": round(mask.mean() * 100, 2) if total_docs > 0 else 0,
            "target_pct": COMPLEXITY_TARGET_PCT[bin_label] * 100,
            "tokens": int(token_counts[mask].sum()),
        }
    stats["complexity_stats"] = {
        "mean": round(float(complexity.mean()), 3),
        "std": round(float(complexity.std()), 3),
        "median": round(float(np.median(complexity)), 3),
    }

    # Multi-label overlap matrix
    active = (topic_scores >= topic_threshold)
    overlap = np.zeros((NUM_LABELS, NUM_LABELS), dtype=np.int64)
    for i in range(NUM_LABELS):
        for j in range(i, NUM_LABELS):
            count = int((active[:, i] & active[:, j]).sum())
            overlap[i, j] = count
            overlap[j, i] = count
    stats["label_overlap_matrix"] = {
        "labels": LABEL_DISPLAY_NAMES,
        "matrix": overlap.tolist(),
    }

    # Multi-label stats
    num_active = active.sum(axis=1)
    stats["multi_label_stats"] = {
        "mean_labels": round(float(num_active.mean()), 2),
        "pct_1_label": round(float((num_active == 1).mean() * 100), 1),
        "pct_2plus": round(float((num_active >= 2).mean() * 100), 1),
        "pct_3plus": round(float((num_active >= 3).mean() * 100), 1),
    }

    # Top 20 domains
    top_domains = sorted(domain_tokens.items(), key=lambda x: x[1], reverse=True)[:20]
    stats["top_domains"] = [
        {"domain": d, "tokens": t, "pct": round(t / total_tokens * 100, 2)}
        for d, t in top_domains
    ]

    # Token count distribution
    stats["token_count_distribution"] = {
        f"p{p}": int(np.percentile(token_counts, p))
        for p in [10, 25, 50, 75, 90]
    }
    stats["token_count_distribution"]["mean"] = int(token_counts.mean())

    return stats, topic_scores, complexity, token_counts


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def generate_plots(stats, topic_scores, complexity, token_counts,
                   topic_threshold, output_dir):
    """Generate summary visualizations."""
    sns.set_theme(style="whitegrid")

    # Topic × Complexity heatmap
    _plot_topic_complexity_heatmap(topic_scores, complexity, token_counts,
                                  topic_threshold, output_dir)

    # Group distribution bar chart
    _plot_group_distribution(stats, output_dir)

    # Label overlap heatmap
    _plot_overlap_heatmap(stats, output_dir)

    # Complexity distribution
    _plot_complexity_distribution(stats, output_dir)


def _plot_topic_complexity_heatmap(topic_scores, complexity, token_counts,
                                  threshold, output_dir):
    """Topic group × complexity bin heatmap of token counts."""
    groups = list(GROUP_MAP.keys())
    bin_labels = [b[0] for b in COMPLEXITY_BINS]
    cbins = complexity_bin_vec(complexity)

    matrix = np.zeros((len(groups), len(bin_labels)), dtype=np.int64)
    for gi, gname in enumerate(groups):
        label_indices = GROUP_MAP[gname]
        mask = np.zeros(len(topic_scores), dtype=bool)
        for li in label_indices:
            mask |= (topic_scores[:, li] >= threshold)
        for bi, bl in enumerate(bin_labels):
            cell_mask = mask & (cbins == bl)
            matrix[gi, bi] = token_counts[cell_mask].sum()

    # Convert to billions for readability
    matrix_b = matrix / 1e9

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix_b, annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=bin_labels,
        yticklabels=[GROUP_DISPLAY_NAMES[g] for g in groups],
        ax=ax,
    )
    ax.set_xlabel("Complexity Level")
    ax.set_ylabel("Topic Group")
    ax.set_title("Token Distribution (Billions): Topic Group × Complexity")
    fig.tight_layout()
    fig.savefig(output_dir / "heatmap_topic_complexity.png", dpi=150)
    plt.close(fig)
    print("  Saved heatmap_topic_complexity.png")


def _plot_group_distribution(stats, output_dir):
    """Actual vs target group distribution."""
    groups = list(stats["group_distribution"].keys())
    actual = [stats["group_distribution"][g]["pct_tokens"] for g in groups]
    target = [stats["group_distribution"][g]["target_pct"] for g in groups]
    names = [GROUP_DISPLAY_NAMES[g] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    w = 0.35
    ax.bar(x - w/2, actual, w, label="Actual %")
    ax.bar(x + w/2, target, w, label="Target %", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of total tokens")
    ax.set_title("Topic Group Distribution: Actual vs Target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "group_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved group_distribution.png")


def _plot_overlap_heatmap(stats, output_dir):
    """Label co-occurrence heatmap."""
    matrix = np.array(stats["label_overlap_matrix"]["matrix"])
    labels = stats["label_overlap_matrix"]["labels"]

    # Normalize by diagonal (co-occurrence rate)
    diag = np.diag(matrix).astype(float)
    diag[diag == 0] = 1
    norm = matrix / diag[:, None]

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, annot_kws={"size": 6},
    )
    ax.set_title("Label Co-occurrence Rate (row-normalized)")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "label_overlap_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved label_overlap_heatmap.png")


def _plot_complexity_distribution(stats, output_dir):
    """Complexity level distribution: actual vs target."""
    bins = list(stats["complexity_distribution"].keys())
    actual = [stats["complexity_distribution"][b]["pct"] for b in bins]
    target = [stats["complexity_distribution"][b]["target_pct"] for b in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bins))
    w = 0.35
    ax.bar(x - w/2, actual, w, label="Actual %")
    ax.bar(x + w/2, target, w, label="Target %", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{stats['complexity_distribution'][b]['range']}" for b in bins])
    ax.set_ylabel("% of documents")
    ax.set_title("Complexity Distribution: Actual vs Target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "complexity_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved complexity_distribution.png")


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def generate_dataset_card(stats, output_dir: Path, topic_threshold: float,
                          repo_id: str | None = None):
    """Generate a HuggingFace dataset card (README.md)."""

    total_tokens_b = stats["total_tokens"] / 1e9

    group_table = "| Group | Target % | Actual % | Tokens | Docs |\n"
    group_table += "|-------|----------|----------|--------|------|\n"
    for gname, gdata in stats["group_distribution"].items():
        group_table += (f"| {gdata['display_name']} | {gdata['target_pct']:.1f}% | "
                        f"{gdata['pct_tokens']:.1f}% | {gdata['tokens']:,} | {gdata['docs']:,} |\n")

    complexity_table = "| Level | Range | Target % | Actual % | Docs |\n"
    complexity_table += "|-------|-------|----------|----------|------|\n"
    for bl, bdata in stats["complexity_distribution"].items():
        complexity_table += (f"| {bl} | {bdata['range']} | {bdata['target_pct']:.1f}% | "
                             f"{bdata['pct']:.1f}% | {bdata['docs']:,} |\n")

    domain_table = "| Domain | Tokens | % |\n|--------|--------|---|\n"
    for d in stats["top_domains"][:10]:
        domain_table += f"| {d['domain']} | {d['tokens']:,} | {d['pct']:.2f}% |\n"

    tc = stats["token_count_distribution"]

    card = f"""---
license: odc-by
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 10B<n<100B
tags:
  - curated
  - fineweb
  - education
  - stem
---

# FineWeb-Edu Curated

A curated subset of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
optimised for training language models with strong STEM and structured-reasoning capabilities.

## Dataset Summary

- **Total documents:** {stats['total_documents']:,}
- **Total tokens:** {total_tokens_b:.1f}B
- **Source:** FineWeb-Edu sample-100BT
- **Curation method:** Multi-label topic classification + complexity scoring + distribution-aware sampling + MinHash deduplication

## Topic Distribution

Documents are classified into 11 target groups using a 17-label multi-label classifier
(ModernBERT-base, sigmoid threshold={topic_threshold}). STEM-core topics (Mathematics,
Computer Science, ML/AI) are boosted relative to their natural distribution.

{group_table}

## Complexity Distribution

Reasoning complexity scored by a ModernBERT-base regression model (1.0-4.0 scale).
Mean complexity: {stats['complexity_stats']['mean']:.2f}, Median: {stats['complexity_stats']['median']:.2f}.

{complexity_table}

## Multi-Label Statistics

- Mean labels per document: {stats['multi_label_stats']['mean_labels']:.1f}
- Documents with 2+ labels: {stats['multi_label_stats']['pct_2plus']:.1f}%
- Documents with 3+ labels: {stats['multi_label_stats']['pct_3plus']:.1f}%

## Token Count Distribution

| Percentile | Tokens |
|------------|--------|
| P10 | {tc['p10']:,} |
| P25 | {tc['p25']:,} |
| P50 (median) | {tc['p50']:,} |
| P75 | {tc['p75']:,} |
| P90 | {tc['p90']:,} |
| Mean | {tc['mean']:,} |

## Top Domains

{domain_table}

## Schema

Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Document text |
| `url` | string | Source URL |
| `token_count` | int | Token count |
| `dump` | string | Common Crawl dump identifier |
| `topic_scores` | list[float] | 17-dim sigmoid scores from topic classifier |
| `complexity` | float | Reasoning complexity score (1.0-4.0) |
| `assigned_groups` | list[string] | Target groups at threshold={topic_threshold} |
| `relevance_score` | float | Composite relevance score used for sampling |

## Methodology

1. **Classification:** Both classifiers (topic: 17-label multi-label, complexity: regression)
   run on each document. Full sigmoid vectors stored for maximum flexibility.
2. **Filtering:** Priority-based sampling targeting STEM-core topics first, with complexity
   distribution targets within each group. Multi-label documents count toward multiple quotas.
3. **Deduplication:** MinHash LSH (128 perms, 13-gram word shingles, 0.7 Jaccard threshold).
   Highest-relevance document kept from each cluster.
"""

    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card)
    print(f"  Dataset card written to {card_path}")


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------

def upload_to_hub(output_dir: Path, repo_id: str):
    """Upload the output directory to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"\nUploading to HuggingFace Hub: {repo_id}")

    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=str(output_dir),
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Upload complete: https://huggingface.co/datasets/{repo_id}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    topic_threshold = args.topic_threshold

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    t0 = time.time()

    # Merge shards
    print("\n--- Merging shards ---")
    num_files = merge_shards(input_dir, output_dir)

    # Compute stats
    print("\n--- Computing statistics ---")
    stats, topic_scores, complexity, token_counts = compute_stats(
        output_dir, topic_threshold,
    )

    # Save stats JSON
    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats written to {stats_path}")

    # Generate plots
    print("\n--- Generating plots ---")
    generate_plots(stats, topic_scores, complexity, token_counts,
                   topic_threshold, output_dir)

    # Generate dataset card
    print("\n--- Generating dataset card ---")
    generate_dataset_card(stats, output_dir, topic_threshold, args.repo_id)

    elapsed = time.time() - t0
    print(f"\nExport complete in {elapsed:.0f}s")

    total_b = stats["total_tokens"] / 1e9
    print(f"  Documents: {stats['total_documents']:,}")
    print(f"  Tokens: {total_b:.1f}B")
    print(f"  Files: {num_files}")
    print(f"  Output: {output_dir}")

    # Upload
    if args.upload and args.repo_id:
        upload_to_hub(output_dir, args.repo_id)
    elif args.upload and not args.repo_id:
        print("\nWARNING: --upload specified but no --repo-id provided. Skipping upload.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Merge, compute stats, generate dataset card, upload",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory of deduped parquet shards from Stage 3")
    parser.add_argument("--output-dir", required=True,
                        help="Final output directory")
    parser.add_argument("--topic-threshold", type=float, default=DEFAULT_TOPIC_THRESHOLD,
                        help=f"Threshold for label-level stats (default: {DEFAULT_TOPIC_THRESHOLD})")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub after export")
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace repo ID (e.g., myuser/fineweb-edu-curated)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
