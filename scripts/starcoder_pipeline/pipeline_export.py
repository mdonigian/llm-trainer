#!/usr/bin/env python3
"""
Stage 4: Export — merge shards, compute stats, generate dataset card, upload.

Merges deduped parquet shards into consistently-sized output files (~500MB each),
computes code-specific statistics, generates a HuggingFace dataset card, and
optionally uploads to the Hub.

Usage:
  python pipeline_export.py --input-dir deduped_shards/ --output-dir final_output/
  python pipeline_export.py --input-dir deduped_shards/ --output-dir final_output/ \
      --upload --repo-id myuser/starcoderdata-curated
"""

import argparse
import json
import logging
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import seaborn as sns
from tqdm.auto import tqdm

from pipeline_config import (
    CONTENT_GROUP_DISPLAY,
    CONTENT_GROUP_MAP,
    CONTENT_TYPES,
    LANGUAGE_SLICES,
    LANGUAGE_SLICE_MAP,
    NUM_CONTENT_TYPES,
    QUALITY_NAMES,
    SD_BINS,
    SD_TARGET_PCT,
    STRUCTURED_DATA_NAMES,
)

logger = logging.getLogger(__name__)

TARGET_SHARD_BYTES = 500 * 1024 * 1024  # 500MB


# ---------------------------------------------------------------------------
# Merge shards
# ---------------------------------------------------------------------------

def merge_shards(input_dir: Path, output_dir: Path):
    shard_files = sorted(input_dir.glob("deduped_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No deduped shard files in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Merging {len(shard_files)} shards into ~500MB output files...")

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

def compute_stats(output_dir: Path):
    data_files = sorted(output_dir.glob("data_*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data files in {output_dir}")

    total_docs = 0
    total_tokens = 0

    all_quality = []
    all_sd = []
    all_ct = []
    all_token_counts = []
    lang_tokens = defaultdict(int)
    lang_docs = defaultdict(int)

    for fp in tqdm(data_files, desc="Computing stats"):
        table = pq.read_table(fp)
        n = table.num_rows
        total_docs += n

        tc = table.column("token_count").to_numpy()
        total_tokens += tc.sum()
        all_token_counts.append(tc)

        quality = table.column("quality").to_numpy().astype(np.float32)
        all_quality.append(quality)

        sd = table.column("structured_data").to_numpy().astype(np.float32)
        all_sd.append(sd)

        ct = table.column("content_type").to_pylist()
        all_ct.extend(ct)

        if "lang" in table.column_names:
            langs = table.column("lang").to_pylist()
            for lang, tokens in zip(langs, tc):
                lang = lang or "unknown"
                lang_tokens[lang] += int(tokens)
                lang_docs[lang] += 1

        del table

    quality = np.concatenate(all_quality)
    sd = np.concatenate(all_sd)
    token_counts = np.concatenate(all_token_counts)

    stats = {
        "total_documents": int(total_docs),
        "total_tokens": int(total_tokens),
        "num_output_files": len(data_files),
    }

    # Content group distribution
    from pipeline_config import assign_content_group_vec
    content_groups = assign_content_group_vec(all_ct)
    stats["group_distribution"] = {}
    for gname in CONTENT_GROUP_MAP:
        mask = np.array([g == gname for g in content_groups])
        group_tokens = int(token_counts[mask].sum()) if mask.any() else 0
        stats["group_distribution"][gname] = {
            "display_name": CONTENT_GROUP_DISPLAY[gname],
            "docs": int(mask.sum()),
            "tokens": group_tokens,
            "pct_tokens": round(group_tokens / total_tokens * 100, 2) if total_tokens > 0 else 0,
            "pct_of_total": round(group_tokens / total_tokens * 100, 2) if total_tokens > 0 else 0,
        }

    # Content type distribution
    ct_counts = Counter(all_ct)
    stats["content_type_distribution"] = {
        ct: int(ct_counts.get(ct, 0)) for ct in CONTENT_TYPES
    }

    # Quality distribution
    q_rounded = np.clip(np.round(quality), 1, 5).astype(int)
    stats["quality_distribution"] = {
        str(i): int((q_rounded == i).sum()) for i in range(1, 6)
    }
    stats["quality_stats"] = {
        "mean": round(float(quality.mean()), 3),
        "std": round(float(quality.std()), 3),
        "median": round(float(np.median(quality)), 3),
    }

    # Structured data distribution
    from pipeline_config import sd_bin_vec
    sd_bins = sd_bin_vec(sd)
    stats["sd_distribution"] = {}
    for bin_label, lo, hi in SD_BINS:
        mask = sd_bins == bin_label
        bin_tokens = int(token_counts[mask].sum())
        stats["sd_distribution"][bin_label] = {
            "range": f"[{lo}, {hi})",
            "docs": int(mask.sum()),
            "tokens": bin_tokens,
            "pct": round(mask.mean() * 100, 2) if total_docs > 0 else 0,
            "target_pct": SD_TARGET_PCT[bin_label] * 100,
        }
    stats["sd_stats"] = {
        "mean": round(float(sd.mean()), 3),
        "std": round(float(sd.std()), 3),
        "median": round(float(np.median(sd)), 3),
    }

    # Language distribution
    top_lang_by_tokens = sorted(lang_tokens.items(), key=lambda x: x[1], reverse=True)
    stats["top_languages"] = [
        {
            "language": lang,
            "tokens": tokens,
            "docs": lang_docs[lang],
            "pct_tokens": round(tokens / total_tokens * 100, 2) if total_tokens > 0 else 0,
        }
        for lang, tokens in top_lang_by_tokens[:25]
    ]

    # Per-slice distribution (the primary axis of our filtering)
    from pipeline_config import resolve_language_to_slice
    slice_tokens: dict[str, int] = defaultdict(int)
    slice_docs: dict[str, int] = defaultdict(int)
    for lang, tok in lang_tokens.items():
        sname = resolve_language_to_slice(lang)
        if sname:
            slice_tokens[sname] += tok
            slice_docs[sname] += lang_docs[lang]

    stats["slice_distribution"] = {}
    for s in LANGUAGE_SLICES:
        stok = slice_tokens.get(s.name, 0)
        sdoc = slice_docs.get(s.name, 0)
        stats["slice_distribution"][s.name] = {
            "description": s.description,
            "strategy": s.strategy,
            "languages": s.languages,
            "target_tokens": s.target_tokens,
            "actual_tokens": stok,
            "docs": sdoc,
            "pct_of_target": round(stok / s.target_tokens * 100, 1) if s.target_tokens > 0 else 0,
            "pct_of_total": round(stok / total_tokens * 100, 1) if total_tokens > 0 else 0,
        }

    # Token count distribution
    stats["token_count_distribution"] = {
        f"p{p}": int(np.percentile(token_counts, p))
        for p in [10, 25, 50, 75, 90]
    }
    stats["token_count_distribution"]["mean"] = int(token_counts.mean())

    return stats, quality, sd, token_counts


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def generate_plots(stats, quality, sd, token_counts, output_dir: Path):
    sns.set_theme(style="whitegrid")

    _plot_slice_distribution(stats, output_dir)
    _plot_group_distribution(stats, output_dir)
    _plot_sd_distribution(stats, output_dir)
    _plot_quality_sd_heatmap(quality, sd, output_dir)
    _plot_language_distribution(stats, output_dir)


def _plot_slice_distribution(stats, output_dir):
    """Bar chart comparing actual vs target tokens per language slice."""
    if "slice_distribution" not in stats:
        return

    slices = list(stats["slice_distribution"].keys())
    actual_m = [stats["slice_distribution"][s]["actual_tokens"] / 1e6 for s in slices]
    target_m = [stats["slice_distribution"][s]["target_tokens"] / 1e6 for s in slices]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(slices))
    w = 0.35
    ax.bar(x - w/2, actual_m, w, label="Actual (M tokens)", color="#2196F3")
    ax.bar(x + w/2, target_m, w, label="Target (M tokens)", alpha=0.7, color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(slices, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Millions of tokens")
    ax.set_title("Language Slice Distribution: Actual vs Target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "slice_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved slice_distribution.png")


def _plot_group_distribution(stats, output_dir):
    groups = list(stats["group_distribution"].keys())
    actual = [stats["group_distribution"][g]["pct_tokens"] for g in groups]
    names = [CONTENT_GROUP_DISPLAY[g] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    ax.bar(x, actual, label="% of total tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of total tokens")
    ax.set_title("Content Group Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "group_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved group_distribution.png")


def _plot_sd_distribution(stats, output_dir):
    bins = list(stats["sd_distribution"].keys())
    actual = [stats["sd_distribution"][b]["pct"] for b in bins]
    target = [stats["sd_distribution"][b]["target_pct"] for b in bins]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(bins))
    w = 0.35
    ax.bar(x - w/2, actual, w, label="Actual %")
    ax.bar(x + w/2, target, w, label="Target %", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n{stats['sd_distribution'][b]['range']}" for b in bins])
    ax.set_ylabel("% of documents")
    ax.set_title("Structured Data Relevance Distribution: Actual vs Target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sd_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved sd_distribution.png")


def _plot_quality_sd_heatmap(quality, sd, output_dir):
    q_rounded = np.clip(np.round(quality), 1, 5).astype(int)
    sd_rounded = np.clip(np.round(sd), 0, 3).astype(int)

    matrix = np.zeros((5, 4), dtype=int)
    for q, s in zip(q_rounded, sd_rounded):
        matrix[q - 1, s] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt=",d", cmap="YlOrRd",
        xticklabels=[STRUCTURED_DATA_NAMES[i] for i in range(4)],
        yticklabels=[f"Q{i}: {QUALITY_NAMES[i]}" for i in range(1, 6)],
        ax=ax,
    )
    ax.set_xlabel("Structured Data Relevance")
    ax.set_ylabel("Quality Level")
    ax.set_title("Quality × Structured Data Relevance (Code Files)")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_sd_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved quality_sd_heatmap.png")


def _plot_language_distribution(stats, output_dir):
    top_langs = stats.get("top_languages", [])[:20]
    if not top_langs:
        return
    labels = [l["language"] for l in top_langs]
    values = [l["pct_tokens"] for l in top_langs]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(labels)), values, align="center")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of total tokens")
    ax.set_title("Programming Language Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "language_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved language_distribution.png")


# ---------------------------------------------------------------------------
# Dataset card
# ---------------------------------------------------------------------------

def generate_dataset_card(stats, output_dir: Path, repo_id: str | None = None):
    total_tokens_b = stats["total_tokens"] / 1e9

    # Language slice table (primary axis)
    slice_table = "| Slice | Strategy | Target | Actual | % |\n"
    slice_table += "|-------|----------|--------|--------|---|\n"
    for sname, sdata in stats.get("slice_distribution", {}).items():
        target_m = sdata["target_tokens"] / 1e6
        actual_m = sdata["actual_tokens"] / 1e6
        slice_table += (f"| {sname} | {sdata['strategy']} | {target_m:.0f}M | "
                        f"{actual_m:.0f}M | {sdata['pct_of_target']:.1f}% |\n")

    # Content group table (secondary axis)
    group_table = "| Group | % of Tokens | Tokens | Files |\n"
    group_table += "|-------|-------------|--------|-------|\n"
    for gname, gdata in stats["group_distribution"].items():
        group_table += (f"| {gdata['display_name']} | "
                        f"{gdata['pct_tokens']:.1f}% | {gdata['tokens']:,} | {gdata['docs']:,} |\n")

    # SD distribution table
    sd_table = "| Level | Range | Target % | Actual % | Files |\n"
    sd_table += "|-------|-------|----------|----------|-------|\n"
    for bl, bdata in stats["sd_distribution"].items():
        sd_table += (f"| {bl} | {bdata['range']} | {bdata['target_pct']:.1f}% | "
                     f"{bdata['pct']:.1f}% | {bdata['docs']:,} |\n")

    # Quality distribution table
    q_table = "| Level | Description | Files |\n"
    q_table += "|-------|-------------|-------|\n"
    for level in range(1, 6):
        count = stats["quality_distribution"].get(str(level), 0)
        q_table += f"| {level} | {QUALITY_NAMES[level]} | {count:,} |\n"

    # Language table
    lang_table = "| Language | % Tokens | Files |\n|----------|----------|-------|\n"
    for l in stats.get("top_languages", [])[:15]:
        lang_table += f"| {l['language']} | {l['pct_tokens']:.1f}% | {l['docs']:,} |\n"

    tc = stats["token_count_distribution"]

    card = f"""---
license: odc-by
task_categories:
  - text-generation
language:
  - code
size_categories:
  - 1B<n<10B
tags:
  - curated
  - starcoderdata
  - code
  - structured-data
  - multi-task-filter
---

# StarCoderData Curated

A curated subset of [StarCoderData](https://huggingface.co/datasets/bigcode/starcoderdata)
optimised for training a 500M parameter model focused on structured data output
(JSON generation, function calling, schema compliance).

## Dataset Summary

- **Total code files:** {stats['total_documents']:,}
- **Total tokens:** {total_tokens_b:.1f}B (target: 3.5B)
- **Source:** bigcode/starcoderdata
- **Classifier:** [mdonigian/code-curator-v1](https://huggingface.co/mdonigian/code-curator-v1) (UniXcoder-base, multi-task)
- **Curation:** Per-language-slice filtering + compression ratio pre-filter + MinHash deduplication

## Filtering Strategy

Each language slice has its own filtering strategy and token budget:

1. **Schema languages** (JSON/YAML/SQL/protobuf/GraphQL): ~800M tokens, light filter (quality floor only)
2. **TypeScript** (relevance ≥ 2): ~600M tokens, relevance classifier
3. **Python** (relevance ≥ 2): ~600M tokens, relevance classifier
4. **Rust/Go/Java** (relevance ≥ 2): ~600M tokens, relevance classifier
5. **Jupyter notebooks**: ~400M tokens, passthrough (already structured)
6. **GitHub Issues** (technical): ~500M tokens, keyword filter

Pre-filtering applied to all slices:
- zlib compression ratio < 0.10 catches extreme repetition
- Quality floor: drop files with predicted quality ≤ 1 (broken/gibberish)
- MinHash LSH deduplication with line-level 5-gram shingles

## Language Slice Distribution

{slice_table}

## Content Group Distribution

{group_table}

## Structured Data Relevance Distribution

The strongest classifier signal (Spearman 0.81). SD2+ files contain significant
structured data patterns (API endpoints, JSON parsing, schema definitions, etc.).

{sd_table}

## Quality Distribution

Quality mean: {stats['quality_stats']['mean']:.2f}, Median: {stats['quality_stats']['median']:.2f}.

{q_table}

## Programming Languages

{lang_table}

## Token Count Distribution

| Percentile | Tokens |
|------------|--------|
| P10 | {tc['p10']:,} |
| P25 | {tc['p25']:,} |
| P50 (median) | {tc['p50']:,} |
| P75 | {tc['p75']:,} |
| P90 | {tc['p90']:,} |
| Mean | {tc['mean']:,} |

## Schema

Each row contains:

| Field | Type | Description |
|-------|------|-------------|
| `content` | string | Source code text |
| `lang` | string | Programming language |
| `size` | int | File size in bytes |
| `token_count` | int | Estimated token count (size // 4) |
| `quality` | float | Code quality score (1-5) |
| `structured_data` | float | Structured data relevance (0-3) |
| `content_type` | string | Content type (9 classes) |
| `language_slice` | string | Language slice (schema_languages, typescript, python, etc.) |
| `relevance_score` | float | Composite relevance score |

## Methodology

1. **Classification:** Single multi-task UniXcoder-base model with three heads runs
   on each code file. Stores raw float scores for maximum flexibility.
2. **Pre-filtering:** zlib compression ratio filter removes repetitive boilerplate
   before GPU inference.
3. **Filtering:** Quality floor + priority-based sampling targeting high structured
   data relevance and balanced content type distribution.
4. **Deduplication:** MinHash LSH (128 perms, 5-line shingles, 0.7 Jaccard threshold).
   Highest-relevance code file kept from each cluster.
"""

    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card)
    print(f"  Dataset card written to {card_path}")


# ---------------------------------------------------------------------------
# HuggingFace upload
# ---------------------------------------------------------------------------

def upload_to_hub(output_dir: Path, repo_id: str):
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

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    t0 = time.time()

    print("\n--- Merging shards ---")
    num_files = merge_shards(input_dir, output_dir)

    print("\n--- Computing statistics ---")
    stats, quality, sd, token_counts = compute_stats(output_dir)

    stats_path = output_dir / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Stats written to {stats_path}")

    print("\n--- Generating plots ---")
    generate_plots(stats, quality, sd, token_counts, output_dir)

    print("\n--- Generating dataset card ---")
    generate_dataset_card(stats, output_dir, args.repo_id)

    elapsed = time.time() - t0
    print(f"\nExport complete in {elapsed:.0f}s")

    total_b = stats["total_tokens"] / 1e9
    print(f"  Code files: {stats['total_documents']:,}")
    print(f"  Tokens: {total_b:.1f}B")
    print(f"  Files: {num_files}")
    print(f"  Output: {output_dir}")

    if args.upload and args.repo_id:
        upload_to_hub(output_dir, args.repo_id)
    elif args.upload and not args.repo_id:
        print("\nWARNING: --upload specified but no --repo-id provided. Skipping upload.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4: Merge, compute stats, generate dataset card, upload (StarCoder)",
    )
    parser.add_argument("--input-dir", required=True,
                        help="Directory of deduped parquet shards from Stage 3")
    parser.add_argument("--output-dir", required=True,
                        help="Final output directory")
    parser.add_argument("--upload", action="store_true",
                        help="Upload to HuggingFace Hub after export")
    parser.add_argument("--repo-id", default=None,
                        help="HuggingFace repo ID (e.g., myuser/starcoderdata-curated)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
