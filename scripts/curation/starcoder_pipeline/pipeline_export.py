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

    # Build a set of slice names that went through the classifier
    classified_slice_names = {s.name for s in LANGUAGE_SLICES if s.strategy == "relevance_filter"}

    total_docs = 0
    total_tokens = 0

    all_quality = []
    all_sd = []
    all_ct = []
    all_token_counts = []
    all_slices = []
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

        if "language_slice" in table.column_names:
            all_slices.extend(table.column("language_slice").to_pylist())
        else:
            all_slices.extend(["unknown"] * n)

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
    slice_arr = np.array(all_slices)

    # Mask for rows that went through the classifier (relevance_filter slices)
    classified_mask = np.isin(slice_arr, list(classified_slice_names))
    classified_quality = quality[classified_mask]
    classified_sd = sd[classified_mask]
    classified_ct = [ct for ct, m in zip(all_ct, classified_mask) if m]
    classified_tc = token_counts[classified_mask]

    stats = {
        "total_documents": int(total_docs),
        "total_tokens": int(total_tokens),
        "num_output_files": len(data_files),
        "classified_documents": int(classified_mask.sum()),
        "classified_tokens": int(classified_tc.sum()),
    }

    # Content group distribution (classified rows only — non-classified have "unclassified" content_type)
    from pipeline_config import assign_content_group_vec
    if classified_ct:
        content_groups = assign_content_group_vec(classified_ct)
        stats["group_distribution"] = {}
        classified_total_tokens = classified_tc.sum()
        for gname in CONTENT_GROUP_MAP:
            mask = np.array([g == gname for g in content_groups])
            group_tokens = int(classified_tc[mask].sum()) if mask.any() else 0
            stats["group_distribution"][gname] = {
                "display_name": CONTENT_GROUP_DISPLAY[gname],
                "docs": int(mask.sum()),
                "tokens": group_tokens,
                "pct_tokens": round(group_tokens / classified_total_tokens * 100, 2) if classified_total_tokens > 0 else 0,
            }
    else:
        stats["group_distribution"] = {}

    # Content type distribution (classified rows only)
    ct_counts = Counter(classified_ct)
    stats["content_type_distribution"] = {
        ct: int(ct_counts.get(ct, 0)) for ct in CONTENT_TYPES
    }

    # Quality distribution (classified rows only)
    if len(classified_quality) > 0:
        q_rounded = np.clip(np.round(classified_quality), 1, 5).astype(int)
        stats["quality_distribution"] = {
            str(i): int((q_rounded == i).sum()) for i in range(1, 6)
        }
        stats["quality_stats"] = {
            "mean": round(float(classified_quality.mean()), 3),
            "std": round(float(classified_quality.std()), 3),
            "median": round(float(np.median(classified_quality)), 3),
        }
    else:
        stats["quality_distribution"] = {str(i): 0 for i in range(1, 6)}
        stats["quality_stats"] = {"mean": 0, "std": 0, "median": 0}

    # Structured data distribution (classified rows only)
    from pipeline_config import sd_bin_vec
    if len(classified_sd) > 0:
        sd_bins = sd_bin_vec(classified_sd)
        stats["sd_distribution"] = {}
        n_classified = len(classified_sd)
        for bin_label, lo, hi in SD_BINS:
            mask = sd_bins == bin_label
            bin_tokens = int(classified_tc[mask].sum())
            stats["sd_distribution"][bin_label] = {
                "range": f"[{lo}, {hi})",
                "docs": int(mask.sum()),
                "tokens": bin_tokens,
                "pct": round(mask.mean() * 100, 2) if n_classified > 0 else 0,
                "target_pct": SD_TARGET_PCT[bin_label] * 100,
            }
        stats["sd_stats"] = {
            "mean": round(float(classified_sd.mean()), 3),
            "std": round(float(classified_sd.std()), 3),
            "median": round(float(np.median(classified_sd)), 3),
        }
    else:
        stats["sd_distribution"] = {}
        stats["sd_stats"] = {"mean": 0, "std": 0, "median": 0}

    # Language distribution (all rows)
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

        slice_entry = {
            "description": s.description,
            "strategy": s.strategy,
            "languages": s.languages,
            "target_tokens": s.target_tokens,
            "actual_tokens": stok,
            "docs": sdoc,
            "pct_of_target": round(stok / s.target_tokens * 100, 1) if s.target_tokens > 0 else 0,
            "pct_of_total": round(stok / total_tokens * 100, 1) if total_tokens > 0 else 0,
        }

        # Per-slice quality/SD stats for classified slices
        if s.name in classified_slice_names:
            smask = slice_arr == s.name
            sq = quality[smask]
            ss = sd[smask]
            if len(sq) > 0:
                slice_entry["quality_mean"] = round(float(sq.mean()), 2)
                slice_entry["sd_mean"] = round(float(ss.mean()), 2)

        stats["slice_distribution"][s.name] = slice_entry

    # Token count distribution (all rows)
    stats["token_count_distribution"] = {
        f"p{p}": int(np.percentile(token_counts, p))
        for p in [10, 25, 50, 75, 90]
    }
    stats["token_count_distribution"]["mean"] = int(token_counts.mean())

    return stats, classified_quality, classified_sd, token_counts


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
    if not stats.get("group_distribution"):
        return
    groups = list(stats["group_distribution"].keys())
    actual = [stats["group_distribution"][g]["pct_tokens"] for g in groups]
    names = [CONTENT_GROUP_DISPLAY[g] for g in groups]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(groups))
    ax.bar(x, actual, label="% of total tokens")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of total tokens")
    ax.set_title("Content Group Distribution (classifier-scored slices only)")
    fig.tight_layout()
    fig.savefig(output_dir / "group_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved group_distribution.png")


def _plot_sd_distribution(stats, output_dir):
    if not stats.get("sd_distribution"):
        return
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
    ax.set_title("Structured Data Relevance (classifier-scored slices only)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sd_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved sd_distribution.png")


def _plot_quality_sd_heatmap(quality, sd, output_dir):
    if len(quality) == 0:
        return
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
    ax.set_title("Quality × SD Relevance (classifier-scored slices only)")
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
    classified_docs = stats.get("classified_documents", 0)
    classified_tokens_b = stats.get("classified_tokens", 0) / 1e9
    non_classified_docs = stats["total_documents"] - classified_docs
    non_classified_tokens_b = total_tokens_b - classified_tokens_b

    # ---- Language slice table with per-slice detail ----
    slice_table = "| Slice | Strategy | Languages | Target | Actual | % of Target |\n"
    slice_table += "|-------|----------|-----------|--------|--------|-------------|\n"
    for sname, sdata in stats.get("slice_distribution", {}).items():
        target_m = sdata["target_tokens"] / 1e6
        actual_m = sdata["actual_tokens"] / 1e6
        langs_str = ", ".join(sdata["languages"][:4])
        if len(sdata["languages"]) > 4:
            langs_str += f" +{len(sdata['languages']) - 4} more"
        slice_table += (f"| {sname} | {sdata['strategy']} | {langs_str} | "
                        f"{target_m:.0f}M | {actual_m:.0f}M | {sdata['pct_of_target']:.1f}% |\n")

    # ---- Per-slice classifier stats (only for classified slices) ----
    classified_slice_table = "| Slice | Files | Tokens | Avg Quality | Avg SD Relevance |\n"
    classified_slice_table += "|-------|-------|--------|-------------|------------------|\n"
    for sname, sdata in stats.get("slice_distribution", {}).items():
        if sdata["strategy"] != "relevance_filter":
            continue
        actual_m = sdata["actual_tokens"] / 1e6
        qm = sdata.get("quality_mean", "—")
        sm = sdata.get("sd_mean", "—")
        if isinstance(qm, float):
            qm = f"{qm:.2f}"
        if isinstance(sm, float):
            sm = f"{sm:.2f}"
        classified_slice_table += (f"| {sname} | {sdata['docs']:,} | "
                                   f"{actual_m:.0f}M | {qm} | {sm} |\n")

    # ---- Non-classified slice summary ----
    non_classified_slice_table = "| Slice | Strategy | Files | Tokens | How Filtered |\n"
    non_classified_slice_table += "|-------|----------|-------|--------|-------------|\n"
    strategy_descriptions = {
        "light_filter": "Quality floor (≥1.5) + token budget, randomly sampled",
        "keyword_filter": "Keyword match for structured-data topics + quality floor",
        "passthrough": "Quality floor only (inherently structured)",
    }
    for sname, sdata in stats.get("slice_distribution", {}).items():
        if sdata["strategy"] == "relevance_filter":
            continue
        actual_m = sdata["actual_tokens"] / 1e6
        how = strategy_descriptions.get(sdata["strategy"], sdata["strategy"])
        non_classified_slice_table += (f"| {sname} | {sdata['strategy']} | "
                                       f"{sdata['docs']:,} | {actual_m:.0f}M | {how} |\n")

    # ---- Content group table (classified rows only) ----
    group_table = "| Group | % of Classified Tokens | Tokens | Files |\n"
    group_table += "|-------|-----------------------|--------|-------|\n"
    for gname, gdata in stats.get("group_distribution", {}).items():
        group_table += (f"| {gdata['display_name']} | "
                        f"{gdata['pct_tokens']:.1f}% | {gdata['tokens']:,} | {gdata['docs']:,} |\n")

    # ---- SD distribution table (classified rows only) ----
    sd_table = "| Level | Range | Target % | Actual % | Files |\n"
    sd_table += "|-------|-------|----------|----------|-------|\n"
    for bl, bdata in stats.get("sd_distribution", {}).items():
        sd_table += (f"| {bl} | {bdata['range']} | {bdata['target_pct']:.1f}% | "
                     f"{bdata['pct']:.1f}% | {bdata['docs']:,} |\n")

    # ---- Quality distribution table (classified rows only) ----
    q_table = "| Level | Description | Files |\n"
    q_table += "|-------|-------------|-------|\n"
    for level in range(1, 6):
        count = stats["quality_distribution"].get(str(level), 0)
        q_table += f"| {level} | {QUALITY_NAMES[level]} | {count:,} |\n"

    # ---- Language table ----
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
- **Classifier-scored files:** {classified_docs:,} ({classified_tokens_b:.1f}B tokens)
- **Non-classified files:** {non_classified_docs:,} ({non_classified_tokens_b:.1f}B tokens) — filtered by heuristics, not the classifier
- **Source:** bigcode/starcoderdata
- **Classifier:** [mdonigian/code-curator-v1](https://huggingface.co/mdonigian/code-curator-v1) (UniXcoder-base, multi-task)
- **Curation:** Per-language-slice filtering + compression ratio pre-filter + MinHash deduplication

## Filtering Strategy

Different language groups need different curation approaches. Not every slice
goes through the GPU classifier — schema languages and GitHub issues are filtered
with cheaper heuristics because the classifier was trained on general-purpose code
and isn't the right tool for inherently structured formats.

**All slices** share these pre-filters:
- zlib compression ratio < 0.10 (catches extreme repetition)
- MinHash LSH deduplication (128 perms, 5-line shingles, 0.7 Jaccard threshold)

### Classifier-Scored Slices (relevance_filter)

These languages were scored by the multi-task classifier. Files were ranked by
structured data relevance and filtered to keep only those with relevance ≥ 2.0
and quality ≥ 1.5, sampled down to the per-slice token budget:

- **TypeScript**: ~600M tokens — strong type system, filter by SD relevance ≥ 2
- **Python**: ~600M tokens — filter by SD relevance ≥ 2
- **Rust/Go/Java**: ~600M tokens — strongly typed, filter by SD relevance ≥ 2
- **Jupyter notebooks**: ~400M tokens — filter by SD relevance ≥ 2

### Non-Classified Slices

These languages were **not** run through the classifier. Their `quality`,
`structured_data`, and `content_type` columns contain default placeholder values
(0.0 / "unclassified") and should be ignored:

- **Schema languages** (JSON/YAML/SQL/protobuf/thrift/XSLT): ~800M tokens — inherently structured data formats; quality floor + random sample to budget
- **GitHub Issues** (technical): ~500M tokens — keyword filter matching structured-data topics (JSON, schema, API, protobuf, gRPC, etc.)
- **General code** (78 other languages): ~1B tokens — random sample for language diversity; quality floor only

## Language Slice Distribution

{slice_table}

## Classifier-Scored Slices — Detail

The quality and structured data scores below apply **only** to the {classified_docs:,} files
({classified_tokens_b:.1f}B tokens) that went through the classifier. Non-classified slices
are excluded from these statistics.

{classified_slice_table}

### Content Group Distribution (classifier-scored files only)

{group_table}

### Structured Data Relevance (classifier-scored files only)

The strongest classifier signal (Spearman 0.81 on held-out data). SD2+ files
contain significant structured data patterns (API endpoints, JSON parsing,
schema definitions, etc.).

Quality mean: {stats['quality_stats']['mean']:.2f}, Median: {stats['quality_stats']['median']:.2f}.

{sd_table}

### Quality Distribution (classifier-scored files only)

{q_table}

## Non-Classified Slices — Detail

These slices were filtered using heuristics. The classifier columns (`quality`,
`structured_data`, `content_type`) are set to defaults and **do not reflect
actual code quality** — the filtering was done by other means:

{non_classified_slice_table}

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
| `quality` | float | Code quality score 1-5 (**classifier-scored slices only**; 0.0 for non-classified) |
| `structured_data` | float | Structured data relevance 0-3 (**classifier-scored slices only**; 0.0 for non-classified) |
| `content_type` | string | Content type — 9 classes (**classifier-scored slices only**; "unclassified" for non-classified) |
| `language_slice` | string | Language slice name (use this to filter by curation strategy) |
| `relevance_score` | float | Composite relevance score (**classifier-scored slices only**; 0.0 for non-classified) |

> **Tip:** To work with only classifier-scored data, filter on
> `language_slice` in `{{"typescript", "python", "rust_go_java", "jupyter"}}`.

## Methodology

1. **Download:** All language folders from `bigcode/starcoderdata`.
2. **Classification:** Multi-task UniXcoder-base model (3 heads: quality, SD relevance,
   content type) runs on TypeScript, Python, Rust, Go, Java, and Jupyter files.
   Schema languages, GitHub issues, and general code skip this step.
3. **Pre-filtering:** zlib compression ratio filter removes repetitive boilerplate
   before GPU inference.
4. **Filtering:** Per-slice strategy — relevance-based ranking for classified languages,
   keyword matching for GitHub issues, random sampling for schema/general code. All
   slices enforce a quality floor.
5. **Deduplication:** MinHash LSH (128 perms, 5-line shingles, 0.7 Jaccard threshold).
   Highest-relevance file kept from each cluster.
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
