#!/usr/bin/env python3
"""
Generate a Topic × Complexity heatmap from classified FineWeb-Edu parquet files.

Reads parquet files with multi-label boolean topic columns and a continuous
reasoning_complexity score, then produces:
  1. A heatmap PNG (topics × complexity levels)
  2. Summary statistics printed to console
  3. A CSV export of the raw count matrix

Usage:
  python scripts/heatmap_topic_complexity.py \
      output/fineweb-edu/fineweb-edu-10BT-categorized/ \
      -o output/fineweb-edu/heatmap_topic_complexity.png
"""

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

TOPIC_COLUMNS = [
    "arts_humanities",
    "life_sciences_biology",
    "business_economics",
    "physical_sciences",
    "computer_science_software_engineering",
    "education_pedagogy",
    "engineering_technology",
    "environmental_science_energy",
    "medicine_health",
    "history_geography",
    "law_government",
    "mathematics_statistics",
    "philosophy_ethics",
    "social_sciences",
    "language_writing",
    "machine_learning_ai",
    "personal_finance_practical_life",
]

TOPIC_LABELS = {
    "arts_humanities": "Arts & Humanities",
    "life_sciences_biology": "Biology & Life Sciences",
    "business_economics": "Business & Economics",
    "physical_sciences": "Physical Sciences",
    "computer_science_software_engineering": "Computer Science & Software Eng.",
    "education_pedagogy": "Education & Pedagogy",
    "engineering_technology": "Engineering & Technology",
    "environmental_science_energy": "Environmental Science & Energy",
    "medicine_health": "Health & Medicine",
    "history_geography": "History & Geography",
    "law_government": "Law & Government",
    "mathematics_statistics": "Mathematics & Statistics",
    "philosophy_ethics": "Philosophy & Ethics",
    "social_sciences": "Social Sciences",
    "language_writing": "Language & Writing",
    "machine_learning_ai": "Machine Learning & AI",
    "personal_finance_practical_life": "Personal Finance & Practical Life",
}

COMPLEXITY_LABELS = {
    1: "1 – Factual/Declarative",
    2: "2 – Single-step Reasoning",
    3: "3 – Multi-step Reasoning",
    4: "4 – Complex Reasoning",
}


def load_data(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if p.is_file():
        files = [str(p)]
    elif p.is_dir():
        files = sorted(glob.glob(str(p / "*.parquet")))
    else:
        files = sorted(glob.glob(input_path))

    if not files:
        print(f"ERROR: No parquet files found at {input_path}", file=sys.stderr)
        sys.exit(1)

    keep_cols = list(TOPIC_COLUMNS) + ["reasoning_complexity", "token_count"]
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=keep_cols)
        frames.append(df)
        print(f"  Loaded {f} ({len(df):,} rows)")

    return pd.concat(frames, ignore_index=True)


def bin_complexity(series: pd.Series) -> pd.Series:
    return series.round().clip(1, 4).astype(int)


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build a topic × complexity count matrix (multi-label: each doc counted per topic)."""
    complexity = bin_complexity(df["reasoning_complexity"])
    records = []
    for col in TOPIC_COLUMNS:
        if col not in df.columns:
            continue
        mask = df[col]
        for level in range(1, 5):
            count = int((mask & (complexity == level)).sum())
            records.append({"topic": col, "complexity": level, "count": count})

    pivot = (
        pd.DataFrame(records)
        .pivot(index="topic", columns="complexity", values="count")
        .fillna(0)
        .astype(int)
    )
    pivot.index = pivot.index.map(TOPIC_LABELS)
    pivot.columns = [COMPLEXITY_LABELS[c] for c in pivot.columns]
    return pivot


def build_token_matrix(df: pd.DataFrame) -> pd.DataFrame | None:
    if "token_count" not in df.columns or df["token_count"].isna().all():
        return None

    complexity = bin_complexity(df["reasoning_complexity"])
    records = []
    for col in TOPIC_COLUMNS:
        if col not in df.columns:
            continue
        mask = df[col]
        for level in range(1, 5):
            sel = mask & (complexity == level)
            tokens = int(df.loc[sel, "token_count"].sum())
            records.append({"topic": col, "complexity": level, "tokens": tokens})

    pivot = (
        pd.DataFrame(records)
        .pivot(index="topic", columns="complexity", values="tokens")
        .fillna(0)
        .astype(int)
    )
    pivot.index = pivot.index.map(TOPIC_LABELS)
    pivot.columns = [COMPLEXITY_LABELS[c] for c in pivot.columns]
    return pivot


def print_summary(count_matrix: pd.DataFrame, token_matrix: pd.DataFrame | None, total_docs: int):
    total_assignments = count_matrix.values.sum()
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total documents:            {total_docs:>12,}")
    print(f"Total topic assignments:    {total_assignments:>12,}  (multi-label, so > total docs)")
    avg_labels = total_assignments / total_docs if total_docs else 0
    print(f"Avg labels per document:    {avg_labels:>12.2f}")

    print(f"\n--- Per-Topic Totals (sorted descending) ---")
    topic_totals = count_matrix.sum(axis=1).sort_values(ascending=False)
    for topic, cnt in topic_totals.items():
        pct = 100.0 * cnt / total_docs
        print(f"  {topic:<42s} {cnt:>10,}  ({pct:5.1f}% of docs)")

    print(f"\n--- Per-Complexity-Level Totals ---")
    level_totals = count_matrix.sum(axis=0)
    for level, cnt in level_totals.items():
        pct = 100.0 * cnt / total_assignments
        print(f"  {level:<35s} {cnt:>10,}  ({pct:5.1f}% of assignments)")

    print(f"\n--- Top 10 Cells by Count ---")
    flat = []
    for topic in count_matrix.index:
        for level in count_matrix.columns:
            flat.append((topic, level, count_matrix.loc[topic, level]))
    flat.sort(key=lambda x: x[2], reverse=True)
    for topic, level, cnt in flat[:10]:
        pct = 100.0 * cnt / total_assignments
        print(f"  {topic:<42s} × {level:<30s} {cnt:>9,}  ({pct:5.2f}%)")

    print(f"\n--- Bottom 5 Cells by Count ---")
    for topic, level, cnt in flat[-5:]:
        pct = 100.0 * cnt / total_assignments
        print(f"  {topic:<42s} × {level:<30s} {cnt:>9,}  ({pct:5.2f}%)")

    if token_matrix is not None:
        print(f"\n--- Token Counts per Topic (sorted descending) ---")
        token_totals = token_matrix.sum(axis=1).sort_values(ascending=False)
        grand_tokens = token_totals.sum()
        print(f"  Total tokens (across all labels):  {grand_tokens:>15,}")
        for topic, toks in token_totals.items():
            pct = 100.0 * toks / grand_tokens
            print(f"  {topic:<42s} {toks:>15,}  ({pct:5.1f}%)")

        print(f"\n--- Token Counts per Complexity Level ---")
        level_tokens = token_matrix.sum(axis=0)
        for level, toks in level_tokens.items():
            pct = 100.0 * toks / grand_tokens
            print(f"  {level:<35s} {toks:>15,}  ({pct:5.1f}%)")


def plot_heatmap(count_matrix: pd.DataFrame, total_docs: int, output_path: str):
    pct_matrix = 100.0 * count_matrix / total_docs

    topic_order = count_matrix.sum(axis=1).sort_values(ascending=True).index
    count_sorted = count_matrix.loc[topic_order]
    pct_sorted = pct_matrix.loc[topic_order]

    annot = count_sorted.copy().astype(str)
    for topic in annot.index:
        for level in annot.columns:
            c = count_sorted.loc[topic, level]
            p = pct_sorted.loc[topic, level]
            if c >= 1_000_000:
                annot.loc[topic, level] = f"{c / 1e6:.1f}M\n({p:.1f}%)"
            elif c >= 1_000:
                annot.loc[topic, level] = f"{c / 1e3:.1f}K\n({p:.1f}%)"
            else:
                annot.loc[topic, level] = f"{c:,}\n({p:.1f}%)"

    fig, ax = plt.subplots(figsize=(16, 11))
    sns.heatmap(
        pct_sorted,
        annot=annot,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "% of total documents", "shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        "FineWeb-Edu 10BT Sample: Topic × Complexity Distribution",
        fontsize=16,
        fontweight="bold",
        pad=16,
    )
    ax.set_xlabel("Complexity Level", fontsize=13, labelpad=10)
    ax.set_ylabel("Topic Category", fontsize=13, labelpad=10)
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Topic × Complexity heatmap from classified FineWeb-Edu data")
    parser.add_argument("input", help="Parquet file or directory of parquet files")
    parser.add_argument("-o", "--output", default="output/fineweb-edu/heatmap_topic_complexity.png",
                        help="Output PNG path (default: output/fineweb-edu/heatmap_topic_complexity.png)")
    parser.add_argument("--csv", default=None,
                        help="Output CSV path (default: same as PNG but .csv)")
    args = parser.parse_args()

    csv_path = args.csv or str(Path(args.output).with_suffix(".csv"))
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data(args.input)
    total_docs = len(df)
    print(f"Total documents: {total_docs:,}")

    print("\nBuilding count matrix...")
    count_matrix = build_matrix(df)

    print("Building token matrix...")
    token_matrix = build_token_matrix(df)

    print_summary(count_matrix, token_matrix, total_docs)

    print("\nGenerating heatmap...")
    plot_heatmap(count_matrix, total_docs, args.output)

    count_matrix.to_csv(csv_path)
    print(f"Count matrix CSV saved to {csv_path}")

    if token_matrix is not None:
        token_csv = str(Path(csv_path).with_stem(Path(csv_path).stem + "_tokens"))
        token_matrix.to_csv(token_csv)
        print(f"Token matrix CSV saved to {token_csv}")


if __name__ == "__main__":
    main()
