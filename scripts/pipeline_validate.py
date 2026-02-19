#!/usr/bin/env python3
"""
Stage 0: Validation run — calibrate thresholds before the full classification.

Streams 50k documents from FineWeb-Edu, runs both classifiers, and produces
diagnostic plots and a summary JSON to guide TOPIC_THRESHOLD and AMBIGUITY_FLOOR
selection.

Diagnostics produced:
  - Sigmoid score distribution per class (17 histograms)
  - Max-sigmoid distribution across documents
  - Multi-label rate at various thresholds
  - Topic distribution at thresholds 0.3, 0.4, 0.5
  - Complexity score distribution
  - Cross-label correlation matrix (17×17 heatmap)

Usage:
  python pipeline_validate.py --output-dir diagnostics/
  python pipeline_validate.py --num-docs 100000 --batch-size 256
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pipeline_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_COMPLEXITY_MODEL,
    DEFAULT_MAX_LENGTH,
    DEFAULT_TOPIC_MODEL,
    GROUP_DISPLAY_NAMES,
    GROUP_MAP,
    LABEL_DISPLAY_NAMES,
    LABEL_NAMES,
    NUM_LABELS,
)


# ---------------------------------------------------------------------------
# Model loading / AMP (mirrors categorize_fineweb_edu_bert.py patterns)
# ---------------------------------------------------------------------------

def load_models(topic_model_path, complexity_model_path, device, compile_models=False):
    """Load topic and complexity models, return (tokenizer, topic_model, complexity_model)."""
    print(f"Loading topic model: {topic_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(topic_model_path)
    topic_model = AutoModelForSequenceClassification.from_pretrained(topic_model_path)
    topic_model.to(device).eval()

    print(f"Loading complexity model: {complexity_model_path}")
    complexity_model = AutoModelForSequenceClassification.from_pretrained(complexity_model_path)
    complexity_model.to(device).eval()

    if compile_models and hasattr(torch, "compile"):
        cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
        mode = "max-autotune" if cc >= (9, 0) else "reduce-overhead"
        print(f"Compiling models (mode={mode})...")
        topic_model = torch.compile(topic_model, mode=mode)
        complexity_model = torch.compile(complexity_model, mode=mode)

    return tokenizer, topic_model, complexity_model


def get_amp_context(device):
    """Return autocast context for inference."""
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Inference on sample
# ---------------------------------------------------------------------------

def run_inference(dataset_iter, tokenizer, topic_model, complexity_model,
                  device, num_docs, batch_size, max_length):
    """Run both classifiers on streamed documents with length-sorted batching.

    Uses the same dynamic-padding strategy as pipeline_classify.py: tokenize
    a chunk of texts, sort by token length, then pad per-batch to the longest
    in that batch. This eliminates wasted padding compute on short texts —
    significant on A100 where batch sizes are large.

    Returns:
        topic_scores: (N, 17) float32 array of sigmoid scores
        complexity_scores: (N,) float32 array
        token_counts: (N,) int array
    """
    amp_ctx = get_amp_context(device)

    # Collect all texts first (50k docs is small, fits easily in RAM)
    texts = []
    token_counts_list = []
    for example in dataset_iter:
        texts.append(example.get("text", ""))
        token_counts_list.append(example.get("token_count", len(texts[-1].split())))
        if len(texts) >= num_docs:
            break

    n = len(texts)
    print(f"Collected {n} documents, tokenizing with length-sorted batching...")

    # Tokenize all at once (fast tokenizer, Rust-backed)
    encodings = tokenizer(
        texts, max_length=max_length, truncation=True,
        padding=False, return_attention_mask=False, return_length=True,
    )
    ids_list = encodings["input_ids"]
    lengths = encodings["length"]

    # Sort by token length for minimal padding waste
    order = sorted(range(n), key=lambda i: lengths[i])

    all_topic = np.empty((n, NUM_LABELS), dtype=np.float32)
    all_complexity = np.empty(n, dtype=np.float32)

    pbar = tqdm(total=n, desc="Classifying sample", unit="doc")

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_order = order[start : start + batch_size]
            batch_ids = [ids_list[i] for i in batch_order]
            padded = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
            input_ids = padded["input_ids"].to(device)
            attention_mask = padded["attention_mask"].to(device)

            with amp_ctx:
                t_logits = topic_model(input_ids=input_ids, attention_mask=attention_mask).logits.clone()
                c_logits = complexity_model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze(-1)

            t_scores = torch.sigmoid(t_logits).float().cpu().numpy()
            c_scores = c_logits.float().cpu().numpy()

            all_topic[batch_order] = t_scores
            all_complexity[batch_order] = c_scores
            pbar.update(len(batch_order))

    pbar.close()

    token_counts = np.array(token_counts_list, dtype=np.int64)
    print(f"Processed {n} documents")
    return all_topic, all_complexity, token_counts


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def generate_diagnostics(topic_scores, complexity_scores, token_counts, output_dir):
    """Generate all diagnostic plots and summary JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_docs = len(topic_scores)
    print(f"\nGenerating diagnostics for {n_docs} documents...")

    sns.set_theme(style="whitegrid")

    # 1. Per-class sigmoid histograms
    _plot_sigmoid_histograms(topic_scores, output_dir)

    # 2. Max-sigmoid distribution
    _plot_max_sigmoid(topic_scores, output_dir)

    # 3. Multi-label rates at various thresholds
    multi_label_stats = _plot_multi_label_rates(topic_scores, output_dir)

    # 4. Topic distribution at different thresholds
    topic_dist_stats = _compute_topic_distributions(topic_scores, token_counts)
    _plot_topic_distributions(topic_dist_stats, output_dir)

    # 5. Complexity distribution
    _plot_complexity(complexity_scores, output_dir)

    # 6. Cross-label correlation matrix
    _plot_correlation_matrix(topic_scores, output_dir)

    # 7. Summary JSON
    summary = _build_summary(topic_scores, complexity_scores, token_counts,
                             multi_label_stats, topic_dist_stats)
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    _print_recommendations(summary)


def _plot_sigmoid_histograms(topic_scores, output_dir):
    """Per-class sigmoid score distributions (4x5 grid of histograms)."""
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    axes = axes.flatten()
    for i in range(NUM_LABELS):
        ax = axes[i]
        scores = topic_scores[:, i]
        ax.hist(scores, bins=50, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.set_title(LABEL_DISPLAY_NAMES[i], fontsize=8)
        ax.set_xlim(0, 1)
        ax.axvline(0.3, color="orange", linestyle="--", linewidth=0.8, label="0.3")
        ax.axvline(0.5, color="red", linestyle="--", linewidth=0.8, label="0.5")
        ax.tick_params(labelsize=6)
    for i in range(NUM_LABELS, len(axes)):
        axes[i].set_visible(False)
    axes[0].legend(fontsize=6)
    fig.suptitle("Sigmoid Score Distribution per Label", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "sigmoid_histograms.png", dpi=150)
    plt.close(fig)
    print("  Saved sigmoid_histograms.png")


def _plot_max_sigmoid(topic_scores, output_dir):
    """Distribution of each document's maximum sigmoid score."""
    max_scores = topic_scores.max(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(max_scores, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    for thresh in [0.2, 0.3, 0.4, 0.5]:
        pct_above = (max_scores >= thresh).mean() * 100
        ax.axvline(thresh, linestyle="--", linewidth=1,
                   label=f"{thresh} ({pct_above:.1f}% above)")
    ax.set_xlabel("Max Sigmoid Score")
    ax.set_ylabel("Count")
    ax.set_title("Max-Sigmoid Distribution (highest label score per document)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "max_sigmoid_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved max_sigmoid_distribution.png")


def _plot_multi_label_rates(topic_scores, output_dir):
    """Multi-label rate at various thresholds."""
    thresholds = np.arange(0.1, 0.71, 0.05)
    stats = {}
    for thresh in thresholds:
        active = (topic_scores >= thresh).sum(axis=1)
        stats[f"{thresh:.2f}"] = {
            "mean_labels": float(active.mean()),
            "pct_0_labels": float((active == 0).mean() * 100),
            "pct_1_label": float((active == 1).mean() * 100),
            "pct_2plus": float((active >= 2).mean() * 100),
            "pct_3plus": float((active >= 3).mean() * 100),
        }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    t_vals = [float(t) for t in thresholds]
    ax1.plot(t_vals, [stats[f"{t:.2f}"]["mean_labels"] for t in thresholds], "o-")
    ax1.set_xlabel("Threshold")
    ax1.set_ylabel("Mean active labels per doc")
    ax1.set_title("Mean Labels per Document vs Threshold")
    ax1.grid(True)

    ax2.plot(t_vals, [stats[f"{t:.2f}"]["pct_0_labels"] for t in thresholds], "o-", label="0 labels")
    ax2.plot(t_vals, [stats[f"{t:.2f}"]["pct_1_label"] for t in thresholds], "s-", label="1 label")
    ax2.plot(t_vals, [stats[f"{t:.2f}"]["pct_2plus"] for t in thresholds], "^-", label="2+ labels")
    ax2.plot(t_vals, [stats[f"{t:.2f}"]["pct_3plus"] for t in thresholds], "d-", label="3+ labels")
    ax2.set_xlabel("Threshold")
    ax2.set_ylabel("% of documents")
    ax2.set_title("Multi-Label Rate vs Threshold")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    fig.savefig(output_dir / "multi_label_rates.png", dpi=150)
    plt.close(fig)
    print("  Saved multi_label_rates.png")
    return stats


def _compute_topic_distributions(topic_scores, token_counts):
    """Compute topic distribution at various thresholds."""
    results = {}
    for thresh in [0.3, 0.4, 0.5]:
        dist = {}
        for group_name, label_indices in GROUP_MAP.items():
            mask = np.zeros(len(topic_scores), dtype=bool)
            for idx in label_indices:
                mask |= (topic_scores[:, idx] >= thresh)
            dist[group_name] = {
                "doc_count": int(mask.sum()),
                "doc_pct": float(mask.mean() * 100),
                "token_sum": int(token_counts[mask].sum()) if mask.any() else 0,
            }
        results[f"{thresh:.1f}"] = dist
    return results


def _plot_topic_distributions(topic_dist_stats, output_dir):
    """Bar chart comparing topic distributions at different thresholds."""
    groups = list(GROUP_MAP.keys())
    group_labels = [GROUP_DISPLAY_NAMES[g] for g in groups]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(groups))
    width = 0.25
    for i, thresh in enumerate(["0.3", "0.4", "0.5"]):
        pcts = [topic_dist_stats[thresh][g]["doc_pct"] for g in groups]
        ax.bar(x + i * width, pcts, width, label=f"threshold={thresh}")

    ax.set_xticks(x + width)
    ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of documents in group")
    ax.set_title("Topic Group Distribution at Various Thresholds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "topic_distributions.png", dpi=150)
    plt.close(fig)
    print("  Saved topic_distributions.png")


def _plot_complexity(complexity_scores, output_dir):
    """Complexity score histogram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(complexity_scores, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    for boundary in [1.75, 2.5, 3.25]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Complexity Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Complexity Score Distribution (mean={complexity_scores.mean():.2f}, "
                 f"std={complexity_scores.std():.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "complexity_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved complexity_distribution.png")


def _plot_correlation_matrix(topic_scores, output_dir):
    """17×17 cross-label correlation heatmap."""
    corr = np.corrcoef(topic_scores.T)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        xticklabels=LABEL_DISPLAY_NAMES, yticklabels=LABEL_DISPLAY_NAMES,
        ax=ax, annot_kws={"size": 6},
    )
    ax.set_title("Cross-Label Correlation Matrix")
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "label_correlation_matrix.png", dpi=150)
    plt.close(fig)
    print("  Saved label_correlation_matrix.png")


def _build_summary(topic_scores, complexity_scores, token_counts,
                   multi_label_stats, topic_dist_stats):
    """Build summary JSON with all key metrics."""
    max_sigmoid = topic_scores.max(axis=1)

    # Recommend thresholds based on distributions
    # TOPIC_THRESHOLD: choose threshold where multi-label rate is reasonable (~20-40% 2+ labels)
    recommended_topic = 0.3
    for thresh_str, stats in sorted(multi_label_stats.items()):
        if stats["pct_2plus"] < 40 and stats["pct_0_labels"] < 10:
            recommended_topic = float(thresh_str)
            break

    # AMBIGUITY_FLOOR: choose where <5% of docs are below it
    recommended_floor = 0.3
    for floor in [0.2, 0.25, 0.3, 0.35, 0.4]:
        if (max_sigmoid < floor).mean() < 0.05:
            recommended_floor = floor
            break

    return {
        "num_docs": len(topic_scores),
        "max_sigmoid_stats": {
            "mean": float(max_sigmoid.mean()),
            "median": float(np.median(max_sigmoid)),
            "p10": float(np.percentile(max_sigmoid, 10)),
            "p90": float(np.percentile(max_sigmoid, 90)),
            "pct_below_0.3": float((max_sigmoid < 0.3).mean() * 100),
            "pct_below_0.4": float((max_sigmoid < 0.4).mean() * 100),
            "pct_below_0.5": float((max_sigmoid < 0.5).mean() * 100),
        },
        "complexity_stats": {
            "mean": float(complexity_scores.mean()),
            "std": float(complexity_scores.std()),
            "min": float(complexity_scores.min()),
            "max": float(complexity_scores.max()),
            "median": float(np.median(complexity_scores)),
            "pct_L1": float(((complexity_scores >= 1.0) & (complexity_scores < 1.75)).mean() * 100),
            "pct_L2": float(((complexity_scores >= 1.75) & (complexity_scores < 2.5)).mean() * 100),
            "pct_L3": float(((complexity_scores >= 2.5) & (complexity_scores < 3.25)).mean() * 100),
            "pct_L4": float((complexity_scores >= 3.25).mean() * 100),
        },
        "token_count_stats": {
            "mean": float(token_counts.mean()),
            "median": float(np.median(token_counts)),
            "p10": float(np.percentile(token_counts, 10)),
            "p90": float(np.percentile(token_counts, 90)),
        },
        "multi_label_rates": multi_label_stats,
        "topic_distributions": topic_dist_stats,
        "recommendations": {
            "TOPIC_THRESHOLD": recommended_topic,
            "AMBIGUITY_FLOOR": recommended_floor,
            "note": "Review the diagnostic plots and adjust these values based on the "
                    "sigmoid histograms and multi-label rate curves.",
        },
    }


def _print_recommendations(summary):
    """Print human-readable recommendations."""
    rec = summary["recommendations"]
    ms = summary["max_sigmoid_stats"]
    cs = summary["complexity_stats"]

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"\nMax-sigmoid stats:")
    print(f"  Mean: {ms['mean']:.3f}, Median: {ms['median']:.3f}")
    print(f"  P10: {ms['p10']:.3f}, P90: {ms['p90']:.3f}")
    print(f"  Below 0.3: {ms['pct_below_0.3']:.1f}%, Below 0.5: {ms['pct_below_0.5']:.1f}%")
    print(f"\nComplexity stats:")
    print(f"  Mean: {cs['mean']:.2f}, Std: {cs['std']:.2f}")
    print(f"  Range: [{cs['min']:.2f}, {cs['max']:.2f}]")
    print(f"  L1: {cs['pct_L1']:.1f}%, L2: {cs['pct_L2']:.1f}%, "
          f"L3: {cs['pct_L3']:.1f}%, L4: {cs['pct_L4']:.1f}%")
    print(f"\nRecommended thresholds:")
    print(f"  TOPIC_THRESHOLD = {rec['TOPIC_THRESHOLD']}")
    print(f"  AMBIGUITY_FLOOR = {rec['AMBIGUITY_FLOOR']}")
    print(f"\n  {rec['note']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Validation run to calibrate classification thresholds",
    )
    parser.add_argument("--output-dir", default="diagnostics",
                        help="Directory for diagnostic output (default: diagnostics)")
    parser.add_argument("--num-docs", type=int, default=50_000,
                        help="Number of documents to sample (default: 50000)")
    parser.add_argument("--topic-model", default=DEFAULT_TOPIC_MODEL)
    parser.add_argument("--complexity-model", default=DEFAULT_COMPLEXITY_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for models")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu",
                        help="HuggingFace dataset name")
    parser.add_argument("--config", default="sample-100BT",
                        help="Dataset config/subset name")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tokenizer, topic_model, complexity_model = load_models(
        args.topic_model, args.complexity_model, device, args.compile,
    )

    print(f"\nStreaming {args.num_docs} docs from {args.dataset} ({args.config})...")
    t0 = time.time()
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)
    ds_iter = iter(ds)

    topic_scores, complexity_scores, token_counts = run_inference(
        ds_iter, tokenizer, topic_model, complexity_model,
        device, args.num_docs, args.batch_size, args.max_length,
    )
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s ({len(topic_scores)/elapsed:.0f} docs/sec)")

    generate_diagnostics(topic_scores, complexity_scores, token_counts, args.output_dir)


if __name__ == "__main__":
    main()
