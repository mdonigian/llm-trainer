#!/usr/bin/env python3
"""
Stage 0: Validation run — calibrate thresholds before the full classification.

Reads 50k code files from local parquet files (preferred) or streams from
HuggingFace, runs the multi-task classifier, and produces diagnostic plots
and a summary JSON.

Diagnostics produced:
  - Quality score distribution (histogram)
  - Structured data relevance distribution (histogram)
  - Content type distribution (bar chart)
  - Quality × structured data heatmap
  - Per-language quality/relevance breakdown
  - Compression ratio distribution

Usage (local — preferred):
  python pipeline_validate.py --local-dir /workspace/starcoder-curation/raw_data
  python pipeline_validate.py --local-dir /workspace/starcoder-curation/raw_data --num-docs 100000

Usage (streaming fallback):
  python pipeline_validate.py --output-dir diagnostics/
"""

import argparse
import contextlib
import json
import time
import zlib
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from pipeline_classify import CodeClassifierModel, load_classifier
from pipeline_config import (
    COMPRESSION_RATIO_FLOOR,
    CONTENT_TYPES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLASSIFIER_MODEL,
    DEFAULT_DATASET,
    DEFAULT_MAX_LENGTH,
    NUM_CONTENT_TYPES,
    QUALITY_NAMES,
    RECOMMENDED_LANGUAGES,
    STRUCTURED_DATA_NAMES,
)


# ---------------------------------------------------------------------------
# Local parquet reading
# ---------------------------------------------------------------------------

def iter_local_parquets(local_dir):
    files = sorted(Path(local_dir).rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {local_dir}")

    print(f"Found {len(files)} local parquet files in {local_dir}")

    for f in files:
        schema = pq.read_schema(f)
        available = schema.names
        read_cols = ["content"]
        lang_col = None
        if "lang" in available:
            read_cols.append("lang")
            lang_col = "lang"
        elif "language" in available:
            read_cols.append("language")
            lang_col = "language"
        if "size" in available:
            read_cols.append("size")

        table = pq.read_table(f, columns=read_cols)
        if table.num_rows == 0:
            continue
        for rb in table.to_batches(max_chunksize=16384):
            texts = rb.column("content").to_pylist()
            langs = rb.column(lang_col).to_pylist() if lang_col and lang_col in rb.schema.names else ["unknown"] * len(texts)
            sizes = rb.column("size").fill_null(0).to_numpy(zero_copy_only=False) if "size" in rb.schema.names else np.array([len(t) for t in texts])
            for i in range(len(texts)):
                yield {
                    "text": texts[i],
                    "token_count": int(sizes[i]) // 4,
                    "lang": langs[i] if langs[i] else "unknown",
                }
        del table


# ---------------------------------------------------------------------------
# AMP
# ---------------------------------------------------------------------------

def get_amp_context(device):
    if device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.amp.autocast("cuda", dtype=dtype)
    return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_PAD_BUCKETS = (64, 128, 192, 256, 384, 512)


def _bucket_pad_length(seq_len: int, max_length: int) -> int:
    for bucket in _PAD_BUCKETS:
        if seq_len <= bucket <= max_length:
            return bucket
    return max_length


def run_inference(dataset_iter, tokenizer, model, device, num_docs, batch_size,
                  max_length, chunk_size=25000):
    amp_ctx = get_amp_context(device)

    def _run_chunk(chunk_texts):
        encodings = tokenizer(
            chunk_texts, max_length=max_length, truncation=True,
            padding=False, return_attention_mask=False, return_length=True,
        )
        ids_list = encodings["input_ids"]
        lengths = encodings["length"]
        order = sorted(range(len(ids_list)), key=lambda i: lengths[i])

        chunk_quality = np.empty(len(chunk_texts), dtype=np.float32)
        chunk_sd = np.empty(len(chunk_texts), dtype=np.float32)
        chunk_ct = np.empty(len(chunk_texts), dtype=np.int32)

        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                batch_order = order[start : start + batch_size]
                batch_ids = [ids_list[i] for i in batch_order]
                longest = max(len(batch_ids[j]) for j in range(len(batch_ids)))
                pad_to = _bucket_pad_length(longest, max_length)
                padded = tokenizer.pad(
                    {"input_ids": batch_ids}, padding="max_length",
                    max_length=pad_to, return_tensors="pt",
                )
                input_ids = padded["input_ids"].to(device, non_blocking=True)
                attention_mask = padded["attention_mask"].to(device, non_blocking=True)
                with amp_ctx:
                    q_pred, sd_pred, ct_logits = model(
                        input_ids=input_ids, attention_mask=attention_mask,
                    )
                chunk_quality[batch_order] = q_pred.float().cpu().numpy()
                chunk_sd[batch_order] = sd_pred.float().cpu().numpy()
                chunk_ct[batch_order] = ct_logits.float().cpu().numpy().argmax(axis=1)

        return chunk_quality, chunk_sd, chunk_ct

    quality_chunks, sd_chunks, ct_chunks = [], [], []
    token_count_chunks, lang_list = [], []
    compression_ratio_chunks = []
    chunk_texts, chunk_token_counts, chunk_langs = [], [], []
    processed = 0
    pbar = tqdm(total=num_docs, desc="Classifying sample", unit="doc")

    for example in dataset_iter:
        text = example.get("text", "") or example.get("content", "")
        if not text:
            continue
        chunk_texts.append(text)
        chunk_token_counts.append(example.get("token_count", len(text.split())))
        chunk_langs.append(example.get("lang", "unknown"))
        if len(chunk_texts) >= chunk_size or processed + len(chunk_texts) >= num_docs:
            # Compression ratios
            cr = np.array([len(zlib.compress(t.encode("utf-8", errors="replace"))) /
                           max(len(t.encode("utf-8", errors="replace")), 1) for t in chunk_texts],
                          dtype=np.float32)
            compression_ratio_chunks.append(cr)

            c_q, c_sd, c_ct = _run_chunk(chunk_texts)
            quality_chunks.append(c_q)
            sd_chunks.append(c_sd)
            ct_chunks.append(c_ct)
            token_count_chunks.append(np.array(chunk_token_counts, dtype=np.int64))
            lang_list.extend(chunk_langs)
            processed += len(chunk_texts)
            pbar.update(len(chunk_texts))
            chunk_texts, chunk_token_counts, chunk_langs = [], [], []
            if processed >= num_docs:
                break

    if chunk_texts and processed < num_docs:
        cr = np.array([len(zlib.compress(t.encode("utf-8", errors="replace"))) /
                       max(len(t.encode("utf-8", errors="replace")), 1) for t in chunk_texts],
                      dtype=np.float32)
        compression_ratio_chunks.append(cr)
        c_q, c_sd, c_ct = _run_chunk(chunk_texts)
        quality_chunks.append(c_q)
        sd_chunks.append(c_sd)
        ct_chunks.append(c_ct)
        token_count_chunks.append(np.array(chunk_token_counts, dtype=np.int64))
        lang_list.extend(chunk_langs)
        processed += len(chunk_texts)
        pbar.update(len(chunk_texts))

    pbar.close()

    if not quality_chunks:
        return (np.empty(0, np.float32), np.empty(0, np.float32),
                np.empty(0, np.int32), np.empty(0, np.int64),
                np.empty(0, np.float32), [])

    return (
        np.concatenate(quality_chunks),
        np.concatenate(sd_chunks),
        np.concatenate(ct_chunks),
        np.concatenate(token_count_chunks),
        np.concatenate(compression_ratio_chunks),
        lang_list,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def generate_diagnostics(quality, sd, ct, token_counts, comp_ratios, langs, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(quality)
    print(f"\nGenerating diagnostics for {n} code files...")
    sns.set_theme(style="whitegrid")

    _plot_quality_distribution(quality, output_dir)
    _plot_sd_distribution(sd, output_dir)
    _plot_content_types(ct, output_dir)
    _plot_quality_sd_heatmap(quality, sd, output_dir)
    _plot_compression_ratio(comp_ratios, output_dir)
    _plot_language_distribution(langs, output_dir)

    summary = _build_summary(quality, sd, ct, token_counts, comp_ratios, langs)
    summary_path = output_dir / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")
    _print_summary(summary)


def _plot_quality_distribution(quality, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(quality, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    for boundary in [1.5, 2.5, 3.5, 4.5]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Count")
    ax.set_title(f"Code Quality Distribution (mean={quality.mean():.2f}, std={quality.std():.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved quality_distribution.png")


def _plot_sd_distribution(sd, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sd, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    for boundary in [0.5, 1.5, 2.5]:
        ax.axvline(boundary, color="red", linestyle="--", linewidth=1)
    ax.set_xlabel("Structured Data Relevance")
    ax.set_ylabel("Count")
    ax.set_title(f"Structured Data Relevance Distribution (mean={sd.mean():.2f})")
    fig.tight_layout()
    fig.savefig(output_dir / "structured_data_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved structured_data_distribution.png")


def _plot_content_types(ct, output_dir):
    counts = Counter(ct)
    labels = [CONTENT_TYPES[i] for i in range(NUM_CONTENT_TYPES)]
    values = [counts.get(i, 0) for i in range(NUM_CONTENT_TYPES)]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(labels, values)
    ax.set_ylabel("Count")
    ax.set_title("Content Type Distribution")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output_dir / "content_type_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved content_type_distribution.png")


def _plot_quality_sd_heatmap(quality, sd, output_dir):
    q_rounded = np.clip(np.round(quality), 1, 5).astype(int)
    sd_rounded = np.clip(np.round(sd), 0, 3).astype(int)

    matrix = np.zeros((5, 4), dtype=int)
    for q, s in zip(q_rounded, sd_rounded):
        matrix[q - 1, s] += 1

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=[STRUCTURED_DATA_NAMES[i] for i in range(4)],
        yticklabels=[f"Q{i}" for i in range(1, 6)],
        ax=ax,
    )
    ax.set_xlabel("Structured Data Relevance")
    ax.set_ylabel("Quality Level")
    ax.set_title("Quality × Structured Data Relevance")
    fig.tight_layout()
    fig.savefig(output_dir / "quality_sd_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved quality_sd_heatmap.png")


def _plot_compression_ratio(comp_ratios, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(comp_ratios, bins=100, alpha=0.7, edgecolor="black", linewidth=0.3)
    ax.axvline(COMPRESSION_RATIO_FLOOR, color="red", linestyle="--", linewidth=1.5,
               label=f"Floor={COMPRESSION_RATIO_FLOOR} ({(comp_ratios < COMPRESSION_RATIO_FLOOR).mean()*100:.1f}% dropped)")
    ax.set_xlabel("Compression Ratio (zlib)")
    ax.set_ylabel("Count")
    ax.set_title("Compression Ratio Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "compression_ratio_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved compression_ratio_distribution.png")


def _plot_language_distribution(langs, output_dir):
    lang_counts = Counter(langs)
    top = lang_counts.most_common(25)
    if not top:
        return
    labels, counts = zip(*top)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(labels)), counts, align="center")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("File count")
    ax.set_title("Programming Language Distribution (Top 25)")
    fig.tight_layout()
    fig.savefig(output_dir / "language_distribution.png", dpi=150)
    plt.close(fig)
    print("  Saved language_distribution.png")


def _build_summary(quality, sd, ct, token_counts, comp_ratios, langs):
    q_rounded = np.clip(np.round(quality), 1, 5).astype(int)
    sd_rounded = np.clip(np.round(sd), 0, 3).astype(int)

    return {
        "num_docs": len(quality),
        "quality_stats": {
            "mean": float(quality.mean()),
            "std": float(quality.std()),
            "median": float(np.median(quality)),
            "per_level": {str(i): int((q_rounded == i).sum()) for i in range(1, 6)},
            "pct_below_floor": float((quality <= 1.5).mean() * 100),
        },
        "structured_data_stats": {
            "mean": float(sd.mean()),
            "std": float(sd.std()),
            "median": float(np.median(sd)),
            "per_level": {str(i): int((sd_rounded == i).sum()) for i in range(0, 4)},
            "pct_sd2_plus": float((sd >= 1.5).mean() * 100),
            "pct_sd3": float((sd >= 2.5).mean() * 100),
        },
        "content_type_distribution": {
            CONTENT_TYPES[i]: int((ct == i).sum()) for i in range(NUM_CONTENT_TYPES)
        },
        "compression_ratio": {
            "mean": float(comp_ratios.mean()),
            "pct_below_threshold": float((comp_ratios < COMPRESSION_RATIO_FLOOR).mean() * 100),
            "threshold": COMPRESSION_RATIO_FLOOR,
        },
        "token_count_stats": {
            "mean": float(token_counts.mean()),
            "median": float(np.median(token_counts)),
            "p10": float(np.percentile(token_counts, 10)),
            "p90": float(np.percentile(token_counts, 90)),
        },
        "language_distribution": dict(Counter(langs).most_common(30)),
    }


def _print_summary(summary):
    qs = summary["quality_stats"]
    ss = summary["structured_data_stats"]
    cr = summary["compression_ratio"]

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY (StarCoder)")
    print("=" * 60)
    print(f"\nQuality: mean={qs['mean']:.2f}, std={qs['std']:.2f}")
    for level in range(1, 6):
        count = qs["per_level"][str(level)]
        print(f"  Q{level} ({QUALITY_NAMES[level]}): {count:,}")
    print(f"  Would drop (quality <= 1.5): {qs['pct_below_floor']:.1f}%")
    print(f"\nStructured Data Relevance: mean={ss['mean']:.2f}")
    for level in range(0, 4):
        count = ss["per_level"][str(level)]
        print(f"  SD{level} ({STRUCTURED_DATA_NAMES[level]}): {count:,}")
    print(f"  SD>=2 (Significant+): {ss['pct_sd2_plus']:.1f}%")
    print(f"  SD>=3 (Primary focus): {ss['pct_sd3']:.1f}%")
    print(f"\nCompression ratio: mean={cr['mean']:.3f}")
    print(f"  Would drop (<{cr['threshold']}): {cr['pct_below_threshold']:.1f}%")
    print(f"\nTop languages:")
    for lang, count in list(summary["language_distribution"].items())[:10]:
        print(f"  {lang}: {count:,}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 0: Validation run to calibrate thresholds (StarCoder)",
    )
    parser.add_argument("--output-dir", default="diagnostics")
    parser.add_argument("--num-docs", type=int, default=50_000)
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_MODEL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--chunk-size", type=int, default=25_000)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--languages", nargs="*", default=RECOMMENDED_LANGUAGES)
    parser.add_argument("--local-dir")
    args = parser.parse_args()

    import logging as _logging
    for _noisy in ("httpx", "urllib3", "huggingface_hub", "datasets"):
        _logging.getLogger(_noisy).setLevel(_logging.WARNING)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        cc = torch.cuda.get_device_capability(device)
        vram_gb = torch.cuda.get_device_properties(device).total_mem / 1e9
        print(f"GPU: {gpu_name} (cc {cc[0]}.{cc[1]}, {vram_gb:.1f} GB)")
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    tokenizer, model = load_classifier(args.classifier_model, device, args.compile)

    t0 = time.time()
    if args.local_dir:
        print(f"\nReading {args.num_docs} docs from local parquets: {args.local_dir}")
        ds_iter = iter_local_parquets(args.local_dir)
    else:
        lang = args.languages[0] if args.languages else "python"
        print(f"\nStreaming {args.num_docs} docs from {args.dataset} ({lang})...")
        ds = load_dataset(args.dataset, lang, split="train", streaming=True)
        ds_iter = ({"text": ex.get("content", ""), "token_count": len(ex.get("content", "").split()),
                     "lang": lang} for ex in ds)

    quality, sd, ct, token_counts, comp_ratios, langs = run_inference(
        ds_iter, tokenizer, model, device,
        args.num_docs, args.batch_size, args.max_length, args.chunk_size,
    )
    elapsed = time.time() - t0
    print(f"Inference completed in {elapsed:.1f}s ({len(quality)/elapsed:.0f} docs/sec)")

    generate_diagnostics(quality, sd, ct, token_counts, comp_ratios, langs, args.output_dir)


if __name__ == "__main__":
    main()
