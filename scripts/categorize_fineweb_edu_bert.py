#!/usr/bin/env python3
"""
Categorize FineWeb-Edu data using trained BERT (ModernBERT) classifiers.

This is the inference counterpart to train_fineweb_edu.py — it takes raw
FineWeb-Edu files (without category labels) and classifies them using:
  1. A fine-tuned multi-label category classifier (17 topic categories)
  2. A fine-tuned regression complexity classifier (reasoning_complexity score)

Features:
  - Processes single files or entire directories of parquet/csv/json/jsonl
  - Runs both classifiers in a single pass (shared tokenization)
  - AMP (bf16/fp16) for fast GPU inference
  - Dynamic padding (pads to longest-in-batch, not max_length)
  - Periodic checkpoint saves (--save-every) for crash resilience
  - Skips already-categorized rows when resuming from a partial output
  - Configurable sigmoid threshold (--threshold) for category classifier

Usage:
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet
  python categorize_fineweb_edu_bert.py data/raw/ -o data/categorized/
  python categorize_fineweb_edu_bert.py input.parquet --model ./my_model --batch-size 128 --threshold 0.4
  python categorize_fineweb_edu_bert.py input.parquet --no-complexity  # skip complexity classifier

Resume from a partial run:
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet
  (automatically detects existing output and skips already-processed rows)
"""

import argparse
import contextlib
import json
import sys
from queue import Queue
from pathlib import Path
from threading import Thread
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "models/fineweb-edu-classifier"
DEFAULT_COMPLEXITY_MODEL = "models/complexity-classifier"
DEFAULT_BATCH_SIZE = 192
DEFAULT_MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_SAVE_EVERY = 20_000
DEFAULT_CHUNK_SIZE = 25_000  # tuned for ~64GB RAM with tokenizer Python object overhead
DEFAULT_PREFETCH_CHUNKS = 1

FALLBACK_CATEGORY_FIELDS = [
    "mathematics_statistics",
    "computer_science_software_engineering",
    "machine_learning_ai",
    "physical_sciences",
    "life_sciences_biology",
    "medicine_health",
    "engineering_technology",
    "business_economics",
    "law_government",
    "social_sciences",
    "history_geography",
    "philosophy_ethics",
    "education_pedagogy",
    "language_writing",
    "arts_humanities",
    "environmental_science_energy",
    "personal_finance_practical_life",
]

# ---------------------------------------------------------------------------
# Tokenization + batch preparation (no DataLoader — all data stays in-process)
# ---------------------------------------------------------------------------


def prepare_chunk_batches(
    indices: List[int],
    texts: List[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    pin_memory: bool = False,
) -> List[tuple]:
    """Tokenize a chunk of texts, sort by length, and return ready-to-go
    padded batch tensors with pinned memory for async GPU transfer.

    Returns a list of (batch_indices, input_ids, attention_mask) tuples.
    Each input_ids/attention_mask is a [B, seq_len] tensor already pinned
    so .to(device, non_blocking=True) is truly asynchronous.

    This replaces the DataLoader entirely — no worker processes, no IPC
    serialization, no collate_fn in the main process. The data is already
    in memory; all we need is padding and a tight iteration loop.
    """
    # Batch-tokenize (fast tokenizer, Rust-backed)
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_length=True,
    )

    ids_list = encodings["input_ids"]
    lengths = encodings["length"]

    # Sort by token length so similar-length sequences land in the same
    # batch, reducing padding waste and making batch compute times uniform
    order = sorted(range(len(ids_list)), key=lambda i: lengths[i])

    batches = []
    for batch_start in range(0, len(order), batch_size):
        batch_order = order[batch_start : batch_start + batch_size]
        batch_indices = [indices[i] for i in batch_order]
        batch_ids = [ids_list[i] for i in batch_order]
        padded = tokenizer.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt")
        input_ids = padded["input_ids"].to(torch.long)
        attention_mask = padded["attention_mask"].to(torch.long)

        if pin_memory:
            input_ids = input_ids.pin_memory()
            attention_mask = attention_mask.pin_memory()

        batches.append((batch_indices, input_ids, attention_mask))

    return batches


def iter_prepared_chunks(
    remaining_indices: List[int],
    texts: List[str],
    tokenizer,
    max_length: int,
    batch_size: int,
    chunk_size: int,
    pin_memory: bool,
    prefetch_chunks: int,
):
    """Yield pre-tokenized chunk batches produced in a background thread.

    This overlaps CPU tokenization/batch-building for chunk N+1 while GPU
    inference is processing chunk N, which removes chunk-boundary GPU idle gaps.
    """
    num_chunks = (len(remaining_indices) + chunk_size - 1) // chunk_size
    prefetch_chunks = max(0, prefetch_chunks)

    # Strict low-memory mode: no background producer, one chunk at a time.
    # This avoids staging extra chunks in RAM.
    if prefetch_chunks == 0:
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, len(remaining_indices))
            chunk_indices = remaining_indices[chunk_start:chunk_end]
            chunk_texts = texts[chunk_start:chunk_end]
            batches = prepare_chunk_batches(
                chunk_indices,
                chunk_texts,
                tokenizer,
                max_length,
                batch_size,
                pin_memory=pin_memory,
            )
            yield (chunk_idx, num_chunks, batches)
        return

    queue = Queue(maxsize=max(1, prefetch_chunks))
    sentinel = object()

    def _producer():
        try:
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, len(remaining_indices))
                chunk_indices = remaining_indices[chunk_start:chunk_end]
                chunk_texts = texts[chunk_start:chunk_end]
                batches = prepare_chunk_batches(
                    chunk_indices,
                    chunk_texts,
                    tokenizer,
                    max_length,
                    batch_size,
                    pin_memory=pin_memory,
                )
                queue.put((chunk_idx, num_chunks, batches))
        except Exception as e:  # pragma: no cover
            queue.put(e)
        finally:
            queue.put(sentinel)

    producer = Thread(target=_producer, daemon=True)
    producer.start()

    while True:
        item = queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------


def load_file(filepath: Path) -> pd.DataFrame:
    """Load a single data file (parquet, csv, json, jsonl)."""
    suffix = filepath.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(filepath)
    elif suffix == ".csv":
        return pd.read_csv(filepath)
    elif suffix == ".json":
        return pd.read_json(filepath)
    elif suffix == ".jsonl":
        return pd.read_json(filepath, lines=True)
    else:
        sys.exit(f"Unsupported file format: {suffix}")


def discover_files(input_path: str) -> List[Path]:
    """Return a list of data files from a path (file or directory)."""
    p = Path(input_path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        exts = ("*.parquet", "*.csv", "*.json", "*.jsonl")
        files = []
        for ext in exts:
            files.extend(sorted(p.glob(ext)))
        if not files:
            sys.exit(f"No data files found in directory: {p}")
        return files
    else:
        sys.exit(f"Input path does not exist: {p}")


def load_label_config(model_path: str) -> dict:
    """Load label_config.json from the model directory."""
    config_path = Path(model_path) / "label_config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    # Fallback: use the standard 17 categories (self-contained, no imports)
    print(f"  Warning: {config_path} not found, using default category fields")
    label_names = [f.replace("_", " ").title() for f in FALLBACK_CATEGORY_FIELDS]
    return {
        "category_fields": FALLBACK_CATEGORY_FIELDS,
        "label_display_names": label_names,
        "num_labels": len(FALLBACK_CATEGORY_FIELDS),
    }


# ---------------------------------------------------------------------------
# AMP helper
# ---------------------------------------------------------------------------


def get_amp_context(device, use_amp: bool):
    """Return an autocast context manager (inference only, no GradScaler)."""
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast("cuda", dtype=dtype)
        print(f"  AMP enabled (dtype={dtype})")
    else:
        ctx = torch.amp.autocast("cuda", enabled=False)
        if use_amp and device.type != "cuda":
            print("  AMP requested but not on CUDA — disabled")
        else:
            print("  AMP disabled")
    return ctx


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def classify_dataframe(
    df: pd.DataFrame,
    model,
    tokenizer,
    category_fields: List[str],
    device: torch.device,
    amp_ctx,
    args,
    output_path: Path | None = None,
    complexity_model=None,
) -> pd.DataFrame:
    """Run classification on a DataFrame using the category model and optionally
    a complexity regression model.  Both models share the same tokenized inputs.

    Supports resuming: rows with all output columns already populated are skipped.
    """
    text_column = args.text_column
    batch_size = args.batch_size
    max_length = args.max_length
    threshold = args.threshold
    save_every = args.save_every

    if text_column not in df.columns:
        sys.exit(f"Text column '{text_column}' not found. Available: {list(df.columns)}")

    # Initialize category columns if missing
    for field in category_fields:
        if field not in df.columns:
            df[field] = None
    if complexity_model is not None and "reasoning_complexity" not in df.columns:
        df["reasoning_complexity"] = None

    # Find rows that still need classification (enables resume)
    needs_classification = df[category_fields[0]].isna()
    if complexity_model is not None:
        needs_classification = needs_classification | df["reasoning_complexity"].isna()
    remaining_indices = df.index[needs_classification].tolist()

    if not remaining_indices:
        print("  All rows already classified, skipping.")
        return df

    total = len(remaining_indices)
    already_done = len(df) - total
    if already_done > 0:
        print(f"  Resuming: {already_done} rows already classified, {total} remaining")

    texts = [str(df.at[idx, text_column]) for idx in remaining_indices]

    model.eval()
    if complexity_model is not None:
        complexity_model.eval()
    classified_count = 0
    chunk_size = args.chunk_size

    # Pre-allocate numpy arrays for predictions; write to DataFrame once at the end
    num_fields = len(category_fields)
    all_preds = np.empty((total, num_fields), dtype=bool)
    complexity_scores = np.full(total, np.nan, dtype=np.float32) if complexity_model is not None else None
    idx_to_pos = {idx: pos for pos, idx in enumerate(remaining_indices)}
    unsaved_positions = []

    # CUDA stream setup for double-buffered H2D transfers
    use_streams = device.type == "cuda"
    if use_streams:
        compute_stream = torch.cuda.default_stream(device)
        transfer_stream = torch.cuda.Stream(device)

    def _to_device(batch_tuple):
        """Transfer a pre-built batch to GPU, optionally on the transfer stream."""
        b_indices, b_ids, b_mask = batch_tuple
        stream = transfer_stream if use_streams else None
        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            ids = b_ids.to(device, non_blocking=True)
            mask = b_mask.to(device, non_blocking=True)
        return ids, mask, b_indices

    pbar = tqdm(total=total, desc="Classifying", unit="row")

    for chunk_idx, num_chunks, batches in iter_prepared_chunks(
        remaining_indices=remaining_indices,
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        chunk_size=chunk_size,
        pin_memory=use_streams,
        prefetch_chunks=args.prefetch_chunks,
    ):

        if not batches:
            continue

        # Prefetch first batch to GPU
        batch_idx = 0
        next_ids, next_mask, next_indices = _to_device(batches[batch_idx])

        with torch.no_grad():
            while batch_idx < len(batches):
                input_ids = next_ids
                attention_mask = next_mask
                batch_indices = next_indices

                if use_streams:
                    compute_stream.wait_stream(transfer_stream)

                # Prefetch next batch while GPU computes
                batch_idx += 1
                if batch_idx < len(batches):
                    next_ids, next_mask, next_indices = _to_device(batches[batch_idx])

                # GPU forward pass
                with amp_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.sigmoid(outputs.logits).float().cpu().numpy()
                preds = (probs > threshold)

                if complexity_model is not None:
                    with amp_ctx:
                        c_outputs = complexity_model(input_ids=input_ids, attention_mask=attention_mask)
                    c_scores = c_outputs.logits.squeeze(-1).float().cpu().numpy()

                positions = [idx_to_pos[idx] for idx in batch_indices]
                all_preds[positions] = preds
                if complexity_scores is not None:
                    complexity_scores[positions] = c_scores
                unsaved_positions.extend(positions)

                classified_count += len(batch_indices)
                pbar.update(len(batch_indices))
                pbar.set_postfix(
                    chunk=f"{chunk_idx + 1}/{num_chunks}",
                    classified=f"{classified_count}/{total}",
                )

                # Periodic checkpoint save
                if output_path and save_every and classified_count % save_every < batch_size:
                    if unsaved_positions:
                        unsaved_rows = [remaining_indices[p] for p in unsaved_positions]
                        for j, field in enumerate(category_fields):
                            df.loc[unsaved_rows, field] = all_preds[unsaved_positions, j]
                        if complexity_scores is not None:
                            df.loc[unsaved_rows, "reasoning_complexity"] = complexity_scores[unsaved_positions]
                        unsaved_positions.clear()
                    _save_output(df, output_path)
                    pbar.set_postfix(
                        chunk=f"{chunk_idx + 1}/{num_chunks}",
                        classified=f"{classified_count}/{total}",
                        saved=True,
                    )

        del batches

    pbar.close()

    # Bulk-assign all predictions into the DataFrame at once
    for j, field in enumerate(category_fields):
        df.loc[remaining_indices, field] = all_preds[:, j]
    if complexity_scores is not None:
        df.loc[remaining_indices, "reasoning_complexity"] = complexity_scores

    return df


def _save_output(df: pd.DataFrame, output_path: Path) -> None:
    """Save DataFrame to the appropriate format based on file extension."""
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
    elif suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
    else:
        # Default to parquet
        df.to_parquet(output_path, index=False)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run(args):
    """Main categorization pipeline."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load label config from model directory
    label_config = load_label_config(args.model)
    category_fields = label_config["category_fields"]
    print(f"Categories: {len(category_fields)}")

    # Load category model and tokenizer
    print(f"Loading category model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)

    # Load complexity model (same tokenizer family — both ModernBERT-based)
    complexity_model = None
    if not args.no_complexity:
        print(f"Loading complexity model: {args.complexity_model}")
        complexity_model = AutoModelForSequenceClassification.from_pretrained(args.complexity_model)
        complexity_model.to(device)

    # Optionally compile
    if args.compile:
        if hasattr(torch, "compile"):
            cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
            compile_mode = "max-autotune" if cc >= (9, 0) else "reduce-overhead"
            print(f"Compiling category model with torch.compile (mode={compile_mode})...")
            model = torch.compile(model, mode=compile_mode)
            if complexity_model is not None:
                print(f"Compiling complexity model with torch.compile (mode={compile_mode})...")
                complexity_model = torch.compile(complexity_model, mode=compile_mode)
        else:
            print("WARNING: --compile requested but torch.compile not available")

    # AMP setup
    use_amp = not args.no_amp
    amp_ctx = get_amp_context(device, use_amp)

    # Discover input files
    input_files = discover_files(args.input)
    print(f"\nFound {len(input_files)} file(s) to process")

    output_path = Path(args.output)

    # If input is a directory, output to a directory (one output per input file)
    input_is_dir = Path(args.input).is_dir()
    if input_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    total_classified = 0

    for file_idx, input_file in enumerate(input_files):
        print(f"\n{'='*60}")
        print(f"[{file_idx + 1}/{len(input_files)}] Processing: {input_file.name}")
        print(f"{'='*60}")

        df = load_file(input_file)
        total_rows += len(df)
        print(f"  Loaded {len(df)} rows")

        if args.max_rows:
            df = df.iloc[:args.max_rows].reset_index(drop=True)
            print(f"  Limited to {len(df)} rows")

        # Determine output file path
        if input_is_dir:
            # One output file per input file, same name in output directory
            file_output = output_path / input_file.name
        else:
            file_output = output_path

        # If resuming, load existing output to pick up where we left off
        if file_output.exists() and not args.overwrite:
            print(f"  Found existing output: {file_output}")
            existing_df = load_file(file_output)
            # Use existing output if row count matches (same data, partial results)
            if len(existing_df) == len(df) and category_fields[0] in existing_df.columns:
                print(f"  Resuming from existing output")
                df = existing_df
            else:
                print(f"  Row count mismatch or no category columns — processing from scratch")

        before_count = df[category_fields[0]].notna().sum() if category_fields[0] in df.columns else 0

        df = classify_dataframe(
            df, model, tokenizer, category_fields, device, amp_ctx, args,
            output_path=file_output,
            complexity_model=complexity_model,
        )

        after_count = df[category_fields[0]].notna().sum()
        newly_classified = after_count - before_count
        total_classified += newly_classified

        # Ensure category columns are boolean
        for field in category_fields:
            df[field] = df[field].astype(bool)

        # Save final output
        _save_output(df, file_output)
        print(f"  Saved: {file_output} ({len(df)} rows, {newly_classified} newly classified)")

        # Print quick distribution summary
        print(f"\n  Category distribution:")
        for field in category_fields:
            count = df[field].sum()
            pct = count / len(df) * 100
            display = field.replace("_", " ").title()
            print(f"    {display:45s} {count:>7d} ({pct:5.1f}%)")

        if "reasoning_complexity" in df.columns:
            scores = df["reasoning_complexity"].dropna()
            if len(scores) > 0:
                print(f"\n  Complexity distribution (mean={scores.mean():.2f}, std={scores.std():.2f}):")
                level_names = {1: "Factual/Declarative", 2: "Single-step", 3: "Multi-step", 4: "Complex"}
                rounded = scores.round().clip(1, 4).astype(int)
                for level in range(1, 5):
                    count = int((rounded == level).sum())
                    pct = count / len(rounded) * 100
                    print(f"    Level {level} - {level_names[level]:25s} {count:>7d} ({pct:5.1f}%)")

    print(f"\n{'='*60}")
    print(f"Done! Processed {total_rows} rows, classified {total_classified}")
    print(f"Output: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Categorize FineWeb-Edu data using a trained BERT classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file (both category + complexity classifiers)
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet

  # Directory of files
  python categorize_fineweb_edu_bert.py data/raw/ -o data/categorized/

  # Custom model and threshold
  python categorize_fineweb_edu_bert.py input.parquet --model ./my_model --threshold 0.4

  # Category-only mode (skip complexity classifier)
  python categorize_fineweb_edu_bert.py input.parquet --no-complexity

  # Resume from partial output (automatic)
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet
""",
    )

    parser.add_argument("input", help="Path to input file or directory of files (parquet, csv, json, jsonl)")
    parser.add_argument("-o", "--output", default="fineweb-edu-bert-categorized.parquet",
                        help="Output file or directory (default: fineweb-edu-bert-categorized.parquet)")

    # Models
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Path to category classifier model (default: {DEFAULT_MODEL})")
    parser.add_argument("--complexity-model", default=DEFAULT_COMPLEXITY_MODEL,
                        help=f"Path to complexity regression model (default: {DEFAULT_COMPLEXITY_MODEL})")
    parser.add_argument("--no-complexity", action="store_true",
                        help="Skip complexity classification (category-only mode)")

    # Data
    parser.add_argument("--text-column", default="text",
                        help="Column containing text (default: text)")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                        help=f"Max token length (default: {DEFAULT_MAX_LENGTH})")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows per file (default: all)")

    # Inference
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Sigmoid threshold for positive classification (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY,
                        help=f"Save checkpoint every N rows (default: {DEFAULT_SAVE_EVERY})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Texts to tokenize at a time; lower = less RAM (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--prefetch-chunks", type=int, default=DEFAULT_PREFETCH_CHUNKS,
                        help=f"Prepared chunks to prefetch on CPU (0 disables prefetch, default: {DEFAULT_PREFETCH_CHUNKS})")

    # Performance
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference (PyTorch 2.x)")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    # Behavior
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output instead of resuming")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Enable TF32 matmul on CUDA for better performance
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    run(args)


if __name__ == "__main__":
    main()
