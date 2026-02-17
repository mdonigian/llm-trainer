#!/usr/bin/env python3
"""
Categorize FineWeb-Edu data using a trained BERT (ModernBERT) classifier.

This is the inference counterpart to train_fineweb_edu.py — it takes raw
FineWeb-Edu files (without category labels) and classifies them using the
fine-tuned multi-label model.

Features:
  - Processes single files or entire directories of parquet/csv/json/jsonl
  - AMP (bf16/fp16) for fast GPU inference
  - Dynamic padding (pads to longest-in-batch, not max_length)
  - Periodic checkpoint saves (--save-every) for crash resilience
  - Skips already-categorized rows when resuming from a partial output
  - Configurable sigmoid threshold (--threshold)

Usage:
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet
  python categorize_fineweb_edu_bert.py data/raw/ -o data/categorized/
  python categorize_fineweb_edu_bert.py input.parquet --model ./my_model --batch-size 128 --threshold 0.4

Resume from a partial run:
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet
  (automatically detects existing output and skips already-processed rows)
"""

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "models/fineweb-edu-classifier"
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_LENGTH = 512
DEFAULT_THRESHOLD = 0.5
DEFAULT_NUM_WORKERS = 4
DEFAULT_SAVE_EVERY = 5000
DEFAULT_CHUNK_SIZE = 50_000  # texts to tokenize at a time (limits peak RAM)

# ---------------------------------------------------------------------------
# Dataset — pre-tokenized, sorted by length for minimal padding
# ---------------------------------------------------------------------------


class TokenizedChunkDataset(Dataset):
    """Dataset wrapping a single pre-tokenized chunk (indices + token id lists).

    Constructed by tokenize_chunk(), not directly by the user.
    Each __getitem__ returns a (row_index, token_ids_tensor) pair so that
    tensor construction is distributed across DataLoader worker processes.
    """

    def __init__(self, indices: List[int], input_ids: list):
        self.indices = indices
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.indices[idx], torch.tensor(self.input_ids[idx], dtype=torch.long)


def tokenize_chunk(
    indices: List[int],
    texts: List[str],
    tokenizer,
    max_length: int,
) -> TokenizedChunkDataset:
    """Tokenize a chunk of texts and return a Dataset sorted by length.

    Sorting by token length within each chunk ensures similar-length sequences
    land in the same batch, drastically reducing padding waste and making batch
    times more uniform.
    """
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_attention_mask=False,
        return_length=True,
    )

    ids = encodings["input_ids"]
    lengths = encodings["length"]

    # Sort by token length within this chunk
    order = sorted(range(len(ids)), key=lambda i: lengths[i])
    ids = [ids[i] for i in order]
    sorted_indices = [indices[i] for i in order]

    return TokenizedChunkDataset(sorted_indices, ids)


# ---------------------------------------------------------------------------
# Pad-only collate function (tokenization already done)
# ---------------------------------------------------------------------------


def pad_collate_fn(batch):
    """Stack pre-tokenized tensor samples with right-padding to longest in batch."""
    indices, tensors = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True, padding_value=0)
    # Vectorized attention mask: compare column indices against each sequence's length
    lengths = torch.tensor([t.shape[0] for t in tensors], dtype=torch.long)
    attention_mask = torch.arange(input_ids.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)

    return {
        "indices": list(indices),
        "input_ids": input_ids,
        "attention_mask": attention_mask.to(torch.long),
    }


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
    # Fallback: use the standard 17 categories
    print(f"  Warning: {config_path} not found, using default category fields")
    from scripts.train_fineweb_edu import CATEGORY_FIELDS, LABEL_DISPLAY_NAMES, NUM_LABELS
    return {
        "category_fields": CATEGORY_FIELDS,
        "label_display_names": LABEL_DISPLAY_NAMES,
        "num_labels": NUM_LABELS,
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
) -> pd.DataFrame:
    """Run multi-label classification on a DataFrame and add category columns.

    Supports resuming: if category columns already exist with non-null values,
    those rows are skipped.
    """
    text_column = args.text_column
    batch_size = args.batch_size
    max_length = args.max_length
    threshold = args.threshold
    num_workers = args.num_workers
    save_every = args.save_every

    if text_column not in df.columns:
        sys.exit(f"Text column '{text_column}' not found. Available: {list(df.columns)}")

    # Initialize category columns if missing
    for field in category_fields:
        if field not in df.columns:
            df[field] = None

    # Find rows that still need classification (enables resume)
    needs_classification = df[category_fields[0]].isna()
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
    classified_count = 0
    chunk_size = args.chunk_size

    # Pre-allocate a numpy array for all category columns; write to DataFrame once at the end
    num_fields = len(category_fields)
    all_preds = np.empty((total, num_fields), dtype=bool)
    # Map from remaining_indices position -> index in remaining_indices
    idx_to_pos = {idx: pos for pos, idx in enumerate(remaining_indices)}

    # Double-buffered inference with CUDA streams:
    # While the GPU computes on the current batch, we transfer the next batch
    # to GPU on a separate stream, eliminating H2D transfer stalls.
    use_streams = device.type == "cuda"
    if use_streams:
        compute_stream = torch.cuda.default_stream(device)
        transfer_stream = torch.cuda.Stream(device)

    def _to_device(batch):
        """Transfer batch tensors to device, optionally on transfer stream."""
        stream = transfer_stream if use_streams else None
        with torch.cuda.stream(stream) if stream else contextlib.nullcontext():
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
        return ids, mask, batch["indices"]

    num_chunks = (total + chunk_size - 1) // chunk_size
    pbar = tqdm(total=total, desc="Classifying", unit="row")

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total)

        chunk_indices = remaining_indices[chunk_start:chunk_end]
        chunk_texts = texts[chunk_start:chunk_end]

        # Tokenize just this chunk (fast tokenizer, Rust-backed)
        dataset = tokenize_chunk(chunk_indices, chunk_texts, tokenizer, max_length)

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=pad_collate_fn,
            persistent_workers=False,  # release workers between chunks to free RAM
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
        dataloader = DataLoader(dataset, **loader_kwargs)

        data_iter = iter(dataloader)

        # Prefetch the first batch
        try:
            next_batch_raw = next(data_iter)
        except StopIteration:
            next_batch_raw = None

        if next_batch_raw is not None:
            next_input_ids, next_attention_mask, next_indices = _to_device(next_batch_raw)

        with torch.no_grad():
            while next_batch_raw is not None:
                # Swap in the prefetched batch
                input_ids = next_input_ids
                attention_mask = next_attention_mask
                batch_indices = next_indices

                # Wait for the transfer to finish before compute
                if use_streams:
                    compute_stream.wait_stream(transfer_stream)

                # Start prefetching the next batch while GPU computes
                try:
                    next_batch_raw = next(data_iter)
                except StopIteration:
                    next_batch_raw = None

                if next_batch_raw is not None:
                    next_input_ids, next_attention_mask, next_indices = _to_device(next_batch_raw)

                # GPU forward pass
                with amp_ctx:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

                probs = torch.sigmoid(outputs.logits).float().cpu().numpy()
                preds = (probs > threshold)

                # Batch-write predictions into the pre-allocated array
                positions = [idx_to_pos[idx] for idx in batch_indices]
                all_preds[positions] = preds

                classified_count += len(batch_indices)
                pbar.update(len(batch_indices))
                pbar.set_postfix(
                    chunk=f"{chunk_idx + 1}/{num_chunks}",
                    classified=f"{classified_count}/{total}",
                )

                # Periodic checkpoint save
                if output_path and save_every and classified_count % save_every < batch_size:
                    for j, field in enumerate(category_fields):
                        df.loc[remaining_indices[:classified_count], field] = all_preds[:classified_count, j]
                    _save_output(df, output_path)
                    pbar.set_postfix(
                        chunk=f"{chunk_idx + 1}/{num_chunks}",
                        classified=f"{classified_count}/{total}",
                        saved=True,
                    )

        # Free this chunk's tokenized data before processing the next
        del dataset, dataloader, data_iter

    pbar.close()

    # Bulk-assign all predictions into the DataFrame at once
    for j, field in enumerate(category_fields):
        df.loc[remaining_indices, field] = all_preds[:, j]

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

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)

    # Optionally compile
    if args.compile:
        if hasattr(torch, "compile"):
            # Use max-autotune on Hopper+ (CC >= 9.0) for aggressive kernel fusion
            cc = torch.cuda.get_device_capability(device) if device.type == "cuda" else (0, 0)
            compile_mode = "max-autotune" if cc >= (9, 0) else "default"
            print(f"Compiling model with torch.compile (mode={compile_mode})...")
            model = torch.compile(model, mode=compile_mode)
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
  # Single file
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet

  # Directory of files
  python categorize_fineweb_edu_bert.py data/raw/ -o data/categorized/

  # Custom model and threshold
  python categorize_fineweb_edu_bert.py input.parquet --model ./my_model --threshold 0.4

  # Resume from partial output (automatic)
  python categorize_fineweb_edu_bert.py input.parquet -o categorized.parquet
""",
    )

    parser.add_argument("input", help="Path to input file or directory of files (parquet, csv, json, jsonl)")
    parser.add_argument("-o", "--output", default="fineweb-edu-bert-categorized.parquet",
                        help="Output file or directory (default: fineweb-edu-bert-categorized.parquet)")

    # Model
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Path to trained model directory (default: {DEFAULT_MODEL})")

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
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"DataLoader workers (default: {DEFAULT_NUM_WORKERS})")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY,
                        help=f"Save checkpoint every N rows (default: {DEFAULT_SAVE_EVERY})")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"Texts to tokenize at a time; lower = less RAM (default: {DEFAULT_CHUNK_SIZE})")

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
