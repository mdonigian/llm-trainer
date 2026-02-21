#!/usr/bin/env python3
"""
Train a ModernBERT multi-label classifier on FineWeb-Edu categorized data.

Takes the output of categorize_fineweb_edu.py (a parquet file with 17 boolean
category columns) and fine-tunes a ModernBERT (or other encoder) model for multi-label classification.

Performance features:
  - Automatic Mixed Precision (AMP) with bf16/fp16 on CUDA (--no-amp to disable)
  - torch.compile for fused kernels (--compile, requires PyTorch 2.x)
  - Dynamic padding (pads to longest-in-batch, not max_length)
  - Parallel data loading with --num-workers

Usage:
  python train_fineweb_edu.py input.parquet
  python train_fineweb_edu.py data/categorized/          # reads all parquet/csv in dir
  python train_fineweb_edu.py input.parquet --model answerdotai/ModernBERT-base --epochs 5
  python train_fineweb_edu.py input.parquet --output-dir ./my_model --batch-size 64 --compile

Evaluation only (on a saved model):
  python train_fineweb_edu.py input.parquet --eval-only --model ./my_model
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_FIELDS = [
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

LABEL_DISPLAY_NAMES = [f.replace("_", " ").title() for f in CATEGORY_FIELDS]

NUM_LABELS = len(CATEGORY_FIELDS)

# ---------------------------------------------------------------------------
# Dataset  (stores raw text — tokenization happens in collate_fn)
# ---------------------------------------------------------------------------


class FineWebEduDataset(Dataset):
    """PyTorch dataset for multi-label text classification.

    Stores raw texts and labels; tokenization is deferred to the collate
    function so that (a) it runs in DataLoader worker processes and (b) we
    can do dynamic padding per-batch instead of padding every sample to
    max_length.
    """

    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx]), self.labels[idx]


# ---------------------------------------------------------------------------
# Dynamic-padding collate function
# ---------------------------------------------------------------------------


def make_collate_fn(tokenizer, max_length: int = 512):
    """Return a collate function that tokenizes + dynamically pads a batch.

    Instead of padding every sample to *max_length*, this pads only to the
    longest sequence in the batch.  For datasets where most texts are much
    shorter than max_length this dramatically reduces wasted compute on
    padding tokens.
    """

    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoding = tokenizer(
            list(texts),
            max_length=max_length,
            padding="longest",       # <-- dynamic padding to longest-in-batch
            truncation=True,
            return_tensors="pt",
        )
        label_tensor = torch.tensor(np.array(labels), dtype=torch.float32)
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": label_tensor,
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_single_file(p: Path) -> pd.DataFrame:
    """Read a single parquet or csv file."""
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    else:
        sys.exit(f"Unsupported file format: {p.suffix}")


def load_data(filepath: str, text_column: str = "text") -> pd.DataFrame:
    """Load categorized parquet/csv data.

    *filepath* can be:
      - a single .parquet or .csv file
      - a directory containing .parquet and/or .csv files (all are concatenated)
    """
    p = Path(filepath)

    if p.is_dir():
        files = sorted(p.glob("*.parquet")) + sorted(p.glob("*.csv"))
        if not files:
            sys.exit(f"No .parquet or .csv files found in directory: {p}")
        print(f"Found {len(files)} file(s) in {p}")
        dfs = []
        for f in files:
            print(f"  Loading {f.name}...")
            dfs.append(_read_single_file(f))
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = _read_single_file(p)

    # Validate required columns exist
    missing = [c for c in CATEGORY_FIELDS if c not in df.columns]
    if missing:
        sys.exit(
            f"Missing category columns: {missing}\n"
            f"Available columns: {list(df.columns)}\n"
            f"Did you run categorize_fineweb_edu.py first?"
        )

    if text_column not in df.columns:
        sys.exit(f"Text column '{text_column}' not found. Available: {list(df.columns)}")

    # Drop rows where labels are missing (uncategorized rows)
    before = len(df)
    df = df.dropna(subset=CATEGORY_FIELDS).reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing labels ({after} remaining)")

    # Ensure booleans
    for col in CATEGORY_FIELDS:
        df[col] = df[col].astype(bool).astype(int)

    print(f"Loaded {len(df)} rows total")
    return df


def prepare_splits(
    df: pd.DataFrame,
    text_column: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
):
    """Split data into train/val/test sets."""
    texts = df[text_column].tolist()
    labels = df[CATEGORY_FIELDS].values

    # First split: separate test set
    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=test_split, random_state=seed
    )

    # Second split: separate validation from training
    val_relative = val_split / (1 - test_split)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts, train_val_labels, test_size=val_relative, random_state=seed
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    return (
        (train_texts, train_labels),
        (val_texts, val_labels),
        (test_texts, test_labels),
    )


# ---------------------------------------------------------------------------
# AMP helpers
# ---------------------------------------------------------------------------


def get_amp_context(device, use_amp: bool):
    """Return an autocast context manager and a GradScaler (or no-ops)."""
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast("cuda", dtype=dtype)
        # GradScaler is only needed for fp16; bf16 doesn't need loss scaling
        scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
        print(f"  AMP enabled (dtype={dtype})")
    else:
        ctx = torch.amp.autocast("cuda", enabled=False)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        if use_amp and device.type != "cuda":
            print("  AMP requested but not on CUDA — disabled")
        else:
            print("  AMP disabled")
    return ctx, scaler


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch_num, amp_ctx, scaler):
    """Run one training epoch with optional AMP."""
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} [train]")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with amp_ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(dataloader)
    print(f"  Train loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, dataloader, device, split_name="val", amp_ctx=None):
    """Evaluate model and return metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    if amp_ctx is None:
        amp_ctx = torch.amp.autocast("cuda", enabled=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [{split_name}]"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with amp_ctx:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            total_loss += outputs.loss.item()

            # Sigmoid -> threshold at 0.5
            probs = torch.sigmoid(outputs.logits)
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss = total_loss / len(dataloader)

    # Per-label and aggregate metrics
    micro_f1 = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    micro_precision = precision_score(all_labels, all_preds, average="micro", zero_division=0)
    micro_recall = recall_score(all_labels, all_preds, average="micro", zero_division=0)

    print(f"\n  {split_name.upper()} Results:")
    print(f"    Loss:       {avg_loss:.4f}")
    print(f"    Micro F1:   {micro_f1:.4f}")
    print(f"    Macro F1:   {macro_f1:.4f}")
    print(f"    Precision:  {micro_precision:.4f}")
    print(f"    Recall:     {micro_recall:.4f}")

    metrics = {
        "loss": avg_loss,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "precision": micro_precision,
        "recall": micro_recall,
    }

    return metrics, all_preds, all_labels


def print_classification_report(preds, labels):
    """Print a per-category classification report."""
    print("\nPer-Category Classification Report:")
    print("=" * 80)
    report = classification_report(
        labels,
        preds,
        target_names=LABEL_DISPLAY_NAMES,
        zero_division=0,
    )
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_training(args):
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split data
    print(f"\nLoading data from {args.input}...")
    df = load_data(args.input, args.text_column)

    # Print label distribution
    print("\nLabel distribution:")
    for field in CATEGORY_FIELDS:
        count = df[field].sum()
        pct = count / len(df) * 100
        print(f"  {field.replace('_', ' ').title():45s} {count:>7d} ({pct:5.1f}%)")

    train_data, val_data, test_data = prepare_splits(
        df, args.text_column, args.val_split, args.test_split, args.seed
    )

    # Load tokenizer and model
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification",
    )
    model.to(device)

    # Optionally compile the model (PyTorch 2.x)
    if args.compile:
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile (first batch will be slow)...")
            model = torch.compile(model)
        else:
            print("WARNING: --compile requested but torch.compile not available (need PyTorch 2.x)")

    # AMP setup
    use_amp = not args.no_amp
    amp_ctx, scaler = get_amp_context(device, use_amp)

    # Create datasets — raw text, tokenization deferred to collate_fn
    train_dataset = FineWebEduDataset(train_data[0], train_data[1])
    val_dataset = FineWebEduDataset(val_data[0], val_data[1])
    test_dataset = FineWebEduDataset(test_data[0], test_data[1])

    collate_fn = make_collate_fn(tokenizer, args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"\nTraining config:")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Max length:    {args.max_length}")
    print(f"  AMP:           {'on' if use_amp else 'off'}")
    print(f"  torch.compile: {'on' if args.compile else 'off'}")
    print(f"  Num workers:   {args.num_workers}")

    # Training loop
    best_val_f1 = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, amp_ctx, scaler)
        val_metrics, _, _ = evaluate(model, val_loader, device, "val", amp_ctx)

        epoch_record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_record)

        # Save best model
        if val_metrics["micro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["micro_f1"]
            print(f"  New best model (F1={best_val_f1:.4f}), saving to {output_dir}")
            # Unwrap compiled model before saving
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            # Save label mapping
            label_config = {
                "category_fields": CATEGORY_FIELDS,
                "label_display_names": LABEL_DISPLAY_NAMES,
                "num_labels": NUM_LABELS,
            }
            with open(output_dir / "label_config.json", "w") as f:
                json.dump(label_config, f, indent=2)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\nTraining history saved to {output_dir / 'training_history.csv'}")

    # Final evaluation on test set with best model
    print(f"\n{'='*60}")
    print("Final evaluation on TEST set (best model)")
    print(f"{'='*60}")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, "test", amp_ctx)
    print_classification_report(test_preds, test_labels)

    # Save test metrics
    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print("Done!")


def run_eval(args):
    """Evaluate a saved model on a dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # AMP setup
    use_amp = not args.no_amp
    amp_ctx, _ = get_amp_context(device, use_amp)

    print(f"\nLoading data from {args.input}...")
    df = load_data(args.input, args.text_column)

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)

    dataset = FineWebEduDataset(
        df[args.text_column].tolist(),
        df[CATEGORY_FIELDS].values,
    )
    collate_fn = make_collate_fn(tokenizer, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )

    metrics, preds, labels = evaluate(model, dataloader, device, "eval", amp_ctx)
    print_classification_report(preds, labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a ModernBERT multi-label classifier on FineWeb-Edu categories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to a categorized parquet/csv file, or a directory containing them")

    # Model
    parser.add_argument("--model", default="answerdotai/ModernBERT-base", help="HuggingFace model name or path (default: answerdotai/ModernBERT-base)")
    parser.add_argument("--output-dir", default="models/fineweb-edu-classifier", help="Directory to save the trained model")

    # Data
    parser.add_argument("--text-column", default="text", help="Column containing text (default: text)")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length (default: 512)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio (default: 0.1)")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Performance
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (PyTorch 2.x)")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision (AMP is on by default for CUDA)")

    # Mode
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved model instead of training")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Enable TF32 matmul on CUDA for better performance (silences inductor warning)
    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    # Set seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_only:
        run_eval(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
