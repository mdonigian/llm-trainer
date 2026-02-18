#!/usr/bin/env python3
"""
Train a ModernBERT regressor for reasoning complexity on FineWeb-Edu data.

Takes merged parquet files with a ``reasoning_complexity`` column (integer 1-5)
and fine-tunes a ModernBERT (or other encoder) model as a regressor predicting
continuous complexity scores.  Level 5 rows are excluded from training data by
default, so the model learns to predict values in the 1.0–4.0 range.

The model uses num_labels=1 (regression head) with MSE loss.  At inference time
the raw float prediction can be rounded to the nearest integer level.

Performance features:
  - Automatic Mixed Precision (AMP) with bf16/fp16 on CUDA (--no-amp to disable)
  - torch.compile for fused kernels (--compile, requires PyTorch 2.x)
  - Dynamic padding (pads to longest-in-batch, not max_length)
  - Parallel data loading with --num-workers

Usage:
  python train_complexity_fineweb_edu.py training_data/complexity/
  python train_complexity_fineweb_edu.py merged.parquet --model answerdotai/ModernBERT-base --epochs 5
  python train_complexity_fineweb_edu.py data/ --output-dir ./complexity_model --batch-size 64 --compile

Resume from a checkpoint:
  python train_complexity_fineweb_edu.py data/ --resume models/complexity-classifier/checkpoint.pt --epochs 10

Evaluation only (on a saved model):
  python train_complexity_fineweb_edu.py data/ --eval-only --model ./complexity_model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
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

COMPLEXITY_LEVELS = {
    1: "Factual/Declarative",
    2: "Single-step reasoning",
    3: "Multi-step reasoning",
    4: "Complex reasoning",
    5: "Formal/Abstract reasoning",
}

EXCLUDED_LEVELS = {5}

TRAIN_LEVELS = sorted(set(COMPLEXITY_LEVELS.keys()) - EXCLUDED_LEVELS)
NUM_LABELS = 1  # regression: single continuous output
LABEL_DISPLAY_NAMES = [f"Level {level} - {COMPLEXITY_LEVELS[level]}" for level in TRAIN_LEVELS]

# ---------------------------------------------------------------------------
# Dataset  (stores raw text — tokenization happens in collate_fn)
# ---------------------------------------------------------------------------


class ComplexityDataset(Dataset):
    """PyTorch dataset for complexity regression.

    Stores raw texts and float targets; tokenization is deferred to
    the collate function for dynamic padding.
    """

    def __init__(self, texts: List[str], labels: np.ndarray):
        self.texts = texts
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx]), float(self.labels[idx])


# ---------------------------------------------------------------------------
# Dynamic-padding collate function
# ---------------------------------------------------------------------------


def make_collate_fn(tokenizer, max_length: int = 512):
    """Return a collate function that tokenizes + dynamically pads a batch."""

    def collate_fn(batch):
        texts, labels = zip(*batch)
        encoding = tokenizer(
            list(texts),
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        label_tensor = torch.tensor(labels, dtype=torch.float32)
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


def load_data(
    filepath: str,
    text_column: str = "text",
    complexity_column: str = "reasoning_complexity",
) -> pd.DataFrame:
    """Load complexity-labeled parquet/csv data.

    *filepath* can be a single file or a directory of files.
    Rows with complexity level in EXCLUDED_LEVELS are dropped.
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

    if complexity_column not in df.columns:
        sys.exit(
            f"Complexity column '{complexity_column}' not found.\n"
            f"Available columns: {list(df.columns)}\n"
            f"Did you run complexity_fineweb_edu.py merge first?"
        )

    if text_column not in df.columns:
        sys.exit(f"Text column '{text_column}' not found. Available: {list(df.columns)}")

    before = len(df)
    df = df.dropna(subset=[complexity_column]).reset_index(drop=True)
    df[complexity_column] = df[complexity_column].astype(int)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing complexity ({after} remaining)")

    excluded = df[complexity_column].isin(EXCLUDED_LEVELS)
    n_excluded = int(excluded.sum())
    if n_excluded > 0:
        excluded_detail = df.loc[excluded, complexity_column].value_counts().sort_index()
        for level, count in excluded_detail.items():
            print(f"Excluding {count} rows with level {level} ({COMPLEXITY_LEVELS.get(level, '?')})")
        df = df[~excluded].reset_index(drop=True)

    valid = df[complexity_column].isin(TRAIN_LEVELS)
    n_invalid = int((~valid).sum())
    if n_invalid > 0:
        print(f"Dropping {n_invalid} rows with unexpected complexity values")
        df = df[valid].reset_index(drop=True)

    df["target"] = df[complexity_column].astype(np.float32)

    print(f"Loaded {len(df)} rows total (levels {TRAIN_LEVELS})")
    return df


def prepare_splits(
    df: pd.DataFrame,
    text_column: str,
    complexity_column: str = "reasoning_complexity",
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
):
    """Split data into train/val/test sets, stratified by integer level."""
    texts = df[text_column].tolist()
    targets = df["target"].values
    strata = df[complexity_column].values  # integer levels for stratification

    train_val_texts, test_texts, train_val_targets, test_targets = train_test_split(
        texts, targets, test_size=test_split, random_state=seed, stratify=strata
    )

    # Recompute strata for the train_val subset
    train_val_strata = np.round(train_val_targets).astype(int)
    val_relative = val_split / (1 - test_split)
    train_texts, val_texts, train_targets, val_targets = train_test_split(
        train_val_texts, train_val_targets, test_size=val_relative, random_state=seed,
        stratify=train_val_strata
    )

    print(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    return (
        (train_texts, train_targets),
        (val_texts, val_targets),
        (test_texts, test_targets),
    )


# ---------------------------------------------------------------------------
# AMP helpers
# ---------------------------------------------------------------------------


def get_amp_context(device, use_amp: bool):
    """Return an autocast context manager and a GradScaler (or no-ops)."""
    if use_amp and device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.amp.autocast("cuda", dtype=dtype)
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
# Checkpointing
# ---------------------------------------------------------------------------

CHECKPOINT_FILENAME = "checkpoint.pt"


def save_checkpoint(output_dir: Path, epoch: int, model, optimizer, scheduler, scaler,
                    best_val_mae: float, history: list):
    """Save a full training checkpoint for resuming later."""
    save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": save_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_mae": best_val_mae,
        "history": history,
    }
    torch.save(checkpoint, output_dir / CHECKPOINT_FILENAME)
    print(f"  Checkpoint saved: {output_dir / CHECKPOINT_FILENAME} (epoch {epoch})")


def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, scaler, device):
    """Load a training checkpoint. Returns (start_epoch, best_val_mae, history)."""
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    base_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    base_model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_mae = checkpoint["best_val_mae"]
    history = checkpoint["history"]

    print(f"  Resumed from epoch {checkpoint['epoch']} (best MAE={best_val_mae:.4f})")
    print(f"  Will start training at epoch {start_epoch}")
    return start_epoch, best_val_mae, history


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
    """Evaluate regression model and return metrics."""
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

            preds = outputs.logits.squeeze(-1).float().cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(dataloader)

    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rounded_preds = np.clip(np.round(all_preds), TRAIN_LEVELS[0], TRAIN_LEVELS[-1]).astype(int)
    rounded_labels = np.round(all_labels).astype(int)
    rounded_acc = (rounded_preds == rounded_labels).mean()
    spearman_corr, spearman_p = spearmanr(all_labels, all_preds)

    print(f"\n  {split_name.upper()} Results:")
    print(f"    Loss (MSE):      {avg_loss:.4f}")
    print(f"    MSE:             {mse:.4f}")
    print(f"    MAE:             {mae:.4f}")
    print(f"    Rounded Acc:     {rounded_acc:.4f}")
    print(f"    Spearman r:      {spearman_corr:.4f} (p={spearman_p:.2e})")

    metrics = {
        "loss": avg_loss,
        "mse": float(mse),
        "mae": float(mae),
        "rounded_accuracy": float(rounded_acc),
        "spearman_r": float(spearman_corr),
        "spearman_p": float(spearman_p),
    }

    return metrics, all_preds, all_labels


def print_regression_report(preds, labels):
    """Print a per-level report using rounded predictions."""
    rounded_preds = np.clip(np.round(preds), TRAIN_LEVELS[0], TRAIN_LEVELS[-1]).astype(int)
    rounded_labels = np.round(labels).astype(int)

    print("\nPer-Level Classification Report (rounded predictions):")
    print("=" * 80)
    report = classification_report(
        rounded_labels,
        rounded_preds,
        labels=TRAIN_LEVELS,
        target_names=LABEL_DISPLAY_NAMES,
        zero_division=0,
    )
    print(report)

    print("Prediction statistics:")
    print(f"  Mean prediction: {preds.mean():.3f}")
    print(f"  Std prediction:  {preds.std():.3f}")
    print(f"  Min prediction:  {preds.min():.3f}")
    print(f"  Max prediction:  {preds.max():.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_training(args):
    """Full training pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"\nLoading data from {args.input}...")
    df = load_data(args.input, args.text_column, args.complexity_column)

    print("\nLabel distribution:")
    for i, level in enumerate(TRAIN_LEVELS):
        count = int((df["reasoning_complexity"] == level).sum())
        pct = count / len(df) * 100
        print(f"  {LABEL_DISPLAY_NAMES[i]:45s} {count:>7d} ({pct:5.1f}%)")

    train_data, val_data, test_data = prepare_splits(
        df, args.text_column, args.complexity_column, args.val_split, args.test_split, args.seed
    )

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_LABELS,
        problem_type="regression",
    )
    model.to(device)

    if args.compile:
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile (first batch will be slow)...")
            model = torch.compile(model)
        else:
            print("WARNING: --compile requested but torch.compile not available (need PyTorch 2.x)")

    use_amp = not args.no_amp
    amp_ctx, scaler = get_amp_context(device, use_amp)

    train_dataset = ComplexityDataset(train_data[0], train_data[1])
    val_dataset = ComplexityDataset(val_data[0], val_data[1])
    test_dataset = ComplexityDataset(test_data[0], test_data[1])

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_mae = float("inf")
    history = []

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"Checkpoint not found: {resume_path}")
        start_epoch, best_val_mae, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device
        )

    print(f"\nTraining config:")
    print(f"  Epochs:        {start_epoch}–{args.epochs} (of {args.epochs} total)")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Max length:    {args.max_length}")
    print(f"  AMP:           {'on' if use_amp else 'off'}")
    print(f"  torch.compile: {'on' if args.compile else 'off'}")
    print(f"  Num workers:   {args.num_workers}")
    print(f"  Mode:          regression (num_labels={NUM_LABELS})")
    print(f"  Levels:        {TRAIN_LEVELS}")
    if args.resume:
        print(f"  Resumed from:  {args.resume} (best MAE so far: {best_val_mae:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, amp_ctx, scaler)
        val_metrics, _, _ = evaluate(model, val_loader, device, "val", amp_ctx)

        epoch_record = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(epoch_record)

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            print(f"  New best model (MAE={best_val_mae:.4f}), saving to {output_dir}")
            save_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            save_model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            label_config = {
                "complexity_levels": {str(k): v for k, v in COMPLEXITY_LEVELS.items()},
                "excluded_levels": sorted(EXCLUDED_LEVELS),
                "train_levels": TRAIN_LEVELS,
                "num_labels": NUM_LABELS,
                "problem_type": "regression",
                "label_display_names": LABEL_DISPLAY_NAMES,
            }
            with open(output_dir / "label_config.json", "w") as f:
                json.dump(label_config, f, indent=2)

        save_checkpoint(output_dir, epoch, model, optimizer, scheduler, scaler, best_val_mae, history)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\nTraining history saved to {output_dir / 'training_history.csv'}")

    print(f"\n{'='*60}")
    print("Final evaluation on TEST set (best model)")
    print(f"{'='*60}")
    model = AutoModelForSequenceClassification.from_pretrained(output_dir)
    model.to(device)
    test_metrics, test_preds, test_labels = evaluate(model, test_loader, device, "test", amp_ctx)
    print_regression_report(test_preds, test_labels)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print("Done!")


def run_eval(args):
    """Evaluate a saved model on a dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    use_amp = not args.no_amp
    amp_ctx, _ = get_amp_context(device, use_amp)

    print(f"\nLoading data from {args.input}...")
    df = load_data(args.input, args.text_column, args.complexity_column)

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.to(device)

    dataset = ComplexityDataset(
        df[args.text_column].tolist(),
        df["target"].values,
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
    print_regression_report(preds, labels)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a ModernBERT regressor for reasoning complexity (levels 1-4, excluding level 5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input", help="Path to a parquet/csv file or directory containing complexity-labeled data")

    parser.add_argument("--model", default="answerdotai/ModernBERT-base", help="HuggingFace model name or path (default: answerdotai/ModernBERT-base)")
    parser.add_argument("--output-dir", default="models/complexity-classifier", help="Directory to save the trained model")

    parser.add_argument("--text-column", default="text", help="Column containing text (default: text)")
    parser.add_argument("--complexity-column", default="reasoning_complexity", help="Column containing complexity level (default: reasoning_complexity)")
    parser.add_argument("--max-length", type=int, default=512, help="Max token length (default: 512)")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-split", type=float, default=0.1, help="Test split ratio (default: 0.1)")

    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio (default: 0.1)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (PyTorch 2.x)")
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision (AMP is on by default for CUDA)")

    parser.add_argument("--resume", default=None, help="Path to checkpoint.pt to resume training from")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate a saved model instead of training")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if torch.cuda.is_available() and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_only:
        run_eval(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
