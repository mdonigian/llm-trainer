#!/usr/bin/env python3
"""
Train a multi-task ModernBERT classifier for StarCoder code classification.

Three tasks trained jointly with a shared encoder:
  - Code Quality (1-5): regression head (MSE loss)
  - Structured Data Relevance (0-3): regression head (MSE loss)
  - Content Type (9 classes): classification head (CrossEntropy loss)

Takes the merged parquet files from classify_starcoderdata.py (with columns
quality, structured_data, content_type) and fine-tunes a single model.

Performance features:
  - Automatic Mixed Precision (AMP) with bf16/fp16 on CUDA (--no-amp to disable)
  - torch.compile for fused kernels (--compile, requires PyTorch 2.x)
  - Dynamic padding (pads to longest-in-batch, not max_length)
  - Parallel data loading with --num-workers

Usage:
  python train_starcoderdata.py training_data/starcoderdata-classified/
  python train_starcoderdata.py data/ --model answerdotai/ModernBERT-base --epochs 5
  python train_starcoderdata.py data/ --output-dir ./code_model --batch-size 64 --compile

Resume from a checkpoint:
  python train_starcoderdata.py data/ --resume models/starcoderdata-classifier/checkpoint.pt

Evaluation only:
  python train_starcoderdata.py data/ --eval-only --model ./code_model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import (
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTENT_TYPES = [
    "library", "application", "test", "config", "tutorial",
    "data", "generated", "script", "other",
]
CONTENT_TYPE_TO_IDX = {ct: i for i, ct in enumerate(CONTENT_TYPES)}
NUM_CONTENT_TYPES = len(CONTENT_TYPES)

QUALITY_RANGE = (1, 5)
STRUCTURED_DATA_RANGE = (0, 3)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class CodeClassifierModel(nn.Module):
    """Multi-task model: shared encoder with three task-specific heads."""

    def __init__(self, encoder_name_or_path: str, num_content_types: int = NUM_CONTENT_TYPES):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
        hidden = self.encoder.config.hidden_size
        drop_rate = getattr(self.encoder.config, "classifier_dropout", None)
        if drop_rate is None:
            drop_rate = getattr(self.encoder.config, "hidden_dropout_prob", 0.1)
        self.dropout = nn.Dropout(drop_rate)
        self.quality_head = nn.Linear(hidden, 1)
        self.structured_data_head = nn.Linear(hidden, 1)
        self.content_type_head = nn.Linear(hidden, num_content_types)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(outputs.last_hidden_state[:, 0])
        return (
            self.quality_head(cls).squeeze(-1),
            self.structured_data_head(cls).squeeze(-1),
            self.content_type_head(cls),
        )


def save_model(model, tokenizer, output_dir: Path):
    """Save encoder (HuggingFace format) + classifier heads + tokenizer."""
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    base.encoder.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.save({
        "quality_head": base.quality_head.state_dict(),
        "structured_data_head": base.structured_data_head.state_dict(),
        "content_type_head": base.content_type_head.state_dict(),
    }, output_dir / "classifier_heads.pt")


def load_model(model_dir: Path, device="cpu") -> CodeClassifierModel:
    """Load a saved multi-task model."""
    model = CodeClassifierModel(str(model_dir))
    heads = torch.load(model_dir / "classifier_heads.pt", map_location=device, weights_only=True)
    model.quality_head.load_state_dict(heads["quality_head"])
    model.structured_data_head.load_state_dict(heads["structured_data_head"])
    model.content_type_head.load_state_dict(heads["content_type_head"])
    return model


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CodeDataset(Dataset):
    """Stores raw texts and three label types; tokenization deferred to collate."""

    def __init__(
        self,
        texts: List[str],
        quality: np.ndarray,
        structured_data: np.ndarray,
        content_type: np.ndarray,
    ):
        self.texts = texts
        self.quality = quality.astype(np.float32)
        self.structured_data = structured_data.astype(np.float32)
        self.content_type = content_type.astype(np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            str(self.texts[idx]),
            float(self.quality[idx]),
            float(self.structured_data[idx]),
            int(self.content_type[idx]),
        )


# ---------------------------------------------------------------------------
# Dynamic-padding collate function
# ---------------------------------------------------------------------------


def make_collate_fn(tokenizer, max_length: int = 512):
    """Return a collate function that tokenizes + dynamically pads a batch."""

    def collate_fn(batch):
        texts, quality, structured_data, content_type = zip(*batch)
        encoding = tokenizer(
            list(texts),
            max_length=max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "quality_labels": torch.tensor(quality, dtype=torch.float32),
            "structured_data_labels": torch.tensor(structured_data, dtype=torch.float32),
            "content_type_labels": torch.tensor(content_type, dtype=torch.long),
        }

    return collate_fn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_single_file(p: Path) -> pd.DataFrame:
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        return pd.read_csv(p)
    else:
        sys.exit(f"Unsupported file format: {p.suffix}")


def load_data(
    filepath: str,
    text_column: str = "content",
) -> pd.DataFrame:
    """Load classified parquet/csv data, dropping unclassified rows."""
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

    for col in ["quality", "structured_data", "content_type"]:
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Available: {list(df.columns)}")
    if text_column not in df.columns:
        sys.exit(f"Text column '{text_column}' not found. Available: {list(df.columns)}")

    before = len(df)
    df = df[df["quality"] > 0].reset_index(drop=True)
    dropped = before - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} unclassified rows (quality=0)")

    unknown_types = set(df["content_type"].unique()) - set(CONTENT_TYPES)
    if unknown_types:
        print(f"Warning: unknown content types will be mapped to 'other': {unknown_types}")
        df.loc[~df["content_type"].isin(CONTENT_TYPES), "content_type"] = "other"

    df["content_type_idx"] = df["content_type"].map(CONTENT_TYPE_TO_IDX)

    print(f"Loaded {len(df):,} rows total")
    return df


def prepare_splits(
    df: pd.DataFrame,
    text_column: str,
    val_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 42,
) -> Tuple:
    """Split data into train/val/test sets, stratified by quality level."""
    texts = df[text_column].tolist()
    quality = df["quality"].values
    structured_data = df["structured_data"].values
    content_type_idx = df["content_type_idx"].values
    strata = quality.astype(int)

    idx = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_split, random_state=seed, stratify=strata,
    )

    val_relative = val_split / (1 - test_split)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_relative, random_state=seed,
        stratify=strata[train_val_idx],
    )

    def _subset(indices):
        return (
            [texts[i] for i in indices],
            quality[indices],
            structured_data[indices],
            content_type_idx[indices],
        )

    train_data = _subset(train_idx)
    val_data = _subset(val_idx)
    test_data = _subset(test_idx)

    print(f"Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")
    return train_data, val_data, test_data


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
# Loss
# ---------------------------------------------------------------------------

_mse_loss = nn.MSELoss()
_ce_loss = nn.CrossEntropyLoss()


def compute_loss(quality_pred, sd_pred, ct_logits, quality_labels, sd_labels, ct_labels):
    """Combined loss: MSE(quality) + MSE(structured_data) + CE(content_type)."""
    q_loss = _mse_loss(quality_pred, quality_labels)
    sd_loss = _mse_loss(sd_pred, sd_labels)
    ct_loss = _ce_loss(ct_logits, ct_labels)
    return q_loss + sd_loss + ct_loss, q_loss, sd_loss, ct_loss


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

CHECKPOINT_FILENAME = "checkpoint.pt"


def save_checkpoint(output_dir: Path, epoch: int, model, optimizer, scheduler, scaler,
                    best_val_metric: float, history: list):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": base.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_metric": best_val_metric,
        "history": history,
    }
    torch.save(checkpoint, output_dir / CHECKPOINT_FILENAME)
    print(f"  Checkpoint saved (epoch {epoch})")


def load_checkpoint(checkpoint_path: Path, model, optimizer, scheduler, scaler, device):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    base.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_val_metric = checkpoint["best_val_metric"]
    history = checkpoint["history"]
    print(f"  Resumed from epoch {checkpoint['epoch']} (best metric={best_val_metric:.4f})")
    return start_epoch, best_val_metric, history


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch_num, amp_ctx, scaler):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch_num} [train]")

    for batch in pbar:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        q_labels = batch["quality_labels"].to(device, non_blocking=True)
        sd_labels = batch["structured_data_labels"].to(device, non_blocking=True)
        ct_labels = batch["content_type_labels"].to(device, non_blocking=True)

        with amp_ctx:
            q_pred, sd_pred, ct_logits = model(input_ids, attention_mask)
            loss, _, _, _ = compute_loss(q_pred, sd_pred, ct_logits, q_labels, sd_labels, ct_labels)

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
    """Evaluate all three tasks. Returns (metrics_dict, predictions_dict)."""
    model.eval()
    all_q_preds, all_sd_preds, all_ct_preds = [], [], []
    all_q_labels, all_sd_labels, all_ct_labels = [], [], []
    total_loss = 0

    if amp_ctx is None:
        amp_ctx = torch.amp.autocast("cuda", enabled=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating [{split_name}]"):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            q_labels = batch["quality_labels"].to(device, non_blocking=True)
            sd_labels = batch["structured_data_labels"].to(device, non_blocking=True)
            ct_labels = batch["content_type_labels"].to(device, non_blocking=True)

            with amp_ctx:
                q_pred, sd_pred, ct_logits = model(input_ids, attention_mask)
                loss, _, _, _ = compute_loss(q_pred, sd_pred, ct_logits, q_labels, sd_labels, ct_labels)

            total_loss += loss.item()
            all_q_preds.append(q_pred.float().cpu().numpy())
            all_sd_preds.append(sd_pred.float().cpu().numpy())
            all_ct_preds.append(ct_logits.float().cpu().numpy())
            all_q_labels.append(q_labels.cpu().numpy())
            all_sd_labels.append(sd_labels.cpu().numpy())
            all_ct_labels.append(ct_labels.cpu().numpy())

    q_preds = np.concatenate(all_q_preds)
    sd_preds = np.concatenate(all_sd_preds)
    ct_preds = np.concatenate(all_ct_preds)
    q_labels = np.concatenate(all_q_labels)
    sd_labels = np.concatenate(all_sd_labels)
    ct_labels = np.concatenate(all_ct_labels)

    avg_loss = total_loss / len(dataloader)

    # Quality metrics
    q_mae = mean_absolute_error(q_labels, q_preds)
    q_mse = mean_squared_error(q_labels, q_preds)
    q_rounded = np.clip(np.round(q_preds), *QUALITY_RANGE).astype(int)
    q_acc = (q_rounded == np.round(q_labels).astype(int)).mean()
    q_spearman, _ = spearmanr(q_labels, q_preds)

    # Structured data metrics
    sd_mae = mean_absolute_error(sd_labels, sd_preds)
    sd_mse = mean_squared_error(sd_labels, sd_preds)
    sd_rounded = np.clip(np.round(sd_preds), *STRUCTURED_DATA_RANGE).astype(int)
    sd_acc = (sd_rounded == np.round(sd_labels).astype(int)).mean()
    sd_spearman, _ = spearmanr(sd_labels, sd_preds)

    # Content type metrics
    ct_pred_classes = ct_preds.argmax(axis=1)
    ct_acc = (ct_pred_classes == ct_labels).mean()
    ct_f1 = f1_score(ct_labels, ct_pred_classes, average="macro", zero_division=0)

    combined_mae = (q_mae + sd_mae) / 2

    print(f"\n  {split_name.upper()} Results (loss={avg_loss:.4f}):")
    print(f"    Quality:         MAE={q_mae:.4f}  Acc={q_acc:.4f}  Spearman={q_spearman:.4f}")
    print(f"    Structured Data: MAE={sd_mae:.4f}  Acc={sd_acc:.4f}  Spearman={sd_spearman:.4f}")
    print(f"    Content Type:    Acc={ct_acc:.4f}  Macro-F1={ct_f1:.4f}")
    print(f"    Combined MAE:    {combined_mae:.4f}")

    metrics = {
        "loss": avg_loss,
        "quality_mae": float(q_mae),
        "quality_mse": float(q_mse),
        "quality_rounded_acc": float(q_acc),
        "quality_spearman": float(q_spearman),
        "structured_data_mae": float(sd_mae),
        "structured_data_mse": float(sd_mse),
        "structured_data_rounded_acc": float(sd_acc),
        "structured_data_spearman": float(sd_spearman),
        "content_type_accuracy": float(ct_acc),
        "content_type_macro_f1": float(ct_f1),
        "combined_mae": float(combined_mae),
    }

    preds = {
        "quality": q_preds,
        "structured_data": sd_preds,
        "content_type": ct_pred_classes,
        "quality_labels": q_labels,
        "structured_data_labels": sd_labels,
        "content_type_labels": ct_labels,
    }

    return metrics, preds


def print_test_report(preds: dict):
    """Print detailed per-class reports for all three dimensions."""
    q_rounded = np.clip(np.round(preds["quality"]), *QUALITY_RANGE).astype(int)
    q_labels = np.round(preds["quality_labels"]).astype(int)
    print("\nQuality — Per-Level Report (rounded predictions):")
    print("=" * 70)
    q_names = [f"Quality {i}" for i in range(QUALITY_RANGE[0], QUALITY_RANGE[1] + 1)]
    print(classification_report(
        q_labels, q_rounded,
        labels=list(range(QUALITY_RANGE[0], QUALITY_RANGE[1] + 1)),
        target_names=q_names, zero_division=0,
    ))

    sd_rounded = np.clip(np.round(preds["structured_data"]), *STRUCTURED_DATA_RANGE).astype(int)
    sd_labels = np.round(preds["structured_data_labels"]).astype(int)
    print("Structured Data — Per-Level Report (rounded predictions):")
    print("=" * 70)
    sd_names = [f"SD {i}" for i in range(STRUCTURED_DATA_RANGE[0], STRUCTURED_DATA_RANGE[1] + 1)]
    print(classification_report(
        sd_labels, sd_rounded,
        labels=list(range(STRUCTURED_DATA_RANGE[0], STRUCTURED_DATA_RANGE[1] + 1)),
        target_names=sd_names, zero_division=0,
    ))

    print("Content Type — Classification Report:")
    print("=" * 70)
    print(classification_report(
        preds["content_type_labels"], preds["content_type"],
        labels=list(range(NUM_CONTENT_TYPES)),
        target_names=CONTENT_TYPES, zero_division=0,
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_training(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    print(f"\nLoading data from {args.input}...")
    df = load_data(args.input, args.text_column)

    print("\nQuality distribution:")
    for level in range(1, 6):
        c = int((df["quality"] == level).sum())
        print(f"  {level}: {c:>7,d} ({c / len(df) * 100:5.1f}%)")

    print("\nStructured Data distribution:")
    for level in range(0, 4):
        c = int((df["structured_data"] == level).sum())
        print(f"  {level}: {c:>7,d} ({c / len(df) * 100:5.1f}%)")

    print("\nContent Type distribution:")
    for ct in CONTENT_TYPES:
        c = int((df["content_type"] == ct).sum())
        print(f"  {ct:<15s} {c:>7,d} ({c / len(df) * 100:5.1f}%)")

    train_data, val_data, test_data = prepare_splits(
        df, args.text_column, args.val_split, args.test_split, args.seed,
    )

    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = CodeClassifierModel(args.model)
    model.to(device)

    if args.compile:
        if hasattr(torch, "compile"):
            print("Compiling model with torch.compile (first batch will be slow)...")
            model = torch.compile(model)
        else:
            print("WARNING: --compile requested but torch.compile not available")

    use_amp = not args.no_amp
    amp_ctx, scaler = get_amp_context(device, use_amp)

    train_dataset = CodeDataset(*train_data)
    val_dataset = CodeDataset(*val_data)
    test_dataset = CodeDataset(*test_data)

    collate_fn = make_collate_fn(tokenizer, args.max_length)
    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    best_val_metric = float("inf")
    history = []

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            sys.exit(f"Checkpoint not found: {resume_path}")
        start_epoch, best_val_metric, history = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device,
        )

    print(f"\nTraining config:")
    print(f"  Epochs:        {start_epoch}–{args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Warmup steps:  {warmup_steps}")
    print(f"  Max length:    {args.max_length}")
    print(f"  AMP:           {'on' if use_amp else 'off'}")
    print(f"  torch.compile: {'on' if args.compile else 'off'}")
    print(f"  Num workers:   {args.num_workers}")
    print(f"  Tasks:         quality (reg) + structured_data (reg) + content_type (cls)")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, amp_ctx, scaler,
        )
        val_metrics, _ = evaluate(model, val_loader, device, "val", amp_ctx)

        epoch_record = {"epoch": epoch, "train_loss": train_loss}
        epoch_record.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(epoch_record)

        if val_metrics["combined_mae"] < best_val_metric:
            best_val_metric = val_metrics["combined_mae"]
            print(f"  New best model (combined MAE={best_val_metric:.4f}), saving to {output_dir}")
            save_model(model, tokenizer, output_dir)

            label_config = {
                "content_types": CONTENT_TYPES,
                "quality_range": list(QUALITY_RANGE),
                "structured_data_range": list(STRUCTURED_DATA_RANGE),
                "num_content_types": NUM_CONTENT_TYPES,
                "tasks": {
                    "quality": "regression",
                    "structured_data": "regression",
                    "content_type": "classification",
                },
            }
            with open(output_dir / "label_config.json", "w") as f:
                json.dump(label_config, f, indent=2)

        save_checkpoint(output_dir, epoch, model, optimizer, scheduler, scaler,
                        best_val_metric, history)

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"\nTraining history saved to {output_dir / 'training_history.csv'}")

    print(f"\n{'='*60}")
    print("Final evaluation on TEST set (best model)")
    print(f"{'='*60}")
    model = load_model(output_dir, device)
    model.to(device)
    test_metrics, test_preds = evaluate(model, test_loader, device, "test", amp_ctx)
    print_test_report(test_preds)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print(f"\nModel saved to: {output_dir}")
    print("Done!")


def run_eval(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    use_amp = not args.no_amp
    amp_ctx, _ = get_amp_context(device, use_amp)

    print(f"\nLoading data from {args.input}...")
    df = load_data(args.input, args.text_column)

    print(f"Loading model from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(Path(args.model), device)
    model.to(device)

    dataset = CodeDataset(
        df[args.text_column].tolist(),
        df["quality"].values.astype(np.float32),
        df["structured_data"].values.astype(np.float32),
        df["content_type_idx"].values,
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

    metrics, preds = evaluate(model, dataloader, device, "eval", amp_ctx)
    print_test_report(preds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a multi-task ModernBERT classifier for code quality, "
                    "structured data relevance, and content type",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_starcoderdata.py training_data/starcoderdata-classified/
  python train_starcoderdata.py data/ --epochs 5 --batch-size 64 --compile
  python train_starcoderdata.py data/ --resume models/starcoderdata-classifier/checkpoint.pt
  python train_starcoderdata.py data/ --eval-only --model models/starcoderdata-classifier/
""",
    )

    parser.add_argument("input", help="Path to classified parquet/csv file or directory")

    parser.add_argument("--model", default="codesage/codesage-small-v2",
                        help="HuggingFace model name or path (default: codesage/codesage-small-v2)")
    parser.add_argument("--output-dir", default="models/starcoderdata-classifier",
                        help="Directory to save the trained model")

    parser.add_argument("--text-column", default="content",
                        help="Column containing code text (default: content)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Max token length (default: 512)")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--test-split", type=float, default=0.1,
                        help="Test split ratio (default: 0.1)")

    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio (default: 0.1)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")

    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster training")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable automatic mixed precision")

    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint.pt to resume training")
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate a saved model instead of training")

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
