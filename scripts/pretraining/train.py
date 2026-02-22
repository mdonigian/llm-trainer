#!/usr/bin/env python3
"""
Pretrain a ~470M LLaMA-style model on pre-tokenized data from HuggingFace.

Designed for single-GPU training on an RTX 4090 (24GB VRAM, 36GB system RAM).
Expects a HuggingFace dataset of binary shards produced by
prepare_tokenized_dataset.py (TKDS format: uint16 packed sequences of 2048 tokens).

Key memory-saving techniques:
  - bf16 mixed precision (Ampere+ native)
  - Gradient checkpointing (trades ~30% speed for ~40% VRAM savings)
  - Gradient accumulation (effective batch size >> micro batch size)
  - Streaming dataset (no full dataset in RAM)
  - Fused AdamW via torch (fewer optimizer buffers)

Estimated throughput: ~45-55k tokens/sec on RTX 4090 with bf16.
Estimated wall time for 15B tokens: ~75-90 hours.

Usage:
  # Full training run
  python train.py \
      --dataset-repo youruser/curated-15B-tokenized \
      --run-name curated-470m-15b \
      --output-dir /workspace/checkpoints

  # Resume from checkpoint
  python train.py \
      --dataset-repo youruser/curated-15B-tokenized \
      --run-name curated-470m-15b \
      --output-dir /workspace/checkpoints \
      --resume

  # Short validation run (1000 steps)
  python train.py \
      --dataset-repo youruser/curated-15B-tokenized \
      --run-name test-run \
      --output-dir /workspace/checkpoints \
      --max-steps 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# ── Architecture constants ────────────────────────────────────────────────────

VOCAB_SIZE = 50_304
CONTEXT_LENGTH = 2048

SHARD_MAGIC = b"TKDS"
SHARD_HEADER_FMT = "<4sHHII"
SHARD_HEADER_SIZE = struct.calcsize(SHARD_HEADER_FMT)


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    context_length: int = CONTEXT_LENGTH
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    hidden_dim: int = 1024
    ffn_dim: int = 2816
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = False


# ── RMSNorm ───────────────────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


# ── RoPE ──────────────────────────────────────────────────────────────────────


def build_rope_cache(
    seq_len: int, head_dim: int, theta: float = 10000.0, device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    angles = torch.outer(pos, freqs)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    B, n_heads, T, head_dim = x.shape
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ── GQA Attention ─────────────────────────────────────────────────────────────


class GQAAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.hidden_dim // config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads

        self.q_proj = nn.Linear(config.hidden_dim, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(y)


# ── SwiGLU FFN ────────────────────────────────────────────────────────────────


class SwiGLUFFN(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ── Transformer block ────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.attn = GQAAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.ffn = SwiGLUFFN(config)

        self.attn.o_proj._is_residual = True
        self.ffn.down_proj._is_residual = True

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Full model ────────────────────────────────────────────────────────────────


class LLaMAModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.drop = nn.Dropout(config.dropout)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "_is_residual"):
                std *= (2 * self.config.n_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._rope_cos is None or self._rope_cos.device != device:
            head_dim = self.config.hidden_dim // self.config.n_heads
            self._rope_cos, self._rope_sin = build_rope_cache(
                self.config.context_length, head_dim, self.config.rope_theta, device,
            )
        return self._rope_cos, self._rope_sin

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cos, sin = self._get_rope(input_ids.device)
        x = self.drop(self.tok_emb(input_ids))
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        return self.lm_head(x)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def enable_gradient_checkpointing(self) -> None:
        for layer in self.layers:
            layer._orig_forward = layer.forward

            def make_ckpt_forward(mod):
                def ckpt_forward(x, cos, sin):
                    return torch.utils.checkpoint.checkpoint(
                        mod._orig_forward, x, cos, sin, use_reentrant=False,
                    )
                return ckpt_forward

            layer.forward = make_ckpt_forward(layer)


# ── Streaming dataset ─────────────────────────────────────────────────────────


class ShardedTokenDataset(IterableDataset):
    """Streams pre-tokenized sequences from binary shards, either local or from HF.

    For HF datasets, downloads shard files on-the-fly and memory-maps them.
    Shuffles at the shard level each epoch for randomness without needing
    the full dataset in memory.
    """

    def __init__(
        self,
        shard_paths: list[str],
        context_length: int = CONTEXT_LENGTH,
        seed: int = 42,
        epoch: int = 0,
    ):
        self.shard_paths = shard_paths
        self.context_length = context_length
        self.seed = seed
        self.epoch = epoch

    def _read_shard(self, path: str) -> np.ndarray:
        with open(path, "rb") as f:
            header = f.read(SHARD_HEADER_SIZE)
            magic, version, ctx_len, num_seq, vocab = struct.unpack(SHARD_HEADER_FMT, header)
            assert magic == SHARD_MAGIC, f"Bad magic in {path}: {magic}"
            data = np.frombuffer(f.read(), dtype=np.uint16)
        return data.reshape(num_seq, ctx_len)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard_list = list(self.shard_paths)

        rng = np.random.default_rng(self.seed + self.epoch)
        rng.shuffle(shard_list)

        if worker_info is not None:
            per_worker = len(shard_list) // worker_info.num_workers
            remainder = len(shard_list) % worker_info.num_workers
            start = worker_info.id * per_worker + min(worker_info.id, remainder)
            end = start + per_worker + (1 if worker_info.id < remainder else 0)
            shard_list = shard_list[start:end]

        for shard_path in shard_list:
            data = self._read_shard(shard_path)
            indices = np.arange(data.shape[0])
            rng.shuffle(indices)
            for idx in indices:
                tokens = torch.from_numpy(data[idx].astype(np.int64))
                yield tokens


def resolve_shards(dataset_repo: str, cache_dir: str | None = None) -> list[str]:
    """Download shard files from a HuggingFace dataset repo and return local paths."""
    from huggingface_hub import snapshot_download

    logger.info(f"Downloading shards from HuggingFace: {dataset_repo}")
    local_dir = snapshot_download(
        repo_id=dataset_repo,
        repo_type="dataset",
        allow_patterns="data/train_*.bin",
        cache_dir=cache_dir,
    )
    data_dir = Path(local_dir) / "data"
    shard_paths = sorted(str(p) for p in data_dir.glob("train_*.bin"))
    logger.info(f"Found {len(shard_paths)} shards in {data_dir}")
    if not shard_paths:
        raise FileNotFoundError(f"No train_*.bin shards found in {data_dir}")
    return shard_paths


# ── Learning rate schedule ────────────────────────────────────────────────────


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ── Training config ───────────────────────────────────────────────────────────


@dataclass
class TrainConfig:
    dataset_repo: str = ""
    output_dir: str = "checkpoints"
    run_name: str = "curated-470m"

    # Batch sizing for RTX 4090 (24GB)
    # micro_batch=4 * seq_len=2048 = 8192 tokens per step
    # grad_accum=16 -> effective batch = 64 * 2048 = 131072 tokens (~128k)
    micro_batch_size: int = 4
    grad_accum_steps: int = 16
    effective_batch_tokens: int = field(init=False)

    # Optimizer
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000

    # Training budget
    total_tokens: int = 20_000_000_000
    max_steps: int = field(init=False)

    # Checkpointing
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 10

    # System
    seed: int = 42
    num_workers: int = 2
    dtype: str = "bfloat16"
    compile: bool = True
    gradient_checkpointing: bool = True

    # Wandb
    wandb_project: str = "curated-llm"
    wandb_enabled: bool = True

    # HF cache
    hf_cache_dir: str | None = None

    def __post_init__(self):
        tokens_per_step = self.micro_batch_size * CONTEXT_LENGTH * self.grad_accum_steps
        self.effective_batch_tokens = tokens_per_step
        self.max_steps = self.total_tokens // tokens_per_step


# ── Checkpoint management ─────────────────────────────────────────────────────


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    train_config: TrainConfig,
    model_config: ModelConfig,
    metrics: dict,
    path: Path,
) -> None:
    checkpoint = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_config": vars(train_config),
        "model_config": vars(model_config),
        "metrics": metrics,
    }
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.rename(path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, dict]:
    logger.info(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    return checkpoint["step"], checkpoint["epoch"], checkpoint.get("metrics", {})


def export_hf_model(model: nn.Module, config: ModelConfig, output_path: Path) -> None:
    """Export model weights and config in a format loadable by HuggingFace transformers."""
    output_path.mkdir(parents=True, exist_ok=True)

    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, output_path / "pytorch_model.bin")

    hf_config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": config.hidden_dim,
        "intermediate_size": config.ffn_dim,
        "num_hidden_layers": config.n_layers,
        "num_attention_heads": config.n_heads,
        "num_key_value_heads": config.n_kv_heads,
        "max_position_embeddings": config.context_length,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "torch_dtype": "bfloat16",
    }
    with open(output_path / "config.json", "w") as f:
        json.dump(hf_config, f, indent=2)

    logger.info(f"HF-format model exported to {output_path}")


# ── Evaluation ────────────────────────────────────────────────────────────────

NUM_EVAL_SHARDS = 2
NUM_EVAL_BATCHES = 50


@torch.no_grad()
def evaluate(
    model: nn.Module,
    eval_shard_paths: list[str],
    micro_batch_size: int,
    compute_dtype: torch.dtype,
    device: torch.device,
    model_cfg: ModelConfig,
) -> dict:
    """Run evaluation on held-out shards and return loss/perplexity."""
    model.eval()
    dataset = ShardedTokenDataset(eval_shard_paths, seed=0, epoch=0)
    loader = DataLoader(dataset, batch_size=micro_batch_size, drop_last=True)

    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        input_ids = batch[:, :-1].to(device, non_blocking=True)
        targets = batch[:, 1:].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, model_cfg.vocab_size), targets.reshape(-1),
            )
        total_loss += loss.item()
        n_batches += 1
        if n_batches >= NUM_EVAL_BATCHES:
            break

    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {"eval/loss": avg_loss, "eval/perplexity": ppl}


# ── Training loop ─────────────────────────────────────────────────────────────


def train(train_cfg: TrainConfig, model_cfg: ModelConfig, resume: bool = False):
    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("No CUDA device found — training will be extremely slow")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    compute_dtype = dtype_map[train_cfg.dtype]

    run_dir = Path(train_cfg.output_dir) / train_cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    with open(run_dir / "model_config.json", "w") as f:
        json.dump(vars(model_cfg), f, indent=2)
    with open(run_dir / "train_config.json", "w") as f:
        json.dump(vars(train_cfg), f, indent=2)

    # ── Model ──
    logger.info("Building model...")
    model = LLaMAModel(model_cfg).to(device)
    param_count = model.param_count()
    logger.info(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    if train_cfg.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        logger.info("Gradient checkpointing enabled")

    if train_cfg.compile and hasattr(torch, "compile"):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Optimizer ──
    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if p.dim() >= 2],
            "weight_decay": train_cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.dim() < 2],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=train_cfg.max_lr,
        betas=(train_cfg.beta1, train_cfg.beta2),
        fused=device.type == "cuda",
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(compute_dtype == torch.float16))

    # ── Resume ──
    start_step = 0
    start_epoch = 0
    if resume:
        ckpt_dir = run_dir / "checkpoints"
        ckpts = sorted(ckpt_dir.glob("step_*.pt"))
        if ckpts:
            start_step, start_epoch, _ = load_checkpoint(
                ckpts[-1], model, optimizer, scaler, device,
            )
            logger.info(f"Resumed from step {start_step}, epoch {start_epoch}")
        else:
            logger.warning("No checkpoints found, starting from scratch")

    # ── Data ──
    all_shard_paths = resolve_shards(train_cfg.dataset_repo, cache_dir=train_cfg.hf_cache_dir)

    first_shard = ShardedTokenDataset(all_shard_paths[:1])
    sample = next(iter(first_shard))
    assert sample.shape == (CONTEXT_LENGTH,), f"Expected shape ({CONTEXT_LENGTH},), got {sample.shape}"
    assert sample.max() < VOCAB_SIZE, f"Token ID {sample.max()} exceeds vocab size {VOCAB_SIZE}"
    logger.info(f"Shard validation passed: shape={sample.shape}, max_id={sample.max()}")

    eval_shard_paths = all_shard_paths[-NUM_EVAL_SHARDS:]
    shard_paths = all_shard_paths[:-NUM_EVAL_SHARDS]
    logger.info(f"Train shards: {len(shard_paths)}, eval shards: {len(eval_shard_paths)}")

    dataset = ShardedTokenDataset(shard_paths, seed=train_cfg.seed, epoch=start_epoch)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg.micro_batch_size,
        num_workers=train_cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Wandb ──
    if train_cfg.wandb_enabled:
        try:
            import wandb
            wandb.init(
                project=train_cfg.wandb_project,
                name=train_cfg.run_name,
                config={**vars(train_cfg), **vars(model_cfg), "param_count": param_count},
                resume="allow" if resume else None,
            )
        except ImportError:
            logger.warning("wandb not installed, disabling logging")
            train_cfg.wandb_enabled = False

    # ── Training ──
    logger.info(
        f"Training config:\n"
        f"  Total tokens:          {train_cfg.total_tokens / 1e9:.1f}B\n"
        f"  Max steps:             {train_cfg.max_steps:,}\n"
        f"  Micro batch size:      {train_cfg.micro_batch_size}\n"
        f"  Gradient accumulation: {train_cfg.grad_accum_steps}\n"
        f"  Effective batch:       {train_cfg.effective_batch_tokens:,} tokens\n"
        f"  Max LR:                {train_cfg.max_lr}\n"
        f"  Warmup steps:          {train_cfg.warmup_steps}\n"
        f"  Compute dtype:         {train_cfg.dtype}\n"
        f"  Compile:               {train_cfg.compile}\n"
        f"  Grad checkpointing:    {train_cfg.gradient_checkpointing}"
    )

    model.train()
    step = start_step
    epoch = start_epoch
    tokens_seen = step * train_cfg.effective_batch_tokens
    running_loss = 0.0
    step_t0 = time.time()

    data_iter = iter(dataloader)

    while step < train_cfg.max_steps:
        lr = get_lr(step, train_cfg.warmup_steps, train_cfg.max_steps, train_cfg.max_lr, train_cfg.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        micro_loss_sum = 0.0

        for micro_step in range(train_cfg.grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                logger.info(f"Epoch {epoch} starting (reshuffling shards)")
                dataset = ShardedTokenDataset(shard_paths, seed=train_cfg.seed, epoch=epoch)
                dataloader = DataLoader(
                    dataset,
                    batch_size=train_cfg.micro_batch_size,
                    num_workers=train_cfg.num_workers,
                    pin_memory=True,
                    drop_last=True,
                )
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch[:, :-1].to(device, non_blocking=True)
            targets = batch[:, 1:].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=compute_dtype):
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, model_cfg.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-1,
                )
                loss = loss / train_cfg.grad_accum_steps

            scaler.scale(loss).backward()
            micro_loss_sum += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        tokens_seen += train_cfg.effective_batch_tokens
        running_loss += micro_loss_sum

        # ── Logging ──
        if step % train_cfg.log_every_steps == 0:
            step_dt = time.time() - step_t0
            avg_loss = running_loss / train_cfg.log_every_steps
            tokens_per_sec = (train_cfg.effective_batch_tokens * train_cfg.log_every_steps) / step_dt
            ppl = math.exp(min(avg_loss, 20))
            eta_hours = (train_cfg.max_steps - step) / max(train_cfg.log_every_steps / step_dt, 1e-9) / 3600

            logger.info(
                f"step {step:>7,d}/{train_cfg.max_steps:,d} | "
                f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                f"tok/s {tokens_per_sec:,.0f} | "
                f"tokens {tokens_seen / 1e9:.3f}B | "
                f"ETA {eta_hours:.1f}h"
            )

            if train_cfg.wandb_enabled:
                import wandb
                wandb.log(
                    {
                        "train/loss": avg_loss,
                        "train/perplexity": ppl,
                        "train/lr": lr,
                        "train/grad_norm": float(grad_norm),
                        "train/tokens_per_sec": tokens_per_sec,
                        "train/tokens_seen": tokens_seen,
                        "train/epoch": epoch,
                        "train/step": step,
                    },
                    step=step,
                )

            running_loss = 0.0
            step_t0 = time.time()

        # ── Evaluation ──
        if step % train_cfg.eval_every_steps == 0:
            eval_metrics = evaluate(
                model, eval_shard_paths, train_cfg.micro_batch_size,
                compute_dtype, device, model_cfg,
            )
            logger.info(
                f"step {step:>7,d} | eval_loss {eval_metrics['eval/loss']:.4f} | "
                f"eval_ppl {eval_metrics['eval/perplexity']:.1f}"
            )
            if train_cfg.wandb_enabled:
                import wandb
                wandb.log(eval_metrics, step=step)

        # ── Checkpointing ──
        if step % train_cfg.save_every_steps == 0 or step == train_cfg.max_steps:
            ckpt_path = run_dir / "checkpoints" / f"step_{step:07d}.pt"
            metrics = {
                "tokens_seen": tokens_seen,
                "last_loss": micro_loss_sum,
            }
            save_checkpoint(
                model, optimizer, scaler, step, epoch,
                train_cfg, model_cfg, metrics, ckpt_path,
            )

            old_ckpts = sorted((run_dir / "checkpoints").glob("step_*.pt"))
            keep_every = 10000
            for ckpt in old_ckpts[:-3]:
                ckpt_step = int(ckpt.stem.split("_")[1])
                if ckpt_step % keep_every != 0:
                    ckpt.unlink()
                    logger.info(f"Removed old checkpoint: {ckpt.name}")

    # ── Final export ──
    logger.info("Training complete!")
    logger.info(f"Total tokens seen: {tokens_seen / 1e9:.2f}B")
    logger.info(f"Total steps: {step}")

    export_path = run_dir / "hf_model"
    export_hf_model(model, model_cfg, export_path)

    if train_cfg.wandb_enabled:
        import wandb
        wandb.finish()

    return run_dir


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain 470M LLaMA-style model on pre-tokenized data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-repo", type=str, required=True, help="HuggingFace dataset repo with TKDS shards")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Base output directory")
    parser.add_argument("--run-name", type=str, default="curated-470m", help="Run name for this experiment")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    parser.add_argument("--micro-batch-size", type=int, default=4, help="Micro batch size (per gradient accumulation step)")
    parser.add_argument("--grad-accum-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--max-lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5, help="Minimum learning rate (end of cosine decay)")
    parser.add_argument("--warmup-steps", type=int, default=2000, help="Warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--total-tokens", type=int, default=15_000_000_000, help="Total training tokens")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps (for short test runs)")

    parser.add_argument("--save-every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
    parser.add_argument("--no-grad-checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--wandb-project", type=str, default="curated-llm", help="Wandb project name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")

    parser.add_argument("--hf-cache-dir", type=str, default=None, help="HuggingFace cache directory")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    model_cfg = ModelConfig()
    train_cfg = TrainConfig(
        dataset_repo=args.dataset_repo,
        output_dir=args.output_dir,
        run_name=args.run_name,
        micro_batch_size=args.micro_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        total_tokens=args.total_tokens,
        save_every_steps=args.save_every,
        log_every_steps=args.log_every,
        eval_every_steps=args.eval_every,
        dtype=args.dtype,
        compile=not args.no_compile,
        gradient_checkpointing=not args.no_grad_checkpoint,
        num_workers=args.num_workers,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_enabled=not args.no_wandb,
        hf_cache_dir=args.hf_cache_dir,
    )

    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps

    train(train_cfg, model_cfg, resume=args.resume)


if __name__ == "__main__":
    main()
