#!/usr/bin/env python3
"""
Prepare pre-tokenized training dataset.

Downloads all 7 data sources from HuggingFace, tokenizes with the GPT-NeoX
tokenizer (50304 vocab), packs into 2048-token sequences, enforces the target
mix ratios, shuffles globally, and uploads the final dataset to HuggingFace.

The output is a HuggingFace dataset of pre-tokenized sequences ready for
training — no tokenization needed on the training cluster.

Data sources and target mix (20B tokens total):
  1. FineWeb-Edu (curated)        — 4.30B tokens (21.5%)  [post-dedup pipeline yield]
  2. FineWeb-Edu (random/uncur.)  — 2.00B tokens (10%)    [uncurated baseline diversity]
  3. StarCoderData (curated)      — 3.90B tokens (19.5%)  [post-dedup pipeline yield]
  4. FineMath-4+                  — 3.20B tokens (16%)    [+200M from Glaive realloc]
  5. Wikipedia (structured)       — 1.50B tokens  (7.5%)  [JSON infoboxes, sections, metadata]
  6. Wikipedia (plain overlap)    — 0.50B tokens  (2.5%)  [dual-representation overlap]
  7. peS2o (CS/math/ML papers)    — 2.20B tokens (11%)   [+200M from Glaive realloc]
  8. StackExchange (technical)    — 1.00B tokens  (5%)    [high-score Q&A]
  9. UltraChat                    — 0.40B tokens  (2%)    [general instruction diversity]

Usage:
  # Full run (download, tokenize, upload)
  python prepare_tokenized_dataset.py \
      --output-dir /workspace/tokenized \
      --upload --repo-id youruser/curated-20B-tokenized

  # Just tokenize, no upload
  python prepare_tokenized_dataset.py --output-dir /workspace/tokenized

  # Resume from a partial run (skips already-tokenized sources)
  python prepare_tokenized_dataset.py --output-dir /workspace/tokenized --resume

  # Override a source with a local/custom HF repo
  python prepare_tokenized_dataset.py --output-dir /workspace/tokenized \
      --fineweb-repo youruser/fineweb-edu-curated \
      --starcoder-repo youruser/starcoderdata-curated
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ── Tokenizer ────────────────────────────────────────────────────────────────

TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"
VOCAB_SIZE = 50_304
CONTEXT_LENGTH = 2048

# ── Data source definitions ──────────────────────────────────────────────────

EOS_SENTINEL = "<|endoftext|>"


@dataclass
class DataSource:
    """A data source with its HuggingFace location, text column, and token budget."""

    name: str
    hf_repo: str
    hf_subset: str | None
    text_column: str
    target_tokens: int
    target_pct: float
    filters: dict | None = None
    streaming: bool = True
    text_fn: str | None = None  # name of custom text extraction function
    no_feature_cast: bool = False  # bypass HF feature schema casting during streaming
    data_files: str | None = None  # load raw data files directly (bypasses loading scripts)
    split: str = "train"


# ── Structured Wikipedia text extraction ────────────────────────────────────
# wikimedia/structured-wikipedia has pre-parsed articles with infoboxes,
# sections, and metadata as nested JSON. We serialize each article into a
# text representation that interleaves prose with structured data so the
# model learns implicit text-to-structure mappings.


def _serialize_infobox(infobox: dict) -> str:
    """Serialize a Wikipedia infobox into a JSON-like text representation."""
    fields = {}
    for part in (infobox.get("has_parts") or []):
        if part.get("type") == "section":
            for field in (part.get("has_parts") or []):
                if field.get("type") == "field" and field.get("name") and field.get("value"):
                    fields[field["name"]] = field["value"]
        elif part.get("type") == "field" and part.get("name") and part.get("value"):
            fields[part["name"]] = part["value"]
    if not fields:
        return ""
    return json.dumps(fields, ensure_ascii=False)


def _serialize_sections(sections: list[dict]) -> str:
    """Extract section text from structured Wikipedia sections."""
    parts = []
    for section in sections:
        name = section.get("name", "")
        text_parts = []
        for part in (section.get("has_parts") or []):
            if part.get("type") == "paragraph" and part.get("value"):
                text_parts.append(part["value"])
            elif part.get("type") == "section":
                sub_text = _serialize_sections([part])
                if sub_text:
                    text_parts.append(sub_text)
        if text_parts:
            section_text = "\n".join(text_parts)
            if name and name != "Abstract":
                parts.append(f"\n## {name}\n{section_text}")
            else:
                parts.append(section_text)
    return "\n".join(parts)


def extract_structured_wikipedia(doc: dict) -> str:
    """Build training text from a structured Wikipedia article.

    Format:
      # Article Title
      Short description

      <infobox>
      {"field": "value", ...}
      </infobox>

      Article sections as prose...

    The <infobox> tags give the model a clear signal of structured data
    embedded within natural language context.
    """
    parts = []

    title = doc.get("name", "")
    if title:
        parts.append(f"# {title}")

    description = doc.get("description", "")
    if description:
        parts.append(description)

    infoboxes = doc.get("infoboxes", [])
    for ib in (infoboxes or []):
        serialized = _serialize_infobox(ib)
        if serialized:
            parts.append(f"\n<infobox>\n{serialized}\n</infobox>")

    abstract = doc.get("abstract", "")
    if abstract:
        parts.append(f"\n{abstract}")

    sections = doc.get("sections", [])
    if sections:
        section_text = _serialize_sections(sections)
        if section_text:
            parts.append(section_text)

    text = "\n".join(parts).strip()
    if len(text) < 100:
        return ""
    return text



def extract_ultrachat(doc: dict) -> str:
    """Build training text from an UltraChat conversation.

    The dataset has a `messages` column with a list of {role, content} dicts.
    We serialize the conversation into a simple multi-turn format.
    """
    messages = doc.get("messages", [])
    if not messages:
        return ""
    parts = []
    for msg in messages:
        role = msg.get("role", "").upper()
        content = msg.get("content", "")
        if role and content:
            parts.append(f"{role}: {content}")
    text = "\n\n".join(parts)
    if len(text) < 100:
        return ""
    return text


def extract_stackexchange(doc: dict) -> str:
    """Build training text from a StackExchange Q&A pair.

    Takes the question text and the highest-scored answer, formatted as a
    Q&A pair. Filters to answers with pm_score >= 3 for quality.
    """
    question = doc.get("question", "")
    if not question:
        return ""
    answers = doc.get("answers", [])
    if not answers:
        return ""
    best = max(answers, key=lambda a: a.get("pm_score", 0))
    if best.get("pm_score", 0) < 3:
        return ""
    answer_text = best.get("text", "")
    if not answer_text:
        return ""
    text = f"QUESTION: {question}\n\nANSWER: {answer_text}"
    if len(text) < 100:
        return ""
    return text


CUSTOM_TEXT_FNS = {
    "structured_wikipedia": extract_structured_wikipedia,
    "ultrachat": extract_ultrachat,
    "stackexchange": extract_stackexchange,
}


DEFAULT_SOURCES: list[DataSource] = [
    DataSource(
        name="fineweb_edu_curated",
        hf_repo="HuggingFaceFW/fineweb-edu",
        hf_subset="CC-MAIN-2024-10",
        text_column="text",
        target_tokens=4_300_000_000,
        target_pct=0.215,
    ),
    DataSource(
        name="fineweb_edu_random",
        hf_repo="HuggingFaceFW/fineweb-edu",
        hf_subset="CC-MAIN-2024-18",
        text_column="text",
        target_tokens=2_000_000_000,
        target_pct=0.10,
    ),
    DataSource(
        name="starcoderdata",
        hf_repo="bigcode/starcoderdata",
        hf_subset=None,
        text_column="content",
        target_tokens=3_900_000_000,
        target_pct=0.195,
    ),
    DataSource(
        name="finemath",
        hf_repo="HuggingFaceTB/finemath",
        hf_subset="finemath-4plus",
        text_column="text",
        target_tokens=3_200_000_000,
        target_pct=0.16,
    ),
    DataSource(
        name="wiki_structured",
        hf_repo="wikimedia/structured-wikipedia",
        hf_subset="20240916.en",
        text_column="",
        text_fn="structured_wikipedia",
        target_tokens=1_500_000_000,
        target_pct=0.075,
        no_feature_cast=True,
    ),
    DataSource(
        name="wiki_plain",
        hf_repo="wikimedia/wikipedia",
        hf_subset="20231101.en",
        text_column="text",
        target_tokens=500_000_000,
        target_pct=0.025,
    ),
    DataSource(
        name="pes2o",
        hf_repo="allenai/peS2o",
        hf_subset="v2",
        text_column="text",
        target_tokens=2_200_000_000,
        target_pct=0.11,
        data_files="hf://datasets/allenai/peS2o/data/v2/train-*.json.gz",
    ),
    DataSource(
        name="stackexchange",
        hf_repo="HuggingFaceH4/stack-exchange-preferences",
        hf_subset=None,
        text_column="",
        text_fn="stackexchange",
        target_tokens=1_000_000_000,
        target_pct=0.05,
    ),
    DataSource(
        name="ultrachat",
        hf_repo="HuggingFaceH4/ultrachat_200k",
        hf_subset=None,
        text_column="",
        text_fn="ultrachat",
        target_tokens=400_000_000,
        target_pct=0.02,
        split="train_sft",
    ),
]

TOTAL_TARGET_TOKENS = 20_000_000_000
TARGET_SEQUENCES = TOTAL_TARGET_TOKENS // CONTEXT_LENGTH  # ~9.8M sequences

# ── Binary shard format ──────────────────────────────────────────────────────
# Each shard is a flat binary file of uint16 token IDs (GPT-NeoX vocab fits in
# uint16). A shard contains N packed sequences of exactly CONTEXT_LENGTH tokens.
# No padding, no separators between sequences — the training loop reads
# contiguous chunks of CONTEXT_LENGTH tokens.
#
# Shard header (16 bytes):
#   magic (4B): b"TKDS"
#   version (2B): 1
#   context_length (2B): 2048
#   num_sequences (4B): number of packed sequences
#   vocab_size (4B): 50304

SHARD_MAGIC = b"TKDS"
SHARD_VERSION = 1
SHARD_HEADER_FMT = "<4sHHII"  # magic, version, ctx_len, num_seq, vocab_size
SHARD_HEADER_SIZE = struct.calcsize(SHARD_HEADER_FMT)
SEQUENCES_PER_SHARD = 8192  # ~32M tokens per shard, ~64MB per shard file


def write_shard(path: Path, sequences: np.ndarray) -> None:
    """Write a shard of packed token sequences to disk."""
    assert sequences.ndim == 2 and sequences.shape[1] == CONTEXT_LENGTH
    num_seq = sequences.shape[0]
    header = struct.pack(
        SHARD_HEADER_FMT,
        SHARD_MAGIC, SHARD_VERSION, CONTEXT_LENGTH, num_seq, VOCAB_SIZE,
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(sequences.astype(np.uint16).tobytes())


def read_shard(path: Path) -> np.ndarray:
    """Read a shard back into a (num_sequences, CONTEXT_LENGTH) uint16 array."""
    with open(path, "rb") as f:
        header_bytes = f.read(SHARD_HEADER_SIZE)
        magic, version, ctx_len, num_seq, vocab = struct.unpack(
            SHARD_HEADER_FMT, header_bytes,
        )
        assert magic == SHARD_MAGIC, f"Bad magic: {magic}"
        assert version == SHARD_VERSION, f"Bad version: {version}"
        data = np.frombuffer(f.read(), dtype=np.uint16)
    return data.reshape(num_seq, ctx_len)


# ── Tokenization + packing ──────────────────────────────────────────────────

TOKENIZE_BATCH_SIZE = 1024  # docs per batch — tokenizer uses Rust parallelism across all cores


class TokenPacker:
    """Packs pre-tokenized documents into fixed-length sequences.

    Documents are concatenated with EOS tokens between them. When a
    concatenation reaches CONTEXT_LENGTH, a sequence is emitted. Partial
    documents carry over to the next sequence.

    This is the standard "packing" approach used by GPT/LLaMA pretraining.
    """

    def __init__(self, eos_id: int, context_length: int = CONTEXT_LENGTH):
        self.context_length = context_length
        self.eos_id = eos_id
        self.buffer: list[int] = []
        self.sequences_emitted = 0
        self.tokens_processed = 0
        self.documents_processed = 0

    def add_tokens(self, token_ids: list[int]) -> list[list[int]]:
        """Add pre-tokenized document tokens (with EOS already appended)."""
        self.buffer.extend(token_ids)
        self.tokens_processed += len(token_ids)
        self.documents_processed += 1

        completed = []
        while len(self.buffer) >= self.context_length:
            seq = self.buffer[: self.context_length]
            self.buffer = self.buffer[self.context_length :]
            completed.append(seq)
            self.sequences_emitted += 1
        return completed

    def flush(self) -> list[int] | None:
        """Flush remaining buffer. Returns None if buffer is too short."""
        if len(self.buffer) >= self.context_length // 2:
            padded = self.buffer + [self.eos_id] * (
                self.context_length - len(self.buffer)
            )
            self.sequences_emitted += 1
            self.buffer = []
            return padded
        self.buffer = []
        return None


def tokenize_source(
    source: DataSource,
    tokenizer,
    output_dir: Path,
    max_tokens: int | None = None,
    resume: bool = False,
) -> dict:
    """Download, tokenize, and pack a single data source into shards.

    Uses batch tokenization to saturate all CPU cores — the HuggingFace
    tokenizer's __call__ with a list of strings dispatches to Rust-level
    parallelism internally.

    Returns a stats dict with token counts, sequence counts, timing, etc.
    """
    from datasets import load_dataset

    source_dir = output_dir / "sources" / source.name
    source_dir.mkdir(parents=True, exist_ok=True)
    stats_path = source_dir / "stats.json"

    if resume and stats_path.exists():
        logger.info(f"[{source.name}] Resuming — already tokenized, skipping")
        with open(stats_path) as f:
            return json.load(f)

    target = max_tokens or source.target_tokens
    logger.info(
        f"[{source.name}] Tokenizing up to {target / 1e9:.2f}B tokens "
        f"from {source.hf_repo}"
        f"{f' ({source.hf_subset})' if source.hf_subset else ''}"
    )

    if source.no_feature_cast:
        from datasets import load_dataset_builder

        builder = load_dataset_builder(source.hf_repo, source.hf_subset)
        builder.info.features = None
        ds = builder.as_streaming_dataset(split=source.split)
    else:
        load_kwargs: dict = {
            "path": source.hf_repo,
            "split": source.split,
            "streaming": source.streaming,
        }
        if source.hf_subset:
            load_kwargs["name"] = source.hf_subset
        if source.data_files:
            load_kwargs["path"] = "json"
            load_kwargs["data_files"] = source.data_files
            load_kwargs.pop("name", None)
        ds = load_dataset(**load_kwargs)

    eos_id = tokenizer.eos_token_id
    packer = TokenPacker(eos_id)
    shard_idx = 0
    shard_buffer: list[list[int]] = []
    t0 = time.time()

    def flush_shard():
        nonlocal shard_idx, shard_buffer
        if not shard_buffer:
            return
        arr = np.array(shard_buffer, dtype=np.uint16)
        shard_path = source_dir / f"shard_{shard_idx:05d}.bin"
        write_shard(shard_path, arr)
        shard_idx += 1
        shard_buffer = []

    pbar = tqdm(
        desc=f"[{source.name}] tokens",
        unit=" tok",
        unit_scale=True,
        total=target,
    )

    text_fn = CUSTOM_TEXT_FNS.get(source.text_fn) if source.text_fn else None
    text_batch: list[str] = []
    done = False

    def process_batch():
        """Batch-tokenize accumulated texts and pack into sequences."""
        nonlocal done
        if not text_batch:
            return
        batch_encodings = tokenizer(
            text_batch, add_special_tokens=False, return_attention_mask=False,
        )
        for token_ids in batch_encodings["input_ids"]:
            token_ids.append(eos_id)
            completed = packer.add_tokens(token_ids)
            for seq in completed:
                shard_buffer.append(seq)
                if len(shard_buffer) >= SEQUENCES_PER_SHARD:
                    flush_shard()
            if packer.tokens_processed >= target:
                done = True
                break
        pbar.update(packer.tokens_processed - pbar.n)
        text_batch.clear()

    for doc in ds:
        if text_fn:
            text = text_fn(doc)
        else:
            text = doc.get(source.text_column, "")
        if not text or not text.strip():
            continue

        if source.filters:
            skip = False
            for key, allowed_values in source.filters.items():
                if doc.get(key) not in allowed_values:
                    skip = True
                    break
            if skip:
                continue

        text_batch.append(text)
        if len(text_batch) >= TOKENIZE_BATCH_SIZE:
            process_batch()
            if done:
                break

    process_batch()

    last_seq = packer.flush()
    if last_seq:
        shard_buffer.append(last_seq)
    flush_shard()

    pbar.close()
    elapsed = time.time() - t0

    stats = {
        "source": source.name,
        "hf_repo": source.hf_repo,
        "hf_subset": source.hf_subset,
        "tokens_processed": packer.tokens_processed,
        "sequences_emitted": packer.sequences_emitted,
        "documents_processed": packer.documents_processed,
        "num_shards": shard_idx,
        "target_tokens": target,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(packer.tokens_processed / max(elapsed, 1)),
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(
        f"[{source.name}] Done: {packer.tokens_processed / 1e9:.2f}B tokens, "
        f"{packer.sequences_emitted} sequences, {shard_idx} shards "
        f"in {elapsed:.0f}s"
    )
    return stats


# ── Shuffle + merge ──────────────────────────────────────────────────────────


def shuffle_and_merge(
    output_dir: Path,
    sources: list[DataSource],
    seed: int = 42,
) -> dict:
    """Shuffle all source shards together and write final training shards.

    Uses a memory-mapped intermediate file so the full dataset never needs
    to fit in RAM. Only the shuffled index permutation is held in memory
    (~74MB for 9.8M sequences). Disk I/O is sequential for both the copy
    and the final shard writes.
    """
    logger.info("Counting sequences across all source shards...")

    shard_manifest: list[tuple[Path, int]] = []
    source_stats = {}

    for source in sources:
        source_dir = output_dir / "sources" / source.name
        shard_paths = sorted(source_dir.glob("shard_*.bin"))
        if not shard_paths:
            logger.warning(f"[{source.name}] No shards found, skipping")
            continue

        source_seqs = 0
        for sp in shard_paths:
            with open(sp, "rb") as f:
                header = f.read(SHARD_HEADER_SIZE)
            _, _, _, num_seq, _ = struct.unpack(SHARD_HEADER_FMT, header)
            shard_manifest.append((sp, num_seq))
            source_seqs += num_seq

        source_stats[source.name] = source_seqs
        logger.info(f"[{source.name}] {source_seqs:,} sequences across {len(shard_paths)} shards")

    total_seqs = sum(n for _, n in shard_manifest)
    total_tokens = total_seqs * CONTEXT_LENGTH
    logger.info(
        f"Total: {total_seqs:,} sequences, {total_tokens / 1e9:.2f}B tokens"
    )

    mmap_path = output_dir / "_shuffle_tmp.bin"
    row_bytes = CONTEXT_LENGTH * 2  # uint16
    logger.info(
        f"Creating memory-mapped file: {mmap_path} "
        f"({total_seqs * row_bytes / 1e9:.1f} GB)"
    )
    mmap_data = np.memmap(
        mmap_path, dtype=np.uint16, mode="w+",
        shape=(total_seqs, CONTEXT_LENGTH),
    )

    offset = 0
    for sp, num_seq in tqdm(shard_manifest, desc="Copying shards to memmap"):
        data = read_shard(sp)
        mmap_data[offset : offset + num_seq] = data
        offset += num_seq
        del data
    mmap_data.flush()
    logger.info("All shards copied to memmap")

    logger.info(f"Generating shuffled index permutation (seed={seed})...")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_seqs)

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    num_final_shards = (total_seqs + SEQUENCES_PER_SHARD - 1) // SEQUENCES_PER_SHARD
    logger.info(f"Writing {num_final_shards} final shards...")

    for i in tqdm(range(num_final_shards), desc="Writing final shards"):
        start = i * SEQUENCES_PER_SHARD
        end = min(start + SEQUENCES_PER_SHARD, total_seqs)
        indices = perm[start:end]
        shard_data = mmap_data[indices]
        shard_path = final_dir / f"train_{i:05d}.bin"
        write_shard(shard_path, shard_data)

    del mmap_data
    mmap_path.unlink(missing_ok=True)
    logger.info("Cleaned up memmap temp file")

    merge_stats = {
        "total_sequences": int(total_seqs),
        "total_tokens": int(total_tokens),
        "num_final_shards": num_final_shards,
        "seed": seed,
        "source_sequences": {k: int(v) for k, v in source_stats.items()},
    }
    return merge_stats


# ── HuggingFace upload ───────────────────────────────────────────────────────


def upload_to_hub(
    output_dir: Path,
    repo_id: str,
    private: bool = True,
) -> None:
    """Upload final tokenized shards to HuggingFace Hub as a dataset repo."""
    from huggingface_hub import HfApi, create_repo

    logger.info(f"Uploading to HuggingFace: {repo_id}")

    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )

    api = HfApi()
    final_dir = output_dir / "final"

    api.upload_folder(
        folder_path=str(final_dir),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="data",
        commit_message="Upload pre-tokenized training shards",
    )

    stats_path = output_dir / "dataset_stats.json"
    if stats_path.exists():
        api.upload_file(
            path_or_fileobj=str(stats_path),
            path_in_repo="dataset_stats.json",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload dataset statistics",
        )

    readme_path = output_dir / "README.md"
    if readme_path.exists():
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Upload dataset card",
        )

    logger.info(f"Upload complete: https://huggingface.co/datasets/{repo_id}")


# ── Dataset card ─────────────────────────────────────────────────────────────


def generate_dataset_card(output_dir: Path, stats: dict) -> None:
    """Generate a HuggingFace dataset card (README.md)."""

    source_table = ""
    for name, seq_count in stats.get("source_sequences", {}).items():
        tokens = seq_count * CONTEXT_LENGTH
        pct = tokens / max(stats["total_tokens"], 1) * 100
        source_table += f"| {name} | {tokens / 1e9:.2f}B | {pct:.1f}% | {seq_count:,} |\n"

    card = f"""---
language:
- en
- code
license: apache-2.0
task_categories:
- text-generation
tags:
- pretokenized
- pretraining
- curated
- structured-output
---

# Curated Pre-Tokenized Training Dataset

Pre-tokenized training data for a 500M parameter LLaMA-style model optimized
for structured output tasks (JSON generation, function calling, schema compliance).

## Format

Binary shards of packed uint16 token sequences. Each shard has a 16-byte header
followed by contiguous sequences of {CONTEXT_LENGTH} tokens.

- **Tokenizer:** `{TOKENIZER_NAME}` (vocab size: {VOCAB_SIZE:,})
- **Context length:** {CONTEXT_LENGTH}
- **Total tokens:** {stats['total_tokens'] / 1e9:.2f}B
- **Total sequences:** {stats['total_sequences']:,}
- **Shards:** {stats['num_final_shards']}
- **Shard format:** TKDS v1 (see `prepare_tokenized_dataset.py` for reader)

## Data Mix

| Source | Tokens | % | Sequences |
|--------|--------|---|-----------|
{source_table}

## Loading

```python
from prepare_tokenized_dataset import read_shard
import numpy as np

data = read_shard("data/train_00000.bin")  # (N, {CONTEXT_LENGTH}) uint16
```

## Shard Header Format

```
Offset  Size  Field
0       4B    Magic: b"TKDS"
4       2B    Version: 1
6       2B    Context length: {CONTEXT_LENGTH}
8       4B    Number of sequences (uint32)
12      4B    Vocab size: {VOCAB_SIZE}
16+     ...   Token data (uint16, row-major)
```
"""
    readme_path = output_dir / "README.md"
    with open(readme_path, "w") as f:
        f.write(card)
    logger.info(f"Dataset card written to {readme_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def build_sources(args: argparse.Namespace) -> list[DataSource]:
    """Build the list of data sources, applying any CLI overrides."""
    sources = []
    for s in DEFAULT_SOURCES:
        override_attr = f"{s.name.replace('-', '_')}_repo"
        if hasattr(args, override_attr) and getattr(args, override_attr):
            s = DataSource(
                name=s.name,
                hf_repo=getattr(args, override_attr),
                hf_subset=None,
                text_column=s.text_column,
                target_tokens=s.target_tokens,
                target_pct=s.target_pct,
                filters=s.filters,
                streaming=s.streaming,
            )
        sources.append(s)
    return sources


def main():
    parser = argparse.ArgumentParser(
        description="Download, tokenize, and pack training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Base output directory for shards and stats",
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload final dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--repo-id", type=str, default=None,
        help="HuggingFace repo ID for upload (e.g. youruser/curated-20B-tokenized)",
    )
    parser.add_argument(
        "--private", action="store_true", default=True,
        help="Make the HuggingFace repo private (default: True)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip sources that already have stats.json (resume partial run)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for shuffling (default: 42)",
    )
    parser.add_argument(
        "--skip-shuffle", action="store_true",
        help="Skip the global shuffle step (for debugging)",
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", default=None,
        help="Only process these sources (by name). Default: all",
    )

    for s in DEFAULT_SOURCES:
        parser.add_argument(
            f"--{s.name.replace('_', '-')}-repo", type=str, default=None,
            help=f"Override HF repo for {s.name} (default: {s.hf_repo})",
        )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Silence noisy HTTP loggers from huggingface/datasets/urllib3/fsspec
    for noisy in [
        "urllib3", "requests", "huggingface_hub", "fsspec",
        "datasets", "filelock", "httpx", "httpcore",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    if args.upload and not args.repo_id:
        parser.error("--repo-id is required when --upload is set")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    logger.info(f"Loading tokenizer: {TOKENIZER_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    assert tokenizer.vocab_size <= VOCAB_SIZE, (
        f"Tokenizer vocab {tokenizer.vocab_size} exceeds expected {VOCAB_SIZE}"
    )
    logger.info(
        f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}, "
        f"eos_token_id={tokenizer.eos_token_id}"
    )

    sources = build_sources(args)
    if args.sources:
        sources = [s for s in sources if s.name in args.sources]
        logger.info(f"Processing subset: {[s.name for s in sources]}")

    all_stats = {}
    for source in sources:
        stats = tokenize_source(
            source, tokenizer, args.output_dir, resume=args.resume,
        )
        all_stats[source.name] = stats

    with open(args.output_dir / "source_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    if not args.skip_shuffle:
        all_sources = build_sources(args)
        merge_stats = shuffle_and_merge(args.output_dir, all_sources, seed=args.seed)

        with open(args.output_dir / "dataset_stats.json", "w") as f:
            json.dump(merge_stats, f, indent=2)

        generate_dataset_card(args.output_dir, merge_stats)

        logger.info(
            f"\nDataset ready: {merge_stats['total_tokens'] / 1e9:.2f}B tokens, "
            f"{merge_stats['total_sequences']:,} sequences, "
            f"{merge_stats['num_final_shards']} shards"
        )

        mix_report = "\nActual mix ratios:\n"
        for name, seq_count in merge_stats["source_sequences"].items():
            tokens = seq_count * CONTEXT_LENGTH
            pct = tokens / max(merge_stats["total_tokens"], 1) * 100
            mix_report += f"  {name:20s}: {tokens / 1e9:.2f}B tokens ({pct:.1f}%)\n"
        logger.info(mix_report)

    if args.upload:
        upload_to_hub(args.output_dir, args.repo_id, private=args.private)

    logger.info("Done!")


if __name__ == "__main__":
    main()
