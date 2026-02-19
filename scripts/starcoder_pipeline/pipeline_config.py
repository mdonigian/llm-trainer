"""
Shared configuration for the StarCoder curation pipeline.

All constants, content type definitions, target distributions, and helper
functions used across pipeline stages.

This pipeline uses a single multi-task UniXcoder-base model with three heads:
  - Code Quality (1-5): regression
  - Structured Data Relevance (0-3): regression
  - Content Type (9 classes): classification

The structured data relevance score is the primary signal for filtering,
since the overall goal is training a model optimized for structured output
(JSON generation, function calling, schema compliance).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Content type definitions (must match models/starcoderdata-classifier/label_config.json)
# ---------------------------------------------------------------------------

CONTENT_TYPES = [
    "library",       # 0 — reusable module/package code
    "application",   # 1 — application/business logic
    "test",          # 2 — unit tests, integration tests
    "config",        # 3 — configuration, setup, build scripts
    "tutorial",      # 4 — educational example, demo
    "data",          # 5 — data file/fixture disguised as code
    "generated",     # 6 — auto-generated code (swagger, protobuf, migrations)
    "script",        # 7 — one-off script, CLI tool, automation
    "other",         # 8 — doesn't fit any category
]

NUM_CONTENT_TYPES = len(CONTENT_TYPES)

CONTENT_TYPE_TO_IDX = {ct: i for i, ct in enumerate(CONTENT_TYPES)}

CONTENT_TYPE_DISPLAY = {ct: ct.title() for ct in CONTENT_TYPES}

# ---------------------------------------------------------------------------
# Quality levels
# ---------------------------------------------------------------------------

QUALITY_RANGE = (1, 5)
QUALITY_NAMES = {
    1: "Broken/gibberish",
    2: "Functional but poor",
    3: "Decent",
    4: "Good",
    5: "Excellent",
}

# ---------------------------------------------------------------------------
# Structured data relevance levels
# ---------------------------------------------------------------------------

STRUCTURED_DATA_RANGE = (0, 3)
STRUCTURED_DATA_NAMES = {
    0: "None",
    1: "Minor",
    2: "Significant",
    3: "Primary focus",
}

# ---------------------------------------------------------------------------
# Content type grouping for target distribution
#
# Groups high-value content types vs. low-value ones for sampling.
# "high_value" types are library/application/script — real functional code.
# "test" gets its own group since test code teaches structured patterns.
# "low_value" includes config, data, generated, tutorial, other —
#   some is useful but heavily downsampled.
# ---------------------------------------------------------------------------

CONTENT_GROUP_MAP: dict[str, list[str]] = {
    "library":     ["library"],
    "application": ["application"],
    "script":      ["script"],
    "test":        ["test"],
    "low_value":   ["config", "data", "generated", "tutorial", "other"],
}

CONTENT_GROUP_DISPLAY: dict[str, str] = {
    "library":     "Library/Package",
    "application": "Application",
    "script":      "Script/CLI",
    "test":        "Test Code",
    "low_value":   "Config/Data/Generated/Other",
}

# Target token percentages per content group (must sum to 1.0)
CONTENT_GROUP_TARGET_PCT: dict[str, float] = {
    "library":     0.35,
    "application": 0.25,
    "script":      0.15,
    "test":        0.15,
    "low_value":   0.10,
}

# Priority tiers for sampling order (lower = filled first)
CONTENT_GROUP_PRIORITY: dict[str, int] = {
    "library":     0,
    "application": 0,
    "test":        1,
    "script":      1,
    "low_value":   2,
}

# ---------------------------------------------------------------------------
# Structured data relevance bins (used for complexity-like distribution targets)
# ---------------------------------------------------------------------------

SD_BINS = [
    ("SD0", 0.0, 0.5),    # no structured data patterns
    ("SD1", 0.5, 1.5),    # minor
    ("SD2", 1.5, 2.5),    # significant
    ("SD3", 2.5, 3.5),    # primary focus
]

# Target: heavily boost high-relevance code
SD_TARGET_PCT: dict[str, float] = {
    "SD0": 0.10,
    "SD1": 0.20,
    "SD2": 0.35,
    "SD3": 0.35,
}

# ---------------------------------------------------------------------------
# Quality thresholds
# Key decision from project: use quality as coarse two-stage filter:
#   - Hard floor: drop predicted quality <= 1 (broken/gibberish)
#   - Soft boost: boost relevance score for predicted quality >= 4
# Don't try to be precise with quality — lean on structured data relevance.
# ---------------------------------------------------------------------------

QUALITY_HARD_FLOOR = 1.5     # drop anything scoring <= this (catches level 1)
QUALITY_SOFT_BOOST_FLOOR = 3.5  # boost relevance for quality >= this

# ---------------------------------------------------------------------------
# Compression ratio pre-filter
# Key decision from project: zlib compression ratio < 0.10 catches extreme
# repetition (SQL migrations, generated configs) without false-flagging.
# Applied BEFORE BERT inference to save GPU time.
# ---------------------------------------------------------------------------

COMPRESSION_RATIO_FLOOR = 0.10

# ---------------------------------------------------------------------------
# Token length filters
# ---------------------------------------------------------------------------

DEFAULT_MIN_TOKENS = 20
DEFAULT_MAX_TOKENS = 100_000

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
DEFAULT_TOTAL_TARGET_TOKENS = 11_000_000_000  # 11B — middle of 10-12B range

DEFAULT_CLASSIFIER_MODEL = "models/starcoderdata-classifier"
DEFAULT_MAX_LENGTH = 512
DEFAULT_SHARD_SIZE = 1_000_000
DEFAULT_OUTPUT_BASE = "/workspace/starcoder-curation"

# Batch size defaults tuned for RTX 5090 (32GB GDDR7, 1.79 TB/s, 680 Tensor Cores).
# UniXcoder-base is 125M params (~250MB bf16), so VRAM is dominated by activations.
# At seq_len=512, bf16: ~4096 comfortably fits in 32GB with headroom for torch.compile.
# At seq_len=128 (most code files): 8192+ is feasible but 4096 avoids OOM on outliers.
DEFAULT_BATCH_SIZE = 4096

DEFAULT_DATASET = "bigcode/starcoderdata"

RECOMMENDED_LANGUAGES = [
    "python",
    "javascript",
    "typescript",
    "java",
    "c",
    "cpp",
    "go",
    "rust",
    "ruby",
    "php",
    "shell",
    "scala",
    "kotlin",
    "swift",
    "sql",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sd_bin(score: float) -> str:
    """Map a structured data relevance score to its bin label."""
    for label, lo, hi in SD_BINS:
        if lo <= score < hi:
            return label
    if score >= 3.0:
        return "SD3"
    return "SD0"


def sd_bin_vec(scores: np.ndarray) -> np.ndarray:
    """Vectorised structured data binning."""
    bins = np.full(len(scores), "SD0", dtype="U3")
    for label, lo, hi in SD_BINS:
        mask = (scores >= lo) & (scores < hi)
        bins[mask] = label
    bins[scores >= 3.0] = "SD3"
    return bins


def assign_content_group(content_type: str) -> str:
    """Map a content type string to its content group."""
    for group, types in CONTENT_GROUP_MAP.items():
        if content_type in types:
            return group
    return "low_value"


def assign_content_group_vec(content_types: list[str]) -> list[str]:
    """Vectorised content group assignment."""
    type_to_group = {}
    for group, types in CONTENT_GROUP_MAP.items():
        for t in types:
            type_to_group[t] = group
    return [type_to_group.get(ct, "low_value") for ct in content_types]


def compute_relevance_score(
    structured_data: float,
    quality: float,
) -> float:
    """Composite relevance score for priority-based sampling.

    Structured data relevance is the primary signal (the whole project
    is about structured output). Quality provides a soft boost.
    """
    base = structured_data / 3.0  # normalize to [0, 1]
    quality_boost = 0.1 if quality >= QUALITY_SOFT_BOOST_FLOOR else 0.0
    return base + quality_boost


def compute_relevance_score_batch(
    structured_data: np.ndarray,
    quality: np.ndarray,
) -> np.ndarray:
    """Vectorised relevance score."""
    base = structured_data / 3.0
    quality_boost = np.where(quality >= QUALITY_SOFT_BOOST_FLOOR, 0.1, 0.0)
    return base + quality_boost
