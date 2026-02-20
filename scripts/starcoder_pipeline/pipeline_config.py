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
# Language slice definitions
#
# Each slice has its own token budget and filtering strategy.
# This reflects the project's insight that different code types need
# different curation approaches:
#   - Schema languages are inherently structured — light filter only
#   - General-purpose languages need the relevance classifier (≥ 2)
#   - Jupyter notebooks are already structured data
#   - GitHub issues provide natural language about code structure
# ---------------------------------------------------------------------------

from dataclasses import dataclass, field


@dataclass
class LanguageSlice:
    """Defines a language slice with its token budget and filtering strategy."""
    name: str
    languages: list[str]
    target_tokens: int
    strategy: str            # "light_filter", "relevance_filter", "passthrough", "keyword_filter"
    min_relevance: float = 0.0
    min_quality: float = 1.5
    description: str = ""


# Token targets are per-slice ceilings, not a hard sum. The combined total
# (~3.7B) slightly exceeds the 3-4B StarCoderData budget in project_summary.md;
# actual yield depends on available data per slice and dedup shrinkage.

LANGUAGE_SLICES: list[LanguageSlice] = [
    LanguageSlice(
        name="schema_languages",
        languages=["json", "yaml", "xml", "sql", "protocol-buffer", "thrift"],
        target_tokens=800_000_000,
        strategy="light_filter",
        min_relevance=0.0,
        min_quality=1.5,
        description="Schema/data/query languages — inherently structured, light filter only",
    ),
    LanguageSlice(
        name="typescript",
        languages=["typescript"],
        target_tokens=600_000_000,
        strategy="relevance_filter",
        min_relevance=2.0,
        min_quality=1.5,
        description="TypeScript — strong type system, filter by structured data relevance ≥ 2",
    ),
    LanguageSlice(
        name="python",
        languages=["python"],
        target_tokens=600_000_000,
        strategy="relevance_filter",
        min_relevance=2.0,
        min_quality=1.5,
        description="Python — filter by structured data relevance ≥ 2",
    ),
    LanguageSlice(
        name="rust_go_java",
        languages=["rust", "go", "java"],
        target_tokens=600_000_000,
        strategy="relevance_filter",
        min_relevance=2.0,
        min_quality=1.5,
        description="Rust/Go/Java — strongly typed, filter by structured data relevance ≥ 2",
    ),
    LanguageSlice(
        name="jupyter",
        languages=["jupyter-scripts-dedup-filtered"],
        target_tokens=400_000_000,
        strategy="relevance_filter",
        min_relevance=2.0,
        min_quality=1.5,
        description="Jupyter notebooks — filter by structured data relevance ≥ 2",
    ),
    LanguageSlice(
        name="github_issues",
        languages=["github-issues-filtered-structured"],
        target_tokens=500_000_000,
        strategy="keyword_filter",
        min_relevance=0.0,
        min_quality=1.5,
        description="GitHub issues (technical) — keyword filter for structured data topics",
    ),
]

LANGUAGE_SLICE_MAP: dict[str, LanguageSlice] = {s.name: s for s in LANGUAGE_SLICES}

# All languages across all slices (for download)
ALL_SLICE_LANGUAGES: list[str] = []
for _s in LANGUAGE_SLICES:
    ALL_SLICE_LANGUAGES.extend(_s.languages)

# Languages that need the classifier (relevance_filter strategy only).
# Schema languages and github issues use cheap heuristics instead.
CLASSIFIER_LANGUAGES: list[str] = []
for _s in LANGUAGE_SLICES:
    if _s.strategy == "relevance_filter":
        CLASSIFIER_LANGUAGES.extend(_s.languages)

# Keywords for the GitHub issues keyword filter.
# Deliberately narrow — we want issues discussing structured data patterns,
# not generic bug reports. Stems like "serializ" match serialize/serialization.
GITHUB_ISSUES_KEYWORDS = [
    "json", "schema", "api endpoint", "serializ", "deserializ",
    "protobuf", "grpc", "graphql", "openapi", "swagger",
    "rest api", "restful",
    "json schema", "json parse", "json format",
    "yaml", "xml schema", "xsd",
    "struct", "interface",
    "marshal", "unmarshal",
    "payload", "request body", "response body",
    "type definition", "type system",
    "function call", "function signature",
]

# ---------------------------------------------------------------------------
# Content type grouping (secondary axis within each language slice)
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

# ---------------------------------------------------------------------------
# Structured data relevance bins (for reporting)
# ---------------------------------------------------------------------------

SD_BINS = [
    ("SD0", 0.0, 0.5),    # no structured data patterns
    ("SD1", 0.5, 1.5),    # minor
    ("SD2", 1.5, 2.5),    # significant
    ("SD3", 2.5, 3.5),    # primary focus
]

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
DEFAULT_TOTAL_TARGET_TOKENS = 3_500_000_000  # 3.5B — sum of all language slice budgets

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

RECOMMENDED_LANGUAGES = CLASSIFIER_LANGUAGES

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


def resolve_language_to_slice(lang: str) -> str | None:
    """Map a language string to its slice name, or None if not in any slice."""
    lang_lower = lang.lower()
    for s in LANGUAGE_SLICES:
        if lang_lower in [l.lower() for l in s.languages]:
            return s.name
    return None


def text_matches_keywords(text: str, keywords: list[str] | None = None) -> bool:
    """Check if text contains any of the GitHub issues keywords."""
    if keywords is None:
        keywords = GITHUB_ISSUES_KEYWORDS
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)
