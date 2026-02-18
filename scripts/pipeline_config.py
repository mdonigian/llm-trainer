"""
Shared configuration for the FineWeb-Edu curation pipeline.

All constants, label definitions, group mappings, target distributions,
and helper functions used across pipeline stages.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Label definitions (must match models/fineweb-edu-classifier/label_config.json)
# ---------------------------------------------------------------------------

LABEL_NAMES = [
    "mathematics_statistics",                  # 0
    "computer_science_software_engineering",    # 1
    "machine_learning_ai",                     # 2
    "physical_sciences",                       # 3
    "life_sciences_biology",                   # 4
    "medicine_health",                         # 5
    "engineering_technology",                  # 6
    "business_economics",                      # 7
    "law_government",                          # 8
    "social_sciences",                         # 9
    "history_geography",                       # 10
    "philosophy_ethics",                       # 11
    "education_pedagogy",                      # 12
    "language_writing",                        # 13
    "arts_humanities",                         # 14
    "environmental_science_energy",            # 15
    "personal_finance_practical_life",         # 16
]

NUM_LABELS = len(LABEL_NAMES)

LABEL_DISPLAY_NAMES = [n.replace("_", " ").title() for n in LABEL_NAMES]

# ---------------------------------------------------------------------------
# Group mapping: 17 fine-grained labels -> 11 target groups
# Each group maps to one or more label indices. A document belongs to a group
# if ANY of its constituent labels score >= TOPIC_THRESHOLD.
# ---------------------------------------------------------------------------

GROUP_MAP: dict[str, list[int]] = {
    "mathematics":        [0],
    "computer_science":   [1],
    "ml_ai":              [2],
    "physical_sciences":  [3],
    "life_sciences":      [4],
    "engineering_tech":   [6],
    "environmental":      [15],
    "medicine_health":    [5],
    "business_economics": [7],
    "law_government":     [8],
    "general":            [9, 10, 11, 12, 13, 14, 16],
}

GROUP_DISPLAY_NAMES: dict[str, str] = {
    "mathematics":        "Mathematics",
    "computer_science":   "Computer Science",
    "ml_ai":              "ML/AI",
    "physical_sciences":  "Physical Sciences",
    "life_sciences":      "Life Sciences",
    "engineering_tech":   "Engineering/Tech",
    "environmental":      "Environmental Sci",
    "medicine_health":    "Medicine/Health",
    "business_economics": "Business/Economics",
    "law_government":     "Law/Government",
    "general":            "General Knowledge",
}

# Target token percentages per group (must sum to 1.0)
GROUP_TARGET_PCT: dict[str, float] = {
    "mathematics":        0.07,
    "computer_science":   0.08,
    "ml_ai":              0.05,
    "physical_sciences":  0.04,
    "life_sciences":      0.03,
    "engineering_tech":   0.05,
    "environmental":      0.02,
    "medicine_health":    0.04,
    "business_economics": 0.04,
    "law_government":     0.03,
    "general":            0.55,
}

# STEM-Core label indices — used for relevance scoring boost
STEM_CORE_INDICES = [0, 1, 2]  # math, cs, ml/ai

# Priority tiers for sampling order (higher priority groups are filled first)
GROUP_PRIORITY: dict[str, int] = {
    "mathematics":        0,   # HIGH — boost
    "computer_science":   0,
    "ml_ai":              0,
    "physical_sciences":  1,   # MEDIUM
    "life_sciences":      1,
    "engineering_tech":   1,
    "environmental":      1,
    "medicine_health":    2,   # LOW — downsample
    "business_economics": 2,
    "law_government":     2,
    "general":            3,   # FILL — last
}

# ---------------------------------------------------------------------------
# Complexity bins and targets
# ---------------------------------------------------------------------------

COMPLEXITY_BINS = [
    ("L1", 1.0, 1.75),    # basic
    ("L2", 1.75, 2.5),    # intermediate
    ("L3", 2.5, 3.25),    # advanced
    ("L4", 3.25, 4.0),    # expert
]

COMPLEXITY_TARGET_PCT: dict[str, float] = {
    "L1": 0.10,
    "L2": 0.20,
    "L3": 0.40,
    "L4": 0.30,
}

# ---------------------------------------------------------------------------
# Thresholds (defaults — calibrated via Stage 0 validation)
# ---------------------------------------------------------------------------

DEFAULT_TOPIC_THRESHOLD = 0.3
DEFAULT_AMBIGUITY_FLOOR = 0.3
DEFAULT_MIN_TOKENS = 50
DEFAULT_MAX_TOKENS = 100_000

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
DEFAULT_TOTAL_TARGET_TOKENS = 23_500_000_000  # 23.5B — middle of 22-25B range

DEFAULT_TOPIC_MODEL = "models/fineweb-edu-classifier"
DEFAULT_COMPLEXITY_MODEL = "models/complexity-classifier"
DEFAULT_BATCH_SIZE = 2048  # A100 80GB: both models use <2GB VRAM, batch activations ~8GB at 2048
DEFAULT_MAX_LENGTH = 512
DEFAULT_SHARD_SIZE = 1_000_000
DEFAULT_OUTPUT_BASE = "/workspace/fineweb-curation"

# CC dumps recommended by HuggingFace for quality
RECOMMENDED_CC_DUMPS = [
    "CC-MAIN-2023-50",
    "CC-MAIN-2024-10",
    "CC-MAIN-2024-18",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def complexity_bin(score: float) -> str:
    """Map a complexity score to its bin label."""
    for label, lo, hi in COMPLEXITY_BINS:
        if lo <= score < hi:
            return label
    if score >= 4.0:
        return "L4"
    return "L1"


def complexity_bin_vec(scores: np.ndarray) -> np.ndarray:
    """Vectorised complexity binning."""
    bins = np.full(len(scores), "L1", dtype="U2")
    for label, lo, hi in COMPLEXITY_BINS:
        mask = (scores >= lo) & (scores < hi)
        bins[mask] = label
    bins[scores >= 4.0] = "L4"
    return bins


def assign_groups(
    topic_scores: np.ndarray,
    threshold: float = DEFAULT_TOPIC_THRESHOLD,
) -> list[str]:
    """Return the list of target groups a document belongs to.

    Args:
        topic_scores: length-17 array of sigmoid scores.
        threshold: minimum score for a label to be considered active.
    """
    groups = []
    for group_name, label_indices in GROUP_MAP.items():
        if any(topic_scores[i] >= threshold for i in label_indices):
            groups.append(group_name)
    return groups


def compute_relevance_score(
    topic_scores: np.ndarray,
    complexity: float,
    threshold: float = DEFAULT_TOPIC_THRESHOLD,
) -> float:
    """Composite relevance score for priority-based sampling.

    Components:
      - STEM-core boost: max sigmoid among math/cs/ml_ai
      - Multi-label bonus: +0.05 per active topic, capped at 0.15
      - Complexity contribution: complexity * 0.1
    """
    stem_core_score = max(topic_scores[i] for i in STEM_CORE_INDICES)
    num_active = sum(1 for s in topic_scores if s >= threshold)
    multi_label_bonus = min(num_active * 0.05, 0.15)
    return stem_core_score + multi_label_bonus + (complexity * 0.1)


def compute_relevance_score_batch(
    topic_scores: np.ndarray,
    complexity: np.ndarray,
    threshold: float = DEFAULT_TOPIC_THRESHOLD,
) -> np.ndarray:
    """Vectorised relevance score for an (N, 17) array of topic scores."""
    stem_core = topic_scores[:, STEM_CORE_INDICES].max(axis=1)
    num_active = (topic_scores >= threshold).sum(axis=1)
    multi_label_bonus = np.minimum(num_active * 0.05, 0.15)
    return stem_core + multi_label_bonus + (complexity * 0.1)
