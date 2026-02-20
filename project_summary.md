# Project Summary: Curated 500M LLM for Structured Output

## What We're Building

A 500M parameter language model trained from scratch on 20B carefully curated tokens, optimized specifically for structured output tasks (JSON generation, function calling, schema compliance, classification). The core thesis is that frontier-model-guided data curation during pretraining creates capabilities that supervised fine-tuning alone cannot replicate — specifically, better generalization to unseen schemas and novel function signatures.

---

## How We Got Here

### Session 1 — Initial Vision (Feb 12)

Started with a broad goal: use a high-quality large LLM to improve training data so a 0.5-1B model could punch above its weight. We identified the core data sources (FineWeb-Edu, The Stack, OpenWebMath, Wikipedia, ArXiv, StackExchange, UltraChat), worked out token-to-parameter ratios (Chinchilla optimal is ~20 tokens/parameter, but modern models train far beyond that), and discussed what mix ratios would work.

**Key decision:** Build BERT-based classifiers to score and filter training data along two dimensions — topic and reasoning complexity. This is the "frontier-model-guided curation" approach: use Claude to generate training labels, train fast ModernBERT classifiers on those labels, then run the classifiers across billions of documents.

**Key decision:** Designed a 17-class topic taxonomy for FineWeb-Edu covering STEM, applied sciences, humanities, and practical knowledge domains.

### Session 2 — Classifiers Built, Project Questioned (Feb 17-18)

Trained both classifiers to completion:

- **Topic classifier:** ModernBERT-base, 17-class multi-label (BCE/sigmoid), ~0.85 F1
- **Complexity classifier:** ModernBERT-base, regression on 1-4 scale, ~0.75 Spearman correlation

The complexity classifier started as a 5-level classification problem, but level 5 had too few examples and the boundaries between adjacent levels were fuzzy. We pivoted to regression, which worked significantly better.

**Critical pivot moment:** After building the classifiers, we questioned whether the whole project made sense. The core tension: xLAM-1B achieved 79% on BFCL with just 60k SFT examples on top of DeepSeek-Coder-1.3B (2T tokens pretraining). Why would curated pretraining on 50-100B tokens beat that? The answer we arrived at: SFT teaches format-following, but pretraining shapes internal representations. If structural patterns (JSON, schemas, type systems, nested hierarchies) are deeply embedded during pretraining, the model should generalize better to *unseen* schemas — not just reproduce trained formats. That's the testable hypothesis.

**Key decision:** Proceed, but reframe as a research contribution. The question isn't "can we beat Qwen?" — it's "does pretraining curation measurably improve structured output beyond what SFT alone achieves?"

### Session 3 — Strategy Refinement (Feb 18)

Researched the competitive landscape in depth. Found that DCLM, SIEVE, and ProX focus on general-purpose data curation for broad benchmarks, while xLAM and similar focus on specialized SFT on existing base models. Nobody has combined multi-dimensional curation (topic × complexity) specifically targeting structured output with the full pretraining pipeline. That's our novelty.

**Key decision:** Train from scratch rather than fine-tune an existing base model. This isolates the variable we're testing (does pretraining data composition matter?) and avoids confounding with whatever the base model already learned.

**Key decision:** The topic classifier is multi-label (sigmoid), not single-label (softmax). Documents can belong to multiple topics. Multi-label documents (e.g., scoring 0.75 ML/AI + 0.68 CS + 0.55 Math) are actually our most valuable training examples — they teach cross-domain structured reasoning.

Analyzed a topic × complexity heatmap from a classified FineWeb-Edu sample. Found that STEM-core topics (CS, Math, ML/AI) are naturally rare (~5% combined) while education/health/business dominate. This confirmed the need for active rebalancing during filtering.

### Session 4 — Feasibility & Baseline Strategy (Feb 18)

Confronted the token count reality: production 1B models train on 10-18T tokens. Our 50B token budget is 100× less than Qwen 2.5's 18T. This places us firmly in the academic research zone, comparable to DCLM ablation studies and curriculum learning papers.

**Key decision:** Cut from 100B to 50B tokens to halve training cost while keeping the research signal intact. Later revised down further — see Session 8.

**Key decision:** Use Pythia as the primary baseline. It's perfect: trained on The Pile (uncurated), full checkpoint history available, all data and training code public, designed for reproducibility research.

Defined three outcome scenarios:
1. **Curation doesn't help** (negative but publishable)
2. **Curation creates specialized advantage** (most likely, validates thesis)
3. **Broad superiority** (unlikely but high-impact)

### Session 5 — FineWeb-Edu Pipeline Spec (Feb 18-19)

Wrote the detailed requirements specification for the FineWeb-Edu curation pipeline. This is the largest component of the training mix.

**Critical correction:** The original spec treated the topic classifier as single-label (argmax + softmax confidence). We caught that the model is actually multi-label (BCE/sigmoid), which changes everything:
- Store full 17-dim sigmoid vector per document, not just top label
- Filter using thresholds on individual dimensions, not confidence of top class
- The 0.5 confidence floor from the original spec was wrong (calibrated for softmax, not sigmoid)

**Key decision:** Added Stage 0 (validation run on 50k docs) to calibrate sigmoid thresholds before committing to a 6-hour classification run. This catches miscalibration, degenerate distributions, and threshold problems in 2 minutes.

**Key decision:** Download specific CC dumps (CC-MAIN-2023-50, 2024-10, 2024-18) rather than streaming the full 100BT sample. Smaller, restartable, and HF recommends these as highest quality.

### Session 6 — Benchmarking Strategy (Feb 19)

Wrote the evaluation and benchmarking specification. Designed a three-tier benchmark suite:
- **Tier 1 (primary):** BFCL v3, custom JSON schema compliance suite (500 examples), structured extraction, classification with structured output
- **Tier 2 (diagnostic):** Standard lm-eval-harness benchmarks (MMLU, HellaSwag, ARC, GSM8k, etc.)
- **Tier 3 (code):** HumanEval, MBPP

**Key decision:** Build the custom JSON Schema Compliance benchmark BEFORE training anything. No existing benchmark specifically stress-tests 1B models on diverse JSON schema compliance. We need 500 examples across 6 categories (flat objects, nested, arrays/enums, conditional types, real-world schemas, adversarial edge cases).

**Key decision:** Evaluate at three stages — base model, post-SFT, and post-DPO — for both our model and Pythia. This disentangles pretraining quality from post-training quality. The headline metric is the structured output delta post-SFT.

### Session 7 — Code Classifiers & Heuristic Filters (Feb 19, current)

Evaluated the StarCoderData classifiers:
- **Structured Data Relevance:** Strong (0.807 Spearman, good per-level F1s). This is the most important code classifier.
- **Content Type:** Solid (87.5% accuracy, high-volume classes all F1 ≥ 0.79).
- **Code Quality:** Weak (0.575 Spearman). Level 2 and Level 5 are broken. Caused by fuzzy label boundaries and severe class imbalance (Level 5 only 1.2% of training data).

**Key decision:** Don't retrain the quality classifier. Use it as a coarse two-stage filter (hard floor at predicted ≤ 1, soft boost for predicted ≥ 4) and lean on the stronger structured relevance score for ranking.

**Key decision:** Use zlib compression ratio as a pre-filter for repetitive code boilerplate. Threshold < 0.10 catches extreme repetition (SQL migrations, generated configs) without false-flagging real code. This runs before BERT inference to save GPU time.

Researched prior art on compression-based filtering. Found that the Gopher/MassiveText repetition filters (used by FineWeb, RefinedWeb, etc.) are the established approach for web text, but they decompose into ~10 individual heuristic checks. Compression ratio is a cleaner single signal for code specifically. FineWeb-Edu already has these filters applied, so compression ratio is only needed for StarCoderData.

### Session 8 — Model Size & Budget Revision (Feb 19)

Revisited the training cost estimates from first principles using `6 * N * D` FLOP calculations. Discovered the original $300-500 estimate for 1B/50B was significantly too low — actual on-demand cost would be ~$3,500 on RunPod (A100 80GB at ~$1.64/hr, ~45% MFU). The original estimate likely assumed spot pricing or used incorrect per-token cost approximations.

Evaluated the tradeoff between 1B and 500M parameters. At 1B, even 30B tokens would cost ~$2,100 on-demand. At 500M, the full range of token budgets becomes affordable.

**Key decision:** Downsize to 500M parameters and 20B tokens. 20B/500M = 40 tokens/parameter, well above Chinchilla-optimal and in line with modern research training runs that push further beyond optimal compute allocation.

**Key decision:** Switch baseline from Pythia-1B to Pythia-410M. Closer architectural match, same family, same full checkpoint history and reproducibility guarantees. Compare at compute-matched and full-training (300B token) checkpoints.

Rationale for 500M over 1B: a capacity-constrained model that still generalizes to unseen schemas after curated pretraining is a *stronger* result than a 1B model doing the same. The smaller model makes the curation signal easier to detect — less capacity to brute-force patterns from noisy data.

### Session 9 — Model Architecture (Feb 19)

Specified the model architecture. Evaluated three families: LLaMA-style (RoPE, RMSNorm, SwiGLU, GQA), GPT-NeoX style (Pythia's architecture — learned pos embeddings, LayerNorm, GeLU), and SSMs (Mamba). Surveyed comparable models at this scale: Pythia-410M, Qwen2.5-0.5B, SmolLM-360M.

**Key decision:** LLaMA-style architecture. The Pythia comparison is about data curation, not architecture — using a different architecture is acceptable and arguably strengthens the result. The entire post-training ecosystem (SFT frameworks, DPO, vLLM, lm-eval-harness) assumes LLaMA-family models. Fighting the tooling at 500M is not where effort should go.

**Key decision:** GPT-NeoX tokenizer (50,304 vocab) for Pythia comparability. A shared tokenizer means token counts are directly comparable, and the NeoX tokenizer has reasonable code coverage. Training a custom tokenizer would optimize compression for our mix but adds complexity and makes the Pythia comparison less clean.

**Key decision:** 2048 context length for pretraining. Most structured output tasks (JSON schemas, function signatures) fit comfortably in 2048 tokens. RoPE supports extending context during SFT if needed. Saves compute vs. 4096.

**Key decision:** GQA with 8 KV heads / 16 query heads (2:1 ratio). Saves ~15% memory during inference with minimal quality loss at this scale.

### Session 10 — Pre-Tokenized Dataset Pipeline (Feb 20)

Wrote the tokenization and dataset assembly script (`scripts/prepare_tokenized_dataset.py`). This script downloads all data sources from HuggingFace, tokenizes with the GPT-NeoX tokenizer, packs into 2048-token sequences, shuffles globally across all sources, and uploads the result to HuggingFace as pre-tokenized binary shards. The goal is to do all tokenization once on a single machine so the training cluster loads pre-tokenized data with zero overhead.

**Shard format:** Custom binary format (TKDS v1) — flat uint16 token IDs packed into fixed 2048-token sequences, 8192 sequences per shard (~64MB). 16-byte header with magic, version, context length, sequence count, and vocab size. The training loop reads contiguous uint16 arrays with no parsing.

**Key decision:** Use `wikimedia/structured-wikipedia` instead of plain Wikipedia. This dataset has pre-parsed articles with infoboxes, tables, and sections as structured JSON. Each article is serialized with its infobox as JSON wrapped in `<infobox>` tags, interleaved with the article prose. This gives the model implicit text-to-structure training signal — it sees real-world entities described in both natural language and structured JSON representations in the same document.

**Key decision:** Split Wikipedia into two sub-sources: structured (80%, 960M tokens) and plain (20%, 240M tokens). Both datasets cover all of English Wikipedia, so the model encounters overlapping articles in both structured and unstructured form. The structured version includes infoboxes serialized as JSON; the plain version is raw article text. This overlap creates implicit paired examples of the same entities in both formats without needing explicit alignment.

**Key decision:** Document packing (standard GPT/LLaMA approach). Documents are concatenated with EOS tokens between them. When the buffer reaches 2048 tokens, a sequence is emitted. Partial documents carry over to the next sequence. No document-level padding waste.

**Key decision:** Revised the token mix to reflect actual post-dedup pipeline yields and redistribute budgets. FineWeb-Edu split into curated (4.3B, post-dedup yield) and random uncurated (2.0B, for baseline diversity). Glaive function calling dropped — the 400M token budget was reallocated to FineMath (+200M) and peS2o (+200M) where it has more impact. UltraChat (400M tokens, 2%) added for general instruction diversity. FineMath and peS2o got the largest increases to strengthen STEM/math/CS coverage.

Script features: `--resume` for restarting partial runs, `--sources` for processing individual sources, per-source `--*-repo` overrides to swap in curated HF repos for FineWeb-Edu and StarCoderData after curation pipelines run, `--skip-shuffle` for debugging.

---

## Current State

### Completed

| Artifact | Status | Location |
|----------|--------|----------|
| Topic classifier (17-class multi-label) | Trained, ~0.85 F1 | `models/topic-classifier` |
| Complexity classifier (1-4 regression) | Trained, ~0.75 Spearman | `models/complexity-classifier` |
| Code quality classifier (1-5 regression) | Trained, 0.575 Spearman | `models/code-quality-classifier` |
| Structured data relevance classifier (0-3) | Trained, 0.807 Spearman | `models/structured-relevance-classifier` |
| Content type classifier (9-class) | Trained, 87.5% accuracy | `models/content-type-classifier` |
| FineWeb-Edu curation pipeline spec | Written | `spec-fineweb-curation-pipeline.md` |
| Benchmarking strategy spec | Written | `spec-benchmarking-pythia-baseline.md` |
| Dataset availability guide | Written | `dataset-availability-50B-mix.md` |
| Code quality labeling prompt | Written | `prompt-code-quality-labeling.md` |
| Topic × complexity heatmap analysis | Done | (in conversation history) |
| Pre-tokenized dataset pipeline | Written | `scripts/prepare_tokenized_dataset.py` |

### Not Yet Started

| Task | Priority | Dependencies |
|------|----------|-------------|
| Run FineWeb-Edu curation pipeline on RunPod | HIGH | Spec complete, classifiers ready |
| Run StarCoderData curation pipeline | HIGH | Classifiers ready, need to write spec |
| Build custom JSON Schema Compliance benchmark | HIGH | None — do before training |
| Run tokenization pipeline (all sources → pre-tokenized shards → HF upload) | HIGH | Curation pipelines complete |
| Download Pythia-410M checkpoints (compute-matched + final) | MEDIUM | None |
| Validate eval harness on Pythia-410M (verify published numbers) | MEDIUM | Pythia download |
| Design SFT dataset for structured output | HIGH | Benchmark design |
| Pretrain 500M model on curated 20B tokens | HIGH | Tokenized dataset uploaded |
| Run SFT → DPO → Evaluation pipeline | HIGH | Pretraining complete |

---

## The 20B Token Mix

| Source | Tokens | % | Notes |
|--------|--------|---|-------|
| FineWeb-Edu curated (post-dedup) | 4.3B | 21.5% | Completed yield |
| FineWeb-Edu random (uncurated) | 2.0B | 10% | Uncurated baseline diversity |
| StarCoderData curated | 3.9B | 19.5% | Completed yield |
| FineMath-4+ | 3.2B | 16% | +200M from Glaive realloc |
| Structured Wikipedia | 1.5B | 7.5% | JSON infoboxes, sections, metadata |
| Wikipedia EN (plain) | 0.5B | 2.5% | Dual-representation overlap |
| peS2o (CS/math/ML) | 2.2B | 11% | +200M from Glaive realloc |
| StackExchange (technical) | 1.0B | 5% | High-score Q&A |
| UltraChat | 0.4B | 2% | General instruction diversity |
| **Total** | **~20B** | | **40 tokens/param** |

All sources tokenized with GPT-NeoX tokenizer, packed into 2048-token sequences, shuffled globally, and uploaded as pre-tokenized binary shards. No tokenization on the training cluster.

---

## Model Architecture

~470M parameter LLaMA-style transformer:

| Parameter | Value |
|-----------|-------|
| Layers | 24 |
| Hidden dimension | 1024 |
| Query heads | 16 |
| KV heads | 8 (GQA, 2:1 ratio) |
| FFN dimension | 2816 (SwiGLU) |
| Context length | 2048 |
| Vocab size | 50,304 (GPT-NeoX tokenizer) |
| Normalization | RMSNorm |
| Position encoding | RoPE |
| Activation | SwiGLU |
| Total parameters | ~470M |

Training framework: LitGPT (best balance of simplicity and feature completeness for a single research run).

---

## Key Architectural Decisions

1. **Multi-label topic classification throughout** — never flatten to single-label. Store full 17-dim sigmoid vectors. Multi-label documents are prioritized, not collapsed.

2. **Regression for complexity/quality** — classification boundaries are too fuzzy for human-labeled ordinal scales. Regression with Spearman correlation as the metric works better.

3. **Train from scratch, not fine-tune** — isolates the pretraining curation variable. Avoids confounding with base model's existing knowledge.

4. **Pythia-410M as baseline, not Qwen** — Pythia answers the scientific question (does curation help?). Qwen/Llama contextualizes absolute performance but can't be meaningfully compared since we can't control their post-training.

5. **500M parameters, 20B tokens** — 40 tokens/parameter, well above Chinchilla-optimal. Smaller model makes curation signal easier to detect and produces a stronger result if the hypothesis holds.

6. **Compression ratio as code pre-filter** — cheap, fast, catches repetitive boilerplate that passes BERT quality checks. Apply before GPU inference.

7. **Build benchmarks before training** — the custom JSON Schema Compliance suite defines what success looks like. Lock it before training starts.

8. **Stage 0 validation run** — 50k docs, 2 minutes, calibrates sigmoid thresholds. Catches problems before committing to 6-hour classification runs.

9. **LLaMA-style architecture, not GPT-NeoX** — ecosystem compatibility (SFT, DPO, vLLM, lm-eval-harness) outweighs the minor advantage of matching Pythia's architecture exactly. The comparison is about data, not architecture.

10. **GPT-NeoX tokenizer** — shared with Pythia for direct token-count comparability. Reasonable code coverage without the complexity of training a custom tokenizer.

11. **Structured Wikipedia over plain Wikipedia** — `wikimedia/structured-wikipedia` includes pre-parsed infoboxes serialized as JSON within article prose. Split 80/20 structured vs. plain, with both covering the same articles. The model sees the same entities in both natural language and structured JSON form — implicit text-to-structure training signal without explicit alignment.

12. **Pre-tokenize everything before training** — tokenize all sources once, pack into 2048-token sequences, upload binary shards to HuggingFace. Training cluster loads uint16 arrays directly with zero tokenization overhead.

---

## Estimated Total Cost

| Component | On-Demand | Spot (est.) |
|-----------|-----------|-------------|
| FineWeb-Edu curation (RunPod A100, 12-16hr) | $20-35 | $20-35 |
| StarCoderData curation (RunPod A100, 8-12hr) | $15-25 | $15-25 |
| Pretraining 500M model on 20B tokens (~430 A100-hrs) | ~$700 | ~$175-270 |
| SFT + DPO (both arms) | $20-40 | $20-40 |
| Evaluation (both arms, all tiers) | $30-60 | $30-60 |
| LLM API costs (labeling, LLM judge) | $20-50 | $20-50 |
| **Total** | **~$805-910** | **~$280-460** |

---

## Documents Produced

| File | Description |
|------|-------------|
| `spec-fineweb-curation-pipeline.md` | Full requirements spec for FineWeb-Edu curation (v2, multi-label corrected) |
| `spec-benchmarking-pythia-baseline.md` | Evaluation strategy with three-tier benchmark suite |
| `dataset-availability-50B-mix.md` | HuggingFace IDs, sizes, download instructions for all data sources (token counts revised to 20B) |
| `prompt-code-quality-labeling.md` | Prompt template for frontier LLM code quality annotation |
| `task-heatmap-classification.md` | Task description for running classifiers on FineWeb-Edu sample |
| `task-heatmap-visualization.md` | Task description for visualizing topic × complexity distribution |
| `scripts/prepare_tokenized_dataset.py` | Download, tokenize (GPT-NeoX), pack, shuffle, upload pipeline for all 8 data sources |
| `project-summary.md` | This document |