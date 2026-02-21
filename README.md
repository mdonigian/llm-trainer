# Curated Pretraining for Structured Output: A 470M Parameter Study

Code and pipeline scripts for training a 470M parameter LLM from scratch on 20B tokens curated via frontier-model-guided data selection, optimized for structured output tasks (JSON generation, function calling, schema compliance).

**Research question:** Does multi-dimensional data curation during pretraining (topic relevance × reasoning complexity) measurably improve structured output capabilities beyond what supervised fine-tuning alone achieves?

## Approach

1. Train BERT-based classifiers (using frontier LLM labels) to score documents along two axes: topic relevance and reasoning complexity
2. Apply these classifiers to filter and rank billions of tokens from FineWeb-Edu and StarCoderData
3. Pretrain a LLaMA-style 470M model on the resulting curated 20B token mix
4. Compare against Pythia-410M (trained on uncurated data) after identical SFT/DPO post-training

The core thesis: structural patterns (JSON, schemas, type systems, nested hierarchies) embedded deeply during pretraining produce better generalization to *unseen* schemas than SFT format-following alone.

## Repository Structure

```
├── requirements.txt                          Python dependencies
├── docs/
│   ├── project_summary.md                    Development narrative and design decisions
│   └── fineweb_curation_spec.md              FineWeb-Edu pipeline requirements specification
│
├── scripts/
│   ├── classifiers/                          Stage 1: BERT classifier training
│   │   ├── categorize_fineweb_edu.py             Generate topic labels via OpenAI Batch API
│   │   ├── categorize_fineweb_edu_bert.py        Run trained topic classifier at scale
│   │   ├── complexity_fineweb_edu.py              Generate complexity labels via OpenAI
│   │   ├── classify_starcoderdata.py              Generate code quality/relevance labels via OpenAI
│   │   ├── train_fineweb_edu.py                   Train 17-class multi-label topic classifier (ModernBERT)
│   │   ├── train_complexity_fineweb_edu.py        Train complexity regression classifier
│   │   └── train_starcoderdata.py                 Train code quality + structured relevance classifiers
│   │
│   ├── curation/                             Stage 2: Data curation pipelines
│   │   ├── download_starcoderdata.py              Download raw StarCoderData from HuggingFace
│   │   ├── fineweb_pipeline/                      FineWeb-Edu curation (classify → filter → dedup → export)
│   │   │   ├── pipeline_config.py                     Shared configuration and thresholds
│   │   │   ├── pipeline_download.py                   Stream FineWeb-Edu from HuggingFace
│   │   │   ├── pipeline_classify.py                   Run topic + complexity classifiers on all docs
│   │   │   ├── pipeline_filter.py                     Apply score thresholds and topic rebalancing
│   │   │   ├── pipeline_dedup.py                      MinHash LSH deduplication
│   │   │   ├── pipeline_validate.py                   Stage 0 validation on 50k doc sample
│   │   │   └── pipeline_export.py                     Export curated subset to parquet
│   │   └── starcoder_pipeline/                    StarCoderData curation (classify → filter → dedup → export)
│   │       ├── pipeline_config.py                     Shared configuration, language slices, thresholds
│   │       ├── pipeline_download.py                   Download StarCoderData language slices
│   │       ├── pipeline_classify.py                   Run quality + structured relevance + content type classifiers
│   │       ├── pipeline_filter.py                     Compression ratio pre-filter + score-based filtering
│   │       ├── pipeline_dedup.py                      MinHash LSH deduplication
│   │       ├── pipeline_validate.py                   Validation and distribution checks
│   │       ├── pipeline_export.py                     Export curated subset to parquet
│   │       └── repair_lang_tags.py                    Fix language metadata inconsistencies
│   │
│   ├── pretraining/                          Stage 3: Tokenization and model training
│   │   ├── prepare_tokenized_dataset.py           Tokenize all sources, pack into 2048-token sequences, upload shards
│   │   └── train.py                               LLM pretraining (LitGPT, LLaMA-style architecture)
│   │
│   └── analysis/                             Supporting analysis
│       ├── heatmap_topic_complexity.py             Topic × complexity distribution visualization
│       └── benchmark_a100_autotune.py              A100 GPU throughput benchmarking
```

## Classifiers

Five BERT classifiers (all ModernBERT-base) power the curation pipeline:

| Classifier | Task | Architecture | Performance |
|---|---|---|---|
| Topic | 17-class multi-label | BCE + sigmoid | ~0.85 F1 |
| Complexity | 1–4 regression | MSE | ~0.75 Spearman ρ |
| Code quality | 1–5 regression | MSE | 0.575 Spearman ρ |
| Structured data relevance | 0–3 regression | MSE | 0.807 Spearman ρ |
| Content type | 9-class single-label | Cross-entropy | 87.5% accuracy |

Training labels were generated by a frontier LLM (Claude/GPT-4) and used to fine-tune fast ModernBERT models for inference at scale. The topic classifier uses multi-label sigmoid outputs (documents can span multiple topics); multi-label documents are prioritized during filtering as they represent cross-domain reasoning.

## Token Mix (20B)

| Source | Tokens | % | Curation |
|---|---|---|---|
| FineWeb-Edu (curated) | 4.3B | 21.5% | Topic + complexity filtered, deduped |
| FineWeb-Edu (random) | 2.0B | 10.0% | Uncurated baseline diversity |
| StarCoderData (curated) | 3.9B | 19.5% | Quality + structured relevance filtered, deduped |
| FineMath-4+ | 3.2B | 16.0% | Pre-filtered by source |
| peS2o (CS/math/ML) | 2.2B | 11.0% | Subject-filtered academic papers |
| Structured Wikipedia | 1.5B | 7.5% | JSON infoboxes + article prose |
| Wikipedia EN (plain) | 0.5B | 2.5% | Dual-representation overlap with structured |
| StackExchange (technical) | 1.0B | 5.0% | High-score Q&A |
| UltraChat | 0.4B | 2.0% | General instruction diversity |
| **Total** | **~20B** | | **40 tokens/parameter** |

All sources tokenized with GPT-NeoX tokenizer (50,304 vocab), packed into 2048-token sequences with EOS separators, globally shuffled, and stored as pre-tokenized binary shards (TKDS v1 format).

## Model Architecture

~470M parameter LLaMA-style transformer:

| Parameter | Value |
|---|---|
| Layers | 24 |
| Hidden dimension | 1024 |
| Query heads | 16 |
| KV heads | 8 (GQA, 2:1) |
| FFN dimension | 2816 (SwiGLU) |
| Context length | 2048 |
| Vocab size | 50,304 (GPT-NeoX) |
| Normalization | RMSNorm |
| Position encoding | RoPE |

**Baseline:** Pythia-410M — trained on The Pile (uncurated), full checkpoint history public, designed for reproducibility research.

## Setup

```bash
pip install -r requirements.txt
```

Scripts that use the OpenAI API for label generation require an `OPENAI_API_KEY` in a `.env` file at the project root.

## Reproduction

### Stage 1: Train classifiers

Generate frontier-LLM labels, then train BERT classifiers:

```bash
# Generate topic labels (OpenAI Batch API)
python scripts/classifiers/categorize_fineweb_edu.py batch submit data.parquet
python scripts/classifiers/categorize_fineweb_edu.py batch wait manifest.json
python scripts/classifiers/categorize_fineweb_edu.py batch download manifest.json data.parquet -o labeled.parquet

# Train topic classifier
python scripts/classifiers/train_fineweb_edu.py labeled.parquet --compile

# Train complexity classifier
python scripts/classifiers/train_complexity_fineweb_edu.py labeled.parquet --compile

# Train code classifiers (quality, structured relevance, content type)
python scripts/classifiers/train_starcoderdata.py labeled_code.parquet --compile
```

### Stage 2: Run curation pipelines

Each pipeline runs as a sequence of standalone stages:

```bash
# FineWeb-Edu: download → classify → filter → dedup → export
cd scripts/curation/fineweb_pipeline
python pipeline_download.py
python pipeline_classify.py
python pipeline_filter.py
python pipeline_dedup.py
python pipeline_export.py

# StarCoderData: download → classify → filter → dedup → export
cd scripts/curation/starcoder_pipeline
python pipeline_download.py
python pipeline_classify.py
python pipeline_filter.py
python pipeline_dedup.py
python pipeline_export.py
```

### Stage 3: Tokenize and train

```bash
# Tokenize all sources → packed binary shards → HuggingFace upload
python scripts/pretraining/prepare_tokenized_dataset.py

# Pretrain 470M model
python scripts/pretraining/train.py
```

## Key Design Decisions

See `docs/project_summary.md` for the full development narrative. The major decisions:

- **Multi-label topic classification** — documents can belong to multiple topics; multi-label overlap is treated as a signal of cross-domain reasoning value
- **Regression over classification for ordinal scores** — fuzzy boundaries between adjacent quality/complexity levels make classification unreliable; regression with Spearman ρ works better
- **Train from scratch** — isolates the pretraining curation variable without confounding from a base model's prior knowledge
- **Compression ratio as code pre-filter** — zlib ratio < 0.10 catches repetitive boilerplate (SQL migrations, generated configs) before expensive BERT inference
- **Stage 0 validation** — 50k document sample calibrates sigmoid thresholds in ~2 minutes before committing to multi-hour classification runs

## License

[TBD]

## Citation

```bibtex
@article{curated-pretraining-structured-output-2025,
  title={[TBD]},
  author={[TBD]},
  year={2025},
  note={Code available at [TBD]}
}
```
