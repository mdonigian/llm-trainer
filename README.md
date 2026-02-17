# FineWeb-Edu LLM Training Project

Two-stage pipeline for building a fast content classifier from FineWeb-Edu data:

1. **Categorize** — Use OpenAI to label each document across 17 categories
2. **Train** — Fine-tune a BERT classifier on those labels for fast, local inference

## Categories

| Field Name | Category |
|---|---|
| `mathematics_statistics` | Mathematics & Statistics |
| `computer_science_software_engineering` | Computer Science & Software Engineering |
| `machine_learning_ai` | Machine Learning & AI |
| `physical_sciences` | Physical Sciences |
| `life_sciences_biology` | Life Sciences & Biology |
| `medicine_health` | Medicine & Health |
| `engineering_technology` | Engineering & Technology |
| `business_economics` | Business & Economics |
| `law_government` | Law & Government |
| `social_sciences` | Social Sciences |
| `history_geography` | History & Geography |
| `philosophy_ethics` | Philosophy & Ethics |
| `education_pedagogy` | Education & Pedagogy |
| `language_writing` | Language & Writing |
| `arts_humanities` | Arts & Humanities |
| `environmental_science_energy` | Environmental Science & Energy |
| `personal_finance_practical_life` | Personal Finance & Practical Life |

## Setup

### 1. Install dependencies

```bash
pip install -r scripts/requirements.txt
```

### 2. Set your OpenAI API key (for categorization only)

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-your-key-here
```

---

## Step 1: Categorize with OpenAI

See `scripts/categorize_fineweb_edu.py`. Supports real-time async and Batch API modes.

### Batch API (recommended for large datasets — 50% cheaper)

```bash
# Submit (auto-splits into multiple batches if file is large)
python scripts/categorize_fineweb_edu.py batch submit data.parquet

# Wait for all batches to complete
python scripts/categorize_fineweb_edu.py batch wait batch_input.manifest.json

# Download and merge results
python scripts/categorize_fineweb_edu.py batch download batch_input.manifest.json data.parquet -o categorized.parquet

# Check label distribution
python scripts/categorize_fineweb_edu.py analyze categorized.parquet
```

### Async (real-time, for smaller jobs)

```bash
python scripts/categorize_fineweb_edu.py async data.parquet -o categorized.parquet --max-rows 1000
```

Run `python scripts/categorize_fineweb_edu.py --help` for all options.

---

## Step 2: Train BERT Classifier

Takes the categorized parquet from Step 1 and fine-tunes a BERT model for multi-label classification.

### Quick Start

```bash
# Train from a single file
python scripts/train_fineweb_edu.py categorized.parquet

# Train from a directory of parquet/csv files (all are concatenated)
python scripts/train_fineweb_edu.py data/categorized/

# Max speed on GPU — compile + large batches
python scripts/train_fineweb_edu.py data/categorized/ --compile --batch-size 64

# Train with a specific model and more epochs
python scripts/train_fineweb_edu.py categorized.parquet --model bert-base-uncased --epochs 5
```

### All Options

```
python scripts/train_fineweb_edu.py <file_or_directory> [options]
```

| Flag | Default | Description |
|---|---|---|
| `--model` | `answerdotai/ModernBERT-base` | HuggingFace model name or path |
| `--output-dir` | `models/fineweb-edu-classifier` | Where to save the trained model |
| `--epochs` | `3` | Number of training epochs |
| `--batch-size` | `32` | Training batch size |
| `--learning-rate` | `2e-5` | Learning rate |
| `--max-length` | `512` | Max token length |
| `--val-split` | `0.1` | Validation split ratio |
| `--test-split` | `0.1` | Test split ratio |
| `--weight-decay` | `0.01` | AdamW weight decay |
| `--warmup-ratio` | `0.1` | Linear warmup ratio |
| `--num-workers` | `4` | DataLoader workers |
| `--seed` | `42` | Random seed |
| `--text-column` | `text` | Column containing text |
| `--compile` | off | Use torch.compile for fused kernels (PyTorch 2.x) |
| `--no-amp` | off | Disable automatic mixed precision (AMP is on by default) |
| `--eval-only` | — | Evaluate a saved model instead of training |

### What it does

1. Loads the categorized parquet/csv file(s) (accepts a single file or a directory)
2. Prints label distribution
3. Splits data into train/val/test (80/10/10)
4. Fine-tunes BERT with a multi-label classification head (BCEWithLogitsLoss)
5. Saves the best model (by validation F1) after each epoch
6. Runs final evaluation on the held-out test set
7. Prints per-category precision/recall/F1

### Output

The trained model is saved to `--output-dir` with:

```
models/fineweb-edu-classifier/
├── config.json              # Model config
├── model.safetensors        # Model weights
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── vocab.txt
├── label_config.json        # Category field names and display names
├── training_history.csv     # Loss and metrics per epoch
└── test_metrics.json        # Final test set metrics
```

### Evaluate a saved model

```bash
python scripts/train_fineweb_edu.py test_data.parquet --eval-only --model models/fineweb-edu-classifier
```

### Performance tips

- **AMP is on by default** — Automatic mixed precision (bf16 on Ampere+, fp16 otherwise) runs ~2x faster and halves memory. Pass `--no-amp` to disable.
- **`--compile`** — Enables `torch.compile` for fused CUDA kernels (~20-40% faster). The first batch is slow while compiling; subsequent batches are faster.
- **Dynamic padding** — Batches are padded to the longest text in the batch, not to `--max-length`. This avoids wasting compute on padding tokens.
- **Batch size** — Default is 32. With AMP you can often go to 64 on a 12 GB GPU. Decrease to 16 or 8 if you run out of memory.
- **`--num-workers`** — Default is 4. Tokenization runs in parallel worker processes, overlapping with GPU compute.
- **GPU recommended** — Training on CPU is slow for large datasets. The script auto-detects CUDA and Apple MPS.
- **Start small** — Use `--max-rows` in the categorization step to create a small labeled set for initial experiments.
- **Model choice** — `answerdotai/ModernBERT-base` is the default. For better accuracy try `microsoft/deberta-v3-base`. For speed try `distilbert-base-uncased`.

---

## Project Structure

```
llm_project/
├── scripts/
│   ├── categorize_fineweb_edu.py   # OpenAI categorization script
│   ├── train_fineweb_edu.py        # BERT training script
│   └── requirements.txt            # Python dependencies
├── models/                         # Trained models (created by training)
├── training_data/                  # Input data
├── .env                            # OpenAI API key (not committed)
├── README.md                       # This file
└── worklog.md
```
