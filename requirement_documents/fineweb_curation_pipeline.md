# FineWeb-Edu Curation Pipeline — Requirements Spec (v2)

## Goal

Curate 22-25B tokens from FineWeb-Edu's `sample-100BT` (~100B tokens, ~200M docs) using our trained topic and complexity classifiers, then deduplicate. Output a clean parquet dataset ready for HuggingFace upload.

---

## Infrastructure

**RunPod instance:**
- GPU: 1x A100 80GB SXM (~$1.64/hr community cloud)
- Volume storage: 1TB+ mounted at `/workspace`
- Template: RunPod PyTorch 2.1+ / CUDA 12.x
- Estimated runtime: 12-16 hours
- Estimated cost: $20-35

**Disk budget:**
- FineWeb-Edu sample-100BT download: ~277GB (streamed, not fully materialized)
- Classification scores (parquet): ~20-40GB
- Filtered subset: ~80-120GB
- Deduped final output: ~60-80GB
- Total: ~500GB headroom needed

---

## Classifier Details

### Topic Classifier
- Architecture: ModernBERT-base, multi-label classification
- Training: BCE loss with sigmoid activation (NOT softmax)
- Output: 17 independent sigmoid scores per document, each in [0, 1]
- A document can belong to multiple topics simultaneously
- Performance: ~0.85 F1

**The 17 categories:**
```
0:  mathematics_statistics
1:  computer_science_software_engineering
2:  machine_learning_ai
3:  physical_sciences
4:  life_sciences_biology
5:  medicine_health
6:  engineering_technology
7:  business_economics
8:  law_government
9:  social_sciences
10: history_geography
11: philosophy_ethics
12: education_pedagogy
13: language_writing
14: arts_humanities
15: environmental_science_energy
16: personal_finance_practical_life
```

### Complexity Classifier
- Architecture: ModernBERT-base, single-value regression
- Output: float score in [1.0, 4.0]
- Performance: ~0.75 Spearman correlation

---

## Topic Grouping for Target Distribution

The 17 fine-grained labels map to 11 target groups for sampling purposes. Documents can belong to multiple groups (multi-label), and a single document can contribute to the quota of multiple groups.

| Target Group | Source Labels (indices) | Target % | Priority |
|---|---|---|---|
| **STEM-Core** | | | |
| Mathematics | `mathematics_statistics` (0) | 7% | HIGH — boost |
| Computer Science | `computer_science_software_engineering` (1) | 8% | HIGH — boost |
| ML/AI | `machine_learning_ai` (2) | 5% | HIGH — boost |
| **STEM-Adjacent** | | | |
| Physical Sciences | `physical_sciences` (3) | 4% | MEDIUM |
| Life Sciences | `life_sciences_biology` (4) | 3% | MEDIUM |
| Engineering/Tech | `engineering_technology` (6) | 5% | MEDIUM |
| Environmental Sci | `environmental_science_energy` (15) | 2% | MEDIUM |
| **Applied/Professional** | | | |
| Medicine/Health | `medicine_health` (5) | 4% | LOW — downsample |
| Business/Economics | `business_economics` (7) | 4% | LOW — downsample |
| Law/Government | `law_government` (8) | 3% | LOW — downsample |
| **General Knowledge** | | | |
| General | `social_sciences` (9), `history_geography` (10), `philosophy_ethics` (11), `education_pedagogy` (12), `language_writing` (13), `arts_humanities` (14), `personal_finance_practical_life` (16) | 55% | FILL — remainder |

**Note:** Percentages are targets, not hard constraints. The General Knowledge bucket is the pressure-relief valve — it absorbs whatever is left after the targeted groups hit their quotas. This avoids fighting the natural distribution too hard.

---

## Pipeline Stages

### Stage 0: Validation Run (CRITICAL — do this first)

**Purpose:** Calibrate thresholds before committing to a 6-hour classification run.

**Process:**
1. Stream 50,000 documents from FineWeb-Edu
2. Run both classifiers
3. Produce diagnostic report:

**Diagnostics needed:**
- **Sigmoid score distribution per class:** Histogram of scores for each of the 17 labels. Are they bimodal (clear yes/no) or spread out?
- **Max-sigmoid distribution:** Histogram of each document's highest sigmoid score across all 17 labels. This tells you where the "ambiguous" floor should be.
- **Multi-label rate:** What % of docs have 2+ labels above threshold? 3+? This determines how much multi-label overlap you're working with.
- **Topic distribution at various thresholds:** At threshold=0.3, 0.4, 0.5, what % of docs land in each topic? Does the distribution shift dramatically?
- **Complexity score distribution:** Histogram of regression outputs. Are they uniformly spread 1-4 or clustered?
- **Cross-label correlation matrix:** Which topics co-occur? (Expect ML/AI + CS to be high, Math + CS moderate, etc.)

**Decision points from validation:**
- Set `TOPIC_THRESHOLD`: the sigmoid score above which a document is "in" a topic. Start with 0.3, adjust based on histograms. If most docs cluster around 0.5+, use 0.5. If sigmoid outputs are spread 0.2-0.8, use 0.3.
- Set `AMBIGUITY_FLOOR`: the minimum max-sigmoid score below which a document is discarded entirely (no clear topic). This replaces the old "0.5 confidence floor" which assumed softmax.
- Verify complexity scores span the expected 1-4 range and aren't degenerate (all clustered at 2.5, etc.)

**Runtime:** ~2 minutes on A100. Do this before anything else.

---

### Stage 1: Classify

**Input:** `HuggingFaceFW/fineweb-edu` config `sample-100BT` (streamed from HF)

**Process:**
1. Stream FineWeb-Edu in batches (batch_size=512, truncate to 512 tokens)
2. Run both classifiers on each batch (both models fit in VRAM simultaneously, ~1.5GB total)
3. For topic: apply sigmoid to logits → 17 independent scores
4. For complexity: take regression output → float score (1.0 - 4.0)
5. Save results as parquet shards

**Inference pseudocode:**
```python
with torch.no_grad(), torch.cuda.amp.autocast():
    topic_logits = topic_model(**inputs).logits          # shape: (batch, 17)
    complexity_scores = complexity_model(**inputs).logits.squeeze(-1)  # shape: (batch,)

# Sigmoid, NOT softmax — this is multi-label
topic_scores = torch.sigmoid(topic_logits)  # each value in [0, 1] independently

# Store the full score vector — don't argmax yet
# Filtering happens in Stage 2 with the calibrated threshold
```

**Output schema per row:**
```
{
  "text": str,                 # original document text
  "url": str,                  # source URL  
  "token_count": int,          # from FineWeb-Edu metadata
  "dump": str,                 # CC dump identifier
  "topic_scores": list[float], # length-17 sigmoid scores (ordered by label index)
  "complexity": float          # regression score 1.0-4.0
}
```

**Why store the full 17-score vector:**
- Enables re-filtering with different thresholds without re-running inference
- Multi-label documents (e.g., scored 0.8 CS + 0.7 Math) are the most valuable for structured output — you want to identify and prioritize them
- Only adds ~140 bytes per row (17 × float32), negligible vs the text field

**Performance target:** 10-15k docs/sec on A100 with batch_size=512. ~4-6 hours for 200M docs.

**Checkpointing:** Save a parquet shard every 1M documents (~200 shards total). Log progress to file. If interrupted, resume from last completed shard.

**Streaming vs download:** Download specific CC dumps (CC-MAIN-2023-50, CC-MAIN-2024-10, CC-MAIN-2024-18) rather than the full 100BT sample. Smaller, restartable, and HF recommends these for small trainings. Add more dumps only if you're short on tokens after filtering.

---

### Stage 2: Filter

**Input:** Scored parquet shards from Stage 1

**Step 2a: Compute actual distributions**

First pass over all scored shards to compute:
- Per-topic document counts at the calibrated `TOPIC_THRESHOLD` 
- Documents per complexity bucket
- Joint distribution (topic × complexity)
- Multi-label overlap statistics
- Total available tokens per group

**Step 2b: Assign documents to topic groups**

For each document, determine which target groups it belongs to:
```python
TOPIC_THRESHOLD = 0.3  # calibrated from Stage 0

# Map label indices to target groups
GROUP_MAP = {
    "mathematics":       [0],   # mathematics_statistics
    "computer_science":  [1],   # computer_science_software_engineering  
    "ml_ai":             [2],   # machine_learning_ai
    "physical_sciences": [3],   # physical_sciences
    "life_sciences":     [4],   # life_sciences_biology
    "engineering_tech":  [6],   # engineering_technology
    "environmental":     [15],  # environmental_science_energy
    "medicine_health":   [5],   # medicine_health
    "business_economics":[7],   # business_economics
    "law_government":    [8],   # law_government
    "general":           [9, 10, 11, 12, 13, 14, 16],  # everything else
}

doc_groups = []
for group_name, label_indices in GROUP_MAP.items():
    if any(topic_scores[i] >= TOPIC_THRESHOLD for i in label_indices):
        doc_groups.append(group_name)

# A document can be in multiple groups, e.g., ["computer_science", "ml_ai"]
```

**Step 2c: Priority-based sampling**

The filtering algorithm accounts for multi-label overlap and prioritizes high-value documents:

1. **Compute "structured output relevance" score** for each document:
   ```python
   # Bonus for STEM-Core topics (the ones we're boosting)
   stem_core_score = max(topic_scores[0], topic_scores[1], topic_scores[2])  # math, cs, ml
   
   # Bonus for multi-label (cross-disciplinary docs are richer)
   num_active_topics = sum(1 for s in topic_scores if s >= TOPIC_THRESHOLD)
   multi_label_bonus = min(num_active_topics * 0.05, 0.15)  # cap at 0.15
   
   # Combined relevance score
   relevance = stem_core_score + multi_label_bonus + (complexity * 0.1)
   ```

2. **For each target group**, compute how many tokens are needed vs available:
   - If available > needed: sample top documents by relevance score
   - If available < needed: take all, log shortfall

3. **Handle multi-label overlap:** A document in both CS and ML/AI counts toward both quotas. This is fine — it means the actual download will be slightly smaller than the sum of group targets, but the topic coverage will be correct.

4. **Fill the General Knowledge bucket** last with whatever quota remains after targeted groups are satisfied. Prefer higher-complexity documents from this pool.

**Step 2d: Quality filters (applied to ALL documents before group sampling)**

- Drop if `max(topic_scores) < AMBIGUITY_FLOOR` (calibrated from Stage 0)
- Drop if `token_count < 50` (too short to be useful)
- Drop if `token_count > 100,000` (likely scraped garbage or full book dumps)

**Target distribution (complexity), applied within each group:**

| Level | Range | Target % |
|-------|-------|----------|
| L1 (basic) | 1.0 - 1.75 | 10% |
| L2 (intermediate) | 1.75 - 2.5 | 20% |
| L3 (advanced) | 2.5 - 3.25 | 40% |
| L4 (expert) | 3.25 - 4.0 | 30% |

**Output:** Filtered parquet shards with added fields:
```
{
  ... (all fields from Stage 1),
  "assigned_groups": list[str],    # which target groups this doc belongs to
  "relevance_score": float         # composite score used for sampling
}
```

**Target output size:** 22-25B tokens. Use a fixed random seed (42) for reproducibility.

---

### Stage 3: Deduplicate

**Input:** Filtered parquet shards from Stage 2

**Method:** MinHash LSH deduplication

**Parameters:**
- Shingling: 13-gram word shingles (consistent with FineWeb's approach)
- MinHash: 128 permutations
- LSH threshold: 0.7 Jaccard similarity
- Library: `datasketch` MinHashLSH or custom with `xxhash` for speed

**Process:**
1. Compute MinHash signature for each document
2. Insert into LSH index
3. For each cluster of near-duplicates, keep the document with highest `relevance_score`
4. Expected dedup rate: 5-15% (FineWeb-Edu is already deduped, but topic filtering may concentrate duplicates)

**Memory:** Working with filtered subset (~50-60M docs). 50M × 128 hashes × 4 bytes = ~25GB — fits in RAM on a standard RunPod instance with 128GB+. If tight, use band-based LSH with disk-backed index.

**Simpler alternative:** If MinHash is too slow or complex, do exact URL dedup (drop docs sharing the same URL) + 50-char prefix dedup (catches boilerplate). FineWeb-Edu is already heavily deduped, so a lighter pass is defensible.

**Output:** Deduped parquet shards, same schema + `dedup_cluster_id` column (for analysis).

---

### Stage 4: Export

**Input:** Deduped parquet shards from Stage 3

**Process:**
1. Keep all metadata columns — they're useful for the HF dataset card and downstream analysis
2. Merge shards into clean, consistently-sized parquet files (~500MB each)
3. Compute and log final statistics:
   - Total tokens, total documents
   - Per-group distribution (actual % vs target %)
   - Per-label distribution (all 17 labels at threshold)
   - Complexity distribution
   - Topic × complexity heatmap
   - Multi-label overlap matrix
   - Top 20 domains by token count
   - Token count distribution (p10, p25, p50, p75, p90)
4. Generate dataset card (README.md) with above stats
5. Upload to HuggingFace Hub

**Final output schema:**
```
{
  "text": str,
  "url": str,
  "token_count": int,
  "dump": str,
  "topic_scores": list[float],   # full 17-dim sigmoid vector
  "complexity": float,
  "assigned_groups": list[str],  # target groups at filtering threshold
  "relevance_score": float       # composite score
}
```

**HuggingFace upload:**
```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("{your-username}/fineweb-edu-curated", repo_type="dataset")
api.upload_folder(
    folder_path="/workspace/fineweb-curation/output",
    repo_id="{your-username}/fineweb-edu-curated",
    repo_type="dataset"
)
```

---

## Dependencies

```
torch>=2.1
transformers>=4.36
datasets>=2.16
accelerate
pyarrow>=14.0
datasketch>=1.6
xxhash
tqdm
huggingface_hub>=0.20
pandas
numpy
```

---

## Key Implementation Notes

### Multi-Label Handling Throughout

The single most important architectural decision: **never flatten to single-label until the final sampling step.** Store the full 17-dim sigmoid vector. Filter using thresholds on individual dimensions. A document about "ML algorithms for drug discovery" scoring 0.75 ML/AI + 0.68 medicine_health + 0.55 computer_science is one of your most valuable training examples — it teaches cross-domain structured reasoning. The pipeline should identify and prioritize these, not force them into a single bucket.

### Streaming vs Download

Download specific CC dumps rather than streaming full 100BT:
- CC-MAIN-2023-50, CC-MAIN-2024-10, CC-MAIN-2024-18 (HF recommended)
- Smaller download, restartable, and these are the highest quality dumps
- Add more dumps only if short on tokens after filtering

### Reproducibility

- Random seed 42 for all sampling
- Log all thresholds (TOPIC_THRESHOLD, AMBIGUITY_FLOOR) and their source
- Save Stage 0 diagnostic plots alongside output
- Pin all dependency versions
- Save filtering config as JSON with output

### Error Handling

- Catch and log documents that fail tokenization
- Resume from last saved shard on interruption
- Validate parquet files after writing (row count check)

---

## Estimated Timeline

| Stage | Duration | Notes |
|-------|----------|-------|
| Setup + download dumps | 1-2 hours | Network dependent |
| Stage 0: Validation | 2-5 minutes | 50k docs, diagnostic plots |
| Stage 1: Classify | 4-6 hours | ~10-15k docs/sec on A100 |
| Stage 2: Filter | 30-60 min | Mostly I/O, two passes |
| Stage 3: Dedup | 2-4 hours | MinHash computation + LSH |
| Stage 4: Export | 30-60 min | Merge + stats + upload |
| **Total** | **8-14 hours** | **~$15-25 on RunPod** |

---

## Success Criteria

- [ ] Stage 0 validation run completed, thresholds calibrated
- [ ] 22-25B tokens in final output
- [ ] STEM-Core topics (Math, CS, ML/AI) boosted to ~20% combined (from ~5% natural)
- [ ] Complexity distribution within ±5% of targets
- [ ] No duplicate documents above 0.7 Jaccard similarity
- [ ] Multi-label documents identified and preserved (not collapsed)
- [ ] All parquet files valid and loadable
- [ ] Dataset card with full statistics generated
- [ ] Uploaded to HuggingFace Hub
- [ ] Filtering config and Stage 0 diagnostics saved alongside dataset