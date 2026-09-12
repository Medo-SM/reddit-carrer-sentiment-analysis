# Plan: Modeling Pipeline Rework

## Context

Data = **Reddit comments and posts** from career/tech subreddits (`cscareerquestions`,
`cscareerquestionsEU`, `datascience`, `dataengineering`, `cybersecurity`,
`SoftwareEngineering`). Text is informal, slang-heavy, sarcastic, with heavy negation,
code snippets, markdown, and emojis. Any sentiment/topic approach must account for this.

The current modeling notebook (`notebooks/1.3_notebook.ipynb`) produces results that
"don't make sense". This plan documents the root causes (6 confirmed bugs) and the
rework to fix them, including a better sentiment model.

**Decisions locked during planning (do not change without discussion):**
- No implementation beyond saving this plan file.
- Sentiment primary model: `cardiffnlp/twitter-roberta-base-sentiment` (3-class).
- VADER: cross-check only, with neutral handling (`|compound| > 0.05`).
- No FinBERT (finance-tuned, wrong domain).
- No custom fine-tune this pass (optional follow-up: hand-label ~1-2k Reddit comments
  and fine-tune roberta for a better domain fit).
- **Titles vs. comments decision:** run topic modeling on titles (they cluster better
  due to concise phrasing), run sentiment on comments (richer opinion signal), and
  cross-reference by linking comments to their parent submission's topic. Track
  `dataset_type` as a column through the entire pipeline. Report metrics split by type.

## 1. Extract a reusable `src/modeling/` module (moved out of notebook)

```
src/modeling/
  __init__.py   # set_seed() utility (see Reproducibility section)
  load.py       # parquet discovery, sampling, dtype-aware created_utc parsing
  text.py       # imports clean_text from src.ingest.sampling; adds modeling-specific transforms
  topics.py     # BERTopic wrapper: build, fit w/ precomputed embeddings, name map, evolution
  sentiment.py  # domain sentiment wrapper + batched inference + aggregation helpers
  cache.py      # intermediate results cache (embeddings, fitted models)
```

Notebook cells become thin calls into these functions.

## 2. Fix the 6 confirmed bugs

### Bug 1 - Timestamps: temporal axes are garbage
Cell 28: `pd.to_datetime(df['created_utc'])` interprets integer epoch *seconds* as
*nanoseconds* -> dates collapse to `1970-01-01` (verified empirically). Sources differ
(scraper JSONL = epoch ints; `zst_to_parquet.py` already converts; sampled JSON = ISO
strings from the sampler's `default=str`), so one blanket call can't be right.

**Fix:** `parse_created_utc()` utility that is dtype-aware:
- integer/float -> `pd.to_datetime(col, unit='s')`
- ISO string -> `pd.to_datetime(col)` (with warning)
- datetime -> no-op
Add a dtype assert/log so wrong formats fail loudly instead of plotting 1970.
Use it everywhere `created_utc` is touched (cells 28/29/33/35/37).

### Bug 2 - Legend labels index the wrong object
Cells 35/37: `label_name = f"Topic {topic_id}: {topics[topic_id]}"`. `topics` is the
per-document assignment array (~45k entries); indexing it by topic *id* yields an
arbitrary document.

**Fix:** plots use a `topic_id -> Name` map built from `topic_model.get_topic_info()`
(see `topics.py::topic_name_map()`).

### Bug 3 - ~half the corpus is unassigned (topic -1)
Stored output: 21,909 / 45,302 docs (48.4%) in the outlier topic.
`HDBSCAN(min_cluster_size=100)` is too strict for submission titles; UMAP
`n_neighbors=15, n_components=5, min_dist=0.0` collapses density.

**Fix — structured parameter sweep with quality evaluation:**

1. **Sweep parameters:**
   - `min_cluster_size`: [15, 25, 35, 50]
   - `min_samples`: [5, 10, 15] (separate from `min_cluster_size` — controls density threshold independently)
   - UMAP `n_neighbors`: [10, 15, 25], `min_dist`: [0.0, 0.05, 0.1]

2. **Quality gates (not just outlier rate):**
   - **Outlier rate:** target <20% unassigned, but this is a soft threshold — do not force clusters at the expense of coherence.
   - **Topic coherence (c_v):** compute via `gensim.models.CoherenceModel` on the top-10 words per topic. Higher is better; compare across sweep runs.
   - **Topic count sanity check:** on 45k short titles, expect 10-30 meaningful topics. If sweep produces 100+ micro-topics, `min_cluster_size` is too low.
   - **Manual inspection checkpoint:** after the sweep, print top-10 words per topic for the best config and have a human sanity-check before committing parameters.

3. **Log results** to `data/processed/sweep_results.json` for comparison (see experiment tracking section below).

4. Keep topic -1 but report/share-plot it rather than silently dropping (current cell 37 drops it, cell 35 does not).

### Bug 4 - Double / wasted embedding
Cell 28 precomputes embeddings from raw `text`, then calls
`fit_transform(df["cleaned_text"])` WITHOUT passing them -> BERTopic re-embeds
`cleaned_text` (~2 min wasted); the precomputed ones (on different text) are discarded.

**Fix:** compute embeddings once on `cleaned_text`, pass to
`fit_transform(..., embeddings=...)`. Ensures alignment + no recompute.

### Bug 5 - Cell 14 is dead code (confirmed TypeError)
`json_to_df(data_samples_path, dataset_type)` passes `"comments"` into the `subreddit`
parameter -> globs `comments_submissions.json` -> returns `None` -> TypeError.

**Fix:** correct the call to `json_to_df(data_samples_path, subreddit, dataset_type)`
or replace with a `load`-module counter.

### Bug 6 - VADER skew / domain mismatch
VADER: mean 0.389, median 0.440, 25th pct = 0.0 (neutral spike). Tuned for generic
social media; weak on career/tech text; mean-by-topic dominated by neutral zeros.

**Fix (locked):**
- Primary: `cardiffnlp/twitter-roberta-base-sentiment` (3-class; trained on informal
  social media - structurally the closest free pretrained model to Reddit).
- VADER kept only as a cross-check, with neutral handling (`|compound| > 0.05`).
- Report **share-of-class (%) per topic and per month** rather than a single mean.

## 3. Batched inference for RoBERTa sentiment

`src/modeling/sentiment.py` must handle large corpora efficiently:

- **Batch inference** with configurable `batch_size` (32-128 depending on GPU memory).
- **`torch.no_grad()`** context for all inference calls.
- **`tqdm` progress bar** for long-running inference.
- **`device` config option** (`cuda`, `cpu`, `mps`) with auto-detection fallback.
- **Fallback sample size** for CPU-only environments (e.g., 5k instead of 1M) — add
  `modeling.max_sample_cpu` to config.

```python
# src/modeling/sentiment.py sketch
def score_batch(texts: list[str], model, tokenizer, device: str, batch_size: int = 64) -> pd.DataFrame:
    """Returns DataFrame with columns [negative, neutral, positive] as probabilities."""
    results = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment"):
            batch = texts[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            output = model(**encoded)
            scores = torch.nn.functional.softmax(output.logits, dim=-1)
            results.append(scores.cpu().numpy())
    return pd.DataFrame(np.vstack(results), columns=["negative", "neutral", "positive"])
```

## 4. VADER vs. RoBERTa comparison visualization

Since VADER is kept as a cross-check, add a specific comparison output:

- **Scatter plot:** VADER compound vs. RoBERTa negative probability per document.
- **Agreement rate:** % of documents where both models agree on polarity (positive/negative/neutral).
- **Disagreement analysis:** top-20 cases where they disagree most — reveals domain-specific failure modes and validates the model switch.

This turns the cross-check from a hand-wave into actionable evidence.

## 5. Reproducibility

The current plan sets `seed: 42` in config but doesn't address the full chain.

**Add `set_seed()` utility in `src/modeling/__init__.py`:**

```python
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

**Ensure determinism across the pipeline:**
- UMAP: pass `random_state=seed` explicitly (non-deterministic by default).
- HDBSCAN: deterministic if input (UMAP output) is deterministic.
- `SentenceTransformer.encode()`: use `normalize_embeddings=True` for consistent cosine similarity.
- Document the exact Python/torch/BERTopic versions that produced the results (add to experiment log).

## 6. Intermediate results cache (`src/modeling/cache.py`)

Embeddings are expensive (~2 min for 45k docs on GPU). If a notebook cell crashes
after embedding but before topic fitting, that work is lost.

```python
# src/modeling/cache.py sketch
CACHE_DIR = Path("data/processed/cache")

def save_embeddings(embeddings: np.ndarray, key: str) -> Path:
    """Save embeddings to .npy file. Key includes subreddit + doc count."""
    path = CACHE_DIR / f"embeddings_{key}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    return path

def load_embeddings(key: str) -> np.ndarray | None:
    """Load cached embeddings if they exist and doc count matches."""
    path = CACHE_DIR / f"embeddings_{key}.npy"
    return np.load(path) if path.exists() else None

def save_topic_model(model, key: str) -> Path:
    """Save fitted BERTopic model."""
    path = CACHE_DIR / f"topic_model_{key}"
    model.save(str(path))
    return path

def load_topic_model(key: str):
    """Load cached BERTopic model if it exists."""
    path = CACHE_DIR / f"topic_model_{key}"
    return BERTopic.load(str(path)) if path.exists() else None
```

This is especially important during the Bug 3 parameter sweep.

## 7. Experiment tracking

Add a lightweight JSON log per run (no MLflow/W&B dependency):

```python
# src/modeling/tracking.py
def log_experiment(params: dict, metrics: dict, output_dir: str = "data/processed/experiments") -> Path:
    """Append experiment results to a JSONL log file."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "params": params,  # n_docs, dataset_type, hdbscan_*, umap_*, sentiment_model, seed
        "metrics": metrics,  # n_topics, outlier_pct, coherence_cv, agreement_rate
        "versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "bertopic": bertopic.__version__,
        }
    }
    path = Path(output_dir) / "experiment_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return path
```

This makes parameter sweep results comparable without re-running everything.

## 8. `dataset_type` as a first-class dimension

The config has `dataset_type` but it's treated as a simple filter. The modeling
module should:

- Track `dataset_type` as a column through the entire pipeline.
- Report metrics split by type (topic distribution for titles, sentiment distribution
  for comments).
- Allow running the pipeline on one or both types via config (`modeling.dataset_types: [submissions, comments]`).
- Cross-reference: link comments to their parent submission's topic for combined analysis.

## 9. Reddit-specific preprocessing notes

- Keep code-block/markdown/URL stripping (already in `process_pipeline`).
- Consider emoji normalization before embedding.
- Keep `\W+` scrubbing (Reddit-lenient) - but do NOT lowercase-strip domain terms that
  matter (`Data Science`, `ML`) before topic modeling.

## 10. `config/config.yaml` additions

```yaml
modeling:
  embedding_model: "all-MiniLM-L6-v2"
  umap:
    n_neighbors: 15
    n_components: 5
    min_dist: 0.0
    metric: cosine
    random_state: 42        # required for reproducibility
  hdbscan:
    min_cluster_size: 100    # to be set after sweep
    min_samples: 10          # separate from min_cluster_size
    metric: euclidean
    cluster_selection_method: eom
    prediction_data: true
  min_topic_size: ...
  top_n_words: 10
  sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment"
  sentiment_batch_size: 64
  device: "auto"             # auto, cuda, cpu, mps
  max_sample_cpu: 5000       # fallback sample size for CPU-only
  dataset_types:             # run pipeline on these types
    - submissions
    - comments
  seed: 42
  cache_dir: "data/processed/cache"
  experiment_log_dir: "data/processed/experiments"
```
(Exact HDBSCAN values to be set after the min_cluster_size sweep.)

## 11. Tests (unified: `pytest`)

Use `pytest` everywhere (aligned with the consolidation plan). Add markers for slow tests.

- `tests/test_text.py` - cleaning (URLs, code blocks, markdown).
- `tests/test_dates.py` - `parse_created_utc` for int/string/datetime.
- `tests/test_sentiment.py` - share-of-class aggregation on synthetic scores; batched inference shape check.
- `tests/test_topics.py` - `topic_name_map` shape + fit on ~1k docs (marked `@pytest.mark.slow`).
- `tests/test_cache.py` - embeddings save/load round-trip; model save/load round-trip.
- `tests/test_tracking.py` - experiment log write/read round-trip.
- `tests/test_reproducibility.py` - `set_seed()` produces identical embeddings on repeated calls (marked `@pytest.mark.slow`).

## 12. Docs

- Update the notebook's "results don't make sense" markdown note (cell 27): reference
  this plan, list fixed bugs, model choice.
- Update the README modeling section: bug fixes, model choices, how to re-run on a
  subsample.

## 13. Validation

**Prerequisites (must complete before validation):**
- Regenerate samples from `data/raw/*.parquet` first (the repo copies in
  `data/samples/` are tiny placeholders — 504 lines in `1.2/`); see notebook cell 7.
  This is a hard blocker for validating any of the 6 fixes on representative data.

**Validation steps:**
- Rebuild 1.3 cells on a small sample (e.g. 5k docs): confirm real date spans
  (2023-2025), <20% outliers, legends match topic names, spot-check sentiment on
  known-negative topics.
- Run VADER vs. RoBERTa comparison visualization; confirm agreement rate > 60%.
- Verify experiment log captures all sweep parameters and metrics.
- `make test` (pytest, see Makefile in consolidation plan).

## 14. Caveats

- Requires a Hugging Face model download for the sentiment model; GPU strongly
  recommended for large samples.
- Samples in `data/samples/` are tiny placeholders - validation needs regenerating
  real samples.
- `gensim` dependency needed for topic coherence (c_v) evaluation — add to requirements.
- Optional follow-up (NOT in this pass): fine-tune roberta on hand-labeled Reddit data.

## Files touched (when implemented)

- `src/modeling/` (new module: `__init__.py`, `load.py`, `text.py`, `topics.py`, `sentiment.py`, `cache.py`, `tracking.py`)
- `notebooks/1.3_notebook.ipynb` (cells rewritten)
- `config/config.yaml` (`modeling:` section)
- `tests/test_{text,dates,sentiment,topics,cache,tracking,reproducibility}.py`
- `requirements_cpu.txt`, `requirements_gpu.txt` (add `gensim`, `transformers`, `tqdm`)
- `README.md` modeling section
