# Notebook Architecture Plan

Oct 19, 2026 · @Mohammed  
**Status: Active — self-build guide, research paper target, fully reproducible**

---

## Core Principle

Notebooks are **documents**. `src/` is **code**.

| Layer | What goes here | Rule |
|---|---|---|
| `src/modeling/` | Functions, classes, utilities | Must be importable and testable |
| `notebooks/` | Narrative + thin calls into `src/` | No function longer than ~5 lines |
| `data/` | Input/output files only | Never raw data in Git |
| `tests/` | Unit tests for `src/` only | Notebooks are not tested |
| `config/config.yaml` | All tunable parameters | No magic numbers in notebook cells |

---

## File Structure

```
reddit-carrer-sentiment-analysis/
│
├── config/
│   └── config.yaml                   ← MODIFY: add modeling: section
│
├── data/
│   ├── raw/                          ← .parquet/.zst dumps (gitignored)
│   ├── samples/
│   │   └── 2.0/                      ← stratified samples (written by notebook 2.0)
│   ├── processed/
│   │   ├── cache/                    ← embeddings .npy, saved topic models
│   │   └── experiments/
│   │       └── experiment_log.jsonl  ← one line per modeling run
│   └── final/                        ← paper-ready CSVs and PNGs
│
├── notebooks/
│   ├── 1.0_data_exploration.ipynb    ← already exists ✅ (keep as-is)
│   ├── 2.0_sampling.ipynb            ← NEW (you write)
│   ├── 3.0_topic_modeling.ipynb      ← NEW (you write)
│   ├── 4.0_sentiment_analysis.ipynb  ← NEW (you write)
│   ├── 5.0_results_and_viz.ipynb     ← NEW (you write)
│   └── archive/
│       ├── 1.1_notebook.ipynb        ← already archived ✅
│       ├── 1.2_notebook.ipynb        ← already archived ✅
│       └── 1.3_notebook.ipynb        ← MOVE HERE when notebooks 2.0–5.0 are done
│
├── src/
│   ├── ingest/                       ← already exists ✅
│   └── modeling/                     ← NEW module (build Phase 1 first)
│       ├── __init__.py               ← set_seed()
│       ├── load.py                   ← parse_created_utc() + load_samples()
│       ├── text.py                   ← apply_pipeline() over a Series
│       ├── sentiment.py              ← score_roberta() + score_vader() + compare_models()
│       ├── topics.py                 ← fit_topic_model() + topic_name_map()
│       ├── cache.py                  ← save/load embeddings + topic models
│       └── tracking.py              ← log_experiment() → JSONL
│
└── tests/
    ├── test_text.py
    ├── test_dates.py
    ├── test_sentiment.py
    ├── test_topics.py                ← mark slow
    ├── test_cache.py
    └── test_tracking.py
```

---

## Pipeline Flow

```
data/raw/ → [2.0 sampling] → data/samples/2.0/
         → [3.0 topic modeling] → data/processed/cache/ + experiment_log.jsonl
         → [4.0 sentiment] → data/processed/sentiment_df.parquet
         → [5.0 results] → data/final/*.png + *.csv
```

Each notebook reads from the previous notebook's output directory. Never skip ahead.

---

## Build Checklist

### Phase 1 — Build `src/modeling/` (do this before any notebook)

- [ ] `src/modeling/__init__.py` — `set_seed(seed=42)`
- [ ] `src/modeling/load.py` — `parse_created_utc()` (fixes Bug 1) + `load_samples()`
- [ ] `src/modeling/text.py` — `apply_pipeline(series)` wrapping `src.ingest.sampling.process_pipeline`
- [ ] `src/modeling/cache.py` — `save_embeddings()`, `load_embeddings()`, `save_topic_model()`, `load_topic_model()`
- [ ] `src/modeling/tracking.py` — `log_experiment(params, metrics)` → JSONL
- [ ] `src/modeling/topics.py` — `fit_topic_model()` (fixes Bugs 3 + 4) + `topic_name_map()` (fixes Bug 2)
- [ ] `src/modeling/sentiment.py` — `score_roberta()`, `score_vader()`, `compare_models()` (fixes Bug 6)

**Smoke test after Phase 1:**
```bash
python -c "from src.modeling import set_seed; set_seed(42); print('OK')"
python -c "from src.modeling.load import parse_created_utc; import pandas as pd; print(parse_created_utc(pd.Series([1700000000])))"
# Should print a 2023 date, not 1970
```

---

### Phase 2 — Build Notebooks (one at a time, in order)

#### `2.0_sampling.ipynb` (~10 cells)
Concern: get clean, representative samples + date audit.
```
[1]  imports + config + set_seed(42)
[2]  load parquet paths from config
[3]  run sample_parquet_file_to_json() → data/samples/2.0/
[4]  load one sample with load_samples() to verify
[5]  print: row count, date range (expect 2022–2025, not 1970), subreddits
[6]  plot: weekly row count per subreddit (temporal coverage audit)
[7]  plot: missing weeks heatmap (spot API blackout gaps)
[8]  save audit to data/final/coverage_audit.csv
```

#### `3.0_topic_modeling.ipynb` (~15 cells)
Concern: fit stable topics, run HDBSCAN sweep.
```
[1]  imports + set_seed(42)
[2]  load_samples() + apply_pipeline()
[3]  load_embeddings() or compute + save_embeddings() [cache-aware]
[4]  HDBSCAN sweep: for min_cluster_size in [15, 25, 35, 50]:
         fit_topic_model(...), compute outlier_pct + coherence_cv, log_experiment(...)
[5]  print sweep results from experiment_log.jsonl
[6]  pick best config (outlier_pct < 20%, coherence_cv reasonable)
[7]  refit final model + save_topic_model()
[8]  print topic_name_map(model)
[9]  2D UMAP scatter plot (color by topic, label from name_map) — fixes Bug 2
[10] save figure to data/final/topic_scatter.png
```

> [!NOTE]
> Sweep `min_cluster_size` values of [15, 25, 35, 50] — lower than the original 100 which caused 48% outliers (Bug 3).

#### `4.0_sentiment_analysis.ipynb` (~12 cells)
Concern: score sentiment, validate model choice (RQ1).
```
[1]  imports + set_seed(42)
[2]  load_samples()
[3]  apply_pipeline() for cleaned_text
[4]  score_vader(df['cleaned_text'])
[5]  score_roberta(df['cleaned_text']) [try loading from cache first]
[6]  merge scores, add polarity label columns
[7]  compare_models(df) → print agreement rate (expect > 60%)
[8]  plot: VADER compound vs RoBERTa pos prob scatter (RQ1 justification figure)
[9]  plot: top-20 disagreements table (shows actual text)
[10] plot: sentiment distribution by subreddit (stacked bar)
[11] plot: sentiment over time (weekly, RoBERTa primary)
[12] save data/processed/sentiment_df.parquet
```

#### `5.0_results_and_viz.ipynb` (~10 cells)
Concern: final paper figures only.
```
[1]  imports + load sentiment_df.parquet + load topic model
[2]  Figure 1: topic distribution by subreddit (stacked bar)
[3]  Figure 2: topic evolution over time (line, top-5 topics) — RQ2
[4]  Figure 3: sentiment trend weekly pre/post ChatGPT — RQ3
[5]  Figure 4: sentiment × topic cross-tab heatmap — RQ3
[6]  Figure 5: VADER vs RoBERTa comparison — RQ1
[7]  export all figures → data/final/ as 300dpi PNG
[8]  export summary tables → data/final/ as CSV
[9]  print key statistics to cite in paper
```

**Every figure in the paper must be generated by a cell in this notebook. Never export manually.**

---

### Phase 3 — Tests

| File | What to test | Difficulty |
|---|---|---|
| `tests/test_text.py` | `apply_pipeline` on URLs, code blocks, None, empty strings | ⭐ Easy |
| `tests/test_dates.py` | `parse_created_utc` for int / float / str / datetime inputs; verify 2023 output | ⭐ Easy |
| `tests/test_cache.py` | save → load round-trip for embeddings and topic models | ⭐⭐ Medium |
| `tests/test_tracking.py` | `log_experiment` writes a valid JSON line; reads back correctly | ⭐⭐ Medium |
| `tests/test_sentiment.py` | VADER on known-positive/negative text; RoBERTa output shape check | ⭐⭐ Medium |
| `tests/test_topics.py` | `topic_name_map` shape; fit on ~1k synthetic docs; `@pytest.mark.slow` | ⭐⭐⭐ Slow |

```bash
pytest tests/ -v -m "not slow"   # fast only
pytest tests/ -v                  # all, including slow
pytest tests/test_dates.py -v     # one file at a time while writing
```

---

### Phase 4 — Config + Cleanup

**Add to `config/config.yaml`:**
```yaml
modeling:
  embedding_model: "all-MiniLM-L6-v2"
  umap:
    n_neighbors: 15
    n_components: 5
    min_dist: 0.0
    metric: cosine
    random_state: 42          # DO NOT change — reproducibility
  hdbscan:
    min_cluster_size: 25      # placeholder — set after sweep in notebook 3.0
    min_samples: 5            # placeholder — set after sweep in notebook 3.0
    metric: euclidean
    cluster_selection_method: eom
    prediction_data: true
  top_n_words: 10
  sentiment_model: "cardiffnlp/twitter-roberta-base-sentiment"
  sentiment_batch_size: 64
  device: "auto"              # auto, cuda, cpu, mps
  max_sample_cpu: 5000        # fallback for CPU-only environments
  seed: 42
  cache_dir: "data/processed/cache"
  experiment_log_dir: "data/processed/experiments"
  dataset_types:
    - submissions
    - comments
```

**Move `notebooks/1.3_notebook.ipynb` → `notebooks/archive/`** only after notebook 5.0 runs end-to-end successfully.

---

## Bug Fix Reference

| Bug | Location in 1.3 | Fix |
|---|---|---|
| Bug 1: 1970-01-01 dates | Cell 28: `pd.to_datetime(df['created_utc'])` | `parse_created_utc()` in `load.py` |
| Bug 2: wrong legend labels | Cells 35/37: `topics[topic_id]` indexes doc array | `topic_name_map(model)` in `topics.py` |
| Bug 3: 48% outliers | `HDBSCAN(min_cluster_size=100)` too strict | Sweep [15, 25, 35, 50] in notebook 3.0 |
| Bug 4: embeddings computed twice | Pre-computed embeddings not passed to `fit_transform` | Compute on `cleaned_text`, pass to `fit_topic_model()` |
| Bug 5: dead code TypeError | Cell 14: `json_to_df(path, dataset_type)` wrong arg order | Use `load_samples()` in notebook 2.0 |
| Bug 6: VADER domain mismatch | Primary sentiment model | Replace with RoBERTa; keep VADER as cross-check |

---

## Learning Resources

| Topic | Resource |
|---|---|
| Why BERTopic over LDA | BERTopic paper: arXiv:2203.05794 |
| Why RoBERTa over VADER | Cardiff NLP TweetEval: arXiv:2010.12421 |
| Reproducibility in ML | Pineau et al. ML Reproducibility Checklist |
| Interrupted time series | Cunningham, *Causal Inference: The Mixtape* (free online) |
| ICWSM paper format | Read 2–3 recent ICWSM papers on Reddit career discourse |
