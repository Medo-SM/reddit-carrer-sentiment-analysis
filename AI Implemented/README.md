# AI Implemented — Reddit Career Sentiment Analysis

This directory contains a complete, independent implementation of the research pipeline for:

> **"Tech Career Sentiment After ChatGPT"**  
> Target Venue: ICWSM 2027  
> Analysis Window: 2020–2025 | Break Date: November 30, 2022 | Seed: 42

This workspace serves as a clean, reference implementation created by AI that mirrors the root project structure while remaining strictly isolated.

---

## Workspace Isolation Rules

- **Zero Outside Mutations:** All outputs, cached models, and generated artifacts are written strictly inside `AI Implemented/`.
- **Shared Data Access:** Shared raw files in `data/raw/` and historical samples in `data/samples/` are accessed in read-only mode via relative paths in `config/config.yaml`.
- **Target Subreddits (6 Existing):** Operates on the 6 subreddits with active parquet data (`r/cscareerquestions`, `r/cscareerquestionsEU`, `r/datascience`, `r/dataengineering`, `r/cybersecurity`, `r/SoftwareEngineering`).

---

## Directory Layout

```
AI Implemented/
├── config/
│   └── config.yaml             # Workspace configuration
├── src/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── sampling.py         # Sampling utilities (stratified + random)
│   │   └── zst_to_parquet.py   # Historical ZST conversion
│   └── modeling/
│       ├── __init__.py         # set_seed() reproducibility utility
│       ├── load.py             # Dtype-aware timestamp parsing + load_samples()
│       ├── text.py             # apply_pipeline() text cleaning
│       ├── cache.py            # Intermediate embedding and model caching
│       ├── tracking.py         # JSONL experiment log
│       ├── topics.py           # BERTopic fitting, name map, HDBSCAN sweep
│       └── sentiment.py        # RoBERTa batched inference + VADER comparison
├── notebooks/
│   ├── 2.0_sampling.ipynb      # Stratified sampling & temporal coverage audit
│   ├── 3.0_topic_modeling.ipynb# BERTopic modeling & HDBSCAN sweep
│   ├── 4.0_sentiment_analysis.ipynb # RoBERTa scoring & VADER cross-check
│   └── 5.0_results_and_viz.ipynb# Publication figures & ITS analysis
├── tests/
│   ├── test_text.py
│   ├── test_dates.py
│   ├── test_cache.py
│   ├── test_tracking.py
│   ├── test_sentiment.py
│   └── test_topics.py
├── plans/
│   └── IMPLEMENTATION_PLAN.md  # Detailed implementation specification
├── requirements_cpu.txt
├── requirements_gpu.txt
├── pytest.ini
└── Makefile
```

---

## Bug Fixes Implemented

1. **Bug 1 (Timestamps collapse to 1970-01-01):** Resolved in `src/modeling/load.py::parse_created_utc` by detecting epoch seconds vs. ISO 8601 strings.
2. **Bug 2 (Legend labels index wrong object):** Resolved in `src/modeling/topics.py::topic_name_map` using `model.get_topic_info()`.
3. **Bug 3 (High outlier rate ~48%):** Resolved in `src/modeling/topics.py::run_hdbscan_sweep` by sweeping smaller `min_cluster_size` values [15, 25, 35, 50] and `min_samples` [5, 10, 15].
4. **Bug 4 (Double embedding computation):** Resolved in `src/modeling/topics.py::fit_topic_model` by pre-computing embeddings once on `cleaned_text` and passing directly to `fit_transform()`.
5. **Bug 5 (Dead code TypeError in sample loader):** Resolved in `src/modeling/load.py::load_samples` with explicit parameters.
6. **Bug 6 (VADER domain mismatch & neutral spike):** Resolved in `src/modeling/sentiment.py` by introducing `cardiffnlp/twitter-roberta-base-sentiment` as primary model with batched GPU/CPU inference, keeping VADER as a cross-check.

---

## Quickstart & Verification

From inside `AI Implemented/`:

```bash
# Verify configuration and timestamp parsing
make smoke-test

# Compile and lint Python modules
make compile
make lint

# Run unit tests
make test-fast
```
