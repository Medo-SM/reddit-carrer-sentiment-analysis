# ToDo

See [notebook-architecture.md](notebook-architecture.md) for the architecture reference and [research-plan.md](research-plan.md) for the research context and RQs.

---

## Step 1 — Complete the Stratified Sampler

- [x] Get the path of the parquet files (`src/ingest/sampling.py::get_parquet_from_path`)
- [x] Run stratified sampler for all 7 existing subreddits (`2018-01-01` to `2022-12-31`)
  - Sample up to 30k rows per subreddit/type file, stratified by ISO week
  - Write samples to `data/samples/2.0/`
- [ ] **Phase 2 (Deferred — requires explicit download approval):** Scrape `2023–2025` + `r/ExperiencedDevs` and `r/csMajors` via `src/ingest/scraper/core_scraper.py`

---

## Step 2 — Build `src/modeling/`

- [x] `src/modeling/__init__.py` — `set_seed()` utility
- [x] `src/modeling/load.py` — dtype-aware `parse_created_utc()`, `load_samples()`, `draw_stratified_subset()` (fixes Bugs 1 & 5)
- [x] `src/modeling/text.py` — dual cleaners: `apply_sentiment_cleaning()` & `apply_topic_cleaning()`
- [x] `src/modeling/cache.py` — embeddings, topic model, and Parquet score cache
- [x] `src/modeling/tracking.py` — lightweight JSONL experiment log
- [x] `src/modeling/topics.py` — BERTopic wrapper + `topic_name_map()` + NPMI/UMass/Diversity evaluation + `run_hdbscan_sweep()` + `reduce_outliers_safe()` (fixes Bugs 2, 3, 4)
- [x] `src/modeling/sentiment.py` — genuine RoBERTa batched inference + VADER cross-check + Cohen's $\kappa$ & macro-F1 (fixes Bug 6)
- [x] `src/modeling/stats.py` — quarterly aggregation, Mann-Kendall trend test, 4-regime Kruskal-Wallis/Mann-Whitney + Holm-Bonferroni tests

---

## Step 3 — Notebooks & Runner

- [x] `notebooks/2.0_sampling.ipynb` — sampling + date coverage audit
- [x] `notebooks/3.0_topic_modeling.ipynb` — BERTopic + HDBSCAN sweep
- [x] `notebooks/4.0_sentiment_analysis.ipynb` — genuine RoBERTa + VADER comparison
- [x] `notebooks/5.0_results_and_viz.ipynb` — final figures & statistical summary
- [x] `scripts/run_pipeline.py` — consolidated CLI orchestrator
- [x] Move `notebooks/1.3_notebook.ipynb` → `notebooks/archive/`

---

## Step 4 — Tests

- [x] `tests/test_cleaning.py`, `tests/test_sampling.py`, `tests/test_scraper_utils.py`
- [x] `tests/test_text.py`, `tests/test_dates.py`, `tests/test_cache.py`, `tests/test_tracking.py`, `tests/test_sentiment.py`, `tests/test_topics.py`, `tests/test_stats.py`

---

## Step 5 — Config + Bibliography Verification

- [x] Consolidated `config/config.yaml` (7 subreddits, `2018-01-01`–`2022-12-31`, `modeling:` section)
- [x] Verified core citations via arXiv & OpenAlex into `data/final/references_verified.md` (excluding `arXiv:2311.15800` mental-health misattribution)

---

## Step 6 — Remaining Research Tasks (Human / Phase 2)

- [ ] Execute Phase 2 scraping (2023–2025 + `r/ExperiencedDevs` + `r/csMajors`) for full post-ChatGPT ITS regression
- [ ] Post pre-registration on OSF
- [ ] Double-annotate 600-post sample for 4-class taxonomy & freeze LLM labeling prompt
