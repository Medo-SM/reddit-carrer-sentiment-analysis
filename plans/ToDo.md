# ToDo

See `plans/notebook-architecture.md` for the full build guide with cell-by-cell instructions.  
See `plans/research-plan.md` for the research context, RQs, and timeline.

---

## Step 1 — Complete the Stratified Sampler

- [x] Get the path of the parquet files (`src/ingest/sampling.py::get_parquet_from_path`)
- [ ] Add `r/ExperiencedDevs` and `r/csMajors` to scraper targets (needed for RQ3 seniority split)
- [ ] Run stratified sampler for all subreddits
  - Sample ~30k rows per subreddit/type file
  - Stratify by week using `created_utc` to preserve temporal distribution
  - Write samples to `data/samples/2.0/`

---

## Step 2 — Build `src/modeling/`

Create the modeling module (see `plans/notebook-architecture.md` for skeletons and `plans/modeling-rework.md` for technical detail):

- [ ] `src/modeling/__init__.py` — `set_seed()` utility
- [ ] `src/modeling/load.py` — dtype-aware `parse_created_utc()` + `load_samples()` (fixes Bug 1)
- [ ] `src/modeling/text.py` — `apply_pipeline()` wrapping `src.ingest.sampling.process_pipeline`
- [ ] `src/modeling/cache.py` — embeddings and topic model save/load
- [ ] `src/modeling/tracking.py` — lightweight JSONL experiment log
- [ ] `src/modeling/topics.py` — BERTopic wrapper + `topic_name_map()` + HDBSCAN sweep (fixes Bugs 2, 3, 4)
- [ ] `src/modeling/sentiment.py` — RoBERTa batched inference + VADER cross-check (fixes Bug 6)

---

## Step 3 — Build New Notebooks

Replace `1.3_notebook.ipynb` with 4 focused notebooks (archive 1.3 when done):

- [ ] `notebooks/2.0_sampling.ipynb` — sampling + date audit (replaces cells 1–16 of 1.3)
- [ ] `notebooks/3.0_topic_modeling.ipynb` — BERTopic + HDBSCAN sweep (replaces cells 28–34 of 1.3)
- [ ] `notebooks/4.0_sentiment_analysis.ipynb` — RoBERTa + VADER comparison (replaces cells 29–31 of 1.3; adds RQ1 figures)
- [ ] `notebooks/5.0_results_and_viz.ipynb` — final paper figures (replaces cells 35–38 of 1.3)
- [ ] Move `notebooks/1.3_notebook.ipynb` → `notebooks/archive/` after notebook 5.0 runs end-to-end

---

## Step 4 — Write Tests

- [ ] `tests/test_text.py` — `apply_pipeline` on URLs, code blocks, None, empty strings
- [ ] `tests/test_dates.py` — `parse_created_utc` for int / float / str / datetime inputs
- [ ] `tests/test_cache.py` — embeddings and topic model save/load round-trip
- [ ] `tests/test_tracking.py` — experiment log write/read round-trip
- [ ] `tests/test_sentiment.py` — VADER on known text; RoBERTa output shape check
- [ ] `tests/test_topics.py` — `topic_name_map` shape + fit on ~1k docs (`@pytest.mark.slow`)

---

## Step 5 — Config + Requirements

- [ ] Add `modeling:` section to `config/config.yaml` (see `plans/notebook-architecture.md` §Phase 4)
- [ ] Add `gensim`, `transformers`, `tqdm` to `requirements_cpu.txt` and `requirements_gpu.txt`

---

## Step 6 — Research Tasks (see `plans/research-plan.md` for full detail)

- [ ] Post pre-registration on OSF (deadline: 24 October 2026)
- [ ] Fix citation errors in original proposal before sharing externally
- [ ] Double-annotate 600-post sample for 4-class taxonomy
- [ ] Freeze LLM labeling prompt on development set
