# Plan: Consolidate & Clean Project Structure

## Detail decisions locked in

- **Canonical scraper:** Merge the robust logic of `core_scraper_fixed.py` (strict monotonic cursor, stall guard, crash-safe `.state.json` resume, `run_main()`/nest_asyncio entry) with the Arctic Shift endpoint + HTTP-status handling + 1s pacing from `core_scraper_arctic.py`.
- **Old scrapers** -> moved to an archive folder (git history preserves originals).
- **All paths -> config.yaml**; repair scripts also get a target-file parameter.
- **Config wiring** spans `src/ingest/` **and** notebooks.
- **Dependencies:** add `pytest` to requirements files; trim `notebooks/requirements.txt` to actual deps.
- **Modeling:** document the known BERTopic issue, defer fixing.
- **Repair scripts** -> move to `scripts/` (operational tools, not importable pipeline modules).

## 1. Canonical merged scraper (`src/ingest/scraper/core_scraper.py`)

One clean file combining:

- Arctic Shift base URL + `API_MODE` posts/comments mapping
- `HourlyRateLimiter` + `RateLimitHardStop` from `core_scraper_fixed.py`
- Monotonic `(created_utc, id)` cursor filtering + stall guard
- Crash-safe sidecar `.state.json` (`load_state`/`save_state`, atomic write)
- Robust `get_last_ts_id_and_count()` tolerant of corrupt trailing lines
- Non-200/429 HTTP-status handling with backoff + failure cap
- `run_main()` working as both a plain script and inside Jupyter/Colab
- **Type hints** on all public functions and state dict structures
- **`logging` module** instead of `print()` — configurable via `logging.basicConfig(level=...)` so output verbosity can be controlled without code changes

Move the three legacy versions to `src/ingest/scraper/archive/`.

### CLI interface

The canonical scraper accepts CLI arguments via `argparse` for one-off runs:

```
python -m src.ingest.scraper.core_scraper \
  --subreddit cscareerquestions \
  --mode posts \
  --start 2023-01-01 \
  --end 2025-01-01 \
  --log-level DEBUG
```

Config values become defaults; CLI args override them. This makes the scraper
usable for ad-hoc data collection without editing `config.yaml`.

## 2. Parameterize everything via `config/config.yaml`

Extend config with:

- `scraper:` - `api_base`, `data_dir`, `subreddits`, `start_date`, `end_date`, `chunk_days`, rate limits (`rate_limit_per_hour`, `min_429_wait_sec`, `max_consecutive_429`, `hard_stop_wait_sec`), `request_delay_sec`, `user_agent`, `max_consecutive_failures`
- `analysis:` - `subreddit`, `sample_size`, `dataset_type` (used by notebooks)
- Scripts read config with a `CONFIG_PATH` env override; no more Colab-hardcoded `/content/...` paths.
- Rewire `zst_to_parquet.py` to read input/output dirs from config instead of CWD-relative paths.
- Refactor `corrupted_line_locator.py`, `corruption_fix.py`, `missing_dates_check.py` to accept a target JSONL path (from config + optional CLI arg) rather than a hardcoded file.
- Notebooks: standardize config-loading cells to resolve paths robustly; move `sample_size`/`subreddit`/`dataset_type` into `analysis:`. Keep `notebook_ver` per-notebook (values differ across notebooks).

## 3. Extract shared notebook utilities into `src/ingest/sampling.py`

All 4 notebooks redefine identical functions inline:

- `get_parquet_from_path()`
- `sample_parquet_file_to_json()`
- `json_to_df()`
- `clean_code_snippets()`
- `clean_text()`
- `process_pipeline()`

Extract these into `src/ingest/sampling.py` so notebooks become thin calls:

```python
from src.ingest.sampling import get_parquet_from_path, sample_parquet_file_to_json, json_to_df
from src.ingest.sampling import process_pipeline
```

This also directly benefits Plan 2 — `src/modeling/text.py` imports `clean_text`
from here instead of re-implementing it.

## 4. Package structure fixes

- Rename misspelled `__innit__.py` -> `__init__.py` in `src/`, `src/ingest/`, `src/ingest/scraper/`, `tests/`.
- Delete stray uncommitted duplicate `data/zst_to_parquet.py`.
- Move repair scripts to `scripts/` directory (they are operational tools, not importable pipeline modules):
  - `src/ingest/scraper/corrupted_line_locator.py` -> `scripts/corrupted_line_locator.py`
  - `src/ingest/scraper/corruption_fix.py` -> `scripts/corruption_fix.py`
  - `src/ingest/scraper/missing_dates_check.py` -> `scripts/missing_dates_check.py`
- Verify package imports cleanly (`from src.ingest import zst_to_parquet`, `from src.ingest.scraper.core_scraper import main`).

## 5. Tests (unified: `pytest`)

**Decision:** use `pytest` everywhere (both this plan and the modeling rework).
`pytest` runs `unittest.TestCase` subclasses natively, so no migration cost. Add
`pytest` to `requirements_cpu.txt` and `requirements_gpu.txt`.

- `tests/test_scraper_utils.py` covering: `HourlyRateLimiter` pacing, monotonic cursor filtering, stall-guard advance, and `load_state`/`save_state` round-trip + corrupt-trailing-line resilience.
- `tests/test_cleaning.py` covering `clean_text()` in `zst_to_parquet.py`.
- `tests/test_sampling.py` covering the extracted `src/ingest/sampling.py` functions.

## 6. Dependency cleanup

- **Add `pytest`** to `requirements_cpu.txt` and `requirements_gpu.txt`.
- **Trim `notebooks/requirements.txt`** from 113 packages to actual dependencies only. The current file is a `pip freeze` dump that includes unrelated packages (`fastapi`, `psycopg`, `BAScraper`, `starlette`, `pigar`, `pipreqs`, `docopt`, `yarg`, etc.). Replace with a curated list of ~30 actually-used packages. Add a comment header noting it's a curated list.
- **Add `.env.example`** documenting `CONFIG_PATH` and other env var overrides (`HF_HOME` for model caching, `CUDA_VISIBLE_DEVICES`).

## 7. Makefile

Add a `Makefile` with targets for repeatable verification:

```makefile
.PHONY: test compile lint smoke-test clean

test:
	pytest tests/ -v

compile:
	python -m compileall src/

smoke-test:
	python -c "from src.ingest.scraper.core_scraper import run_main; print('scraper OK')"
	python -c "from src.ingest.zst_to_parquet import main; print('zst_to_parquet OK')"

lint:
	python -m py_compile src/ingest/scraper/core_scraper.py
	python -m py_compile src/ingest/zst_to_parquet.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

all: compile test smoke-test
```

## 8. Modeling documentation

- Add a markdown note in `notebooks/1.3_notebook.ipynb` and a short README section stating BERTopic topic coherence and VADER-vs-subjectivity are unvalidated; re-tuning deferred.

## Verification

- `make test` (runs `pytest tests/ -v`)
- `make compile` (compiles all `src/`)
- `make smoke-test` (config load + import check for canonical scraper)
- Notebooks: re-open/parse JSON to confirm cells still valid (outputs preserved where possible)

## Files touched

- `src/ingest/scraper/core_scraper.py` (rebuilt), `archive/` (3 moved files)
- `src/ingest/zst_to_parquet.py`
- `src/ingest/sampling.py` (new — extracted from notebooks)
- `scripts/corrupted_line_locator.py`, `scripts/corruption_fix.py`, `scripts/missing_dates_check.py` (moved from `src/ingest/scraper/`)
- `config/config.yaml`
- `__innit__.py` x4 -> `__init__.py`, `tests/test_*.py` (new)
- `notebooks/*.ipynb` (config cells + import shared module)
- `requirements_cpu.txt`, `requirements_gpu.txt` (add pytest)
- `notebooks/requirements.txt` (trimmed)
- `Makefile` (new)
- `.env.example` (new)
- `data/README.md` (new — documents the data pipeline flow)
- README, delete `data/zst_to_parquet.py`

## Nuance flagged during planning

Since each notebook uses its own `notebook_ver`/subreddit, those stay per-notebook rather than forcing one shared value; only the genuinely shared params go into `analysis:`.
