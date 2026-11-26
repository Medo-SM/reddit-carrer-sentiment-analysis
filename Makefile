.PHONY: test test-fast compile smoke-test lint clean pipeline all help

VENV_PY := $(shell if [ -x "$$HOME/.venvs/data/bin/python" ]; then echo "$$HOME/.venvs/data/bin/python"; else echo "python3"; fi)
PYTHON ?= $(VENV_PY)

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run all tests with pytest (including slow model integration test)
	$(PYTHON) -m pytest tests/ -v

test-fast: ## Run fast unit tests excluding slow-marked tests
	$(PYTHON) -m pytest tests/ -v -m "not slow"

compile: ## Compile-check all source files (excluding archive)
	$(PYTHON) -m compileall -q -x '(archive|__pycache__)' src/ scripts/ && echo "compile OK"

smoke-test: ## Verify config, ingestion, and modeling imports resolve cleanly
	@$(PYTHON) -c "import yaml; yaml.safe_load(open('config/config.yaml')); print('config OK')"
	@$(PYTHON) -c "from src.modeling import set_seed; from src.modeling.load import parse_created_utc; import pandas as pd; s = parse_created_utc(pd.Series([1700000000])); assert s.dt.year.iloc[0] == 2023; print('modeling OK:', s.iloc[0])"
	@$(PYTHON) -c "from src.ingest.sampling import clean_sentiment_text, process_pipeline; print('ingest OK')"

lint: ## Run py_compile on all Python files in src/ and scripts/ (excluding archive)
	@find src/ -path '*/archive' -prune -o -name '*.py' -print -exec $(PYTHON) -m py_compile {} \; >/dev/null 2>&1 && echo "src lint OK"
	@find scripts/ -name '*.py' -exec $(PYTHON) -m py_compile {} \; >/dev/null 2>&1 && echo "scripts lint OK"

pipeline: ## Run full consolidated modeling pipeline (Phases 2-5)
	$(PYTHON) scripts/run_pipeline.py

clean: ## Remove __pycache__ dirs and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "clean OK"

all: compile test-fast smoke-test ## Run compile + test-fast + smoke-test
