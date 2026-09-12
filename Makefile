.PHONY: test compile smoke-test lint clean all help

PYTHON ?= python3

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run all tests with pytest
	pytest tests/ -v

test-fast: ## Run tests excluding slow-marked tests
	pytest tests/ -v -m "not slow"

compile: ## Compile-check all source files (excluding archive)
	$(PYTHON) -m compileall -q -x '(archive|__pycache__)' src/ && echo "compile OK"

smoke-test: ## Verify key imports and config resolve correctly
	@$(PYTHON) -c "import yaml; yaml.safe_load(open('config/config.yaml'))" && echo "config OK"
	@$(PYTHON) -m py_compile src/ingest/zst_to_parquet.py && echo "zst_to_parquet OK"
	@$(PYTHON) -m py_compile src/ingest/sampling.py && echo "sampling OK"
	@$(PYTHON) -m py_compile src/ingest/scraper/core_scraper.py && echo "core_scraper OK"

lint: ## Run py_compile on all Python files in src/ and scripts/ (excluding archive)
	@find src/ -path '*/archive' -prune -o -name '*.py' -print -exec $(PYTHON) -m py_compile {} \; >/dev/null 2>&1 && echo "src lint OK"
	@find scripts/ -name '*.py' -exec $(PYTHON) -m py_compile {} \; >/dev/null 2>&1 && echo "scripts lint OK"

clean: ## Remove __pycache__ dirs and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "clean OK"

all: compile test smoke-test ## Run compile + test + smoke-test
