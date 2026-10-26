.PHONY: test test-fast smoke-test lint compile clean all help

PYTHON ?= $(shell if [ -f $(HOME)/.venvs/data/bin/python ]; then echo $(HOME)/.venvs/data/bin/python; else echo python3; fi)
PYTEST ?= $(shell if [ -f $(HOME)/.venvs/data/bin/pytest ]; then echo $(HOME)/.venvs/data/bin/pytest; else echo pytest; fi)

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

test: ## Run all tests including slow
	$(PYTEST) tests/ -v

test-fast: ## Run tests excluding slow (< 60 s)
	$(PYTEST) tests/ -v -m "not slow"

smoke-test: ## Verify key imports work and timestamps parse correctly
	@$(PYTHON) -c "from src.modeling import set_seed; set_seed(42); print('set_seed OK')"
	@$(PYTHON) -c "\
from src.modeling.load import parse_created_utc; import pandas as pd; \
ts = parse_created_utc(pd.Series([1700000000])); \
assert ts.dt.year[0] == 2023, f'Expected 2023, got {ts.dt.year[0]}'; \
print('parse_created_utc OK — got', ts[0])"

lint: ## py_compile all source files
	@find src/ -name '*.py' -exec $(PYTHON) -m py_compile {} \; && echo "lint OK"

compile: ## compileall on src/
	$(PYTHON) -m compileall -q src/ && echo "compile OK"

clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	@echo "clean OK"

all: compile lint test-fast smoke-test ## Full check (excludes slow tests)
