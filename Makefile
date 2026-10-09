PY ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: help venv install lint format test seed eval triage api docker clean

help:  ## Show targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

venv:  ## Create virtualenv
	$(PY) -m venv $(VENV)

install: venv  ## Install package + dev deps (editable)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

lint:  ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

test:  ## Run tests
	$(BIN)/pytest

seed:  ## Build the synthetic SQLite DB
	$(BIN)/python -m claims_agent.data.seed --db var/claims.db

eval: seed  ## Run offline evaluation -> reports/eval_report.md
	$(BIN)/python -m claims_agent.evaluate --out reports/eval_report.md

triage:  ## Example: make triage CLAIM=CLM-1004
	$(BIN)/python -m claims_agent triage $(or $(CLAIM),CLM-1004)

api:  ## Run the FastAPI service on :8000
	$(BIN)/uvicorn claims_agent.api:app --reload --port 8000

docker:  ## Build the container image
	docker build -t claims-triage-agent:local .

clean:  ## Remove caches, DB and traces
	rm -rf var traces .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
