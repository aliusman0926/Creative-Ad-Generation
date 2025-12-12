PYTHON := python

.PHONY: install lint format test build run-local

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install .[dev]

lint:
	pre-commit run --all-files

format:
	black src tests
	isort src tests
	ruff check --fix src tests

test:
	pytest

build:
	$(PYTHON) -m build

run-local:
	uvicorn src.api.main:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000}
