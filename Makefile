.PHONY: clean install format lint test security build all

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -f .coverage
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

install:
	uv sync --all-extras

format: ## Format code with ruff (parallel)
	uv run ruff format deepfabric/ tests/

lint:
	uv run ruff check . --exclude notebooks/

test-unit:
	uv run pytest tests/unit/

PYTEST_ADDOPTS ?=

test-integration:
	pytest $(PYTEST_ADDOPTS) tests/integration/

.PHONY: test-integration-verbose
test-integration-verbose:
	pytest -v -rA --durations=10 tests/integration/

security:
	uv run bandit -r deepfabric/

build: clean test
	uv build

all: clean install format lint test security build
