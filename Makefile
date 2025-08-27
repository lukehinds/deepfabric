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

format:
	uv run black .
	uv run ruff check --fix .

lint:
	uv run ruff check .

test:
	uv run pytest

security:
	uv run bandit -r promptwright/

build: clean test
	uv build

all: clean install format lint test security build