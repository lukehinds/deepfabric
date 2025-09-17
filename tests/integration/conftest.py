import os
import tempfile

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir():
    """Return path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def test_config_path(fixtures_dir):
    """Return path to test configuration file."""
    return fixtures_dir / "test_config.yaml"


@pytest.fixture
def graph_config_path(fixtures_dir):
    """Return path to graph configuration file."""
    return fixtures_dir / "graph_config.yaml"


@pytest.fixture
def sample_tree_path(fixtures_dir):
    """Return path to sample tree file."""
    return fixtures_dir / "sample_tree.jsonl"


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(autouse=True)
def github_token_check():
    """Skip integration tests if GITHUB_TOKEN or MODELS_TOKEN is not available."""
    if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
        pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available - skipping integration test")


@pytest.fixture
def minimal_test_config():
    """Provide minimal configuration for fast testing."""
    return {
        "provider": "github",
        "model_name": "openai/gpt-4o-mini",
        "temperature": 0.1,
        "max_retries": 1,
        "request_timeout": 30,
    }
