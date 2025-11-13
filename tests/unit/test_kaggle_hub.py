"""Tests for the Kaggle Hub uploader module."""

import json
import tempfile

from pathlib import Path
from unittest.mock import patch

import pytest

from deepfabric.kaggle_hub import KaggleUploader


@pytest.fixture
def uploader():
    """Create a KaggleUploader instance with mock credentials."""
    with patch.dict("os.environ", {"KAGGLE_USERNAME": "test_user", "KAGGLE_KEY": "test_key"}):
        return KaggleUploader("test_user", "test_key")


def test_kaggle_uploader_init_with_credentials():
    """Test KaggleUploader initialization with provided credentials."""
    with patch.dict("os.environ", clear=True):
        uploader = KaggleUploader("user", "key")
        assert uploader.kaggle_username == "user"
        assert uploader.kaggle_key == "key"


def test_kaggle_uploader_init_from_env():
    """Test KaggleUploader initialization from environment variables."""
    with patch.dict("os.environ", {"KAGGLE_USERNAME": "env_user", "KAGGLE_KEY": "env_key"}):
        uploader = KaggleUploader()
        assert uploader.kaggle_username == "env_user"
        assert uploader.kaggle_key == "env_key"


def test_kaggle_uploader_init_missing_credentials():
    """Test KaggleUploader initialization with missing credentials."""
    with (
        patch.dict("os.environ", clear=True),
        pytest.raises(ValueError, match="Kaggle credentials not provided"),
    ):
        KaggleUploader()


def test_create_dataset_metadata(uploader):
    """Test dataset metadata creation."""
    metadata = uploader.create_dataset_metadata(
        "username/dataset-name", tags=["tag1", "tag2"], description="Test dataset"
    )

    assert metadata["id"] == "username/dataset-name"
    assert metadata["title"] == "Dataset Name"
    assert "deepfabric" in metadata["tags"]
    assert "synthetic" in metadata["tags"]
    assert "tag1" in metadata["tags"]
    assert "tag2" in metadata["tags"]
    assert metadata["description"] == "Test dataset"
    assert metadata["licenses"] == [{"name": "CC0-1.0"}]


def test_create_dataset_metadata_invalid_handle(uploader):
    """Test metadata creation with invalid handle format."""
    with pytest.raises(ValueError, match="Invalid dataset handle format"):
        uploader.create_dataset_metadata("invalid-handle")


def test_push_to_hub_success(uploader):
    """Test successful dataset push to Kaggle."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write('{"test": "data"}\\n')
        tmp_file_path = tmp_file.name

    try:
        with patch("deepfabric.kaggle_hub.kagglehub.dataset_upload") as mock_upload:
            result = uploader.push_to_hub(
                "test/dataset",
                tmp_file_path,
                tags=["test"],
                version_notes="Test upload",
                description="Test description",
            )

            mock_upload.assert_called_once()
            call_args = mock_upload.call_args
            assert call_args.kwargs["handle"] == "test/dataset"
            assert call_args.kwargs["version_notes"] == "Test upload"
            assert "local_dataset_dir" in call_args.kwargs

            assert result["status"] == "success"
            assert "test/dataset" in result["message"]
    finally:
        Path(tmp_file_path).unlink()


def test_push_to_hub_file_not_found(uploader):
    """Test push to hub with non-existent file."""
    result = uploader.push_to_hub("test/dataset", "nonexistent.jsonl")
    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_push_to_hub_dataset_not_found(uploader):
    """Test push to hub with non-existent Kaggle dataset."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write('{"test": "data"}\\n')
        tmp_file_path = tmp_file.name

    try:
        with patch("deepfabric.kaggle_hub.kagglehub.dataset_upload") as mock_upload:
            mock_upload.side_effect = Exception("404 Not Found")

            result = uploader.push_to_hub("test/dataset", tmp_file_path)
            assert result["status"] == "error"
            assert "not found" in result["message"].lower()
            assert "create it first" in result["message"]
    finally:
        Path(tmp_file_path).unlink()


def test_push_to_hub_authentication_error(uploader):
    """Test push to hub with authentication error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write('{"test": "data"}\\n')
        tmp_file_path = tmp_file.name

    try:
        with patch("deepfabric.kaggle_hub.kagglehub.dataset_upload") as mock_upload:
            mock_upload.side_effect = Exception("401 Unauthorized")

            result = uploader.push_to_hub("test/dataset", tmp_file_path)
            assert result["status"] == "error"
            assert "Authentication failed" in result["message"]
    finally:
        Path(tmp_file_path).unlink()


def test_push_to_hub_permission_denied(uploader):
    """Test push to hub with permission denied error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write('{"test": "data"}\\n')
        tmp_file_path = tmp_file.name

    try:
        with patch("deepfabric.kaggle_hub.kagglehub.dataset_upload") as mock_upload:
            mock_upload.side_effect = Exception("403 Forbidden")

            result = uploader.push_to_hub("test/dataset", tmp_file_path)
            assert result["status"] == "error"
            assert "Permission denied" in result["message"]
    finally:
        Path(tmp_file_path).unlink()


def test_push_to_hub_generic_error(uploader):
    """Test push to hub with generic error."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write('{"test": "data"}\\n')
        tmp_file_path = tmp_file.name

    try:
        with patch("deepfabric.kaggle_hub.kagglehub.dataset_upload") as mock_upload:
            mock_upload.side_effect = Exception("Some other error")

            result = uploader.push_to_hub("test/dataset", tmp_file_path)
            assert result["status"] == "error"
            assert "Some other error" in result["message"]
    finally:
        Path(tmp_file_path).unlink()


def test_metadata_file_creation(uploader):
    """Test that metadata file is created correctly during upload."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
        tmp_file.write('{"test": "data"}\\n')
        tmp_file_path = tmp_file.name

    try:
        with patch("deepfabric.kaggle_hub.kagglehub.dataset_upload") as mock_upload:
            # We need to capture what files are created
            created_files = {}

            def capture_upload(**kwargs):
                path = Path(kwargs["local_dataset_dir"])
                for file_path in path.iterdir():
                    if file_path.name == "dataset-metadata.json":
                        with open(file_path) as f:
                            created_files["metadata"] = json.load(f)

            mock_upload.side_effect = capture_upload

            uploader.push_to_hub(
                "test_user/test_dataset",
                tmp_file_path,
                tags=["ml", "dataset"],
                description="Test dataset for ML",
            )

            # Verify metadata was created correctly
            assert "metadata" in created_files
            metadata = created_files["metadata"]
            assert metadata["id"] == "test_user/test_dataset"
            assert "ml" in metadata["tags"]
            assert "dataset" in metadata["tags"]
            assert "deepfabric" in metadata["tags"]
            assert metadata["description"] == "Test dataset for ML"

    finally:
        Path(tmp_file_path).unlink()
