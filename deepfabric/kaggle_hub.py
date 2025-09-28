import json
import os
import shutil
import tempfile

from contextlib import contextmanager
from pathlib import Path

import kagglehub

from .constants import DEFAULT_KAGGLE_TAGS

# Constants
EXPECTED_HANDLE_PARTS = 2


class KaggleUploader:
    """
    KaggleUploader is a class for uploading datasets to Kaggle.

    Methods
    -------
    __init__(kaggle_username, kaggle_key)

    push_to_hub(dataset_handle, jsonl_file_path, tags=None, version_notes=None)

        Parameters
        ----------
        dataset_handle : str
            The dataset handle in the format 'username/dataset-name'.
        jsonl_file_path : str
            Path to the JSONL file.
        tags : list[str], optional
            List of tags to add to the dataset.
        version_notes : str, optional
            Notes for the dataset version.

        Returns
        -------
        dict
            A dictionary containing the status and a message.
    """

    def __init__(self, kaggle_username: str | None = None, kaggle_key: str | None = None):
        """
        Initialize the uploader with Kaggle authentication credentials.

        Parameters:
        kaggle_username (str, optional): Kaggle username (can also be set via KAGGLE_USERNAME env var).
        kaggle_key (str, optional): Kaggle API key (can also be set via KAGGLE_KEY env var).
        """
        self.kaggle_username = kaggle_username or os.getenv("KAGGLE_USERNAME")
        self.kaggle_key = kaggle_key or os.getenv("KAGGLE_KEY")

        if not self.kaggle_username or not self.kaggle_key:
            raise ValueError(
                "Kaggle credentials not provided. "
                "Set via constructor params or KAGGLE_USERNAME/KAGGLE_KEY env vars."
            )

    @contextmanager
    def _kaggle_credentials(self):
        """Context manager to temporarily set Kaggle credentials in environment."""
        # Store original values to restore later
        original_username = os.environ.get("KAGGLE_USERNAME")
        original_key = os.environ.get("KAGGLE_KEY")

        try:
            # Set credentials for kagglehub
            os.environ["KAGGLE_USERNAME"] = self.kaggle_username  # type: ignore
            os.environ["KAGGLE_KEY"] = self.kaggle_key  # type: ignore
            yield
        finally:
            # Restore original environment state
            if original_username is None:
                os.environ.pop("KAGGLE_USERNAME", None)
            else:
                os.environ["KAGGLE_USERNAME"] = original_username

            if original_key is None:
                os.environ.pop("KAGGLE_KEY", None)
            else:
                os.environ["KAGGLE_KEY"] = original_key

    def create_dataset_metadata(
        self, dataset_handle: str, tags: list[str] | None = None, description: str | None = None
    ) -> dict:
        """
        Create metadata for the Kaggle dataset.

        Parameters:
        dataset_handle (str): The dataset handle in the format 'username/dataset-name'.
        tags (list[str], optional): List of tags for the dataset.
        description (str, optional): Description for the dataset.

        Returns:
        dict: Metadata dictionary for the dataset.
        """
        # Parse the dataset handle
        parts = dataset_handle.split("/")
        if len(parts) != EXPECTED_HANDLE_PARTS:
            raise ValueError(
                f"Invalid dataset handle format: {dataset_handle}. Expected 'username/dataset-name'"
            )

        username, dataset_name = parts

        # Add default deepfabric tags
        all_tags = set(DEFAULT_KAGGLE_TAGS)
        if tags:
            all_tags.update(tags)

        metadata = {
            "title": dataset_name.replace("-", " ").title(),
            "id": f"{username}/{dataset_name}",
            "licenses": [{"name": "CC0-1.0"}],
            "tags": list(all_tags),
        }

        if description:
            metadata["description"] = description
        else:
            metadata["description"] = "Synthetic dataset generated using DeepFabric"

        return metadata

    def _handle_upload_error(self, error: Exception, dataset_handle: str) -> dict | None:
        """Handle specific upload errors and return appropriate error response."""
        error_msg = str(error)
        if "404" in error_msg or "not found" in error_msg.lower():
            return {
                "status": "error",
                "message": (
                    f"Dataset '{dataset_handle}' not found. "
                    "You may need to create it first on Kaggle.com"
                ),
            }
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            return {
                "status": "error",
                "message": "Authentication failed. Please check your Kaggle credentials.",
            }
        if "403" in error_msg or "forbidden" in error_msg.lower():
            return {
                "status": "error",
                "message": f"Permission denied. You may not have access to update {dataset_handle}.",
            }
        return None

    def push_to_hub(
        self,
        dataset_handle: str,
        jsonl_file_path: str,
        tags: list[str] | None = None,
        version_notes: str | None = None,
        description: str | None = None,
    ) -> dict[str, str]:
        """
        Push a JSONL dataset to Kaggle.

        Parameters:
        dataset_handle (str): The dataset handle in the format 'username/dataset-name'.
        jsonl_file_path (str): Path to the JSONL file.
        tags (list[str], optional): List of tags to add to the dataset.
        version_notes (str, optional): Notes for the dataset version.
        description (str, optional): Description for the dataset.

        Returns:
        dict: A dictionary containing the status and a message.
        """
        result = {"status": "error", "message": ""}

        try:
            # Create a temporary directory for the dataset
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Copy the JSONL file to the temp directory
                dest_file = tmpdir_path / Path(jsonl_file_path).name
                shutil.copy2(jsonl_file_path, dest_file)

                # Create dataset metadata
                metadata = self.create_dataset_metadata(dataset_handle, tags, description)
                metadata_path = tmpdir_path / "dataset-metadata.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Upload the dataset using kagglehub
                version_notes = version_notes or "Dataset uploaded via DeepFabric"

                try:
                    # Upload the dataset with temporary credentials
                    with self._kaggle_credentials():
                        kagglehub.dataset_upload(
                            handle=dataset_handle,
                            local_dataset_dir=str(tmpdir_path),
                            version_notes=version_notes,
                        )

                except Exception as upload_error:
                    # Handle specific Kaggle errors
                    error_result = self._handle_upload_error(upload_error, dataset_handle)
                    if error_result:
                        return error_result
                    raise
                else:
                    result["status"] = "success"
                    result["message"] = f"Dataset pushed successfully to Kaggle: {dataset_handle}"

        except FileNotFoundError:
            result["message"] = f"File '{jsonl_file_path}' not found. Please check your file path."

        except ValueError as e:
            result["message"] = f"Invalid configuration: {str(e)}"

        except Exception as e:
            result["message"] = f"An unexpected error occurred: {str(e)}"

        return result
