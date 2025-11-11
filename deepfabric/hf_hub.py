import json
import tempfile

from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError
from pydantic import BaseModel, field_serializer

from .constants import DEFAULT_HF_TAGS
from .tui import get_tui


class HubUploadSample(BaseModel):
    """
    Model for preparing samples for HuggingFace Hub upload.

    Complex nested fields (tools, tool_context, reasoning, etc.) are serialized to JSON strings
    to avoid HuggingFace's Parquet schema merging issues. This is the industry-standard approach
    used by datasets like Salesforce xLAM (60k examples).
    """

    messages: list[dict[str, Any]]
    metadata: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    tool_context: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    agent_context: dict[str, Any] | None = None
    structured_data: dict[str, Any] | None = None
    question: str = ""
    final_answer: str = ""

    @field_serializer("metadata", "reasoning", "tool_context", "agent_context", "structured_data")
    def serialize_dict_field(self, value: dict[str, Any] | None) -> str | None:
        """Serialize dict fields to JSON strings for HF Hub storage."""
        if value is None:
            return None
        return json.dumps(value)

    @field_serializer("tools")
    def serialize_tools(self, value: list[dict[str, Any]] | None) -> str | None:
        """Serialize tools list to JSON string for HF Hub storage."""
        if value is None:
            return None
        return json.dumps(value)

    def to_upload_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary ready for HF Hub upload.

        Removes empty question/final_answer fields to avoid empty columns in dataset viewers.
        """
        result = self.model_dump(mode="json")

        # Remove empty question/final_answer fields to avoid empty columns
        if result.get("question") == "":
            result.pop("question", None)
        if result.get("final_answer") == "":
            result.pop("final_answer", None)

        return result


class HFUploader:
    """
    HFUploader is a class for uploading datasets to the Hugging Face Hub.

    Methods
    -------
    __init__(hf_token)

    push_to_hub(hf_dataset_repo, jsonl_file_path, tags=None)

        Parameters
        ----------
        hf_dataset_repo : str
            The repository name in the format 'username/dataset_name'.
        jsonl_file_path : str
            Path to the JSONL file.
        tags : list[str], optional
            List of tags to add to the dataset card.

        Returns
        -------
        dict
            A dictionary containing the status and a message.
    """

    def __init__(self, hf_token):
        """
        Initialize the uploader with the Hugging Face authentication token.

        Parameters:
        hf_token (str): Hugging Face Hub authentication token.
        """
        self.hf_token = hf_token

    def _clean_dataset_for_upload(self, jsonl_file_path: str) -> str:
        """
        Prepare dataset for HuggingFace Hub upload.

        This method:
        1. Serializes complex nested fields (tools, tool_context, etc.) to JSON strings
           to avoid HuggingFace's Parquet schema merging issues
        2. Removes empty question/final_answer fields to prevent empty columns in viewers

        Parameters:
        jsonl_file_path (str): Path to the original JSONL file.

        Returns:
        str: Path to prepared file (always a temp file with serialized data).
        """
        # Read and process all samples
        with open(jsonl_file_path) as f:
            samples = [json.loads(line) for line in f if line.strip()]

        # Create a temporary file with serialized data
        tui = get_tui()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
            for sample in samples:
                try:
                    # Use Pydantic model for clean serialization
                    upload_sample = HubUploadSample(**sample)
                    cleaned_sample = upload_sample.to_upload_dict()
                    tmp_file.write(json.dumps(cleaned_sample) + "\n")
                except Exception as e:
                    # Log but don't fail - write original sample if serialization fails
                    tui.warning(f"Failed to serialize sample: {e}. Using original.")
                    tmp_file.write(json.dumps(sample) + "\n")

            return tmp_file.name

    def update_dataset_card(self, repo_id: str, tags: list[str] | None = None):
        """
        Update the dataset card with tags.

        Parameters:
        repo_id (str): The repository ID in the format 'username/dataset_name'.
        tags (list[str], optional): List of tags to add to the dataset card.
        """
        try:
            card = DatasetCard.load(repo_id)

            # Initialize tags if not present - use getattr for safe access
            current_tags = getattr(card.data, "tags", None)
            if not current_tags or not isinstance(current_tags, list):
                current_tags = []
                setattr(card.data, "tags", current_tags)  # noqa: B010

            # Add default deepfabric tags
            for tag in DEFAULT_HF_TAGS:
                if tag not in current_tags:
                    current_tags.append(tag)

            # Add custom tags if provided
            if tags:
                for tag in tags:
                    if tag not in current_tags:
                        current_tags.append(tag)

            # Use getattr to safely access push_to_hub method
            push_method = getattr(card, "push_to_hub", None)
            if push_method:
                push_method(repo_id)
            return True  # noqa: TRY300
        except Exception as e:
            tui = get_tui()
            tui.warning(f"Failed to update dataset card: {str(e)}")
            return False

    def push_to_hub(
        self, hf_dataset_repo: str, jsonl_file_path: str, tags: list[str] | None = None
    ):
        """
        Push a JSONL dataset to Hugging Face Hub.

        Parameters:
        hf_dataset_repo (str): The repository name in the format 'username/dataset_name'.
        jsonl_file_path (str): Path to the JSONL file.
        tags (list[str], optional): List of tags to add to the dataset card.

        Returns:
        dict: A dictionary containing the status and a message.
        """
        try:
            login(token=self.hf_token)

            # Clean empty question/final_answer fields to avoid empty columns in dataset viewers
            cleaned_file = self._clean_dataset_for_upload(jsonl_file_path)

            # Bandit locally produced and sourced
            dataset = load_dataset("json", data_files={"train": cleaned_file})  # nosec

            # Use getattr to safely access push_to_hub method
            push_method = getattr(dataset, "push_to_hub", None)
            if push_method:
                push_method(hf_dataset_repo, token=self.hf_token)
            else:
                raise AttributeError("Dataset object does not support push_to_hub")  # noqa: TRY003, TRY301

            # Update dataset card with tags
            self.update_dataset_card(hf_dataset_repo, tags)

            # Clean up temp file if we created one
            if cleaned_file != jsonl_file_path:
                Path(cleaned_file).unlink(missing_ok=True)

        except RepositoryNotFoundError:
            return {
                "status": "error",
                "message": f"Repository '{hf_dataset_repo}' not found. Please check your repository name.",
            }

        except HfHubHTTPError as e:
            return {
                "status": "error",
                "message": f"Hugging Face Hub HTTP Error: {str(e)}",
            }

        except FileNotFoundError:
            return {
                "status": "error",
                "message": f"File '{jsonl_file_path}' not found. Please check your file path.",
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"An unexpected error occurred: {str(e)}",
            }

        else:
            return {
                "status": "success",
                "message": f"Dataset pushed successfully to {hf_dataset_repo}.",
            }
