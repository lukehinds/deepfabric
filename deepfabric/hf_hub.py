import json
import tempfile

from pathlib import Path

from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from .constants import DEFAULT_HF_TAGS


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
        Clean dataset by removing empty question/final_answer fields.

        This prevents empty columns from appearing in HuggingFace/Kaggle dataset viewers.

        Parameters:
        jsonl_file_path (str): Path to the original JSONL file.

        Returns:
        str: Path to cleaned file (temp file if cleaning was needed, original if not).
        """
        # Read the dataset and check if cleaning is needed
        needs_cleaning = False
        samples = []

        with open(jsonl_file_path) as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                samples.append(sample)

                # Check if any sample has empty question/final_answer
                if sample.get("question") == "" or sample.get("final_answer") == "":
                    needs_cleaning = True

        # If no cleaning needed, return original file
        if not needs_cleaning:
            return jsonl_file_path

        # Create a temporary file with cleaned data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp_file:
            for sample in samples:
                # Remove empty question/final_answer fields
                if sample.get("question") == "":
                    sample.pop("question", None)
                if sample.get("final_answer") == "":
                    sample.pop("final_answer", None)

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
            print(f"Warning: Failed to update dataset card: {str(e)}")  # nosec
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
