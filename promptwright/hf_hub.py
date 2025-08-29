from datasets import load_dataset
from huggingface_hub import DatasetCard, login
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

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

    def update_dataset_card(self, repo_id: str, tags: list[str] | None = None):
        """
        Update the dataset card with tags.

        Parameters:
        repo_id (str): The repository ID in the format 'username/dataset_name'.
        tags (list[str], optional): List of tags to add to the dataset card.
        """
        try:
            card = DatasetCard.load(repo_id)

            # Initialize tags if not present
            if not hasattr(card.data, "tags") or not isinstance(card.data.tags, list):
                card.data.tags = []

            # Add default promptwright tags
            for tag in DEFAULT_HF_TAGS:
                if tag not in card.data.tags:
                    card.data.tags.append(tag)

            # Add custom tags if provided
            if tags:
                for tag in tags:
                    if tag not in card.data.tags:
                        card.data.tags.append(tag)

            card.push_to_hub(repo_id)
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
            dataset = load_dataset("json", data_files={"train": jsonl_file_path})
            dataset.push_to_hub(hf_dataset_repo, token=self.hf_token)

            # Update dataset card with tags
            self.update_dataset_card(hf_dataset_repo, tags)

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
