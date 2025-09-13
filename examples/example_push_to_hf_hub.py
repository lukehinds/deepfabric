import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import HFUploader  # type: ignore  # noqa: I001


def main():
    print("deepfabric - Uploading to Hugging Face Hub")
    print("============================================")

    dataset_file = "my_dataset.jsonl"

    # Upload the dataset to Hugging Face Hub
    hf_dataset_repo = "huggingface_username/my_dataset"
    jsonl_file_path = dataset_file

    # Get the Hugging Face authentication token from env
    hf_token = os.getenv("HF_TOKEN")

    # Create an instance of HFUploader
    uploader = HFUploader(hf_token)

    # Push dataset to Hugging Face and print the status
    status = uploader.push_to_hub(hf_dataset_repo, jsonl_file_path)
    print(status)


if __name__ == "__main__":
    main()
