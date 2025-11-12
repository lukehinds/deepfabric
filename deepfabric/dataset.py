import json
import re

from typing import Any, overload

from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError, UnexpectedSplitsError
from pydantic import ValidationError

from .formatters import FormatterRegistry
from .formatters.base import FormatterError
from .formatters.hf_template import HFChatTemplateFormatter
from .formatters.models import FormatterConfigModel
from .schemas import Conversation


class Dataset:
    """
    A class to represent a dataset consisting of samples, where each sample contains messages with specific roles.
    Methods:
        __init__():
            Initialize an empty dataset.
        from_jsonl(file_path: str) -> "Dataset":
            Create a Dataset instance from a JSONL file.
        from_list(sample_list: list[dict]) -> "Dataset":
            Create a Dataset instance from a list of samples.
        validate_sample(sample: dict) -> bool:
            Validate if a sample has the correct format.
        add_samples(samples: list[dict]) -> tuple[list[dict], list[str]]:
            Add multiple samples to the dataset and return any failures.
        remove_linebreaks_and_spaces(input_string: str) -> str:
            Clean up a string by removing extra whitespace and normalizing linebreaks.
        save(save_path: str):
            Save the dataset to a JSONL file.
        __len__() -> int:
            Get the number of samples in the dataset.
        __getitem__(idx: int) -> dict:
            Get a sample from the dataset by index.
        filter_by_role(role: str) -> list[dict]:
            Filter samples to only include messages with a specific role.
        get_statistics() -> dict:
            Calculate basic statistics about the dataset.
    """

    def __init__(self):
        """Initialize an empty dataset."""
        self.samples = []
        self.failed_samples = []
        self.formatter_registry = FormatterRegistry()
        self.tool_registry = None

    @classmethod
    def from_jsonl(cls, file_path: str) -> "Dataset":
        """Create a Dataset instance from a JSONL file.

        Args:
            file_path: Path to the JSONL file containing the dataset.

        Returns:
            A new Dataset instance populated with the data from the file.
        """
        instance = cls()
        with open(file_path) as f:
            for line in f:
                sample = json.loads(line)
                instance.samples.append(sample)

        return instance

    @classmethod
    def from_list(cls, sample_list: list[dict]) -> "Dataset":
        """Create a Dataset instance from a list of samples.

        Args:
            sample_list: List of dictionaries containing the samples.

        Returns:
            A new Dataset instance populated with the provided samples.
        """
        instance = cls()
        instance.samples.extend(sample_list)
        return instance

    @classmethod
    def from_hub(cls, repo_id: str, split: str = "train") -> "Dataset":
        """Create a Dataset instance from a HuggingFace Hub dataset.

        Args:
            repo_id: HuggingFace dataset repo ID (e.g., "org/dataset-name")
            split: Dataset split to load (default: "train")

        Returns:
            A new Dataset instance populated with data from HuggingFace Hub

        Raises:
            ImportError: If datasets library is not installed
            RuntimeError: If dataset cannot be loaded from Hub

        Example:
            >>> dataset = Dataset.from_hub("deepfabric/customer-support")
            >>> formatted = dataset.format(target_model="meta-llama/Llama-3.1-8B")
        """
        try:
            hf_ds = load_dataset(repo_id, split=split)  #  nosec
            samples = list(hf_ds)

            # Clean up samples: HuggingFace datasets may convert empty strings to None
            cleaned_samples = []
            for sample in samples:
                cleaned = dict(sample)
                # Convert None back to empty strings for optional string fields
                if cleaned.get("question") is None:
                    cleaned["question"] = ""
                if cleaned.get("final_answer") is None:
                    cleaned["final_answer"] = ""
                cleaned_samples.append(cleaned)

            return cls.from_list(cleaned_samples)
        except (DatasetNotFoundError, UnexpectedSplitsError) as e:
            raise RuntimeError(
                f"Failed to load dataset from HuggingFace Hub '{repo_id}' with split '{split}': {e}"
            ) from e

    def add_samples(self, samples: list, tool_registry=None) -> tuple[list, list[str]]:
        """Add multiple samples to the dataset and return any failures.

        Args:
            samples: List of Pydantic models containing the samples to add.
            tool_registry: Optional tool registry for agent tool-calling samples.

        Returns:
            tuple: (list of failed samples, list of failure descriptions)
        """
        if tool_registry is not None:
            self.tool_registry = tool_registry

        self.samples.extend(samples)
        return [], []

    @staticmethod
    def remove_linebreaks_and_spaces(input_string: str) -> str:
        """Clean up a string by removing extra whitespace and normalizing linebreaks.

        Args:
            input_string: The string to clean up.

        Returns:
            str: The cleaned string.
        """
        # Remove line breaks
        no_linebreaks = re.sub(r"\s+", " ", input_string)
        # Remove extra spaces
        return " ".join(no_linebreaks.split())

    def save(self, save_path: str):
        """Save the dataset to a JSONL file.

        Args:
            save_path: Path where the JSONL file should be saved.
        """
        with open(save_path, "w") as f:
            for sample in self.samples:
                sample_dict = (
                    sample if isinstance(sample, dict) else sample.model_dump(exclude_none=True)
                )

                # Special handling for text-only format (e.g., ChatML text output)
                # If sample is a dict with only "text" key, write raw text for training
                if isinstance(sample_dict, dict) and list(sample_dict.keys()) == ["text"]:
                    f.write(sample_dict["text"] + "\n")
                else:
                    # Standard JSON output for all other formats
                    clean_json = self.remove_linebreaks_and_spaces(json.dumps(sample_dict))
                    f.write(clean_json + "\n")

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample from the dataset by index.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            dict: The sample at the specified index.
        """
        return self.samples[idx]

    def filter_by_role(self, role: str) -> list[dict]:
        """Filter samples to only include messages with a specific role.

        Args:
            role: The role to filter by ('user', 'assistant', or 'system').

        Returns:
            List[dict]: Filtered list of samples.
        """
        filtered_samples = []
        for sample in self.samples:
            filtered_messages = [msg for msg in sample["messages"] if msg["role"] == role]
            if filtered_messages:
                filtered_sample = sample.copy()
                filtered_sample["messages"] = filtered_messages
                filtered_samples.append(filtered_sample)
        return filtered_samples

    def get_statistics(self) -> dict:
        """Calculate basic statistics about the dataset.

        Returns:
            dict: Statistics about the dataset including:
                - Total number of samples
                - Average messages per sample
                - Role distribution
                - Average content length
        """
        if not self.samples:
            return {"error": "Dataset is empty"}

        total_samples = len(self.samples)
        total_messages = sum(len(sample["messages"]) for sample in self.samples)
        role_counts = {"user": 0, "assistant": 0, "system": 0}
        total_content_length = 0
        message_count = 0

        for sample in self.samples:
            for message in sample["messages"]:
                role_counts[message["role"]] += 1
                total_content_length += len(message["content"])
                message_count += 1

        return {
            "total_samples": total_samples,
            "avg_messages_per_sample": total_messages / total_samples,
            "role_distribution": {
                role: count / message_count for role, count in role_counts.items()
            },
            "avg_content_length": total_content_length / message_count,
        }

    def apply_formatters(self, formatter_configs: list[dict[str, Any]]) -> dict[str, "Dataset"]:
        """
        Apply formatters to the dataset and return formatted datasets.

        Args:
            formatter_configs: list of formatter configuration dictionaries or FormatterConfig objects

        Returns:
            Dictionary mapping formatter names to formatted Dataset instances

        Raises:
            FormatterError: If any formatter fails to process the dataset
        """

        formatted_datasets = {}

        for config in formatter_configs:
            # Parse config using Pydantic model for validation
            try:
                if isinstance(config, dict):
                    formatter_config_model = FormatterConfigModel(**config)
                else:
                    formatter_config_model = config
            except ValidationError as e:
                raise FormatterError(f"Invalid formatter configuration: {e}") from e

            formatter_name = formatter_config_model.name
            template = formatter_config_model.template
            formatter_config = formatter_config_model.config
            output_path = formatter_config_model.output

            try:
                formatter = self.formatter_registry.load_formatter(
                    template, formatter_config, tool_registry=self.tool_registry
                )

                # Use the new format_with_metadata method for better error reporting
                if hasattr(formatter, "format_with_metadata"):
                    result = formatter.format_with_metadata(self.samples)
                    formatted_samples = result.samples

                    # Log statistics if available
                    if result.stats.failed_samples > 0:
                        print(
                            f"Warning: {result.stats.failed_samples} samples failed to format with {formatter_name}"
                        )
                    if result.errors:
                        print(
                            f"Formatter errors for {formatter_name}: {result.errors[:3]}..."
                        )  # Show first 3 errors
                else:
                    # Fallback to basic format method
                    formatted_result = formatter.format(self.samples)
                    # Extract samples from DatasetOutput if needed
                    if hasattr(formatted_result, "samples"):
                        formatted_samples = formatted_result.samples
                    else:
                        formatted_samples = formatted_result

                # Create a new dataset with the formatted samples
                formatted_dataset = Dataset()
                # Convert FormattedOutput objects to dicts if needed
                if formatted_samples:
                    first_sample = formatted_samples[0]
                    if hasattr(first_sample, "model_dump"):
                        dump = "model_dump"
                    elif hasattr(first_sample, "dict"):
                        dump = "dict"
                    else:
                        dump = None

                    if dump is not None:
                        formatted_dataset.samples = [
                            getattr(sample, dump)(exclude_none=True) for sample in formatted_samples
                        ]
                    else:
                        formatted_dataset.samples = formatted_samples
                else:
                    formatted_dataset.samples = formatted_samples if formatted_samples else []

                # Save to file if output path is specified
                if output_path:
                    formatted_dataset.save(output_path)
                    print(
                        f"Formatted dataset saved to {output_path} using {formatter_name} formatter"
                    )

                formatted_datasets[formatter_name] = formatted_dataset

            except Exception as e:
                raise FormatterError(
                    f"Failed to apply formatter '{formatter_name}': {str(e)}"
                ) from e

        return formatted_datasets

    def list_available_formatters(self) -> list[str]:
        """
        list all available built-in formatters.

        Returns:
            list of built-in formatter names
        """
        return self.formatter_registry.list_builtin_formatters()

    @overload
    def format(
        self,
        target_model: str | None = None,
        model_config: str | None = None,
        use_transformers: bool = True,
        return_info: bool = False,
        tokenizer: Any = None,
        **kwargs,
    ) -> "Dataset": ...

    @overload
    def format(
        self,
        target_model: str | None = None,
        model_config: str | None = None,
        use_transformers: bool = True,
        *,
        return_info: bool = True,
        tokenizer: Any = None,
        **kwargs,
    ) -> tuple["Dataset", dict[str, Any]]: ...

    def format(
        self,
        target_model: str | None = None,
        model_config: str | None = None,
        use_transformers: bool = True,
        return_info: bool = False,
        tokenizer: Any = None,
        **kwargs,
    ) -> "Dataset | tuple[Dataset, dict[str, Any]]":
        """
        Format dataset for a specific model using HuggingFace chat templates.

        ⚠️  WARNING: Most users don't need this method!

        If you're using TRL, Axolotl, or other modern HuggingFace training frameworks,
        they automatically call apply_chat_template() during training - DO NOT pre-format
        your dataset. The DeepFabric evaluator also handles formatting automatically.

        Only use this method for:
        - Debugging/inspecting formatted prompts before training
        - Exporting to llama.cpp, MLX, or other non-HuggingFace inference tools
        - Custom training scripts that don't use TRL/HuggingFace Trainer
        - Manual quality review of how training samples will look
        - OpenAI/Anthropic API fine-tuning (requires pre-formatted text)

        For TRL training: Use the raw dataset with 'messages' and 'tools' columns.
        For DeepFabric evaluation: The evaluator calls apply_chat_template() directly.

        This is the universal formatting method that works with any HuggingFace model
        that has a chat template. It automatically detects model capabilities and
        applies correct formatting.

        Args:
            target_model: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct").
                         Optional if tokenizer is provided.
            model_config: Optional path to custom model mappings YAML file
            use_transformers: Whether to use transformers library (default: True).
                            Ignored if tokenizer is provided.
            return_info: If True, return (dataset, info_dict) instead of just dataset
            tokenizer: Optional pre-loaded transformers tokenizer. If provided, skips
                      tokenizer loading and uses this instance directly. Useful for
                      fine-tuning workflows where tokenizer is already loaded.
            **kwargs: Additional arguments passed to apply_chat_template

        Returns:
            If return_info=False: New Dataset instance with formatted samples
            If return_info=True: Tuple of (Dataset, info_dict) where info_dict contains:
                - capabilities: Detected model capabilities
                - model_id: Target model ID
                - success_count: Number of successfully formatted samples
                - failed_count: Number of failed samples
                - errors: Dictionary of error messages and counts

        Raises:
            ImportError: If required dependencies are not installed
            ValueError: If neither target_model nor tokenizer is provided

        Example:
            >>> # Option 1: Automatic tokenizer loading
            >>> dataset = Dataset.from_jsonl("dataset.jsonl")
            >>> formatted = dataset.format(target_model="meta-llama/Llama-3.1-8B-Instruct")
            >>> formatted.save("llama-formatted.jsonl")
            >>>
            >>> # Option 2: Use existing tokenizer (efficient for fine-tuning workflows)
            >>> from transformers import AutoTokenizer
            >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            >>> formatted = dataset.format(tokenizer=tokenizer)
            >>> formatted.save("qwen-formatted.jsonl")
            >>>
            >>> # With info
            >>> formatted, info = dataset.format(
            ...     target_model="meta-llama/Llama-3.1-8B-Instruct",
            ...     return_info=True
            ... )
            >>> print(f"Reasoning support: {info['capabilities']['reasoning']['native_support']}")
        """

        # Validate inputs and determine model_id
        if tokenizer is None and target_model is None:
            raise ValueError("Either 'target_model' or 'tokenizer' must be provided")

        # Determine the model_id to use
        model_id: str
        if tokenizer is not None:
            if target_model is None:
                # Try to get model name from tokenizer
                if hasattr(tokenizer, "name_or_path"):
                    model_id = tokenizer.name_or_path
                else:
                    raise ValueError(
                        "Could not determine model ID from tokenizer. "
                        "Please provide 'target_model' parameter."
                    )
            else:
                model_id = target_model
            use_transformers = True  # Force transformers mode when tokenizer provided
        else:
            # target_model must be non-None here due to validation above
            model_id = target_model  # type: ignore[assignment]

        # Initialize formatter
        formatter = HFChatTemplateFormatter(
            model_id=model_id, model_config_path=model_config, use_transformers=use_transformers
        )

        # Override formatter's tokenizer if provided
        if tokenizer is not None:
            formatter.tokenizer = tokenizer
            formatter.use_transformers = True

        # Note: We don't warn about missing reasoning for models with thinking support
        # Many models (like Qwen3) support BOTH thinking and non-thinking modes
        # The presence of <think> tags means it's available, not required

        # Format all samples
        formatted_samples = []
        failed_count = 0
        error_messages = {}  # Track unique error messages and their counts

        for sample in self.samples:
            try:
                # Convert sample to Conversation if needed
                conversation = Conversation(**sample) if isinstance(sample, dict) else sample

                # Format using HF chat template
                formatted_text = formatter.format(conversation, **kwargs)

                # Store as dict with text field
                formatted_samples.append({"text": formatted_text})

            except Exception as e:
                error_msg = str(e)
                # Track unique error messages
                if error_msg not in error_messages:
                    error_messages[error_msg] = 0
                error_messages[error_msg] += 1
                failed_count += 1
                continue

        # Create new dataset with formatted samples
        formatted_dataset = Dataset()
        formatted_dataset.samples = formatted_samples

        # Print summary
        total = len(self.samples)
        success = len(formatted_samples)
        print(f"Formatted {success}/{total} samples for {model_id}")

        if failed_count > 0:
            print(f"Warning: {failed_count} samples failed to format")
            # Print unique error messages with counts
            for error_msg, count in error_messages.items():
                if count == 1:
                    print(f"  - {error_msg}")
                else:
                    print(f"  - {error_msg} ({count} samples)")

        # Return with info if requested
        if return_info:
            info = {
                "model_id": model_id,
                "capabilities": formatter.get_capabilities(),
                "success_count": success,
                "failed_count": failed_count,
                "total_count": total,
                "errors": error_messages,
            }
            return formatted_dataset, info

        return formatted_dataset
