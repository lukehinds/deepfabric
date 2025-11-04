"""Dataset splitting functionality for train/eval set creation."""

import random

from collections import defaultdict
from pathlib import Path
from typing import Literal

from datasets import Dataset as HFDataset
from datasets import DatasetDict
from pydantic import BaseModel, Field, field_validator

from ..dataset import Dataset
from ..schemas import Conversation


class SplitConfig(BaseModel):
    """Configuration for dataset splitting."""

    test_size: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Fraction of data for evaluation set (0.0 to 1.0)",
    )
    stratify_by: Literal["topic", "tool", "conversation_type"] | None = Field(
        default=None,
        description="Field to stratify by for balanced splitting",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    shuffle: bool = Field(
        default=True,
        description="Whether to shuffle before splitting",
    )

    @field_validator("test_size")
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        """Validate test_size is reasonable."""
        if v == 0.0:
            raise ValueError("test_size must be greater than 0.0")
        if v == 1.0:
            raise ValueError("test_size must be less than 1.0")
        return v


class SplitResult(BaseModel):
    """Result of dataset splitting operation."""

    train_size: int = Field(description="Number of samples in training set")
    eval_size: int = Field(description="Number of samples in evaluation set")
    total_size: int = Field(description="Total number of samples")
    stratification_used: str | None = Field(
        default=None,
        description="Stratification field used, if any",
    )
    strata_distribution: dict[str, dict[str, int]] | None = Field(
        default=None,
        description="Distribution of samples across strata",
    )


def _extract_stratification_key(sample: Conversation, stratify_by: str) -> str:  # noqa: PLR0911
    """Extract stratification key from a sample.

    Args:
        sample: Conversation sample
        stratify_by: Field to stratify by (topic, tool, conversation_type)

    Returns:
        Stratification key as string
    """
    if stratify_by == "topic":
        # Extract topic from metadata if present
        if sample.metadata and "topic" in sample.metadata:
            return str(sample.metadata["topic"])
        return "unknown"

    if stratify_by == "tool":
        # Extract tool name from tool_context if present
        if sample.tool_context and sample.tool_context.executions:
            # Use first tool execution as key
            return sample.tool_context.executions[0].function_name
        return "no_tool"

    if stratify_by == "conversation_type":
        # Determine conversation type based on capabilities
        if sample.reasoning is not None:
            return "chain_of_thought"
        return "basic"

    return "unknown"


def _stratified_split(
    samples: list[Conversation],
    test_size: float,
    stratify_by: str,
    seed: int,
) -> tuple[list[Conversation], list[Conversation], dict[str, dict[str, int]]]:
    """Perform stratified split maintaining distribution across strata.

    Args:
        samples: List of Conversation samples
        test_size: Fraction for evaluation set
        stratify_by: Field to stratify by
        seed: Random seed

    Returns:
        Tuple of (train_samples, eval_samples, strata_distribution)
    """
    rng = random.Random(seed)  # noqa: S311  #  nosec

    # Group samples by stratification key
    strata: dict[str, list[Conversation]] = defaultdict(list)
    for sample in samples:
        key = _extract_stratification_key(sample, stratify_by)
        strata[key].append(sample)

    # Split each stratum
    train_samples: list[Conversation] = []
    eval_samples: list[Conversation] = []
    distribution: dict[str, dict[str, int]] = {}

    for stratum_key, stratum_samples in strata.items():
        # Shuffle within stratum
        rng.shuffle(stratum_samples)

        # Calculate split point
        n_eval = max(1, int(len(stratum_samples) * test_size))
        n_train = len(stratum_samples) - n_eval

        # Split
        stratum_eval = stratum_samples[:n_eval]
        stratum_train = stratum_samples[n_eval:]

        train_samples.extend(stratum_train)
        eval_samples.extend(stratum_eval)

        # Track distribution
        distribution[stratum_key] = {
            "train": n_train,
            "eval": n_eval,
            "total": len(stratum_samples),
        }

    return train_samples, eval_samples, distribution


def _simple_split(
    samples: list[Conversation],
    test_size: float,
    seed: int,
    shuffle: bool,
) -> tuple[list[Conversation], list[Conversation]]:
    """Perform simple random split without stratification.

    Args:
        samples: List of Conversation samples
        test_size: Fraction for evaluation set
        seed: Random seed
        shuffle: Whether to shuffle

    Returns:
        Tuple of (train_samples, eval_samples)
    """
    samples_copy = samples.copy()

    if shuffle:
        rng = random.Random(seed)  # noqa: S311  #  nosec
        rng.shuffle(samples_copy)

    # Calculate split point
    n_eval = max(1, int(len(samples_copy) * test_size))
    eval_samples = samples_copy[:n_eval]
    train_samples = samples_copy[n_eval:]

    return train_samples, eval_samples


def split_dataset(
    dataset_path: str,
    train_output: str,
    eval_output: str,
    config: SplitConfig | None = None,
) -> SplitResult:
    """Split dataset into train/eval sets with optional stratification.

    This function loads a DeepFabric dataset, splits it into training and
    evaluation sets, and saves them to separate files. It preserves all
    conversation structure and metadata.

    Args:
        dataset_path: Path to original dataset JSONL file
        train_output: Output path for training set
        eval_output: Output path for evaluation set
        config: Split configuration (uses defaults if None)

    Returns:
        SplitResult with split statistics

    Raises:
        FileNotFoundError: If dataset_path does not exist
        ValueError: If dataset is too small to split
        ValueError: If stratification field is not available in samples

    Example:
        >>> config = SplitConfig(test_size=0.2, stratify_by="topic", seed=42)
        >>> result = split_dataset(
        ...     "dataset.jsonl",
        ...     "train.jsonl",
        ...     "eval.jsonl",
        ...     config
        ... )
        >>> print(f"Train: {result.train_size}, Eval: {result.eval_size}")
    """
    # Use default config if not provided
    if config is None:
        config = SplitConfig()

    # Validate paths
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset
    dataset = Dataset.from_jsonl(dataset_path)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if len(dataset) < 2:  # noqa: PLR2004
        raise ValueError("Dataset must have at least 2 samples to split")

    # Convert samples to Conversation objects for type safety
    conversations: list[Conversation] = []
    for sample in dataset.samples:
        if isinstance(sample, dict):
            conversations.append(Conversation(**sample))
        else:
            conversations.append(sample)

    # Perform split
    strata_distribution: dict[str, dict[str, int]] | None = None

    if config.stratify_by is not None:
        # Stratified split
        train_samples, eval_samples, strata_distribution = _stratified_split(
            conversations,
            config.test_size,
            config.stratify_by,
            config.seed,
        )
    else:
        # Simple random split
        train_samples, eval_samples = _simple_split(
            conversations,
            config.test_size,
            config.seed,
            config.shuffle,
        )

    # Create datasets and save
    train_dataset = Dataset()
    train_dataset.samples = train_samples
    train_dataset.tool_registry = dataset.tool_registry
    train_dataset.save(train_output)

    eval_dataset = Dataset()
    eval_dataset.samples = eval_samples
    eval_dataset.tool_registry = dataset.tool_registry
    eval_dataset.save(eval_output)

    # Create and return result
    return SplitResult(
        train_size=len(train_samples),
        eval_size=len(eval_samples),
        total_size=len(conversations),
        stratification_used=config.stratify_by,
        strata_distribution=strata_distribution,
    )


def split_to_hf_dataset(
    dataset_path: str,
    config: SplitConfig | None = None,
) -> tuple[DatasetDict, SplitResult]:
    """Split dataset and return HuggingFace DatasetDict with train/test splits.

    This function splits a DeepFabric dataset into training and evaluation sets
    and returns them as a HuggingFace DatasetDict, which can be used directly
    with transformers training pipelines or pushed to the HuggingFace Hub.

    Args:
        dataset_path: Path to original dataset JSONL file
        config: Split configuration (uses defaults if None)

    Returns:
        Tuple of (DatasetDict with 'train' and 'test' splits, SplitResult)

    Raises:
        FileNotFoundError: If dataset_path does not exist
        ValueError: If dataset is too small to split

    Example:
        >>> config = SplitConfig(test_size=0.2, stratify_by="topic", seed=42)
        >>> dataset_dict, result = split_to_hf_dataset("dataset.jsonl", config)
        >>> print(f"Train: {len(dataset_dict['train'])}, Test: {len(dataset_dict['test'])}")
        >>> dataset_dict.push_to_hub("username/dataset-name")
    """
    # Use default config if not provided
    if config is None:
        config = SplitConfig()

    # Validate paths
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Load dataset
    dataset = Dataset.from_jsonl(dataset_path)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    if len(dataset) < 2:  # noqa: PLR2004
        raise ValueError("Dataset must have at least 2 samples to split")

    # Convert samples to Conversation objects for type safety
    conversations: list[Conversation] = []
    for sample in dataset.samples:
        if isinstance(sample, dict):
            conversations.append(Conversation(**sample))
        else:
            conversations.append(sample)

    # Perform split
    strata_distribution: dict[str, dict[str, int]] | None = None

    if config.stratify_by is not None:
        # Stratified split
        train_samples, eval_samples, strata_distribution = _stratified_split(
            conversations,
            config.test_size,
            config.stratify_by,
            config.seed,
        )
    else:
        # Simple random split
        train_samples, eval_samples = _simple_split(
            conversations,
            config.test_size,
            config.seed,
            config.shuffle,
        )

    # Convert to dictionaries for HuggingFace Dataset
    train_dicts = [sample.model_dump() for sample in train_samples]
    eval_dicts = [sample.model_dump() for sample in eval_samples]

    # Create HuggingFace datasets
    train_hf_dataset = HFDataset.from_list(train_dicts)
    eval_hf_dataset = HFDataset.from_list(eval_dicts)

    # Create DatasetDict
    dataset_dict = DatasetDict(
        {
            "train": train_hf_dataset,
            "test": eval_hf_dataset,
        }
    )

    # Create result
    result = SplitResult(
        train_size=len(train_samples),
        eval_size=len(eval_samples),
        total_size=len(conversations),
        stratification_used=config.stratify_by,
        strata_distribution=strata_distribution,
    )

    return dataset_dict, result
