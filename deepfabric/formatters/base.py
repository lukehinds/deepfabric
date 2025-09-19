"""
Base formatter interface for DeepFabric.

All formatters (built-in and custom) must inherit from BaseFormatter.
"""

import time

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from pydantic import BaseModel, ValidationError

from .models import (
    ConversationSample,
    DatasetInput,
    DatasetOutput,
    DatasetSample,
    FormattedOutput,
    FormatterMetadata,
    FormatterResult,
    FormatterStats,
    GenericSample,
    InstructionSample,
    QASample,
    ValidationResult,
)


class BaseFormatter(ABC):
    """
    Abstract base class for all dataset formatters.

    Formatters transform datasets from DeepFabric's internal format
    to training framework-specific formats (e.g., Alpaca, GRPO, DPO).
    """

    def __init__(self, config: dict | BaseModel | None = None):
        """
        Initialize the formatter with configuration.

        Args:
            config: Optional configuration dictionary or Pydantic model specific to this formatter
        """
        if isinstance(config, BaseModel):
            self.config = config.model_dump()
            self._config_model = config
        else:
            self.config = config or {}
            self._config_model = None
            self._setup_config()

    def _setup_config(self):
        """Set up configuration using Pydantic model if available."""
        config_model_class = self.get_config_model()
        if config_model_class:
            try:
                self._config_model = config_model_class(**self.config)
            except ValidationError as e:
                raise FormatterError(f"Invalid configuration: {e}") from e

    @abstractmethod
    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample. This is the core method that subclasses must implement.

        Args:
            sample: Single sample dictionary to format

        Returns:
            Formatted sample dictionary or None if formatting fails
        """
        raise NotImplementedError

    def format(self, dataset: DatasetInput | list) -> DatasetOutput:
        """
        Transform the dataset to the target format.

        This method provides the standard implementation that iterates through samples
        and calls _format_single_sample for each one. Subclasses should implement
        _format_single_sample instead of overriding this method.

        Args:
            dataset: Input dataset (DatasetInput model or list of samples)

        Returns:
            DatasetOutput containing formatted samples

        Raises:
            FormatterError: If formatting fails
        """
        # Convert to DatasetInput if it's a list
        if isinstance(dataset, list):
            dataset_samples = []
            for sample in dataset:
                if isinstance(sample, dict):
                    dataset_samples.append(GenericSample(data=sample))
                else:
                    dataset_samples.append(sample)
            dataset_input = DatasetInput(samples=dataset_samples)
        else:
            dataset_input = dataset

        formatted_samples = []

        for i, sample in enumerate(dataset_input.samples):
            try:
                # Convert Pydantic model to dict for formatting
                sample_dict = getattr(sample, "data", None) or sample.model_dump()
                formatted_sample = self._format_single_sample(sample_dict)
                if formatted_sample:
                    formatted_samples.append(FormattedOutput(**formatted_sample))
            except Exception as e:
                raise FormatterError(f"Failed to format sample {i}: {str(e)}") from e

        return DatasetOutput(samples=formatted_samples)

    def format_with_metadata(self, dataset: DatasetInput | list) -> FormatterResult:
        """
        Transform dataset with comprehensive metadata and statistics.

        Args:
            dataset: Input dataset (DatasetInput model or list of samples)

        Returns:
            FormatterResult with samples, metadata, and statistics
        """
        # Convert to DatasetInput if it's a list
        if isinstance(dataset, list):
            # Convert dict samples to GenericSample models
            dataset_samples = []
            for sample in dataset:
                if isinstance(sample, dict):
                    dataset_samples.append(GenericSample(data=sample))
                else:
                    dataset_samples.append(sample)
            dataset_input = DatasetInput(samples=dataset_samples)
        else:
            dataset_input = dataset

        start_time = time.time()
        processed_samples = []
        errors = []
        failed_count = 0
        skipped_count = 0

        for i, sample in enumerate(dataset_input.samples):
            try:
                # Convert Pydantic model to dict for validation
                sample_dict = (
                    sample.data if isinstance(sample, GenericSample) else sample.model_dump()
                )

                validation_result = self.validate_sample(sample_dict)
                if not validation_result.is_valid:
                    skipped_count += 1
                    errors.extend([f"Sample {i}: {error}" for error in validation_result.errors])
                    continue

                formatted_sample = self._format_single_sample(sample_dict)
                if formatted_sample:
                    processed_samples.append(formatted_sample)
                else:
                    failed_count += 1
                    errors.append(f"Sample {i}: Failed to format")

            except Exception as e:
                failed_count += 1
                errors.append(f"Sample {i}: {str(e)}")

        processing_time = time.time() - start_time

        metadata = FormatterMetadata(
            formatter_name=self.__class__.__name__,
            formatter_version="1.0",
            processing_timestamp=datetime.now(tz=UTC).isoformat(),
            validation_passed=len(errors) == 0,
            original_format="generic",
        )

        stats = FormatterStats(
            total_samples=len(dataset_input.samples),
            processed_samples=len(processed_samples),
            failed_samples=failed_count,
            skipped_samples=skipped_count,
            processing_time_seconds=processing_time,
        )

        return FormatterResult(
            samples=processed_samples, metadata=metadata, stats=stats, errors=errors
        )

    def validate_sample(self, entry: dict) -> ValidationResult:
        """
        Validate that an entry meets the formatter's requirements.

        Args:
            entry: A single dataset entry to validate

        Returns:
            ValidationResult with validation status and any errors/warnings
        """
        errors = []
        warnings = []

        # Basic validation
        if not isinstance(entry, dict):
            errors.append("Entry must be a dictionary")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Try to parse the entry into a known format
        sample = self._parse_sample(entry)
        if sample is None:
            errors.append("Could not parse entry into a supported format")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        # Additional custom validation
        custom_validation = self.validate(entry)
        if not custom_validation:
            errors.append("Custom validation failed")

        return ValidationResult(
            is_valid=len(errors) == 0 and custom_validation, errors=errors, warnings=warnings
        )

    def _parse_sample(self, entry: dict) -> DatasetSample | None:
        """
        Try to parse an entry into a typed sample.

        Args:
            entry: Raw entry dictionary

        Returns:
            Parsed sample or None if parsing fails
        """
        # Try different sample types in order of specificity
        sample_types = [ConversationSample, QASample, InstructionSample, GenericSample]

        for sample_type in sample_types:
            try:
                if sample_type == GenericSample:
                    return sample_type(data=entry)
                return sample_type(**entry)
            except ValidationError:
                continue

        return None

    def validate(self, entry: dict) -> bool:
        """
        Custom validation logic. Override in subclasses.

        Args:
            entry: A single dataset entry to validate

        Returns:
            True if the entry is valid, False otherwise
        """
        # Default validation - subclasses can override
        return isinstance(entry, dict)

    def get_config_model(self) -> type[BaseModel] | None:
        """
        Get the Pydantic model class for this formatter's configuration.

        Returns:
            Pydantic model class or None if no specific model is defined
        """
        return None

    def get_description(self) -> str:
        """
        Get a human-readable description of this formatter.

        Returns:
            String description of what this formatter does
        """
        return self.__class__.__doc__ or "No description available"

    def get_supported_formats(self) -> list[str]:
        """
        Get list of input formats this formatter can handle.

        Returns:
            List of supported input format names
        """
        return ["messages"]  # Default: support standard chat format

    def get_output_model(self) -> type[BaseModel] | None:
        """
        Get the Pydantic model class for this formatter's output.

        Returns:
            Pydantic model class or None if no specific model is defined
        """
        return None

    def validate_output(self, output: dict) -> ValidationResult:
        """
        Validate formatter output using the output model.

        Args:
            output: Formatted output to validate

        Returns:
            ValidationResult with validation status
        """
        output_model = self.get_output_model()
        if not output_model:
            # No specific output model, use basic validation
            return ValidationResult(is_valid=isinstance(output, dict))

        try:
            output_model(**output)
            return ValidationResult(is_valid=True)
        except ValidationError as e:
            errors = [str(error) for error in e.errors()]
            return ValidationResult(is_valid=False, errors=errors)


class FormatterError(Exception):
    """Exception raised when formatting operations fail."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


# Import here to avoid circular imports
from .models import ConversationSample, InstructionSample, QASample  # noqa: E402, F811
