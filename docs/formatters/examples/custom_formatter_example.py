"""
Example custom formatter for a hypothetical training framework.

This formatter demonstrates common patterns and best practices for
creating custom formatters in DeepFabric.
"""

import re

from typing import Any

from deepfabric.formatters.base import BaseFormatter, FormatterError


class ExampleCustomFormatter(BaseFormatter):
    """
    Example custom formatter for demonstration purposes.

    This formatter creates a specialized format that includes:
    - Custom metadata fields
    - Template-based text generation
    - Multi-format input support
    - Comprehensive validation
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Configuration options with defaults
        self.output_template = self.config.get(
            "output_template",
            "Task: {instruction}\nContext: {context}\nResponse: {response}"
        )
        self.include_metadata = self.config.get("include_metadata", True)
        self.max_context_length = self.config.get("max_context_length", 1000)
        self.required_fields = self.config.get("required_fields", ["instruction", "response"])

        # Validate configuration
        self._validate_config()

        # Compile regex patterns for efficiency
        self._compile_patterns()

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.max_context_length <= 0:
            raise FormatterError("max_context_length must be positive")

        if not isinstance(self.required_fields, list):
            raise FormatterError("required_fields must be a list")

        # Validate template
        try:
            self.output_template.format(instruction="test", context="test", response="test")
        except KeyError as e:
            raise FormatterError(f"Invalid template, missing placeholder: {e}") from e

    def _compile_patterns(self):
        """Compile regex patterns for text processing."""
        self.url_pattern = re.compile(r'https?://[^\s]+')
        self.whitespace_pattern = re.compile(r'\s+')

    def format(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Transform dataset to custom format.

        Args:
            dataset: list of samples in DeepFabric format

        Returns:
            list of samples in custom format

        Raises:
            FormatterError: If formatting fails
        """
        formatted_samples = []

        for i, sample in enumerate(dataset):
            try:
                if not self.validate(sample):
                    continue

                formatted_sample = self._format_sample(sample)
                if formatted_sample:
                    formatted_samples.append(formatted_sample)

            except Exception as e:
                raise FormatterError(f"Failed to format sample {i}: {str(e)}") from e

        return formatted_samples

    def _format_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """
        Format a single sample.

        Args:
            sample: Input sample to format

        Returns:
            Formatted sample or None if formatting fails
        """
        # Extract components based on input format
        components = self._extract_components(sample)
        if not components:
            return None

        # Apply transformations
        components = self._transform_components(components)

        # Generate output using template
        formatted_text = self._apply_template(components)

        # Build output sample
        output_sample: dict[str, Any] = {
            "text": formatted_text,
            "format_version": "1.0"
        }

        # Add metadata if configured
        if self.include_metadata:
            output_sample["metadata"] = self._generate_metadata(sample, components)

        return output_sample

    def _extract_components(self, sample: dict[str, Any]) -> dict[str, str] | None:
        """
        Extract instruction, context, and response from various input formats.

        Args:
            sample: Input sample

        Returns:
            dictionary with extracted components or None if extraction fails
        """
        # Handle messages format
        if "messages" in sample:
            return self._extract_from_messages(sample["messages"])

        # Handle Q&A format
        if "question" in sample and ("answer" in sample or "final_answer" in sample):
            return self._extract_from_qa(sample)

        # Handle instruction format
        if "instruction" in sample and "output" in sample:
            return self._extract_from_instruction(sample)

        # Handle generic format
        return self._extract_generic(sample)

    def _extract_from_messages(self, messages: list[dict[str, str]]) -> dict[str, str] | None:
        """Extract components from chat messages format."""
        instruction = ""
        context = ""
        response = ""

        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")

            if role == "system":
                context = content
            elif role == "user":
                if instruction:
                    context += f"\n{content}"
                else:
                    instruction = content
            elif role == "assistant":
                response = content

        if not instruction or not response:
            return None

        return {
            "instruction": instruction,
            "context": context,
            "response": response
        }

    def _extract_from_qa(self, sample: dict[str, Any]) -> dict[str, str]:
        """Extract components from Q&A format."""
        instruction = sample["question"]
        response = sample.get("answer") or sample.get("final_answer", "")
        context = sample.get("context", "")

        # Include reasoning if available
        if "chain_of_thought" in sample:
            reasoning = sample["chain_of_thought"]
            response = f"Reasoning: {reasoning}\n\nAnswer: {response}"

        return {
            "instruction": instruction,
            "context": context,
            "response": response
        }

    def _extract_from_instruction(self, sample: dict[str, Any]) -> dict[str, str]:
        """Extract components from instruction format."""
        return {
            "instruction": sample["instruction"],
            "context": sample.get("input", ""),
            "response": sample["output"]
        }

    def _extract_generic(self, sample: dict[str, Any]) -> dict[str, str] | None:
        """Extract components from generic format by field name matching."""
        instruction_fields = ["instruction", "prompt", "question", "task"]
        response_fields = ["response", "output", "answer", "solution"]
        context_fields = ["context", "input", "background"]

        instruction = ""
        response = ""
        context = ""

        # Find instruction
        for field in instruction_fields:
            if field in sample and sample[field]:
                instruction = sample[field]
                break

        # Find response
        for field in response_fields:
            if field in sample and sample[field]:
                response = sample[field]
                break

        # Find context
        for field in context_fields:
            if field in sample and sample[field]:
                context = sample[field]
                break

        if not instruction or not response:
            return None

        return {
            "instruction": instruction,
            "context": context,
            "response": response
        }

    def _transform_components(self, components: dict[str, str]) -> dict[str, str]:
        """
        Apply transformations to extracted components.

        Args:
            components: dictionary with instruction, context, response

        Returns:
            Transformed components
        """
        transformed = {}

        for key, value in components.items():
            # Clean whitespace
            cleaned = self.whitespace_pattern.sub(' ', value.strip())

            # Remove URLs if they exist
            cleaned = self.url_pattern.sub('[URL]', cleaned)

            # Truncate context if too long
            if key == "context" and len(cleaned) > self.max_context_length:
                cleaned = cleaned[:self.max_context_length] + "..."

            transformed[key] = cleaned

        return transformed

    def _apply_template(self, components: dict[str, str]) -> str:
        """
        Apply the output template to generate formatted text.

        Args:
            components: dictionary with instruction, context, response

        Returns:
            Formatted text
        """
        return self.output_template.format(**components)

    def _generate_metadata(self, original_sample: dict[str, Any],
                          components: dict[str, str]) -> dict[str, Any]:
        """
        Generate metadata for the formatted sample.

        Args:
            original_sample: Original input sample
            components: Extracted components

        Returns:
            Metadata dictionary
        """
        metadata = {
            "formatter": "example_custom_formatter",
            "version": "1.0",
            "original_format": self._detect_format(original_sample),
            "component_lengths": {
                key: len(value) for key, value in components.items()
            }
        }

        # Preserve original metadata if present
        if "metadata" in original_sample:
            metadata["original_metadata"] = original_sample["metadata"]

        return metadata

    def _detect_format(self, sample: dict[str, Any]) -> str:
        """Detect the input format of a sample."""
        if "messages" in sample:
            return "messages"
        if "question" in sample:
            return "qa"
        if "instruction" in sample:
            return "instruction"
        return "generic"

    def validate(self, entry: dict[str, Any]) -> bool:
        """
        Validate that an entry can be formatted.

        Args:
            entry: Dataset entry to validate

        Returns:
            True if the entry can be formatted, False otherwise
        """
        if not super().validate(entry):
            return False

        # Check if we can extract required components
        components = self._extract_components(entry)
        if not components:
            return False

        # Verify required fields are present and non-empty
        for field in self.required_fields:
            if field not in components or not components[field].strip():
                return False

        return True

    def validate_output(self, entry: dict[str, Any]) -> bool:
        """
        Validate that a formatted entry meets requirements.

        Args:
            entry: Formatted entry to validate

        Returns:
            True if the entry meets format requirements
        """
        if not isinstance(entry, dict):
            return False

        # Check required output fields
        if "text" not in entry or "format_version" not in entry:
            return False

        # Validate text content
        text = entry["text"]
        if not isinstance(text, str) or len(text.strip()) == 0:
            return False

        # Validate metadata if present
        if "metadata" in entry:
            metadata = entry["metadata"]
            if not isinstance(metadata, dict):
                return False

        return True

    def get_description(self) -> str:
        """Get description of this formatter."""
        return """
        Example custom formatter demonstrating best practices.

        Features:
        - Template-based text generation
        - Multi-format input support (messages, Q&A, instruction, generic)
        - Configurable transformations and validation
        - Comprehensive metadata generation
        - Robust error handling

        This formatter serves as a template for creating specialized
        formatters for custom training frameworks.
        """

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "qa", "instruction", "generic"]


# Alternative: Specialized formatter for a specific use case
class JsonlFormatter(BaseFormatter):
    """
    Simple formatter that outputs JSONL with specific field names.

    Useful for frameworks that expect specific field naming conventions.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # Field name mappings
        self.field_mapping = self.config.get("field_mapping", {
            "input": "prompt",
            "output": "completion"
        })

    def format(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform to JSONL format with custom field names."""
        formatted_samples = []

        for sample in dataset:
            if "messages" not in sample:
                continue

            messages = sample["messages"]

            # Extract user and assistant messages
            user_content = ""
            assistant_content = ""

            for msg in messages:
                if msg["role"] == "user":
                    user_content += msg["content"] + " "
                elif msg["role"] == "assistant":
                    assistant_content += msg["content"] + " "

            if user_content and assistant_content:
                formatted_sample = {
                    self.field_mapping["input"]: user_content.strip(),
                    self.field_mapping["output"]: assistant_content.strip()
                }
                formatted_samples.append(formatted_sample)

        return formatted_samples

    def get_description(self) -> str:
        return "Simple JSONL formatter with customizable field names"
