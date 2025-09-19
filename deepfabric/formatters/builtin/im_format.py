"""
Formatter for the <|im_start|>/<|im_end|> conversation format.

This formatter converts DeepFabric datasets to the format used by models
that expect conversation delimiters with <|im_start|> and <|im_end|> tokens.
"""

from pydantic import BaseModel, Field

from ..base import BaseFormatter, FormatterError
from ..models import (
    ConversationSample,
    DatasetInput,
    DatasetOutput,
    FormattedOutput,
    GenericSample,
    InstructionSample,
    QASample,
)


class ImFormatConfig(BaseModel):
    """Configuration for the <|im_start|>/<|im_end|> formatter."""

    include_system: bool = Field(
        default=False, description="Whether to include system messages in the output"
    )
    system_message: str | None = Field(
        default=None, description="Optional system message to prepend to conversations"
    )
    roles_map: dict = Field(
        default={"user": "user", "assistant": "assistant", "system": "system"},
        description="Mapping of roles from input to output format",
    )


class ImFormatter(BaseFormatter):
    """
    Formats conversations using <|im_start|> and <|im_end|> delimiters.

    This formatter is compatible with models that use the ChatML format
    or similar conversation formats with explicit role markers.
    """

    def get_config_model(self):
        """Return the configuration model for this formatter."""
        return ImFormatConfig

    def format(self, dataset: DatasetInput | list) -> DatasetOutput:
        """
        Format dataset to <|im_start|>/<|im_end|> format.

        Args:
            dataset: Input dataset containing conversations

        Returns:
            DatasetOutput with formatted samples
        """
        # Convert list to DatasetInput if needed
        if isinstance(dataset, list):
            samples = []
            for item in dataset:
                if isinstance(item, dict):
                    samples.append(GenericSample(data=item))
                else:
                    samples.append(item)
            dataset = DatasetInput(samples=samples)

        formatted_samples = []
        config: ImFormatConfig = (
            self._config_model
            if isinstance(self._config_model, ImFormatConfig)
            else ImFormatConfig(**self.config)
        )

        for sample in dataset.samples:
            try:
                formatted_text = self._format_sample(sample, config)
                if formatted_text:
                    formatted_samples.append(FormattedOutput(**{"text": formatted_text}))
            except Exception as e:
                raise FormatterError(f"Failed to format sample: {str(e)}") from e

        return DatasetOutput(samples=formatted_samples)

    def _format_sample(
        self,
        sample: GenericSample | ConversationSample | QASample | InstructionSample,
        config: ImFormatConfig,
    ) -> str:
        """
        Format a single sample to <|im_start|>/<|im_end|> format.

        Args:
            sample: Sample to format
            config: Formatter configuration

        Returns:
            Formatted string
        """
        messages = self._extract_messages(sample)

        if not messages:
            raise FormatterError("No messages found in sample")

        formatted_parts = []

        # Add system message if configured
        if config.include_system and config.system_message:
            formatted_parts.append(f"<|im_start|>system\n{config.system_message}<|im_end|>")

        # Format each message
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Map role if needed
            mapped_role = config.roles_map.get(role, role)

            # Skip system messages if already added or not wanted
            if mapped_role == "system" and not config.include_system:
                continue

            formatted_parts.append(f"<|im_start|>{mapped_role}\n{content}<|im_end|>")

        return "\n".join(formatted_parts)

    def _extract_messages(self, sample) -> list[dict]:  # noqa: PLR0911
        """
        Extract messages from different sample types.

        Args:
            sample: Sample to extract messages from

        Returns:
            List of message dictionaries
        """
        # Handle ConversationSample
        if isinstance(sample, ConversationSample):
            return [{"role": msg.role, "content": msg.content} for msg in sample.messages]

        # Handle QASample
        if isinstance(sample, QASample):
            messages = []
            if hasattr(sample, "question") and sample.question:
                messages.append({"role": "user", "content": sample.question})
            if hasattr(sample, "answer") and sample.answer:
                messages.append({"role": "assistant", "content": sample.answer})
            return messages

        # Handle InstructionSample
        if isinstance(sample, InstructionSample):
            messages = []
            if hasattr(sample, "instruction") and sample.instruction:
                content = sample.instruction
                if hasattr(sample, "input") and sample.input:
                    content = f"{content}\n\nInput: {sample.input}"
                messages.append({"role": "user", "content": content})
            if hasattr(sample, "output") and sample.output:
                messages.append({"role": "assistant", "content": sample.output})
            return messages

        # Handle GenericSample or dict
        data = sample.data if isinstance(sample, GenericSample) else sample

        # Try to extract messages from common formats
        if isinstance(data, dict):
            # Check for messages field
            if "messages" in data:
                return data["messages"]

            # Check for question/answer format
            if "question" in data and "answer" in data:
                messages = []
                messages.append({"role": "user", "content": data["question"]})
                messages.append({"role": "assistant", "content": data["answer"]})
                return messages

            # Check for instruction format
            if "instruction" in data:
                messages = []
                content = data["instruction"]
                if "input" in data and data["input"]:
                    content = f"{content}\n\nInput: {data['input']}"
                messages.append({"role": "user", "content": content})
                if "output" in data:
                    messages.append({"role": "assistant", "content": data["output"]})
                return messages

            # Check for user/assistant fields directly
            if "user" in data and "assistant" in data:
                messages = []
                messages.append({"role": "user", "content": data["user"]})
                messages.append({"role": "assistant", "content": data["assistant"]})
                return messages

        raise FormatterError(f"Cannot extract messages from sample type: {type(sample)}")

    def validate(self, entry: dict) -> bool:
        """
        Validate that an entry can be formatted.

        Args:
            entry: Entry to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Create a temporary sample to test extraction
            sample = GenericSample(data=entry)
            messages = self._extract_messages(sample)
            return len(messages) > 0
        except Exception:
            return False

    def get_description(self) -> str:
        """Get formatter description."""
        return (
            "Formats conversations using <|im_start|> and <|im_end|> delimiters. "
            "Compatible with ChatML and similar formats that use explicit role markers."
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "conversation", "qa", "instruction", "question_answer"]
