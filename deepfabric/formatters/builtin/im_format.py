"""
Formatter for the <|im_start|>/<|im_end|> conversation format.

This formatter converts DeepFabric datasets to the format used by models
that expect conversation delimiters with <|im_start|> and <|im_end|> tokens.
"""

from pydantic import BaseModel, Field

from ..base import BaseFormatter
from ..utils import extract_messages


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

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to <|im_start|>/<|im_end|> format.

        Args:
            sample: Sample to format

        Returns:
            Formatted sample with text key
        """
        config: ImFormatConfig = (
            self._config_model
            if isinstance(self._config_model, ImFormatConfig)
            else ImFormatConfig(**self.config)
        )

        try:
            messages = extract_messages(sample)
        except ValueError:
            return None

        if not messages:
            return None

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

        return {"text": "\n".join(formatted_parts)}

    def validate(self, entry: dict) -> bool:
        """
        Validate that an entry can be formatted.

        Args:
            entry: Entry to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            messages = extract_messages(entry)
            return len(messages) > 0
        except (ValueError, Exception):
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
