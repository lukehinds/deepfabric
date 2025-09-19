"""
Formatter for Unsloth training framework.

This formatter converts DeepFabric datasets to the format expected by Unsloth,
using the conversations structure with role/content pairs.
"""

from pydantic import BaseModel, Field

from ..base import BaseFormatter
from ..utils import extract_messages


class UnslothConfig(BaseModel):
    """Configuration for the Unsloth formatter."""

    include_system: bool = Field(
        default=False, description="Whether to include system messages in conversations"
    )
    system_message: str | None = Field(
        default=None, description="Optional system message to add to conversations"
    )
    roles_map: dict = Field(
        default={"user": "user", "assistant": "assistant", "system": "system"},
        description="Mapping of roles from input to output format",
    )


class UnslothFormatter(BaseFormatter):
    """
    Formats datasets for Unsloth training framework.

    This formatter outputs datasets in the conversations format that Unsloth expects,
    with role/content pairs that can be processed by Unsloth's chat templates.
    """

    def get_config_model(self):
        """Return the configuration model for this formatter."""
        return UnslothConfig

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to Unsloth conversations format.

        Args:
            sample: Sample to format

        Returns:
            Formatted sample with conversations key
        """
        config: UnslothConfig = (
            self._config_model
            if isinstance(self._config_model, UnslothConfig)
            else UnslothConfig(**self.config)
        )

        try:
            messages = extract_messages(sample)
        except ValueError:
            return None

        if not messages:
            return None

        conversations = []

        # Add system message if configured
        if config.include_system and config.system_message:
            conversations.append(
                {"role": config.roles_map.get("system", "system"), "content": config.system_message}
            )

        # Add conversation messages
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Map role if needed
            mapped_role = config.roles_map.get(role, role)

            # Skip system messages if already added or not wanted
            if mapped_role == "system" and not config.include_system:
                continue

            conversations.append({"role": mapped_role, "content": content})

        return {"conversations": conversations}

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
            "Formats datasets for Unsloth training framework. "
            "Outputs conversations with role/content pairs compatible with Unsloth's chat templates."
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "conversation", "qa", "instruction", "question_answer"]
