"""
Conversations formatter for generic training frameworks.

This formatter converts DeepFabric datasets to the standard conversations format
using role/content pairs. Compatible with Unsloth, Axolotl, HuggingFace TRL, and
other frameworks that use the conversations field structure.
"""

from pydantic import BaseModel, Field

from ..base import BaseFormatter
from ..utils import extract_messages


class ConversationsConfig(BaseModel):
    """Configuration for the conversations formatter."""

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


class ConversationsFormatter(BaseFormatter):
    """
    Formats datasets to standard conversations format.

    This formatter outputs datasets with a 'conversations' field containing
    role/content pairs. Compatible with multiple training frameworks including
    Unsloth, Axolotl, and HuggingFace TRL.
    """

    def get_config_model(self):
        """Return the configuration model for this formatter."""
        return ConversationsConfig

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to conversations format.

        Args:
            sample: Sample to format

        Returns:
            Formatted sample with conversations key
        """
        config: ConversationsConfig = (
            self._config_model
            if isinstance(self._config_model, ConversationsConfig)
            else ConversationsConfig(**self.config)
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
            "Formats datasets to the standard conversations format with role/content pairs. "
            "Compatible with Unsloth, Axolotl, HuggingFace TRL, and other frameworks that use the conversations field structure."
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "conversation", "qa", "instruction", "question_answer"]
