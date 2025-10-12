"""
TRL SFT Tools formatter.

This formatter transforms DeepFabric agent reasoning datasets into the format
required by HuggingFace TRL's SFTTrainer for tool/function calling fine-tuning.

Key features:
- Converts `available_tools` to `tools` field in OpenAI schema format
- Ensures proper message structure with tool calls and tool responses
- Compatible with TRL SFTTrainer's tool calling mode
- Supports multiple conversation types (agent_cot_tools, agent_cot_hybrid, etc.)

The formatter converts from DeepFabric's internal format:
{
  "messages": [...],
  "available_tools": [{"name": "...", "parameters": [...], ...}, ...]
}

To TRL SFT format:
{
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "...",
        "description": "...",
        "parameters": {"type": "object", "properties": {...}, "required": [...]}
      }
    },
    ...
  ]
}

Reference:
- https://huggingface.co/docs/trl/en/sft_trainer#tool-calling-with-sft
- https://www.stephendiehl.com/posts/fine_tuning_tools/
"""

import logging

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ...schemas import ToolDefinition
from ..base import BaseFormatter
from ..models import ConversationSample

logger = logging.getLogger(__name__)


class TRLSFTToolsConfig(BaseModel):
    """Configuration for TRL SFT Tools formatter."""

    include_system_prompt: bool = Field(
        default=True,
        description="Whether to include system prompt in messages (recommended for tool calling)",
    )
    system_prompt_override: str | None = Field(
        default=None,
        description="Override the system prompt with custom text (None uses original)",
    )
    validate_tool_schemas: bool = Field(
        default=True,
        description="Validate that tool schemas are properly formatted",
    )
    remove_available_tools_field: bool = Field(
        default=False,
        description="Remove the 'available_tools' field from output (keep only 'tools')",
    )


class TRLSFTToolsFormatter(BaseFormatter):
    """
    Formatter for HuggingFace TRL SFTTrainer tool calling format.

    This formatter prepares DeepFabric datasets for training with TRL's
    SFTTrainer in tool calling mode. It converts DeepFabric's tool definitions
    to OpenAI function calling schema format required by TRL.

    The formatter is specifically designed for:
    - Fine-tuning models with tool/function calling capabilities
    - Training with HuggingFace TRL SFTTrainer
    - Creating datasets compatible with trl.trainer.sft_trainer
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def get_config_model(self) -> type[BaseModel] | None:
        """Return the configuration model for this formatter."""
        return TRLSFTToolsConfig

    def validate(self, sample: dict) -> bool:
        """Validate that sample has required fields for TRL format."""
        # Must have messages
        if "messages" not in sample or not isinstance(sample["messages"], list):
            return False

        # Must have at least one message
        # Return whether there is at least one message
        # Should have available_tools for tool calling
        # (though we'll still process samples without it)
        return len(sample["messages"]) != 0

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to TRL SFT tool calling format.

        Args:
            sample: DeepFabric sample with messages and available_tools

        Returns:
            Formatted sample with 'tools' field in OpenAI schema format
        """
        if not self.validate(sample):
            return None

        # Get configuration
        config: TRLSFTToolsConfig = (
            self._config_model
            if isinstance(self._config_model, TRLSFTToolsConfig)
            else TRLSFTToolsConfig()
        )

        # Start with a copy of the sample
        formatted_sample = sample.copy()

        # Convert available_tools to TRL format if present
        if "available_tools" in sample and sample["available_tools"]:
            try:
                # Convert to ToolDefinition objects and then to OpenAI schema
                tool_defs = [
                    ToolDefinition.model_validate(tool) for tool in sample["available_tools"]
                ]
                formatted_sample["tools"] = [tool.to_openai_schema() for tool in tool_defs]

                # Optionally validate tool schemas
                if config.validate_tool_schemas:
                    self._validate_tool_schemas(formatted_sample["tools"])

                # Optionally remove available_tools field
                if config.remove_available_tools_field:
                    formatted_sample.pop("available_tools", None)

            except (ValidationError, TypeError, KeyError) as e:
                # If tool conversion fails, log but don't fail the entire sample
                # This allows processing of samples without proper tool definitions
                logger.warning(
                    "Failed to convert 'available_tools' for a sample due to: %s. Skipping tool conversion.",
                    e,
                    exc_info=True,
                )
                formatted_sample["tools"] = []

        # Handle system prompt
        messages = formatted_sample.get("messages", [])
        if messages and config.include_system_prompt:
            # Check if first message is system message
            has_system = messages[0].get("role") == "system" if messages else False

            if config.system_prompt_override and has_system:
                # Override existing system prompt
                messages[0]["content"] = config.system_prompt_override
            elif config.system_prompt_override and not has_system:
                # Add new system prompt at the beginning
                messages.insert(
                    0,
                    {"role": "system", "content": config.system_prompt_override},
                )

        return formatted_sample

    def _validate_tool_schemas(self, tools: list[dict]) -> None:
        """
        Validate that tool schemas are properly formatted for TRL.

        Args:
            tools: List of tool schemas in OpenAI format

        Raises:
            ValueError: If tool schemas are invalid
        """
        for i, tool in enumerate(tools):
            # Check required top-level fields
            if "type" not in tool or tool["type"] != "function":
                raise ValueError(f"Tool {i}: Missing or invalid 'type' field (must be 'function')")

            if "function" not in tool:
                raise ValueError(f"Tool {i}: Missing 'function' field")

            func = tool["function"]

            # Check required function fields
            required_fields = ["name", "description", "parameters"]
            for field in required_fields:
                if field not in func:
                    raise ValueError(f"Tool {i}: Missing required field '{field}' in function")

            # Validate parameters structure
            params = func["parameters"]
            if "type" not in params or params["type"] != "object":
                raise ValueError(
                    f"Tool {i}: parameters must have type='object', got {params.get('type')}"
                )

            if "properties" not in params:
                raise ValueError(f"Tool {i}: parameters must have 'properties' field")

    def format_conversation_sample(self, sample: ConversationSample) -> dict[str, Any]:
        """Format a ConversationSample (if needed for compatibility)."""
        return {"messages": [msg.model_dump() for msg in sample.messages]}

    def get_example_config(self) -> dict[str, Any]:
        """Return example configuration for this formatter."""
        return {
            "include_system_prompt": True,
            "system_prompt_override": None,
            "validate_tool_schemas": True,
            "remove_available_tools_field": False,
        }
