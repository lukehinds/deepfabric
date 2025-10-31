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

import json
import logging

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from ...schemas import ToolDefinition
from ..base import BaseFormatter, get_field, has_field, to_dict
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

    def __init__(self, config: dict[str, Any] | None = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

    def get_config_model(self) -> type[BaseModel] | None:
        """Return the configuration model for this formatter."""
        return TRLSFTToolsConfig

    def validate(self, sample) -> bool:
        """Validate that sample has required fields for TRL format."""
        # Accept either messages format OR agent_cot_tools format

        # Check for messages format
        if has_field(sample, "messages"):
            messages = get_field(sample, "messages")
            if isinstance(messages, list):
                return len(messages) > 0

        # Check for agent_cot_tools format (question + tool_used + answer/final_answer)
        if has_field(sample, "question") and has_field(sample, "tool_used"):
            return has_field(sample, "answer") or has_field(sample, "final_answer")

        return False

    def _format_single_sample(self, sample) -> dict | None:
        """
        Format a single sample to TRL SFT tool calling format.

        Args:
            sample: DeepFabric sample with messages and available_tools (dict or Pydantic model)

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

        # Convert agent_cot_tools format to messages format if needed
        if not has_field(sample, "messages"):
            sample_dict = self._convert_agent_to_messages(to_dict(sample), config)
        else:
            sample_dict = to_dict(sample)

        # Start with a copy of the sample
        formatted_sample = sample_dict.copy()

        # Convert available_tools to TRL format if present
        # Support both legacy format (available_tools) and unified schema (tool_context.available_tools)
        available_tools = get_field(sample, "available_tools")
        tool_executions = []
        if not available_tools and has_field(sample, "tool_context"):
            tool_context = get_field(sample, "tool_context")
            if isinstance(tool_context, dict):
                if "available_tools" in tool_context:
                    available_tools = tool_context["available_tools"]
                if "executions" in tool_context:
                    tool_executions = tool_context["executions"]
            elif hasattr(tool_context, "available_tools"):
                available_tools = tool_context.available_tools  # type: ignore
                if hasattr(tool_context, "executions"):
                    tool_executions = tool_context.executions  # type: ignore

        if available_tools:
            try:
                # Convert to ToolDefinition objects
                tool_defs = [ToolDefinition.model_validate(tool) for tool in available_tools]

                # Filter to only tools actually used (saves tokens)
                if tool_executions:
                    tool_defs = self._get_conversation_used_tools(tool_defs, tool_executions)

                # Convert to OpenAI schema
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

        # Clean up empty fields that shouldn't be in the output
        # For basic conversations (no CoT), question and final_answer are empty strings
        if "question" in formatted_sample and not formatted_sample["question"]:
            del formatted_sample["question"]
        if "final_answer" in formatted_sample and not formatted_sample["final_answer"]:
            del formatted_sample["final_answer"]

        return formatted_sample

    def _convert_agent_to_messages(self, sample: dict, config: TRLSFTToolsConfig) -> dict:
        """
        Convert agent_cot_tools format to messages format.

        Args:
            sample: Sample in agent_cot_tools format
            config: Formatter configuration

        Returns:
            Sample with messages field
        """
        messages = []

        # Add system message if configured
        if config.include_system_prompt:
            system_content = config.system_prompt_override or (
                "You are a helpful AI assistant with access to various tools and functions. "
                "When a user asks a question that requires information or actions you cannot "
                "directly provide, use the available tools to help answer the question."
            )
            messages.append({"role": "system", "content": system_content})

        # Add user question
        question = sample.get("question", "")
        messages.append({"role": "user", "content": question})

        # Extract tool usage information
        tool_used = sample.get("tool_used", "")
        tool_input = sample.get("tool_input", "{}")
        tool_output = sample.get("tool_output", "")
        answer = sample.get("answer") or sample.get("final_answer", "")

        # Parse tool input
        if isinstance(tool_input, str):
            try:
                # First try parsing as standard JSON
                tool_args = json.loads(tool_input)
            except json.JSONDecodeError:
                try:
                    # Fallback: try replacing single quotes with double quotes
                    tool_args = json.loads(tool_input.replace("'", '"'))
                except json.JSONDecodeError:
                    # Final fallback: wrap in a simple structure
                    tool_args = {"input": tool_input}
        else:
            tool_args = tool_input

        # Add assistant message with tool call
        # Using OpenAI-compatible function calling format
        tool_call = {
            "id": "call_1",  # Placeholder ID
            "type": "function",
            "function": {"name": tool_used, "arguments": json.dumps(tool_args)},
        }

        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        )

        # Add tool response message
        messages.append(
            {
                "role": "tool",
                "content": str(tool_output),
                "tool_call_id": "call_1",
            }
        )

        # Add final assistant answer
        messages.append({"role": "assistant", "content": answer})

        # Return sample with messages and preserve available_tools
        return {
            "messages": messages,
            "available_tools": sample.get("available_tools", []),
        }

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
