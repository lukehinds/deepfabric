"""
XLAM 2.0 (APIGen-MT) Multi-Turn Formatter.

This formatter transforms XlamMultiTurnAgent conversations into the Salesforce
XLAM 2.0 (APIGen-MT-5k) format designed for training multi-turn function-calling models.

XLAM 2.0 format structure:
{
  "conversations": [
    {"from": "human", "value": "user query"},
    {"from": "gpt", "value": "agent response"},
    {"from": "function_call", "value": '{"name": "...", "arguments": {...}}'},
    {"from": "observation", "value": "tool execution result"}
  ],
  "tools": "[{...tool definitions...}]",
  "system": "system prompt with domain policy"
}

Based on: https://huggingface.co/datasets/Salesforce/APIGen-MT-5k
"""

import json

from typing import Any

from pydantic import BaseModel, Field

from ..base import BaseFormatter


class XlamV2Config(BaseModel):
    """Configuration for XLAM v2 formatter."""

    validate_strict: bool = Field(
        default=True,
        description="Enable strict validation of conversation flow and function calls",
    )
    include_system_prompt: bool = Field(
        default=True,
        description="Include system/domain policy prompt in output",
    )
    min_turns: int = Field(
        default=3,
        ge=1,
        description="Minimum number of conversation turns",
    )
    max_turns: int = Field(
        default=15,
        le=20,
        description="Maximum number of conversation turns",
    )


class XlamV2Formatter(BaseFormatter):
    """
    Formatter for XLAM 2.0 (APIGen-MT) multi-turn format.

    Transforms DeepFabric XlamMultiTurnAgent conversations into the standardized
    XLAM 2.0 format with multi-turn conversations, tool definitions, and system prompts.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

        # _config_model is guaranteed to be XlamV2Config due to get_config_model()
        xlam_config: XlamV2Config = self._config_model  # type: ignore[assignment]
        self.validate_strict = xlam_config.validate_strict
        self.include_system_prompt = xlam_config.include_system_prompt
        self.min_turns = xlam_config.min_turns
        self.max_turns = xlam_config.max_turns

    def get_config_model(self) -> type[BaseModel]:
        """Return the configuration model for this formatter."""
        return XlamV2Config

    @classmethod
    def get_default_config(cls) -> dict:
        """Return the default configuration for this formatter."""
        return {
            "validate_strict": True,
            "include_system_prompt": True,
            "min_turns": 3,
            "max_turns": 15,
        }

    def _get_from_field(self, turn: dict | Any) -> str | None:
        """
        Extract 'from' field from a turn, handling both dict and Pydantic model.

        Args:
            turn: Conversation turn (dict or XlamConversationTurn)

        Returns:
            The 'from' field value or None if not found
        """
        if isinstance(turn, dict):
            return turn.get("from") or turn.get("from_")
        # Handle Pydantic model
        return getattr(turn, "from_", None) or getattr(turn, "from", None)

    def validate(self, entry: dict) -> bool:  # noqa: PLR0911
        """
        Validate that an entry can be formatted for XLAM 2.0.

        XLAM 2.0 requires:
        - A list of conversation turns with proper structure
        - At least min_turns turns in the conversation
        - At least one tool available
        - Valid conversation flow (human starts, function_call → observation)

        Args:
            entry: Dataset entry to validate

        Returns:
            True if the entry can be formatted, False otherwise
        """
        if not super().validate(entry):
            return False

        # Check for required fields
        has_turns = "turns" in entry and isinstance(entry["turns"], list)
        if not has_turns:
            return False

        turns = entry["turns"]

        # Check turn count
        if len(turns) < self.min_turns or len(turns) > self.max_turns:
            return False

        # Check that we have available tools
        has_tools = "available_tools" in entry and entry["available_tools"]
        if not has_tools:
            return False

        # Validate conversation flow if strict mode
        if self.validate_strict:
            # Must start with human
            if not turns or self._get_from_field(turns[0]) != "human":
                return False

            # function_call must be followed by observation
            for i, turn in enumerate(turns[:-1]):
                if (
                    self._get_from_field(turn) == "function_call"
                    and i + 1 < len(turns)
                    and self._get_from_field(turns[i + 1]) != "observation"
                ):
                    return False

            # Validate function_call turns have valid JSON
            for turn in turns:
                if self._get_from_field(turn) == "function_call":
                    value = (
                        turn.get("value", "{}")
                        if isinstance(turn, dict)
                        else getattr(turn, "value", "{}")
                    )
                    try:
                        call_data = json.loads(value)
                        if "name" not in call_data or "arguments" not in call_data:
                            return False
                    except json.JSONDecodeError:
                        return False

        return True

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to XLAM 2.0 format.

        Args:
            sample: A single dataset sample from XlamMultiTurnAgent generation

        Returns:
            Formatted sample in XLAM 2.0 format or None if formatting fails
        """
        if not self.validate(sample):
            return None

        # Extract conversations (just convert turns to the right format)
        conversations = self._extract_conversations(sample)
        if not conversations:
            return None

        # Extract and format tools
        tools = self._format_tools(sample.get("available_tools", []))

        # Extract system/domain prompt
        system = self._extract_system_prompt(sample)

        xlam_sample = {
            "conversations": conversations,
            "tools": tools,
            "system": system,
        }

        # Final validation
        if self.validate_strict and not self.validate_output(xlam_sample):
            return None

        return xlam_sample

    def _extract_conversations(self, sample: dict) -> list[dict[str, str]]:
        """
        Extract conversation turns from sample.

        Args:
            sample: Dataset sample with 'turns' field

        Returns:
            List of conversation turns in XLAM 2.0 format
        """
        conversations = []

        for turn in sample.get("turns", []):
            from_field = self._get_from_field(turn)
            value = turn.get("value", "") if isinstance(turn, dict) else getattr(turn, "value", "")

            if from_field and value is not None:
                conversations.append({"from": from_field, "value": str(value)})

        return conversations

    def _format_tools(self, tools: list) -> str:
        """
        Format tools as JSON string for XLAM 2.0.

        Args:
            tools: List of tool definitions

        Returns:
            JSON string of tool definitions
        """
        formatted_tools = []

        for tool in tools:
            # Handle both dict and ToolDefinition objects
            if isinstance(tool, dict):
                formatted_tool = self._format_tool_dict(tool)
            else:
                # Convert Pydantic model to dict
                tool_dict = tool.model_dump() if hasattr(tool, "model_dump") else tool.__dict__
                formatted_tool = self._format_tool_dict(tool_dict)

            if formatted_tool:
                formatted_tools.append(formatted_tool)

        return json.dumps(formatted_tools)

    def _format_tool_dict(self, tool: dict) -> dict | None:
        """
        Format a single tool dictionary to XLAM 2.0 JSON Schema format.

        Args:
            tool: Tool definition as dictionary

        Returns:
            Formatted tool in XLAM 2.0 schema or None if invalid
        """
        if "name" not in tool:
            return None

        # Build JSON Schema parameters
        properties = {}
        required = []

        # Handle parameters list format (DeepFabric ToolDefinition format)
        if "parameters" in tool and isinstance(tool["parameters"], list):
            for param in tool["parameters"]:
                if isinstance(param, dict) and "name" in param:
                    param_name = param["name"]
                    properties[param_name] = {
                        "type": self._convert_param_type(param.get("type", "string")),
                        "description": param.get("description", ""),
                    }
                    if param.get("required", True):
                        required.append(param_name)

        # Handle JSON Schema format (already in correct format)
        elif "parameters" in tool and isinstance(tool["parameters"], dict):
            parameters = tool["parameters"]
            if "properties" in parameters:
                properties = parameters["properties"]
            if "required" in parameters:
                required = parameters["required"]

        return {
            "name": tool["name"],
            "description": tool.get("description", f"Function {tool['name']}"),
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _convert_param_type(self, param_type: str) -> str:
        """
        Convert DeepFabric parameter type to JSON Schema type.

        Args:
            param_type: DeepFabric type string

        Returns:
            JSON Schema type string
        """
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }
        return type_mapping.get(param_type, "string")

    def _extract_system_prompt(self, sample: dict) -> str:
        """
        Extract system/domain prompt from sample.

        Args:
            sample: Dataset sample

        Returns:
            System prompt string
        """
        if not self.include_system_prompt:
            return ""

        # Check for various system prompt fields (prioritize domain_policy for unique policies)
        system = (
            sample.get("domain_policy")  # Unique domain-specific policy (preferred)
            or sample.get("system")
            or sample.get("domain_context")
            or sample.get("scenario_description")
            or ""
        )

        return str(system)

    def validate_output(self, xlam_sample: dict) -> bool:  # noqa: PLR0911
        """
        Validate that the formatted XLAM 2.0 sample is valid.

        Args:
            xlam_sample: Formatted XLAM 2.0 sample

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        if not all(key in xlam_sample for key in ["conversations", "tools", "system"]):
            return False

        # Validate conversations structure
        conversations = xlam_sample["conversations"]
        if not isinstance(conversations, list) or len(conversations) < self.min_turns:
            return False

        # Each conversation turn must have 'from' and 'value'
        for turn in conversations:
            if not isinstance(turn, dict):
                return False
            if "from" not in turn or "value" not in turn:
                return False
            if turn["from"] not in ["human", "gpt", "function_call", "observation"]:
                return False

        # Validate tools is a string (JSON serialized)
        if not isinstance(xlam_sample["tools"], str):
            return False

        # Try to parse tools JSON
        try:
            tools = json.loads(xlam_sample["tools"])
            if not isinstance(tools, list):
                return False
        except json.JSONDecodeError:
            return False

        # Validate system is a string
        return isinstance(xlam_sample["system"], str)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["xlam_multi_turn", "XlamMultiTurnAgent"]

    def get_output_model(self) -> type[BaseModel] | None:
        """Get the Pydantic model for XLAM v2 output."""
        # Could define an output model if needed for validation
        return None
