"""
XLAM 2.0 (APIGen-MT) Multi-Turn Formatter.

This formatter transforms Conversation samples with multi-turn agent interactions into the Salesforce
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
from ..models import ValidationResult


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
        le=50,
        description="Maximum number of conversation turns",
    )


class XlamV2Formatter(BaseFormatter):
    """
    Formatter for XLAM 2.0 (APIGen-MT) multi-turn format.

    Transforms DeepFabric Conversation samples with agent_context into the standardized
    XLAM 2.0 format with multi-turn conversations, tool definitions, and system prompts.
    """

    def __init__(self, config: dict[str, Any] | None = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

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

    def _get_field(self, obj: dict | Any, field: str, default=None):
        """
        Extract field from object, handling both dict and Pydantic model.

        Args:
            obj: Object (dict or Pydantic model)
            field: Field name to extract
            default: Default value if field not found

        Returns:
            Field value or default
        """
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    def _has_field(self, obj: dict | Any, field: str) -> bool:
        """
        Check if object has field, handling both dict and Pydantic model.

        Args:
            obj: Object (dict or Pydantic model)
            field: Field name to check

        Returns:
            True if field exists, False otherwise
        """
        if isinstance(obj, dict):
            return field in obj
        return hasattr(obj, field)

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
        - A list of conversation turns/messages with proper structure
        - At least min_turns turns in the conversation
        - At least one tool available (from tool_context)
        - Valid conversation flow (human starts, function_call → observation)

        Args:
            entry: Dataset entry to validate (unified Conversation schema)

        Returns:
            True if the entry can be formatted, False otherwise
        """
        if not super().validate(entry):
            return False

        # Check for messages (unified Conversation schema) - handle both dict and Pydantic
        messages = self._get_field(entry, "messages")
        if not messages or not isinstance(messages, list):
            return False

        turns = messages

        # Check turn count
        if len(turns) < self.min_turns or len(turns) > self.max_turns:
            return False

        # Check for available tools in tool_context (unified schema) - handle both dict and Pydantic
        tool_context = self._get_field(entry, "tool_context")
        if tool_context:
            available_tools = self._get_field(tool_context, "available_tools", [])
            has_tools = bool(available_tools)
        else:
            has_tools = False

        if not has_tools:
            return False

        # Validate conversation flow if strict mode
        if self.validate_strict:
            # Check if already in XLAM format (has function_call role)
            has_function_call_role = any(
                (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None))
                == "function_call"
                for msg in turns
            )

            if has_function_call_role:
                # Validate pre-structured XLAM format
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
            else:
                # Validate standard format - first non-system message should be user
                non_system_messages = [
                    msg
                    for msg in turns
                    if (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None))
                    != "system"
                ]
                if non_system_messages:
                    first_role = (
                        non_system_messages[0].get("role")
                        if isinstance(non_system_messages[0], dict)
                        else getattr(non_system_messages[0], "role", None)
                    )
                    if first_role != "user":
                        return False

        return True

    def _format_single_sample(self, sample: dict | Any) -> dict | None:
        """
        Format a single sample to XLAM 2.0 format.

        Args:
            sample: A single dataset sample from unified Conversation schema (dict or Pydantic model)

        Returns:
            Formatted sample in XLAM 2.0 format or None if formatting fails
        """
        # Convert Pydantic models to dict
        if hasattr(sample, "model_dump") and callable(getattr(sample, "model_dump", None)):
            sample = sample.model_dump()  # type: ignore[assignment]

        if not self.validate(sample):
            return None

        # Extract conversations (just convert turns to the right format)
        conversations = self._extract_conversations(sample)
        if not conversations:
            return None

        # Extract and format tools - support both legacy and new format, handle Pydantic
        available_tools = self._get_field(sample, "available_tools")
        if not available_tools:
            tool_context = self._get_field(sample, "tool_context")
            if tool_context:
                available_tools = self._get_field(tool_context, "available_tools", [])

        tools = self._format_tools(available_tools or [])

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
        Extract conversation turns from sample and transform to XLAM 2.0 format.

        Handles two input formats:
        1. Pre-structured XLAM format (messages already have function_call/observation roles)
        2. Standard multi-turn format (user/assistant/tool messages + tool_context.executions)

        For standard format, reconstructs XLAM interleaved structure:
        human → gpt → function_call → observation → gpt

        Args:
            sample: Dataset sample with 'messages' field (unified Conversation schema)

        Returns:
            List of conversation turns in XLAM 2.0 format
        """
        messages = self._get_field(sample, "messages", [])
        tool_executions = []

        # Get tool executions from tool_context - handle both dict and Pydantic
        tool_context = self._get_field(sample, "tool_context")
        if tool_context:
            tool_executions = self._get_field(tool_context, "executions", [])

        # Validate messages is not None
        if messages is None:
            return []

        # Check if already in XLAM format (has function_call role)
        has_function_call_role = any(
            (msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None))
            == "function_call"
            for msg in messages
        )

        if has_function_call_role:
            # Already in XLAM format, just do simple mapping
            return self._simple_role_mapping(messages)
        # Standard multi-turn format, needs transformation
        return self._transform_to_xlam_format(messages, tool_executions if tool_executions else [])

    def _map_role_to_xlam_from(self, role: str) -> str | None:
        """
        Map ChatMessage role to XLAM 'from' field.

        Args:
            role: ChatMessage role (system, user, assistant, tool)

        Returns:
            XLAM 'from' field value or None
        """
        role_mapping = {
            "user": "human",
            "assistant": "gpt",
            "tool": "observation",
            "function_call": "function_call",  # Special role for function calls
        }
        return role_mapping.get(role)

    def _simple_role_mapping(self, messages: list) -> list[dict[str, str]]:
        """
        Simple role mapping for pre-structured XLAM format messages.

        Args:
            messages: List of messages already in XLAM structure

        Returns:
            List of XLAM conversation turns
        """
        conversations = []
        for msg in messages:
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )
            xlam_from = self._map_role_to_xlam_from(role)
            if xlam_from and content is not None:
                conversations.append({"from": xlam_from, "value": str(content)})
        return conversations

    def _transform_to_xlam_format(
        self, messages: list, tool_executions: list
    ) -> list[dict[str, str]]:
        """
        Transform standard multi-turn conversation to XLAM format.

        Reconstructs XLAM interleaved structure:
        - Maps user → human
        - Maps assistant → gpt
        - Inserts function_call turns from tool_executions
        - Maps tool → observation

        Args:
            messages: Standard conversation messages (user/assistant/tool)
            tool_executions: List of tool executions to interleave

        Returns:
            List of XLAM conversation turns
        """
        conversations = []
        execution_index = 0

        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
            content = (
                msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
            )

            if role == "user":
                # User message → human turn
                conversations.append({"from": "human", "value": str(content)})

            elif role == "assistant":
                # Check if next message is a tool message (indicates tool calls happened)
                has_tool_after = (
                    i + 1 < len(messages)
                    and (
                        messages[i + 1].get("role")
                        if isinstance(messages[i + 1], dict)
                        else getattr(messages[i + 1], "role", "")
                    )
                    == "tool"
                )

                if has_tool_after:
                    # This assistant message precedes tool calls
                    # Add gpt turn with the assistant's thinking/response
                    conversations.append({"from": "gpt", "value": str(content)})

                    # Process all consecutive tool messages and insert function_call + observation
                    i += 1
                    while i < len(messages):
                        next_msg = messages[i]
                        next_role = (
                            next_msg.get("role")
                            if isinstance(next_msg, dict)
                            else getattr(next_msg, "role", "")
                        )

                        if next_role != "tool":
                            break

                        # Only process if we have a corresponding execution
                        if execution_index < len(tool_executions):
                            execution = tool_executions[execution_index]
                            func_name = (
                                execution.get("function_name")
                                if isinstance(execution, dict)
                                else getattr(execution, "function_name", "")
                            )

                            # Get arguments - handle both parsed_arguments (dict) and arguments (str)
                            if isinstance(execution, dict):
                                args = execution.get("parsed_arguments")
                                if args is None:
                                    # Try to parse from arguments string
                                    args_str = execution.get("arguments", "{}")
                                    try:
                                        args = (
                                            json.loads(args_str)
                                            if isinstance(args_str, str)
                                            else args_str
                                        )
                                    except json.JSONDecodeError:
                                        args = {}
                            else:
                                args = getattr(execution, "parsed_arguments", None)
                                if args is None:
                                    args_str = getattr(execution, "arguments", "{}")
                                    try:
                                        args = (
                                            json.loads(args_str)
                                            if isinstance(args_str, str)
                                            else args_str
                                        )
                                    except json.JSONDecodeError:
                                        args = {}

                            # Insert function_call turn
                            function_call_value = json.dumps({"name": func_name, "arguments": args})
                            conversations.append(
                                {"from": "function_call", "value": function_call_value}
                            )

                            # Insert observation turn
                            tool_content = (
                                next_msg.get("content")
                                if isinstance(next_msg, dict)
                                else getattr(next_msg, "content", "")
                            )
                            conversations.append(
                                {"from": "observation", "value": str(tool_content)}
                            )

                            execution_index += 1

                        i += 1

                    # The loop will increment i again, so decrement here
                    i -= 1
                else:
                    # Final assistant message (no tools after)
                    conversations.append({"from": "gpt", "value": str(content)})

            i += 1

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

        Supports both legacy format and new unified Conversation schema (metadata field).

        Args:
            sample: Dataset sample

        Returns:
            System prompt string
        """
        if not self.include_system_prompt:
            return ""

        # Check for various system prompt fields (prioritize domain_policy for unique policies)
        # Check top-level fields first - handle both dict and Pydantic
        system = (
            self._get_field(sample, "domain_policy")  # Unique domain-specific policy (preferred)
            or self._get_field(sample, "system")
            or self._get_field(sample, "domain_context")
            or self._get_field(sample, "scenario_description")
        )

        # Fall back to metadata fields
        if not system:
            metadata = self._get_field(sample, "metadata")
            if metadata:
                system = (
                    self._get_field(metadata, "domain_policy")
                    or self._get_field(metadata, "domain_context")
                    or self._get_field(metadata, "system")
                    or self._get_field(metadata, "scenario_description")
                )

        return str(system) if system else ""

    def validate_output(self, output: dict) -> ValidationResult:  # noqa: PLR0911
        """
        Validate that the formatted XLAM 2.0 sample is valid.

        Args:
            output: Formatted XLAM 2.0 sample

        Returns:
            ValidationResult with validation status
        """
        xlam_sample = output

        # Check required fields
        if not all(key in xlam_sample for key in ["conversations", "tools", "system"]):
            return ValidationResult(is_valid=False, errors=["Missing required fields"])

        # Validate conversations structure
        conversations = xlam_sample["conversations"]
        if not isinstance(conversations, list) or len(conversations) < self.min_turns:
            return ValidationResult(
                is_valid=False,
                errors=[f"Conversations must be a list with at least {self.min_turns} turns"],
            )

        # Each conversation turn must have 'from' and 'value'
        for i, turn in enumerate(conversations):
            if not isinstance(turn, dict):
                return ValidationResult(is_valid=False, errors=[f"Turn {i} is not a dictionary"])
            if "from" not in turn or "value" not in turn:
                return ValidationResult(
                    is_valid=False, errors=[f"Turn {i} missing 'from' or 'value' field"]
                )
            if turn["from"] not in ["human", "gpt", "function_call", "observation"]:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Turn {i} has invalid 'from' value: {turn['from']}"],
                )

        # Validate tools is a string (JSON serialized)
        if not isinstance(xlam_sample["tools"], str):
            return ValidationResult(is_valid=False, errors=["'tools' must be a JSON string"])

        # Try to parse tools JSON
        try:
            tools = json.loads(xlam_sample["tools"])
            if not isinstance(tools, list):
                return ValidationResult(is_valid=False, errors=["'tools' must be a JSON array"])
        except json.JSONDecodeError as e:
            return ValidationResult(is_valid=False, errors=[f"Invalid tools JSON: {e}"])

        # Validate system is a string
        if not isinstance(xlam_sample["system"], str):
            return ValidationResult(is_valid=False, errors=["'system' must be a string"])

        return ValidationResult(is_valid=True)

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["Conversation"]

    def get_output_model(self) -> type[BaseModel] | None:
        """Get the Pydantic model for XLAM v2 output."""
        # Could define an output model if needed for validation
        return None
