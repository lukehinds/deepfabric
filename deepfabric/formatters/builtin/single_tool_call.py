"""
Single Tool Call formatter.

This formatter transforms agent reasoning datasets into a format where each tool call
is in its own message exchange, rather than embedding multiple tools in a single response.

The formatter converts from:
{
  "question": "What's the weather in Paris and the time in Tokyo?",
  "reasoning": "Need to check weather and time...",
  "tool_used": "get_weather",
  "tool_input": "{'location': 'Paris'}",
  "tool_output": "15°C, partly cloudy",
  "answer": "The weather in Paris is 15°C and partly cloudy."
}

To:
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant with access to functions..."},
    {"role": "user", "content": "What's the weather in Paris and the time in Tokyo?"},
    {"role": "assistant", "content": "I'll check the weather in Paris for you.\n\n<tool_call>\n{'name': 'get_weather', 'arguments': {'location': 'Paris'}}\n</tool_call>"},
    {"role": "tool", "content": "{'temperature': '15°C', 'conditions': 'Partly cloudy'}"},
    {"role": "assistant", "content": "The weather in Paris is currently 15°C and partly cloudy."}
  ]
}
"""

import json

from typing import Any

from pydantic import BaseModel, Field

from ..base import BaseFormatter
from ..models import ConversationSample


class SingleToolCallConfig(BaseModel):
    """Configuration for single tool call formatter."""

    system_prompt: str = Field(
        default="You are a helpful assistant with access to the following functions. Use them if required:",
        description="System prompt that explains tool calling behavior",
    )
    include_tools_in_system: bool = Field(
        default=True, description="Whether to include available tools in system message"
    )
    include_reasoning_prefix: bool = Field(
        default=True, description="Whether to include a reasoning prefix before the tool call"
    )
    reasoning_prefix_template: str = Field(
        default="I'll {action} for you.",
        description="Template for the reasoning prefix. {action} will be replaced with tool action description",
    )
    tool_call_format: str = Field(
        default="<tool_call>\n{tool_call}\n</tool_call>",
        description="Format for tool call tags",
    )
    tool_response_as_json: bool = Field(
        default=True,
        description="Whether to format tool response as JSON string",
    )


class SingleToolCallFormatter(BaseFormatter):
    """
    Formatter for single tool calling format.

    Transforms agent reasoning datasets into conversational format
    where each tool call is in its own message exchange.
    """

    def __init__(self, config: dict[str, Any] | None = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

    def get_config_model(self) -> type[BaseModel] | None:
        """Return the configuration model for this formatter."""
        return SingleToolCallConfig

    def validate(self, entry: dict | BaseModel) -> bool:  # type: ignore[override]
        """Validate that entry has required fields for tool calling format."""
        # Support both legacy schema and unified Conversation schema
        if hasattr(entry, "tool_executions"):
            return len(entry.tool_executions) > 0  # type: ignore[union-attr]
        if hasattr(entry, "tool_context") and entry.tool_context:  # type: ignore[union-attr]
            return len(entry.tool_context.executions) > 0  # type: ignore[union-attr]
        # Dict format
        if isinstance(entry, dict):
            if "tool_executions" in entry and entry["tool_executions"]:
                return True
            if "tool_context" in entry and entry["tool_context"]:
                tool_context = entry["tool_context"]
                if isinstance(tool_context, dict) and "executions" in tool_context:
                    return bool(tool_context["executions"])
        return False

    def _format_single_sample(self, sample: dict | BaseModel) -> dict | None:
        """
        Format a single agent reasoning sample to single tool call conversation format.

        Uses unified Conversation schema with tool_context.

        Args:
            sample: Agent conversation (Pydantic model or dict)

        Returns:
            Formatted conversation sample with single tool call per message
        """
        if not self.validate(sample):
            return None

        config: SingleToolCallConfig = (
            self._config_model
            if isinstance(self._config_model, SingleToolCallConfig)
            else SingleToolCallConfig()
        )

        # Extract data from either legacy or unified schema
        question = self._get_question(sample)
        tool_executions = self._get_tool_executions(sample)
        answer = self._get_answer(sample)
        available_tools = self._get_available_tools(sample)

        messages = []

        # Add system message with tools
        if config.include_tools_in_system and available_tools:
            tools_text = self._format_tools_for_system(available_tools)
            system_content = f"{config.system_prompt}\n\n{tools_text}"
            messages.append({"role": "system", "content": system_content})
        elif config.include_tools_in_system:
            system_content = f"{config.system_prompt}\n\n{self._get_generic_tools_text()}"
            messages.append({"role": "system", "content": system_content})

        # Add user question
        messages.append({"role": "user", "content": question})

        for execution in tool_executions:
            assistant_content = ""

            # Extract execution details from either format
            func_name = (
                execution.function_name
                if hasattr(execution, "function_name")
                else execution.get("function_name", "")
            )

            if config.include_reasoning_prefix:
                action = self._get_tool_action_description(func_name, execution)
                reasoning_prefix = config.reasoning_prefix_template.format(action=action)
                assistant_content = f"{reasoning_prefix}\n\n"

            # Get parsed arguments
            if hasattr(execution, "parsed_arguments"):
                parsed_args = execution.parsed_arguments
            elif isinstance(execution, dict) and "parsed_arguments" in execution:
                parsed_args = execution["parsed_arguments"]
            elif isinstance(execution, dict) and "arguments" in execution:
                args = execution["arguments"]
                if isinstance(args, str):
                    try:
                        parsed_args = json.loads(args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                else:
                    parsed_args = args
            else:
                parsed_args = {}

            tool_call_json = json.dumps({"name": func_name, "arguments": parsed_args})
            tool_call = config.tool_call_format.format(tool_call=tool_call_json)
            assistant_content += tool_call

            messages.append({"role": "assistant", "content": assistant_content})

            # Get result
            result = (
                execution.result if hasattr(execution, "result") else execution.get("result", "")  # type: ignore[union-attr]
            )

            if config.tool_response_as_json:
                tool_response_content = json.dumps({"result": result})
            else:
                tool_response_content = result

            messages.append({"role": "tool", "content": tool_response_content})

        # Add final assistant answer (only once, after all tool calls)
        messages.append({"role": "assistant", "content": answer})

        return {"messages": messages}

    def _get_question(self, sample) -> str:
        """Extract question from either legacy or unified schema."""
        if isinstance(sample, dict):
            return sample.get("question", "")
        return getattr(sample, "question", "")

    def _get_answer(self, sample) -> str:
        """Extract answer from either legacy or unified schema."""
        if isinstance(sample, dict):
            return sample.get("answer") or sample.get("final_answer", "")
        # Try various possible fields
        return getattr(sample, "answer", None) or getattr(sample, "final_answer", None) or ""

    def _get_tool_executions(self, sample):
        """Extract tool executions from unified schema."""
        # Unified schema (Pydantic model)
        if hasattr(sample, "tool_context") and sample.tool_context:
            return sample.tool_context.executions
        # Unified schema (dict format)
        if isinstance(sample, dict) and "tool_context" in sample:
            tool_context = sample["tool_context"]
            if isinstance(tool_context, dict):
                return tool_context.get("executions", [])
        return []

    def _get_available_tools(self, sample):
        """Extract available tools from sample or tool registry."""
        # Try sample first (unified schema)
        if (
            hasattr(sample, "tool_context")
            and sample.tool_context
            and hasattr(sample.tool_context, "available_tools")
        ):
            return sample.tool_context.available_tools
        # Dict format
        if isinstance(sample, dict) and "tool_context" in sample:
            tool_context = sample["tool_context"]
            if isinstance(tool_context, dict) and "available_tools" in tool_context:
                return tool_context["available_tools"]
        # Fall back to tool registry
        return self.tool_registry.tools if self.tool_registry else []

    def _get_tool_action_description(self, tool_name: str, execution) -> str:
        """Generate a natural language description of the tool action."""
        tool_actions = {
            "get_weather": "check the weather",
            "get_time": "check the current time",
            "calculator": "perform the calculation",
            "web_search": "search for that information",
            "database_query": "query the database",
            "api_call": "make the API call",
        }

        action = tool_actions.get(tool_name, f"use the {tool_name} tool")

        # Extract arguments from either format
        if hasattr(execution, "parsed_arguments"):
            tool_args = execution.parsed_arguments
        elif isinstance(execution, dict) and "parsed_arguments" in execution:
            tool_args = execution["parsed_arguments"]
        elif isinstance(execution, dict) and "arguments" in execution:
            # Try to parse arguments if it's a JSON string
            args = execution["arguments"]
            if isinstance(args, str):
                try:
                    tool_args = json.loads(args)
                except json.JSONDecodeError:
                    tool_args = {}
            else:
                tool_args = args
        else:
            tool_args = {}
        if tool_args:
            if tool_name == "get_weather" and "location" in tool_args:
                action = f"check the weather in {tool_args['location']}"
            elif tool_name == "get_time" and "timezone" in tool_args:
                action = f"check the time in {tool_args['timezone']}"
            elif tool_name == "calculator" and "expression" in tool_args:
                action = f"calculate {tool_args['expression']}"
            elif tool_name == "web_search" and "query" in tool_args:
                action = f"search for {tool_args['query']}"

        return action

    def _get_generic_tools_text(self) -> str:
        """Get generic tools text when no specific tools are provided."""
        return json.dumps(
            {
                "functions": [
                    {
                        "name": "generic_function",
                        "description": "A generic function that can perform various operations",
                        "parameters": {"type": "object", "properties": {}, "required": []},
                    }
                ]
            },
            indent=2,
        )

    def _format_tools_for_system(self, available_tools: list) -> str:
        """Format available tools for inclusion in system message."""
        functions = []
        for tool in available_tools:
            func_def = {
                "name": tool.name,
                "description": tool.description,
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

            for param in tool.parameters:
                func_def["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    func_def["parameters"]["required"].append(param.name)

            functions.append(func_def)

        return json.dumps({"functions": functions}, indent=2)

    def format_conversation_sample(self, sample: ConversationSample) -> dict[str, Any]:
        """Format a conversation sample (if needed for compatibility)."""
        return {"messages": [msg.model_dump(exclude_none=True) for msg in sample.messages]}

    def get_example_config(self) -> dict[str, Any]:
        """Return example configuration for this formatter."""
        return {
            "system_prompt": "You are a helpful assistant with access to the following functions. Use them if required:",
            "include_tools_in_system": True,
            "include_reasoning_prefix": True,
            "reasoning_prefix_template": "I'll {action} for you.",
            "tool_call_format": "<tool_call>\n{tool_call}\n</tool_call>",
            "tool_response_as_json": True,
        }
