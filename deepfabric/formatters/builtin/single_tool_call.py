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

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def get_config_model(self) -> type[BaseModel] | None:
        """Return the configuration model for this formatter."""
        return SingleToolCallConfig

    def validate(self, entry: dict) -> bool:
        """Validate that entry has required fields for tool calling format."""
        required_fields = ["question", "tool_used"]
        # Check for either "answer" (SimpleAgentCoT) or "final_answer" (HybridAgentCoT)
        has_answer = "answer" in entry or "final_answer" in entry
        return all(field in entry for field in required_fields) and has_answer

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single agent reasoning sample to single tool call conversation format.

        Args:
            sample: Agent reasoning sample with tool usage

        Returns:
            Formatted conversation sample with single tool call per message
        """
        if not self.validate(sample):
            return None

        # Get configuration instance - cast to proper type for type hints
        config: SingleToolCallConfig = (
            self._config_model
            if isinstance(self._config_model, SingleToolCallConfig)
            else SingleToolCallConfig()
        )

        # Extract components from agent reasoning sample
        question = sample["question"]
        tool_used = sample["tool_used"]
        tool_input = sample.get("tool_input", "{}")
        tool_output = sample.get("tool_output", "Tool execution completed.")
        answer = sample.get("answer") or sample.get("final_answer", "No answer provided")
        available_tools = sample.get("available_tools", [])

        # Parse tool input if it's a string
        if isinstance(tool_input, str):
            try:
                # Try to parse as JSON
                tool_args = json.loads(tool_input.replace("'", '"'))
            except (json.JSONDecodeError, AttributeError):
                # If parsing fails, create a simple structure
                tool_args = {"input": tool_input}
        else:
            tool_args = tool_input

        # Create messages list
        messages = []

        # Add system message with tools
        if config.include_tools_in_system and available_tools:
            tools_text = self._format_tools_for_system(available_tools)
            system_content = f"{config.system_prompt}\n\n{tools_text}"
            messages.append({"role": "system", "content": system_content})
        elif config.include_tools_in_system:
            # Even without specific tools, include the system prompt with generic functions note
            system_content = f"{config.system_prompt}\n\n{self._get_generic_tools_text()}"
            messages.append({"role": "system", "content": system_content})

        # Add user question
        messages.append({"role": "user", "content": question})

        # Add assistant message with tool call
        assistant_content = ""

        if config.include_reasoning_prefix:
            # Generate action description based on tool name
            action = self._get_tool_action_description(tool_used, sample)
            reasoning_prefix = config.reasoning_prefix_template.format(action=action)
            assistant_content = f"{reasoning_prefix}\n\n"

        # Add the tool call
        tool_call_json = json.dumps({"name": tool_used, "arguments": tool_args})
        tool_call = config.tool_call_format.format(tool_call=tool_call_json)
        assistant_content += tool_call

        messages.append({"role": "assistant", "content": assistant_content})

        # Add tool response
        if config.tool_response_as_json:
            # Format tool output as JSON
            try:
                # If tool_output is already a dict/JSON, use it directly
                if isinstance(tool_output, dict):
                    tool_response_content = json.dumps(tool_output)
                elif isinstance(tool_output, str):
                    # Try to parse it as JSON first
                    try:
                        parsed_output = json.loads(tool_output)
                        tool_response_content = json.dumps(parsed_output)
                    except json.JSONDecodeError:
                        # If not JSON, create a simple JSON structure
                        tool_response_content = json.dumps({"result": tool_output})
                else:
                    tool_response_content = json.dumps({"result": str(tool_output)})
            except Exception:
                tool_response_content = json.dumps({"result": str(tool_output)})
        else:
            tool_response_content = str(tool_output)

        messages.append({"role": "tool", "content": tool_response_content})

        # Handle multiple tool calls if present
        # Check if there are additional tool calls in the sample
        if "additional_tools" in sample:
            for additional_tool in sample["additional_tools"]:
                # Add assistant message for next tool call
                tool_name = additional_tool.get("name", "unknown_tool")
                tool_args = additional_tool.get("arguments", {})
                tool_result = additional_tool.get("result", "Tool executed")

                assistant_content = ""
                if config.include_reasoning_prefix:
                    action = self._get_tool_action_description(tool_name, additional_tool)
                    reasoning_prefix = config.reasoning_prefix_template.format(action=action)
                    assistant_content = f"{reasoning_prefix}\n\n"

                tool_call_json = json.dumps({"name": tool_name, "arguments": tool_args})
                tool_call = config.tool_call_format.format(tool_call=tool_call_json)
                assistant_content += tool_call

                messages.append({"role": "assistant", "content": assistant_content})

                # Add tool response
                if config.tool_response_as_json:
                    tool_response_content = json.dumps({"result": tool_result})
                else:
                    tool_response_content = str(tool_result)

                messages.append({"role": "tool", "content": tool_response_content})

        # Add final assistant answer (only once, after all tool calls)
        messages.append({"role": "assistant", "content": answer})

        return {"messages": messages}

    def _get_tool_action_description(self, tool_name: str, sample: dict) -> str:
        """Generate a natural language description of the tool action."""
        tool_actions = {
            "get_weather": "check the weather",
            "get_time": "check the current time",
            "calculator": "perform the calculation",
            "web_search": "search for that information",
            "database_query": "query the database",
            "api_call": "make the API call",
        }

        # Try to get a specific action or use a generic one
        action = tool_actions.get(tool_name, f"use the {tool_name} tool")

        # Check both tool_input and arguments keys for parameters
        if "tool_input" in sample or "arguments" in sample:
            try:
                tool_args = None
                if "tool_input" in sample:
                    tool_input = sample["tool_input"]
                    if isinstance(tool_input, str):
                        tool_args = json.loads(tool_input.replace("'", '"'))
                    else:
                        tool_args = tool_input
                elif "arguments" in sample:
                    tool_args = sample["arguments"]

                if tool_args:
                    if tool_name == "get_weather" and "location" in tool_args:
                        action = f"check the weather in {tool_args['location']}"
                    elif tool_name == "get_time" and "timezone" in tool_args:
                        action = f"check the time in {tool_args['timezone']}"
                    elif tool_name == "calculator" and "expression" in tool_args:
                        action = f"calculate {tool_args['expression']}"
                    elif tool_name == "web_search" and "query" in tool_args:
                        action = f"search for {tool_args['query']}"
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

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

    def _format_tools_for_system(self, available_tools: list[dict]) -> str:
        """Format available tools for inclusion in system message."""
        functions = []
        for tool in available_tools:
            # Convert to function format
            func_def = {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }

            # Add parameters if present
            for param in tool.get("parameters", []):
                func_def["parameters"]["properties"][param["name"]] = {
                    "type": param.get("type", "string"),
                    "description": param.get("description", ""),
                }
                if param.get("required", True):
                    func_def["parameters"]["required"].append(param["name"])

            functions.append(func_def)

        return json.dumps({"functions": functions}, indent=2)

    def format_conversation_sample(self, sample: ConversationSample) -> dict[str, Any]:
        """Format a conversation sample (if needed for compatibility)."""
        return {"messages": [msg.model_dump() for msg in sample.messages]}

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
