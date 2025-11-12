"""
Tool Calling format formatter.

This formatter transforms agent reasoning datasets into embedded tool execution
format with proper tool call traces, similar to how real agents execute tools
during conversations. Supports multiple tool executions per query.

The formatter converts from:
{
  "question": "What's the weather like in NYC and London?",
  "reasoning": "Need to check weather in both cities...",I
  "tool_executions": [
    {"function": "get_weather", "arguments": {"location": "NYC"}, "result": "72°F, sunny"},
    {"function": "get_weather", "arguments": {"location": "London"}, "result": "15°C, rainy"}
  ],
  "answer": "NYC is sunny at 72°F, while London is rainy at 15°C"
}

To:
{
  "messages": [
    {"role": "user", "content": "What's the weather like in NYC and London?"},
    {"role": "assistant", "content": "<think>Need to check weather in both cities...</think><tool_call>\n{'name': 'get_weather', 'arguments': {'location': 'NYC'}}\n</tool_call><tool_call>\n{'name': 'get_weather', 'arguments': {'location': 'London'}}\n</tool_call>"},
    {"role": "tool", "content": "<tool_response>\n72°F, sunny\n</tool_response>"},
    {"role": "tool", "content": "<tool_response>\n15°C, rainy\n</tool_response>"},
    {"role": "assistant", "content": "NYC is sunny at 72°F, while London is rainy at 15°C"}
  ]
}
"""

import json

from typing import Any

from pydantic import BaseModel, Field

from ...schemas import Conversation, ToolDefinition
from ..base import BaseFormatter
from ..models import ConversationSample


class ToolCallingConfig(BaseModel):
    """Configuration for tool calling formatter."""

    system_prompt: str = Field(
        default="You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags.",
        description="System prompt that explains tool calling behavior",
    )
    include_tools_in_system: bool = Field(
        default=True, description="Whether to include available tools in system message"
    )
    thinking_format: str = Field(
        default="<think>{reasoning}</think>",
        description="Format for embedding reasoning (thinking) in model responses",
    )
    tool_call_format: str = Field(
        default="<tool_call>\n{tool_call}\n</tool_call>",
        description="Format for tool call XML tags",
    )
    tool_response_format: str = Field(
        default="<tool_response>\n{tool_output}\n</tool_response>",
        description="Format for tool response XML tags",
    )


class ToolCallingFormatter(BaseFormatter):
    """
    Formatter for embedded tool calling execution traces.

    Transforms agent reasoning datasets into conversational format
    that shows the actual execution flow of tool calls.
    """

    def __init__(self, config: dict[str, Any] | None = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

    def get_config_model(self) -> type[BaseModel] | None:
        """Return the configuration model for this formatter."""
        return ToolCallingConfig

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

    def _format_single_sample(self, sample: Conversation | dict | Any) -> dict | None:
        """
        Format a single agent reasoning sample to tool calling conversation format.

        Uses unified Conversation schema with tool_context.

        Args:
            sample: A Conversation Pydantic model, dict, or anything with model_dump()

        Returns:
            Formatted conversation sample with embedded tool execution
        """
        # Convert to Conversation model
        if isinstance(sample, Conversation):
            conversation = sample
        elif isinstance(sample, dict):
            try:
                conversation = Conversation(**sample)
            except Exception:
                return None
        elif hasattr(sample, "model_dump"):
            try:
                conversation = Conversation(**sample.model_dump())
            except Exception:
                return None
        else:
            return None

        # Validate has tool executions
        if not conversation.tool_context or not conversation.tool_context.executions:
            return None

        config: ToolCallingConfig = (
            self._config_model
            if isinstance(self._config_model, ToolCallingConfig)
            else ToolCallingConfig()
        )

        # Extract data from Conversation model using typed access
        question = conversation.question if conversation.question else ""
        tool_executions = conversation.tool_context.executions
        available_tools = (
            conversation.tool_context.available_tools
            if conversation.tool_context.available_tools
            else []
        )

        # Filter to only tools actually used in this conversation (saves tokens)
        used_tools = self._get_conversation_used_tools(available_tools, tool_executions)

        # Get answer from final_answer field
        answer = conversation.final_answer or "No answer provided"
        reasoning_parts = self._extract_reasoning(conversation)

        # Join with single newline - no need for blank lines in training data
        reasoning = "\n".join(reasoning_parts)

        # Create messages list
        messages = []

        # Add system message with tools (if configured)
        if config.include_tools_in_system and used_tools:
            tools_text = self._format_tools_for_system(used_tools)
            system_content = f"{config.system_prompt}\n\nHere are the available tools:\n<tools>\n{tools_text}\n</tools>"
            messages.append({"role": "system", "content": system_content})

        # Add user question
        messages.append({"role": "user", "content": question})

        # Add model response with thinking and tool calls
        thinking = config.thinking_format.format(reasoning=reasoning)

        tool_calls_text = ""
        for execution in tool_executions:
            try:
                # Try to get parsed arguments
                arguments = execution.parsed_arguments
                tool_call = {"name": execution.function_name, "arguments": arguments}
                tool_calls_text += config.tool_call_format.format(tool_call=json.dumps(tool_call))
            except json.JSONDecodeError:
                # Skip malformed tool calls
                continue

        model_response = f"{thinking}{tool_calls_text}"
        messages.append({"role": "assistant", "content": model_response})

        # Add tool responses (one per execution)
        for execution in tool_executions:
            result = execution.result
            tool_response = config.tool_response_format.format(tool_output=result)
            messages.append({"role": "tool", "content": tool_response})

        # Add final model answer
        # Check for result_interpretation in structured_data
        result_interpretation = ""
        if conversation.structured_data:
            result_interpretation = conversation.structured_data.get("result_interpretation", "")
        # Single newline between interpretation and answer - training data doesn't need blank lines
        final_response = f"{result_interpretation}\n{answer}" if result_interpretation else answer
        messages.append({"role": "assistant", "content": final_response})

        return {"messages": messages}

    def _extract_reasoning(self, conversation: Conversation) -> list[str]:
        """Extract reasoning parts from Conversation model."""
        reasoning_parts = []

        if not conversation.reasoning:
            return reasoning_parts

        reasoning = conversation.reasoning
        content = reasoning.content

        if isinstance(content, str):
            reasoning_parts.append(content)
        elif isinstance(content, list):
            reasoning_parts.append("Step-by-step reasoning:")
            for i, step in enumerate(content, 1):
                if hasattr(step, "thought"):
                    step_text = f"{i}. {step.thought}"
                    if hasattr(step, "action") and step.action:
                        step_text += f" → {step.action}"
                    reasoning_parts.append(step_text)
                else:
                    reasoning_parts.append(f"{i}. {step}")

        return reasoning_parts

    def _format_tools_for_system(self, available_tools: list) -> str:
        """
        Format available tools for inclusion in system message.

        Args:
            available_tools: List of ToolDefinition Pydantic models

        Returns:
            JSON string of tools in OpenAI schema format
        """
        tools_list = []
        for tool in available_tools:
            # Tools should already be ToolDefinition objects from Conversation model
            if not isinstance(tool, ToolDefinition):
                continue

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

            for param in tool.parameters:
                tool_def["function"]["parameters"]["properties"][param.name] = {
                    "type": param.type,
                    "description": param.description,
                }
                if param.required:
                    tool_def["function"]["parameters"]["required"].append(param.name)

            tools_list.append(tool_def)

        return json.dumps(tools_list, indent=2)

    def format_conversation_sample(self, sample: ConversationSample) -> dict[str, Any]:
        """Format a conversation sample (if needed for compatibility)."""
        return {"messages": [msg.model_dump(exclude_none=True) for msg in sample.messages]}

    def get_example_config(self) -> dict[str, Any]:
        """Return example configuration for this formatter."""
        return {
            "system_prompt": "You are a helpful AI assistant with access to tools.",
            "include_tools_in_system": True,
            "thinking_format": "<think>{reasoning}</think>",
            "tool_call_format": "<tool_call>\n{tool_call}\n</tool_call>",
            "tool_response_format": "<tool_response>\n{tool_output}\n</tool_response>",
        }
