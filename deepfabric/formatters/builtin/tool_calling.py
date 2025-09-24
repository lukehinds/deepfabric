"""
Tool Calling format formatter.

This formatter transforms agent reasoning datasets into embedded tool execution
format with proper tool call traces, similar to how real agents execute tools
during conversations.

The formatter converts from:
{
  "question": "What's the weather like?",
  "reasoning": "Need to check weather...",
  "tool_used": "get_weather",
  "tool_input": "{'location': 'NYC'}",
  "tool_output": "72°F, sunny",
  "answer": "It's sunny and 72°F"
}

To:
{
  "messages": [
    {"role": "user", "content": "What's the weather like?"},
    {"role": "assistant", "content": "<think>Need to check weather...</think><tool_call>\n{'name': 'get_weather', 'arguments': {'location': 'NYC'}}\n</tool_call>"},
    {"role": "tool", "content": "<tool_response>\n72°F, sunny\n</tool_response>"},
    {"role": "assistant", "content": "It's sunny and 72°F"}
  ]
}
"""

import json

from typing import Any

from pydantic import BaseModel, Field

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

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)

    def get_config_model(self) -> type[BaseModel] | None:
        """Return the configuration model for this formatter."""
        return ToolCallingConfig

    def validate(self, sample: dict) -> bool:
        """Validate that sample has required fields for tool calling format."""
        required_fields = ["question", "tool_used"]
        # Check for either "answer" (SimpleAgentCoT) or "final_answer" (HybridAgentCoT)
        has_answer = "answer" in sample or "final_answer" in sample
        return all(field in sample for field in required_fields) and has_answer

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single agent reasoning sample to tool calling conversation format.

        Args:
            sample: Agent reasoning sample with tool usage

        Returns:
            Formatted conversation sample with embedded tool execution
        """
        if not self.validate(sample):
            return None

        # Get configuration instance (created by base class _setup_config)
        config = self._config_model or ToolCallingConfig()

        # Extract components from agent reasoning sample
        question = sample["question"]
        tool_used = sample["tool_used"]
        tool_input = sample.get("tool_input", "{}")
        tool_output = sample.get("tool_output", "Tool execution completed.")
        answer = sample.get("answer") or sample.get("final_answer", "No answer provided")
        available_tools = sample.get("available_tools", [])

        # Build comprehensive reasoning from available fields
        reasoning_parts = []

        # Check if this is rich agent CoT format (SimpleAgentCoT)
        if "initial_analysis" in sample:
            reasoning_parts.append(f"Analysis: {sample['initial_analysis']}")

            if "reasoning_steps" in sample and sample["reasoning_steps"]:
                reasoning_parts.append("Step-by-step reasoning:")
                for i, step in enumerate(sample["reasoning_steps"], 1):
                    reasoning_parts.append(f"{i}. {step}")

            if "tool_selection_rationale" in sample:
                reasoning_parts.append(f"Tool selection: {sample['tool_selection_rationale']}")

            if "parameter_reasoning" in sample:
                reasoning_parts.append(f"Parameters: {sample['parameter_reasoning']}")

        # Check if this is hybrid agent CoT format (HybridAgentCoT)
        elif "chain_of_thought" in sample and "reasoning_trace" in sample:
            reasoning_parts.append(f"Analysis: {sample['chain_of_thought']}")

            if sample["reasoning_trace"]:
                reasoning_parts.append("Step-by-step reasoning:")
                for step in sample["reasoning_trace"]:
                    step_text = f"Step {step['step_number']}: {step['thought']}"
                    if step.get("action"):
                        step_text += f" → {step['action']}"
                    reasoning_parts.append(step_text)

            if "tool_selection_rationale" in sample:
                reasoning_parts.append(f"Tool selection: {sample['tool_selection_rationale']}")

            if "parameter_reasoning" in sample:
                reasoning_parts.append(f"Parameters: {sample['parameter_reasoning']}")

        else:
            # Fallback to simple reasoning field
            reasoning_parts.append(
                sample.get("reasoning", "I need to use a tool to help answer this question.")
            )

        reasoning = "\n\n".join(reasoning_parts)

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

        # Add system message with tools (if configured)
        if config.include_tools_in_system and available_tools:
            tools_text = self._format_tools_for_system(available_tools)
            system_content = f"{config.system_prompt}\n\nHere are the available tools:\n<tools>\n{tools_text}\n</tools>"
            messages.append({"role": "system", "content": system_content})

        # Add user question
        messages.append({"role": "user", "content": question})

        # Add model response with thinking and tool call
        thinking = config.thinking_format.format(reasoning=reasoning)
        tool_call_json = json.dumps({"name": tool_used, "arguments": tool_args})
        tool_call = config.tool_call_format.format(tool_call=tool_call_json)
        model_response = f"{thinking}{tool_call}"
        messages.append({"role": "assistant", "content": model_response})

        # Add tool response
        tool_response = config.tool_response_format.format(tool_output=tool_output)
        messages.append({"role": "tool", "content": tool_response})

        # Add final model answer with optional result interpretation
        final_response = answer
        if "result_interpretation" in sample:
            final_response = f"{sample['result_interpretation']}\n\n{answer}"

        messages.append({"role": "assistant", "content": final_response})

        return {"messages": messages}

    def _format_tools_for_system(self, available_tools: list[dict]) -> str:
        """Format available tools for inclusion in system message."""
        tools_list = []
        for tool in available_tools:
            # Convert to OpenAI-style function format
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

            # Add parameters
            for param in tool.get("parameters", []):
                tool_def["function"]["parameters"]["properties"][param["name"]] = {
                    "type": param["type"],
                    "description": param["description"],
                }
                if param.get("required", True):
                    tool_def["function"]["parameters"]["required"].append(param["name"])

            tools_list.append(tool_def)

        return json.dumps(tools_list, indent=2)

    def format_conversation_sample(self, sample: ConversationSample) -> dict[str, Any]:
        """Format a conversation sample (if needed for compatibility)."""
        return {"messages": [msg.model_dump() for msg in sample.messages]}

    def get_example_config(self) -> dict[str, Any]:
        """Return example configuration for this formatter."""
        return {
            "system_prompt": "You are a helpful AI assistant with access to tools.",
            "include_tools_in_system": True,
            "thinking_format": "<think>{reasoning}</think>",
            "tool_call_format": "<tool_call>\n{tool_call}\n</tool_call>",
            "tool_response_format": "<tool_response>\n{tool_output}\n</tool_response>",
        }
