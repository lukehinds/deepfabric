"""
ChatML (Chat Markup Language) format formatter.

This formatter transforms datasets to the ChatML format, which is a standardized
way of representing conversations with clear role delineation and special tokens.

ChatML format uses special tokens to mark conversation boundaries:
- <|im_start|>role
- <|im_end|>

Supports tool calling via <tool_call> and <tool_response> XML tags.
Supports reasoning via <think> tags.

Reference: https://github.com/openai/openai-python/blob/main/chatml.md
"""

import json
import re

from typing import Any

from pydantic import BaseModel

from ...schemas import Conversation, ToolDefinition
from ..base import BaseFormatter
from ..models import ChatmlConfig, ChatmlStructuredOutput, ChatmlTextOutput


class ChatmlFormatter(BaseFormatter):
    """
    Formatter for ChatML (Chat Markup Language) format.

    Transforms DeepFabric datasets to ChatML format with proper
    role delineation and conversation structure.
    """

    # Common English words that are unlikely to be function names
    # Used as a heuristic filter in function name extraction
    # Note: This blacklist may filter out legitimate functions named 'tool' or 'function'
    _COMMON_WORDS_BLACKLIST = {"a", "an", "the", "this", "that", "it", "tool", "function"}

    def __init__(self, config: "dict[str, Any] | None" = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

        # Access configuration through typed model if available
        if self._config_model:
            if isinstance(self._config_model, ChatmlConfig):
                chatml_config: ChatmlConfig = self._config_model
            else:
                chatml_config = ChatmlConfig.model_validate(self._config_model)
            self.start_token = chatml_config.start_token
            self.end_token = chatml_config.end_token
            self.output_format = chatml_config.output_format
            self.default_system_message = chatml_config.default_system_message
            self.require_system_message = chatml_config.require_system_message

    def _format_single_sample(self, sample: Conversation | dict | Any) -> dict | None:
        """
        Format a single sample to ChatML format.

        Args:
            sample: A Conversation Pydantic model, dict, or anything with model_dump()

        Returns:
            Formatted sample in ChatML format or None if formatting fails
        """
        # Convert to Conversation model
        if isinstance(sample, Conversation):
            conversation = sample
        elif isinstance(sample, dict):
            # Convert dict to Conversation
            try:
                conversation = Conversation(**sample)
            except Exception:
                return None
        elif hasattr(sample, "model_dump"):
            # Convert from other Pydantic models
            try:
                conversation = Conversation(**sample.model_dump())
            except Exception:
                return None
        else:
            return None

        # Format using Pydantic model
        return self._format_conversation(conversation)

    def _format_conversation(self, conversation: Conversation) -> dict[str, Any]:
        """
        Format a Conversation model to ChatML format.

        Args:
            conversation: Pydantic Conversation model

        Returns:
            Formatted ChatML output
        """
        # Check if this is a multi-turn agent conversation
        # Only use multi-turn formatting if mode is multi_turn AND there are multiple user messages
        user_message_count = sum(1 for msg in conversation.messages if msg.role == "user")
        is_multi_turn = (
            conversation.agent_context
            and hasattr(conversation.agent_context, "mode")
            and conversation.agent_context.mode == "multi_turn"
            and user_message_count > 1
        )

        if is_multi_turn:
            messages = self._format_multi_turn(conversation)
        else:
            messages = self._format_single_turn(conversation)

        # Return in requested format
        if self.output_format == "text":
            return {"text": self._messages_to_chatml_text(messages)}
        return {"messages": messages}

    def _format_single_turn(self, conversation: Conversation) -> list[dict[str, Any]]:
        """
        Format a single-turn conversation (original logic).

        Args:
            conversation: Pydantic Conversation model

        Returns:
            List of formatted messages
        """
        # Build messages list from conversation
        messages = []

        # Get conversation-used tools only (same as multi-turn)
        used_tools = []
        if conversation.tool_context and conversation.tool_context.available_tools:
            used_tools = self._get_conversation_used_tools(
                conversation.tool_context.available_tools,
                conversation.tool_context.executions or [],
            )

        # Add system message with filtered tools
        system_content = self._build_system_message(conversation, tools_override=used_tools)
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Track if we've added reasoning and tool calls to first assistant message
        first_assistant_processed = False

        # Process conversation messages with tool calls and reasoning
        for msg in conversation.messages:
            # Skip system messages - already handled above
            if msg.role == "system":
                continue

            if msg.role == "assistant":
                if not first_assistant_processed:
                    # First assistant message: add reasoning and tool calls
                    content = self._build_assistant_content(msg, conversation)
                    messages.append({"role": "assistant", "content": content})
                    first_assistant_processed = True
                else:
                    # Subsequent assistant messages: just use content as-is
                    messages.append({"role": "assistant", "content": msg.content})
            elif msg.role == "tool":
                # Tool messages are already in the conversation, just format them
                if not msg.content.startswith("<tool_response>"):
                    tool_response = f"<tool_response>\n{msg.content}\n</tool_response>"
                    messages.append({"role": "tool", "content": tool_response})
                else:
                    messages.append({"role": "tool", "content": msg.content})
            else:
                # Regular user messages (not system, not assistant, not tool)
                messages.append({"role": msg.role, "content": msg.content})

        return messages

    def _detect_turns(self, conversation: Conversation) -> list[dict[str, Any]]:
        """
        Detect turn boundaries in multi-turn agent conversations.

        A turn consists of:
        1. User message
        2. Assistant message(s) (may include reasoning and tool calls)
        3. Tool response messages (if tools were called)
        4. Optional final assistant message (summary/response)

        Args:
            conversation: Conversation model

        Returns:
            List of turn dictionaries, each containing:
            - user_msg: ChatMessage
            - assistant_msgs: list[ChatMessage]
            - reasoning_steps: list[ReasoningStep] for this turn
            - tool_executions: list[ToolExecution] for this turn
        """
        turns = []
        current_turn = None
        execution_index = 0

        # Get total executions and reasoning steps
        total_executions = conversation.tool_context.executions if conversation.tool_context else []
        total_reasoning_steps = (
            conversation.reasoning.content
            if conversation.reasoning and isinstance(conversation.reasoning.content, list)
            else []
        )

        for msg in conversation.messages:
            if msg.role == "system":
                continue

            if msg.role == "user":
                # Start a new turn when we see a user message
                if current_turn is not None:
                    turns.append(current_turn)

                current_turn = {
                    "user_msg": msg,
                    "assistant_msgs": [],
                    "reasoning_steps": [],
                    "tool_executions": [],
                }

            elif msg.role == "assistant" and current_turn is not None:
                current_turn["assistant_msgs"].append(msg)

            elif msg.role == "tool" and current_turn is not None:
                # Tool messages can be either "calls" (intent) or "responses" (results)
                # Responses contain " -> " pattern indicating execution result
                # Only count actual responses (with results) for execution mapping
                if (
                    " -> " in msg.content or "successfully" in msg.content.lower()
                ) and execution_index < len(total_executions):
                    # This is a tool response message
                    current_turn["tool_executions"].append(total_executions[execution_index])
                    execution_index += 1

        # Add the last turn
        if current_turn is not None:
            turns.append(current_turn)

        # Distribute reasoning steps across turns by matching actions to executions sequentially
        if total_reasoning_steps and turns:
            # Build a sequential list of (function_name, turn_idx) pairs
            execution_order = []
            for turn_idx, turn in enumerate(turns):
                for execution in turn["tool_executions"]:
                    execution_order.append((execution.function_name, turn_idx))

            # Track which execution we're on
            execution_idx = 0
            current_turn_idx = 0

            for step in total_reasoning_steps:
                # Check if step has an action that matches an execution
                action = getattr(step, "action", None)
                if action and isinstance(action, str):
                    # Try to extract function name from action string
                    # This handles multiple formats:
                    # 1. Function call format: "function_name(...)"
                    # 2. Plain function name: "function_name"
                    # 3. Descriptive text containing function name (best-effort)
                    func_name = self._extract_function_name(action)

                    # Check if this matches the next expected execution
                    if execution_idx < len(execution_order):
                        expected_func, expected_turn = execution_order[execution_idx]
                        if func_name and func_name == expected_func:
                            # This step corresponds to the next execution
                            turns[expected_turn]["reasoning_steps"].append(step)
                            current_turn_idx = expected_turn
                            execution_idx += 1
                        # Step mentions a function but doesn't match next execution
                        # (e.g., it's explaining previous turn or planning ahead)
                        # Assign to current turn
                        elif current_turn_idx < len(turns):
                            turns[current_turn_idx]["reasoning_steps"].append(step)
                    # No more executions, assign to current turn
                    elif current_turn_idx < len(turns):
                        turns[current_turn_idx]["reasoning_steps"].append(step)
                # No action field (e.g., conclusion or planning step)
                # Assign to current turn
                elif current_turn_idx < len(turns):
                    turns[current_turn_idx]["reasoning_steps"].append(step)

        return turns

    def _extract_function_name(self, action: str) -> str | None:
        """
        Extract function name from action string with robust parsing.

        Handles multiple formats:
        1. Function call format: "function_name(...)" -> "function_name"
        2. Plain function name: "function_name" -> "function_name"
        3. Descriptive text: "I will call get_weather tool" -> "get_weather"

        Args:
            action: Action string from ReasoningStep

        Returns:
            Extracted function name, or None if no function name found

        Note:
            For best results, LLMs should be prompted to use one of these formats in the action field:
            - Plain function name: "get_weather"
            - Function call: "get_weather(city='Paris')"
            Descriptive text parsing is best-effort and may fail for complex sentences.

        Limitations:
            The common words blacklist (_COMMON_WORDS_BLACKLIST) filters out words like
            'tool' and 'function', which could prevent extraction if a function is
            legitimately named 'tool' or 'function'. This is an accepted trade-off to
            reduce false positives from natural language text.
        """
        if not action:
            return None

        action = action.strip()

        # Format 1: Function call format "function_name(...)"
        if "(" in action:
            func_name = action.split("(")[0].strip()
            # Validate it looks like a function name (alphanumeric + underscore)
            if func_name and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", func_name):
                return func_name

        # Format 2: Check if the entire string is a valid function name
        if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", action):
            return action

        # Format 3: Try to extract from descriptive text (best-effort)
        # Look for patterns like "call X", "use X", "execute X", "run X"
        # where X is a valid function name (not common words like "the", "a")
        match = re.search(
            r"\b(?:call|use|execute|run|invoke)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b",
            action,
            re.IGNORECASE,
        )
        if match:
            func_name = match.group(1)
            # Filter out common English words that aren't function names
            if func_name.lower() not in self._COMMON_WORDS_BLACKLIST:
                return func_name

        # If all else fails, return None - this reasoning step doesn't clearly reference a tool
        return None

    def _format_multi_turn(self, conversation: Conversation) -> list[dict[str, Any]]:
        """
        Format multi-turn agent conversation with proper turn boundaries.

        Creates structure:
        - Single system message with conversation-used tools only
        - Per-turn: user → assistant (with turn reasoning & tool calls) → tool responses

        Args:
            conversation: Conversation model with agent_context.mode == "multi_turn"

        Returns:
            List of formatted messages
        """
        messages = []

        # Get conversation-used tools only
        used_tools = []
        if conversation.tool_context and conversation.tool_context.available_tools:
            used_tools = self._get_conversation_used_tools(
                conversation.tool_context.available_tools,
                conversation.tool_context.executions or [],
            )

        # Build system message with filtered tools
        system_content = self._build_system_message(conversation, tools_override=used_tools)
        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Detect turns and format each one
        turns = self._detect_turns(conversation)

        for turn in turns:
            # Add user message
            messages.append({"role": "user", "content": turn["user_msg"].content})

            # Build assistant message with turn-specific reasoning and tool calls
            # Use the first assistant message as the base
            if turn["assistant_msgs"]:
                assistant_msg = turn["assistant_msgs"][0]
                content = self._build_assistant_content(
                    assistant_msg,
                    conversation,
                    reasoning_steps=turn["reasoning_steps"],
                    tool_executions=turn["tool_executions"],
                )
                messages.append({"role": "assistant", "content": content})

                # Add tool response messages for this turn
                for execution in turn["tool_executions"]:
                    tool_response = f"<tool_response>\n{execution.result}\n</tool_response>"
                    messages.append({"role": "tool", "content": tool_response})

                # Add final assistant response if there's more than one assistant message
                if len(turn["assistant_msgs"]) > 1:
                    for additional_msg in turn["assistant_msgs"][1:]:
                        messages.append({"role": "assistant", "content": additional_msg.content})

        return messages

    def _build_system_message(
        self, conversation: Conversation, tools_override: list[ToolDefinition] | None = None
    ) -> str:
        """
        Build system message with optional tools section.

        Args:
            conversation: Conversation model
            tools_override: Optional list of tools to use instead of all available tools.
                          Used for multi-turn to include only conversation-used tools.

        Returns:
            System message content
        """
        # Get base system message from conversation
        system_content = ""
        for msg in conversation.messages:
            if msg.role == "system":
                system_content = msg.content
                break

        # If no system message and required, use default
        if not system_content and self.require_system_message:
            system_content = self.default_system_message

        # Determine which tools to include
        tools_to_include = (
            tools_override
            if tools_override is not None
            else (conversation.tool_context.available_tools if conversation.tool_context else None)
        )

        # Add tools section if present
        if tools_to_include:
            tools_json = self._format_tools_as_json(tools_to_include)

            tools_header = (
                "\n\nYou are provided with function signatures within <tools></tools> XML tags. "
                "You may call one or more functions to assist with the user query. "
                "For each function call return a json object with function name and arguments "
                "within <tool_call></tool_call> XML tags.\n\n"
                f"Here are the available tools:\n<tools>\n{tools_json}\n</tools>"
            )

            if system_content:
                # Append to existing system message (preserves dataset_system_prompt)
                system_content = f"{system_content}{tools_header}"
            else:
                # Create new system message with tools
                system_content = (
                    "You are a function calling AI model. You are provided with function signatures\n"
                    "within <tools></tools> XML tags. You may call one or more functions to assist\n"
                    "with the user query. For each function call return a json object with function\n"
                    "name and arguments within <tool_call></tool_call> XML tags.\n\n"
                    f"Here are the available tools:\n<tools>\n{tools_json}\n</tools>"
                )

        return system_content

    def _build_assistant_content(
        self,
        message: Any,
        conversation: Conversation,
        reasoning_steps: list[Any] | None = None,
        tool_executions: list[Any] | None = None,
    ) -> str:
        """
        Build assistant message content with reasoning and tool calls.

        Args:
            message: ChatMessage from conversation
            conversation: Parent Conversation model
            reasoning_steps: Optional list of reasoning steps for this specific turn.
                           If None, uses all reasoning from conversation (single-turn behavior).
            tool_executions: Optional list of tool executions for this specific turn.
                           If None, uses all executions from conversation (single-turn behavior).

        Returns:
            Formatted assistant content with <think> and <tool_call> tags
        """
        content_parts = []
        reasoning_extracted = False

        # Determine which reasoning to use
        if reasoning_steps is not None:
            # Multi-turn: use turn-specific reasoning
            if reasoning_steps:
                reasoning_text = self._format_reasoning_steps(
                    reasoning_steps,
                    conversation.reasoning.style if conversation.reasoning else "structured",
                )
                if reasoning_text:
                    content_parts.append(f"<think>\n{reasoning_text}\n</think>\n")
                    reasoning_extracted = True
        # Single-turn: use all reasoning from conversation
        elif conversation.reasoning:
            reasoning_text = self._format_reasoning(conversation.reasoning)
            if reasoning_text:
                content_parts.append(f"<think>\n{reasoning_text}\n</think>\n")
                reasoning_extracted = True

        # For chain-of-thought datasets: if reasoning was extracted and final_answer exists,
        # use final_answer instead of full message content to avoid duplication
        if reasoning_extracted and conversation.final_answer:
            content_parts.append(conversation.final_answer)
        # Otherwise, use original message content
        elif message.content:
            content_parts.append(message.content)

        # Determine which tool executions to use
        executions_to_format = (
            tool_executions
            if tool_executions is not None
            else (conversation.tool_context.executions if conversation.tool_context else [])
        )

        # Add tool calls if present
        if executions_to_format:
            for execution in executions_to_format:
                try:
                    # Parse arguments if string, otherwise use as-is
                    if isinstance(execution.arguments, str):
                        arguments = json.loads(execution.arguments)
                    else:
                        arguments = execution.arguments

                    tool_call_json = {"name": execution.function_name, "arguments": arguments}
                    content_parts.append(
                        f"\n<tool_call>\n{json.dumps(tool_call_json)}\n</tool_call>"
                    )
                except json.JSONDecodeError:
                    # Skip malformed tool calls - log warning but continue formatting
                    continue

        return "".join(content_parts)

    def _format_reasoning(self, reasoning: Any) -> str:
        """
        Format reasoning trace for <think> tags.

        Args:
            reasoning: ReasoningTrace from Conversation

        Returns:
            Formatted reasoning text with normalized whitespace
        """
        if not reasoning:
            return ""

        if reasoning.style == "freetext":
            content = reasoning.content if isinstance(reasoning.content, str) else ""
            # Normalize whitespace: replace blank lines with single space
            return self._normalize_whitespace(content)
        if reasoning.style in ("structured", "hybrid") and isinstance(reasoning.content, list):
            # Format structured reasoning steps
            return self._format_reasoning_steps(reasoning.content, reasoning.style)
        return ""

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace in reasoning text for training efficiency.

        Replaces multiple newlines (blank lines) with single spaces while
        preserving single line breaks between sentences.

        Args:
            text: Raw reasoning text

        Returns:
            Text with normalized whitespace
        """
        # Replace multiple newlines (blank lines) with single space
        # Pattern: \n\n+ matches two or more consecutive newlines
        normalized = re.sub(r"\n\n+", " ", text)

        # Replace remaining single newlines with spaces
        normalized = normalized.replace("\n", " ")

        # Collapse multiple spaces into single space
        normalized = re.sub(r" +", " ", normalized)

        return normalized.strip()

    def _format_reasoning_steps(self, steps: list[Any], _style: str) -> str:
        """
        Format a list of reasoning steps.

        Args:
            steps: List of ReasoningStep objects
            _style: Reasoning style ("structured", "hybrid", etc.) - unused but kept for compatibility

        Returns:
            Formatted reasoning text
        """
        if not steps:
            return ""

        formatted_steps = []
        for step in steps:
            if hasattr(step, "thought"):
                thought = step.thought
                action = getattr(step, "action", None)
                # Format thought and action naturally without prefixes
                if action:
                    # Combine thought and action in a natural flow
                    formatted_steps.append(f"{thought} → {action}")
                else:
                    formatted_steps.append(thought)
        # Join steps with single space instead of newlines
        return " ".join(formatted_steps)

    def _format_tools_as_json(self, tools: list[ToolDefinition]) -> str:
        """
        Format tools as JSON array for ChatML <tools> section.

        Args:
            tools: List of ToolDefinition models

        Returns:
            JSON string of tools in OpenAI format
        """
        formatted_tools = []

        for tool in tools:
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }

            # Add parameters
            if tool.parameters:
                properties = {}
                required = []

                for param in tool.parameters:
                    properties[param.name] = {"type": param.type, "description": param.description}
                    if param.required:
                        required.append(param.name)

                formatted_tool["function"]["parameters"]["properties"] = properties
                formatted_tool["function"]["parameters"]["required"] = required

            formatted_tools.append(formatted_tool)

        return json.dumps(formatted_tools, indent=1)

    def _messages_to_chatml_text(self, messages: list[dict[str, str]]) -> str:
        """Convert messages list to ChatML text format."""
        chatml_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"].rstrip()  # Remove trailing whitespace

            chatml_parts.append(f"{self.start_token}{role}")
            chatml_parts.append(content)
            chatml_parts.append(self.end_token)

        return "\n".join(chatml_parts)

    def _is_valid_message(self, message: dict[str, Any]) -> bool:
        """Check if a message has valid structure."""
        if not isinstance(message, dict):
            return False

        if "role" not in message or "content" not in message:
            return False

        role = message["role"]
        content = message["content"]

        # Valid roles
        valid_roles = ["system", "user", "assistant", "function", "tool"]
        if role not in valid_roles:
            return False

        # Content should be a non-empty string
        return isinstance(content, str) and bool(content.strip())

    def validate(self, entry: Conversation | Any) -> bool:
        """
        Validate that a Conversation can be formatted for ChatML.

        Args:
            entry: Conversation Pydantic model

        Returns:
            True if the entry can be formatted, False otherwise
        """
        # Try to convert to Conversation if not already
        if isinstance(entry, Conversation):
            conversation = entry
        elif hasattr(entry, "model_dump"):
            try:
                conversation = Conversation(**entry.model_dump())
            except Exception:
                return False
        else:
            return False

        # Must have at least one message
        if not conversation.messages or len(conversation.messages) == 0:
            return False

        # Should have at least one user and one assistant message
        roles = [msg.role for msg in conversation.messages]
        return "user" in roles and "assistant" in roles

    def validate_output(self, entry: dict[str, Any]) -> bool:  # noqa: PLR0911
        """
        Validate that a formatted entry meets ChatML requirements.

        Args:
            entry: Formatted entry to validate

        Returns:
            True if the entry meets ChatML format requirements
        """
        if not isinstance(entry, dict):
            return False

        if self.output_format == "text":
            # Check for text format
            if "text" not in entry:
                return False

            text = entry["text"]
            if not isinstance(text, str):
                return False

            # Should contain ChatML tokens
            return self.start_token in text and self.end_token in text

        # Check for structured format
        if "messages" not in entry:
            return False

        messages = entry["messages"]
        if not isinstance(messages, list):
            return False

        # Validate each message
        for message in messages:
            if not self._is_valid_message(message):
                return False

        # Should have at least user and assistant roles
        roles = [msg["role"] for msg in messages]
        return "user" in roles and "assistant" in roles

    def get_description(self) -> str:
        """Get description of the ChatML formatter."""
        return """
        ChatML (Chat Markup Language) format formatter.

        Transforms datasets to ChatML format with proper role delineation:
        - Supports both structured messages and text formats
        - Uses configurable start/end tokens for role boundaries
        - Handles system, user, and assistant roles
        - Can enforce system message presence

        Output formats:
        - structured: {"messages": [...]} format
        - text: Single text string with ChatML markup
        """

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "question_answer", "instruction_response", "generic"]

    def get_config_model(self) -> type[BaseModel]:
        """Get the Pydantic model for ChatML configuration."""
        return ChatmlConfig

    def get_output_model(self) -> type[BaseModel]:
        """Get the Pydantic model for ChatML output."""
        # Return different models based on output format
        if self.output_format == "text":
            return ChatmlTextOutput
        return ChatmlStructuredOutput
