"""
OpenAI Harmony format formatter.

This formatter transforms datasets to the OpenAI Harmony format, which is designed
for gpt-oss models and provides structured conversation with channels and tool support.

Harmony format uses special tokens and channels:
- <|start|>role<|message|> to start messages
- <|end|> to end messages
- Channels: final (user-facing), analysis (reasoning), commentary (tool calls)
- Supports TypeScript-style function definitions

Reference: https://github.com/openai/harmony
"""

from typing import Any

from pydantic import BaseModel

from ..base import BaseFormatter, FormatterError
from ..models import HarmonyConfig, HarmonyMessage, HarmonyStructuredOutput, HarmonyTextOutput


class HarmonyFormatter(BaseFormatter):
    """
    Formatter for OpenAI Harmony format.

    Transforms DeepFabric datasets to Harmony format with proper
    role hierarchy, channels, and tool support for gpt-oss models.
    """

    def __init__(self, config: "dict[str, Any] | None" = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

        # BaseFormatter's __init__ calls get_config_model() and populates self._config_model
        # with a validated HarmonyConfig instance from the config dict
        if not self._config_model:
            # If no config was provided, create with defaults
            self._config_model = HarmonyConfig()
        elif not isinstance(self._config_model, HarmonyConfig):
            # This shouldn't happen if get_config_model() is properly implemented
            raise FormatterError("Configuration model is not a valid HarmonyConfig instance.")

        harmony_config: HarmonyConfig = self._config_model

        # Set instance attributes from the config model
        self.start_token = harmony_config.start_token
        self.end_token = harmony_config.end_token
        self.message_token = harmony_config.message_token
        self.output_format = harmony_config.output_format
        self.default_channel = harmony_config.default_channel
        self.include_developer_role = harmony_config.include_developer_role
        self.developer_instructions = harmony_config.developer_instructions
        self.system_message = harmony_config.system_message
        self.reasoning_level = harmony_config.reasoning_level
        self.knowledge_cutoff = harmony_config.knowledge_cutoff
        self.current_date = harmony_config.current_date
        self.include_metadata = harmony_config.include_metadata
        self.tool_namespace = harmony_config.tool_namespace

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to Harmony format.

        Args:
            sample: A single dataset sample

        Returns:
            Formatted sample in Harmony format or None if formatting fails
        """
        if not self.validate(sample):
            return None

        # Handle different input formats
        if "messages" in sample:
            return self._format_messages_sample(sample)
        if "question" in sample and ("answer" in sample or "final_answer" in sample):
            return self._format_qa_sample(sample)
        return self._format_generic_sample(sample)

    def _format_messages_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format a sample that already has messages structure."""
        messages = []
        input_messages = sample["messages"].copy()

        # Add system message with metadata
        system_content = self._build_system_message(sample)
        messages.append(
            HarmonyMessage(role="system", content=system_content, channel=None, recipient=None)
        )

        # Add developer message if configured
        if self.include_developer_role:
            dev_message = self._build_developer_message(sample)
            if dev_message:  # Only add if there's content
                messages.append(
                    HarmonyMessage(
                        role="developer",
                        content=dev_message,
                        channel=None,
                        recipient=None,
                    )
                )

        # Process input messages
        for msg in input_messages:
            if not self._is_valid_message(msg):
                continue

            role = msg["role"]
            content = msg["content"]

            # Skip system messages as we handle them separately
            if role == "system":
                continue

            # Determine channel for assistant messages
            channel = None
            recipient = None
            if role == "assistant":
                # Check if this is a tool call or reasoning
                if "tool_calls" in msg and msg["tool_calls"]:
                    # Handle multiple tool calls by creating separate messages for each
                    for tool_call in msg["tool_calls"]:
                        tool_recipient = None
                        if "function" in tool_call:
                            tool_recipient = (
                                f"{self.tool_namespace}.{tool_call['function']['name']}"
                            )
                        messages.append(
                            HarmonyMessage(
                                role=role,
                                content=content,
                                channel="commentary",
                                recipient=tool_recipient,
                            )
                        )
                    continue  # Skip the regular append since we handled it above
                if "function_call" in msg:
                    channel = "commentary"
                    # Handle legacy single function_call format
                    if "name" in msg["function_call"]:
                        recipient = f"{self.tool_namespace}.{msg['function_call']['name']}"
                elif "reasoning" in msg or "thinking" in msg:
                    channel = "analysis"
                else:
                    channel = self.default_channel

            messages.append(
                HarmonyMessage(role=role, content=content, channel=channel, recipient=recipient)
            )

        if self.output_format == "text":
            return {"text": self._messages_to_harmony_text(messages)}
        return {"messages": [msg.model_dump(exclude_none=True) for msg in messages]}

    def _format_qa_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format a Q&A sample to Harmony format."""
        messages = []

        # Add system message
        system_content = self._build_system_message(sample)
        messages.append(
            HarmonyMessage(role="system", content=system_content, channel=None, recipient=None)
        )

        # Add developer message if configured
        if self.include_developer_role:
            # Use context or instructions from sample if available
            dev_content = sample.get("instructions") or sample.get(
                "context", self.developer_instructions
            )
            if dev_content:
                messages.append(
                    HarmonyMessage(
                        role="developer", content=dev_content, channel=None, recipient=None
                    )
                )

        # Add user question
        question = sample["question"]
        messages.append(HarmonyMessage(role="user", content=question, channel=None, recipient=None))

        # Add assistant answer with appropriate channel
        answer = sample.get("answer") or sample.get("final_answer", "")

        # Check if there's reasoning/chain of thought
        if "chain_of_thought" in sample and sample["chain_of_thought"]:
            # Add analysis channel for reasoning
            messages.append(
                HarmonyMessage(
                    role="assistant",
                    content=sample["chain_of_thought"],
                    channel="analysis",
                    recipient=None,
                )
            )
            # Add final answer
            messages.append(
                HarmonyMessage(role="assistant", content=answer, channel="final", recipient=None)
            )
        else:
            # Single assistant message
            messages.append(
                HarmonyMessage(
                    role="assistant", content=answer, channel=self.default_channel, recipient=None
                )
            )

        if self.output_format == "text":
            return {"text": self._messages_to_harmony_text(messages)}
        return {"messages": [msg.model_dump(exclude_none=True) for msg in messages]}

    def _format_generic_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Try to format any sample by detecting conversation patterns."""
        messages = []

        # Add system message
        system_content = self._build_system_message(sample)
        messages.append(
            HarmonyMessage(role="system", content=system_content, channel=None, recipient=None)
        )

        # Look for instruction-like fields
        user_fields = ["instruction", "prompt", "question", "input", "query"]
        assistant_fields = ["output", "response", "answer", "solution", "final_answer"]
        developer_fields = ["instructions", "context", "background"]

        # Find developer content if enabled
        if self.include_developer_role:
            for field in developer_fields:
                if field in sample and sample[field]:
                    messages.append(
                        HarmonyMessage(
                            role="developer",
                            content=sample[field],
                            channel=None,
                            recipient=None,
                        )
                    )
                    break

        # Find user content
        user_content = None
        for field in user_fields:
            if field in sample and sample[field]:
                user_content = sample[field]
                break

        # Find assistant content
        assistant_content = None
        for field in assistant_fields:
            if field in sample and sample[field]:
                assistant_content = sample[field]
                break

        if not user_content or not assistant_content:
            return None

        messages.append(
            HarmonyMessage(role="user", content=user_content, channel=None, recipient=None)
        )

        # Check for reasoning content
        reasoning_fields = ["reasoning", "chain_of_thought", "thinking", "analysis"]
        reasoning_content = None
        for field in reasoning_fields:
            if field in sample and sample[field]:
                reasoning_content = sample[field]
                break

        if reasoning_content:
            # Add reasoning in analysis channel
            messages.append(
                HarmonyMessage(
                    role="assistant",
                    content=reasoning_content,
                    channel="analysis",
                    recipient=None,
                )
            )
            # Add final answer
            messages.append(
                HarmonyMessage(
                    role="assistant",
                    content=assistant_content,
                    channel="final",
                    recipient=None,
                )
            )
        else:
            messages.append(
                HarmonyMessage(
                    role="assistant",
                    content=assistant_content,
                    channel=self.default_channel,
                    recipient=None,
                )
            )

        if self.output_format == "text":
            return {"text": self._messages_to_harmony_text(messages)}
        return {"messages": [msg.model_dump(exclude_none=True) for msg in messages]}

    def _build_system_message(self, sample: dict[str, Any]) -> str:
        """Build the system message with optional metadata."""
        parts = [self.system_message]

        if self.include_metadata:
            # Add knowledge cutoff
            if self.knowledge_cutoff:
                parts.append(f"Knowledge cutoff: {self.knowledge_cutoff}")

            # Add current date if provided in config
            if self.current_date:
                parts.append(f"Current date: {self.current_date}")

            # Add reasoning level
            parts.append(f"Reasoning: {self.reasoning_level}")

            # Add valid channels info
            parts.append("# Valid channels: analysis, commentary, final")

        # Check for additional system content in sample
        if "system_prompt" in sample and sample["system_prompt"]:
            parts.append(sample["system_prompt"])

        return "\n".join(parts)

    def _build_developer_message(self, sample: dict[str, Any]) -> str:
        """Build the developer message with instructions and tools."""
        parts = []

        # Add developer instructions
        if self.developer_instructions:
            parts.append("# Instructions")
            parts.append(self.developer_instructions)

        # Check for tool definitions in sample
        if "tools" in sample and sample["tools"]:
            if parts:  # Add separator if we have instructions
                parts.append("")
            parts.append("# Tools")
            parts.append(f"namespace {self.tool_namespace} {{")

            for tool in sample["tools"]:
                if isinstance(tool, dict) and "name" in tool:
                    # Format as TypeScript-style function type - skip tools without names
                    name = tool["name"]
                    params = tool.get("parameters", {})
                    parts.append(f"  type {name} = ({self._format_tool_params(params)}) => any;")

            parts.append("}")

        return "\n".join(parts) if parts else ""

    def _format_tool_params(self, params: dict[str, Any]) -> str:
        """Format tool parameters as TypeScript-style type definition."""
        if not params or "properties" not in params:
            return ""

        properties = params.get("properties", {})
        required = params.get("required", [])

        param_strings = []
        for name, schema in properties.items():
            param_type = self._json_schema_to_ts_type(schema)
            optional = "?" if name not in required else ""
            param_strings.append(f"{name}{optional}: {param_type}")

        if param_strings:
            return f"_: {{ {', '.join(param_strings)} }}"
        return ""

    def _json_schema_to_ts_type(self, schema: dict[str, Any]) -> str:  # noqa: PLR0911
        """Convert JSON schema type to TypeScript type."""
        schema_type = schema.get("type", "any")

        if schema_type == "string":
            if "enum" in schema:
                return " | ".join(f'"{v}"' for v in schema["enum"])
            return "string"
        if schema_type in {"number", "integer"}:
            return "number"
        if schema_type == "boolean":
            return "boolean"
        if schema_type == "array":
            item_type = self._json_schema_to_ts_type(schema.get("items", {}))
            return f"{item_type}[]"
        if schema_type == "object":
            return "object"
        return "any"

    def _messages_to_harmony_text(self, messages: list[HarmonyMessage]) -> str:
        """Convert messages list to Harmony text format."""
        harmony_parts = []

        for message in messages:
            # Start token and role
            parts = [f"{self.start_token}{message.role}"]

            # Add channel for assistant messages
            if message.role == "assistant" and message.channel:
                parts.append(f"<|channel|>{message.channel}")

            # Add recipient for tool calls
            if message.recipient:
                parts.append(f"<|recipient|>{message.recipient}")

            # Add message token and content
            parts.append(self.message_token)
            harmony_parts.append("".join(parts))
            harmony_parts.append(message.content)
            harmony_parts.append(self.end_token)

        return "\n".join(harmony_parts)

    def _is_valid_message(self, message: dict[str, Any]) -> bool:
        """Check if a message has valid structure."""
        if not isinstance(message, dict):
            return False

        if "role" not in message or "content" not in message:
            return False

        role = message["role"]
        content = message["content"]

        # Valid roles in Harmony
        valid_roles = ["system", "developer", "user", "assistant", "tool"]
        if role not in valid_roles:
            return False

        # Content should be a non-empty string
        return isinstance(content, str) and bool(content.strip())

    def validate(self, entry: dict[str, Any]) -> bool:
        """
        Validate that an entry can be formatted for Harmony.

        Args:
            entry: Dataset entry to validate

        Returns:
            True if the entry can be formatted, False otherwise
        """
        if not super().validate(entry):
            return False

        # Check for messages format
        if "messages" in entry:
            messages = entry["messages"]
            if not isinstance(messages, list):
                return False

            # Should have at least one user and one assistant message
            roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
            return "user" in roles and "assistant" in roles

        # Check for Q&A format
        if "question" in entry and ("answer" in entry or "final_answer" in entry):
            return True

        # Check for any conversation pattern
        user_fields = ["instruction", "prompt", "question", "input", "query"]
        assistant_fields = ["output", "response", "answer", "solution", "final_answer"]

        has_user_content = any(field in entry for field in user_fields)
        has_assistant_content = any(field in entry for field in assistant_fields)

        return has_user_content and has_assistant_content

    def get_description(self) -> str:
        """Get description of the Harmony formatter."""
        return """
        OpenAI Harmony format formatter for gpt-oss models.

        Transforms datasets to Harmony format with:
        - Special tokens: <|start|>, <|end|>, <|message|>
        - Role hierarchy: system > developer > user > assistant > tool
        - Channel support: final (user-facing), analysis (reasoning), commentary (tool calls)
        - TypeScript-style function definitions for tools
        - Configurable metadata (knowledge cutoff, reasoning level)

        Output formats:
        - text: Single text string with Harmony tokens
        - structured: {"messages": [...]} format with channel info
        """

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "question_answer", "instruction_response", "generic"]

    def get_config_model(self) -> type[BaseModel]:
        """Get the Pydantic model for Harmony configuration."""
        return HarmonyConfig

    def get_output_model(self) -> type[BaseModel]:
        """Get the Pydantic model for Harmony output."""
        # Return different models based on output format
        if self.output_format == "text":
            return HarmonyTextOutput
        return HarmonyStructuredOutput
