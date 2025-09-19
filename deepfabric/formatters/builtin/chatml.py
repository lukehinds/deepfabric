"""
ChatML (Chat Markup Language) format formatter.

This formatter transforms datasets to the ChatML format, which is a standardized
way of representing conversations with clear role delineation and special tokens.

ChatML format uses special tokens to mark conversation boundaries:
- <|im_start|>role
- <|im_end|>

Reference: https://github.com/openai/openai-python/blob/main/chatml.md
"""

from typing import Any

from pydantic import BaseModel

from ..base import BaseFormatter
from ..models import ChatmlConfig, ChatmlStructuredOutput, ChatmlTextOutput


class ChatmlFormatter(BaseFormatter):
    """
    Formatter for ChatML (Chat Markup Language) format.

    Transforms DeepFabric datasets to ChatML format with proper
    role delineation and conversation structure.
    """

    def __init__(self, config: "dict[str, Any] | None" = None):
        super().__init__(config)

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
        else:
            # Fallback to dict-based config for backward compatibility
            self.start_token = self.config.get("start_token", "<|im_start|>")
            self.end_token = self.config.get("end_token", "<|im_end|>")
            self.output_format = self.config.get("output_format", "structured")
            self.default_system_message = self.config.get(
                "default_system_message", "You are a helpful assistant."
            )
            self.require_system_message = self.config.get("require_system_message", False)

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to ChatML format.

        Args:
            sample: A single dataset sample

        Returns:
            Formatted sample in ChatML format or None if formatting fails
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
        messages = sample["messages"].copy()

        # Ensure we have a system message if required
        if self.require_system_message:
            has_system = any(msg["role"] == "system" for msg in messages)
            if not has_system:
                messages.insert(0, {"role": "system", "content": self.default_system_message})

        # Validate and clean up messages
        cleaned_messages = []
        for msg in messages:
            if self._is_valid_message(msg):
                cleaned_messages.append({"role": msg["role"], "content": msg["content"]})

        if self.output_format == "text":
            return {"text": self._messages_to_chatml_text(cleaned_messages)}
        return {"messages": cleaned_messages}

    def _format_qa_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format a Q&A sample to ChatML format."""
        question = sample["question"]
        answer = sample.get("answer") or sample.get("final_answer", "")

        # Build messages list
        messages = []

        # Add system message if required or if context is available
        system_content = None
        if self.require_system_message:
            system_content = self.default_system_message

        # Check for additional context that could be system message
        context_fields = ["context", "background", "instructions", "system_prompt"]
        for field in context_fields:
            if field in sample and sample[field]:
                system_content = sample[field]
                break

        if system_content:
            messages.append({"role": "system", "content": system_content})

        # Add user question
        messages.append({"role": "user", "content": question})

        # Add assistant answer (include reasoning if available)
        output_content = answer
        if "chain_of_thought" in sample and sample["chain_of_thought"]:
            output_content = f"{sample['chain_of_thought']}\n\n{answer}"

        messages.append({"role": "assistant", "content": output_content})

        if self.output_format == "text":
            return {"text": self._messages_to_chatml_text(messages)}
        return {"messages": messages}

    def _format_generic_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        """Try to format any sample by detecting conversation patterns."""
        # Look for instruction-like fields for user messages
        user_fields = ["instruction", "prompt", "question", "input", "query"]
        assistant_fields = ["output", "response", "answer", "solution", "final_answer"]
        system_fields = ["system", "context", "background", "instructions"]

        user_content = None
        assistant_content = None
        system_content = None

        # Find user content
        for field in user_fields:
            if field in sample and sample[field]:
                user_content = sample[field]
                break

        # Find assistant content
        for field in assistant_fields:
            if field in sample and sample[field]:
                assistant_content = sample[field]
                break

        # Find system content
        for field in system_fields:
            if field in sample and sample[field]:
                system_content = sample[field]
                break

        if not user_content or not assistant_content:
            return None

        # Build messages
        messages = []

        if system_content or self.require_system_message:
            messages.append(
                {"role": "system", "content": system_content or self.default_system_message}
            )

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})

        if self.output_format == "text":
            return {"text": self._messages_to_chatml_text(messages)}
        return {"messages": messages}

    def _messages_to_chatml_text(self, messages: list[dict[str, str]]) -> str:
        """Convert messages list to ChatML text format."""
        chatml_parts = []

        for message in messages:
            role = message["role"]
            content = message["content"]

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

    def validate(self, entry: dict[str, Any]) -> bool:
        """
        Validate that an entry can be formatted for ChatML.

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
