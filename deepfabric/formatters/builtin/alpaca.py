"""
Alpaca instruction-following format formatter.

This formatter transforms datasets to the Alpaca format used for instruction-following
fine-tuning. The Alpaca format consists of:
- instruction: The task description
- input: Optional context or input data
- output: The expected response

Reference: https://github.com/tatsu-lab/stanford_alpaca
"""

from typing import Any

from pydantic import BaseModel

from ...schemas import Conversation
from ..base import BaseFormatter
from ..models import (
    AlpacaConfig,
    AlpacaOutput,
)


class AlpacaFormatter(BaseFormatter):
    """
    Formatter for Alpaca instruction-following format.

    Transforms DeepFabric datasets to the standardized Alpaca format
    used for instruction-following fine-tuning.
    """

    def __init__(self, config: dict[str, Any] | None = None, tool_registry=None):
        super().__init__(config, tool_registry=tool_registry)

        # Access configuration through typed model if available
        if self._config_model:
            alpaca_config: AlpacaConfig = self._config_model  # type: ignore
            self.instruction_field = alpaca_config.instruction_field
            self.input_field = alpaca_config.input_field
            self.output_field = alpaca_config.output_field
            self.include_empty_input = alpaca_config.include_empty_input
            self.instruction_template = alpaca_config.instruction_template
        else:
            # Fallback to dict-based config for backward compatibility
            self.instruction_field = self.config.get("instruction_field", "instruction")
            self.input_field = self.config.get("input_field", "input")
            self.output_field = self.config.get("output_field", "output")
            self.include_empty_input = self.config.get("include_empty_input", True)
            self.instruction_template = self.config.get("instruction_template", None)

    def _format_single_sample(self, sample: Conversation | dict | Any) -> dict | None:
        """
        Format a single sample to Alpaca format.

        Args:
            sample: A Conversation Pydantic model, dict, or anything with model_dump()

        Returns:
            Formatted sample in Alpaca format or None if formatting fails
        """
        # Handle dict samples that might be Q&A or instruction format
        if isinstance(sample, dict):
            # Check if it's Q&A format (not messages format)
            if "question" in sample and ("answer" in sample or "final_answer" in sample):
                return self._format_qa_sample(sample)

            # Check if it's already in Alpaca instruction format
            if "instruction" in sample and "output" in sample:
                return self._format_instruction_sample(sample)

            # Try to convert to Conversation model for messages format
            try:
                conversation = Conversation(**sample)
            except Exception:
                return None
        elif isinstance(sample, Conversation):
            conversation = sample
        elif hasattr(sample, "model_dump"):
            try:
                conversation = Conversation(**sample.model_dump())
            except Exception:
                return None
        else:
            return None

        # Format using Conversation model
        return self._format_conversation(conversation)

    def _format_conversation(self, conversation: Conversation) -> dict[str, Any]:
        """
        Format a Conversation model to Alpaca format.

        Args:
            conversation: Pydantic Conversation model

        Returns:
            Formatted Alpaca output with instruction, input, and output fields
        """
        instruction = ""
        input_text = ""
        output_text = ""

        # Extract instruction from system message
        system_messages = [msg for msg in conversation.messages if msg.role == "system"]
        if system_messages:
            instruction = system_messages[0].content

        # Extract input from user messages
        user_messages = [msg for msg in conversation.messages if msg.role == "user"]
        if user_messages:
            # If we have a system instruction, use user content as input
            if instruction:
                input_text = " ".join(msg.content for msg in user_messages)
            else:
                # If no system instruction, use first user message as instruction
                instruction = user_messages[0].content
                if len(user_messages) > 1:
                    input_text = " ".join(msg.content for msg in user_messages[1:])

        # Extract output from assistant messages
        assistant_messages = [msg for msg in conversation.messages if msg.role == "assistant"]
        if assistant_messages:
            output_text = " ".join(msg.content for msg in assistant_messages)
        # If no assistant message but has reasoning and final_answer
        elif conversation.reasoning and conversation.final_answer:
            output_parts = []
            content = conversation.reasoning.content

            if isinstance(content, str):
                output_parts.append(content)
            elif isinstance(content, list):
                for step in content:
                    if hasattr(step, "thought") and step.thought:
                        output_parts.append(step.thought)

            if output_parts:
                output_text = (
                    " ".join(output_parts) + f" The answer is {conversation.final_answer}."
                )
            else:
                output_text = conversation.final_answer

        # Apply instruction template if provided
        if self.instruction_template:
            instruction = self.instruction_template.format(instruction=instruction)

        # Build result
        result = {
            self.instruction_field: instruction,
            self.output_field: output_text,
        }

        # Include input field based on configuration
        if self.include_empty_input or input_text:
            result[self.input_field] = input_text

        return result

    def _format_qa_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format a Q&A sample to Alpaca format."""
        question = sample["question"]
        answer = sample.get("answer") or sample.get("final_answer", "")

        # Use question as instruction
        instruction = question

        # Check for additional context that could be input
        input_text = ""
        context_fields = ["context", "passage", "background", "input"]
        for field in context_fields:
            if field in sample and sample[field]:
                input_text = sample[field]
                break

        # If we have chain of thought reasoning, include it in the output
        output_text = answer
        if "chain_of_thought" in sample and sample["chain_of_thought"]:
            output_text = f"{sample['chain_of_thought']}\n\n{answer}"

        # Apply custom instruction template if provided
        if self.instruction_template:
            instruction = self.instruction_template.format(instruction=instruction)

        alpaca_sample = {"instruction": instruction, "output": output_text}

        if input_text or self.include_empty_input:
            alpaca_sample["input"] = input_text

        return alpaca_sample

    def _format_instruction_sample(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Format an instruction sample (already in Alpaca format) applying templates and config."""
        instruction = sample["instruction"]
        output = sample["output"]
        input_text = sample.get("input", "")

        # Apply custom instruction template if provided
        if self.instruction_template:
            instruction = self.instruction_template.format(instruction=instruction)

        alpaca_sample = {
            self.instruction_field: instruction,
            self.output_field: output,
        }

        if input_text or self.include_empty_input:
            alpaca_sample[self.input_field] = input_text

        return alpaca_sample

    def validate(self, entry: dict[str, Any]) -> bool:
        """
        Validate that an entry can be formatted for Alpaca.

        Args:
            entry: Dataset entry to validate

        Returns:
            True if the entry can be formatted, False otherwise
        """
        if not super().validate(entry):
            return False

        # Check for direct Alpaca format
        if self.instruction_field in entry and self.output_field in entry:
            return True

        # Check for messages format
        if "messages" in entry:
            messages = entry["messages"]
            if not isinstance(messages, list):
                return False
            # Should have at least one user or system message and one assistant message
            roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
            return "assistant" in roles and ("user" in roles or "system" in roles)

        # Check for Q&A format
        if "question" in entry and ("answer" in entry or "final_answer" in entry):
            return True

        # Check for any instruction/output pattern
        instruction_fields = ["instruction", "prompt", "question", "task", "query", "problem"]
        output_fields = ["output", "response", "answer", "solution", "result", "final_answer"]

        has_instruction = any(field in entry for field in instruction_fields)
        has_output = any(field in entry for field in output_fields)

        return has_instruction and has_output

    def validate_output(self, entry: dict[str, Any]) -> bool:
        """
        Validate that a formatted entry meets Alpaca requirements.

        Args:
            entry: Formatted entry to validate

        Returns:
            True if the entry meets Alpaca format requirements
        """
        if not isinstance(entry, dict):
            return False

        # Must have instruction and output
        if "instruction" not in entry or "output" not in entry:
            return False

        # Instruction and output must be non-empty strings
        instruction = entry["instruction"]
        output = entry["output"]

        if not isinstance(instruction, str) or not isinstance(output, str):
            return False

        if not instruction.strip() or not output.strip():
            return False

        # Input is optional but must be string if present
        if "input" in entry:
            input_text = entry["input"]
            if not isinstance(input_text, str):
                return False

        return True

    def get_description(self) -> str:
        """Get description of the Alpaca formatter."""
        return """
        Alpaca instruction-following format formatter.

        Transforms datasets to the standard Alpaca format with:
        - instruction: The task description or question
        - input: Optional context or input data
        - output: The expected response or answer

        Supports various input formats including messages, Q&A, and generic
        instruction-response patterns. Configurable field mappings and templates.
        """

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "instruction_output", "question_answer", "generic"]

    def get_config_model(self) -> type[BaseModel]:
        """Get the Pydantic model for Alpaca configuration."""
        return AlpacaConfig

    def get_output_model(self) -> type[BaseModel]:
        """Get the Pydantic model for Alpaca output."""
        return AlpacaOutput
