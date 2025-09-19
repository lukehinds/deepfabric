"""
Formatter for Unsloth GRPO training with reasoning.

This formatter combines Unsloth's conversations format with GRPO-style
reasoning tags for training models with chain-of-thought capabilities.
"""

from pydantic import BaseModel, Field

from ..base import BaseFormatter
from ..utils import extract_data


class UnslothGrpoConfig(BaseModel):
    """Configuration for the Unsloth GRPO formatter."""

    reasoning_start_tag: str = Field(
        default="<think>", description="Tag to mark start of reasoning"
    )
    reasoning_end_tag: str = Field(default="</think>", description="Tag to mark end of reasoning")
    include_reasoning_instructions: bool = Field(
        default=True, description="Whether to include reasoning instructions in system message"
    )
    system_message: str | None = Field(
        default=None,
        description="Custom system message (if None, auto-generated based on reasoning tags)",
    )


class UnslothGrpoFormatter(BaseFormatter):
    """
    Formats datasets for Unsloth GRPO training with reasoning.

    This formatter outputs conversations format with embedded reasoning tags,
    suitable for training models that show their thinking process.
    """

    def get_config_model(self):
        """Return the configuration model for this formatter."""
        return UnslothGrpoConfig

    def _format_single_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample to Unsloth GRPO conversations format.

        Args:
            sample: Sample to format

        Returns:
            Formatted sample with conversations key
        """
        config: UnslothGrpoConfig = (
            self._config_model
            if isinstance(self._config_model, UnslothGrpoConfig)
            else UnslothGrpoConfig(**self.config)
        )

        data = extract_data(sample)
        conversations = []

        # Add system message if configured
        if config.include_reasoning_instructions:
            if config.system_message:
                system_msg = config.system_message
            else:
                system_msg = (
                    f"You are a helpful assistant. When solving problems, "
                    f"show your reasoning process between {config.reasoning_start_tag} "
                    f"and {config.reasoning_end_tag} tags, then provide your final answer."
                )
            conversations.append({"role": "system", "content": system_msg})

        # Extract question/answer with reasoning
        question = self._extract_question(data)
        answer = self._extract_answer(data)
        reasoning = self._extract_reasoning(data)

        if question:
            conversations.append({"role": "user", "content": question})

        if answer:
            # Format answer with reasoning if available
            if reasoning:
                formatted_answer = (
                    f"{config.reasoning_start_tag}\n"
                    f"{reasoning}\n"
                    f"{config.reasoning_end_tag}\n\n"
                    f"{answer}"
                )
            else:
                formatted_answer = answer

            conversations.append({"role": "assistant", "content": formatted_answer})

        return {"conversations": conversations}

    def _extract_question(self, data: dict) -> str:
        """Extract question/user input from data."""
        # Check various possible fields
        if "question" in data:
            return data["question"]
        if "instruction" in data:
            instruction = data["instruction"]
            if "input" in data and data["input"]:
                return f"{instruction}\n\nInput: {data['input']}"
            return instruction
        if "user" in data:
            return data["user"]
        if "messages" in data:
            for msg in data["messages"]:
                if msg.get("role") == "user":
                    return msg.get("content", "")
        return ""

    def _extract_answer(self, data: dict) -> str:  # noqa: PLR0911
        """Extract answer/response from data."""
        # Check various possible fields
        if "answer" in data:
            return data["answer"]
        if "final_answer" in data:
            return data["final_answer"]
        if "solution" in data:
            return data["solution"]
        if "output" in data:
            return data["output"]
        if "assistant" in data:
            return data["assistant"]
        if "response" in data:
            return data["response"]
        if "messages" in data:
            for msg in data["messages"]:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Check if reasoning tags are already present
                    if "<think>" in content or "<start_working_out>" in content:
                        # Extract just the final answer part
                        if "</think>" in content:
                            parts = content.split("</think>")
                            if len(parts) > 1:
                                return parts[-1].strip()
                        elif "<end_working_out>" in content:
                            parts = content.split("<end_working_out>")
                            if len(parts) > 1:
                                # Look for solution tags
                                final_part = parts[-1]
                                if "<SOLUTION>" in final_part:
                                    solution_start = final_part.index("<SOLUTION>") + len(
                                        "<SOLUTION>"
                                    )
                                    solution_end = (
                                        final_part.index("</SOLUTION>")
                                        if "</SOLUTION>" in final_part
                                        else len(final_part)
                                    )
                                    return final_part[solution_start:solution_end].strip()
                                return final_part.strip()
                    return content
        return ""

    def _extract_reasoning(self, data: dict) -> str:  # noqa: PLR0911
        """Extract reasoning/chain-of-thought from data."""
        # Check for explicit reasoning fields
        if "reasoning" in data:
            return data["reasoning"]
        if "chain_of_thought" in data:
            return data["chain_of_thought"]
        if "working_out" in data:
            return data["working_out"]
        if "thinking" in data:
            return data["thinking"]
        if "explanation" in data:
            return data["explanation"]

        # Check if reasoning is embedded in messages
        if "messages" in data:
            for msg in data["messages"]:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Extract reasoning from existing tags
                    if "<think>" in content and "</think>" in content:
                        start = content.index("<think>") + len("<think>")
                        end = content.index("</think>")
                        return content[start:end].strip()
                    if "<start_working_out>" in content and "<end_working_out>" in content:
                        start = content.index("<start_working_out>") + len("<start_working_out>")
                        end = content.index("<end_working_out>")
                        return content[start:end].strip()

        return ""

    def validate(self, entry: dict) -> bool:
        """
        Validate that an entry can be formatted.

        Args:
            entry: Entry to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            data = extract_data(entry)
            question = self._extract_question(data)
            answer = self._extract_answer(data)
            return bool(question and answer)
        except Exception:
            return False

    def get_description(self) -> str:
        """Get formatter description."""
        return (
            "Formats datasets for Unsloth GRPO training with reasoning. "
            "Outputs conversations format with embedded reasoning tags for chain-of-thought training."
        )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "conversation", "qa", "instruction", "reasoning", "cot"]
