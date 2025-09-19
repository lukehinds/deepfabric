"""
GRPO (Generalized Reward-based Policy Optimization) formatter.

This formatter transforms datasets for GRPO training, which requires:
- Reasoning sections wrapped in configurable tags
- Solution sections wrapped in configurable tags
- Mathematical problems with extractable numerical answers
- Chat format with system/user/assistant structure

Based on the Qwen3-4B GRPO training pipeline format requirements.
"""

import re

from pydantic import BaseModel

from ..base import BaseFormatter, FormatterError
from ..models import (
    DatasetInput,
    DatasetOutput,
    FormattedOutput,
    GenericSample,
    GrpoConfig,
    GrpoOutput,
)

PARTS_LENGTH = 3  # Minimum parts length when splitting on answer/solution keywords


class GrpoFormatter(BaseFormatter):
    """
    Formatter for GRPO (Generalized Reward-based Policy Optimization) training.

    Transforms DeepFabric datasets to the format required by GRPO training,
    with reasoning and solution tags for mathematical reasoning tasks.
    """

    def __init__(self, config: dict | None = None):
        super().__init__(config)

        # Access configuration through typed model if available
        if self._config_model:
            grpo_config: GrpoConfig = GrpoConfig.model_validate(self._config_model)
            self.reasoning_start_tag = grpo_config.reasoning_start_tag
            self.reasoning_end_tag = grpo_config.reasoning_end_tag
            self.solution_start_tag = grpo_config.solution_start_tag
            self.solution_end_tag = grpo_config.solution_end_tag
            self.system_prompt = grpo_config.system_prompt or self._get_default_system_prompt()
            self.validate_numerical = grpo_config.validate_numerical
        else:
            # Fallback to dict-based config for backward compatibility
            self.reasoning_start_tag = self.config.get("reasoning_start_tag", "<start_working_out>")
            self.reasoning_end_tag = self.config.get("reasoning_end_tag", "<end_working_out>")
            self.solution_start_tag = self.config.get("solution_start_tag", "<SOLUTION>")
            self.solution_end_tag = self.config.get("solution_end_tag", "</SOLUTION>")
            self.system_prompt = self.config.get("system_prompt", self._get_default_system_prompt())
            self.validate_numerical = self.config.get("validate_numerical", True)

        # Compile regex for answer extraction if validation is enabled
        if self.validate_numerical:
            self._compile_answer_regex()

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for GRPO reasoning."""
        return f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {self.reasoning_start_tag} and {self.reasoning_end_tag}.
Then, provide your solution between {self.solution_start_tag} and {self.solution_end_tag}."""

    def _compile_answer_regex(self):
        """Compile regex patterns for answer extraction and validation."""
        # Regex to match the complete GRPO format
        self.format_regex = re.compile(
            rf"{re.escape(self.reasoning_end_tag)}.*?"
            rf"{re.escape(self.solution_start_tag)}(.+?){re.escape(self.solution_end_tag)}"
            rf"[\s]*$",
            flags=re.MULTILINE | re.DOTALL,
        )

        # Regex to extract numerical answers
        self.number_regex = re.compile(
            rf"{re.escape(self.solution_start_tag)}.*?[\s]*([+-]?[\d\.,]+)",
            flags=re.MULTILINE | re.DOTALL,
        )

    def format(self, dataset: DatasetInput | list) -> DatasetOutput:
        """
        Transform dataset to GRPO format.

        Args:
            dataset: Input dataset (DatasetInput model or list of samples)

        Returns:
            DatasetOutput containing GRPO-formatted samples

        Raises:
            FormatterError: If formatting fails
        """
        # Convert to DatasetInput if it's a list
        if isinstance(dataset, list):
            dataset_samples = []
            for sample in dataset:
                if isinstance(sample, dict):
                    dataset_samples.append(GenericSample(data=sample))
                else:
                    dataset_samples.append(sample)
            dataset_input = DatasetInput(samples=dataset_samples)
        else:
            dataset_input = dataset

        formatted_samples = []

        for i, sample in enumerate(dataset_input.samples):
            try:
                # Convert Pydantic model to dict
                sample_dict = getattr(sample, "data", None) or sample.model_dump()
                formatted_sample = self._format_sample(sample_dict)
                if formatted_sample:
                    formatted_samples.append(FormattedOutput(**formatted_sample))
            except Exception as e:
                raise FormatterError(f"Failed to format sample {i}: {str(e)}") from e

        return DatasetOutput(samples=formatted_samples)

    def _format_sample(self, sample: dict) -> dict | None:
        """
        Format a single sample for GRPO training.

        Args:
            sample: A single dataset sample

        Returns:
            Formatted sample or None if formatting fails
        """
        if not self.validate(sample):
            return None

        # Handle different input formats
        if "messages" in sample:
            return self._format_messages_sample(sample)
        if "question" in sample and "final_answer" in sample:
            return self._format_qa_sample(sample)
        # Try to extract problem and solution from any available fields
        return self._format_generic_sample(sample)

    def _format_messages_sample(self, sample: dict) -> dict:
        """Format a sample that already has a messages structure."""
        messages = sample["messages"].copy()

        # Find the assistant message and wrap it in GRPO format
        for message in messages:
            if message["role"] == "assistant":
                content = message["content"]

                # If not already in GRPO format, wrap it
                if not self._is_grpo_formatted(content):
                    # Try to extract reasoning and answer
                    reasoning, answer = self._extract_reasoning_and_answer(content)
                    if reasoning and answer:
                        formatted_content = (
                            f"{self.reasoning_start_tag}{reasoning}{self.reasoning_end_tag}"
                            f"{self.solution_start_tag}{answer}{self.solution_end_tag}"
                        )
                        message["content"] = formatted_content

        # Ensure system prompt is appropriate for GRPO
        has_system = any(msg["role"] == "system" for msg in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": self.system_prompt})

        return {"messages": messages}

    def _format_qa_sample(self, sample: dict) -> dict:
        """Format a Q&A sample to GRPO messages format."""
        question = sample["question"]
        answer = sample["final_answer"]

        # Get reasoning if available
        reasoning = ""
        if "chain_of_thought" in sample:
            reasoning = sample["chain_of_thought"]
        elif "reasoning_trace" in sample:
            reasoning = sample["reasoning_trace"]

        # Format the assistant response
        if reasoning:
            assistant_content = (
                f"{self.reasoning_start_tag}{reasoning}{self.reasoning_end_tag}"
                f"{self.solution_start_tag}{answer}{self.solution_end_tag}"
            )
        else:
            # Generate basic reasoning wrapper
            assistant_content = (
                f"{self.reasoning_start_tag}Let me solve this step by step.{self.reasoning_end_tag}"
                f"{self.solution_start_tag}{answer}{self.solution_end_tag}"
            )

        return {
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content},
            ]
        }

    def _format_generic_sample(self, sample: dict) -> dict | None:
        """Try to format any sample by extracting question/answer patterns."""
        # Look for common field names that might contain questions
        question_fields = ["question", "prompt", "problem", "input", "instruction"]
        answer_fields = ["answer", "output", "response", "solution", "final_answer"]

        question = None
        answer = None

        for field in question_fields:
            if field in sample and sample[field]:
                question = sample[field]
                break

        for field in answer_fields:
            if field in sample and sample[field]:
                answer = sample[field]
                break

        if question and answer:
            return self._format_qa_sample(
                {
                    "question": question,
                    "final_answer": answer,
                    "chain_of_thought": sample.get("reasoning", ""),
                }
            )

        return None

    def _extract_reasoning_and_answer(self, content: str) -> tuple[str, str]:
        """
        Extract reasoning and final answer from assistant content.

        Args:
            content: Assistant message content

        Returns:
            Tuple of (reasoning, answer)
        """
        # Try to split on common patterns
        if "answer:" in content.lower() or "solution:" in content.lower():
            parts = re.split(r"(?i)(answer|solution):\s*", content, maxsplit=1)
            if len(parts) >= PARTS_LENGTH:
                reasoning = parts[0].strip()
                answer = parts[2].strip()
                return reasoning, answer

        # Fallback: treat entire content as reasoning, try to extract numerical answer
        reasoning = content.strip()

        # Look for numbers at the end
        number_match = re.search(r"([+-]?[\d\.,]+)\s*$", content)
        if number_match:
            answer = number_match.group(1)
            # Remove the number from reasoning
            reasoning = content[: number_match.start()].strip()
            return reasoning, answer

        # If no clear numerical answer, use last sentence
        sentences = content.split(".")
        if len(sentences) > 1:
            answer = sentences[-1].strip()
            reasoning = ".".join(sentences[:-1]).strip()
            return reasoning, answer

        return reasoning, content.strip()

    def _is_grpo_formatted(self, content: str) -> bool:
        """Check if content is already in GRPO format."""
        return (
            self.reasoning_start_tag in content
            and self.reasoning_end_tag in content
            and self.solution_start_tag in content
            and self.solution_end_tag in content
        )

    def validate(self, entry: dict) -> bool:
        """
        Validate that an entry can be formatted for GRPO.

        Args:
            entry: Dataset entry to validate

        Returns:
            True if the entry can be formatted, False otherwise
        """
        if not super().validate(entry):
            return False

        # Check for supported formats
        if "messages" in entry:
            messages = entry["messages"]
            if not isinstance(messages, list):
                return False

            # Should have at least user and assistant messages
            roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
            return "user" in roles and "assistant" in roles

        # Check for Q&A format
        if "question" in entry and ("final_answer" in entry or "answer" in entry):
            return True

        # Check for any question/answer pattern
        question_fields = ["question", "prompt", "problem", "input", "instruction"]
        answer_fields = ["answer", "output", "response", "solution", "final_answer"]

        has_question = any(field in entry for field in question_fields)
        has_answer = any(field in entry for field in answer_fields)

        return has_question and has_answer

    def validate_output(self, entry: dict) -> bool:  # noqa: PLR0911
        """
        Validate that a formatted entry meets GRPO requirements.

        Args:
            entry: Formatted entry to validate

        Returns:
            True if the entry meets GRPO format requirements
        """
        if "messages" not in entry:
            return False

        messages = entry["messages"]
        if not isinstance(messages, list):
            return False

        # Should have system, user, and assistant messages
        roles = [msg.get("role") for msg in messages]
        if not all(role in roles for role in ["system", "user", "assistant"]):
            return False

        # Find assistant message and validate GRPO format
        assistant_content = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_content = msg.get("content", "")
                break

        if not assistant_content:
            return False

        # Check for GRPO formatting tags
        if not self._is_grpo_formatted(assistant_content):
            return False

        # If numerical validation is enabled, check for extractable answers
        if self.validate_numerical:
            return self._validate_numerical_answer(assistant_content)

        return True

    def _validate_numerical_answer(self, content: str) -> bool:
        """Validate that the content contains an extractable numerical answer."""
        if not hasattr(self, "number_regex"):
            return True  # Skip validation if regex not compiled

        return self.number_regex.search(content) is not None

    def get_description(self) -> str:
        """Get description of the GRPO formatter."""
        return """
        GRPO (Generalized Reward-based Policy Optimization) formatter.

        Transforms datasets for mathematical reasoning training with:
        - Reasoning sections wrapped in working_out tags
        - Solution sections wrapped in SOLUTION tags
        - Chat format with system/user/assistant structure
        - Numerical answer extraction for reward functions

        Supports configuration of all formatting tags and validation options.
        """

    def get_supported_formats(self) -> list[str]:
        """Get list of supported input formats."""
        return ["messages", "question_answer", "chain_of_thought", "generic"]

    def get_config_model(self) -> type[BaseModel]:
        """Get the Pydantic model for GRPO configuration."""
        return GrpoConfig

    def get_output_model(self) -> type[BaseModel]:
        """Get the Pydantic model for GRPO output."""
        return GrpoOutput
