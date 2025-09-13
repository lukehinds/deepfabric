import json
import math
import random
import re

from typing import TYPE_CHECKING, Any

import litellm

from pydantic import BaseModel, Field, field_validator
from tqdm.rich import tqdm

from .constants import (
    API_ERROR_INDICATORS,
    DEFAULT_MAX_RETRIES,
    DEFAULT_REQUEST_TIMEOUT,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    ERROR_CATEGORIES,
    ERROR_DATASET_FILENAME,
    INTERRUPTED_DATASET_FILENAME,
)
from .dataset import Dataset
from .exceptions import (
    DataSetGeneratorError,
)
from .prompts import ENGINE_JSON_INSTRUCTIONS, SAMPLE_GENERATION_PROMPT
from .topic_model import TopicModel
from .tui import get_tui, get_dataset_tui


# Handle circular import for type hints
if TYPE_CHECKING:
    from .topic_model import TopicModel


def validate_json_response(json_str: str, schema: dict[str, Any] | None = None) -> dict | None:
    """Validate and clean JSON response from LLM."""
    try:
        json_match = re.search(r"(?s)\{.*\}", json_str)
        if not json_match:
            return None

        cleaned_json = json_match.group(0)
        cleaned_json = re.sub(r"```json\s*|\s*```", "", cleaned_json)

        parsed = json.loads(cleaned_json)

        if schema is not None:
            # Schema validation could be added here
            pass
        else:
            return parsed
    except (json.JSONDecodeError, ValueError):
        return None


class DataSetGeneratorArguments(BaseModel):
    """Arguments for configuring the data engine."""

    instructions: str = Field("", description="Additional instructions for data generation")
    system_prompt: str = Field(..., min_length=1, description="System prompt for the model")
    model_name: str = Field(..., min_length=1, description="Name of the model to use")
    prompt_template: str | None = Field(None, description="Custom prompt template")
    example_data: Dataset | None = Field(None, description="Example dataset for few-shot learning")
    temperature: float = Field(
        ENGINE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_retries: int = Field(
        DEFAULT_MAX_RETRIES,
        ge=1,
        le=10,
        description="Maximum number of retries for failed requests",
    )
    default_batch_size: int = Field(
        ENGINE_DEFAULT_BATCH_SIZE,
        ge=1,
        le=100,
        description="Default batch size for generation",
    )
    default_num_examples: int = Field(
        ENGINE_DEFAULT_NUM_EXAMPLES,
        ge=0,
        le=10,
        description="Default number of examples to include",
    )
    request_timeout: int = Field(
        DEFAULT_REQUEST_TIMEOUT, ge=5, le=300, description="Request timeout in seconds"
    )
    sys_msg: bool = Field(True, description="Whether to include system message in dataset")

    class Config:
        arbitrary_types_allowed = True  # Allow Dataset type

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v):
        if not v or not v.strip():
            raise ValueError("required")
        return v.strip()

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("required")
        return v.strip()


class DataSetGenerator:
    def __init__(self, args: DataSetGeneratorArguments):
        """Initialize DataSetGenerator with validated arguments."""
        if not isinstance(args, DataSetGeneratorArguments):
            try:
                # Handle dict input for backward compatibility
                if isinstance(args, dict):
                    args = DataSetGeneratorArguments(**args)
                else:
                    raise DataSetGeneratorError("invalid")  # noqa: TRY301
            except Exception as e:
                raise DataSetGeneratorError("invalid") from e

        self.args = args
        self.model_name = args.model_name
        self.dataset = Dataset()
        self.failed_samples = []
        self.failure_analysis = {category: [] for category in ERROR_CATEGORIES}

        # Store original system prompt for dataset inclusion
        self.original_system_prompt = args.system_prompt
        # Use ENGINE_JSON_INSTRUCTIONS only for generation prompt
        self.generation_system_prompt = ENGINE_JSON_INSTRUCTIONS + args.system_prompt

    def _validate_create_data_params(
        self,
        num_steps: int,
        batch_size: int,
        topic_model: "TopicModel | None" = None,
    ) -> None:
        """Validate parameters for data creation."""
        if num_steps is None or num_steps <= 0:
            raise DataSetGeneratorError("positive")

        if batch_size <= 0:
            raise DataSetGeneratorError("positive")

        if topic_model and len(topic_model.get_all_paths()) == 0:
            raise DataSetGeneratorError("")

    def _prepare_topic_paths(
        self,
        num_steps: int,
        batch_size: int,
        topic_model: "TopicModel | None" = None,
    ) -> tuple[list | None, int]:
        """Prepare and validate topic paths for data generation."""
        topic_paths = None
        if topic_model is not None:
            topic_paths = topic_model.get_all_paths()
            total_paths = len(topic_paths)
            required_samples = num_steps * batch_size

            if required_samples > total_paths:
                raise DataSetGeneratorError("insufficient")
            # Bandit: not a security function
            topic_paths = random.sample(topic_paths, required_samples)  # nosec
            num_steps = math.ceil(len(topic_paths) / batch_size)

        return topic_paths, num_steps

    def _generate_batch_prompts(
        self,
        batch_size: int,
        start_idx: int,
        topic_paths: list,
        data_creation_prompt: str,
        num_example_demonstrations: int,
    ) -> list[str]:
        """Generate prompts for a batch."""
        prompts = []
        for i in range(batch_size):
            path = None
            if topic_paths:
                current_idx = start_idx + i
                if current_idx < len(topic_paths):
                    path = topic_paths[current_idx]
                else:
                    break

            sample_prompt = self.build_prompt(
                data_creation_prompt=data_creation_prompt,
                num_example_demonstrations=num_example_demonstrations,
                subtopics_list=path,
            )
            prompts.append(sample_prompt)
        return prompts

    def _process_batch_responses(  # noqa: PLR0912
        self,
        responses: list,
        include_sys_msg: bool,
    ) -> tuple[list, list]:
        """Process batch responses and return samples and failed responses."""
        samples = []
        failed_responses = []

        for response in responses:
            try:
                # Extract content from the response - use getattr to safely access attributes
                response_content = None

                # Try standard OpenAI format first
                choices = getattr(response, "choices", None)
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message = getattr(choice, "message", None)
                    if message:
                        response_content = getattr(message, "content", None)
                    if not response_content:
                        response_content = getattr(choice, "text", None)

                # Try direct content access if no content found yet
                if not response_content:
                    response_content = getattr(response, "content", None)

                if not response_content:
                    response_content = getattr(response, "text", None)

                # Try getting content from response as dict if still no content
                if not response_content and isinstance(response, dict):
                    try:
                        if "choices" in response and response["choices"]:
                            choice = response["choices"][0]
                            if isinstance(choice, dict):
                                if "message" in choice and "content" in choice["message"]:
                                    response_content = choice["message"]["content"]
                                elif "text" in choice:
                                    response_content = choice["text"]
                        elif "content" in response:
                            response_content = response["content"]
                    except (KeyError, IndexError, TypeError):
                        pass

                if not response_content:
                    failed_responses.append("empty model response")
                    self.failure_analysis["empty_responses"].append("empty model response")
                    continue

                parsed_json = validate_json_response(response_content)

                if parsed_json and include_sys_msg and "messages" in parsed_json:
                    # Add system message at the start if sys_msg is True
                    parsed_json["messages"].insert(
                        0,
                        {
                            "role": "system",
                            "content": self.original_system_prompt,
                        },
                    )

                if parsed_json:
                    samples.append(parsed_json)
                else:
                    failed_responses.append(response_content)
                    failure_type = self.analyze_failure(response_content)
                    self.failure_analysis[failure_type].append(response_content)

            except Exception as e:
                failed_responses.append(str(e))
                failure_type = self.analyze_failure(str(e), error=e)
                self.failure_analysis[failure_type].append(str(e))

        return samples, failed_responses

    def _handle_batch_completion(self, prompts: list[str]) -> list:
        """Handle batch completion with the LLM."""
        completion_args = {
            "model": self.model_name,
            "messages": [[{"role": "user", "content": p}] for p in prompts],
            "temperature": self.args.temperature,
            "stream": False,  # Ensure we get complete responses
        }
        return litellm.batch_completion(**completion_args)

    def analyze_failure(self, response_content: str, error: Exception | None = None) -> str:
        """Analyze the failure reason for a sample."""
        if error:
            error_str = str(error)
            if "schema" in error_str.lower():
                return "invalid_schema"
            if any(api_err in error_str.lower() for api_err in API_ERROR_INDICATORS):
                return "api_errors"
            return "other_errors"

        if not response_content or response_content.isspace():
            return "empty_responses"

        # Check if response seems to be attempting JSON but failing
        if any(char in response_content for char in "{}[]"):
            return "json_parsing_errors"
        return "malformed_responses"

    def summarize_failures(self) -> dict:
        """Generate a summary of all failures."""
        summary = {
            "total_failures": len(self.failed_samples),
            "failure_types": {k: len(v) for k, v in self.failure_analysis.items()},
            "failure_examples": {},
        }

        # Add example failures for each category
        for _category, failures in self.failure_analysis.items():
            if failures:
                # Get up to 3 examples for each category
                examples = failures[:3]
                summary["failure_examples"] = [
                    (
                        str(ex)[:200] + "..."
                        if len(str(ex)) > 200  # noqa: PLR2004
                        else str(ex)  # noqa: PLR2004
                    )  # noqa: PLR2004
                    for ex in examples
                ]
        return summary

    def create_data(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
    ):
        # Set default value for num_steps if None
        if num_steps is None:
            num_steps = 1

        # Validate inputs
        self._validate_create_data_params(num_steps, batch_size, topic_model)

        # Use instance model_name as fallback if none provided
        if model_name:
            self.model_name = model_name.strip()

        if not self.model_name:
            raise DataSetGeneratorError("")

        # Use provided sys_msg or fall back to args.sys_msg
        include_sys_msg = sys_msg if sys_msg is not None else self.args.sys_msg

        # Prepare topic paths and adjust num_steps if necessary
        topic_paths, num_steps = self._prepare_topic_paths(num_steps, batch_size, topic_model)

        total_samples = num_steps * batch_size

        # Initialize dataset generation TUI
        dataset_tui = get_dataset_tui()
        dataset_tui.show_generation_header(self.model_name, num_steps, batch_size)

        # Enable JSON schema validation
        litellm.enable_json_schema_validation = True

        data_creation_prompt = SAMPLE_GENERATION_PROMPT

        return self._run_generation_loop(
            num_steps=num_steps,
            batch_size=batch_size,
            total_samples=total_samples,
            topic_paths=topic_paths or [],
            data_creation_prompt=data_creation_prompt,
            num_example_demonstrations=num_example_demonstrations,
            include_sys_msg=include_sys_msg,
            dataset_tui=dataset_tui,
        )

    def _run_generation_loop(  # noqa: PLR0912
        self,
        num_steps: int,
        batch_size: int,
        total_samples: int,
        topic_paths: list,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
        dataset_tui=None,
    ):
        """Run the main generation loop."""
        try:
            # Create rich progress bar
            progress = dataset_tui.create_rich_progress() if dataset_tui else None
            if progress:
                progress.start()
                progress_task = progress.add_task("Generating dataset samples", total=total_samples)
            else:
                progress_task = None

            # Fallback to regular tqdm if no TUI
            pbar = None if progress else tqdm(total=total_samples, desc="Progress")

            try:
                for step in range(num_steps):
                    start_idx = step * batch_size
                    prompts = self._generate_batch_prompts(
                        batch_size,
                        start_idx,
                        topic_paths,
                        data_creation_prompt,
                        num_example_demonstrations,
                    )

                    success = self._process_batch_with_retries(
                        prompts, include_sys_msg, progress, progress_task, pbar
                    )
                    if not success:
                        if dataset_tui:
                            dataset_tui.tui.warning(
                                f"Failed to process batch {step + 1} after all retries"
                            )
                        else:
                            print(f"Failed to process batch {step + 1} after all retries")

            finally:
                if progress:
                    progress.stop()
                elif pbar:
                    pbar.close()

        except KeyboardInterrupt:
            if dataset_tui:
                dataset_tui.tui.warning("Generation interrupted by user.")
            else:
                print("\nGeneration interrupted by user.")
            self.print_failure_summary()
            self.save_dataset(INTERRUPTED_DATASET_FILENAME)
            return self.dataset

        except Exception as e:
            if dataset_tui:
                dataset_tui.tui.error(f"Unexpected error: {str(e)}")
            else:
                print(f"\nUnexpected error: {str(e)}")
            self.print_failure_summary()
            self.save_dataset(ERROR_DATASET_FILENAME)
            raise DataSetGeneratorError("failed") from e

        if dataset_tui:
            dataset_tui.tui.success(f"Successfully Generated {len(self.dataset)} samples.")
        else:
            print(f"Successfully Generated {len(self.dataset)} samples.")
        self.print_failure_summary()
        return self.dataset

    def _process_batch_with_retries(
        self,
        prompts: list[str],
        include_sys_msg: bool,
        progress=None,
        progress_task=None,
        pbar=None,
    ) -> bool:
        """Process a batch with retry logic."""
        for attempt in range(self.args.max_retries):
            try:
                responses = self._handle_batch_completion(prompts)
                samples, failed_responses = self._process_batch_responses(
                    responses, include_sys_msg
                )

                # Update failed samples
                self.failed_samples.extend(failed_responses)

                if samples:
                    failed_samples, failure_descriptions = self.dataset.add_samples(samples)
                    if failed_samples:
                        for sample, desc in zip(failed_samples, failure_descriptions, strict=True):
                            self.failed_samples.append(sample)
                            self.failure_analysis["invalid_schema"].append(desc)

                    successful_samples = len(samples) - len(failed_samples)

                    # Update progress
                    if progress and progress_task is not None:
                        progress.advance(progress_task, successful_samples)
                    elif pbar:
                        pbar.update(successful_samples)

                    return True  # Success - exit retry loop

            except litellm.AuthenticationError as e:
                # Handle authentication errors specifically - don't retry, they won't succeed
                provider = self.model_name.split("/")[0] if "/" in self.model_name else "unknown"
                error_msg = f"Authentication failed for provider '{provider}'. Please set the required API key environment variable."
                self.failed_samples.append(error_msg)
                self.failure_analysis["authentication_error"].append(error_msg)

                # Use TUI to display error if available
                try:
                    tui = get_tui()
                    tui.error(error_msg)
                except Exception:
                    print(f"Authentication Error: {error_msg}")

                return False  # Don't retry authentication errors
            except Exception as e:
                if attempt == self.args.max_retries - 1:
                    # Don't print here, let TUI or calling code handle final messaging
                    self.failed_samples.append(str(e))
                    failure_type = self.analyze_failure(str(e), error=e)
                    self.failure_analysis[failure_type].append(str(e))
                    return False
                # Don't print retry messages to avoid clutter with TUI

        return False

    def print_failure_summary(self):
        """Print a detailed summary of all failures."""
        summary = self.summarize_failures()

        print("\n=== Failure Analysis Summary ===")
        print(f"Total Failed Samples: {summary['total_failures']}")
        print("\nFailure Types Breakdown:")
        for failure_type, count in summary["failure_types"].items():
            if count > 0:
                print(f"\n{failure_type.replace('_', ' ').title()}: {count}")
                if failure_type in summary["failure_examples"]:
                    print("Example failures:")
                    for i, example in enumerate(summary["failure_examples"][failure_type], 1):
                        print(f"  {i}. {example}")
        print("\n=============================")

    def build_prompt(
        self,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        subtopics_list: list[str] | None = None,
    ) -> str:
        prompt = data_creation_prompt.replace(
            "{{{{system_prompt}}}}", self.generation_system_prompt
        )
        prompt = prompt.replace("{{{{instructions}}}}", self.build_custom_instructions_text())
        prompt = prompt.replace(
            "{{{{examples}}}}", self.build_examples_text(num_example_demonstrations)
        )
        return prompt.replace("{{{{subtopics}}}}", self.build_subtopics_text(subtopics_list))

    def build_system_prompt(self):
        """Return the original system prompt for dataset inclusion."""
        return self.original_system_prompt

    def build_custom_instructions_text(self) -> str:
        if self.args.instructions is None:
            return ""
        return f"\nHere are additional instructions:\n<instructions>\n{self.args.instructions}\n</instructions>\n"

    def build_examples_text(self, num_example_demonstrations: int):
        if self.args.example_data is None or num_example_demonstrations == 0:
            return ""
        # Bandit: not a security function
        examples = random.sample(self.args.example_data.samples, num_example_demonstrations)  # nosec
        examples_text = "Here are output examples:\n\n"
        examples_text += "\n".join(f"Example {i + 1}: \n\n{ex}\n" for i, ex in enumerate(examples))
        return f"\nHere are output examples:\n<examples>\n{examples_text}\n</examples>\n"

    def build_subtopics_text(self, subtopic_list: list[str] | None):
        if subtopic_list is None:
            return ""
        return f"\nLastly, the topic of the training data should be related to the following subtopics: {' -> '.join(subtopic_list)}"

    def save_dataset(self, save_path: str):
        """Save the dataset to a file."""
        self.dataset.save(save_path)
