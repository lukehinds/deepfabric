import asyncio
import logging
import math
import random

from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

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
from .exceptions import DataSetGeneratorError
from .llm import LLMClient
from .metrics import trace
from .prompts import (
    AGENT_COT_HYBRID_PROMPT,
    AGENT_COT_MULTI_TURN_PROMPT,
    AGENT_COT_TOOLS_PROMPT,
    CONVERSATION_GENERATION_PROMPT,
    FREETEXT_COT_MATHEMATICAL_PROMPT,
    FREETEXT_COT_PROMPT,
    HYBRID_COT_MATHEMATICAL_PROMPT,
    HYBRID_COT_PROMPT,
    STRUCTURED_COT_MATHEMATICAL_PROMPT,
    STRUCTURED_COT_PROMPT,
    XLAM_MULTI_TURN_PROMPT,
    AgentPromptBuilder,
)
from .schemas import ToolRegistry, get_conversation_schema
from .tools.loader import get_available_tools, load_tools_from_dict, load_tools_from_file
from .topic_model import TopicModel
from .utils import ensure_not_running_loop

# Handle circular import for type hints
if TYPE_CHECKING:
    from .topic_model import TopicModel

logger = logging.getLogger(__name__)


class DataSetGeneratorConfig(BaseModel):
    """Configuration for the data engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    instructions: str = Field(default="", description="Additional instructions for data generation")
    generation_system_prompt: str = Field(
        ..., min_length=1, description="System prompt for content generation"
    )
    dataset_system_prompt: str | None = Field(
        None,
        description="System prompt that goes into the final dataset (falls back to generation_system_prompt if not provided)",
    )
    provider: str = Field(
        ..., min_length=1, description="LLM provider (openai, anthropic, gemini, ollama)"
    )
    model_name: str = Field(..., min_length=1, description="Name of the model to use")
    prompt_template: str | None = Field(default=None, description="Custom prompt template")
    example_data: Dataset | None = Field(
        default=None, description="Example dataset for few-shot learning"
    )
    temperature: float = Field(
        default=ENGINE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=1,
        le=10,
        description="Maximum number of retries for failed requests (deprecated, use rate_limit config)",
    )
    max_tokens: int = Field(
        default=2000, ge=1, description="Maximum tokens to generate in a single call to the llm"
    )
    default_batch_size: int = Field(
        default=ENGINE_DEFAULT_BATCH_SIZE,
        ge=1,
        le=100,
        description="Default batch size for generation",
    )
    default_num_examples: int = Field(
        default=ENGINE_DEFAULT_NUM_EXAMPLES,
        ge=0,
        le=10,
        description="Default number of examples to include",
    )
    request_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT, ge=5, le=300, description="Request timeout in seconds"
    )
    sys_msg: bool = Field(default=True, description="Whether to include system message in dataset")

    # Rate limiting configuration
    rate_limit: dict[str, int | float | str | bool] | None = Field(
        default=None,
        description="Rate limiting and retry configuration (uses provider defaults if not specified)",
    )

    # Chain of Thought parameters
    conversation_type: Literal[
        "basic",
        "structured",
        "tool_calling",
        "cot_freetext",
        "cot_structured",
        "cot_hybrid",
        "agent_cot_tools",
        "agent_cot_hybrid",
        "agent_cot_multi_turn",
        "xlam_multi_turn",
    ] = Field(default="basic", description="Type of conversation to generate")
    reasoning_style: Literal["mathematical", "logical", "general"] = Field(
        default="general", description="Style of reasoning for CoT generation"
    )

    # Agent-specific parameters
    available_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names available to the agent (empty means all tools)",
    )
    custom_tools: list[dict] = Field(
        default_factory=list, description="Custom tool definitions as dictionaries"
    )
    max_tools_per_query: int = Field(
        default=3, ge=1, le=10, description="Maximum number of tools an agent can use per query"
    )
    tool_registry_path: str | None = Field(
        default=None, description="Path to custom tool definitions file (JSON/YAML)"
    )
    domain_context: str | None = Field(
        default=None,
        description="Domain/scenario context for XLAM multi-turn (becomes 'system' field)",
    )


class DataSetGenerator:
    def __init__(self, **kwargs):
        """Initialize DataSetGenerator with parameters."""
        try:
            self.config = DataSetGeneratorConfig.model_validate(kwargs)
        except Exception as e:  # noqa: TRY003
            raise DataSetGeneratorError(f"Invalid generator configuration: {str(e)}") from e

        # Initialize from config
        self.provider = self.config.provider
        self.model_name = self.config.model_name
        self.dataset = Dataset()
        self.failed_samples = []
        self.failure_analysis = {category: [] for category in ERROR_CATEGORIES}

        # Initialize LLM client with rate limiting configuration
        self.llm_client = LLMClient(
            provider=self.provider,
            model_name=self.model_name,
            rate_limit_config=self.config.rate_limit,  # Pass rate limit config (can be None)
        )
        trace(
            "generator_created",
            {
                "provider": self.provider,
                "model_name": self.model_name,
                "conversation_type": self.config.conversation_type,
            },
        )

        # Store dataset system prompt for dataset inclusion (with fallback)
        self.dataset_system_prompt = (
            self.config.dataset_system_prompt or self.config.generation_system_prompt
        )
        # Store generation prompt for content generation
        self.generation_prompt = self.config.generation_system_prompt

        # Initialize tool registry for agent conversation types
        self.tool_registry = None
        if self.config.conversation_type in {
            "agent_cot_tools",
            "agent_cot_hybrid",
            "agent_cot_multi_turn",
            "xlam_multi_turn",
        }:
            self._initialize_tool_registry()

    def _initialize_tool_registry(self):
        """Initialize tool registry from configuration."""
        try:
            custom_registry = None

            # Load tools from file if specified
            if self.config.tool_registry_path:
                custom_registry = load_tools_from_file(self.config.tool_registry_path)

            # Load custom tools from dict and merge with file-based tools if both exist
            if self.config.custom_tools:
                dict_registry = load_tools_from_dict(self.config.custom_tools)
                if custom_registry:
                    # Merge both registries - tools from dict take precedence
                    # Create a new registry with combined tools
                    combined_tools = custom_registry.tools + dict_registry.tools
                    custom_registry = ToolRegistry(tools=combined_tools)
                else:
                    custom_registry = dict_registry

            # Get available tools based on configuration
            self.tool_registry = get_available_tools(
                available_tool_names=self.config.available_tools or None,
                custom_registry=custom_registry,
            )

        except Exception as e:  # noqa: BLE001
            raise DataSetGeneratorError(f"Failed to initialize tool registry: {str(e)}") from e

    def _validate_create_data_params(
        self,
        num_steps: int,
        batch_size: int,
        topic_model: "TopicModel | None" = None,
    ) -> None:
        """Validate parameters for data creation."""
        if num_steps is None or num_steps <= 0:
            raise DataSetGeneratorError("num_steps must be a positive integer")

        if batch_size <= 0:
            raise DataSetGeneratorError("batch_size must be a positive integer")

        if topic_model and len(topic_model.get_all_paths()) == 0:
            raise DataSetGeneratorError(
                "Topic model has no paths. Ensure the topic tree was built successfully."
            )

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
                # Provide detailed error with recommendations
                max_steps_for_batch = total_paths // batch_size
                max_batch_for_steps = total_paths // num_steps if num_steps > 0 else total_paths

                error_msg = (
                    f"Insufficient topic paths for dataset generation:\n"
                    f"  • Available paths: {total_paths}\n"
                    f"  • Requested samples: {required_samples} ({num_steps} steps × {batch_size} batch size)\n"
                    f"  • Shortfall: {required_samples - total_paths} samples\n\n"
                    f"Recommendations:\n"
                    f"  • Reduce --num-steps to {max_steps_for_batch} (with current batch size {batch_size})\n"
                    f"  • Reduce --batch-size to {max_batch_for_steps} (with current {num_steps} steps)\n"
                    f"  • Increase topic tree/graph depth or degree to generate more paths"
                )
                raise DataSetGeneratorError(error_msg)

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

    async def _generate_structured_samples_async(
        self,
        prompts: list[str],
        include_sys_msg: bool,
    ) -> tuple[list, list]:
        """Generate structured samples using Outlines with appropriate schema."""
        samples = []
        failed_responses = []

        schema_class = get_conversation_schema(
            self.config.conversation_type, self.config.reasoning_style
        )

        async def _generate(prompt: str) -> tuple[bool, dict | Exception]:
            try:
                # Use higher max_tokens for complex conversation types to prevent truncation
                max_tokens = self.config.max_tokens

                # Structured CoT types need more tokens for reasoning traces + messages
                if self.config.conversation_type in {
                    "cot_structured",
                    "cot_hybrid",
                    "agent_cot_hybrid",
                    "agent_cot_multi_turn",
                    "xlam_multi_turn",
                }:
                    max_tokens = max(max_tokens, 4000)

                conversation = await self.llm_client.generate_async(
                    prompt=prompt,
                    schema=schema_class,
                    max_retries=self.config.max_retries,
                    max_tokens=max_tokens,
                    temperature=self.config.temperature,
                )

                sample = conversation.model_dump()

                if include_sys_msg and "messages" in sample:
                    sample["messages"].insert(
                        0,
                        {
                            "role": "system",
                            "content": self.dataset_system_prompt,
                        },
                    )

                if (
                    self.config.conversation_type
                    in {
                        "agent_cot_tools",
                        "agent_cot_hybrid",
                        "agent_cot_multi_turn",
                        "xlam_multi_turn",
                    }
                    and self.tool_registry is not None
                ):
                    sample["available_tools"] = [
                        tool.model_dump() for tool in self.tool_registry.tools
                    ]

                # Add domain context for XLAM multi-turn
                if self.config.conversation_type == "xlam_multi_turn":
                    if self.config.domain_context:
                        sample["domain_context"] = self.config.domain_context
                    elif "scenario_description" not in sample:
                        sample["scenario_description"] = "AI assistant conversation"

            except Exception as e:  # noqa: BLE001
                if self.config.conversation_type == "xlam_multi_turn":
                    logger.error("XLAM generation error: %s", e, exc_info=True)
                return False, e
            else:
                return True, sample

        tasks = [asyncio.create_task(_generate(prompt)) for prompt in prompts]
        results = await asyncio.gather(*tasks)

        for success, payload in results:
            if success:
                samples.append(payload)  # type: ignore[arg-type]
            else:
                error = payload
                error_msg = f"Generation failed: {error}"
                failed_responses.append(error_msg)
                failure_type = self.analyze_failure(
                    str(error), error=error if isinstance(error, Exception) else None
                )
                self.failure_analysis[failure_type].append(error_msg)

        return samples, failed_responses

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
                summary["failure_examples"].append(
                    (
                        str(ex)[:200] + "..."
                        if len(str(ex)) > 200  # noqa: PLR2004
                        else str(ex)
                    )
                    for ex in examples
                )
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
        ensure_not_running_loop("DataSetGenerator.create_data")
        return asyncio.run(
            self.create_data_async(
                num_steps=num_steps,
                num_example_demonstrations=num_example_demonstrations,
                batch_size=batch_size,
                topic_model=topic_model,
                model_name=model_name,
                sys_msg=sys_msg,
            )
        )

    def create_data_with_events(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
    ):
        ensure_not_running_loop("DataSetGenerator.create_data_with_events")

        async def _async_generator() -> AsyncGenerator[dict | Dataset, None]:
            async for event in self.create_data_with_events_async(
                num_steps=num_steps,
                num_example_demonstrations=num_example_demonstrations,
                batch_size=batch_size,
                topic_model=topic_model,
                model_name=model_name,
                sys_msg=sys_msg,
            ):
                yield event

        agen = _async_generator()

        def _sync_generator():
            loop = asyncio.new_event_loop()
            try:
                while True:
                    try:
                        event = loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
                    else:
                        yield event
            finally:
                loop.run_until_complete(agen.aclose())
                loop.close()

        return _sync_generator()

    async def create_data_async(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
    ) -> Dataset:
        if num_steps is None:
            num_steps = 1

        self._validate_create_data_params(num_steps, batch_size, topic_model)

        if model_name:
            self.model_name = model_name.strip()

        if not self.model_name:
            raise DataSetGeneratorError("")

        include_sys_msg = sys_msg if sys_msg is not None else self.config.sys_msg

        topic_paths, num_steps = self._prepare_topic_paths(num_steps, batch_size, topic_model)

        total_samples = num_steps * batch_size
        data_creation_prompt = self._get_cot_prompt_template()

        final_result: Dataset | dict | None = None
        async for event in self._run_generation_loop_async(
            num_steps=num_steps,
            batch_size=batch_size,
            total_samples=total_samples,
            topic_paths=topic_paths or [],
            data_creation_prompt=data_creation_prompt,
            num_example_demonstrations=num_example_demonstrations,
            include_sys_msg=include_sys_msg,
        ):
            final_result = event

        if isinstance(final_result, Dataset):
            trace(
                "dataset_created",
                {
                    "provider": self.provider,
                    "model_name": self.model_name,
                    "conversation_type": self.config.conversation_type,
                    "samples_count": len(final_result.samples),
                    "failed_samples": len(self.failed_samples),
                    "success": len(final_result.samples) > 0,
                },
            )
            return final_result

        msg = "Dataset generation failed"
        raise DataSetGeneratorError(msg)

    async def create_data_with_events_async(
        self,
        num_steps: int | None = None,
        num_example_demonstrations: int = 3,
        batch_size: int = 10,
        topic_model: TopicModel | None = None,
        model_name: str | None = None,
        sys_msg: bool | None = None,
    ) -> AsyncGenerator[dict | Dataset, None]:
        if num_steps is None:
            num_steps = 1

        self._validate_create_data_params(num_steps, batch_size, topic_model)

        if model_name:
            self.model_name = model_name.strip()

        if not self.model_name:
            raise DataSetGeneratorError("")

        include_sys_msg = sys_msg if sys_msg is not None else self.config.sys_msg

        topic_paths, num_steps = self._prepare_topic_paths(num_steps, batch_size, topic_model)

        total_samples = num_steps * batch_size
        data_creation_prompt = self._get_cot_prompt_template()

        async for event in self._run_generation_loop_async(
            num_steps=num_steps,
            batch_size=batch_size,
            total_samples=total_samples,
            topic_paths=topic_paths or [],
            data_creation_prompt=data_creation_prompt,
            num_example_demonstrations=num_example_demonstrations,
            include_sys_msg=include_sys_msg,
        ):
            yield event

    async def _run_generation_loop_async(  # noqa: PLR0912
        self,
        num_steps: int,
        batch_size: int,
        total_samples: int,
        topic_paths: list,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        include_sys_msg: bool,
    ) -> AsyncGenerator[dict | Dataset, None]:
        """Run the main generation loop yielding progress events."""
        try:
            yield {
                "event": "generation_start",
                "model_name": self.model_name,
                "num_steps": num_steps,
                "batch_size": batch_size,
                "total_samples": total_samples,
            }

            for step in range(num_steps):
                yield {"event": "step_start", "step": step + 1, "total_steps": num_steps}

                start_idx = step * batch_size
                prompts = self._generate_batch_prompts(
                    batch_size,
                    start_idx,
                    topic_paths,
                    data_creation_prompt,
                    num_example_demonstrations,
                )

                failed_before = len(self.failed_samples)

                success, samples_generated = await self._process_batch_with_retries_async(
                    prompts, include_sys_msg
                )

                failed_in_batch = len(self.failed_samples) - failed_before
                failure_reasons = []
                if failed_in_batch > 0 and self.failed_samples:
                    recent_failures = self.failed_samples[-failed_in_batch:]
                    failure_reasons = recent_failures[:3]

                yield {
                    "event": "step_complete",
                    "step": step + 1,
                    "samples_generated": samples_generated,
                    "success": success,
                    "failed_in_step": failed_in_batch,
                    "failure_reasons": failure_reasons,
                }

                if not success:
                    yield {
                        "event": "step_failed",
                        "step": step + 1,
                        "message": f"Failed to process batch {step + 1} after all retries",
                    }

            yield {
                "event": "generation_complete",
                "total_samples": len(self.dataset),
                "failed_samples": len(self.failed_samples),
            }

        except KeyboardInterrupt:
            yield {"event": "generation_interrupted", "message": "Generation interrupted by user."}
            self.print_failure_summary()
            self.save_dataset(INTERRUPTED_DATASET_FILENAME)

        except Exception as e:  # noqa: BLE001
            yield {"event": "generation_error", "error": str(e)}
            self.print_failure_summary()
            self.save_dataset(ERROR_DATASET_FILENAME)
            raise DataSetGeneratorError("failed") from e

        yield self.dataset

    async def _process_batch_with_retries_async(
        self,
        prompts: list[str],
        include_sys_msg: bool,
    ) -> tuple[bool, int]:
        """Process a batch with retry logic."""
        for attempt in range(self.config.max_retries):
            try:
                samples, failed_responses = await self._generate_structured_samples_async(
                    prompts, include_sys_msg
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
                    return True, successful_samples  # Success - exit retry loop

            except DataSetGeneratorError as e:
                # Authentication and API errors are now wrapped in DataSetGeneratorError
                error_str = str(e).lower()
                if any(
                    keyword in error_str
                    for keyword in ["api_key", "api key", "authentication", "unauthorized"]
                ):
                    error_msg = f"Authentication failed for provider '{self.provider}'. Please set the required API key environment variable."
                    self.failure_analysis["authentication_error"].append(error_msg)
                else:
                    error_msg = f"API error for provider '{self.provider}': {str(e)[:100]}..."
                    self.failure_analysis["api_errors"].append(error_msg)

                self.failed_samples.append(error_msg)
                logger.exception("API error: %s", error_msg)
                return False, 0  # Don't retry authentication/API errors
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    self.failed_samples.append(str(e))
                    failure_type = self.analyze_failure(str(e), error=e)
                    self.failure_analysis[failure_type].append(str(e))
                    return False, 0
            else:
                # If no exception and no samples, return False, 0
                return False, 0

        return False, 0

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
                    for i, example in enumerate(
                        summary["failure_examples"].get(failure_type, []), 1
                    ):
                        print(f"  {i}. {example}")
        print("\n=============================")

    def build_prompt(
        self,
        data_creation_prompt: str,
        num_example_demonstrations: int,
        subtopics_list: list[str] | None = None,
    ) -> str:
        prompt = data_creation_prompt.replace("{{{{system_prompt}}}}", self.generation_prompt)
        prompt = prompt.replace("{{{{instructions}}}}", self.build_custom_instructions_text())
        prompt = prompt.replace(
            "{{{{examples}}}}", self.build_examples_text(num_example_demonstrations)
        )
        prompt = prompt.replace("{{{{subtopics}}}}", self.build_subtopics_text(subtopics_list))

        # XLAM multi-turn specific replacements
        if self.config.conversation_type == "xlam_multi_turn":
            tools_text = self.build_tools_text()
            domain_context = (
                self.config.domain_context or "AI assistant with access to various tools"
            )
            prompt = prompt.replace("{{{{tools}}}}", tools_text)
            prompt = prompt.replace("{{{{domain_context}}}}", domain_context)

        return prompt

    def build_system_prompt(self):
        """Return the original system prompt for dataset inclusion."""
        return self.dataset_system_prompt

    def build_custom_instructions_text(self) -> str:
        if self.config.instructions is None or self.config.instructions == "":
            return ""
        return f"\nHere are additional instructions:\n<instructions>\n{self.config.instructions}\n</instructions>\n"

    def build_examples_text(self, num_example_demonstrations: int):
        if self.config.example_data is None or num_example_demonstrations == 0:
            return ""
        # Bandit: not a security function
        examples = random.sample(self.config.example_data.samples, num_example_demonstrations)  # nosec
        examples_text = "Here are output examples:\n\n"
        examples_text += "\n".join(f"Example {i + 1}: \n\n{ex}\n" for i, ex in enumerate(examples))
        return f"\nHere are output examples:\n<examples>\n{examples_text}\n</examples>\n"

    def build_tools_text(self) -> str:
        """Build formatted tools text for XLAM multi-turn prompts."""
        if not self.tool_registry:
            return "No tools available"

        tools_text = []
        for tool in self.tool_registry.tools:
            params_text = []
            for param in tool.parameters:
                req = " (required)" if param.required else " (optional)"
                params_text.append(f"  - {param.name} ({param.type}){req}: {param.description}")

            tool_text = f"• {tool.name}: {tool.description}\n  Parameters:\n" + "\n".join(
                params_text
            )
            tools_text.append(tool_text)

        return "\n\n".join(tools_text)

    def build_subtopics_text(self, subtopic_list: list[str] | None):
        if subtopic_list is None:
            return ""
        return f"\nLastly, the topic of the training data should be related to the following subtopics: {' -> '.join(subtopic_list)}"

    def _get_cot_prompt_template(self) -> str:  # noqa: PLR0911
        """Get the appropriate CoT prompt template based on conversation type and reasoning style."""
        # Use mathematical prompts when reasoning_style is mathematical
        if self.config.reasoning_style == "mathematical":
            mathematical_prompts = {
                "cot_freetext": FREETEXT_COT_MATHEMATICAL_PROMPT,
                "cot_structured": STRUCTURED_COT_MATHEMATICAL_PROMPT,
                "cot_hybrid": HYBRID_COT_MATHEMATICAL_PROMPT,
            }
            if self.config.conversation_type in mathematical_prompts:
                return mathematical_prompts[self.config.conversation_type]

        # Agent tool-calling prompts use structured generation
        if self.config.conversation_type == "agent_cot_tools":
            if self.tool_registry is not None:
                return AgentPromptBuilder.build_tool_context_prompt(self.tool_registry)
            return AGENT_COT_TOOLS_PROMPT

        if self.config.conversation_type == "agent_cot_hybrid":
            if self.tool_registry is not None:
                return AgentPromptBuilder.build_tool_context_prompt(self.tool_registry)
            return AGENT_COT_HYBRID_PROMPT

        if self.config.conversation_type == "agent_cot_multi_turn":
            if self.tool_registry is not None:
                return AgentPromptBuilder.build_multi_turn_context_prompt(self.tool_registry)
            return AGENT_COT_MULTI_TURN_PROMPT

        if self.config.conversation_type == "xlam_multi_turn":
            # For XLAM multi-turn, we use a custom prompt with tools and domain context

            return XLAM_MULTI_TURN_PROMPT

        # Default CoT prompts for non-mathematical reasoning
        cot_prompts = {
            "cot_freetext": FREETEXT_COT_PROMPT,
            "cot_structured": STRUCTURED_COT_PROMPT,
            "cot_hybrid": HYBRID_COT_PROMPT,
        }
        return cot_prompts.get(self.config.conversation_type, CONVERSATION_GENERATION_PROMPT)

    def save_dataset(self, save_path: str):
        """Save the dataset to a file."""
        self.dataset.save(save_path)
