from typing import Any, Literal

import yaml

from pydantic import BaseModel, Field, model_validator

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import ConfigurationError
from .metrics import trace


class TopicTreeConfig(BaseModel):
    """Configuration for topic tree generation."""

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic tree"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=TOPIC_TREE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    degree: int = Field(
        default=TOPIC_TREE_DEFAULT_DEGREE,
        ge=1,
        le=50,
        description="Number of subtopics per node",
    )
    depth: int = Field(
        default=TOPIC_TREE_DEFAULT_DEPTH,
        ge=1,
        le=10,
        description="Depth of the tree",
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated topic tree")


class TopicGraphConfig(BaseModel):
    """Configuration for topic graph generation."""

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic graph"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=0.6,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    degree: int = Field(default=3, ge=1, le=10, description="The branching factor of the graph")
    depth: int = Field(default=2, ge=1, le=5, description="The depth of the graph")
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated topic graph")


class DataEngineConfig(BaseModel):
    """Configuration for data engine generation."""

    instructions: str = Field(default="", description="Additional instructions for data generation")
    generation_system_prompt: str = Field(
        ..., min_length=1, description="System prompt for content generation"
    )
    provider: str = Field(
        default=DEFAULT_PROVIDER,
        min_length=1,
        description="LLM provider (openai, anthropic, gemini, ollama)",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        min_length=1,
        description="The name of the model to be used",
    )
    temperature: float = Field(
        default=ENGINE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description="Maximum number of retries for failed generations (deprecated, use rate_limit)",
    )
    max_tokens: int = Field(
        default=2000, ge=1, description="Maximum tokens to generate in a single call to the llm"
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for API endpoint (e.g., custom OpenAI-compatible servers)",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated data")

    # Rate limiting configuration
    rate_limit: dict[str, int | float | str | bool] | None = Field(
        default=None,
        description="Rate limiting and retry configuration (uses provider defaults if not specified)",
    )

    # Modular conversation configuration
    conversation_type: Literal["basic", "chain_of_thought"] = Field(
        default="basic",
        description="Base conversation type: basic (simple chat), chain_of_thought (with reasoning traces)",
    )

    reasoning_style: Literal["freetext", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style for chain_of_thought type: freetext (natural language), structured (step-by-step), hybrid (both)",
    )

    agent_mode: Literal["single_turn", "multi_turn"] | None = Field(
        default=None,
        description="Agent mode: single_turn (one-shot tool use), multi_turn (extended agent conversations). Requires tools to be configured.",
    )

    # Tool configuration (used when agent_mode is enabled or for tool_calling)
    available_tools: list[str] = Field(
        default_factory=list,
        description="List of tool names available (empty means all tools from registry)",
    )
    custom_tools: list[dict] = Field(
        default_factory=list, description="Custom tool definitions as dictionaries"
    )
    max_tools_per_query: int = Field(
        default=3, ge=1, le=10, description="Maximum number of tools per query/turn"
    )
    max_tools_strict: bool = Field(
        default=True,
        description="If True, discard samples exceeding max_tools_per_query. If False, keep sample but truncate executions to limit.",
    )
    tool_registry_path: str | None = Field(
        default=None, description="Path to custom tool definitions file (JSON/YAML)"
    )

    @model_validator(mode="after")
    def validate_configuration(self):
        """Validate that configuration combinations are consistent."""
        # Validate reasoning_style is only used with chain_of_thought
        if self.reasoning_style is not None and self.conversation_type != "chain_of_thought":
            raise ValueError(
                f"reasoning_style can only be set when conversation_type='chain_of_thought', "
                f"got conversation_type='{self.conversation_type}'"
            )

        # Validate chain_of_thought has a reasoning_style
        if self.conversation_type == "chain_of_thought" and self.reasoning_style is None:
            raise ValueError(
                "reasoning_style must be specified when conversation_type='chain_of_thought'. "
                "Choose from: 'freetext', 'structured', 'hybrid'"
            )

        # Validate agent_mode requires tools
        if self.agent_mode is not None:
            has_tools = bool(self.available_tools or self.custom_tools or self.tool_registry_path)
            if not has_tools:
                raise ValueError(
                    "agent_mode requires tools to be configured. "
                    "Specify at least one of: available_tools, custom_tools, or tool_registry_path"
                )

        return self


class DatasetCreationConfig(BaseModel):
    """Configuration for dataset creation parameters."""

    num_steps: int = Field(
        default=ENGINE_DEFAULT_NUM_EXAMPLES,
        ge=1,
        description="Number of training examples to generate",
    )
    batch_size: int = Field(
        default=ENGINE_DEFAULT_BATCH_SIZE,
        ge=1,
        description="Number of examples to process at a time",
    )
    sys_msg: bool | None = Field(
        default=None,
        description="Include system messages in output format",
    )
    provider: str | None = Field(
        default=None,
        description="Optional provider override for dataset creation",
    )
    model: str | None = Field(
        default=None,
        description="Optional model override for dataset creation",
    )


class FormatterConfig(BaseModel):
    """Configuration for a single formatter."""

    name: str = Field(..., min_length=1, description="Name identifier for this formatter")
    template: str = Field(..., min_length=1, description="Template path (builtin:// or file://)")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Formatter-specific configuration"
    )
    output: str | None = Field(None, description="Output file path for this formatter")


class DatasetConfig(BaseModel):
    """Configuration for dataset assembly and output."""

    creation: DatasetCreationConfig = Field(
        default_factory=DatasetCreationConfig,
        description="Dataset creation parameters",
    )
    save_as: str = Field(..., min_length=1, description="Where to save the final dataset")
    formatters: list[FormatterConfig] = Field(
        default_factory=list, description="List of formatters to apply to the dataset"
    )


class HuggingFaceConfig(BaseModel):
    """Configuration for Hugging Face Hub integration."""

    repository: str = Field(..., min_length=1, description="HuggingFace repository name")
    tags: list[str] = Field(default_factory=list, description="Tags for the dataset")


class KaggleConfig(BaseModel):
    """Configuration for Kaggle integration."""

    handle: str = Field(
        ..., min_length=1, description="Kaggle dataset handle (username/dataset-name)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for the dataset")
    description: str | None = Field(None, description="Description for the dataset")
    version_notes: str | None = Field(None, description="Version notes for dataset update")


class EvaluationConfig(BaseModel):
    """Configuration for model evaluation."""

    conversation_type: Literal["basic", "chain_of_thought"] = Field(
        ...,
        description="Conversation type (must match dataset generation)",
    )
    reasoning_style: Literal["freetext", "structured", "hybrid"] | None = Field(
        default=None,
        description="Reasoning style for chain_of_thought type",
    )
    agent_mode: Literal["single_turn", "multi_turn"] | None = Field(
        default=None,
        description="Agent mode if tools are used",
    )
    metrics: list[str] = Field(
        default_factory=lambda: [
            "tool_selection_accuracy",
            "parameter_accuracy",
            "execution_success_rate",
            "response_quality",
        ],
        description="Metrics to compute during evaluation",
    )
    thresholds: dict[str, float] = Field(
        default_factory=dict,
        description="Pass/fail thresholds for metrics",
    )
    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "tool_selection": 0.40,
            "parameter_accuracy": 0.30,
            "execution_success": 0.20,
            "response_quality": 0.10,
        },
        description="Metric weights for overall score calculation",
    )
    output_dir: str = Field(
        default="./eval_results",
        description="Output directory for evaluation results",
    )
    output_formats: list[Literal["json", "html", "csv"]] = Field(
        default_factory=lambda: ["json", "html", "csv"],
        description="Output formats to generate",
    )
    include_failures: bool = Field(
        default=True,
        description="Include failed examples in output",
    )
    generate_charts: bool = Field(
        default=True,
        description="Generate visualization charts",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for model inference",
    )
    max_samples: int | None = Field(
        default=None,
        description="Maximum number of samples to evaluate (None for all)",
    )

    @model_validator(mode="after")
    def validate_evaluation_config(self) -> "EvaluationConfig":
        """Validate evaluation configuration consistency."""
        # Validate reasoning_style is only used with chain_of_thought
        if self.reasoning_style is not None and self.conversation_type != "chain_of_thought":
            raise ValueError(
                f"reasoning_style can only be set when conversation_type='chain_of_thought', "
                f"got conversation_type='{self.conversation_type}'"
            )

        # Validate chain_of_thought has a reasoning_style
        if self.conversation_type == "chain_of_thought" and self.reasoning_style is None:
            raise ValueError(
                "reasoning_style must be specified when conversation_type='chain_of_thought'. "
                "Choose from: 'freetext', 'structured', 'hybrid'"
            )

        return self


class DeepFabricConfig(BaseModel):
    """Configuration for DeepFabric tasks."""

    dataset_system_prompt: str | None = Field(
        None,
        description="System prompt that goes into the final dataset as the system message (falls back to generation_system_prompt if not provided)",
    )
    topic_tree: TopicTreeConfig | None = Field(None, description="Topic tree configuration")
    topic_graph: TopicGraphConfig | None = Field(None, description="Topic graph configuration")
    data_engine: DataEngineConfig = Field(..., description="Data engine configuration")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    evaluation: EvaluationConfig | None = Field(None, description="Evaluation configuration")
    huggingface: HuggingFaceConfig | None = Field(None, description="Hugging Face configuration")
    kaggle: KaggleConfig | None = Field(None, description="Kaggle configuration")

    @model_validator(mode="after")
    def validate_formatter_compatibility(self) -> "DeepFabricConfig":
        """Warn about potentially incompatible conversation types and formatters."""
        if not self.dataset or not self.dataset.formatters:
            return self

        conversation_type = self.data_engine.conversation_type
        agent_mode = self.data_engine.agent_mode
        reasoning_style = self.data_engine.reasoning_style

        # Define formatter compatibility rules
        formatter_warnings = {
            "alpaca": {
                "incompatible_with": {
                    "conversation_type": ["chain_of_thought"],
                    "agent_mode": ["multi_turn"],
                },
                "warning": "Alpaca format works best with simple Q&A. Chain-of-thought reasoning and multi-turn conversations may have high rejection rates.",
            },
            "chatml": {
                "incompatible_with": {
                    "reasoning_style": ["structured", "hybrid"],
                },
                "warning": "ChatML format may reject samples with complex reasoning structures. Consider using 'conversations' format for chain-of-thought data.",
            },
            "xlam_v2": {
                "requires": {
                    "agent_mode": ["single_turn", "multi_turn"],
                },
                "warning": "XLAM v2 format requires agent_mode to be set (single_turn or multi_turn) for tool-calling data.",
            },
            "tool_calling": {
                "requires": {
                    "agent_mode": ["single_turn", "multi_turn"],
                },
                "warning": "Tool calling format requires agent_mode to be set for tool-calling data.",
            },
            "single_tool_call": {
                "requires": {
                    "agent_mode": ["single_turn"],
                },
                "warning": "Single tool call format requires agent_mode='single_turn' and will reject multi-turn conversations.",
            },
        }

        # Check each formatter
        for formatter_config in self.dataset.formatters:
            template = formatter_config.template
            formatter_name = None

            # Extract formatter name from template (builtin://formatter_name)
            if isinstance(template, str) and "builtin://" in template:
                formatter_name = template.replace("builtin://", "").replace(".py", "")

            if not formatter_name or formatter_name not in formatter_warnings:
                continue

            rules = formatter_warnings[formatter_name]

            # Check incompatibilities
            if "incompatible_with" in rules:
                incompatible = rules["incompatible_with"]

                if (
                    "conversation_type" in incompatible
                    and conversation_type in incompatible["conversation_type"]
                ):
                    print(
                        f"\n⚠️  Warning: Formatter '{formatter_config.name}' may have high rejection rates with conversation_type='{conversation_type}'"
                    )
                    print(f"   {rules['warning']}\n")

                if "agent_mode" in incompatible and agent_mode in incompatible["agent_mode"]:
                    print(
                        f"\n⚠️  Warning: Formatter '{formatter_config.name}' may have high rejection rates with agent_mode='{agent_mode}'"
                    )
                    print(f"   {rules['warning']}\n")

                if (
                    "reasoning_style" in incompatible
                    and reasoning_style in incompatible["reasoning_style"]
                ):
                    print(
                        f"\n⚠️  Warning: Formatter '{formatter_config.name}' may have high rejection rates with reasoning_style='{reasoning_style}'"
                    )
                    print(f"   {rules['warning']}\n")

            # Check requirements
            if "requires" in rules:
                requirements = rules["requires"]

                if "agent_mode" in requirements and (
                    not agent_mode or agent_mode not in requirements["agent_mode"]
                ):
                    print(
                        f"\n⚠️  Warning: Formatter '{formatter_config.name}' requires agent_mode to be one of {requirements['agent_mode']}"
                    )
                    print(f"   Current: agent_mode={agent_mode}")
                    print(f"   {rules['warning']}\n")

        return self

    @classmethod
    def _migrate_legacy_format(cls, config_dict: dict) -> dict:
        """Migrate legacy 'args' wrapper format to new flat structure."""
        migrated = config_dict.copy()

        # Handle topic_tree args wrapper
        if (
            "topic_tree" in migrated
            and isinstance(migrated["topic_tree"], dict)
            and "args" in migrated["topic_tree"]
        ):
            print(
                "Warning: 'args' wrapper in topic_tree config is deprecated. Please update your config."
            )
            args = migrated["topic_tree"]["args"]
            save_as = migrated["topic_tree"].get("save_as")
            migrated["topic_tree"] = args.copy()
            if save_as:
                migrated["topic_tree"]["save_as"] = save_as

        # Handle topic_graph args wrapper
        if (
            "topic_graph" in migrated
            and isinstance(migrated["topic_graph"], dict)
            and "args" in migrated["topic_graph"]
        ):
            print(
                "Warning: 'args' wrapper in topic_graph config is deprecated. Please update your config."
            )
            args = migrated["topic_graph"]["args"]
            save_as = migrated["topic_graph"].get("save_as")
            migrated["topic_graph"] = args.copy()
            if save_as:
                migrated["topic_graph"]["save_as"] = save_as

        # Handle data_engine args wrapper
        if (
            "data_engine" in migrated
            and isinstance(migrated["data_engine"], dict)
            and "args" in migrated["data_engine"]
        ):
            print(
                "Warning: 'args' wrapper in data_engine config is deprecated. Please update your config."
            )
            args = migrated["data_engine"]["args"]
            save_as = migrated["data_engine"].get("save_as")
            migrated["data_engine"] = args.copy()
            if save_as:
                migrated["data_engine"]["save_as"] = save_as

        return migrated

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeepFabricConfig":
        """Load configuration from a YAML file."""
        try:
            with open(yaml_path, encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        except FileNotFoundError as e:
            raise ConfigurationError(f"not found: {yaml_path}") from e  # noqa: TRY003
        except yaml.YAMLError as e:
            raise ConfigurationError(f"invalid YAML: {str(e)}") from e  # noqa: TRY003
        except Exception as e:
            raise ConfigurationError(f"read error: {str(e)}") from e  # noqa: TRY003

        if not isinstance(config_dict, dict):
            raise ConfigurationError("must be dictionary")  # noqa: TRY003

        # Handle backward compatibility for nested "args" format
        config_dict = cls._migrate_legacy_format(config_dict)

        try:
            config = cls(**config_dict)
            trace(
                "config_loaded",
                {
                    "method": "yaml",
                    "has_topic_tree": config.topic_tree is not None,
                    "has_topic_graph": config.topic_graph is not None,
                    "has_huggingface": config.huggingface is not None,
                    "has_kaggle": config.kaggle is not None,
                },
            )
        except Exception as e:
            raise ConfigurationError(  # noqa: TRY003
                f"invalid structure: {str(e)}"
            ) from e  # noqa: TRY003
        else:
            return config

    def get_topic_tree_params(self, **overrides) -> dict:
        """Get parameters for Tree instantiation."""
        if not self.topic_tree:
            raise ConfigurationError("missing 'topic_tree' configuration")  # noqa: TRY003
        try:
            # Convert Pydantic model to dict and exclude save_as
            params = self.topic_tree.model_dump(exclude={"save_as"})

            # Handle provider and model separately if present
            override_provider = overrides.pop("provider", None)
            override_model = overrides.pop("model", None)
            config_provider = params.pop("provider", None)
            config_model = params.pop("model", None)

            # Apply remaining overrides
            params.update(overrides)

            # Determine final provider
            final_provider = override_provider or config_provider or DEFAULT_PROVIDER
            params["provider"] = final_provider

            # Determine final model and model_name
            if override_model:
                # If model is overridden, use just the model name (provider is separate)
                params["model_name"] = override_model
            elif config_model:
                # If model comes from config, use as-is for model_name
                params["model_name"] = config_model
            elif "model_name" not in params:
                params["model_name"] = DEFAULT_MODEL

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        else:
            return params

    def get_topic_graph_params(self, **overrides) -> dict:
        """Get parameters for Graph instantiation."""
        if not self.topic_graph:
            raise ConfigurationError("missing 'topic_graph' configuration")  # noqa: TRY003
        try:
            # Convert Pydantic model to dict and exclude save_as
            params = self.topic_graph.model_dump(exclude={"save_as"})

            # Handle provider and model separately if present
            override_provider = overrides.pop("provider", None)
            override_model = overrides.pop("model", None)
            config_provider = params.pop("provider", None)
            config_model = params.pop("model", None)

            # Apply remaining overrides
            params.update(overrides)

            # Determine final provider
            final_provider = override_provider or config_provider or DEFAULT_PROVIDER
            params["provider"] = final_provider

            # Determine final model and model_name
            if override_model:
                # If model is overridden, use just the model name (provider is separate)
                params["model_name"] = override_model
            elif config_model:
                # If model comes from config, use as-is for model_name
                params["model_name"] = config_model
            elif "model_name" not in params:
                params["model_name"] = DEFAULT_MODEL

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        return params

    def get_engine_params(self, **overrides) -> dict:
        """Get parameters for DataSetGenerator instantiation."""
        try:
            # Convert Pydantic model to dict and exclude save_as
            params = self.data_engine.model_dump(exclude={"save_as"})

            # Handle provider and model separately if present
            override_provider = overrides.pop("provider", None)
            override_model = overrides.pop("model", None)
            config_provider = params.pop("provider", None)
            config_model = params.pop("model", None)

            # Apply remaining overrides
            params.update(overrides)

            # Determine final provider
            final_provider = override_provider or config_provider or DEFAULT_PROVIDER
            params["provider"] = final_provider

            # Determine final model and model_name
            if override_model:
                # If model is overridden, use just the model name (provider is separate)
                params["model_name"] = override_model
            elif config_model:
                # If model comes from config, use as-is for model_name
                params["model_name"] = config_model
            elif "model_name" not in params:
                params["model_name"] = DEFAULT_MODEL

            # Get sys_msg from dataset config, defaulting to True
            sys_msg_value = self.dataset.creation.sys_msg
            if sys_msg_value is not None:
                params.setdefault("sys_msg", sys_msg_value)
            else:
                params.setdefault("sys_msg", True)

            # Set the dataset_system_prompt for the data engine, fall back to generation_system_prompt
            dataset_prompt = self.dataset_system_prompt or params.get("generation_system_prompt")
            if dataset_prompt:
                params.setdefault("dataset_system_prompt", dataset_prompt)

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        else:
            return params

    def get_dataset_config(self) -> dict:
        """Get dataset configuration."""
        return self.dataset.model_dump(exclude_none=True)

    def get_huggingface_config(self) -> dict:
        """Get Hugging Face configuration."""
        return self.huggingface.model_dump() if self.huggingface else {}

    def get_kaggle_config(self) -> dict:
        """Get Kaggle configuration."""
        return self.kaggle.model_dump() if self.kaggle else {}

    def get_formatter_configs(self) -> list[dict]:
        """Get list of formatter configurations."""
        return [formatter.model_dump() for formatter in self.dataset.formatters]
