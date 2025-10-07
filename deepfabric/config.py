from typing import Any, Literal

import yaml

from pydantic import BaseModel, Field, model_validator

from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import ConfigurationError
from .metrics import trace


class LLMProviderConfig(BaseModel):
    """Configuration for LLM provider settings."""

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
        default=DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    max_tokens: int | None = Field(
        default=DEFAULT_MAX_TOKENS,
        ge=1,
        description="Maximum tokens for generation",
    )
    max_retries: int = Field(
        default=DEFAULT_MAX_RETRIES,
        ge=0,
        le=10,
        description="Maximum number of retries for failed generations",
    )
    request_timeout: int = Field(
        default=DEFAULT_REQUEST_TIMEOUT,
        ge=5,
        le=300,
        description="Request timeout in seconds",
    )


class TopicTreeConfig(BaseModel):
    """Configuration for topic tree generation."""

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic tree"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    llm_config: LLMProviderConfig | None = Field(
        default=None,
        description="LLM provider configuration (takes precedence over individual fields)",
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
    save_as: str | None = Field(default=None, description="Where to save the generated topic tree")

    @model_validator(mode="after")
    def ensure_llm_config(self) -> "TopicTreeConfig":
        """Ensure llm_config exists, creating from individual fields if needed (backward compatibility)."""
        if self.llm_config is None:
            self.llm_config = LLMProviderConfig(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
                max_retries=DEFAULT_MAX_RETRIES,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            )
        return self


class TopicGraphConfig(BaseModel):
    """Configuration for topic graph generation."""

    topic_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic graph"
    )
    topic_system_prompt: str = Field(
        default="", description="System prompt for topic exploration and generation"
    )
    llm_config: LLMProviderConfig | None = Field(
        default=None,
        description="LLM provider configuration (takes precedence over individual fields)",
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
    save_as: str | None = Field(default=None, description="Where to save the generated topic graph")

    @model_validator(mode="after")
    def ensure_llm_config(self) -> "TopicGraphConfig":
        """Ensure llm_config exists, creating from individual fields if needed (backward compatibility)."""
        if self.llm_config is None:
            self.llm_config = LLMProviderConfig(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=DEFAULT_MAX_TOKENS,
                max_retries=DEFAULT_MAX_RETRIES,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
            )
        return self


class DataEngineConfig(BaseModel):
    """Configuration for data engine generation."""

    instructions: str = Field(default="", description="Additional instructions for data generation")
    generation_system_prompt: str = Field(
        ..., min_length=1, description="System prompt for content generation"
    )
    llm_config: LLMProviderConfig | None = Field(
        default=None,
        description="LLM provider configuration (takes precedence over individual fields)",
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
        description="Maximum number of retries for failed generations",
    )
    save_as: str | None = Field(default=None, description="Where to save the generated data")

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
    max_tokens: int | None = Field(
        default=2000, description="Max Tokens for Single sample generation"
    )

    @model_validator(mode="after")
    def ensure_llm_config(self) -> "DataEngineConfig":
        """Ensure llm_config exists, creating from individual fields if needed (backward compatibility)."""
        if self.llm_config is None:
            self.llm_config = LLMProviderConfig(
                provider=self.provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens or DEFAULT_MAX_TOKENS,
                max_retries=self.max_retries,
                request_timeout=DEFAULT_REQUEST_TIMEOUT,
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
    huggingface: HuggingFaceConfig | None = Field(None, description="Hugging Face configuration")
    kaggle: KaggleConfig | None = Field(None, description="Kaggle configuration")

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
            # Get LLM config - validators ensure it always exists
            llm_config = self.topic_tree.llm_config
            assert llm_config is not None, "Validator should ensure llm_config exists"

            # Extract LLM-related overrides
            llm_override_keys = {"provider", "model", "temperature", "max_retries", "max_tokens", "request_timeout"}
            llm_overrides = {k: overrides.pop(k) for k in list(overrides.keys()) if k in llm_override_keys}

            # Convert Pydantic model to dict, excluding llm fields and llm_config
            params = self.topic_tree.model_dump(
                exclude={"save_as", "llm_config"} | llm_override_keys
            )

            # Apply remaining overrides to params
            params.update(overrides)

            # Apply LLM overrides using model_copy (or pass through unchanged)
            if llm_overrides:
                params["llm_config"] = llm_config.model_copy(update=llm_overrides)
            else:
                params["llm_config"] = llm_config

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        else:
            return params

    def get_topic_graph_params(self, **overrides) -> dict:
        """Get parameters for Graph instantiation."""
        if not self.topic_graph:
            raise ConfigurationError("missing 'topic_graph' configuration")  # noqa: TRY003
        try:
            # Convert Pydantic model to dict and exclude save_as and llm_config
            params = self.topic_graph.model_dump(exclude={"save_as", "llm_config"})

            # Get LLM config - validators ensure it always exists
            llm_config = self.topic_graph.llm_config
            assert llm_config is not None, "Validator should ensure llm_config exists"

            # Handle provider and model separately if present
            override_provider = overrides.pop("provider", None)
            override_model = overrides.pop("model", None)
            override_temperature = overrides.pop("temperature", None)
            override_max_retries = overrides.pop("max_retries", None)
            override_max_tokens = overrides.pop("max_tokens", None)
            override_request_timeout = overrides.pop("request_timeout", None)

            # Remove deprecated individual fields from params
            params.pop("provider", None)
            params.pop("model", None)
            params.pop("temperature", None)
            params.pop("max_retries", None)

            # Apply remaining overrides
            params.update(overrides)

            # Priority: CLI overrides > llm_config (2 levels)
            resolved_provider = override_provider or llm_config.provider
            resolved_model = override_model or llm_config.model
            resolved_temperature = (
                override_temperature
                if override_temperature is not None
                else llm_config.temperature
            )
            resolved_max_retries = (
                override_max_retries
                if override_max_retries is not None
                else llm_config.max_retries
            )
            resolved_max_tokens = (
                override_max_tokens
                if override_max_tokens is not None
                else llm_config.max_tokens
            )
            resolved_request_timeout = (
                override_request_timeout
                if override_request_timeout is not None
                else llm_config.request_timeout
            )

            # Create LLMProviderConfig instance
            llm_config_obj = LLMProviderConfig(
                provider=resolved_provider,
                model=resolved_model,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                max_retries=resolved_max_retries,
                request_timeout=resolved_request_timeout,
            )

            # Return params with llm_config
            params["llm_config"] = llm_config_obj

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        return params

    def get_engine_params(self, **overrides) -> dict:
        """Get parameters for DataSetGenerator instantiation."""
        try:
            # Convert Pydantic model to dict and exclude save_as and llm_config
            params = self.data_engine.model_dump(exclude={"save_as", "llm_config"})

            # Get LLM config - validators ensure it always exists
            llm_config = self.data_engine.llm_config
            assert llm_config is not None, "Validator should ensure llm_config exists"

            # Handle LLM-related overrides separately
            override_provider = overrides.pop("provider", None)
            override_model = overrides.pop("model", None)
            override_temperature = overrides.pop("temperature", None)
            override_max_retries = overrides.pop("max_retries", None)
            override_max_tokens = overrides.pop("max_tokens", None)
            override_request_timeout = overrides.pop("request_timeout", None)

            # Remove deprecated individual fields from params
            params.pop("provider", None)
            params.pop("model", None)
            params.pop("temperature", None)
            params.pop("max_retries", None)

            # Apply remaining overrides
            params.update(overrides)

            # Priority: CLI overrides > llm_config (2 levels)
            resolved_provider = override_provider or llm_config.provider
            resolved_model = override_model or llm_config.model
            resolved_temperature = (
                override_temperature
                if override_temperature is not None
                else llm_config.temperature
            )
            resolved_max_retries = (
                override_max_retries
                if override_max_retries is not None
                else llm_config.max_retries
            )
            resolved_max_tokens = (
                override_max_tokens
                if override_max_tokens is not None
                else llm_config.max_tokens
            )
            resolved_request_timeout = (
                override_request_timeout
                if override_request_timeout is not None
                else llm_config.request_timeout
            )

            # Create LLMProviderConfig instance from resolved parameters
            llm_config_obj = LLMProviderConfig(
                provider=resolved_provider,
                model=resolved_model,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                max_retries=resolved_max_retries,
                request_timeout=resolved_request_timeout,
            )
            params["llm_config"] = llm_config_obj

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
