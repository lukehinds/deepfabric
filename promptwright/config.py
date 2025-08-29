"""Configuration handling for YAML-based promptwright configurations."""

from typing import Any

import yaml

from pydantic import BaseModel, Field, field_validator

from .constants import DEFAULT_MODEL, DEFAULT_PROVIDER, SYSTEM_PROMPT_PLACEHOLDER
from .engine import EngineArguments
from .exceptions import ConfigurationError
from .topic_tree import TopicTreeArguments


def construct_model_string(provider: str, model: str) -> str:
    """Construct the full model string for LiteLLM."""
    return f"{provider}/{model}"


class PromptWrightConfig(BaseModel):
    """Configuration for PromptWright tasks."""

    system_prompt: str = Field(..., min_length=1, description="System prompt for the model")
    topic_tree: dict[str, Any] = Field(..., description="Topic tree configuration")
    data_engine: dict[str, Any] = Field(..., description="Data engine configuration")
    dataset: dict[str, Any] = Field(..., description="Dataset configuration")
    huggingface: dict[str, Any] | None = Field(None, description="Hugging Face configuration")

    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v):
        if not v or not v.strip():
            raise ValueError("required")
        return v.strip()

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PromptWrightConfig":
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

        try:
            return cls(**config_dict)
        except Exception as e:
            raise ConfigurationError(  # noqa: TRY003
                f"invalid structure: {str(e)}"
            ) from e  # noqa: TRY003

    def get_topic_tree_args(self, **overrides) -> TopicTreeArguments:
        """Get TopicTreeArguments from config with optional overrides."""
        try:
            args = self.topic_tree.get("args", {}).copy()

            # Replace system prompt placeholder
            if "model_system_prompt" in args and isinstance(args["model_system_prompt"], str):
                args["model_system_prompt"] = args["model_system_prompt"].replace(
                    SYSTEM_PROMPT_PLACEHOLDER, self.system_prompt
                )

            # Handle provider and model separately
            provider = overrides.pop("provider", args.pop("provider", DEFAULT_PROVIDER))
            model = overrides.pop("model", args.pop("model", DEFAULT_MODEL))

            # Apply remaining overrides
            args.update(overrides)

            # Construct full model string
            args["model_name"] = construct_model_string(provider, model)

            return TopicTreeArguments(**args)
        except Exception as e:
            raise ConfigurationError(f"args error: {str(e)}") from e  # noqa: TRY003

    def get_engine_args(self, **overrides) -> EngineArguments:
        """Get EngineArguments from config with optional overrides."""
        try:
            args = self.data_engine.get("args", {}).copy()

            # Replace system prompt placeholder
            if "system_prompt" in args and isinstance(args["system_prompt"], str):
                args["system_prompt"] = args["system_prompt"].replace(
                    SYSTEM_PROMPT_PLACEHOLDER, self.system_prompt
                )

            # Handle provider and model separately
            provider = overrides.pop("provider", args.pop("provider", DEFAULT_PROVIDER))
            model = overrides.pop("model", args.pop("model", DEFAULT_MODEL))

            # Apply remaining overrides
            args.update(overrides)

            # Construct full model string
            args["model_name"] = construct_model_string(provider, model)

            # Get sys_msg from dataset config, defaulting to True
            dataset_config = self.get_dataset_config()
            args.setdefault("sys_msg", dataset_config.get("creation", {}).get("sys_msg", True))

            return EngineArguments(**args)
        except Exception as e:
            raise ConfigurationError(f"args error: {str(e)}") from e  # noqa: TRY003

    def get_dataset_config(self) -> dict:
        """Get dataset configuration."""
        return self.dataset

    def get_huggingface_config(self) -> dict:
        """Get Hugging Face configuration."""
        return self.huggingface or {}
