from typing import Any

import yaml

from pydantic import BaseModel, Field

from .constants import DEFAULT_MODEL, DEFAULT_PROVIDER
from .exceptions import ConfigurationError

# Config classes are no longer imported directly as they're not used in this module


def construct_model_string(provider: str, model: str) -> str:
    """Construct the full model string for LiteLLM."""
    return f"{provider}/{model}"


class DeepFabricConfig(BaseModel):
    """Configuration for DeepFabric tasks."""

    dataset_system_prompt: str | None = Field(
        None,
        description="System prompt that goes into the final dataset as the system message (falls back to generation_system_prompt if not provided)",
    )
    topic_tree: dict[str, Any] | None = Field(None, description="Topic tree configuration")
    topic_graph: dict[str, Any] | None = Field(None, description="Topic graph configuration")
    data_engine: dict[str, Any] = Field(..., description="Data engine configuration")
    dataset: dict[str, Any] = Field(..., description="Dataset configuration")
    huggingface: dict[str, Any] | None = Field(None, description="Hugging Face configuration")

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

        try:
            return cls(**config_dict)
        except Exception as e:
            raise ConfigurationError(  # noqa: TRY003
                f"invalid structure: {str(e)}"
            ) from e  # noqa: TRY003

    def get_topic_tree_params(self, **overrides) -> dict:
        """Get parameters for Tree instantiation."""
        if not self.topic_tree:
            raise ConfigurationError("missing 'topic_tree' configuration")  # noqa: TRY003
        try:
            # Check for old format with deprecation warning
            if "args" in self.topic_tree:
                params = self.topic_tree["args"].copy()
                print(
                    "Warning: 'args' wrapper in topic_tree config is deprecated. Please update your config."
                )
            else:
                params = self.topic_tree.copy()

            # Remove non-constructor params
            params.pop("save_as", None)

            # Handle provider and model separately if present
            provider = overrides.pop("provider", params.pop("provider", None))
            model = overrides.pop("model", params.pop("model", None))

            # Apply remaining overrides
            params.update(overrides)

            # Construct full model string if provider/model were specified
            if provider and model:
                params["model_name"] = construct_model_string(provider, model)
            elif "model_name" not in params:
                params["model_name"] = construct_model_string(DEFAULT_PROVIDER, DEFAULT_MODEL)

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        else:
            return params

    def get_topic_graph_params(self, **overrides) -> dict:
        """Get parameters for Graph instantiation."""
        if not self.topic_graph:
            raise ConfigurationError("missing 'topic_graph' configuration")  # noqa: TRY003
        try:
            # Check for old format with deprecation warning
            if "args" in self.topic_graph:
                params = self.topic_graph["args"].copy()
                print(
                    "Warning: 'args' wrapper in topic_graph config is deprecated. Please update your config."
                )
            else:
                params = self.topic_graph.copy()

            # Remove non-constructor params
            params.pop("save_as", None)

            # Handle provider and model separately if present
            provider = overrides.pop("provider", params.pop("provider", None))
            model = overrides.pop("model", params.pop("model", None))

            # Apply remaining overrides
            params.update(overrides)

            # Construct full model string if provider/model were specified
            if provider and model:
                params["model_name"] = construct_model_string(provider, model)
            elif "model_name" not in params:
                params["model_name"] = construct_model_string(DEFAULT_PROVIDER, DEFAULT_MODEL)

        except Exception as e:
            raise ConfigurationError(f"config error: {str(e)}") from e  # noqa: TRY003
        return params

    def get_engine_params(self, **overrides) -> dict:
        """Get parameters for DataSetGenerator instantiation."""
        try:
            # Check for old format with deprecation warning
            if "args" in self.data_engine:
                params = self.data_engine["args"].copy()
                print(
                    "Warning: 'args' wrapper in data_engine config is deprecated. Please update your config."
                )
            else:
                params = self.data_engine.copy()

            # Remove non-constructor params
            params.pop("save_as", None)

            # Handle provider and model separately if present
            provider = overrides.pop("provider", params.pop("provider", None))
            model = overrides.pop("model", params.pop("model", None))

            # Apply remaining overrides
            params.update(overrides)

            # Construct full model string if provider/model were specified
            if provider and model:
                params["model_name"] = construct_model_string(provider, model)
            elif "model_name" not in params:
                params["model_name"] = construct_model_string(DEFAULT_PROVIDER, DEFAULT_MODEL)

            # Get sys_msg from dataset config, defaulting to True
            dataset_config = self.get_dataset_config()
            params.setdefault("sys_msg", dataset_config.get("creation", {}).get("sys_msg", True))

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
        return self.dataset

    def get_huggingface_config(self) -> dict:
        """Get Hugging Face configuration."""
        return self.huggingface or {}
