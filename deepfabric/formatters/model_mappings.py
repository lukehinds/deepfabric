"""
Model mapping configuration loader for HF chat template formatter.

This module handles loading and resolving model-specific configurations from YAML files.
"""

import fnmatch
import logging

from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ModelMappingLoader:
    """
    Load and resolve model-specific formatting configurations.

    Supports:
    - Exact model ID matching
    - Glob pattern matching (e.g., "Qwen/*", "meta-llama/Llama-3*")
    - Model family aliases
    - Configuration merging (user overrides defaults)
    """

    def __init__(self, config_path: str | Path | None = None):
        """
        Initialize the mapping loader.

        Args:
            config_path: Path to YAML configuration file. If None, uses defaults only.
        """
        self.config_path = Path(config_path) if config_path else None
        self.config: dict[str, Any] = {}
        self.defaults: dict[str, Any] = self._get_builtin_defaults()

        if self.config_path and self.config_path.exists():
            self._load_config()

    def _get_builtin_defaults(self) -> dict[str, Any]:
        """
        Get built-in default configuration.

        Returns:
            Default configuration dictionary
        """
        return {
            "reasoning": {
                "inject_mode": "inline",
                "native_support": False,
                "prefix": "",
                "separator": "\n\n",
                "style": "compact",
            },
            "tools": {"format": "native", "native_support": False},
            "preprocessing": [],
        }

    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path) as f:  # type: ignore
                self.config = yaml.safe_load(f) or {}
            logger.info(f"Loaded model mappings from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load model mappings from {self.config_path}: {e}")
            self.config = {}

    def resolve(self, model_id: str) -> dict[str, Any]:
        """
        Resolve configuration for a specific model ID.

        Resolution order:
        1. Exact model ID match
        2. Glob pattern match
        3. Model family alias
        4. Defaults

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")

        Returns:
            Resolved configuration dictionary
        """
        resolved = self.defaults.copy()

        if not self.config:
            logger.debug(f"No custom config loaded, using defaults for {model_id}")
            return resolved

        models = self.config.get("models", {})

        # 1. Try exact match
        if model_id in models:
            logger.debug(f"Found exact match for {model_id}")
            return self._merge_configs(resolved, models[model_id])

        # 2. Try glob pattern match
        for pattern, pattern_config in models.items():
            if self._matches_pattern(model_id, pattern):
                logger.debug(f"Matched {model_id} to pattern {pattern}")
                return self._merge_configs(resolved, pattern_config)

        # 3. Try model family aliases
        model_families = self.config.get("model_families", {})
        for family_name, patterns in model_families.items():
            for pattern in patterns:
                if self._matches_pattern(model_id, pattern):
                    logger.debug(f"Matched {model_id} to family {family_name}")
                    # Look for family-specific config in models
                    if family_name in models:
                        return self._merge_configs(resolved, models[family_name])

        # 4. Return defaults
        logger.debug(f"No match found for {model_id}, using defaults")
        return resolved

    def _matches_pattern(self, model_id: str, pattern: str) -> bool:
        """
        Check if model ID matches a glob pattern.

        Args:
            model_id: Model ID to check
            pattern: Glob pattern (e.g., "Qwen/*", "meta-llama/Llama-3*")

        Returns:
            True if matches, False otherwise
        """
        return fnmatch.fnmatch(model_id, pattern)

    def _merge_configs(self, base: dict, override: dict) -> dict:
        """
        Deep merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Configuration to merge on top

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Deep merge for nested dicts
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override
                result[key] = value

        return result

    def get_training_framework_config(self, framework: str) -> dict[str, Any]:
        """
        Get configuration for a specific training framework.

        Args:
            framework: Framework name (e.g., "grpo", "dpo", "sft")

        Returns:
            Framework-specific configuration or empty dict
        """
        frameworks = self.config.get("training_frameworks", {})
        return frameworks.get(framework, {})

    def get_capability_indicators(self) -> dict[str, list[str]]:
        """
        Get token indicators for automatic capability detection.

        Returns:
            Dictionary mapping capability names to token indicator lists
        """
        detection = self.config.get("capability_detection", {})
        return {
            "reasoning": detection.get("reasoning", {}).get("token_indicators", []),
            "tools": detection.get("tools", {}).get("token_indicators", []),
        }


def load_model_mappings(config_path: str | Path | None = None) -> ModelMappingLoader:
    """
    Convenience function to load model mappings.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        ModelMappingLoader instance
    """
    return ModelMappingLoader(config_path)
