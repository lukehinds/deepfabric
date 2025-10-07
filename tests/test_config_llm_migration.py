"""Comprehensive tests for LLM config migration.

Tests the three config styles: old-style (individual fields), new-style (llm_config),
and hybrid (mix of both). Ensures backward compatibility and proper precedence.
"""

import os
import tempfile

import pytest
import yaml

from pydantic import ValidationError

from deepfabric.config import DeepFabricConfig, LLMProviderConfig
from deepfabric.exceptions import ConfigurationError


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def old_style_config_dict():
    """Old-style config: individual provider/model/temperature fields."""
    return {
        "dataset_system_prompt": "Old style test prompt",
        "topic_tree": {
            "topic_prompt": "Old style tree prompt",
            "topic_system_prompt": "Old style tree system prompt",
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "degree": 2,
            "depth": 2,
            "save_as": "old_tree.jsonl",
        },
        "topic_graph": {
            "topic_prompt": "Old style graph prompt",
            "topic_system_prompt": "Old style graph system prompt",
            "provider": "anthropic",
            "model": "claude-3",
            "temperature": 0.6,
            "degree": 3,
            "depth": 2,
            "save_as": "old_graph.json",
        },
        "data_engine": {
            "generation_system_prompt": "Old style engine prompt",
            "provider": "gemini",
            "model": "gemini-pro",
            "temperature": 0.8,
            "max_retries": 3,
        },
        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 1,
                "sys_msg": True,
            },
            "save_as": "old_dataset.jsonl",
        },
    }


@pytest.fixture
def new_style_config_dict():
    """New-style config: llm_config objects."""
    return {
        "dataset_system_prompt": "New style test prompt",
        "topic_tree": {
            "topic_prompt": "New style tree prompt",
            "topic_system_prompt": "New style tree system prompt",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "max_retries": 4,
                "request_timeout": 60,
            },
            "degree": 2,
            "depth": 2,
            "save_as": "new_tree.jsonl",
        },
        "topic_graph": {
            "topic_prompt": "New style graph prompt",
            "topic_system_prompt": "New style graph system prompt",
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3",
                "temperature": 0.6,
                "max_tokens": 1500,
                "max_retries": 5,
                "request_timeout": 45,
            },
            "degree": 3,
            "depth": 2,
            "save_as": "new_graph.json",
        },
        "data_engine": {
            "generation_system_prompt": "New style engine prompt",
            "llm_config": {
                "provider": "gemini",
                "model": "gemini-pro",
                "temperature": 0.8,
                "max_tokens": 3000,
                "max_retries": 6,
                "request_timeout": 90,
            },
        },
        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 1,
                "sys_msg": True,
            },
            "save_as": "new_dataset.jsonl",
        },
    }


@pytest.fixture
def hybrid_config_dict():
    """Hybrid config: mix of old and new styles across components."""
    return {
        "dataset_system_prompt": "Hybrid test prompt",
        "topic_tree": {
            "topic_prompt": "Hybrid tree prompt",
            "topic_system_prompt": "Hybrid tree system prompt",
            # Old-style for tree
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "degree": 2,
            "depth": 2,
            "save_as": "hybrid_tree.jsonl",
        },
        "topic_graph": {
            "topic_prompt": "Hybrid graph prompt",
            "topic_system_prompt": "Hybrid graph system prompt",
            # New-style for graph
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3",
                "temperature": 0.6,
                "max_tokens": 1500,
                "max_retries": 5,
                "request_timeout": 45,
            },
            "degree": 3,
            "depth": 2,
            "save_as": "hybrid_graph.json",
        },
        "data_engine": {
            "generation_system_prompt": "Hybrid engine prompt",
            # New-style for engine
            "llm_config": {
                "provider": "gemini",
                "model": "gemini-pro",
                "temperature": 0.8,
                "max_tokens": 3000,
                "max_retries": 6,
                "request_timeout": 90,
            },
        },
        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 1,
                "sys_msg": True,
            },
            "save_as": "hybrid_dataset.jsonl",
        },
    }


@pytest.fixture
def conflicting_config_dict():
    """Config with both styles having different values (llm_config should win)."""
    return {
        "dataset_system_prompt": "Conflicting test prompt",
        "topic_tree": {
            "topic_prompt": "Conflicting tree prompt",
            "topic_system_prompt": "Conflicting tree system prompt",
            # Old-style values (should be ignored)
            "provider": "ollama",
            "model": "mistral:latest",
            "temperature": 0.5,
            # New-style values (should win)
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.9,
                "max_tokens": 2000,
                "max_retries": 5,
                "request_timeout": 60,
            },
            "degree": 2,
            "depth": 2,
            "save_as": "conflicting_tree.jsonl",
        },
        "data_engine": {
            "generation_system_prompt": "Conflicting engine prompt",
            "provider": "test",
            "model": "test-model",
        },
        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 1,
            },
            "save_as": "conflicting_dataset.jsonl",
        },
    }


def _create_temp_yaml_file(config_dict):
    """Helper to create temporary YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name
    return temp_path


def _cleanup_temp_file(temp_path):
    """Helper to cleanup temporary file."""
    if os.path.exists(temp_path):
        os.unlink(temp_path)


# ============================================================================
# CONFIG LOADING TESTS
# ============================================================================


def test_load_old_style_config(old_style_config_dict):
    """Test loading old-style config creates llm_config automatically."""
    temp_path = _create_temp_yaml_file(old_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # Verify topic_tree llm_config was created
        assert config.topic_tree is not None
        assert config.topic_tree.llm_config is not None
        assert isinstance(config.topic_tree.llm_config, LLMProviderConfig)
        assert config.topic_tree.llm_config.provider == "openai"
        assert config.topic_tree.llm_config.model == "gpt-4"
        assert config.topic_tree.llm_config.temperature == 0.7

        # Verify topic_graph llm_config was created
        assert config.topic_graph is not None
        assert config.topic_graph.llm_config is not None
        assert isinstance(config.topic_graph.llm_config, LLMProviderConfig)
        assert config.topic_graph.llm_config.provider == "anthropic"
        assert config.topic_graph.llm_config.model == "claude-3"
        assert config.topic_graph.llm_config.temperature == 0.6

        # Verify data_engine llm_config was created
        assert config.data_engine.llm_config is not None
        assert isinstance(config.data_engine.llm_config, LLMProviderConfig)
        assert config.data_engine.llm_config.provider == "gemini"
        assert config.data_engine.llm_config.model == "gemini-pro"
        assert config.data_engine.llm_config.temperature == 0.8
        assert config.data_engine.llm_config.max_retries == 3
    finally:
        _cleanup_temp_file(temp_path)


def test_load_new_style_config(new_style_config_dict):
    """Test loading new-style config preserves llm_config."""
    temp_path = _create_temp_yaml_file(new_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # Verify topic_tree llm_config is exactly as specified
        assert config.topic_tree is not None
        assert config.topic_tree.llm_config is not None
        assert config.topic_tree.llm_config.provider == "openai"
        assert config.topic_tree.llm_config.model == "gpt-4"
        assert config.topic_tree.llm_config.temperature == 0.7
        assert config.topic_tree.llm_config.max_tokens == 2000
        assert config.topic_tree.llm_config.max_retries == 4
        assert config.topic_tree.llm_config.request_timeout == 60

        # Verify topic_graph llm_config
        assert config.topic_graph is not None
        assert config.topic_graph.llm_config is not None
        assert config.topic_graph.llm_config.provider == "anthropic"
        assert config.topic_graph.llm_config.model == "claude-3"
        assert config.topic_graph.llm_config.max_tokens == 1500
        assert config.topic_graph.llm_config.max_retries == 5

        # Verify data_engine llm_config
        assert config.data_engine.llm_config is not None
        assert config.data_engine.llm_config.provider == "gemini"
        assert config.data_engine.llm_config.model == "gemini-pro"
        assert config.data_engine.llm_config.max_tokens == 3000
        assert config.data_engine.llm_config.max_retries == 6
    finally:
        _cleanup_temp_file(temp_path)


def test_load_hybrid_config(hybrid_config_dict):
    """Test loading hybrid config handles mixed styles correctly."""
    temp_path = _create_temp_yaml_file(hybrid_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # topic_tree uses old-style - should have auto-created llm_config
        assert config.topic_tree is not None
        assert config.topic_tree.llm_config is not None
        assert config.topic_tree.llm_config.provider == "openai"
        assert config.topic_tree.llm_config.model == "gpt-4"
        assert config.topic_tree.llm_config.temperature == 0.7

        # topic_graph uses new-style - should have explicit llm_config
        assert config.topic_graph is not None
        assert config.topic_graph.llm_config is not None
        assert config.topic_graph.llm_config.provider == "anthropic"
        assert config.topic_graph.llm_config.model == "claude-3"
        assert config.topic_graph.llm_config.temperature == 0.6
        assert config.topic_graph.llm_config.max_tokens == 1500

        # data_engine uses new-style - should have explicit llm_config
        assert config.data_engine.llm_config is not None
        assert config.data_engine.llm_config.provider == "gemini"
        assert config.data_engine.llm_config.model == "gemini-pro"
        assert config.data_engine.llm_config.max_tokens == 3000
    finally:
        _cleanup_temp_file(temp_path)


def test_conflicting_fields_llm_config_wins(conflicting_config_dict):
    """Test that llm_config takes precedence over individual fields."""
    temp_path = _create_temp_yaml_file(conflicting_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # llm_config values should win over individual fields
        assert config.topic_tree.llm_config.provider == "openai"  # NOT "ollama"
        assert config.topic_tree.llm_config.model == "gpt-4"  # NOT "mistral:latest"
        assert config.topic_tree.llm_config.temperature == 0.9  # NOT 0.5
        assert config.topic_tree.llm_config.max_tokens == 2000
        assert config.topic_tree.llm_config.max_retries == 5
    finally:
        _cleanup_temp_file(temp_path)


# ============================================================================
# PARAMETER EXTRACTION TESTS - TOPIC TREE
# ============================================================================


def test_get_topic_tree_params_old_style(old_style_config_dict):
    """Test extracting topic tree params from old-style config."""
    temp_path = _create_temp_yaml_file(old_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_tree_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Verify llm_config has correct values from individual fields
        assert params["llm_config"].provider == "openai"
        assert params["llm_config"].model == "gpt-4"
        assert params["llm_config"].temperature == 0.7

        # Verify individual fields are NOT in params
        assert "provider" not in params
        assert "model" not in params
        assert "temperature" not in params
    finally:
        _cleanup_temp_file(temp_path)


def test_get_topic_tree_params_new_style(new_style_config_dict):
    """Test extracting topic tree params from new-style config."""
    temp_path = _create_temp_yaml_file(new_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_tree_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Verify llm_config has correct values
        assert params["llm_config"].provider == "openai"
        assert params["llm_config"].model == "gpt-4"
        assert params["llm_config"].temperature == 0.7
        assert params["llm_config"].max_tokens == 2000
        assert params["llm_config"].max_retries == 4

        # Verify individual fields are NOT in params
        assert "provider" not in params
        assert "model" not in params
    finally:
        _cleanup_temp_file(temp_path)


def test_get_topic_tree_params_hybrid_style(hybrid_config_dict):
    """Test extracting topic tree params from hybrid config (tree uses old-style)."""
    temp_path = _create_temp_yaml_file(hybrid_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_tree_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Tree uses old-style, so values come from individual fields
        assert params["llm_config"].provider == "openai"
        assert params["llm_config"].model == "gpt-4"
        assert params["llm_config"].temperature == 0.7
    finally:
        _cleanup_temp_file(temp_path)


# ============================================================================
# PARAMETER EXTRACTION TESTS - TOPIC GRAPH
# ============================================================================


def test_get_topic_graph_params_old_style(old_style_config_dict):
    """Test extracting topic graph params from old-style config."""
    temp_path = _create_temp_yaml_file(old_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_graph_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Verify llm_config has correct values
        assert params["llm_config"].provider == "anthropic"
        assert params["llm_config"].model == "claude-3"
        assert params["llm_config"].temperature == 0.6

        # Verify individual fields are NOT in params
        assert "provider" not in params
        assert "model" not in params
    finally:
        _cleanup_temp_file(temp_path)


def test_get_topic_graph_params_new_style(new_style_config_dict):
    """Test extracting topic graph params from new-style config."""
    temp_path = _create_temp_yaml_file(new_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_graph_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Verify llm_config has correct values
        assert params["llm_config"].provider == "anthropic"
        assert params["llm_config"].model == "claude-3"
        assert params["llm_config"].max_tokens == 1500
        assert params["llm_config"].max_retries == 5
    finally:
        _cleanup_temp_file(temp_path)


def test_get_topic_graph_params_hybrid_style(hybrid_config_dict):
    """Test extracting topic graph params from hybrid config (graph uses new-style)."""
    temp_path = _create_temp_yaml_file(hybrid_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_graph_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Graph uses new-style, so values come from llm_config
        assert params["llm_config"].provider == "anthropic"
        assert params["llm_config"].model == "claude-3"
        assert params["llm_config"].temperature == 0.6
        assert params["llm_config"].max_tokens == 1500
    finally:
        _cleanup_temp_file(temp_path)


# ============================================================================
# PARAMETER EXTRACTION TESTS - DATA ENGINE
# ============================================================================


def test_get_engine_params_old_style(old_style_config_dict):
    """Test extracting engine params from old-style config."""
    temp_path = _create_temp_yaml_file(old_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_engine_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Verify llm_config has correct values
        assert params["llm_config"].provider == "gemini"
        assert params["llm_config"].model == "gemini-pro"
        assert params["llm_config"].temperature == 0.8
        assert params["llm_config"].max_retries == 3

        # Verify individual fields are NOT in params
        assert "provider" not in params
        assert "model" not in params
    finally:
        _cleanup_temp_file(temp_path)


def test_get_engine_params_new_style(new_style_config_dict):
    """Test extracting engine params from new-style config."""
    temp_path = _create_temp_yaml_file(new_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_engine_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Verify llm_config has correct values
        assert params["llm_config"].provider == "gemini"
        assert params["llm_config"].model == "gemini-pro"
        assert params["llm_config"].temperature == 0.8
        assert params["llm_config"].max_tokens == 3000
        assert params["llm_config"].max_retries == 6
    finally:
        _cleanup_temp_file(temp_path)


def test_get_engine_params_hybrid_style(hybrid_config_dict):
    """Test extracting engine params from hybrid config (engine uses new-style)."""
    temp_path = _create_temp_yaml_file(hybrid_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_engine_params()

        # Key assertion: llm_config must be in params
        assert "llm_config" in params
        assert isinstance(params["llm_config"], LLMProviderConfig)

        # Engine uses new-style, so values come from llm_config
        assert params["llm_config"].provider == "gemini"
        assert params["llm_config"].model == "gemini-pro"
        assert params["llm_config"].max_tokens == 3000
    finally:
        _cleanup_temp_file(temp_path)


# ============================================================================
# OVERRIDE TESTS
# ============================================================================


def test_cli_overrides_with_old_style_config(old_style_config_dict):
    """Test CLI overrides work with old-style config."""
    temp_path = _create_temp_yaml_file(old_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_tree_params(
            provider="override-provider",
            model="override-model",
            temperature=0.99,
        )

        # Overrides should be applied
        assert params["llm_config"].provider == "override-provider"
        assert params["llm_config"].model == "override-model"
        assert params["llm_config"].temperature == 0.99
    finally:
        _cleanup_temp_file(temp_path)


def test_cli_overrides_with_new_style_config(new_style_config_dict):
    """Test CLI overrides take precedence over llm_config."""
    temp_path = _create_temp_yaml_file(new_style_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)
        params = config.get_topic_tree_params(
            provider="override-provider",
            model="override-model",
            temperature=0.99,
        )

        # Overrides should take precedence over llm_config
        assert params["llm_config"].provider == "override-provider"
        assert params["llm_config"].model == "override-model"
        assert params["llm_config"].temperature == 0.99

        # Non-overridden values should come from llm_config
        assert params["llm_config"].max_tokens == 2000
        assert params["llm_config"].max_retries == 4
    finally:
        _cleanup_temp_file(temp_path)


def test_cli_overrides_with_hybrid_config(hybrid_config_dict):
    """Test CLI overrides work consistently with hybrid config."""
    temp_path = _create_temp_yaml_file(hybrid_config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # Override tree params (tree uses old-style)
        tree_params = config.get_topic_tree_params(
            provider="tree-override",
            model="tree-model",
        )
        assert tree_params["llm_config"].provider == "tree-override"
        assert tree_params["llm_config"].model == "tree-model"

        # Override graph params (graph uses new-style)
        graph_params = config.get_topic_graph_params(
            provider="graph-override",
            model="graph-model",
        )
        assert graph_params["llm_config"].provider == "graph-override"
        assert graph_params["llm_config"].model == "graph-model"
    finally:
        _cleanup_temp_file(temp_path)


# ============================================================================
# VALIDATION FAILURE TESTS
# ============================================================================


def test_invalid_temperature_too_high_fails():
    """Test that temperature > 2.0 fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "topic_tree": {
            "topic_prompt": "Test",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 5.0,  # Invalid: > 2.0
            },
        },
        "data_engine": {
            "generation_system_prompt": "Test",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


def test_invalid_temperature_negative_fails():
    """Test that negative temperature fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "topic_tree": {
            "topic_prompt": "Test",
            "provider": "openai",
            "model": "gpt-4",
            "temperature": -0.5,  # Invalid: < 0.0
        },
        "data_engine": {
            "generation_system_prompt": "Test",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


def test_invalid_max_retries_too_high_fails():
    """Test that max_retries > 10 fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "data_engine": {
            "generation_system_prompt": "Test",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "max_retries": 20,  # Invalid: > 10
            },
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


def test_empty_provider_fails():
    """Test that empty provider string fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "topic_tree": {
            "topic_prompt": "Test",
            "llm_config": {
                "provider": "",  # Invalid: min_length=1
                "model": "gpt-4",
            },
        },
        "data_engine": {
            "generation_system_prompt": "Test",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


def test_empty_model_fails():
    """Test that empty model string fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "topic_tree": {
            "topic_prompt": "Test",
            "llm_config": {
                "provider": "openai",
                "model": "",  # Invalid: min_length=1
            },
        },
        "data_engine": {
            "generation_system_prompt": "Test",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


def test_invalid_max_tokens_zero_fails():
    """Test that max_tokens = 0 fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "data_engine": {
            "generation_system_prompt": "Test",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "max_tokens": 0,  # Invalid: must be >= 1
            },
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


def test_invalid_request_timeout_too_low_fails():
    """Test that request_timeout < 5 fails validation."""
    config_dict = {
        "dataset_system_prompt": "Test",
        "data_engine": {
            "generation_system_prompt": "Test",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                "request_timeout": 2,  # Invalid: must be >= 5
            },
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "test.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        _cleanup_temp_file(temp_path)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_all_components_different_styles():
    """Test config with all components using different style combinations."""
    config_dict = {
        "dataset_system_prompt": "Edge case test",
        "topic_tree": {
            "topic_prompt": "Tree prompt",
            # Old-style
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
        },
        "topic_graph": {
            "topic_prompt": "Graph prompt",
            # New-style
            "llm_config": {
                "provider": "anthropic",
                "model": "claude-3",
                "temperature": 0.6,
            },
        },
        "data_engine": {
            "generation_system_prompt": "Engine prompt",
            # Hybrid: both (llm_config should win)
            "provider": "test",
            "model": "test-model",
            "llm_config": {
                "provider": "gemini",
                "model": "gemini-pro",
                "temperature": 0.8,
            },
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "edge.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # Verify all components have llm_config
        assert config.topic_tree.llm_config.provider == "openai"
        assert config.topic_graph.llm_config.provider == "anthropic"
        assert config.data_engine.llm_config.provider == "gemini"  # llm_config wins

        # Verify params extraction works
        tree_params = config.get_topic_tree_params()
        graph_params = config.get_topic_graph_params()
        engine_params = config.get_engine_params()

        assert "llm_config" in tree_params
        assert "llm_config" in graph_params
        assert "llm_config" in engine_params
    finally:
        _cleanup_temp_file(temp_path)


def test_partial_llm_config_uses_defaults():
    """Test that partial llm_config fills in missing fields with defaults."""
    config_dict = {
        "dataset_system_prompt": "Partial test",
        "topic_tree": {
            "topic_prompt": "Tree prompt",
            "llm_config": {
                "provider": "openai",
                "model": "gpt-4",
                # Missing: temperature, max_tokens, max_retries, request_timeout
            },
        },
        "data_engine": {
            "generation_system_prompt": "Engine prompt",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "partial.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        # Verify defaults are filled in
        assert config.topic_tree.llm_config.provider == "openai"
        assert config.topic_tree.llm_config.model == "gpt-4"
        assert config.topic_tree.llm_config.temperature == 0.7  # Default
        assert config.topic_tree.llm_config.max_tokens == 1000  # Default
        assert config.topic_tree.llm_config.max_retries == 3  # Default
        assert config.topic_tree.llm_config.request_timeout == 30  # Default
    finally:
        _cleanup_temp_file(temp_path)


def test_graph_config_independent_from_tree():
    """Test that topic_tree and topic_graph configs don't contaminate each other."""
    config_dict = {
        "dataset_system_prompt": "Independence test",
        "topic_tree": {
            "topic_prompt": "Tree prompt",
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
        },
        "topic_graph": {
            "topic_prompt": "Graph prompt",
            "provider": "anthropic",
            "model": "claude-3",
            "temperature": 0.6,
        },
        "data_engine": {
            "generation_system_prompt": "Engine prompt",
        },
        "dataset": {
            "creation": {"num_steps": 1, "batch_size": 1},
            "save_as": "independence.jsonl",
        },
    }

    temp_path = _create_temp_yaml_file(config_dict)

    try:
        config = DeepFabricConfig.from_yaml(temp_path)

        tree_params = config.get_topic_tree_params()
        graph_params = config.get_topic_graph_params()

        # Verify they have different configs
        assert tree_params["llm_config"].provider == "openai"
        assert graph_params["llm_config"].provider == "anthropic"
        assert tree_params["llm_config"].model == "gpt-4"
        assert graph_params["llm_config"].model == "claude-3"
        assert tree_params["llm_config"].temperature == 0.7
        assert graph_params["llm_config"].temperature == 0.6
    finally:
        _cleanup_temp_file(temp_path)
