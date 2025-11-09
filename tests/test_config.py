"""Tests for the configuration module."""

import os
import tempfile

import pytest
import yaml

from deepfabric.config import DeepFabricConfig
from deepfabric.exceptions import ConfigurationError


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "dataset_system_prompt": "Test system prompt",
        "topic_tree": {
            "topic_prompt": "Test root prompt",
            "topic_system_prompt": "Test system prompt",
            "degree": 3,
            "depth": 2,
            "temperature": 0.7,
            "provider": "test",
            "model": "model",
            "save_as": "test_tree.jsonl",
        },
        "data_engine": {
            "instructions": "Test instructions",
            "generation_system_prompt": "Test system prompt",
            "provider": "test",
            "model": "model",
            "temperature": 0.9,
            "max_retries": 2,
            "conversation_type": "basic",
        },
        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 1,
                "provider": "test",
                "model": "model",
                "sys_msg": True,
            },
            "save_as": "test_dataset.jsonl",
            "formatters": [],
        },
    }


@pytest.fixture
def sample_config_dict_no_sys_msg():
    """Sample configuration dictionary without sys_msg setting."""
    return {
        "dataset_system_prompt": "Test system prompt",
        "topic_tree": {
            "topic_prompt": "Test root prompt",
            "topic_system_prompt": "Test system prompt",
            "degree": 3,
            "depth": 2,
            "temperature": 0.7,
            "provider": "test",
            "model": "model",
            "save_as": "test_tree.jsonl",
        },
        "data_engine": {
            "instructions": "Test instructions",
            "generation_system_prompt": "Test system prompt",
            "provider": "test",
            "model": "model",
            "temperature": 0.9,
            "max_retries": 2,
            "conversation_type": "basic",
        },
        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 1,
                "provider": "test",
                "model": "model",
            },
            "save_as": "test_dataset.jsonl",
            "formatters": [],
        },
    }


@pytest.fixture
def sample_yaml_file(sample_config_dict):
    """Create a temporary YAML file with sample configuration."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_yaml_file_no_sys_msg(sample_config_dict_no_sys_msg):
    """Create a temporary YAML file without sys_msg setting."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(sample_config_dict_no_sys_msg, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_load_from_yaml(sample_yaml_file, sample_config_dict):
    """Test loading configuration from YAML file."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)

    assert config.dataset_system_prompt == sample_config_dict["dataset_system_prompt"]
    assert config.topic_tree.model_dump(exclude_none=True) == sample_config_dict["topic_tree"]

    # Add the new fields to the expected config for comparison
    expected_data_engine = sample_config_dict["data_engine"].copy()
    expected_data_engine["available_tools"] = []  # Default value
    expected_data_engine["custom_tools"] = []  # Default value
    expected_data_engine["max_tools_per_query"] = 3  # Default value
    expected_data_engine["max_tools_strict"] = True  # Default value
    expected_data_engine["max_tokens"] = 2000

    actual_data_engine = config.data_engine.model_dump(exclude_none=True)
    assert actual_data_engine == expected_data_engine
    assert config.dataset.model_dump(exclude_none=True) == sample_config_dict["dataset"]


def test_get_topic_tree_args(sample_yaml_file):
    """Test getting tree arguments from config."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_topic_tree_params()

    assert isinstance(args, dict)
    assert args["topic_prompt"] == "Test root prompt"
    assert args["topic_system_prompt"] == "Test system prompt"
    assert args["degree"] == 3  # noqa: PLR2004
    assert args["depth"] == 2  # noqa: PLR2004
    assert args["temperature"] == 0.7  # noqa: PLR2004
    assert args["provider"] == "test"
    assert args["model_name"] == "model"


def test_get_engine_args(sample_yaml_file):
    """Test getting engine arguments from config."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_engine_params()

    assert isinstance(args, dict)
    assert args["instructions"] == "Test instructions"
    assert args["generation_system_prompt"] == "Test system prompt"
    assert args["model_name"] == "model"
    assert args["provider"] == "test"
    assert args["temperature"] == 0.9  # noqa: PLR2004
    assert args["max_retries"] == 2  # noqa: PLR2004
    assert args["sys_msg"] is True  # Default from dataset config


def test_get_engine_args_no_sys_msg(sample_yaml_file_no_sys_msg):
    """Test getting engine arguments without sys_msg setting."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file_no_sys_msg)
    args = config.get_engine_params()

    assert isinstance(args, dict)
    assert args["sys_msg"] is True  # Default value when not specified


def test_get_topic_tree_args_with_overrides(sample_yaml_file):
    """Test getting tree arguments with overrides."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_topic_tree_params(
        provider="override",
        model="model",
        temperature=0.5,
    )

    assert args["model_name"] == "model"
    assert args["temperature"] == 0.5  # noqa: PLR2004


def test_get_engine_args_with_overrides(sample_yaml_file):
    """Test getting engine arguments with overrides."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    args = config.get_engine_params(
        provider="override",
        model="model",
        temperature=0.5,
    )

    assert args["model_name"] == "model"
    assert args["temperature"] == 0.5  # noqa: PLR2004


def test_get_dataset_config(sample_yaml_file, sample_config_dict):
    """Test getting dataset configuration."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file)
    dataset_config = config.get_dataset_config()

    assert dataset_config == sample_config_dict["dataset"]
    assert dataset_config["creation"]["sys_msg"] is True


def test_get_dataset_config_no_sys_msg(sample_yaml_file_no_sys_msg):
    """Test getting dataset configuration without sys_msg setting."""
    config = DeepFabricConfig.from_yaml(sample_yaml_file_no_sys_msg)
    dataset_config = config.get_dataset_config()

    assert "sys_msg" not in dataset_config["creation"]


def test_missing_yaml_file():
    """Test handling of missing YAML file."""
    with pytest.raises(ConfigurationError):
        DeepFabricConfig.from_yaml("nonexistent.yaml")


def test_invalid_yaml_content():
    """Test handling of invalid YAML content."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_path = f.name

    try:
        with pytest.raises(ConfigurationError):
            DeepFabricConfig.from_yaml(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
