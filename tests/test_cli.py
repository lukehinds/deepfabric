"""Tests for the CLI module."""

import os
import tempfile

from unittest.mock import Mock, patch

import pytest

from click.testing import CliRunner

from promptwright.cli import cli


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing."""
    return CliRunner()


@pytest.fixture
def sample_yaml_content():
    """Sample YAML content for testing."""
    return """
system_prompt: "Test system prompt"
topic_tree:
  args:
    root_prompt: "Test root prompt"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 3
    tree_depth: 2
    temperature: 0.7
    provider: "test"
    model: "model"
  save_as: "test_tree.jsonl"
data_engine:
  args:
    instructions: "Test instructions"
    system_prompt: "<system_prompt_placeholder>"
    provider: "test"
    model: "model"
    temperature: 0.9
    max_retries: 2
dataset:
  creation:
    num_steps: 5
    batch_size: 1
    provider: "test"
    model: "model"
    sys_msg: true
  save_as: "test_dataset.jsonl"
"""


@pytest.fixture
def sample_yaml_content_no_sys_msg():
    """Sample YAML content without sys_msg setting."""
    return """
system_prompt: "Test system prompt"
topic_tree:
  args:
    root_prompt: "Test root prompt"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 3
    tree_depth: 2
    temperature: 0.7
    provider: "test"
    model: "model"
  save_as: "test_tree.jsonl"
data_engine:
  args:
    instructions: "Test instructions"
    system_prompt: "<system_prompt_placeholder>"
    provider: "test"
    model: "model"
    temperature: 0.9
    max_retries: 2
dataset:
  creation:
    num_steps: 5
    batch_size: 1
    provider: "test"
    model: "model"
  save_as: "test_dataset.jsonl"
"""


@pytest.fixture
def sample_config_file(sample_yaml_content):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_config_file_no_sys_msg(sample_yaml_content_no_sys_msg):
    """Create a temporary config file without sys_msg setting."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(sample_yaml_content_no_sys_msg)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


def test_cli_help(cli_runner):
    """Test CLI help command."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "PromptWright CLI" in result.output


def test_start_help(cli_runner):
    """Test start command help."""
    result = cli_runner.invoke(cli, ["start", "--help"])
    assert result.exit_code == 0
    assert "Generate training data from a YAML configuration file" in result.output
    assert "--sys-msg" in result.output


@patch("promptwright.cli.create_topic_generator")
@patch("promptwright.cli.DataEngine")
def test_start_command_basic(
    mock_data_engine, mock_create_topic_generator, cli_runner, sample_config_file
):
    """Test basic start command execution."""
    # Setup mocks
    from promptwright.topic_tree import TopicTree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=TopicTree)
    mock_engine_instance = Mock()
    mock_dataset = Mock()

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data.return_value = mock_dataset

    # Run command
    result = cli_runner.invoke(cli, ["start", sample_config_file])

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify mocks were called correctly
    mock_create_topic_generator.assert_called_once()
    mock_tree_instance.build.assert_called_once()
    mock_tree_instance.save.assert_called_once()
    mock_data_engine.assert_called_once()
    mock_engine_instance.create_data.assert_called_once()
    mock_dataset.save.assert_called_once()


@patch("promptwright.cli.create_topic_generator")
@patch("promptwright.cli.DataEngine")
def test_start_command_with_sys_msg_override(
    mock_data_engine, mock_create_topic_generator, cli_runner, sample_config_file
):
    """Test start command with sys_msg override."""
    # Setup mocks
    from promptwright.topic_tree import TopicTree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=TopicTree)
    mock_engine_instance = Mock()
    mock_dataset = Mock()

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data.return_value = mock_dataset

    # Run command with sys_msg override
    result = cli_runner.invoke(
        cli,
        [
            "start",
            sample_config_file,
            "--sys-msg",
            "false",
        ],
    )

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify create_data was called with sys_msg=False
    args, kwargs = mock_engine_instance.create_data.call_args
    assert kwargs["sys_msg"] is False


@patch("promptwright.cli.create_topic_generator")
@patch("promptwright.cli.DataEngine")
def test_start_command_default_sys_msg(
    mock_data_engine, mock_create_topic_generator, cli_runner, sample_config_file_no_sys_msg
):
    """Test start command with default sys_msg behavior."""
    # Setup mocks
    from promptwright.topic_tree import TopicTree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=TopicTree)
    mock_engine_instance = Mock()
    mock_dataset = Mock()

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data.return_value = mock_dataset

    # Run command without sys_msg override
    result = cli_runner.invoke(cli, ["start", sample_config_file_no_sys_msg])

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify create_data was called with default sys_msg (should be None to use engine default)
    args, kwargs = mock_engine_instance.create_data.call_args
    assert "sys_msg" not in kwargs or kwargs["sys_msg"] is None


@patch("promptwright.cli.create_topic_generator")
@patch("promptwright.cli.DataEngine")
def test_start_command_with_overrides(
    mock_data_engine, mock_create_topic_generator, cli_runner, sample_config_file
):
    """Test start command with parameter overrides."""
    # Setup mocks
    from promptwright.topic_tree import TopicTree  # noqa: PLC0415

    mock_tree_instance = Mock(spec=TopicTree)
    mock_engine_instance = Mock()
    mock_dataset = Mock()

    mock_create_topic_generator.return_value = mock_tree_instance
    mock_data_engine.return_value = mock_engine_instance
    mock_engine_instance.create_data.return_value = mock_dataset

    # Run command with overrides
    result = cli_runner.invoke(
        cli,
        [
            "start",
            sample_config_file,
            "--topic-tree-save-as",
            "override_tree.jsonl",
            "--dataset-save-as",
            "override_dataset.jsonl",
            "--provider",
            "override",
            "--model",
            "model",
            "--temperature",
            "0.5",
            "--tree-degree",
            "4",
            "--tree-depth",
            "3",
            "--num-steps",
            "10",
            "--batch-size",
            "2",
            "--sys-msg",
            "false",
        ],
    )

    # Verify command executed successfully
    assert result.exit_code == 0

    # Verify mocks were called
    mock_create_topic_generator.assert_called_once()

    mock_tree_instance.save.assert_called_once_with("override_tree.jsonl")
    mock_dataset.save.assert_called_once_with("override_dataset.jsonl")

    args, kwargs = mock_engine_instance.create_data.call_args
    assert kwargs["num_steps"] == 10  # noqa: PLR2004
    assert kwargs["batch_size"] == 2  # noqa: PLR2004
    assert kwargs["model_name"] == "override/model"
    assert kwargs["sys_msg"] is False


@patch("promptwright.cli.read_topic_tree_from_jsonl")
@patch("promptwright.cli.TopicTree")
@patch("promptwright.cli.DataEngine")
def test_start_command_with_jsonl(
    mock_data_engine,
    mock_topic_tree,
    mock_read_topic_tree_from_jsonl,
    cli_runner,
    sample_config_file,
):
    """Test start command with JSONL file."""
    mock_tree_instance = Mock()
    mock_tree_instance.build = Mock()  # Add build method
    mock_topic_tree.return_value = mock_tree_instance
    mock_read_topic_tree_from_jsonl.return_value = [{"path": ["root", "child"]}]

    mock_engine_instance = Mock()
    mock_data_engine.return_value = mock_engine_instance
    mock_dataset = Mock()
    mock_engine_instance.create_data.return_value = mock_dataset
    # Create a temporary JSONL file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"path": ["root", "child"]}\n')
        temp_jsonl_path = f.name

    try:
        # Run command with JSONL file
        result = cli_runner.invoke(
            cli,
            ["start", sample_config_file, "--topic-tree-jsonl", temp_jsonl_path],
        )

        # Print output if command fails
        if result.exit_code != 0:
            print(result.output)

        # Verify command executed successfully
        assert result.exit_code == 0

        # Verify JSONL read function was called
        mock_read_topic_tree_from_jsonl.assert_called_once_with(temp_jsonl_path)

        # Verify from_dict_list was called with the correct data
        mock_tree_instance.from_dict_list.assert_called_once_with([{"path": ["root", "child"]}])

        # Verify save was not called since JSONL file was provided
        mock_tree_instance.save.assert_not_called()

    finally:
        # Cleanup the temporary JSONL file
        if os.path.exists(temp_jsonl_path):
            os.unlink(temp_jsonl_path)


def test_start_command_missing_config(cli_runner):
    """Test start command with missing config file."""
    result = cli_runner.invoke(cli, ["start", "nonexistent.yaml"])
    assert result.exit_code != 0
    assert "Error" in result.output


def test_start_command_invalid_yaml(cli_runner):
    """Test start command with invalid YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content:")
        temp_path = f.name

    try:
        result = cli_runner.invoke(cli, ["start", temp_path])
        assert result.exit_code != 0
        assert "Error" in result.output
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


@patch("promptwright.cli.create_topic_generator")
@patch("promptwright.cli.DataEngine")
def test_start_command_error_handling(
    _mock_data_engine,
    mock_create_topic_generator,
    cli_runner,
    sample_config_file,  # noqa: ARG001
):
    """Test error handling in start command."""
    # Setup mock to raise an exception
    mock_create_topic_generator.side_effect = Exception("Test error")

    # Run command
    result = cli_runner.invoke(cli, ["start", sample_config_file])

    # Verify command failed with error
    assert result.exit_code != 0
    assert "Error" in result.output
    assert "Test error" in result.output
