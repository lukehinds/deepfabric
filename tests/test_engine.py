from unittest.mock import MagicMock, patch

import pytest

from promptwright.engine import DataEngine, Dataset, EngineArguments
from promptwright.exceptions import DataEngineError


@pytest.fixture
def engine_args():
    return EngineArguments(
        instructions="Test instructions",
        system_prompt="Test system prompt",
        model_name="test-model",
        prompt_template=None,
        example_data=None,
        temperature=0.7,
        max_retries=3,
        default_batch_size=5,
        default_num_examples=3,
        request_timeout=30,
        sys_msg=True,
    )


@pytest.fixture
def data_engine(engine_args):
    return DataEngine(engine_args)


def test_engine_initialization(engine_args):
    engine = DataEngine(engine_args)
    assert engine.args == engine_args
    assert isinstance(engine.dataset, Dataset)
    assert engine.failed_samples == []


def test_create_data_no_steps(data_engine):
    with pytest.raises(DataEngineError, match="positive"):
        data_engine.create_data(num_steps=0)


@patch("promptwright.engine.litellm.batch_completion")
def test_create_data_success(mock_batch_completion, data_engine):
    # Mock valid JSON responses to match the expected structure for 10 samples
    mock_batch_completion.return_value = [
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"messages": [{"role": "user", "content": "example"}, {"role": "assistant", "content": "response"}]}'
                    )
                )
            ]
        )
    ] * 10  # Mock 10 responses to match the batch size

    topic_tree = MagicMock()
    topic_tree.tree_paths = [
        "path1",
        "path2",
        "path3",
        "path4",
        "path5",
        "path6",
        "path7",
        "path8",
        "path9",
        "path10",
    ]

    # Define a constant for the expected number of samples
    expected_num_samples = 10

    # Generate the data
    dataset = data_engine.create_data(num_steps=1, batch_size=10, topic_tree=topic_tree)

    # Assert that the dataset contains exactly the expected number of samples
    assert len(dataset.samples) == expected_num_samples


@patch("promptwright.engine.litellm.batch_completion")
def test_create_data_with_sys_msg_default(mock_batch_completion, data_engine):
    # Mock valid JSON response
    mock_batch_completion.return_value = [
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"messages": [{"role": "user", "content": "example"}, {"role": "assistant", "content": "response"}]}'
                    )
                )
            ]
        )
    ]

    topic_tree = MagicMock()
    topic_tree.tree_paths = ["path1"]

    # Generate data with default sys_msg (True)
    dataset = data_engine.create_data(num_steps=1, batch_size=1, topic_tree=topic_tree)

    # Verify system message is included
    assert len(dataset.samples) == 1
    assert len(dataset.samples[0]["messages"]) == 3  # noqa: PLR2004
    assert dataset.samples[0]["messages"][0]["role"] == "system"
    assert dataset.samples[0]["messages"][0]["content"] == data_engine.args.system_prompt


@patch("promptwright.engine.litellm.batch_completion")
def test_create_data_without_sys_msg(mock_batch_completion, data_engine):
    # Mock valid JSON response
    mock_batch_completion.return_value = [
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"messages": [{"role": "user", "content": "example"}, {"role": "assistant", "content": "response"}]}'
                    )
                )
            ]
        )
    ]

    topic_tree = MagicMock()
    topic_tree.tree_paths = ["path1"]

    # Generate data with sys_msg=False
    dataset = data_engine.create_data(
        num_steps=1, batch_size=1, topic_tree=topic_tree, sys_msg=False
    )

    # Verify system message is not included
    assert len(dataset.samples) == 1
    assert len(dataset.samples[0]["messages"]) == 2  # noqa: PLR2004
    assert dataset.samples[0]["messages"][0]["role"] == "user"


@patch("promptwright.engine.litellm.batch_completion")
def test_create_data_sys_msg_override(mock_batch_completion):
    # Create engine with sys_msg=False
    args = EngineArguments(
        instructions="Test instructions",
        system_prompt="Test system prompt",
        model_name="test-model",
        prompt_template=None,
        example_data=None,
        temperature=0.7,
        max_retries=3,
        default_batch_size=5,
        default_num_examples=3,
        request_timeout=30,
        sys_msg=False,  # Default to False
    )
    engine = DataEngine(args)

    # Mock valid JSON response
    mock_batch_completion.return_value = [
        MagicMock(
            choices=[
                MagicMock(
                    message=MagicMock(
                        content='{"messages": [{"role": "user", "content": "example"}, {"role": "assistant", "content": "response"}]}'
                    )
                )
            ]
        )
    ]

    topic_tree = MagicMock()
    topic_tree.tree_paths = ["path1"]

    # Override sys_msg=False with True in create_data
    dataset = engine.create_data(num_steps=1, batch_size=1, topic_tree=topic_tree, sys_msg=True)

    # Verify system message is included despite engine default
    assert len(dataset.samples) == 1
    assert len(dataset.samples[0]["messages"]) == 3  # noqa: PLR2004
    assert dataset.samples[0]["messages"][0]["role"] == "system"


def test_build_prompt(data_engine):
    prompt = data_engine.build_prompt("Test prompt", 3, ["subtopic1", "subtopic2"])
    assert "{{system_prompt}}" not in prompt
    assert "{{instructions}}" not in prompt
    assert "{{examples}}" not in prompt
    assert "{{subtopics}}" not in prompt


def test_build_system_prompt(data_engine):
    system_prompt = data_engine.build_system_prompt()
    assert system_prompt == data_engine.args.system_prompt


def test_build_custom_instructions_text(data_engine):
    instructions_text = data_engine.build_custom_instructions_text()
    assert "<instructions>" in instructions_text
    assert data_engine.args.instructions in instructions_text


def test_build_examples_text_no_examples(data_engine):
    examples_text = data_engine.build_examples_text(3)
    assert examples_text == ""


def test_build_subtopics_text(data_engine):
    subtopics_text = data_engine.build_subtopics_text(["subtopic1", "subtopic2"])
    assert "subtopic1 -> subtopic2" in subtopics_text


@patch.object(Dataset, "save")
def test_save_dataset(mock_save, data_engine):
    data_engine.save_dataset("test_path.jsonl")
    mock_save.assert_called_once_with("test_path.jsonl")
