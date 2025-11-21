import pytest

from datasets import Dataset as HFDataset
from pydantic import ValidationError

from deepfabric import Dataset
from deepfabric.schemas import ChatMessage, Conversation


def test_dataset_initialization():
    """Test Dataset class initialization."""
    dataset = Dataset()

    assert len(dataset) == 0
    assert dataset.samples == []


def test_dataset_add_samples():
    """Test adding samples to dataset."""

    dataset = Dataset()

    samples = [
        Conversation(
            messages=[
                ChatMessage(role="user", content="test1"),
                ChatMessage(role="assistant", content="response1"),
            ]
        ),
        Conversation(
            messages=[
                ChatMessage(role="user", content="test2"),
                ChatMessage(role="assistant", content="response2"),
            ]
        ),
    ]

    dataset.add_samples(samples)
    assert len(dataset) == 2  # noqa: PLR2004
    assert dataset[0].messages[0].content == "test1"  # type: ignore
    assert dataset[0].messages[1].content == "response1"  # type: ignore


def test_dataset_add_samples_with_system_messages():
    """Test adding samples with system messages to dataset."""

    dataset = Dataset()

    samples = [
        Conversation(
            messages=[
                ChatMessage(role="system", content="system prompt"),
                ChatMessage(role="user", content="test1"),
                ChatMessage(role="assistant", content="response1"),
            ]
        ),
        Conversation(
            messages=[
                ChatMessage(role="system", content="system prompt"),
                ChatMessage(role="user", content="test2"),
                ChatMessage(role="assistant", content="response2"),
            ]
        ),
    ]

    dataset.add_samples(samples)
    assert len(dataset) == 2  # noqa: PLR2004
    assert dataset[0].messages[0].role == "system"  # type: ignore
    assert dataset[0].messages[0].content == "system prompt"  # type: ignore


def test_dataset_filter_by_role():
    """Test filtering samples by role."""

    dataset = Dataset()

    samples = [
        Conversation(
            messages=[
                ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="test1"),
                ChatMessage(role="assistant", content="response1"),
            ]
        )
    ]

    dataset.add_samples(samples)
    user_messages = dataset.filter_by_role("user")
    assert len(user_messages) == 1
    assert user_messages[0].messages[0].content == "test1"

    system_messages = dataset.filter_by_role("system")
    assert len(system_messages) == 1
    assert system_messages[0].messages[0].content == "sys"


def test_dataset_get_statistics():
    """Test getting dataset statistics."""

    dataset = Dataset()

    samples = [
        Conversation(
            messages=[
                ChatMessage(role="system", content="sys"),
                ChatMessage(role="user", content="test1"),
                ChatMessage(role="assistant", content="response1"),
            ]
        )
    ]

    dataset.add_samples(samples)
    stats = dataset.get_statistics()

    assert stats["total_samples"] == 1
    assert stats["avg_messages_per_sample"] == 3  # noqa: PLR2004
    assert "system" in stats["role_distribution"]
    assert stats["role_distribution"]["system"] == 1 / 3  # noqa: PLR2004


def test_dataset_to_hf_dataset():
    """Test converting to HuggingFace Dataset."""
    dataset = Dataset()

    samples = [
        Conversation(
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ]
        ),
    ]

    dataset.add_samples(samples)
    hf_dataset = dataset.to_hf_dataset()

    assert isinstance(hf_dataset, HFDataset)
    assert len(hf_dataset) == 1
    assert "messages" in hf_dataset[0]
    assert hf_dataset[0]["messages"][0]["role"] == "user"


def test_dataset_from_list_validation():
    """Test that from_list validates data."""
    # Valid data should work
    valid_samples = [
        {"messages": [{"role": "user", "content": "test"}]},
    ]
    dataset = Dataset.from_list(valid_samples)
    assert len(dataset) == 1

    # Invalid data should raise ValidationError
    invalid_samples = [
        {"messages": "not a list"},  # messages must be a list
    ]
    with pytest.raises(ValidationError):
        Dataset.from_list(invalid_samples)
