from deepfabric import Dataset


def test_dataset_initialization():
    """Test Dataset class initialization."""
    dataset = Dataset()

    assert len(dataset) == 0
    assert dataset.samples == []


def test_dataset_add_samples():
    """Test adding samples to dataset."""

    dataset = Dataset()

    samples = [
        {
            "messages": [
                {"role": "user", "content": "test1"},
                {"role": "assistant", "content": "response1"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "test2"},
                {"role": "assistant", "content": "response2"},
            ]
        },
    ]

    dataset.add_samples(samples)
    assert len(dataset) == 2  # noqa: PLR2004
    assert dataset[0] == samples[0]


def test_dataset_add_samples_with_system_messages():
    """Test adding samples with system messages to dataset."""

    dataset = Dataset()

    samples = [
        {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "test1"},
                {"role": "assistant", "content": "response1"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "test2"},
                {"role": "assistant", "content": "response2"},
            ]
        },
    ]

    dataset.add_samples(samples)
    assert len(dataset) == 2  # noqa: PLR2004
    assert dataset[0] == samples[0]
    assert dataset[0]["messages"][0]["role"] == "system"


def test_dataset_filter_by_role():
    """Test filtering samples by role."""

    dataset = Dataset()

    samples = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "test1"},
                {"role": "assistant", "content": "response1"},
            ]
        }
    ]

    dataset.add_samples(samples)
    user_messages = dataset.filter_by_role("user")
    assert len(user_messages) == 1
    assert user_messages[0]["messages"][0]["content"] == "test1"

    system_messages = dataset.filter_by_role("system")
    assert len(system_messages) == 1
    assert system_messages[0]["messages"][0]["content"] == "sys"


def test_dataset_get_statistics():
    """Test getting dataset statistics."""

    dataset = Dataset()

    samples = [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "test1"},
                {"role": "assistant", "content": "response1"},
            ]
        }
    ]

    dataset.add_samples(samples)
    stats = dataset.get_statistics()

    assert stats["total_samples"] == 1
    assert stats["avg_messages_per_sample"] == 3  # noqa: PLR2004
    assert "system" in stats["role_distribution"]
    assert stats["role_distribution"]["system"] == 1 / 3  # noqa: PLR2004
