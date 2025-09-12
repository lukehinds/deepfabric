from deepfabric import Dataset


def test_dataset_initialization():
    """Test Dataset class initialization."""
    dataset = Dataset()

    assert len(dataset) == 0
    assert dataset.samples == []


def test_dataset_validation():
    """Test sample validation."""

    valid_sample = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
    }

    invalid_sample = {"messages": [{"role": "invalid", "content": "test"}]}

    assert Dataset.validate_sample(valid_sample) is True
    assert Dataset.validate_sample(invalid_sample) is False


def test_dataset_validation_with_system_message():
    """Test sample validation with system message."""

    valid_sample = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
    }

    assert Dataset.validate_sample(valid_sample) is True


def test_dataset_validation_system_message_order():
    """Test sample validation with system message in different positions."""

    # System message should be valid in any position
    valid_sample_start = {
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
    }

    valid_sample_middle = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "system", "content": "system prompt"},
            {"role": "assistant", "content": "response"},
        ]
    }

    valid_sample_end = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
            {"role": "system", "content": "system prompt"},
        ]
    }

    assert Dataset.validate_sample(valid_sample_start) is True
    assert Dataset.validate_sample(valid_sample_middle) is True
    assert Dataset.validate_sample(valid_sample_end) is True


def test_dataset_validation_multiple_system_messages():
    """Test sample validation with multiple system messages."""

    # Multiple system messages should be valid
    valid_sample = {
        "messages": [
            {"role": "system", "content": "system prompt 1"},
            {"role": "system", "content": "system prompt 2"},
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]
    }

    assert Dataset.validate_sample(valid_sample) is True


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
