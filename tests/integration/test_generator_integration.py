"""
Integration tests for DataSetGenerator.
"""

import os

import pytest

from deepfabric import Dataset, DataSetGenerator, Tree
from deepfabric.exceptions import DataSetGeneratorError

MESSAGE_LIMIT = 2


class TestDataSetGeneratorIntegration:
    """Integration tests for DataSetGenerator functionality."""

    def test_generator_creation_basic(self, minimal_test_config):
        """Test basic generator creation with minimal parameters."""
        generator = DataSetGenerator(
            instructions="Create simple Python Q&A pairs",
            generation_system_prompt="You are a Python tutor.",
            **minimal_test_config,
        )

        assert generator.config.instructions == "Create simple Python Q&A pairs"
        assert generator.config.generation_system_prompt == "You are a Python tutor."
        assert isinstance(generator.dataset, Dataset)

    def test_generator_with_tree(self, minimal_test_config):
        """Test generator creating data from a tree."""
        # Create a simple tree
        tree = Tree(
            topic_prompt="Python basics",
            topic_system_prompt="You are a Python educator.",
            degree=2,
            depth=1,  # Keep small
            **minimal_test_config,
        )

        # Build the tree
        list(tree.build())

        # Create generator
        generator = DataSetGenerator(
            instructions="Create beginner Python questions and answers",
            generation_system_prompt="You are a Python programming tutor.",
            **minimal_test_config,
        )

        # Generate small dataset
        dataset = generator.create_data(
            num_steps=2,
            batch_size=1,
            topic_model=tree,
        )

        # Verify dataset structure
        assert isinstance(dataset, Dataset)
        assert len(dataset.samples) > 0

        # Verify sample format
        for sample in dataset.samples:
            assert "messages" in sample
            messages = sample["messages"]
            assert len(messages) >= MESSAGE_LIMIT  # Should have user and assistant
            assert any(msg["role"] == "user" for msg in messages)
            assert any(msg["role"] == "assistant" for msg in messages)

    def test_generator_conversation_types(self, minimal_test_config):
        """Test different conversation types."""
        # Create minimal tree
        tree = Tree(
            topic_prompt="Math concepts",
            degree=2,
            depth=1,
            **minimal_test_config,
        )
        list(tree.build())

        # Test different conversation types
        conversation_types = ["basic", "cot_freetext", "cot_structured"]

        for conv_type in conversation_types:
            generator = DataSetGenerator(
                instructions="Create educational content",
                generation_system_prompt="You are an educator.",
                conversation_type=conv_type,
                **minimal_test_config,
            )

            dataset = generator.create_data(
                num_steps=1,
                batch_size=1,
                topic_model=tree,
            )

            assert len(dataset.samples) > 0  # type: ignore

    def test_generator_with_system_message(self, minimal_test_config):
        """Test generator with sys_msg parameter."""
        tree = Tree(
            topic_prompt="Science topics",
            degree=2,
            depth=1,
            **minimal_test_config,
        )
        list(tree.build())

        # Test with system message
        generator_with_sys = DataSetGenerator(
            instructions="Create science Q&A",
            generation_system_prompt="You are a science teacher.",
            dataset_system_prompt="You are a helpful science tutor.",
            sys_msg=True,
            **minimal_test_config,
        )

        dataset_with_sys = generator_with_sys.create_data(
            num_steps=1,
            batch_size=1,
            topic_model=tree,
        )

        # Verify system message is included
        assert len(dataset_with_sys.samples) > 0  # type: ignore
        sample = dataset_with_sys.samples[0]  # type: ignore
        messages = sample["messages"]
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert len(system_messages) > 0

        # Test without system message
        generator_no_sys = DataSetGenerator(
            instructions="Create science Q&A",
            generation_system_prompt="You are a science teacher.",
            sys_msg=False,
            **minimal_test_config,
        )

        dataset_no_sys = generator_no_sys.create_data(
            num_steps=1,
            batch_size=1,
            topic_model=tree,
        )

        # Verify no system message
        sample = dataset_no_sys.samples[0]  # type: ignore
        messages = sample["messages"]
        system_messages = [msg for msg in messages if msg["role"] == "system"]
        assert len(system_messages) == 0

    def test_generator_error_handling(self, minimal_test_config):
        """Test generator error handling."""
        generator = DataSetGenerator(
            instructions="Test",
            generation_system_prompt="Test",
            **minimal_test_config,
        )

        # Test with invalid num_steps
        tree = Tree(topic_prompt="Test", degree=2, depth=1, **minimal_test_config)
        list(tree.build())

        with pytest.raises(DataSetGeneratorError):
            generator.create_data(num_steps=0, batch_size=1, topic_model=tree)

    def test_generator_with_retries(self, minimal_test_config):
        """Test generator retry mechanism."""
        config = minimal_test_config.copy()
        config["max_retries"] = 2

        generator = DataSetGenerator(
            instructions="Create content",
            generation_system_prompt="You are helpful.",
            **config,
        )

        tree = Tree(topic_prompt="Simple topic", degree=2, depth=1, **minimal_test_config)
        list(tree.build())

        # Should handle retries gracefully
        dataset = generator.create_data(
            num_steps=1,
            batch_size=1,
            topic_model=tree,
        )

        assert isinstance(dataset, Dataset)

    def test_generator_with_github_provider(self):
        """Test generator specifically with GitHub provider."""
        if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
            pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available")

        # Create tree
        tree = Tree(
            topic_prompt="Programming fundamentals",
            topic_system_prompt="You are a programming instructor.",
            provider="github",
            model_name="openai/gpt-4o-mini",
            degree=2,
            depth=1,
            temperature=0.1,
        )
        list(tree.build())

        # Create generator
        generator = DataSetGenerator(
            instructions="Create programming Q&A pairs for beginners",
            generation_system_prompt="You are a helpful programming tutor.",
            provider="github",
            model_name="openai/gpt-4o-mini",
            temperature=0.1,
            max_retries=1,
        )

        # Generate data
        dataset = generator.create_data(
            num_steps=1,
            batch_size=1,
            topic_model=tree,
        )

        assert len(dataset.samples) > 0  # type: ignore

    def test_generator_batch_processing(self, minimal_test_config):
        """Test generator batch processing."""
        tree = Tree(
            topic_prompt="General knowledge",
            degree=2,
            depth=1,
            **minimal_test_config,
        )
        list(tree.build())

        generator = DataSetGenerator(
            instructions="Create general knowledge questions",
            generation_system_prompt="You are knowledgeable.",
            **minimal_test_config,
        )

        # Test with batch size > 1
        dataset = generator.create_data(
            num_steps=1,
            batch_size=2,  # Process 2 samples at once
            topic_model=tree,
        )

        assert len(dataset.samples) >= MESSAGE_LIMIT  # type: ignore

    def test_generator_dataset_validation(self, minimal_test_config):
        """Test that generated datasets are properly validated."""
        tree = Tree(
            topic_prompt="Test validation",
            degree=2,
            depth=1,
            **minimal_test_config,
        )
        list(tree.build())

        generator = DataSetGenerator(
            instructions="Create simple content",
            generation_system_prompt="You create valid responses.",
            **minimal_test_config,
        )

        dataset = generator.create_data(
            num_steps=1,
            batch_size=1,
            topic_model=tree,
        )

        # Verify all samples pass validation
        for sample in dataset.samples:  # type: ignore
            assert Dataset.validate_sample(sample)

        # Check for failed samples
        assert len(generator.failed_samples) == 0 or generator.failed_samples is None
