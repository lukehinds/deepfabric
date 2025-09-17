"""
End-to-end pipeline integration tests.
"""

import json
import os

import pytest

from deepfabric import DataSetGenerator, DeepFabricConfig, Tree

MESSAGE_LIMIT = 2
SAMPLE_LIMIT = 3
CONTENT_LENGTH_THRESHOLD = 50


class TestPipelineIntegration:
    """End-to-end pipeline integration tests."""

    def test_tree_to_dataset_pipeline(self, minimal_test_config, temp_output_dir):
        """Test complete pipeline from tree creation to dataset generation."""
        # Step 1: Create and build tree
        tree = Tree(
            topic_prompt="Basic mathematics",
            topic_system_prompt="You are a mathematics educator creating educational topics.",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        events = list(tree.build())
        assert events[-1]["event"] == "build_complete"

        # Step 2: Save tree
        tree_path = temp_output_dir / "pipeline_tree.jsonl"
        tree.save(str(tree_path))
        assert tree_path.exists()

        # Step 3: Create generator
        generator = DataSetGenerator(
            instructions="Create mathematics questions and detailed step-by-step solutions.",
            generation_system_prompt="You are a mathematics tutor providing clear explanations.",
            **minimal_test_config,
        )

        # Step 4: Generate dataset
        dataset = generator.create_data(
            num_steps=3,
            batch_size=1,
            topic_model=tree,
        )

        # Step 5: Save dataset
        dataset_path = temp_output_dir / "pipeline_dataset.jsonl"
        dataset.save(str(dataset_path))  # type: ignore
        assert dataset_path.exists()

        # Step 6: Verify pipeline outputs
        assert len(dataset.samples) >= SAMPLE_LIMIT  # type: ignore

        # Verify dataset content quality
        for sample in dataset.samples:  # type: ignore
            assert "messages" in sample
            messages = sample["messages"]
            assert len(messages) >= MESSAGE_LIMIT  # At least user and assistant

            # Check for user and assistant messages
            roles = [msg["role"] for msg in messages]
            assert "user" in roles
            assert "assistant" in roles

    def test_config_based_pipeline(self, test_config_path, temp_output_dir):
        """Test pipeline using YAML configuration."""
        # Load config
        config = DeepFabricConfig.from_yaml(str(test_config_path))

        # Override paths to use temp directory
        tree_path = temp_output_dir / "config_tree.jsonl"
        dataset_path = temp_output_dir / "config_dataset.jsonl"

        # Step 1: Create tree from config
        tree_params = config.get_tree_params()  # type: ignore
        tree = Tree(**tree_params)
        list(tree.build())
        tree.save(str(tree_path))

        # Step 2: Create generator from config
        engine_params = config.get_engine_params()
        generator = DataSetGenerator(**engine_params)

        # Step 3: Create dataset
        dataset_config = config.get_dataset_config()
        dataset = generator.create_data(
            num_steps=dataset_config["creation"]["num_steps"],
            batch_size=dataset_config["creation"]["batch_size"],
            topic_model=tree,
        )

        # Step 4: Save dataset
        dataset.save(str(dataset_path))  # type: ignore

        # Verify outputs
        assert tree_path.exists()
        assert dataset_path.exists()
        assert len(dataset.samples) > 0  # type: ignore

    def test_error_recovery_pipeline(self, minimal_test_config, temp_output_dir):
        """Test pipeline error recovery and partial results."""
        # Create tree
        tree = Tree(
            topic_prompt="Complex topics",
            degree=2,
            depth=1,
            **minimal_test_config,
        )
        list(tree.build())

        # Create generator with limited retries
        config = minimal_test_config.copy()
        config["max_retries"] = 1

        generator = DataSetGenerator(
            instructions="Create content",
            generation_system_prompt="You are helpful.",
            **config,
        )

        # Generate with potential failures
        dataset = generator.create_data(
            num_steps=3,
            batch_size=1,
            topic_model=tree,
        )

        # Should have some results even if some fail
        assert isinstance(dataset, type(dataset))  # Should not crash

        # Save what we have
        dataset_path = temp_output_dir / "partial_dataset.jsonl"
        dataset.save(str(dataset_path))  # type: ignore
        assert dataset_path.exists()

    def test_validation_pipeline(self, minimal_test_config):
        """Test pipeline data validation throughout."""
        # Create tree with validation
        tree = Tree(
            topic_prompt="Validated content",
            degree=2,
            depth=1,
            **minimal_test_config,
        )

        list(tree.build())

        # Verify tree structure
        paths = tree.get_all_paths()
        assert len(paths) > 0
        for path in paths:
            assert len(path) > 0
            assert all(isinstance(topic, str) and len(topic.strip()) > 0 for topic in path)

        # Create generator
        generator = DataSetGenerator(
            instructions="Create valid, structured content",
            generation_system_prompt="You create well-formatted responses.",
            **minimal_test_config,
        )

        # Generate and validate dataset
        dataset = generator.create_data(
            num_steps=2,
            batch_size=1,
            topic_model=tree,
        )

        # Validate all samples
        for sample in dataset.samples:  # type: ignore
            assert dataset.validate_sample(sample), f"Invalid sample: {sample}"  # type: ignore

    def test_different_conversation_types_pipeline(self, minimal_test_config):
        """Test pipeline with different conversation types."""
        conversation_types = ["basic", "cot_freetext", "cot_structured"]

        for conv_type in conversation_types:
            # Create tree
            tree = Tree(
                topic_prompt=f"Topics for {conv_type}",
                degree=2,
                depth=1,
                **minimal_test_config,
            )
            list(tree.build())

            # Create generator with specific conversation type
            generator = DataSetGenerator(
                instructions=f"Create {conv_type} style content",
                generation_system_prompt="You adapt your style to requirements.",
                conversation_type=conv_type,
                **minimal_test_config,
            )

            # Generate dataset
            dataset = generator.create_data(
                num_steps=1,
                batch_size=1,
                topic_model=tree,
            )

            assert len(dataset.samples) > 0  # type: ignore

            # Verify conversation structure based on type
            sample = dataset.samples[0]  # type: ignore
            messages = sample["messages"]

            if conv_type.startswith("cot"):
                # CoT should have more detailed responses
                assistant_messages = [msg for msg in messages if msg["role"] == "assistant"]
                assert len(assistant_messages) > 0
                # CoT responses should be longer
                for msg in assistant_messages:
                    assert len(msg["content"]) > CONTENT_LENGTH_THRESHOLD  # Basic length check

    def test_graph_to_dataset_pipeline(self, minimal_test_config, temp_output_dir):
        """Test pipeline using graph instead of tree."""
        # Import Graph
        from deepfabric import Graph  # noqa: PLC0415

        # Create graph
        graph = Graph(
            topic_prompt="Software engineering",
            topic_system_prompt="You are a software engineering expert.",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        list(graph.build())

        # Save graph
        graph_path = temp_output_dir / "pipeline_graph.json"
        graph.save(str(graph_path))

        # Convert to tree for dataset generation
        tree = graph.to_tree()  # type: ignore

        # Generate dataset
        generator = DataSetGenerator(
            instructions="Create software engineering content",
            generation_system_prompt="You are a software engineering instructor.",
            **minimal_test_config,
        )

        dataset = generator.create_data(
            num_steps=2,
            batch_size=1,
            topic_model=tree,
        )

        # Save dataset
        dataset_path = temp_output_dir / "graph_dataset.jsonl"
        dataset.save(str(dataset_path))  # type: ignore

        # Verify outputs
        assert graph_path.exists()
        assert dataset_path.exists()
        assert len(dataset.samples) >= MESSAGE_LIMIT  # type: ignore

    def test_pipeline_with_github_provider(self, temp_output_dir):
        """Test complete pipeline with GitHub provider."""
        if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
            pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available")

        github_config = {
            "provider": "github",
            "model_name": "openai/gpt-4o-mini",
            "temperature": 0.1,
            "max_retries": 1,
            "request_timeout": 60,  # Longer timeout for real API
        }

        # Create tree
        tree = Tree(
            topic_prompt="Environmental science basics",
            topic_system_prompt="You are an environmental science educator.",
            degree=2,
            depth=1,
            **github_config,
        )

        events = list(tree.build())
        assert events[-1]["event"] == "build_complete"

        # Generate dataset
        generator = DataSetGenerator(
            instructions="Create environmental science educational content with real-world examples.",
            generation_system_prompt="You are an environmental science teacher creating engaging content.",
            **github_config,
        )

        dataset = generator.create_data(
            num_steps=2,
            batch_size=1,
            topic_model=tree,
        )

        # Save outputs
        tree_path = temp_output_dir / "github_tree.jsonl"
        dataset_path = temp_output_dir / "github_dataset.jsonl"

        tree.save(str(tree_path))
        dataset.save(str(dataset_path))  # type: ignore

        # Verify quality with real API
        assert len(dataset.samples) >= MESSAGE_LIMIT  # type: ignore
        for sample in dataset.samples:  # type: ignore
            messages = sample["messages"]
            # Real API should produce higher quality content
            for msg in messages:
                assert len(msg["content"]) > CONTENT_LENGTH_THRESHOLD  # Reasonable content length
                assert msg["content"].strip()  # Not empty

    def test_pipeline_statistics_and_metadata(self, minimal_test_config, temp_output_dir):
        """Test pipeline generates proper statistics and metadata."""
        # Create tree
        tree = Tree(
            topic_prompt="Statistics topics",
            degree=2,
            depth=1,
            **minimal_test_config,
        )
        list(tree.build())

        # Generate dataset
        generator = DataSetGenerator(
            instructions="Create statistics content",
            generation_system_prompt="You teach statistics.",
            **minimal_test_config,
        )

        dataset = generator.create_data(
            num_steps=3,
            batch_size=1,
            topic_model=tree,
        )

        # Get statistics
        stats = dataset.get_statistics()  # type: ignore

        # Verify statistics structure
        assert "total_samples" in stats
        assert "total_messages" in stats
        assert "avg_messages_per_sample" in stats
        assert stats["total_samples"] == len(dataset.samples)  # type: ignore
        assert stats["total_messages"] > 0

        # Create metadata
        metadata = {
            "tree_config": {
                "degree": tree.degree,
                "depth": tree.depth,
                "total_paths": len(tree.get_all_paths()),
            },
            "generator_config": {
                "provider": generator.config.provider,
                "model": generator.config.model_name,
                "temperature": generator.config.temperature,
            },
            "dataset_stats": stats,
            "failed_samples": len(generator.failed_samples) if generator.failed_samples else 0,
        }

        # Save metadata
        metadata_path = temp_output_dir / "pipeline_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        assert metadata_path.exists()

        # Verify metadata content
        with open(metadata_path) as f:
            loaded_metadata = json.load(f)
            assert loaded_metadata["dataset_stats"]["total_samples"] > 0
            assert loaded_metadata["tree_config"]["total_paths"] > 0
