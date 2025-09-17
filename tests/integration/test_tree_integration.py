"""
Integration tests for Tree generation.
"""

import os

import pytest

from deepfabric import Tree
from deepfabric.exceptions import DataSetGeneratorError, TreeError

MESSAGE_LIMIT = 2
TOTAL_PATHS = 4
PATHS_DEPTH = 3
TREE_DEPTH_LIMIT = 2
TREE_DEGREE_LIMIT = 2


class TestTreeIntegration:
    """Integration tests for Tree generation functionality."""

    def test_tree_creation_basic(self, minimal_test_config):
        """Test basic tree creation with minimal parameters."""
        tree = Tree(
            topic_prompt="Basic Python concepts",
            topic_system_prompt="You are a Python educator.",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        assert tree.topic_prompt == "Basic Python concepts"
        assert tree.degree == TREE_DEGREE_LIMIT
        assert tree.depth == TREE_DEPTH_LIMIT

    def test_tree_build_small(self, minimal_test_config):
        """Test building a small tree and validate structure."""
        tree = Tree(
            topic_prompt="Python data types",
            topic_system_prompt="You are a Python expert creating educational topics.",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        # Build the tree
        events = list(tree.build())

        # Check that we get events
        assert len(events) > 0

        # Check final event
        final_event = events[-1]
        assert final_event["event"] == "build_complete"
        assert final_event["total_paths"] == TOTAL_PATHS  # 2^2 = 4 paths

        # Verify tree has paths
        paths = tree.get_all_paths()
        assert len(paths) == TOTAL_PATHS

        # Verify each path has correct depth
        for path in paths:
            assert len(path) == PATHS_DEPTH  # root + 2 levels

    def test_tree_save_load(self, minimal_test_config, temp_output_dir):
        """Test tree persistence to JSONL format."""
        tree = Tree(
            topic_prompt="Python functions",
            topic_system_prompt="You are a Python expert.",
            degree=2,
            depth=1,  # Keep small for speed
            **minimal_test_config,
        )

        # Build the tree
        list(tree.build())

        # Save to file
        save_path = temp_output_dir / "test_tree.jsonl"
        tree.save(str(save_path))

        # Verify file exists and has content
        assert save_path.exists()
        with open(save_path) as f:
            lines = f.readlines()
            assert len(lines) > 0

            # Check JSONL format
            import json  # noqa: PLC0415

            for line in lines:
                data = json.loads(line)
                assert "topic" in data
                assert "topic_path" in data

    def test_tree_validation_errors(self, minimal_test_config):
        """Test tree validation for invalid configurations."""
        # Test invalid degree
        with pytest.raises((TreeError, ValueError)):
            Tree(
                topic_prompt="Test",
                degree=0,  # Invalid
                depth=2,
                **minimal_test_config,
            )

        # Test invalid depth
        with pytest.raises((TreeError, ValueError)):
            Tree(
                topic_prompt="Test",
                degree=2,
                depth=0,  # Invalid
                **minimal_test_config,
            )

    def test_tree_path_calculation(self, minimal_test_config):
        """Test tree path calculation is correct."""
        tree = Tree(
            topic_prompt="Test topic",
            degree=3,
            depth=2,
            **minimal_test_config,
        )

        # Build tree
        list(tree.build())

        # Verify path count
        paths = tree.get_all_paths()
        expected_paths = 3**2  # degree^depth
        assert len(paths) == expected_paths

    def test_tree_with_github_provider(self):
        """Test tree creation specifically with GitHub provider."""
        # Skip if no GitHub token
        if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
            pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available")

        tree = Tree(
            topic_prompt="Machine learning basics",
            topic_system_prompt="You are an ML expert creating educational content.",
            provider="github",
            model_name="openai/gpt-4o-mini",
            degree=2,
            depth=1,  # Keep minimal for speed
            temperature=0.1,
        )

        # Build should succeed
        events = list(tree.build())
        assert len(events) > 0

        final_event = events[-1]
        assert final_event["event"] == "build_complete"

    def test_tree_error_handling(self, minimal_test_config):
        """Test tree handles API errors gracefully."""
        # Use invalid model to trigger error
        config = minimal_test_config.copy()
        config["model_name"] = "invalid/model"

        tree = Tree(
            topic_prompt="Test",
            degree=2,
            depth=1,
            **config,
        )

        # Build should raise appropriate error
        with pytest.raises(DataSetGeneratorError):
            list(tree.build())
