"""
Integration tests for Graph generation.
"""

import json

import pytest

from deepfabric import Graph

MESSAGE_LIMIT = 2


class TestGraphIntegration:
    """Integration tests for Graph generation functionality."""

    def test_graph_creation_basic(self, minimal_test_config):
        """Test basic graph creation with minimal parameters."""
        graph = Graph(
            topic_prompt="Basic Python concepts",
            topic_system_prompt="You are a Python educator.",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        assert graph.topic_prompt == "Basic Python concepts"
        assert graph.degree == MESSAGE_LIMIT
        assert graph.depth == MESSAGE_LIMIT

    def test_graph_build_small(self, minimal_test_config):
        """Test building a small graph and validate structure."""
        graph = Graph(
            topic_prompt="Python data structures",
            topic_system_prompt="You are a Python expert creating educational topics about data structures.",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        # Build the graph
        events = list(graph.build())

        # Check that we get events
        assert len(events) > 0

        # Check final event
        final_event = events[-1]
        assert final_event["event"] == "build_complete"

        # Graph should have nodes
        assert graph.root is not None  # type: ignore
        assert len(graph.nodes) > 1  # Should have root + additional nodes

    def test_graph_save_load(self, minimal_test_config, temp_output_dir):
        """Test graph persistence to JSON format."""
        graph = Graph(
            topic_prompt="Python functions",
            topic_system_prompt="You are a Python expert.",
            degree=2,
            depth=1,  # Keep small for speed
            **minimal_test_config,
        )

        # Build the graph
        list(graph.build())

        # Save to file
        save_path = temp_output_dir / "test_graph.json"
        graph.save(str(save_path))

        # Verify file exists and has content
        assert save_path.exists()

        with open(save_path) as f:
            data = json.load(f)
            assert "nodes" in data
            assert "root_id" in data
            assert "degree" in data
            assert "depth" in data

    def test_graph_from_json(self, minimal_test_config, temp_output_dir):
        """Test loading graph from JSON file."""
        # Create and save a graph
        original_graph = Graph(
            topic_prompt="Python basics",
            degree=2,
            depth=1,
            **minimal_test_config,
        )

        list(original_graph.build())
        save_path = temp_output_dir / "test_graph.json"
        original_graph.save(str(save_path))

        # Load graph from JSON
        loaded_graph = Graph.from_json(
            str(save_path),
            {
                "topic_prompt": "Python basics",
                "degree": 2,
                "depth": 1,
                **minimal_test_config,
            },
        )

        # Verify loaded graph has same structure
        assert loaded_graph.degree == original_graph.degree
        assert loaded_graph.depth == original_graph.depth
        assert len(loaded_graph.nodes) == len(original_graph.nodes)

    def test_graph_to_tree_conversion(self, minimal_test_config):
        """Test converting graph to tree format."""
        graph = Graph(
            topic_prompt="Python control flow",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        # Build the graph
        list(graph.build())

        # Convert to tree
        tree = graph.to_tree()  # type: ignore

        # Verify tree structure
        assert tree is not None
        paths = tree.get_all_paths()
        assert len(paths) > 0

        # Each path should contain topics
        for path in paths:
            assert len(path) > 0
            assert all(isinstance(topic, str) for topic in path)

    def test_graph_visualization(self, minimal_test_config, temp_output_dir):
        """Test graph visualization generation."""
        graph = Graph(
            topic_prompt="Python modules",
            degree=2,
            depth=1,
            **minimal_test_config,
        )

        # Build the graph
        list(graph.build())

        # Create visualization
        output_path = temp_output_dir / "test_graph_viz"
        graph.visualize(str(output_path))

        # Check SVG file was created
        svg_path = temp_output_dir / "test_graph_viz.svg"
        assert svg_path.exists()

        # Verify SVG content
        with open(svg_path) as f:
            content = f.read()
            assert "<svg" in content
            assert "</svg>" in content

    def test_graph_with_github_provider(self):
        """Test graph creation specifically with GitHub provider."""

        graph = Graph(
            topic_prompt="Web development basics",
            topic_system_prompt="You are a web development expert creating educational content.",
            provider="github",
            model_name="openai/gpt-4o-mini",
            degree=2,
            depth=1,  # Keep minimal for speed
            temperature=0.1,
        )

        # Build should succeed
        events = list(graph.build())
        assert len(events) > 0

        final_event = events[-1]
        assert final_event["event"] == "build_complete"

    def test_graph_node_relationships(self, minimal_test_config):
        """Test that graph nodes have proper parent-child relationships."""
        graph = Graph(
            topic_prompt="Database concepts",
            degree=2,
            depth=2,
            **minimal_test_config,
        )

        # Build the graph
        list(graph.build())

        # Check root node exists
        assert graph.root is not None  # type: ignore

        # Verify relationships
        nodes_with_children = [node for node in graph.nodes.values() if node.children]
        nodes_with_parents = [node for node in graph.nodes.values() if node.parents]

        # Should have some nodes with children (non-leaf nodes)
        assert len(nodes_with_children) > 0

        # Should have some nodes with parents (non-root nodes)
        assert len(nodes_with_parents) > 0

    def test_graph_error_handling(self, minimal_test_config):
        """Test graph handles API errors gracefully."""
        # Use completely invalid provider to trigger error
        config = minimal_test_config.copy()
        config["provider"] = "nonexistent_provider"
        config["model_name"] = "invalid_model"

        # Creating graph with invalid provider should raise error
        with pytest.raises(Exception):  # noqa: B017
            Graph(
                topic_prompt="Test",
                degree=2,
                depth=1,
                **config,
            )
