"""
Integration tests for CLI functionality.
"""

import json
import os
import subprocess
import tempfile

from pathlib import Path

import pytest
import yaml


def run_cli_command(*args, **kwargs):
    """Helper function to run deepfabric CLI commands with common settings.

    Args:
        *args: Command arguments (after 'deepfabric')
        **kwargs: Additional arguments for subprocess.run

    Returns:
        subprocess.CompletedProcess result
    """
    cmd = ["uv", "run", "deepfabric"] + list(args)

    # Default settings
    defaults = {
        "check": False,
        "capture_output": True,
        "text": True,
        "cwd": Path(__file__).parent.parent.parent,
    }

    # Override with any provided kwargs
    defaults.update(kwargs)

    return subprocess.run(cmd, **defaults)  # noqa: PLW1510, S603


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_generate_with_config(self, test_config_path, temp_output_dir):
        """Test CLI generate command with config file."""
        # Create a temporary config with updated paths
        temp_config = temp_output_dir / "cli_test_config.yaml"

        # Load original config and update paths using YAML parser
        with open(test_config_path) as f:
            config_data = yaml.safe_load(f)

        # Update paths in config data structure
        config_data["topic_tree"]["save_as"] = str(temp_output_dir / "cli_tree.jsonl")
        config_data["dataset"]["save_as"] = str(temp_output_dir / "cli_dataset.jsonl")

        # Write updated config
        with open(temp_config, "w") as f:
            yaml.safe_dump(config_data, f, default_flow_style=False)

        # Run CLI command
        result = run_cli_command("generate", str(temp_config))

        # Check command succeeded
        assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"

        # Verify outputs were created
        tree_file = temp_output_dir / "cli_tree.jsonl"
        dataset_file = temp_output_dir / "cli_dataset.jsonl"

        assert tree_file.exists(), "Tree file was not created"
        assert dataset_file.exists(), "Dataset file was not created"

        # Verify file contents
        with open(dataset_file) as f:
            lines = f.readlines()
            assert len(lines) > 0, "Dataset file is empty"

            # Verify JSONL format
            for line in lines:
                data = json.loads(line)
                assert "messages" in data

    def test_cli_generate_with_overrides(self, test_config_path, temp_output_dir):
        """Test CLI generate with parameter overrides."""
        tree_path = temp_output_dir / "override_tree.jsonl"
        dataset_path = temp_output_dir / "override_dataset.jsonl"

        result = run_cli_command(
            "generate",
            str(test_config_path),
            "--save-tree",
            str(tree_path),
            "--dataset-save-as",
            str(dataset_path),
            "--num-steps",
            "2",
            "--batch-size",
            "1",
            "--temperature",
            "0.1",
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert tree_path.exists()
        assert dataset_path.exists()

    def test_cli_validate_config(self, test_config_path):
        """Test CLI config validation."""
        result = run_cli_command("validate", str(test_config_path))

        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        assert "valid" in result.stdout.lower()

    def test_cli_info_command(self):
        """Test CLI info command."""
        result = run_cli_command("info")

        assert result.returncode == 0
        assert "DeepFabric" in result.stdout
        assert "generate" in result.stdout  # Should list commands

    def test_cli_tree_mode(self, temp_output_dir):
        """Test CLI with tree mode explicitly."""

        tree_path = temp_output_dir / "cli_tree_mode.jsonl"
        dataset_path = temp_output_dir / "cli_tree_dataset.jsonl"

        result = run_cli_command(
            "generate",
            "--mode",
            "tree",
            "--topic-prompt",
            "Simple programming concepts",
            "--generation-system-prompt",
            "You are a programming tutor.",
            "--provider",
            "github",
            "--model",
            "openai/gpt-4o-mini",
            "--degree",
            "2",
            "--depth",
            "1",
            "--num-steps",
            "2",
            "--batch-size",
            "1",
            "--temperature",
            "0.1",
            "--save-tree",
            str(tree_path),
            "--dataset-save-as",
            str(dataset_path),
        )

        assert result.returncode == 0, f"CLI tree mode failed: {result.stderr}"
        assert tree_path.exists()
        assert dataset_path.exists()

    def test_cli_graph_mode(self, temp_output_dir):
        """Test CLI with graph mode."""

        graph_path = temp_output_dir / "cli_graph.json"
        dataset_path = temp_output_dir / "cli_graph_dataset.jsonl"

        result = run_cli_command(
            "generate",
            "--mode",
            "graph",
            "--topic-prompt",
            "Data science fundamentals",
            "--generation-system-prompt",
            "You are a data science educator.",
            "--provider",
            "github",
            "--model",
            "openai/gpt-4o-mini",
            "--degree",
            "2",
            "--depth",
            "1",
            "--num-steps",
            "2",
            "--batch-size",
            "1",
            "--temperature",
            "0.1",
            "--save-graph",
            str(graph_path),
            "--dataset-save-as",
            str(dataset_path),
        )

        assert result.returncode == 0, f"CLI graph mode failed: {result.stderr}"
        assert graph_path.exists()
        assert dataset_path.exists()

    def test_cli_load_existing_tree(self, sample_tree_path, temp_output_dir):
        """Test CLI loading existing tree file."""
        dataset_path = temp_output_dir / "loaded_tree_dataset.jsonl"

        result = run_cli_command(
            "generate",
            "--load-tree",
            str(sample_tree_path),
            "--generation-system-prompt",
            "You are an educational assistant.",
            "--provider",
            "github",
            "--model",
            "openai/gpt-4o-mini",
            "--temperature",
            "0.1",
            "--num-steps",
            "2",
            "--batch-size",
            "1",
            "--dataset-save-as",
            str(dataset_path),
        )

        # Skip if no token available
        if result.returncode != 0 and "TOKEN" in result.stderr:
            pytest.skip("GitHub token not available")

        assert result.returncode == 0, f"CLI load tree failed: {result.stderr}"
        assert dataset_path.exists()

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid inputs."""
        # Test with non-existent config file
        result = run_cli_command("generate", "nonexistent.yaml")

        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "does not exist" in result.stderr.lower()

        # Test validation with invalid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            invalid_config = f.name

        try:
            result = run_cli_command("validate", invalid_config)

            assert result.returncode != 0
        finally:
            os.unlink(invalid_config)

    def test_cli_visualize_command(self, temp_output_dir):
        """Test CLI visualize command."""

        # First create a graph
        graph_path = temp_output_dir / "viz_graph.json"

        result = run_cli_command(
            "generate",
            "--mode",
            "graph",
            "--topic-prompt",
            "Visualization test",
            "--provider",
            "github",
            "--model",
            "openai/gpt-4o-mini",
            "--degree",
            "2",
            "--depth",
            "1",
            "--temperature",
            "0.1",
            "--save-graph",
            str(graph_path),
            "--dataset-save-as",
            str(temp_output_dir / "viz_dataset.jsonl"),
            "--num-steps",
            "1",
            "--batch-size",
            "1",
        )

        if result.returncode != 0:
            pytest.skip(f"Graph generation failed: {result.stderr}")

        # Now visualize it
        viz_output = temp_output_dir / "test_viz"

        result = run_cli_command("visualize", str(graph_path), "--output", str(viz_output))

        assert result.returncode == 0, f"Visualize failed: {result.stderr}"

        # Check SVG file was created
        svg_file = Path(f"{viz_output}.svg")
        assert svg_file.exists(), "SVG file was not created"

    def test_cli_help_commands(self):
        """Test CLI help functionality."""
        # Test main help
        result = run_cli_command("--help")

        assert result.returncode == 0
        assert "generate" in result.stdout
        assert "validate" in result.stdout

        # Test generate help
        result = run_cli_command("generate", "--help")

        assert result.returncode == 0
        assert "--provider" in result.stdout
        assert "--model" in result.stdout

    def test_cli_with_different_providers_fallback(self, temp_output_dir):
        """Test CLI gracefully handles missing provider configurations."""
        # Test with missing provider (should use config defaults)
        result = run_cli_command(
            "generate",
            "--topic-prompt",
            "Fallback test",
            "--generation-system-prompt",
            "Test system prompt",
            "--provider",
            "nonexistent_provider",
            "--model",
            "test/model",
            "--degree",
            "2",
            "--depth",
            "1",
            "--num-steps",
            "1",
            "--batch-size",
            "1",
            "--dataset-save-as",
            str(temp_output_dir / "fallback_dataset.jsonl"),
        )

        # Should fail with appropriate error message
        assert result.returncode != 0
        assert (
            "unsupported" in result.stderr.lower()
            or "error" in result.stderr.lower()
            or "provider" in result.stderr.lower()
        )
