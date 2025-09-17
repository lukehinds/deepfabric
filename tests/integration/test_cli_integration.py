"""
Integration tests for CLI functionality.
"""

import json
import os
import subprocess
import tempfile

from pathlib import Path

import pytest


class TestCLIIntegration:
    """Integration tests for CLI commands."""

    def test_cli_generate_with_config(self, test_config_path, temp_output_dir):
        """Test CLI generate command with config file."""
        # Create a temporary config with updated paths
        temp_config = temp_output_dir / "cli_test_config.yaml"

        # Read original config and update paths
        with open(test_config_path) as f:
            config_content = f.read()

        # Update paths in config
        config_content = config_content.replace(
            'save_as: "test_tree.jsonl"', f'save_as: "{temp_output_dir}/cli_tree.jsonl"'
        )
        config_content = config_content.replace(
            'save_as: "test_dataset.jsonl"', f'save_as: "{temp_output_dir}/cli_dataset.jsonl"'
        )

        with open(temp_config, "w") as f:
            f.write(config_content)

        # Run CLI command
        result = subprocess.run(  # noqa: S603
            ["uv", "run", "deepfabric", "generate", str(temp_config)],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

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

        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "deepfabric",
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
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert tree_path.exists()
        assert dataset_path.exists()

    def test_cli_validate_config(self, test_config_path):
        """Test CLI config validation."""
        result = subprocess.run(  # noqa: S603
            ["uv", "run", "deepfabric", "validate", str(test_config_path)],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, f"Validation failed: {result.stderr}"
        assert "valid" in result.stdout.lower()

    def test_cli_info_command(self):
        """Test CLI info command."""
        result = subprocess.run(
            ["uv", "run", "deepfabric", "info"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "DeepFabric" in result.stdout
        assert "generate" in result.stdout  # Should list commands

    def test_cli_tree_mode(self, temp_output_dir):
        """Test CLI with tree mode explicitly."""
        if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
            pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available")

        tree_path = temp_output_dir / "cli_tree_mode.jsonl"
        dataset_path = temp_output_dir / "cli_tree_dataset.jsonl"

        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "deepfabric",
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
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, f"CLI tree mode failed: {result.stderr}"
        assert tree_path.exists()
        assert dataset_path.exists()

    def test_cli_graph_mode(self, temp_output_dir):
        """Test CLI with graph mode."""
        if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
            pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available")

        graph_path = temp_output_dir / "cli_graph.json"
        dataset_path = temp_output_dir / "cli_graph_dataset.jsonl"

        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "deepfabric",
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
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, f"CLI graph mode failed: {result.stderr}"
        assert graph_path.exists()
        assert dataset_path.exists()

    def test_cli_load_existing_tree(self, sample_tree_path, temp_output_dir):
        """Test CLI loading existing tree file."""
        dataset_path = temp_output_dir / "loaded_tree_dataset.jsonl"

        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "deepfabric",
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
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Skip if no token available
        if result.returncode != 0 and "TOKEN" in result.stderr:
            pytest.skip("GitHub token not available")

        assert result.returncode == 0, f"CLI load tree failed: {result.stderr}"
        assert dataset_path.exists()

    def test_cli_error_handling(self):
        """Test CLI error handling for invalid inputs."""
        # Test with non-existent config file
        result = subprocess.run(
            ["uv", "run", "deepfabric", "generate", "nonexistent.yaml"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "does not exist" in result.stderr.lower()

        # Test validation with invalid config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content:")
            invalid_config = f.name

        try:
            result = subprocess.run(  # noqa: S603
                ["uv", "run", "deepfabric", "validate", invalid_config],  # noqa: S607
                check=False,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent,
            )

            assert result.returncode != 0
        finally:
            os.unlink(invalid_config)

    def test_cli_visualize_command(self, temp_output_dir):
        """Test CLI visualize command."""
        if not (os.environ.get("GITHUB_TOKEN") or os.environ.get("MODELS_TOKEN")):
            pytest.skip("GITHUB_TOKEN or MODELS_TOKEN not available")

        # First create a graph
        graph_path = temp_output_dir / "viz_graph.json"

        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "deepfabric",
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
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        if result.returncode != 0:
            pytest.skip(f"Graph generation failed: {result.stderr}")

        # Now visualize it
        viz_output = temp_output_dir / "test_viz"

        result = subprocess.run(  # noqa: S603
            ["uv", "run", "deepfabric", "visualize", str(graph_path), "--output", str(viz_output)],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0, f"Visualize failed: {result.stderr}"

        # Check SVG file was created
        svg_file = Path(f"{viz_output}.svg")
        assert svg_file.exists(), "SVG file was not created"

    def test_cli_help_commands(self):
        """Test CLI help functionality."""
        # Test main help
        result = subprocess.run(
            ["uv", "run", "deepfabric", "--help"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "generate" in result.stdout
        assert "validate" in result.stdout

        # Test generate help
        result = subprocess.run(
            ["uv", "run", "deepfabric", "generate", "--help"],  # noqa: S607
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "--provider" in result.stdout
        assert "--model" in result.stdout

    def test_cli_with_different_providers_fallback(self, temp_output_dir):
        """Test CLI gracefully handles missing provider configurations."""
        # Test with missing provider (should use config defaults)
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "uv",
                "run",
                "deepfabric",
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
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Should fail with appropriate error message
        assert result.returncode != 0
        assert (
            "unsupported" in result.stderr.lower()
            or "error" in result.stderr.lower()
            or "provider" in result.stderr.lower()
        )
