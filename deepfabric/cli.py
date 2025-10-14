import os
import sys

from typing import Literal, NoReturn, cast

import click
import yaml

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from .config import DeepFabricConfig
from .config_manager import apply_cli_overrides, get_final_parameters, load_config
from .dataset_manager import create_dataset, save_dataset
from .exceptions import ConfigurationError
from .format_command import format_cli
from .generator import DataSetGenerator
from .graph import Graph
from .metrics import set_trace_debug, trace
from .topic_manager import load_or_build_topic_model, save_topic_model
from .topic_model import TopicModel
from .tui import get_tui
from .validation import show_validation_success, validate_path_requirements

OverrideValue = str | int | float | bool | None
OverrideMap = dict[str, OverrideValue]


def handle_error(ctx: click.Context, error: Exception) -> NoReturn:
    """Handle errors in CLI commands."""
    _ = ctx  # Unused but required for click context
    tui = get_tui()

    # Check if this is formatted error from our event handlers
    error_msg = str(error)
    if not error_msg.startswith("Error: "):
        tui.error(f"Error: {error_msg}")
    else:
        tui.error(error_msg)

    sys.exit(1)


@click.group()
@click.version_option()
def cli():
    """DeepFabric CLI - Generate synthetic training data for language models."""
    pass


class GenerateOptions(BaseModel):
    """
    Validated command options for dataset generation.

    These options can be provided via CLI arguments or a configuration file.
    so they are marked as optional here.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config_file: str | None = None
    dataset_system_prompt: str | None = None
    topic_prompt: str | None = None
    topic_system_prompt: str | None = None
    generation_system_prompt: str | None = None
    save_tree: str | None = None
    load_tree: str | None = None
    save_graph: str | None = None
    load_graph: str | None = None
    dataset_save_as: str | None = None
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    degree: int | None = None
    depth: int | None = None
    num_steps: int | None = None
    batch_size: int | None = None
    base_url: str | None = None
    sys_msg: bool | None = None
    mode: Literal["tree", "graph"] = Field(default="tree")
    debug: bool = False
    topic_only: bool = False

    @model_validator(mode="after")
    def validate_mode_constraints(self) -> "GenerateOptions":
        if self.mode == "graph" and self.save_tree:
            raise ValueError(
                "Cannot use --save-tree when mode is graph. Use --save-graph to persist graph data.",
            )
        if self.mode == "tree" and self.save_graph:
            raise ValueError(
                "Cannot use --save-graph when mode is tree. Use --save-tree to persist tree data.",
            )
        if self.topic_only and (self.load_tree or self.load_graph):
            raise ValueError("--topic-only cannot be used with --load-tree or --load-graph")
        return self


class GenerationPreparation(BaseModel):
    """Validated state required to run dataset generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: DeepFabricConfig
    tree_overrides: OverrideMap = Field(default_factory=dict)
    graph_overrides: OverrideMap = Field(default_factory=dict)
    engine_overrides: OverrideMap = Field(default_factory=dict)
    num_steps: int
    batch_size: int
    depth: int
    degree: int
    loading_existing: bool

    @model_validator(mode="after")
    def validate_positive_dimensions(self) -> "GenerationPreparation":
        if self.num_steps <= 0:
            raise ValueError("num_steps must be greater than zero")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if self.depth <= 0:
            raise ValueError("depth must be greater than zero")
        if self.degree <= 0:
            raise ValueError("degree must be greater than zero")
        return self


def _load_and_prepare_generation_context(options: GenerateOptions) -> GenerationPreparation:
    """Load configuration, compute overrides, and validate derived parameters."""

    config = load_config(
        config_file=options.config_file,
        topic_prompt=options.topic_prompt,
        dataset_system_prompt=options.dataset_system_prompt,
        generation_system_prompt=options.generation_system_prompt,
        provider=options.provider,
        model=options.model,
        temperature=options.temperature,
        degree=options.degree,
        depth=options.depth,
        num_steps=options.num_steps,
        batch_size=options.batch_size,
        save_tree=options.save_tree,
        save_graph=options.save_graph,
        dataset_save_as=options.dataset_save_as,
        sys_msg=options.sys_msg,
        mode=options.mode,
    )

    tree_overrides_raw, graph_overrides_raw, engine_overrides_raw = apply_cli_overrides(
        config=config,
        dataset_system_prompt=options.dataset_system_prompt,
        topic_prompt=options.topic_prompt,
        topic_system_prompt=options.topic_system_prompt,
        generation_system_prompt=options.generation_system_prompt,
        provider=options.provider,
        model=options.model,
        temperature=options.temperature,
        degree=options.degree,
        depth=options.depth,
        base_url=options.base_url,
    )

    final_num_steps, final_batch_size, final_depth, final_degree = get_final_parameters(
        config=config,
        num_steps=options.num_steps,
        batch_size=options.batch_size,
        depth=options.depth,
        degree=options.degree,
    )

    loading_existing = bool(options.load_tree or options.load_graph)

    validate_path_requirements(
        mode=options.mode,
        depth=final_depth,
        degree=final_degree,
        num_steps=final_num_steps,
        batch_size=final_batch_size,
        loading_existing=loading_existing,
    )

    show_validation_success(
        mode=options.mode,
        depth=final_depth,
        degree=final_degree,
        num_steps=final_num_steps,
        batch_size=final_batch_size,
        loading_existing=loading_existing,
    )

    try:
        return GenerationPreparation(
            config=config,
            tree_overrides=cast(OverrideMap, tree_overrides_raw),
            graph_overrides=cast(OverrideMap, graph_overrides_raw),
            engine_overrides=cast(OverrideMap, engine_overrides_raw),
            num_steps=final_num_steps,
            batch_size=final_batch_size,
            depth=final_depth,
            degree=final_degree,
            loading_existing=loading_existing,
        )
    except (ValueError, PydanticValidationError) as error:
        raise ConfigurationError(str(error)) from error


def _initialize_topic_model(
    *,
    preparation: GenerationPreparation,
    options: GenerateOptions,
) -> TopicModel:
    """Load existing topic structures or build new ones and persist when needed."""

    topic_model = load_or_build_topic_model(
        config=preparation.config,
        load_tree=options.load_tree,
        load_graph=options.load_graph,
        tree_overrides=preparation.tree_overrides,
        graph_overrides=preparation.graph_overrides,
        provider=options.provider,
        model=options.model,
        base_url=options.base_url,
        debug=options.debug,
    )

    if not options.load_tree and not options.load_graph:
        save_topic_model(
            topic_model=topic_model,
            config=preparation.config,
            save_tree=options.save_tree,
            save_graph=options.save_graph,
        )

    return topic_model


def _run_generation(
    *,
    preparation: GenerationPreparation,
    topic_model: TopicModel,
    options: GenerateOptions,
) -> None:
    """Create the dataset using the prepared configuration and topic model."""

    engine_params = preparation.config.get_engine_params(**preparation.engine_overrides)
    engine = DataSetGenerator(**engine_params)

    dataset = create_dataset(
        engine=engine,
        topic_model=topic_model,
        config=preparation.config,
        num_steps=preparation.num_steps,
        batch_size=preparation.batch_size,
        sys_msg=options.sys_msg,
        provider=options.provider,
        model=options.model,
        engine_overrides=preparation.engine_overrides,
        debug=options.debug,
    )

    dataset_config = preparation.config.get_dataset_config()
    dataset_save_path = options.dataset_save_as or dataset_config["save_as"]
    save_dataset(dataset, dataset_save_path, preparation.config)

    trace(
        "dataset_generated",
        {"samples": len(dataset.samples) if hasattr(dataset, "samples") else 0},
    )


@cli.command()
@click.argument("config_file", type=click.Path(exists=True), required=False)
@click.option(
    "--dataset-system-prompt", help="System prompt for final dataset (if sys_msg is true)"
)
@click.option("--topic-prompt", help="Starting topic/seed for tree/graph generation")
@click.option("--topic-system-prompt", help="System prompt for tree/graph topic generation")
@click.option("--generation-system-prompt", help="System prompt for dataset content generation")
@click.option("--save-tree", help="Save path for the tree")
@click.option(
    "--load-tree",
    type=click.Path(exists=True),
    help="Path to the JSONL file containing the tree.",
)
@click.option("--save-graph", help="Save path for the graph")
@click.option(
    "--load-graph",
    type=click.Path(exists=True),
    help="Path to the JSON file containing the graph.",
)
@click.option("--dataset-save-as", help="Save path for the dataset")
@click.option("--provider", help="LLM provider (e.g., ollama)")
@click.option("--model", help="Model name (e.g., mistral:latest)")
@click.option("--temperature", type=float, help="Temperature setting")
@click.option("--degree", type=int, help="Degree (branching factor)")
@click.option("--depth", type=int, help="Depth setting")
@click.option("--num-steps", type=int, help="Number of generation steps")
@click.option("--batch-size", type=int, help="Batch size")
@click.option("--base-url", help="Base URL for LLM provider API endpoint")
@click.option(
    "--sys-msg",
    type=bool,
    help="Include system message in dataset (default: true)",
)
@click.option(
    "--mode",
    type=click.Choice(["tree", "graph"]),
    default="tree",
    help="Topic generation mode (default: tree)",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug mode for detailed error output",
)
@click.option(
    "--topic-only",
    is_flag=True,
    help="Only create topic assets, no dataset",
)
def generate(  # noqa: PLR0913
    config_file: str | None,
    dataset_system_prompt: str | None = None,
    topic_prompt: str | None = None,
    topic_system_prompt: str | None = None,
    generation_system_prompt: str | None = None,
    save_tree: str | None = None,
    load_tree: str | None = None,
    save_graph: str | None = None,
    load_graph: str | None = None,
    dataset_save_as: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    degree: int | None = None,
    depth: int | None = None,
    num_steps: int | None = None,
    batch_size: int | None = None,
    base_url: str | None = None,
    sys_msg: bool | None = None,
    mode: Literal["tree", "graph"] = "tree",
    debug: bool = False,
    topic_only: bool = False,
) -> None:
    """Generate training data from a YAML configuration file or CLI parameters."""
    set_trace_debug(debug)
    trace(
        "cli_generate",
        {"mode": mode, "has_config": config_file is not None, "provider": provider, "model": model},
    )

    try:
        options = GenerateOptions(
            config_file=config_file,
            dataset_system_prompt=dataset_system_prompt,
            topic_prompt=topic_prompt,
            topic_system_prompt=topic_system_prompt,
            generation_system_prompt=generation_system_prompt,
            save_tree=save_tree,
            load_tree=load_tree,
            save_graph=save_graph,
            load_graph=load_graph,
            dataset_save_as=dataset_save_as,
            provider=provider,
            model=model,
            temperature=temperature,
            degree=degree,
            depth=depth,
            num_steps=num_steps,
            batch_size=batch_size,
            base_url=base_url,
            sys_msg=sys_msg,
            mode=mode,
            debug=debug,
            topic_only=topic_only,
        )
    except PydanticValidationError as error:
        handle_error(click.get_current_context(), ConfigurationError(str(error)))
        return

    try:
        preparation = _load_and_prepare_generation_context(options)

        topic_model = _initialize_topic_model(
            preparation=preparation,
            options=options,
        )

        if topic_only:
            return

        _run_generation(
            preparation=preparation,
            topic_model=topic_model,
            options=options,
        )

    except ConfigurationError as e:
        handle_error(click.get_current_context(), e)
    except Exception as e:
        tui = get_tui()
        tui.error(f"Unexpected error: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option(
    "--repo",
    required=True,
    help="Hugging Face repository (e.g., username/dataset-name)",
)
@click.option(
    "--token",
    help="Hugging Face API token (can also be set via HF_TOKEN env var)",
)
@click.option(
    "--tags",
    multiple=True,
    help="Tags for the dataset (can be specified multiple times)",
)
def upload(
    dataset_file: str,
    repo: str,
    token: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Upload a dataset to Hugging Face Hub."""
    trace("cli_upload", {"has_tags": len(tags) > 0 if tags else False})

    try:
        # Get token from CLI arg or env var
        token = token or os.getenv("HF_TOKEN")
        if not token:
            handle_error(
                click.get_current_context(),
                ValueError("Hugging Face token not provided. Set via --token or HF_TOKEN env var."),
            )

        # Lazy import to avoid slow startup when not using HF features
        from .hf_hub import HFUploader  # noqa: PLC0415

        uploader = HFUploader(token)
        result = uploader.push_to_hub(str(repo), dataset_file, tags=list(tags) if tags else [])

        tui = get_tui()
        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.error(result["message"])
            sys.exit(1)

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error uploading to Hugging Face Hub: {str(e)}")
        sys.exit(1)


@cli.command("upload-kaggle")
@click.argument("dataset_file", type=click.Path(exists=True))
@click.option(
    "--handle",
    required=True,
    help="Kaggle dataset handle (e.g., username/dataset-name)",
)
@click.option(
    "--username",
    help="Kaggle username (can also be set via KAGGLE_USERNAME env var)",
)
@click.option(
    "--key",
    help="Kaggle API key (can also be set via KAGGLE_KEY env var)",
)
@click.option(
    "--tags",
    multiple=True,
    help="Tags for the dataset (can be specified multiple times)",
)
@click.option(
    "--version-notes",
    help="Version notes for the dataset update",
)
@click.option(
    "--description",
    help="Description for the dataset",
)
def upload_kaggle(
    dataset_file: str,
    handle: str,
    username: str | None = None,
    key: str | None = None,
    tags: list[str] | None = None,
    version_notes: str | None = None,
    description: str | None = None,
) -> None:
    """Upload a dataset to Kaggle."""
    trace("cli_upload_kaggle", {"has_tags": len(tags) > 0 if tags else False})

    try:
        # Get credentials from CLI args or env vars
        username = username or os.getenv("KAGGLE_USERNAME")
        key = key or os.getenv("KAGGLE_KEY")

        if not username or not key:
            handle_error(
                click.get_current_context(),
                ValueError(
                    "Kaggle credentials not provided. "
                    "Set via --username/--key or KAGGLE_USERNAME/KAGGLE_KEY env vars."
                ),
            )

        # Lazy import to avoid slow startup when not using Kaggle features
        from .kaggle_hub import KaggleUploader  # noqa: PLC0415

        uploader = KaggleUploader(username, key)
        result = uploader.push_to_hub(
            str(handle),
            dataset_file,
            tags=list(tags) if tags else [],
            version_notes=version_notes,
            description=description,
        )

        tui = get_tui()
        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.error(result["message"])
            sys.exit(1)

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error uploading to Kaggle: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    required=True,
    help="Output SVG file path",
)
def visualize(graph_file: str, output: str) -> None:
    """Visualize a topic graph as an SVG file."""
    try:
        # Load the graph
        with open(graph_file) as f:
            import json  # noqa: PLC0415

            graph_data = json.load(f)

        # Create a minimal Graph object for visualization
        # We need to get the args from somewhere - for now, use defaults
        from .constants import (  # noqa: PLC0415
            TOPIC_GRAPH_DEFAULT_DEGREE,
            TOPIC_GRAPH_DEFAULT_DEPTH,
        )

        # Create parameters for Graph instantiation
        graph_params = {
            "topic_prompt": "placeholder",  # Not needed for visualization
            "model_name": "placeholder/model",  # Not needed for visualization
            "degree": graph_data.get("degree", TOPIC_GRAPH_DEFAULT_DEGREE),
            "depth": graph_data.get("depth", TOPIC_GRAPH_DEFAULT_DEPTH),
            "temperature": 0.7,  # Default, not used for visualization
        }

        # Use the Graph.from_json method to properly load the graph structure
        import tempfile  # noqa: PLC0415

        # Create a temporary file with the graph data and use from_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(graph_data, tmp_file)
            temp_path = tmp_file.name

        try:
            graph = Graph.from_json(temp_path, graph_params)
        finally:
            import os  # noqa: PLC0415

            os.unlink(temp_path)

        # Visualize the graph
        graph.visualize(output)
        tui = get_tui()
        tui.success(f"Graph visualization saved to: {output}.svg")

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error visualizing graph: {str(e)}")
        sys.exit(1)


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
def validate(config_file: str) -> None:  # noqa: PLR0912
    """Validate a DeepFabric configuration file."""
    try:
        # Try to load the configuration
        config = DeepFabricConfig.from_yaml(config_file)

        # Check required sections
        errors = []
        warnings = []

        # Check for system prompt (with fallback check)
        engine_params = config.get_engine_params()
        if not config.dataset_system_prompt and not engine_params.get("generation_system_prompt"):
            warnings.append("No dataset_system_prompt or generation_system_prompt defined")

        # Check for either topic_tree or topic_graph
        if not config.topic_tree and not config.topic_graph:
            errors.append("Either topic_tree or topic_graph must be defined")

        if config.topic_tree and config.topic_graph:
            warnings.append("Both topic_tree and topic_graph defined - only one will be used")

        # Check data_engine section
        if not config.data_engine:
            errors.append("data_engine section is required")
        elif not config.data_engine.instructions:
            warnings.append("No instructions defined in data_engine")

        # Check dataset section
        if not config.dataset:
            errors.append("dataset section is required")
        else:
            dataset_config = config.get_dataset_config()
            if not dataset_config.get("save_as"):
                warnings.append("No save_as path defined for dataset")

        # Report results
        tui = get_tui()
        if errors:
            tui.error("Configuration validation failed:")
            for error in errors:
                tui.console.print(f"  - {error}", style="red")
            sys.exit(1)
        else:
            tui.success("Configuration is valid")

        if warnings:
            tui.console.print("\nWarnings:", style="yellow bold")
            for warning in warnings:
                tui.warning(warning)

        # Print configuration summary
        tui.console.print("\nConfiguration Summary:", style="cyan bold")
        if config.topic_tree:
            tui.info(
                f"Topic Tree: depth={config.topic_tree.depth}, degree={config.topic_tree.degree}"
            )
        if config.topic_graph:
            tui.info(
                f"Topic Graph: depth={config.topic_graph.depth}, degree={config.topic_graph.degree}"
            )

        dataset_params = config.get_dataset_config()["creation"]
        tui.info(
            f"Dataset: steps={dataset_params['num_steps']}, batch_size={dataset_params['batch_size']}"
        )

        if config.huggingface:
            hf_config = config.get_huggingface_config()
            tui.info(f"Hugging Face: repo={hf_config.get('repository', 'not set')}")

        if config.kaggle:
            kaggle_config = config.get_kaggle_config()
            tui.info(f"Kaggle: handle={kaggle_config.get('handle', 'not set')}")

    except FileNotFoundError:
        handle_error(
            click.get_current_context(),
            ValueError(f"Config file not found: {config_file}"),
        )
    except yaml.YAMLError as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Invalid YAML in config file: {str(e)}"),
        )
    except Exception as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Error validating config file: {str(e)}"),
        )


@cli.command()
def info() -> None:
    """Show DeepFabric version and configuration information."""
    try:
        import importlib.metadata  # noqa: PLC0415

        # Get version
        try:
            version = importlib.metadata.version("deepfabric")
        except importlib.metadata.PackageNotFoundError:
            version = "development"

        tui = get_tui()
        header = tui.create_header(
            f"DeepFabric v{version}", "Large Scale Topic based Synthetic Data Generation"
        )
        tui.console.print(header)

        tui.console.print("\n📋 Available Commands:", style="cyan bold")
        commands = [
            ("generate", "Generate training data from configuration"),
            ("validate", "Validate a configuration file"),
            ("visualize", "Create SVG visualization of a topic graph"),
            ("upload", "Upload dataset to Hugging Face Hub"),
            ("upload-kaggle", "Upload dataset to Kaggle"),
            ("info", "Show this information"),
        ]
        for cmd, desc in commands:
            tui.console.print(f"  [cyan]{cmd}[/cyan] - {desc}")

        tui.console.print("\n🔑 Environment Variables:", style="cyan bold")
        env_vars = [
            ("OPENAI_API_KEY", "OpenAI API key"),
            ("ANTHROPIC_API_KEY", "Anthropic API key"),
            ("HF_TOKEN", "Hugging Face API token"),
        ]
        for var, desc in env_vars:
            tui.console.print(f"  [yellow]{var}[/yellow] - {desc}")

        tui.console.print(
            "\n🔗 For more information, visit: [link]https://github.com/RedDotRocket/deepfabric[/link]"
        )

    except Exception as e:
        tui = get_tui()
        tui.error(f"Error getting info: {str(e)}")
        sys.exit(1)


# Add the format command to the CLI group
cli.add_command(format_cli)

if __name__ == "__main__":
    cli()
