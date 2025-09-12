import os
import sys

from typing import NoReturn

import click
import yaml

from .config import DeepFabricConfig, construct_model_string
from .constants import (
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .factory import create_topic_generator
from .generator import DataSetGenerator
from .graph import Graph
from .tree import Tree, TreeArguments
from .utils import read_topic_tree_from_jsonl


def handle_error(ctx: click.Context, error: Exception) -> NoReturn:  # noqa: ARG001
    """Handle errors in CLI commands."""
    click.echo(f"Error: {str(error)}", err=True)
    sys.exit(1)


@click.group()
@click.version_option()
def cli():
    """DeepFabric CLI - Generate synthetic training data for language models."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--save-tree", help="Override the save path for the tree")
@click.option(
    "--load-tree",
    type=click.Path(exists=True),
    help="Path to the JSONL file containing the tree.",
)
@click.option("--save-graph", help="Override the save path for the graph")
@click.option(
    "--load-graph",
    type=click.Path(exists=True),
    help="Path to the JSON file containing the graph.",
)
@click.option("--dataset-save-as", help="Override the save path for the dataset")
@click.option("--provider", help="Override the LLM provider (e.g., ollama)")
@click.option("--model", help="Override the model name (e.g., mistral:latest)")
@click.option("--temperature", type=float, help="Override the temperature")
@click.option("--tree-degree", type=int, help="Override the tree degree")
@click.option("--tree-depth", type=int, help="Override the tree depth")
@click.option("--graph-degree", type=int, help="Override the graph degree")
@click.option("--graph-depth", type=int, help="Override the graph depth")
@click.option("--num-steps", type=int, help="Override number of generation steps")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option(
    "--sys-msg",
    type=bool,
    help="Include system message in dataset (default: true)",
)
def generate(  # noqa: PLR0912, PLR0913
    config_file: str,
    save_tree: str | None = None,
    load_tree: str | None = None,
    save_graph: str | None = None,
    load_graph: str | None = None,
    graph_degree: int | None = None,
    graph_depth: int | None = None,
    dataset_save_as: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    tree_degree: int | None = None,
    tree_depth: int | None = None,
    num_steps: int | None = None,
    batch_size: int | None = None,
    sys_msg: bool | None = None,
) -> None:
    """Generate training data from a YAML configuration file."""
    try:
        # Load configuration
        try:
            config = DeepFabricConfig.from_yaml(config_file)
        except FileNotFoundError:
            handle_error(
                click.get_current_context(), ValueError(f"Config file not found: {config_file}")
            )
        except yaml.YAMLError as e:
            handle_error(
                click.get_current_context(), ValueError(f"Invalid YAML in config file: {str(e)}")
            )
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error loading config file: {str(e)}")
            )
        # Get dataset parameters
        dataset_config = config.get_dataset_config()
        dataset_params = dataset_config.get("creation", {})

        # Prepare topic tree overrides
        tree_overrides = {}
        if provider:
            tree_overrides["provider"] = provider
        if model:
            tree_overrides["model"] = model
        if temperature:
            tree_overrides["temperature"] = temperature
        if tree_degree:
            tree_overrides["tree_degree"] = tree_degree
        if tree_depth:
            tree_overrides["tree_depth"] = tree_depth

        # Prepare topic graph overrides
        graph_overrides = {}
        if provider:
            graph_overrides["provider"] = provider
        if model:
            graph_overrides["model"] = model
        if temperature:
            graph_overrides["temperature"] = temperature
        if graph_degree:
            graph_overrides["graph_degree"] = graph_degree
        if graph_depth:
            graph_overrides["graph_depth"] = graph_depth

        # Construct model name
        model_name = construct_model_string(
            provider or dataset_params.get("provider", "default_provider"),
            model or dataset_params.get("model", "default_model"),
        )

        # Create and build topic model
        try:
            if load_tree:
                click.echo(f"Reading topic tree from JSONL file: {load_tree}")
                dict_list = read_topic_tree_from_jsonl(load_tree)
                default_args = TreeArguments(
                    root_prompt="default",
                    model_name=model_name,
                    model_system_prompt="",
                    tree_degree=TOPIC_TREE_DEFAULT_DEGREE,
                    tree_depth=TOPIC_TREE_DEFAULT_DEPTH,
                    temperature=TOPIC_TREE_DEFAULT_TEMPERATURE,
                )
                topic_model = Tree(args=default_args)
                topic_model.from_dict_list(dict_list)
            elif load_graph:
                click.echo(f"Reading topic graph from JSON file: {load_graph}")
                graph_args = config.get_topic_graph_args(**graph_overrides)
                topic_model = Graph.from_json(load_graph, graph_args)
            else:
                topic_model = create_topic_generator(
                    config, tree_overrides=tree_overrides, graph_overrides=graph_overrides
                )
                topic_model.build()
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error building topic model: {str(e)}")
            )

        # Save topic model
        if not load_tree and not load_graph:
            if isinstance(topic_model, Tree):
                try:
                    tree_save_path = save_tree or (config.topic_tree or {}).get(
                        "save_as", "topic_tree.jsonl"
                    )
                    topic_model.save(tree_save_path)
                    click.echo(f"Topic tree saved to: {tree_save_path}")
                except Exception as e:
                    handle_error(
                        click.get_current_context(),
                        ValueError(f"Error saving topic tree: {str(e)}"),
                    )
            elif isinstance(topic_model, Graph):
                try:
                    graph_save_path = save_graph or (config.topic_graph or {}).get(
                        "save_as", "topic_graph.json"
                    )
                    topic_model.save(graph_save_path)
                    click.echo(f"Topic graph saved to: {graph_save_path}")
                except Exception as e:
                    handle_error(
                        click.get_current_context(),
                        ValueError(f"Error saving topic graph: {str(e)}"),
                    )

        # Prepare engine overrides
        engine_overrides = {}
        if provider:
            engine_overrides["provider"] = provider
        if model:
            engine_overrides["model"] = model
        if temperature:
            engine_overrides["temperature"] = temperature

        # Create data engine
        try:
            engine = DataSetGenerator(args=config.get_engine_args(**engine_overrides))
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error creating data engine: {str(e)}")
            )

        # Construct model name for dataset creation
        model_name = construct_model_string(
            provider or dataset_params.get("provider", "ollama"),
            model or dataset_params.get("model", "mistral:latest"),
        )

        # Create dataset with overrides
        try:
            dataset = engine.create_data(
                num_steps=num_steps or dataset_params.get("num_steps", 5),
                batch_size=batch_size or dataset_params.get("batch_size", 1),
                topic_model=topic_model,
                model_name=model_name,
                sys_msg=sys_msg,  # Pass sys_msg to create_data
            )
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error creating dataset: {str(e)}")
            )

        # Save dataset
        try:
            dataset_save_path = dataset_save_as or dataset_config.get("save_as", "dataset.jsonl")
            dataset.save(dataset_save_path)
            click.echo(f"Dataset saved to: {dataset_save_path}")
        except Exception as e:
            handle_error(click.get_current_context(), Exception(f"Error saving dataset: {str(e)}"))

    except Exception as e:
        handle_error(click.get_current_context(), ValueError(f"Unexpected error: {str(e)}"))


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

        if result["status"] == "success":
            click.echo(result["message"])
        else:
            handle_error(click.get_current_context(), ValueError(result["message"]))

    except Exception as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Error uploading to Hugging Face Hub: {str(e)}"),
        )


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
        from .graph import GraphArguments  # noqa: PLC0415

        args = GraphArguments(
            root_prompt="placeholder",  # Not needed for visualization
            model_name="placeholder/model",  # Not needed for visualization
            graph_degree=graph_data.get("degree", TOPIC_GRAPH_DEFAULT_DEGREE),
            graph_depth=graph_data.get("depth", TOPIC_GRAPH_DEFAULT_DEPTH),
            temperature=0.7,  # Default, not used for visualization
        )

        # Use the Graph.from_json method to properly load the graph structure
        import tempfile  # noqa: PLC0415

        # Create a temporary file with the graph data and use from_json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(graph_data, tmp_file)
            temp_path = tmp_file.name

        try:
            graph = Graph.from_json(temp_path, args)
        finally:
            import os  # noqa: PLC0415

            os.unlink(temp_path)

        # Visualize the graph
        graph.visualize(output)
        click.echo(f"Graph visualization saved to: {output}.svg")

    except Exception as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Error visualizing graph: {str(e)}"),
        )


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

        # Check for system prompt
        if not config.system_prompt:
            warnings.append("No system_prompt defined")

        # Check for either topic_tree or topic_graph
        if not config.topic_tree and not config.topic_graph:
            errors.append("Either topic_tree or topic_graph must be defined")

        if config.topic_tree and config.topic_graph:
            warnings.append("Both topic_tree and topic_graph defined - only one will be used")

        # Check data_engine section
        if not config.data_engine:
            errors.append("data_engine section is required")
        else:
            engine_args = config.data_engine.get("args", {})
            if not engine_args.get("instructions"):
                warnings.append("No instructions defined in data_engine")

        # Check dataset section
        if not config.dataset:
            errors.append("dataset section is required")
        else:
            dataset_config = config.get_dataset_config()
            if not dataset_config.get("save_as"):
                warnings.append("No save_as path defined for dataset")

        # Report results
        if errors:
            click.echo("❌ Configuration validation failed:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(1)
        else:
            click.echo("✅ Configuration is valid")

        if warnings:
            click.echo("\nWarnings:")
            for warning in warnings:
                click.echo(f"  ⚠️  {warning}")

        # Print configuration summary
        click.echo("\nConfiguration Summary:")
        if config.topic_tree:
            tree_args = config.topic_tree.get("args", {})
            click.echo(
                f"  Topic Tree: depth={tree_args.get('tree_depth', 'default')}, degree={tree_args.get('tree_degree', 'default')}"
            )
        if config.topic_graph:
            graph_args = config.topic_graph.get("args", {})
            click.echo(
                f"  Topic Graph: depth={graph_args.get('graph_depth', 'default')}, degree={graph_args.get('graph_degree', 'default')}"
            )

        dataset_params = config.get_dataset_config().get("creation", {})
        click.echo(
            f"  Dataset: steps={dataset_params.get('num_steps', 'default')}, batch_size={dataset_params.get('batch_size', 'default')}"
        )

        if config.huggingface:
            hf_config = config.get_huggingface_config()
            click.echo(f"  Hugging Face: repo={hf_config.get('repository', 'not set')}")

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

        click.echo(f"DeepFabric v{version}")
        click.echo("\nAvailable Commands:")
        click.echo("  generate   - Generate training data from configuration")
        click.echo("  validate   - Validate a configuration file")
        click.echo("  visualize  - Create SVG visualization of a topic graph")
        click.echo("  upload     - Upload dataset to Hugging Face Hub")
        click.echo("  info       - Show this information")

        click.echo("\nEnvironment Variables:")
        click.echo("  OPENAI_API_KEY     - OpenAI API key")
        click.echo("  ANTHROPIC_API_KEY  - Anthropic API key")
        click.echo("  HF_TOKEN           - Hugging Face API token")

        click.echo("\nFor more information, visit: https://github.com/stacklok/deepfabric")

    except Exception as e:
        handle_error(
            click.get_current_context(),
            ValueError(f"Error getting info: {str(e)}"),
        )


if __name__ == "__main__":
    cli()
