import os
import sys

from typing import NoReturn

import click
import yaml

from .config import PromptWrightConfig, construct_model_string
from .constants import (
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .engine import DataEngine
from .factory import create_topic_generator
from .topic_graph import TopicGraph
from .topic_tree import TopicTree, TopicTreeArguments
from .utils import read_topic_tree_from_jsonl


def handle_error(ctx: click.Context, error: Exception) -> NoReturn:  # noqa: ARG001
    """Handle errors in CLI commands."""
    click.echo(f"Error: {str(error)}", err=True)
    sys.exit(1)


@click.group()
def cli():
    """PromptWright CLI - Generate training data for language models."""
    pass


@cli.command()
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--topic-tree-save-as", help="Override the save path for the topic tree")
@click.option(
    "--topic-tree-jsonl",
    type=click.Path(exists=True),
    help="Path to the JSONL file containing the topic tree.",
)
@click.option("--graph-save-as", help="Override the save path for the topic graph")
@click.option(
    "--graph-load-from",
    type=click.Path(exists=True),
    help="Path to the JSON file containing the topic graph.",
)
@click.option("--graph-visualize-as", help="Save a visualization of the graph to an SVG file")
@click.option("--dataset-save-as", help="Override the save path for the dataset")
@click.option("--provider", help="Override the LLM provider (e.g., ollama)")
@click.option("--model", help="Override the model name (e.g., mistral:latest)")
@click.option("--temperature", type=float, help="Override the temperature")
@click.option("--tree-degree", type=int, help="Override the tree degree")
@click.option("--tree-depth", type=int, help="Override the tree depth")
@click.option("--num-steps", type=int, help="Override number of generation steps")
@click.option("--batch-size", type=int, help="Override batch size")
@click.option(
    "--hf-repo",
    help="Hugging Face repository to upload dataset (e.g., username/dataset-name)",
)
@click.option("--hf-token", help="Hugging Face API token (can also be set via HF_TOKEN env var)")
@click.option(
    "--hf-tags",
    multiple=True,
    help="Additional tags for the dataset (can be specified multiple times)",
)
@click.option(
    "--sys-msg",
    type=bool,
    help="Include system message in dataset (default: true)",
)
def start(  # noqa: PLR0912, PLR0913
    config_file: str,
    topic_tree_save_as: str | None = None,
    topic_tree_jsonl: str | None = None,
    graph_save_as: str | None = None,
    graph_load_from: str | None = None,
    graph_visualize_as: str | None = None,
    dataset_save_as: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    tree_degree: int | None = None,
    tree_depth: int | None = None,
    num_steps: int | None = None,
    batch_size: int | None = None,
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_tags: list[str] | None = None,
    sys_msg: bool | None = None,
) -> None:
    """Generate training data from a YAML configuration file."""
    try:
        # Load configuration
        try:
            config = PromptWrightConfig.from_yaml(config_file)
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

        # Construct model name
        model_name = construct_model_string(
            provider or dataset_params.get("provider", "default_provider"),
            model or dataset_params.get("model", "default_model"),
        )

        # Create and build topic model
        try:
            if topic_tree_jsonl:
                click.echo(f"Reading topic tree from JSONL file: {topic_tree_jsonl}")
                dict_list = read_topic_tree_from_jsonl(topic_tree_jsonl)
                default_args = TopicTreeArguments(
                    root_prompt="default",
                    model_name=model_name,
                    model_system_prompt="",
                    tree_degree=TOPIC_TREE_DEFAULT_DEGREE,
                    tree_depth=TOPIC_TREE_DEFAULT_DEPTH,
                    temperature=TOPIC_TREE_DEFAULT_TEMPERATURE,
                )
                topic_model = TopicTree(args=default_args)
                topic_model.from_dict_list(dict_list)
            elif graph_load_from:
                click.echo(f"Reading topic graph from JSON file: {graph_load_from}")
                graph_args = config.get_topic_graph_args(**tree_overrides)
                topic_model = TopicGraph.from_json(graph_load_from, graph_args)
            else:
                topic_model = create_topic_generator(config)
                topic_model.build()
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error building topic model: {str(e)}")
            )

        # Save topic model
        if not topic_tree_jsonl and not graph_load_from:
            if isinstance(topic_model, TopicTree):
                try:
                    tree_save_path = topic_tree_save_as or (config.topic_tree or {}).get(
                        "save_as", "topic_tree.jsonl"
                    )
                    topic_model.save(tree_save_path)
                    click.echo(f"Topic tree saved to: {tree_save_path}")
                except Exception as e:
                    handle_error(
                        click.get_current_context(),
                        ValueError(f"Error saving topic tree: {str(e)}"),
                    )
            elif isinstance(topic_model, TopicGraph):
                try:
                    graph_save_path = graph_save_as or (config.topic_graph or {}).get(
                        "save_as", "topic_graph.json"
                    )
                    topic_model.save(graph_save_path)
                    click.echo(f"Topic graph saved to: {graph_save_path}")
                except Exception as e:
                    handle_error(
                        click.get_current_context(),
                        ValueError(f"Error saving topic graph: {str(e)}"),
                    )

        # Visualize graph if requested
        if isinstance(topic_model, TopicGraph) and graph_visualize_as:
            try:
                topic_model.visualize(graph_visualize_as)
                click.echo(f"Graph visualization saved to: {graph_visualize_as}.svg")
            except Exception as e:
                handle_error(
                    click.get_current_context(), ValueError(f"Error visualizing graph: {str(e)}")
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
            engine = DataEngine(args=config.get_engine_args(**engine_overrides))
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

        # Handle Hugging Face upload if configured
        hf_config = config.get_huggingface_config()
        if hf_repo or hf_config.get("repository"):
            try:
                # Get token from CLI arg, env var, or config
                token = hf_token or os.getenv("HF_TOKEN") or hf_config.get("token")
                if not token:
                    handle_error(
                        click.get_current_context(),
                        ValueError(
                            "Hugging Face token not provided. Set via --hf-token, HF_TOKEN env var, or config file."
                        ),
                    )

                # Get repository from CLI arg or config
                repo = hf_repo or hf_config.get("repository")
                if not repo:
                    handle_error(
                        click.get_current_context(),
                        ValueError(
                            "Hugging Face repository not provided. Set via --hf-repo or config file."
                        ),
                    )

                # Get tags from CLI args and config
                config_tags = hf_config.get("tags", [])
                all_tags = list(hf_tags) if hf_tags else []
                all_tags.extend(config_tags)

                # Upload to Hugging Face
                # Lazy import to avoid slow startup when not using HF features
                from .hf_hub import HFUploader  # noqa: PLC0415

                uploader = HFUploader(token)
                result = uploader.push_to_hub(str(repo), dataset_save_path, tags=all_tags)

                if result["status"] == "success":
                    click.echo(result["message"])
                else:
                    handle_error(click.get_current_context(), ValueError(result["message"]))

            except Exception as e:
                handle_error(
                    click.get_current_context(),
                    ValueError(f"Error uploading to Hugging Face Hub: {str(e)}"),
                )

    except Exception as e:
        handle_error(click.get_current_context(), ValueError(f"Unexpected error: {str(e)}"))


if __name__ == "__main__":
    cli()
