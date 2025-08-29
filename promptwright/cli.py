"""Command-line interface for PromptWright."""

import os
import sys

import click
import yaml

from .config import PromptWrightConfig, construct_model_string
from .engine import DataEngine
from .hf_hub import HFUploader
from .topic_tree import TopicTree, TopicTreeArguments
from .utils import read_topic_tree_from_jsonl


def handle_error(ctx: click.Context, error: Exception) -> None:  # noqa: ARG001
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
@click.option(
    "--hf-token", help="Hugging Face API token (can also be set via HF_TOKEN env var)"
)
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
def start(  # noqa: PLR0912
    config_file: str,
    topic_tree_save_as: str | None = None,
    topic_tree_jsonl: str | None = None,
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
                click.get_current_context(), f"Config file not found: {config_file}"
            )
        except yaml.YAMLError as e:
            handle_error(
                click.get_current_context(), f"Invalid YAML in config file: {str(e)}"
            )
        except Exception as e:
            handle_error(
                click.get_current_context(), f"Error loading config file: {str(e)}"
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

        # Create and build topic tree
        try:
            if topic_tree_jsonl:
                click.echo(f"Reading topic tree from JSONL file: {topic_tree_jsonl}")
                dict_list = read_topic_tree_from_jsonl(topic_tree_jsonl)
                default_args = TopicTreeArguments(
                    root_prompt="default", model_name=model_name
                )
                tree = TopicTree(args=default_args)
                tree.from_dict_list(dict_list)
            else:
                if hasattr(config, "topic_tree"):
                    tree_args = config.get_topic_tree_args(**tree_overrides)
                else:
                    tree_args = TopicTreeArguments(
                        root_prompt="default", model_name=model_name
                    )
                tree = TopicTree(args=tree_args)
                tree.build_tree()
        except Exception as e:
            handle_error(
                click.get_current_context(), f"Error building topic tree: {str(e)}"
            )

        # Save topic tree if JSONL file is not provided
        if not topic_tree_jsonl:
            try:
                tree_save_path = topic_tree_save_as or config.topic_tree.get(
                    "save_as", "topic_tree.jsonl"
                )
                tree.save(tree_save_path)
                click.echo(f"Topic tree saved to: {tree_save_path}")
            except Exception as e:
                handle_error(
                    click.get_current_context(), f"Error saving topic tree: {str(e)}"
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
                click.get_current_context(), f"Error creating data engine: {str(e)}"
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
                topic_tree=tree,
                model_name=model_name,
                sys_msg=sys_msg,  # Pass sys_msg to create_data
            )
        except Exception as e:
            handle_error(
                click.get_current_context(), f"Error creating dataset: {str(e)}"
            )

        # Save dataset
        try:
            dataset_save_path = dataset_save_as or dataset_config.get(
                "save_as", "dataset.jsonl"
            )
            dataset.save(dataset_save_path)
            click.echo(f"Dataset saved to: {dataset_save_path}")
        except Exception as e:
            handle_error(click.get_current_context(), f"Error saving dataset: {str(e)}")

        # Handle Hugging Face upload if configured
        hf_config = config.get_huggingface_config()
        if hf_repo or hf_config.get("repository"):
            try:
                # Get token from CLI arg, env var, or config
                token = hf_token or os.getenv("HF_TOKEN") or hf_config.get("token")
                if not token:
                    handle_error(
                        click.get_current_context(),
                        "Hugging Face token not provided. Set via --hf-token, HF_TOKEN env var, or config file.",
                    )

                # Get repository from CLI arg or config
                repo = hf_repo or hf_config.get("repository")
                if not repo:
                    handle_error(
                        click.get_current_context(),
                        "Hugging Face repository not provided. Set via --hf-repo or config file.",
                    )

                # Get tags from CLI args and config
                config_tags = hf_config.get("tags", [])
                all_tags = list(hf_tags) if hf_tags else []
                all_tags.extend(config_tags)

                # Upload to Hugging Face
                uploader = HFUploader(token)
                result = uploader.push_to_hub(repo, dataset_save_path, tags=all_tags)

                if result["status"] == "success":
                    click.echo(result["message"])
                else:
                    handle_error(click.get_current_context(), result["message"])

            except Exception as e:
                handle_error(
                    click.get_current_context(),
                    f"Error uploading to Hugging Face Hub: {str(e)}",
                )

    except Exception as e:
        handle_error(click.get_current_context(), f"Unexpected error: {str(e)}")


if __name__ == "__main__":
    cli()
