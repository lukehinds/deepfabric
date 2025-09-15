import os
import sys
import tempfile

from typing import NoReturn

import click
import yaml

from .config import DeepFabricConfig
from .constants import (
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .dataset import Dataset
from .factory import create_topic_generator
from .generator import DataSetGenerator
from .graph import Graph
from .metrics import trace
from .tree import Tree
from .tui import get_dataset_tui, get_graph_tui, get_tree_tui, get_tui
from .utils import read_topic_tree_from_jsonl


def calculate_expected_paths(mode: str, depth: int, degree: int) -> int:
    """Calculate expected number of paths for tree/graph generation."""
    if mode == "tree":
        # Tree paths = degree^depth (exact - each leaf is a unique path)
        return degree**depth
    # mode == "graph"
    # Graph paths vary widely due to cross-connections
    # Can range from degree^depth * 0.5 to degree^depth * 2+
    # Use base estimate as rough middle ground, but warn it's approximate
    return degree**depth


def validate_path_requirements(
    mode: str,
    depth: int,
    degree: int,
    num_steps: int,
    batch_size: int,
    loading_existing: bool = False,
) -> None:
    """Validate that the topic generation parameters will produce enough paths."""
    if loading_existing:
        # Can't validate existing files without loading them
        return

    expected_paths = calculate_expected_paths(mode, depth, degree)
    required_samples = num_steps * batch_size

    if required_samples > expected_paths:
        # Alternative: provide exact combinations that use all paths
        optimal_combinations = []
        for test_steps in range(1, expected_paths + 1):
            test_batch = expected_paths // test_steps
            if test_steps * test_batch <= expected_paths and test_batch > 0:
                optimal_combinations.append((test_steps, test_batch))

        # Sort by preference (fewer steps first, then larger batches)
        optimal_combinations.sort(key=lambda x: (x[0], -x[1]))

        tui = get_tui()
        tui.error(" Path validation failed - stopping before topic generation")

        # Build recommendations - focus on optimal combinations rather than misleading individual params
        recommendations = []

        if optimal_combinations:
            recommendations.append(
                f"  • Use one of these combinations to utilize the {expected_paths} paths:"
            )
            for steps, batch in optimal_combinations[:3]:  # Show top 3
                total_samples = steps * batch
                recommendations.append(
                    f"    --num-steps {steps} --batch-size {batch}  (generates {total_samples} samples)"
                )

        recommendations.extend(
            [
                f"  • Or increase --depth (currently {depth}) or --degree (currently {degree})",
            ]
        )

        estimation_note = ""
        if mode == "graph":
            estimation_note = " (estimated - graphs vary due to cross-connections)"

        error_msg = (
            f"Insufficient expected paths for dataset generation:\n"
            f"  • Expected {mode} paths: ~{expected_paths}{estimation_note} (depth={depth}, degree={degree})\n"
            f"  • Requested samples: {required_samples} ({num_steps} steps × {batch_size} batch size)\n"
            f"  • Shortfall: ~{required_samples - expected_paths} samples\n\n"
            f"Recommendations:\n" + "\n".join(recommendations)
        )

        if mode == "graph":
            error_msg += f"\n\nNote: Graph path counts are estimates. The actual graph may produce {expected_paths // 2}-{expected_paths * 2} paths due to cross-connections."

        handle_error(click.get_current_context(), ValueError(error_msg))


def handle_graph_events(graph):
    """Build graph with TUI progress."""
    tui = get_graph_tui()
    tui_started = False

    final_event = None
    try:
        for event in graph.build():
            if event["event"] == "depth_start":
                if not tui_started:
                    tui.start_building(graph.model_name, graph.depth, graph.degree)
                    tui_started = True
                tui.start_depth_level(event["depth"], event.get("leaf_count", 0))
            elif event["event"] == "node_expanded":
                tui.complete_node_expansion(
                    event["node_topic"], event["subtopics_added"], event["connections_added"]
                )
            elif event["event"] == "depth_complete":
                tui.complete_depth_level(event["depth"])
            elif event["event"] == "build_complete":
                tui.finish_building(event.get("failed_generations", 0))
                final_event = event
    except Exception as e:
        # The LLM module now handles proper error formatting
        get_tui().error(f"Graph build failed: {str(e)}")
        raise

    return final_event


def handle_tree_events(tree):
    """Build tree with TUI progress."""
    tui = get_tree_tui()

    final_event = None
    try:
        for event in tree.build():
            if event["event"] == "build_start":
                tui.start_building(event["model_name"], event["depth"], event["degree"])
            elif event["event"] == "subtopics_generated":
                if not event["success"]:
                    tui.add_failure()
            elif event["event"] == "build_complete":
                tui.finish_building(event["total_paths"], event["failed_generations"])
                final_event = event
    except Exception as e:
        # The LLM module now handles proper error formatting
        get_tui().error(f"Tree build failed: {str(e)}")
        raise

    return final_event


def handle_dataset_events(generator) -> Dataset | None:  # noqa: PLR0912
    """Handle dataset generation with TUI progress."""
    tui = get_dataset_tui()
    progress = None
    task = None

    final_result: Dataset | None = None
    try:
        for event in generator:
            if isinstance(event, dict) and "event" in event:
                if event["event"] == "generation_start":
                    tui.show_generation_header(
                        event["model_name"], event["num_steps"], event["batch_size"]
                    )
                    progress = tui.create_rich_progress()
                    progress.start()
                    task = progress.add_task(
                        "  Generating dataset samples", total=event["total_samples"]
                    )
                elif event["event"] == "step_complete":
                    if progress and task is not None:
                        samples_generated = event.get("samples_generated", 0)
                        if samples_generated > 0:
                            progress.update(task, advance=samples_generated)
                elif event["event"] == "generation_complete":
                    if progress:
                        progress.stop()
                    tui.success(f"Successfully generated {event['total_samples']} samples")
                    if event["failed_samples"] > 0:
                        tui.warning(f"Failed to generate {event['failed_samples']} samples")
            elif isinstance(event, Dataset):
                final_result = event
            else:
                # Handle unexpected non-dict, non-Dataset events
                get_tui().warning(f"Unexpected event type: {type(event)}")
    except Exception as e:
        if progress:
            progress.stop()
        get_tui().error(f"Dataset generation failed: {str(e)}")
        raise

    return final_result


def handle_error(ctx: click.Context, error: Exception) -> NoReturn:  # noqa: ARG001
    """Handle errors in CLI commands."""
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
def generate(  # noqa: PLR0912, PLR0913
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
    mode: str = "tree",
) -> None:
    """Generate training data from a YAML configuration file or CLI parameters."""
    trace(
        "cli_generate",
        {"mode": mode, "has_config": config_file is not None, "provider": provider, "model": model},
    )

    try:
        # Load configuration or create minimal config from CLI args
        if config_file:
            try:
                config = DeepFabricConfig.from_yaml(config_file)
            except FileNotFoundError:
                handle_error(
                    click.get_current_context(), ValueError(f"Config file not found: {config_file}")
                )
            except yaml.YAMLError as e:
                handle_error(
                    click.get_current_context(),
                    ValueError(f"Invalid YAML in config file: {str(e)}"),
                )
            except Exception as e:
                handle_error(
                    click.get_current_context(), ValueError(f"Error loading config file: {str(e)}")
                )
        else:
            # No config file provided - validate required CLI parameters
            if not topic_prompt:
                handle_error(
                    click.get_current_context(),
                    ValueError("--topic-prompt is required when no config file is provided"),
                )

            # Create minimal configuration from CLI args
            tui = get_tui()
            tui.info("No config file provided - using CLI parameters")

            # Create a minimal config dict
            # Use generation_system_prompt as fallback for dataset_system_prompt if not provided
            default_prompt = generation_system_prompt or "You are a helpful AI assistant."
            minimal_config = {
                "dataset_system_prompt": dataset_system_prompt,  # Can be None, will fall back in config
                "data_engine": {
                    "instructions": "Generate diverse and educational examples",
                    "generation_system_prompt": default_prompt,
                    "provider": provider or "gemini",
                    "model": model or "gemini-2.5-flash-lite",
                    "temperature": temperature or 0.9,
                    "max_retries": 3,
                },
                "dataset": {
                    "creation": {
                        "num_steps": num_steps or 5,
                        "batch_size": batch_size or 2,
                        "provider": provider or "gemini",
                        "model": model or "gemini-2.5-flash-lite",
                        "sys_msg": sys_msg if sys_msg is not None else True,
                    },
                    "save_as": dataset_save_as or "dataset.jsonl",
                },
            }

            # Add topic generation config based on mode parameter
            if mode == "graph":
                minimal_config["topic_graph"] = {
                    "topic_prompt": topic_prompt,
                    "provider": provider or "gemini",
                    "model": model or "gemini-2.5-flash-lite",
                    "temperature": temperature or 0.7,
                    "degree": degree or 3,
                    "depth": depth or 2,
                    "save_as": save_graph or "topic_graph.json",
                }
            else:  # mode == "tree" (default)
                minimal_config["topic_tree"] = {
                    "topic_prompt": topic_prompt,
                    "provider": provider or "gemini",
                    "model": model or "gemini-2.5-flash-lite",
                    "temperature": temperature or 0.7,
                    "degree": degree or 3,
                    "depth": depth or 3,
                    "save_as": save_tree or "topic_tree.jsonl",
                }

            # Create config object from dict

            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump(minimal_config, f)
                temp_config_path = f.name

            try:
                config = DeepFabricConfig.from_yaml(temp_config_path)
            finally:
                os.unlink(temp_config_path)

        # Apply dataset system prompt override if provided
        if dataset_system_prompt:
            config.dataset_system_prompt = dataset_system_prompt

        # Get dataset parameters
        dataset_config = config.get_dataset_config()
        dataset_params = dataset_config.get("creation", {})

        # Prepare topic tree overrides
        tree_overrides = {}
        if topic_prompt:
            tree_overrides["topic_prompt"] = topic_prompt
        if topic_system_prompt:
            tree_overrides["topic_system_prompt"] = topic_system_prompt
        if provider:
            tree_overrides["provider"] = provider
        if model:
            tree_overrides["model"] = model
        if temperature:
            tree_overrides["temperature"] = temperature
        if degree:
            tree_overrides["degree"] = degree
        if depth:
            tree_overrides["depth"] = depth
        if base_url:
            tree_overrides["base_url"] = base_url

        # Prepare topic graph overrides
        graph_overrides = {}
        if topic_prompt:
            graph_overrides["topic_prompt"] = topic_prompt
        if topic_system_prompt:
            graph_overrides["topic_system_prompt"] = topic_system_prompt
        if provider:
            graph_overrides["provider"] = provider
        if model:
            graph_overrides["model"] = model
        if temperature:
            graph_overrides["temperature"] = temperature
        if degree:
            graph_overrides["degree"] = degree
        if depth:
            graph_overrides["depth"] = depth
        if base_url:
            graph_overrides["base_url"] = base_url

        # Set provider and model
        final_provider = provider or dataset_params.get("provider", "ollama")
        final_model = model or dataset_params.get("model", "mistral:latest")

        # Validate path requirements before topic generation
        final_num_steps = num_steps or dataset_params.get("num_steps", 5)
        final_batch_size = batch_size or dataset_params.get("batch_size", 1)
        final_depth = depth or (config.topic_tree or config.topic_graph or {}).get("args", {}).get(
            "depth", 3
        )
        final_degree = degree or (config.topic_tree or config.topic_graph or {}).get(
            "args", {}
        ).get("degree", 3)

        # Early validation to prevent wasted token usage
        validate_path_requirements(
            mode=mode,
            depth=final_depth,
            degree=final_degree,
            num_steps=final_num_steps,
            batch_size=final_batch_size,
            loading_existing=bool(load_tree or load_graph),
        )

        # Show validation passed message
        if not (load_tree or load_graph):
            expected_paths = calculate_expected_paths(mode, final_depth, final_degree)
            total_samples = final_num_steps * final_batch_size

            tui = get_tui()
            tui.success("📊 Path Validation Passed")
            tui.info(
                f"  • Expected {mode} paths: ~{expected_paths} (depth={final_depth}, degree={final_degree})"
            )
            tui.info(
                f"  • Requested samples: {total_samples} ({final_num_steps} steps × {final_batch_size} batch size)"
            )
            tui.info(
                f"  • Path utilization: ~{min(100, (total_samples / expected_paths) * 100):.1f}%"
            )

            if mode == "graph":
                tui.info("  • Note: Graph paths may vary due to cross-connections")
            print()  # Extra space before topic generation

        # Create and build topic model
        try:
            tui = get_tui()
            if load_tree:
                tui.info(f"Reading topic tree from JSONL file: {load_tree}")
                dict_list = read_topic_tree_from_jsonl(load_tree)
                topic_model = Tree(
                    topic_prompt="default",
                    provider=final_provider,
                    model_name=final_model,
                    topic_system_prompt="",
                    degree=TOPIC_TREE_DEFAULT_DEGREE,
                    depth=TOPIC_TREE_DEFAULT_DEPTH,
                    temperature=TOPIC_TREE_DEFAULT_TEMPERATURE,
                    base_url=base_url,
                )
                topic_model.from_dict_list(dict_list)
            elif load_graph:
                tui.info(f"Reading topic graph from JSON file: {load_graph}")
                graph_params = config.get_topic_graph_params(**graph_overrides)
                topic_model = Graph.from_json(load_graph, graph_params)
            else:
                topic_model = create_topic_generator(
                    config, tree_overrides=tree_overrides, graph_overrides=graph_overrides
                )
                # Build with appropriate event handler
                if isinstance(topic_model, Graph):
                    handle_graph_events(topic_model)
                elif isinstance(topic_model, Tree):
                    handle_tree_events(topic_model)
        except Exception as e:
            handle_error(click.get_current_context(), e)

        # Save topic model (TUI messaging is handled in save methods)
        if not load_tree and not load_graph:
            if isinstance(topic_model, Tree):
                try:
                    tree_save_path = save_tree or (config.topic_tree or {}).get(
                        "save_as", "topic_tree.jsonl"
                    )
                    topic_model.save(tree_save_path)
                    tui = get_tui()
                    tui.success(f"Topic tree saved to {tree_save_path}")
                    tui.info(f"Total paths: {len(topic_model.tree_paths)}")
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
                    tui = get_tui()
                    tui.success(f"Topic graph saved to {graph_save_path}")
                except Exception as e:
                    handle_error(
                        click.get_current_context(),
                        ValueError(f"Error saving topic graph: {str(e)}"),
                    )

        # Prepare engine overrides
        engine_overrides = {}
        if generation_system_prompt:
            engine_overrides["generation_system_prompt"] = generation_system_prompt
        if provider:
            engine_overrides["provider"] = provider
        if model:
            engine_overrides["model"] = model
        if temperature:
            engine_overrides["temperature"] = temperature
        if base_url:
            engine_overrides["base_url"] = base_url

        # Create data engine
        try:
            engine = DataSetGenerator(**config.get_engine_params(**engine_overrides))
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error creating data engine: {str(e)}")
            )

        # Set provider and model for dataset creation
        engine_params = config.get_engine_params(**engine_overrides)
        final_provider = provider or engine_params.get("provider", "ollama")
        final_model = model or engine_params.get("model_name", "mistral:latest")

        # Create dataset with overrides - using generator pattern for TUI
        try:
            generator = engine.create_data_with_events(
                num_steps=num_steps or dataset_params.get("num_steps", 5),
                batch_size=batch_size or dataset_params.get("batch_size", 1),
                topic_model=topic_model,
                model_name=final_model,
                sys_msg=sys_msg,
                num_example_demonstrations=dataset_params.get("num_example_demonstrations", 3),
            )
            dataset = handle_dataset_events(generator)
        except Exception as e:
            handle_error(
                click.get_current_context(), ValueError(f"Error creating dataset: {str(e)}")
            )

        # Validate dataset was created
        if dataset is None:
            handle_error(
                click.get_current_context(),
                ValueError("Dataset generation failed - no dataset returned"),
            )

        # Save dataset
        try:
            dataset_save_path = dataset_save_as or dataset_config.get("save_as", "dataset.jsonl")
            dataset.save(dataset_save_path)
            tui.success(f"Dataset saved to: {dataset_save_path}")
            trace(
                "dataset_generated",
                {"samples": len(dataset.samples) if hasattr(dataset, "samples") else 0},
            )
        except Exception as e:
            handle_error(click.get_current_context(), Exception(f"Error saving dataset: {str(e)}"))

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
            tree_args = config.topic_tree.get("args", {})
            tui.info(
                f"Topic Tree: depth={tree_args.get('depth', 'default')}, degree={tree_args.get('degree', 'default')}"
            )
        if config.topic_graph:
            graph_args = config.topic_graph.get("args", {})
            tui.info(
                f"Topic Graph: depth={graph_args.get('depth', 'default')}, degree={graph_args.get('degree', 'default')}"
            )

        dataset_params = config.get_dataset_config().get("creation", {})
        tui.info(
            f"Dataset: steps={dataset_params.get('num_steps', 'default')}, batch_size={dataset_params.get('batch_size', 'default')}"
        )

        if config.huggingface:
            hf_config = config.get_huggingface_config()
            tui.info(f"Hugging Face: repo={hf_config.get('repository', 'not set')}")

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


if __name__ == "__main__":
    cli()
