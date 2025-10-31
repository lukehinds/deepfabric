import yaml

from pydantic import ValidationError

from .config import DeepFabricConfig
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    ENGINE_DEFAULT_BATCH_SIZE,
    ENGINE_DEFAULT_NUM_EXAMPLES,
    ENGINE_DEFAULT_TEMPERATURE,
    TOPIC_GRAPH_DEFAULT_DEGREE,
    TOPIC_GRAPH_DEFAULT_DEPTH,
    TOPIC_GRAPH_DEFAULT_TEMPERATURE,
    TOPIC_TREE_DEFAULT_DEGREE,
    TOPIC_TREE_DEFAULT_DEPTH,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .exceptions import ConfigurationError
from .tui import get_tui


def load_config(  # noqa: PLR0913
    config_file: str | None,
    topic_prompt: str | None = None,
    dataset_system_prompt: str | None = None,
    generation_system_prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    degree: int | None = None,
    depth: int | None = None,
    num_steps: int | None = None,
    batch_size: int | None = None,
    save_tree: str | None = None,
    save_graph: str | None = None,
    dataset_save_as: str | None = None,
    sys_msg: bool | None = None,
    mode: str = "tree",
    # Modular conversation configuration
    conversation_type: str | None = None,
    reasoning_style: str | None = None,
    agent_mode: str | None = None,
) -> DeepFabricConfig:
    """
    Load configuration from YAML file or create minimal config from CLI arguments.

    Args:
        config_file: Path to YAML configuration file
        topic_prompt: Starting topic/seed for tree/graph generation
        dataset_system_prompt: System prompt for final dataset
        generation_system_prompt: System prompt for dataset content generation
        provider: LLM provider
        model: Model name
        temperature: Temperature setting
        degree: Branching factor
        depth: Depth of tree/graph
        num_steps: Number of generation steps
        batch_size: Batch size for generation
        save_tree: Path to save tree
        save_graph: Path to save graph
        dataset_save_as: Path to save dataset
        sys_msg: Include system message in dataset
        mode: Topic generation mode (tree or graph)
        conversation_type: Base conversation type (basic, chain_of_thought)
        reasoning_style: Reasoning style for chain_of_thought (freetext, structured, hybrid)
        agent_mode: Agent mode (single_turn, multi_turn)

    Returns:
        DeepFabricConfig object

    Raises:
        ConfigurationError: If config file is invalid or required parameters are missing
    """
    if config_file:
        try:
            return DeepFabricConfig.from_yaml(config_file)
        except FileNotFoundError as e:
            raise ConfigurationError(f"Config file not found: {config_file}") from e
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in config file: {str(e)}") from e
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {str(e)}") from e

    # No config file provided - create minimal configuration from CLI args
    if not topic_prompt:
        raise ConfigurationError("--topic-prompt is required when no config file is provided")

    tui = get_tui()
    tui.info("No config file provided - using CLI parameters")

    # Create minimal config dict
    default_prompt = generation_system_prompt or "You are a helpful AI assistant."
    minimal_config = {
        "dataset_system_prompt": dataset_system_prompt,
        "data_engine": {
            "instructions": "Generate diverse and educational examples",
            "generation_system_prompt": default_prompt,
            "provider": provider or DEFAULT_PROVIDER,
            "model": model or DEFAULT_MODEL,
            "temperature": temperature or ENGINE_DEFAULT_TEMPERATURE,
            "max_retries": DEFAULT_MAX_RETRIES,
            # Add modular conversation config if provided
            **({"conversation_type": conversation_type} if conversation_type else {}),
            **({"reasoning_style": reasoning_style} if reasoning_style else {}),
            **({"agent_mode": agent_mode} if agent_mode else {}),
        },
        "dataset": {
            "creation": {
                "num_steps": num_steps or ENGINE_DEFAULT_NUM_EXAMPLES,
                "batch_size": batch_size or ENGINE_DEFAULT_BATCH_SIZE,
                "provider": provider or DEFAULT_PROVIDER,
                "model": model or DEFAULT_MODEL,
                "sys_msg": sys_msg if sys_msg is not None else True,
            },
            "save_as": dataset_save_as or "dataset.jsonl",
        },
    }

    # Add topic generation config based on mode
    if mode == "graph":
        minimal_config["topic_graph"] = {
            "topic_prompt": topic_prompt,
            "provider": provider or DEFAULT_PROVIDER,
            "model": model or DEFAULT_MODEL,
            "temperature": temperature or TOPIC_GRAPH_DEFAULT_TEMPERATURE,
            "degree": degree or TOPIC_GRAPH_DEFAULT_DEGREE,
            "depth": depth or TOPIC_GRAPH_DEFAULT_DEPTH,
            "save_as": save_graph or "topic_graph.json",
        }
    else:  # mode == "tree" (default)
        minimal_config["topic_tree"] = {
            "topic_prompt": topic_prompt,
            "provider": provider or DEFAULT_PROVIDER,
            "model": model or DEFAULT_MODEL,
            "temperature": temperature or TOPIC_TREE_DEFAULT_TEMPERATURE,
            "degree": degree or TOPIC_TREE_DEFAULT_DEGREE,
            "depth": depth or TOPIC_TREE_DEFAULT_DEPTH,
            "save_as": save_tree or "topic_tree.jsonl",
        }

    try:
        return DeepFabricConfig.model_validate(minimal_config)
    except ValidationError as e:
        raise ConfigurationError(f"Invalid configuration: {str(e)}") from e


def apply_cli_overrides(
    config: DeepFabricConfig,
    dataset_system_prompt: str | None = None,
    topic_prompt: str | None = None,
    topic_system_prompt: str | None = None,
    generation_system_prompt: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    degree: int | None = None,
    depth: int | None = None,
    base_url: str | None = None,
) -> tuple[dict, dict, dict]:
    """
    Apply CLI overrides to configuration and return override dictionaries.

    Args:
        config: DeepFabricConfig object to update
        dataset_system_prompt: Override for dataset system prompt
        topic_prompt: Override for topic prompt
        topic_system_prompt: Override for topic system prompt
        generation_system_prompt: Override for generation system prompt
        provider: Override for LLM provider
        model: Override for model name
        temperature: Override for temperature
        degree: Override for branching factor
        depth: Override for depth
        base_url: Override for base URL

    Returns:
        Tuple of (tree_overrides, graph_overrides, engine_overrides) dictionaries
    """
    # Apply dataset system prompt override if provided
    if dataset_system_prompt:
        config.dataset_system_prompt = dataset_system_prompt

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

    # Graph overrides are the same as tree overrides
    graph_overrides = tree_overrides.copy()

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

    return tree_overrides, graph_overrides, engine_overrides


def get_final_parameters(
    config: DeepFabricConfig,
    num_steps: int | None = None,
    batch_size: int | None = None,
    depth: int | None = None,
    degree: int | None = None,
) -> tuple[int, int, int, int]:
    """
    Get final parameters from config and CLI overrides.

    Args:
        config: DeepFabricConfig object
        num_steps: CLI override for num_steps
        batch_size: CLI override for batch_size
        depth: CLI override for depth
        degree: CLI override for degree

    Returns:
        Tuple of (num_steps, batch_size, depth, degree)
    """
    dataset_config = config.get_dataset_config()
    dataset_params = dataset_config["creation"]

    final_num_steps = num_steps or dataset_params["num_steps"]
    final_batch_size = batch_size or dataset_params["batch_size"]

    # Get depth and degree from config if not provided
    config_depth = None
    config_degree = None
    if config.topic_tree:
        config_depth = config.topic_tree.depth
        config_degree = config.topic_tree.degree
    elif config.topic_graph:
        config_depth = config.topic_graph.depth
        config_degree = config.topic_graph.degree

    final_depth = depth or config_depth or TOPIC_TREE_DEFAULT_DEPTH
    final_degree = degree or config_degree or TOPIC_TREE_DEFAULT_DEGREE

    return final_num_steps, final_batch_size, final_depth, final_degree
