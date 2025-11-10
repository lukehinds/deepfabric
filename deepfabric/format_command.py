import click
import yaml

from .dataset import Dataset
from .tui import get_tui


def _detect_dataset_characteristics(dataset: Dataset) -> dict[str, str | None]:
    """
    Detect dataset characteristics by inspecting sample fields.

    Args:
        dataset: Dataset to inspect

    Returns:
        Dict with conversation_type, agent_mode, and reasoning_style
    """
    # Sample a few items to detect characteristics
    max_samples = 10
    min_multi_turn_messages = 3
    samples = (
        dataset.samples[:max_samples] if len(dataset.samples) > max_samples else dataset.samples
    )

    if not samples:
        return {
            "conversation_type": None,
            "agent_mode": None,
            "reasoning_style": None,
        }

    # Detect characteristics from samples
    has_reasoning = False
    has_tool_context = False
    has_agent_context = False
    reasoning_style = None
    is_multi_turn = False

    for sample in samples:
        # Check for reasoning (chain_of_thought indicator)
        if "reasoning" in sample and sample["reasoning"]:
            has_reasoning = True
            # Try to detect reasoning style from structure
            reasoning_data = sample["reasoning"]
            if isinstance(reasoning_data, dict):
                style = reasoning_data.get("style")
                if style:
                    reasoning_style = style

        # Check for tool context (agent indicator)
        if "tool_context" in sample and sample["tool_context"]:
            has_tool_context = True

        # Check for agent context (multi-turn indicator)
        if "agent_context" in sample and sample["agent_context"]:
            has_agent_context = True
            agent_data = sample["agent_context"]
            if isinstance(agent_data, dict):
                turns = agent_data.get("turns", 0)
                if turns > 1:
                    is_multi_turn = True

        # Check messages length as fallback for multi-turn detection
        # More than 3 messages (system, user, assistant) suggests multi-turn
        if (
            "messages" in sample
            and isinstance(sample["messages"], list)
            and len(sample["messages"]) > min_multi_turn_messages
        ):
            is_multi_turn = True

    # Determine conversation_type
    conversation_type = "chain_of_thought" if has_reasoning else "basic"

    # Determine agent_mode
    if has_tool_context or has_agent_context:
        agent_mode = "multi_turn" if is_multi_turn else "single_turn"
    else:
        agent_mode = None

    return {
        "conversation_type": conversation_type,
        "agent_mode": agent_mode,
        "reasoning_style": reasoning_style,
    }


def _warn_formatter_incompatibilities(dataset: Dataset, formatter_configs: list[dict], tui) -> bool:
    """
    Warn about potential incompatibilities between dataset and formatters.

    Args:
        dataset: Dataset to check
        formatter_configs: List of formatter configurations
        tui: TUI instance for displaying warnings

    Returns:
        True if fatal incompatibility detected (should abort), False otherwise
    """
    # Detect dataset characteristics
    characteristics = _detect_dataset_characteristics(dataset)
    conversation_type = characteristics["conversation_type"]
    agent_mode = characteristics["agent_mode"]
    reasoning_style = characteristics["reasoning_style"]

    # Define formatter compatibility rules (same as config.py)
    formatter_warnings = {
        "alpaca": {
            "incompatible_with": {
                "conversation_type": ["chain_of_thought"],
                "agent_mode": ["multi_turn"],
            },
            "warning": "Alpaca format works best with simple Q&A. Chain-of-thought reasoning and multi-turn conversations may have high rejection rates.",
        },
        "chatml": {
            "incompatible_with": {
                "reasoning_style": ["structured", "hybrid"],
            },
            "warning": "ChatML format may reject samples with complex reasoning structures. Consider using 'conversations' format for chain-of-thought data.",
        },
        "xlam_v2": {
            "requires": {
                "agent_mode": ["single_turn", "multi_turn"],
            },
            "warning": "XLAM v2 format requires agent_mode to be set (single_turn or multi_turn) for tool-calling data.",
            "fatal": True,  # Fatal incompatibility - will fail all samples
        },
        "tool_calling": {
            "requires": {
                "agent_mode": ["single_turn", "multi_turn"],
            },
            "warning": "Tool calling format requires agent_mode to be set for tool-calling data.",
            "fatal": True,
        },
        "single_tool_call": {
            "requires": {
                "agent_mode": ["single_turn"],
            },
            "warning": "Single tool call format requires agent_mode='single_turn' and will reject multi-turn conversations.",
            "fatal": True,
        },
    }

    has_fatal_error = False

    # Check each formatter
    for formatter_config in formatter_configs:
        template = formatter_config.get("template", "")
        formatter_name = None

        # Extract formatter name from template (builtin://formatter_name)
        if isinstance(template, str) and "builtin://" in template:
            formatter_name = template.replace("builtin://", "").replace(".py", "")

        if not formatter_name or formatter_name not in formatter_warnings:
            continue

        rules = formatter_warnings[formatter_name]
        is_fatal = rules.get("fatal", False)

        # Check incompatibilities
        if "incompatible_with" in rules:
            incompatible = rules["incompatible_with"]

            if (
                "conversation_type" in incompatible
                and conversation_type in incompatible["conversation_type"]
            ):
                tui.warning(
                    f"Formatter '{formatter_config['name']}' may have high rejection rates with conversation_type='{conversation_type}'"
                )
                tui.info(f"  {rules['warning']}")

            if "agent_mode" in incompatible and agent_mode in incompatible["agent_mode"]:
                tui.warning(
                    f"Formatter '{formatter_config['name']}' may have high rejection rates with agent_mode='{agent_mode}'"
                )
                tui.info(f"  {rules['warning']}")

            if (
                "reasoning_style" in incompatible
                and reasoning_style in incompatible["reasoning_style"]
            ):
                tui.warning(
                    f"Formatter '{formatter_config['name']}' may have high rejection rates with reasoning_style='{reasoning_style}'"
                )
                tui.info(f"  {rules['warning']}")

        # Check requirements
        if "requires" in rules:
            requirements = rules["requires"]

            if "agent_mode" in requirements and (
                not agent_mode or agent_mode not in requirements["agent_mode"]
            ):
                tui.warning(
                    f"Formatter '{formatter_config['name']}' requires agent_mode to be one of {requirements['agent_mode']}"
                )
                tui.info(f"  Detected: agent_mode={agent_mode}")
                tui.info(f"  {rules['warning']}")
                if is_fatal:
                    has_fatal_error = True

    return has_fatal_error


def format_command(
    input_file: str | None = None,
    *,
    repo: str | None = None,
    split: str | None = None,
    config_file: str | None = None,
    formatter: str | None = None,
    target_model: str | None = None,
    model_config: str | None = None,
    output: str | None = None,
) -> None:
    """
    Apply formatters to an existing dataset.

    Args:
        input_file: Path to the input JSONL dataset file
        repo: Optional Hugging Face dataset repo id (e.g., "org/dataset-name")
        split: Optional split to load from the Hugging Face dataset (default: train)
        config_file: Optional YAML config file with formatter settings
        formatter: Optional formatter name (e.g., 'chatml')
        target_model: Optional HuggingFace model ID to format for (e.g., 'meta-llama/Llama-3.1-8B')
        model_config: Optional path to custom model mappings YAML for --target-model
        output: Optional output file path
    """
    tui = get_tui()

    if (input_file is None and repo is None) or (input_file and repo):
        raise ValueError("Specify exactly one of INPUT_FILE or --repo")

    # Load the existing dataset from local file or Hugging Face repo
    if input_file:
        tui.info(f"Loading dataset from {input_file}...")
        dataset = Dataset.from_jsonl(input_file)
        tui.success(f"Loaded {len(dataset)} samples")
    elif repo:
        # Use Dataset.from_hub instead of loading manually
        hf_split = split or "train"
        tui.info(f"Loading dataset from Hugging Face repo '{repo}' (split: {hf_split})...")
        try:
            dataset = Dataset.from_hub(repo, split=hf_split)
            tui.success(f"Loaded {len(dataset)} samples from {repo}:{hf_split}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset from HuggingFace: {e}") from e
    else:
        raise ValueError("Must specify either INPUT_FILE or --repo")

    # Handle --target-model (HF chat template formatter)
    if target_model:
        tui.info(f"Formatting for target model: {target_model}")

        # Determine output file
        if output:
            output_file = output
        elif input_file:
            output_file = f"{input_file.rsplit('.', 1)[0]}_{target_model.split('/')[-1]}.jsonl"
        else:
            output_file = f"{target_model.split('/')[-1]}-formatted.jsonl"

        # Format using HF chat template
        formatted_dataset = dataset.format(
            target_model=target_model, model_config=model_config, use_transformers=True
        )

        # Save formatted dataset
        formatted_dataset.save(output_file)
        tui.success(f"✓ Formatted dataset saved to {output_file}")
        tui.info(f"  Samples: {len(formatted_dataset)}")
        return

    # Determine formatter configuration (legacy builtin formatters)
    formatter_configs = []

    if config_file:
        # Load formatters from config file
        with open(config_file) as f:
            config_data = yaml.safe_load(f)

        # Check for formatters in dataset section
        if "dataset" in config_data and "formatters" in config_data["dataset"]:
            formatter_configs = config_data["dataset"]["formatters"]
        else:
            raise ValueError("No formatters found in config file")
    elif formatter:
        # Use specified formatter with default settings
        if input_file:
            output_file = output or f"{input_file.rsplit('.', 1)[0]}_{formatter}.jsonl"
        else:
            # When loading from --repo, default to a simple formatted.jsonl unless specified
            output_file = output or "formatted.jsonl"

        # Default configs for common formatters
        default_configs = {
            "conversations": {
                "include_system": False,
                "system_message": None,
                "roles_map": {"user": "user", "assistant": "assistant", "system": "system"},
            },
            "alpaca": {
                "instruction_template": "### Instruction:\n{instruction}\n\n### Response:",
                "include_empty_input": False,
            },
            "chatml": {
                "output_format": "text",
                "start_token": "<|im_start|>",
                "end_token": "<|im_end|>",
                "include_system": False,
            },
            "harmony": {
                "output_format": "text",
                "default_channel": "final",
                "include_developer_role": False,
                "reasoning_level": "high",
                "include_metadata": True,
            },
            # OpenAI Schema formatter defaults
            "openai": {},
            "xlam_v2": {},
        }

        # Map aliases to actual builtin module names
        template_name = formatter

        formatter_configs = [
            {
                "name": formatter,
                "template": f"builtin://{template_name}.py",
                "output": output_file,
                "config": default_configs.get(formatter, {}),
            }
        ]
    else:
        raise ValueError("Either --config-file or --formatter must be specified")

    # Detect dataset characteristics and warn about potential incompatibilities
    has_fatal_error = _warn_formatter_incompatibilities(dataset, formatter_configs, tui)

    # Abort if fatal incompatibility detected
    if has_fatal_error:
        return

    # Apply formatters
    tui.info("Applying formatters...")
    formatted_datasets = dataset.apply_formatters(formatter_configs)

    # Report results
    for formatter_config in formatter_configs:
        name = formatter_config["name"]
        output_path = formatter_config.get("output", f"{name}.jsonl")
        if name in formatted_datasets:
            formatted_dataset = formatted_datasets[name]
            num_samples = len(formatted_dataset)

            if num_samples == 0:
                tui.error(f"✗ Formatter '{name}' failed - no samples were successfully formatted")
                tui.info(
                    "  This is likely due to incompatibility between the dataset and formatter."
                )
                tui.info("  See warnings above for details.")
            else:
                tui.success(f"✓ Formatter '{name}' applied successfully")
                tui.info(f"  Output: {output_path}")
                tui.info(f"  Samples: {num_samples}")


@click.command(name="format")
@click.argument("input_file", type=click.Path(exists=True), required=False)
@click.option(
    "--repo",
    help="Hugging Face dataset repo id (e.g., 'org/dataset-name')",
)
@click.option(
    "--split",
    help="Split to load from Hugging Face dataset (default: train)",
)
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="YAML config file with formatter settings",
)
@click.option(
    "--formatter",
    "-f",
    type=click.Choice(
        [
            "conversations",
            "alpaca",
            "chatml",
            "harmony",
            "openai",
            "xlam_v2",
        ]
    ),
    help="Formatter to apply",
)
@click.option(
    "--target-model",
    "-m",
    help="HuggingFace model ID to format for (e.g., 'meta-llama/Llama-3.1-8B-Instruct')",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True),
    help="Path to custom model mappings YAML for --target-model",
)
@click.option(
    "--output",
    "-o",
    help="Output file path (default: input_file_formatter.jsonl)",
)
@click.pass_context
def format_cli(
    ctx,
    input_file: str | None,
    repo: str | None,
    split: str | None,
    config_file: str | None,
    formatter: str | None,
    target_model: str | None,
    model_config: str | None,
    output: str | None,
) -> None:
    """Apply formatters to an existing dataset."""
    try:
        format_command(
            input_file,
            repo=repo,
            split=split,
            config_file=config_file,
            formatter=formatter,
            target_model=target_model,
            model_config=model_config,
            output=output,
        )
    except FileNotFoundError as e:
        ctx.fail(f"Input file not found: {e}")
    except Exception as e:
        ctx.fail(f"Error: {e}")
