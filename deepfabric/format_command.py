"""Format command implementation for DeepFabric CLI."""

import click
import yaml

from .dataset import Dataset
from .tui import get_tui


def format_command(
    input_file: str,
    config_file: str | None = None,
    formatter: str | None = None,
    output: str | None = None,
) -> None:
    """
    Apply formatters to an existing dataset.

    Args:
        input_file: Path to the input JSONL dataset file
        config_file: Optional YAML config file with formatter settings
        formatter: Optional formatter name (e.g., 'im_format')
        output: Optional output file path
    """
    tui = get_tui()

    # Load the existing dataset
    tui.info(f"Loading dataset from {input_file}...")
    dataset = Dataset.from_jsonl(input_file)
    tui.success(f"Loaded {len(dataset)} samples")

    # Determine formatter configuration
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
        output_file = output or f"{input_file.rsplit('.', 1)[0]}_{formatter}.jsonl"

        # Default configs for common formatters
        default_configs = {
            "im_format": {
                "include_system": True,
                "system_message": "You are a helpful assistant.",
                "roles_map": {"user": "user", "assistant": "assistant", "system": "system"},
            },
            "unsloth": {
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
            "grpo": {
                "reasoning_start_tag": "<start_working_out>",
                "reasoning_end_tag": "<end_working_out>",
                "solution_start_tag": "<SOLUTION>",
                "solution_end_tag": "</SOLUTION>",
            },
            "harmony": {
                "output_format": "text",
                "default_channel": "final",
                "include_developer_role": False,
                "reasoning_level": "high",
                "include_metadata": True,
            },
        }

        formatter_configs = [
            {
                "name": formatter,
                "template": f"builtin://{formatter}.py",
                "output": output_file,
                "config": default_configs.get(formatter, {}),
            }
        ]
    else:
        raise ValueError("Either --config-file or --formatter must be specified")

    # Apply formatters
    tui.info("Applying formatters...")
    formatted_datasets = dataset.apply_formatters(formatter_configs)

    # Report results
    for formatter_config in formatter_configs:
        name = formatter_config["name"]
        output_path = formatter_config.get("output", f"{name}.jsonl")
        if name in formatted_datasets:
            formatted_dataset = formatted_datasets[name]
            tui.success(f"✓ Formatter '{name}' applied successfully")
            tui.info(f"  Output: {output_path}")
            tui.info(f"  Samples: {len(formatted_dataset)}")


@click.command(name="format")
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True),
    help="YAML config file with formatter settings",
)
@click.option(
    "--formatter",
    "-f",
    type=click.Choice(["im_format", "unsloth", "alpaca", "chatml", "grpo", "harmony"]),
    help="Formatter to apply",
)
@click.option(
    "--output",
    "-o",
    help="Output file path (default: input_file_formatter.jsonl)",
)
@click.pass_context
def format_cli(
    ctx, input_file: str, config_file: str | None, formatter: str | None, output: str | None
) -> None:
    """Apply formatters to an existing dataset."""
    try:
        format_command(input_file, config_file, formatter, output)
    except FileNotFoundError as e:
        ctx.fail(f"Input file not found: {e}")
    except Exception as e:
        ctx.fail(f"Error: {e}")
