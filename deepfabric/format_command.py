import click
import yaml

from .dataset import Dataset
from .tui import get_tui


def format_command(
    input_file: str | None = None,
    *,
    repo: str | None = None,
    split: str | None = None,
    config_file: str | None = None,
    formatter: str | None = None,
    output: str | None = None,
) -> None:
    """
    Apply formatters to an existing dataset.

    Args:
        input_file: Path to the input JSONL dataset file
        repo: Optional Hugging Face dataset repo id (e.g., "org/dataset-name")
        split: Optional split to load from the Hugging Face dataset (default: train)
        config_file: Optional YAML config file with formatter settings
        formatter: Optional formatter name (e.g., 'im_format')
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
    else:
        # Lazy import to avoid overhead when not needed
        try:
            from datasets import load_dataset  # type: ignore  # noqa: PLC0415
            from datasets.exceptions import (  # type: ignore  # noqa: PLC0415
                DatasetNotFoundError,
                UnexpectedSplitsError,
            )
        except ImportError as e:  # pragma: no cover - import path
            raise RuntimeError(
                "The 'datasets' library is required to load from --repo. Please install it."
            ) from e

        hf_split = split or "train"
        tui.info(f"Loading dataset from Hugging Face repo '{repo}' (split: {hf_split})...")
        try:
            # Bandit nosec, as no digest is set.
            hf_ds = load_dataset(str(repo), split=hf_split) #  nosec
        except (DatasetNotFoundError, UnexpectedSplitsError) as e:
            msg = (
                "Failed to load dataset from Hugging Face repo "
                f"'{repo}' with split '{hf_split}': {e}"
            )
            raise RuntimeError(msg) from e

        # Convert to DeepFabric Dataset from list of dicts
        samples = list(hf_ds)
        dataset = Dataset.from_list(samples)
        tui.success(f"Loaded {len(dataset)} samples from {repo}:{hf_split}")

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
        if input_file:
            output_file = output or f"{input_file.rsplit('.', 1)[0]}_{formatter}.jsonl"
        else:
            # When loading from --repo, default to a simple formatted.jsonl unless specified
            output_file = output or "formatted.jsonl"

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
            # TRL SFT Tools formatter defaults
            "trl_sft_tools": {},
            "trl": {},  # alias
            "xlam_v2": {},
        }

        # Map aliases to actual builtin module names
        template_name = formatter
        if formatter == "trl":
            template_name = "trl_sft_tools"

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
            "im_format",
            "unsloth",
            "alpaca",
            "chatml",
            "grpo",
            "harmony",
            "trl",
            "trl_sft_tools",
            "xlam_v2",
        ]
    ),
    help="Formatter to apply",
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
            output=output,
        )
    except FileNotFoundError as e:
        ctx.fail(f"Input file not found: {e}")
    except Exception as e:
        ctx.fail(f"Error: {e}")
