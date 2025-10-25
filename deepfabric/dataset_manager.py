import asyncio
import os
import traceback

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from rich.console import Group
from rich.live import Live
from rich.progress import Progress as RichProgress
from rich.progress import SpinnerColumn, TextColumn

from .config import DeepFabricConfig
from .config_manager import DEFAULT_MODEL
from .dataset import Dataset
from .exceptions import ConfigurationError
from .generator import DataSetGenerator
from .progress import ProgressReporter
from .tui import get_dataset_tui, get_tui
from .utils import ensure_not_running_loop

if TYPE_CHECKING:
    from .topic_model import TopicModel

# Constants for debug output
DEBUG_MAX_FAILURES_TO_SHOW = 10


async def handle_dataset_events_async(
    generator: AsyncIterator[dict | Dataset], engine=None, debug: bool = False
) -> Dataset | None:
    """Handle dataset generation with TUI progress and streaming feedback."""
    tui = get_dataset_tui()
    progress = None
    task = None
    live = None

    final_result: Dataset | None = None
    try:
        async for event in generator:
            if isinstance(event, dict) and "event" in event:
                if event["event"] == "generation_start":
                    tui.show_generation_header(
                        event["model_name"], event["num_steps"], event["batch_size"]
                    )
                    tui.progress = tui.create_rich_progress()
                    progress = tui.progress
                    task = progress.add_task(
                        "  Generating dataset samples", total=event["total_samples"]
                    )

                    # Create streaming progress with animated spinner
                    tui.stream_progress = RichProgress(
                        SpinnerColumn(),
                        TextColumn("{task.description}", style="dim"),
                        console=tui.console,
                    )
                    tui.stream_task = tui.stream_progress.add_task(
                        "  Processing: (waiting for LLM output...)"
                    )

                    # Start Live display with both progress bars
                    live = Live(
                        Group(progress, tui.stream_progress),
                        console=tui.console,
                        refresh_per_second=10,
                    )
                    tui.live_display = live  # Give TUI reference to update it
                    live.start()
                elif event["event"] == "step_complete":
                    if progress and task is not None:
                        samples_generated = event.get("samples_generated", 0)
                        if samples_generated > 0:
                            progress.update(task, advance=samples_generated)

                        # Show debug info for failed samples in this step
                        if debug and "failed_in_step" in event and event["failed_in_step"] > 0:
                            get_tui().error(
                                f"🔍 Debug: {event['failed_in_step']} samples failed in this step"
                            )
                            if "failure_reasons" in event:
                                for reason in event.get("failure_reasons", [])[
                                    :3
                                ]:  # Show first 3 failures
                                    get_tui().error(f"    - {reason}")

                elif event["event"] == "generation_complete":
                    if live:
                        live.stop()
                    tui.console.print()  # Add blank line after live display
                    tui.success(f"Successfully generated {event['total_samples']} samples")
                    if event["failed_samples"] > 0:
                        tui.warning(f"Failed to generate {event['failed_samples']} samples")

                        # Show detailed failure information in debug mode
                        if debug and engine and hasattr(engine, "failed_samples"):
                            get_tui().error("\n🔍 Debug: Dataset generation failures:")
                            for idx, failure in enumerate(
                                engine.failed_samples[:DEBUG_MAX_FAILURES_TO_SHOW], 1
                            ):
                                get_tui().error(f"  [{idx}] {failure}")
                            if len(engine.failed_samples) > DEBUG_MAX_FAILURES_TO_SHOW:
                                remaining = len(engine.failed_samples) - DEBUG_MAX_FAILURES_TO_SHOW
                                get_tui().error(f"  ... and {remaining} more failures")

            elif isinstance(event, Dataset):
                final_result = event
            else:
                # Handle unexpected non-dict, non-Dataset events
                get_tui().warning(f"Unexpected event type: {type(event)}")
    except Exception as e:
        if live:
            live.stop()
        if debug:
            get_tui().error(f"🔍 Debug: Full traceback:\n{traceback.format_exc()}")
        get_tui().error(f"Dataset generation failed: {str(e)}")
        raise

    return final_result


def handle_dataset_events(generator, engine=None, debug: bool = False) -> Dataset | None:
    """Synchronous wrapper for async dataset event handling."""
    ensure_not_running_loop("handle_dataset_events")
    return asyncio.run(handle_dataset_events_async(generator, engine=engine, debug=debug))


def create_dataset(
    engine: DataSetGenerator,
    topic_model: "TopicModel",
    config: DeepFabricConfig,
    num_steps: int | None = None,
    batch_size: int | None = None,
    sys_msg: bool | None = None,
    provider: str | None = None,  # noqa: ARG001
    model: str | None = None,
    engine_overrides: dict | None = None,
    debug: bool = False,
) -> Dataset:
    """
    Create dataset using the data engine and topic model.

    Args:
        engine: DataSetGenerator instance
        topic_model: TopicModel (Tree or Graph) to use for generation
        config: DeepFabricConfig object
        num_steps: Override for number of steps
        batch_size: Override for batch size
        sys_msg: Override for including system message
        provider: Override for LLM provider
        model: Override for model name
        engine_overrides: Additional engine parameter overrides

    Returns:
        Generated Dataset object

    Raises:
        ConfigurationError: If dataset generation fails
    """
    ensure_not_running_loop("create_dataset")
    return asyncio.run(
        create_dataset_async(
            engine=engine,
            topic_model=topic_model,
            config=config,
            num_steps=num_steps,
            batch_size=batch_size,
            sys_msg=sys_msg,
            provider=provider,
            model=model,
            engine_overrides=engine_overrides,
            debug=debug,
        )
    )


async def create_dataset_async(
    engine: DataSetGenerator,
    topic_model: "TopicModel",
    config: DeepFabricConfig,
    num_steps: int | None = None,
    batch_size: int | None = None,
    sys_msg: bool | None = None,
    provider: str | None = None,  # noqa: ARG001
    model: str | None = None,
    engine_overrides: dict | None = None,
    debug: bool = False,
) -> Dataset:
    dataset_config = config.get_dataset_config()
    dataset_params = dataset_config["creation"]

    final_num_steps = num_steps or dataset_params["num_steps"]
    final_batch_size = batch_size or dataset_params["batch_size"]

    engine_params = config.get_engine_params(**(engine_overrides or {}))
    final_model = model or engine_params.get("model_name", DEFAULT_MODEL)

    # Create progress reporter and attach TUI as observer for streaming feedback
    progress_reporter = ProgressReporter()
    tui = get_dataset_tui()
    progress_reporter.attach(tui)

    # Attach progress reporter to engine
    engine.progress_reporter = progress_reporter

    try:
        generator = engine.create_data_with_events_async(
            num_steps=final_num_steps,
            batch_size=final_batch_size,
            topic_model=topic_model,
            model_name=final_model,
            sys_msg=sys_msg,
            num_example_demonstrations=dataset_params.get("num_example_demonstrations") or 3,
        )
        dataset = await handle_dataset_events_async(generator, engine=engine, debug=debug)
    except Exception as e:  # noqa: BLE001
        raise ConfigurationError(f"Error creating dataset: {str(e)}") from e

    if dataset is None:
        raise ConfigurationError("Dataset generation failed - no dataset returned")

    return dataset


def _upload_to_service(
    service_name: str,
    dataset_path: str,
    config: dict,
    credential_check_func,
    uploader_import_func,
    uploader_args_func,
    push_args_func,
    tui,
) -> None:
    """Generic function to upload dataset to any configured service."""
    try:
        tui.info(f"Uploading dataset to {service_name}...")

        # Check credentials
        credentials = credential_check_func()
        if not credentials:
            return

        # Import uploader class
        uploader_class = uploader_import_func()

        # Create uploader instance
        uploader_args = uploader_args_func(credentials)
        uploader = (
            uploader_class(*uploader_args)
            if isinstance(uploader_args, tuple)
            else uploader_class(**uploader_args)
        )

        # Prepare push arguments
        push_args = push_args_func(config, dataset_path)

        # Upload dataset
        result = uploader.push_to_hub(**push_args)

        if result["status"] == "success":
            tui.success(result["message"])
        else:
            tui.warning(f"{service_name} upload failed: {result['message']}")

    except Exception as e:
        tui.warning(f"Error uploading to {service_name}: {str(e)}")


def _upload_to_huggingface(dataset_path: str, hf_config: dict, tui) -> None:
    """Upload dataset to HuggingFace Hub if configured."""

    def check_credentials():
        token = os.getenv("HF_TOKEN")
        if not token:
            tui.warning("HF_TOKEN not set. Skipping HuggingFace upload.")
            return None
        return token

    def import_uploader():
        from .hf_hub import HFUploader  # noqa: PLC0415

        return HFUploader

    def get_uploader_args(credentials):
        return (credentials,)  # HFUploader takes token as single argument

    def get_push_args(config, dataset_path):
        return {
            "hf_dataset_repo": config["repository"],
            "jsonl_file_path": dataset_path,
            "tags": config.get("tags", []),
        }

    _upload_to_service(
        "HuggingFace Hub",
        dataset_path,
        hf_config,
        check_credentials,
        import_uploader,
        get_uploader_args,
        get_push_args,
        tui,
    )


def _upload_to_kaggle(dataset_path: str, kaggle_config: dict, tui) -> None:
    """Upload dataset to Kaggle if configured."""

    def check_credentials():
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if not username or not key:
            tui.warning("KAGGLE_USERNAME or KAGGLE_KEY not set. Skipping Kaggle upload.")
            return None
        return (username, key)

    def import_uploader():
        from .kaggle_hub import KaggleUploader  # noqa: PLC0415

        return KaggleUploader

    def get_uploader_args(credentials):
        return credentials  # KaggleUploader takes username, key as tuple

    def get_push_args(config, dataset_path):
        return {
            "dataset_handle": config["handle"],
            "jsonl_file_path": dataset_path,
            "tags": config.get("tags", []),
            "version_notes": config.get("version_notes"),
            "description": config.get("description"),
        }

    _upload_to_service(
        "Kaggle",
        dataset_path,
        kaggle_config,
        check_credentials,
        import_uploader,
        get_uploader_args,
        get_push_args,
        tui,
    )


def save_dataset(dataset: Dataset, save_path: str, config: DeepFabricConfig | None = None) -> None:
    """
    Save dataset to file and apply formatters if configured.

    Args:
        dataset: Dataset object to save
        save_path: Path where to save the dataset
        config: Optional configuration containing formatter settings

    Raises:
        ConfigurationError: If saving fails
    """
    tui = get_tui()
    try:
        # Save the raw dataset
        dataset.save(save_path)
        tui.success(f"Dataset saved to: {save_path}")

        # Apply formatters if configured
        if config:
            formatter_configs = config.get_formatter_configs()
            if formatter_configs:
                tui.info("Applying formatters...")
                try:
                    formatted_datasets = dataset.apply_formatters(formatter_configs)

                    for formatter_name, formatted_dataset in formatted_datasets.items():
                        if hasattr(formatted_dataset, "samples"):
                            sample_count = len(formatted_dataset.samples)
                            tui.success(
                                f"Applied '{formatter_name}' formatter: {sample_count} samples"
                            )
                        else:
                            tui.success(f"Applied '{formatter_name}' formatter")

                except Exception as e:
                    tui.error(f"Error applying formatters: {str(e)}")
                    # Don't raise here - we want to continue even if formatters fail

        # Handle automatic uploads if configured
        if config:
            # HuggingFace upload
            if config.huggingface:
                _upload_to_huggingface(save_path, config.get_huggingface_config(), tui)

            # Kaggle upload
            if config.kaggle:
                _upload_to_kaggle(save_path, config.get_kaggle_config(), tui)

    except Exception as e:
        raise ConfigurationError(f"Error saving dataset: {str(e)}") from e
