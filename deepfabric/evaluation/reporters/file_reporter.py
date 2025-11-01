"""File-based reporter for writing results to local JSON files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from .base import BaseReporter

if TYPE_CHECKING:
    from ..evaluator import EvaluationResult
    from ..metrics import SampleEvaluation

console = Console()


class FileReporter(BaseReporter):
    """Writes evaluation results to local JSON file.

    This is the default reporter that maintains backwards compatibility
    with the original file-based output.
    """

    def __init__(self, config: dict | None = None):
        """Initialize file reporter.

        Args:
            config: Optional configuration with 'path' key
        """
        super().__init__(config)
        self.output_path = config.get("path") if config else None

    def report(self, result: EvaluationResult) -> None:
        """Write evaluation results to JSON file.

        Args:
            result: Complete evaluation result
        """
        # Use path from config, or fall back to result's config
        output_path = self.output_path or result.config.output_path

        if output_path is None:
            console.print("[yellow]No output path specified, skipping file write[/yellow]")
            return

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w") as f:
            f.write(result.model_dump_json(indent=2))

        console.print(f"[green]Results saved to {path}[/green]")

    def report_sample(self, sample_eval: SampleEvaluation) -> None:
        """File reporter doesn't support streaming (waits for final results).

        Args:
            sample_eval: Individual sample evaluation (ignored)
        """
