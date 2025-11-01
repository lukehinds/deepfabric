"""Multi-reporter for running multiple reporters simultaneously."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from rich.console import Console

from .base import BaseReporter

if TYPE_CHECKING:
    from ..evaluator import EvaluationResult
    from ..metrics import SampleEvaluation

console = Console()


class MultiReporter(BaseReporter):
    """Runs multiple reporters (e.g., file + cloud).

    This reporter allows sending results to multiple destinations
    simultaneously. Errors in one reporter don't affect others.
    """

    def __init__(self, reporters: list[BaseReporter]):
        """Initialize multi-reporter.

        Args:
            reporters: List of reporter instances to run
        """
        super().__init__()
        self.reporters = reporters

    def report(self, result: EvaluationResult) -> None:
        """Report to all reporters.

        Args:
            result: Complete evaluation result
        """
        for reporter in self.reporters:
            try:
                reporter.report(result)
            except Exception as e:  # noqa: BLE001
                console.print(f"[red]Reporter {reporter.get_name()} failed: {e}[/red]")

    def report_sample(self, sample_eval: SampleEvaluation) -> None:
        """Report sample to all reporters.

        Args:
            sample_eval: Individual sample evaluation result
        """
        for reporter in self.reporters:
            # Silently fail on sample streaming errors
            with suppress(Exception):
                reporter.report_sample(sample_eval)
