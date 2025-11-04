"""Base classes for evaluation result reporting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..evaluator import EvaluationResult
    from ..metrics import SampleEvaluation


class BaseReporter(ABC):
    """Base class for evaluation result reporters.

    Reporters handle the output of evaluation results. They can write to
    local files, send to cloud services, or perform other actions with
    the evaluation data.
    """

    def __init__(self, config: dict | None = None):
        """Initialize reporter with optional configuration.

        Args:
            config: Optional reporter-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    def report(self, result: EvaluationResult) -> None:
        """Report complete evaluation results.

        Called once at the end of evaluation with all results.

        Args:
            result: Complete evaluation result
        """

    def report_sample(self, sample_eval: SampleEvaluation) -> None:  # noqa: B027
        """Report individual sample evaluation (optional, for streaming).

        Called after each sample is evaluated, allowing real-time reporting.
        Default implementation does nothing.

        Args:
            sample_eval: Individual sample evaluation result
        """
        pass  # Optional hook for subclasses

    def get_name(self) -> str:
        """Return reporter identifier.

        Returns:
            Reporter name (e.g., "file", "cloud")
        """
        return self.__class__.__name__.replace("Reporter", "").lower()
