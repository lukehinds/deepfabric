"""Base classes for evaluation system."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from ...schemas import ToolDefinition
from ..inference import ModelResponse
from ..parser import GroundTruth


class EvaluationContext(BaseModel):
    """Context passed to evaluators."""

    messages: list[dict[str, str]] = Field(description="Messages sent to model")
    tools: list[ToolDefinition] | None = Field(
        default=None,
        description="Available tools for the evaluation",
    )
    sample_id: int = Field(description="Sample index in dataset")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context metadata",
    )


class EvaluatorResult(BaseModel):
    """Result from a single evaluator."""

    evaluator_name: str = Field(description="Name of the evaluator")
    metrics: dict[str, float] = Field(
        description="Metrics produced by this evaluator",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details about the evaluation",
    )
    error: str | None = Field(
        default=None,
        description="Error message if evaluation failed",
    )


class BaseEvaluator(ABC):
    """Base class for all evaluators.

    Evaluators assess specific aspects of model outputs (e.g., tool calling,
    safety, answer quality). They are modular and can be enabled/disabled
    via configuration.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize evaluator with optional configuration.

        Args:
            config: Optional evaluator-specific configuration
        """
        self.config = config or {}

    @abstractmethod
    def get_name(self) -> str:
        """Return unique identifier for this evaluator.

        Returns:
            Evaluator name (e.g., "tool_calling", "safety")
        """

    def get_metrics(self) -> list[str]:
        """Return list of metric names this evaluator produces.

        Returns:
            List of metric names
        """
        return []

    def applicable_to(self, ground_truth: GroundTruth) -> bool:  # noqa: ARG002
        """Check if this evaluator should run for the given sample.

        Args:
            ground_truth: Ground truth for the sample

        Returns:
            True if evaluator should run, False to skip
        """
        return True

    @abstractmethod
    def evaluate(
        self,
        ground_truth: GroundTruth,
        prediction: ModelResponse,
        context: EvaluationContext,
    ) -> EvaluatorResult | None:
        """Evaluate a single sample.

        Args:
            ground_truth: Expected values from dataset
            prediction: Model's generated response
            context: Additional evaluation context

        Returns:
            EvaluatorResult with metrics and details, or None to skip
        """
