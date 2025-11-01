"""Cloud-based reporter for sending results to DeepFabric SaaS (future)."""

from __future__ import annotations

import os
import uuid

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.console import Console

from .base import BaseReporter

if TYPE_CHECKING:
    from ..evaluator import EvaluationResult
    from ..metrics import SampleEvaluation

console = Console()


class CloudReporter(BaseReporter):
    """Posts evaluation results to DeepFabric cloud service.

    This is a stub implementation for future cloud service integration.
    When the DeepFabric SaaS is ready, this reporter will upload results
    to the cloud for centralized tracking, dashboards, and analytics.
    """

    def __init__(self, config: dict | None = None):
        """Initialize cloud reporter.

        Args:
            config: Optional configuration with:
                - api_key: DeepFabric API key (or use DEEPFABRIC_API_KEY env var)
                - endpoint: API endpoint URL (optional, uses default)
                - enabled: Whether to enable cloud reporting (default: False)
        """
        super().__init__(config)

        # Get API key from config or environment
        self.api_key = None
        if config:
            self.api_key = config.get("api_key") or os.getenv("DEEPFABRIC_API_KEY")
        else:
            self.api_key = os.getenv("DEEPFABRIC_API_KEY")

        # Get endpoint (use default if not specified)
        self.endpoint = (
            config.get("endpoint") if config else "https://api.deepfabric.ai/v1/evaluations"
        )

        # Cloud reporting is disabled by default until service is ready
        self.enabled = config.get("enabled", False) if config else False

        # Generate unique run ID for this evaluation
        self.run_id = str(uuid.uuid4())

    def report(self, result: EvaluationResult) -> None:  # noqa: ARG002
        """Upload complete evaluation results to cloud service.

        Args:
            result: Complete evaluation result (unused until service ready)
        """
        if not self.enabled:
            console.print(
                "[yellow]Cloud reporting not yet available. "
                "Set enabled=True when service is ready.[/yellow]"
            )
            return

        if not self.api_key:
            console.print(
                "[red]Cloud reporting enabled but no API key provided. "
                "Set DEEPFABRIC_API_KEY environment variable or pass api_key in config.[/red]"
            )
            return

        # TODO: When cloud service is ready, uncomment this:
        # payload = self._prepare_payload(result)
        # try:
        #     response = requests.post(
        #         f"{self.endpoint}/runs",
        #         json=payload,
        #         headers={
        #             "Authorization": f"Bearer {self.api_key}",
        #             "Content-Type": "application/json",
        #         },
        #         timeout=30,
        #     )
        #
        #     if response.ok:
        #         data = response.json()
        #         console.print(f"[green]Results uploaded to cloud: {data.get('url')}[/green]")
        #     else:
        #         console.print(f"[red]Cloud upload failed: {response.text}[/red]")
        # except Exception as e:
        #     console.print(f"[red]Cloud upload error: {e}[/red]")

        console.print("[yellow]Cloud upload would be triggered here (stub)[/yellow]")

    def report_sample(self, sample_eval: SampleEvaluation) -> None:  # noqa: ARG002
        """Stream individual sample to cloud for real-time progress tracking.

        Args:
            sample_eval: Individual sample evaluation result (unused until service ready)
        """
        if not self.enabled or not self.api_key:
            return

        # TODO: When cloud service is ready, implement streaming:
        # try:
        #     requests.post(
        #         f"{self.endpoint}/runs/{self.run_id}/samples",
        #         json=sample_eval.model_dump(),
        #         headers={
        #             "Authorization": f"Bearer {self.api_key}",
        #             "Content-Type": "application/json",
        #         },
        #         timeout=10,
        #     )
        # except Exception:
        #     pass  # Silently fail on streaming errors

    def _prepare_payload(self, result: EvaluationResult) -> dict:
        """Prepare evaluation result for cloud API.

        Args:
            result: Evaluation result

        Returns:
            Payload dict for API
        """
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "model": result.config.model_path,
            "metrics": result.metrics.model_dump(),
            "samples": [s.model_dump() for s in result.predictions],
            "config": {
                "evaluators": getattr(result.config, "evaluators", ["tool_calling"]),
                "inference": result.config.inference_config.model_dump(),
            },
            "metadata": {
                "samples_evaluated": result.metrics.samples_evaluated,
                "samples_processed": result.metrics.samples_processed,
                "processing_errors": result.metrics.processing_errors,
            },
        }
