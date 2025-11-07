"""Main evaluator for running model evaluation."""

import json

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..metrics import trace
from ..schemas import ToolDefinition
from .evaluators import EvaluationContext, EvaluatorRegistry, EvaluatorResult
from .inference import InferenceConfig, ModelResponse, create_inference_backend
from .metrics import (
    EvaluationMetrics,
    SampleEvaluation,
    compute_metrics,
)
from .parser import GroundTruth, GroundTruthParser
from .reporters import BaseReporter, CloudReporter, FileReporter, MultiReporter

console = Console()


class EvaluatorConfig(BaseModel):
    """Configuration for evaluation run."""

    model_path: str = Field(description="Path to model to evaluate")
    dataset_path: str = Field(description="Path to evaluation dataset (JSONL)")
    output_path: str | None = Field(
        default=None,
        description="Path to save evaluation results",
    )
    inference_config: InferenceConfig = Field(
        description="Inference backend configuration",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for evaluation",
    )
    max_samples: int | None = Field(
        default=None,
        description="Maximum number of samples to evaluate (None for all)",
    )
    save_predictions: bool = Field(
        default=True,
        description="Save individual predictions to output file",
    )
    metric_weights: dict[str, float] | None = Field(
        default=None,
        description="Custom weights for overall score computation",
    )
    evaluators: list[str] | dict[str, dict] = Field(
        default=["tool_calling"],
        description="List of evaluator names or dict of name -> config",
    )
    reporters: list[str] | dict[str, dict] = Field(
        default=["file"],
        description="List of reporter names or dict of name -> config",
    )
    cloud_api_key: str | None = Field(
        default=None,
        description="DeepFabric cloud API key (or use DEEPFABRIC_API_KEY env var)",
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    metrics: EvaluationMetrics = Field(description="Computed metrics")
    predictions: list[SampleEvaluation] = Field(
        description="Individual sample evaluations",
    )
    config: EvaluatorConfig = Field(description="Evaluation configuration used")


class Evaluator:
    """Orchestrates model evaluation on tool-calling tasks."""

    def __init__(self, config: EvaluatorConfig):
        """Initialize evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.backend = create_inference_backend(config.inference_config)
        # Parser will be configured per-sample based on conversation metadata
        self.parser: GroundTruthParser | None = None

        # Initialize evaluator registry and active evaluators
        self.registry = EvaluatorRegistry()
        self.active_evaluators = self._initialize_evaluators()

        # Initialize reporters
        self.reporter = self._initialize_reporters()

        # Track evaluator creation
        trace(
            "evaluator_created",
            {
                "backend": self.config.inference_config.backend,
                "model_path": self.config.model_path,
                "has_adapter": self.config.inference_config.adapter_path is not None,
                "evaluators": (
                    list(self.config.evaluators)
                    if isinstance(self.config.evaluators, list)
                    else list(self.config.evaluators.keys())
                ),
                "reporters": (
                    list(self.config.reporters)
                    if isinstance(self.config.reporters, list)
                    else list(self.config.reporters.keys())
                ),
            },
        )

    def _initialize_evaluators(self) -> list:
        """Initialize evaluators based on config.

        Returns:
            List of active evaluator instances
        """
        evaluators = []

        if isinstance(self.config.evaluators, list):
            # Simple list of names
            for name in self.config.evaluators:
                evaluators.append(self.registry.get(name))
        else:
            # Dict with configs
            for name, eval_config in self.config.evaluators.items():
                evaluators.append(self.registry.get(name, config=eval_config))

        return evaluators

    def _initialize_reporters(self) -> BaseReporter:
        """Initialize reporters based on config.

        Returns:
            Reporter instance (may be MultiReporter)
        """
        reporters: list[BaseReporter] = []

        if isinstance(self.config.reporters, list):
            # Simple list of names
            for name in self.config.reporters:
                if name == "file":
                    reporters.append(FileReporter({"path": self.config.output_path}))
                elif name == "cloud":
                    reporters.append(CloudReporter({"api_key": self.config.cloud_api_key}))
        else:
            # Dict with configs
            for name, reporter_config in self.config.reporters.items():
                if name == "file":
                    # Merge output_path if not in config
                    if "path" not in reporter_config and self.config.output_path:
                        reporter_config["path"] = self.config.output_path
                    reporters.append(FileReporter(reporter_config))
                elif name == "cloud":
                    # Merge api_key if not in config
                    if "api_key" not in reporter_config and self.config.cloud_api_key:
                        reporter_config["api_key"] = self.config.cloud_api_key
                    reporters.append(CloudReporter(reporter_config))

        # Return single reporter or MultiReporter
        if len(reporters) == 0:
            # Default to file reporter
            return FileReporter({"path": self.config.output_path})
        if len(reporters) == 1:
            return reporters[0]
        return MultiReporter(reporters)

    def load_dataset(self) -> list[dict[str, Any]]:
        """Load evaluation dataset from JSONL file.

        Returns:
            List of dataset samples

        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset format is invalid
        """
        dataset_path = Path(self.config.dataset_path)
        if not dataset_path.exists():
            msg = f"Dataset file not found: {dataset_path}"
            raise FileNotFoundError(msg)

        samples = []
        with dataset_path.open() as f:
            for line_num, line in enumerate(f, 1):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    msg = f"Invalid JSON on line {line_num}: {e}"
                    raise ValueError(msg) from e

        if self.config.max_samples is not None:
            samples = samples[: self.config.max_samples]

        return samples

    def extract_ground_truth(self, sample: dict[str, Any]) -> GroundTruth:
        """Extract ground truth from sample.

        Args:
            sample: Dataset sample

        Returns:
            Parsed ground truth
        """
        # Create parser for this sample's conversation type
        from ..schemas import Conversation  # noqa: PLC0415

        # Convert sample dict to Conversation object
        conversation = Conversation.model_validate(sample)

        # Determine conversation type from metadata
        metadata = conversation.metadata or {}
        conv_type = metadata.get("conversation_type", "basic")
        reasoning_style = metadata.get("reasoning_style")
        agent_mode = metadata.get("agent_mode")

        # Create parser with appropriate config
        parser = GroundTruthParser(
            conversation_type=conv_type,  # type: ignore[arg-type]
            reasoning_style=reasoning_style,  # type: ignore[arg-type]
            agent_mode=agent_mode,  # type: ignore[arg-type]
        )

        return parser.parse(conversation)

    def prepare_messages(self, sample: dict[str, Any]) -> list[dict[str, str]]:
        """Prepare messages for model inference.

        Extracts conversation up to the assistant's tool call.

        Args:
            sample: Dataset sample

        Returns:
            List of messages for inference
        """
        messages = []
        for msg in sample["messages"]:
            # Stop before first assistant message (where tool call should be generated)
            if msg["role"] == "assistant":
                break
            messages.append({"role": msg["role"], "content": msg["content"]})

        return messages

    def prepare_tools(self, sample: dict[str, Any]) -> list[ToolDefinition]:
        """Prepare tool definitions from sample.

        Args:
            sample: Dataset sample

        Returns:
            List of available tools
        """
        from ..schemas import Conversation  # noqa: PLC0415

        # Convert to Conversation to access tool_context
        conversation = Conversation.model_validate(sample)

        if conversation.tool_context is None or not conversation.tool_context.available_tools:
            return []

        return conversation.tool_context.available_tools

    def evaluate_sample(
        self,
        sample: dict[str, Any],
        sample_id: int,
    ) -> SampleEvaluation:
        """Evaluate a single sample using configured evaluators.

        Args:
            sample: Dataset sample
            sample_id: Sample index

        Returns:
            Evaluation result for this sample
        """
        try:
            # Extract ground truth
            ground_truth = self.extract_ground_truth(sample)

            # Prepare inputs
            messages = self.prepare_messages(sample)
            tools = self.prepare_tools(sample)

            # Run inference
            response: ModelResponse = self.backend.generate(messages, tools)

            # Create evaluation context
            context = EvaluationContext(
                messages=messages,
                tools=tools,
                sample_id=sample_id,
            )

            # Run all active evaluators
            evaluator_results: list[EvaluatorResult] = []
            for evaluator in self.active_evaluators:
                result = evaluator.evaluate(ground_truth, response, context)
                if result is not None:  # Evaluator may skip
                    evaluator_results.append(result)

            # Aggregate results for backwards compatibility
            return self._aggregate_results(
                sample_id=sample_id,
                ground_truth=ground_truth,
                response=response,
                evaluator_results=evaluator_results,
            )

        except Exception as e:  # noqa: BLE001
            # Return failed evaluation with safe defaults
            query = ""
            expected_tool = None
            expected_params: dict[str, Any] = {}
            expected_answer = None

            # Try to extract ground truth if available
            try:
                gt = self.extract_ground_truth(sample)
                query = gt.query
                expected_tool = gt.expected_tool
                expected_params = gt.expected_parameters
                expected_answer = gt.expected_answer
            except Exception:  # noqa: BLE001, S110
                pass  # nosec

            return SampleEvaluation(
                sample_id=sample_id,
                query=query,
                expected_tool=expected_tool,
                predicted_tool=None,
                expected_parameters=expected_params,
                predicted_parameters={},
                expected_answer=expected_answer,
                predicted_answer=None,
                tool_selection_correct=False,
                parameters_correct=False,
                execution_valid=False,
                response_score=0.0,
                error=str(e),
            )

    def _aggregate_results(
        self,
        sample_id: int,
        ground_truth: GroundTruth,
        response: ModelResponse,
        evaluator_results: list[EvaluatorResult],
    ) -> SampleEvaluation:
        """Aggregate evaluator results into SampleEvaluation.

        Args:
            sample_id: Sample index
            ground_truth: Expected values
            response: Model response
            evaluator_results: Results from all evaluators

        Returns:
            SampleEvaluation with aggregated metrics
        """
        # Extract tool calling metrics from evaluator results
        tool_correct = False
        params_correct = False
        execution_valid = False
        predicted_tool = None
        predicted_params = {}

        # Extract predictions from response
        if response.tool_call:
            predicted_tool = response.tool_call.get("name")
            predicted_params = response.tool_call.get("arguments", {})

        # Get metrics from tool_calling evaluator
        for result in evaluator_results:
            if result.evaluator_name == "tool_calling":
                metrics = result.metrics
                tool_correct = metrics.get("tool_selection_accuracy", 0.0) == 1.0
                params_correct = metrics.get("parameter_accuracy", 0.0) == 1.0
                execution_valid = metrics.get("execution_valid", 0.0) == 1.0

        # Return backwards-compatible SampleEvaluation
        return SampleEvaluation(
            sample_id=sample_id,
            query=ground_truth.query,
            expected_tool=ground_truth.expected_tool,
            predicted_tool=predicted_tool,
            expected_parameters=ground_truth.expected_parameters,
            predicted_parameters=predicted_params,
            expected_answer=ground_truth.expected_answer,
            predicted_answer=response.content,
            tool_selection_correct=tool_correct,
            parameters_correct=params_correct,
            execution_valid=execution_valid,
            response_score=0.0,  # TODO: Could use semantic similarity for response quality evaluation in the future, but disabled for tool-calling mode
            error=None,
        )

    def evaluate(self) -> EvaluationResult:
        """Run full evaluation.

        Returns:
            Complete evaluation result with metrics and predictions
        """
        console.print("[bold blue]Loading dataset...[/bold blue]")
        samples = self.load_dataset()
        console.print(f"Loaded {len(samples)} samples")

        console.print("[bold blue]Running evaluation...[/bold blue]")
        evaluations = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Evaluating {len(samples)} samples...",
                total=len(samples),
            )

            for idx, sample in enumerate(samples):
                progress.update(task, description=f"Evaluating sample {idx + 1}/{len(samples)}...")
                eval_result = self.evaluate_sample(sample, idx)
                evaluations.append(eval_result)

                # Stream sample to reporters (for cloud real-time tracking)
                self.reporter.report_sample(eval_result)

                progress.update(task, advance=1)

        console.print("[bold green]Evaluation complete![/bold green]")

        # Compute metrics
        metrics = compute_metrics(evaluations, self.config.metric_weights)

        # Create result
        result = EvaluationResult(
            metrics=metrics,
            predictions=evaluations,
            config=self.config,
        )

        # Track evaluation completion
        trace(
            "evaluation_completed",
            {
                "backend": self.config.inference_config.backend,
                "model_path": self.config.model_path,
                "has_adapter": self.config.inference_config.adapter_path is not None,
                "samples_evaluated": metrics.samples_evaluated,
                "samples_processed": metrics.samples_processed,
                "processing_errors": metrics.processing_errors,
                "tool_selection_accuracy": round(metrics.tool_selection_accuracy, 4),
                "parameter_accuracy": round(metrics.parameter_accuracy, 4),
                "execution_success_rate": round(metrics.execution_success_rate, 4),
                "overall_score": round(metrics.overall_score, 4),
                "success": metrics.processing_errors == 0,
            },
        )

        # Report results using configured reporters
        if self.config.save_predictions:
            self.reporter.report(result)

        return result

    def cleanup(self) -> None:
        """Clean up resources."""
        self.backend.cleanup()

    def print_summary(self, metrics: EvaluationMetrics) -> None:
        """Print evaluation summary.

        Args:
            metrics: Computed metrics
        """
        console.print("\n[bold]Evaluation Summary[/bold]")
        console.print(f"Samples Evaluated: {metrics.samples_evaluated}")
        console.print(f"Processed Successfully: {metrics.samples_processed}")
        console.print(f"Processing Errors: {metrics.processing_errors}")
        console.print("\n[bold]Metrics[/bold]")
        console.print(f"Tool Selection Accuracy: {metrics.tool_selection_accuracy:.2%}")
        console.print(f"Parameter Accuracy: {metrics.parameter_accuracy:.2%}")
        console.print(f"Execution Success Rate: {metrics.execution_success_rate:.2%}")
        console.print(f"Response Quality: {metrics.response_quality:.2%}")
        console.print(f"\n[bold green]Overall Score: {metrics.overall_score:.2%}[/bold green]")
