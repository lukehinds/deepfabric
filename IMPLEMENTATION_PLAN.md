# DeepFabric Evaluation System - Implementation Plan

## 📋 Document Information

- **Version:** 1.0
- **Created:** 2025-10-31
- **Status:** Planning
- **Scope:** Local evaluation system (Phases 1-4)
- **Timeline:** 4-5 months
- **Team Size:** 2-3 engineers

> **Note:** The Cloud SaaS platform is a separate project. See `CLOUD_SAAS_DESIGN.md` for the cloud platform specification.

---

## 🎯 Executive Summary

This implementation plan delivers a production-ready local evaluation system for DeepFabric in 4 phases over 4-5 months. The system enables users to:

1. Split datasets into train/eval sets with stratification
2. Evaluate fine-tuned models with comprehensive metrics
3. Generate rich reports (JSON, HTML, CSV, model cards)
4. Compare models and track improvements over time
5. Integrate with MLflow, W&B, and HuggingFace Hub

**Key Principle:** Evaluation operates on original DeepFabric format (not formatted training data) to preserve ground truth and enable accurate metric calculation.

---

## 📊 Phase Overview

| Phase | Duration | Focus | Status |
|-------|----------|-------|--------|
| Phase 1 | Weeks 1-3 | Foundation & Dataset Splitting | ⏸️ Not Started |
| Phase 2 | Weeks 4-8 | Core Evaluation Engine | ⏸️ Not Started |
| Phase 3 | Weeks 9-12 | Enhanced Metrics & Reporting | ⏸️ Not Started |
| Phase 4 | Weeks 13-16 | Comparison & Integrations | ⏸️ Not Started |

---

## 🚀 Phase 1: Foundation & Dataset Splitting

**Duration:** Weeks 1-3
**Status:** ⏸️ Not Started
**Engineer:** TBD

### Objectives

Establish the foundation for evaluation by implementing dataset splitting and basic configuration infrastructure.

### Deliverables

#### 1.1 Dataset Splitting System

**Files to Create:**
- `deepfabric/evaluation/__init__.py`
- `deepfabric/evaluation/split.py`

**Implementation:**

```python
# deepfabric/evaluation/split.py

def split_dataset(
    dataset_path: str,
    train_output: str,
    eval_output: str,
    test_size: float = 0.2,
    stratify_by: Optional[str] = None,
    seed: int = 42,
    shuffle: bool = True
) -> Tuple[Dataset, Dataset]:
    """
    Split existing dataset into train/eval sets with stratification.

    Args:
        dataset_path: Path to original dataset JSONL file
        train_output: Output path for training set
        eval_output: Output path for evaluation set
        test_size: Fraction of data for evaluation (0.0 to 1.0)
        stratify_by: Field to stratify by (topic, tool, conversation_type)
        seed: Random seed for reproducibility
        shuffle: Whether to shuffle before splitting

    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    pass
```

**Key Features:**
- Load existing Dataset from JSONL
- Use HuggingFace datasets `.train_test_split()` for stratification
- Preserve all metadata and conversation structure
- Validate conversation integrity (multi-turn conversations stay together)
- Support stratification by topic, tool, or conversation_type

**Tests:**
- `tests/evaluation/test_split.py`
- Test stratification correctness
- Test metadata preservation
- Test edge cases (empty datasets, single sample, etc.)

---

#### 1.2 Evaluation Configuration Schema

**Files to Modify:**
- `deepfabric/config.py`
- `deepfabric/schemas.py`

**Implementation:**

Add `EvaluationConfig` class to `config.py`:

```python
@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""

    # Conversation settings (must match dataset generation)
    conversation_type: str
    reasoning_style: Optional[str] = None
    agent_mode: str = "single_turn"

    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "tool_selection_accuracy",
        "parameter_accuracy",
        "execution_success_rate",
        "response_quality"
    ])

    # Pass/fail thresholds
    thresholds: Dict[str, float] = field(default_factory=dict)

    # Metric weights for overall score
    weights: Dict[str, float] = field(default_factory=lambda: {
        "tool_selection": 0.40,
        "parameter_accuracy": 0.30,
        "execution_success": 0.20,
        "response_quality": 0.10
    })

    # Output configuration
    output_dir: str = "./eval_results"
    output_formats: List[str] = field(default_factory=lambda: ["json", "html", "csv"])
    include_failures: bool = True
    generate_charts: bool = True

    # Inference settings
    batch_size: int = 1
    max_samples: Optional[int] = None
```

Add evaluation section to YAML schema:

```yaml
# Example config.yaml with evaluation section
evaluation:
  conversation_type: "chain_of_thought"
  reasoning_style: "hybrid"
  agent_mode: "single_turn"

  metrics:
    - tool_selection_accuracy
    - parameter_accuracy
    - execution_success_rate

  thresholds:
    tool_selection_accuracy: 0.85
    overall_score: 0.80

  weights:
    tool_selection: 0.40
    parameter_accuracy: 0.30
    execution_success: 0.20
    response_quality: 0.10
```

**Tests:**
- `tests/test_config.py` (extend existing)
- Test YAML parsing with evaluation section
- Test config validation
- Test default values

---

#### 1.3 Ground Truth Parser

**Files to Create:**
- `deepfabric/evaluation/parser.py`

**Implementation:**

```python
# deepfabric/evaluation/parser.py

@dataclass
class GroundTruth:
    """Parsed ground truth from original dataset sample."""

    query: str
    expected_tool: str
    expected_parameters: Dict[str, Any]
    tool_schema: Dict[str, Any]
    expected_answer: Optional[str] = None
    conversation_type: str = "basic"
    metadata: Dict[str, Any] = field(default_factory=dict)


class GroundTruthParser:
    """Parse ground truth from original DeepFabric JSONL format."""

    def __init__(self, conversation_type: str):
        self.conversation_type = conversation_type

    def parse(self, sample: Dict[str, Any]) -> GroundTruth:
        """
        Extract ground truth from dataset sample.

        Handles all conversation types:
        - basic: Simple Q&A
        - structured: With metadata
        - chain_of_thought: With reasoning traces
        - agent (single_turn, multi_turn): With tool calls
        """
        pass

    def extract_tool_call(self, message: str) -> Tuple[str, Dict[str, Any]]:
        """Extract tool name and parameters from assistant message."""
        pass

    def get_tool_schema(self, sample: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
        """Get tool schema from tool_context.available_tools."""
        pass
```

**Key Features:**
- Parse all conversation types (basic, structured, chain_of_thought)
- Handle both agent modes (single_turn, multi_turn)
- Extract tool schemas from `tool_context.available_tools`
- Support multiple tool call formats (XML, JSON, Python-like)
- Robust regex extraction with fallbacks

**Tests:**
- `tests/evaluation/test_parser.py`
- Test all conversation types
- Test tool call extraction with various formats
- Test schema extraction
- Test error handling for malformed data

---

#### 1.4 CLI Command: `deepfabric split`

**Files to Modify:**
- `deepfabric/cli.py`

**Implementation:**

```python
@cli.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--train-output", required=True, help="Output path for training set")
@click.option("--eval-output", required=True, help="Output path for evaluation set")
@click.option("--test-size", default=0.2, help="Fraction for eval (default: 0.2)")
@click.option("--stratify-by", type=click.Choice(["topic", "tool", "conversation_type"]), help="Field to stratify by")
@click.option("--seed", default=42, help="Random seed for reproducibility")
@click.option("--shuffle/--no-shuffle", default=True, help="Shuffle before splitting")
def split(dataset_path, train_output, eval_output, test_size, stratify_by, seed, shuffle):
    """Split dataset into train/eval sets with stratification."""

    from deepfabric.evaluation.split import split_dataset

    console.print(f"[bold]Splitting dataset:[/bold] {dataset_path}")
    console.print(f"  Test size: {test_size*100:.1f}%")
    if stratify_by:
        console.print(f"  Stratify by: {stratify_by}")

    train_dataset, eval_dataset = split_dataset(
        dataset_path=dataset_path,
        train_output=train_output,
        eval_output=eval_output,
        test_size=test_size,
        stratify_by=stratify_by,
        seed=seed,
        shuffle=shuffle
    )

    console.print(f"\n✓ [green]Split complete![/green]")
    console.print(f"  Train: {len(train_dataset)} samples → {train_output}")
    console.print(f"  Eval:  {len(eval_dataset)} samples → {eval_output}")
```

**Tests:**
- `tests/test_cli.py` (extend existing)
- Test CLI with various options
- Test error handling for invalid paths
- Test output validation

---

### Phase 1 Acceptance Criteria

- [ ] Dataset splitting works correctly with stratification
- [ ] All conversation types are preserved during split
- [ ] Evaluation config loads from YAML
- [ ] Ground truth parser handles all conversation types
- [ ] CLI command `deepfabric split` is functional
- [ ] 100% test coverage for all Phase 1 components
- [ ] Documentation updated with split command examples

### Phase 1 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Stratification breaks conversation integrity | High | Group multi-turn conversations before splitting |
| Tool schema not present in old datasets | Medium | Fallback to schema inference from parameters |
| Conversation type detection fails | High | Require explicit conversation_type in config |

---

## 🔧 Phase 2: Core Evaluation Engine

**Duration:** Weeks 4-8
**Status:** ⏸️ Not Started
**Engineer:** TBD

### Objectives

Build the core evaluation engine with model inference, metric calculation, and JSON output.

### Deliverables

#### 2.1 Evaluation Engine Architecture

**Files to Create:**
- `deepfabric/evaluation/engine.py`
- `deepfabric/evaluation/inference.py`
- `deepfabric/evaluation/metrics/__init__.py`

**Implementation:**

```python
# deepfabric/evaluation/engine.py

@dataclass
class EvaluationResults:
    """Container for all evaluation results."""

    metadata: Dict[str, Any]
    summary: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    per_topic_breakdown: Dict[str, Any]
    failure_analysis: Dict[str, Any]
    threshold_checks: Dict[str, bool]

    def to_json(self, path: str):
        """Export as JSON."""
        pass

    def to_html(self, path: str):
        """Generate HTML report."""
        pass

    def to_csv(self, path: str):
        """Export detailed CSV."""
        pass

    def generate_model_card(self, path: str):
        """Create model card section."""
        pass


class EvaluationEngine:
    """Main evaluation orchestrator."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = self._initialize_metrics()
        self.parser = GroundTruthParser(config.conversation_type)

    def evaluate(
        self,
        model_path: str,
        eval_dataset: str,
        output_dir: str
    ) -> EvaluationResults:
        """
        Run full evaluation pipeline.

        Steps:
        1. Load model and dataset
        2. Run inference on eval samples
        3. Parse model outputs
        4. Compute all metrics
        5. Analyze failures
        6. Generate outputs
        7. Check thresholds
        """
        pass

    def _initialize_metrics(self) -> List[Metric]:
        """Initialize metric calculators based on config."""
        pass

    def _run_inference_batch(self, model, samples: List[Dict]) -> List[str]:
        """Run batch inference."""
        pass

    def _parse_model_output(self, output: str) -> Prediction:
        """Parse model output to extract tool call."""
        pass
```

**Key Design Decisions:**
- Plugin architecture for metrics (easy to add new metrics)
- Conversation-type-aware parsing
- Batch inference with progress tracking
- Graceful error handling (failed samples don't crash evaluation)
- Extensible output system

---

#### 2.2 Model Inference Runner

**Files to Create:**
- `deepfabric/evaluation/inference.py`

**Implementation:**

```python
# deepfabric/evaluation/inference.py

class ModelInferenceRunner:
    """Handle model loading and batch inference."""

    def __init__(self, model_path: str, batch_size: int = 1):
        self.model_path = model_path
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load model from path.

        Supports:
        - HuggingFace Hub paths (org/model-name)
        - Local directories
        - GGUF files (via llama-cpp-python)
        - Quantized models (GPTQ, AWQ)
        """
        pass

    def generate(self, prompts: List[str]) -> List[str]:
        """Run batch inference."""
        pass

    def format_prompt(self, query: str, conversation_type: str) -> str:
        """Format query for model input."""
        pass


@dataclass
class Prediction:
    """Model prediction for a single sample."""

    raw_output: str
    tool_name: Optional[str] = None
    tool_parameters: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    answer: Optional[str] = None
    parse_error: Optional[str] = None


class ResponseParser:
    """Parse model outputs to extract structured predictions."""

    def parse(self, output: str, conversation_type: str) -> Prediction:
        """
        Parse model output.

        Handles multiple formats:
        - XML: <tool_call>func(args)</tool_call>
        - JSON: {"tool_calls": [...]}
        - Python-like: func(arg1, arg2)
        - Plain text (extract with regex)
        """
        pass

    def _extract_tool_call_xml(self, output: str) -> Tuple[str, Dict]:
        """Extract from XML format."""
        pass

    def _extract_tool_call_json(self, output: str) -> Tuple[str, Dict]:
        """Extract from JSON format."""
        pass
```

**Key Features:**
- Support multiple model formats (HF, GGUF, quantized)
- Robust parsing with fallbacks
- Error tracking (parse failures)
- Progress tracking with rich progress bars

---

#### 2.3 Basic Metric Calculators

**Files to Create:**
- `deepfabric/evaluation/metrics/base.py`
- `deepfabric/evaluation/metrics/tool_selection.py`
- `deepfabric/evaluation/metrics/parameter_accuracy.py`

**Implementation:**

```python
# deepfabric/evaluation/metrics/base.py

class Metric(ABC):
    """Base class for all metrics."""

    @abstractmethod
    def compute(
        self,
        predictions: List[Prediction],
        ground_truth: List[GroundTruth]
    ) -> Dict[str, Any]:
        """Compute metric values."""
        pass

    @abstractmethod
    def name(self) -> str:
        """Metric identifier."""
        pass


# deepfabric/evaluation/metrics/tool_selection.py

class ToolSelectionMetric(Metric):
    """
    Evaluate tool selection correctness.

    Computes:
    - Accuracy: Percentage of correct tool selections
    - Precision: TP / (TP + FP) for multi-tool scenarios
    - Recall: TP / (TP + FN) for multi-tool scenarios
    - F1: Harmonic mean of precision and recall
    - Confusion matrix: Tool → Tool mapping
    """

    def compute(self, predictions, ground_truth):
        accuracy = self._compute_accuracy(predictions, ground_truth)
        precision, recall, f1 = self._compute_prf(predictions, ground_truth)
        confusion_matrix = self._compute_confusion_matrix(predictions, ground_truth)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix
        }


# deepfabric/evaluation/metrics/parameter_accuracy.py

class ParameterAccuracyMetric(Metric):
    """
    Evaluate parameter correctness.

    Computes:
    - Exact match rate: All parameters exactly correct
    - Field-level accuracy: Per-field correctness
    - Type correctness: Parameters have correct types
    - Per-tool accuracy: Breakdown by tool
    """

    def compute(self, predictions, ground_truth):
        exact_match = self._compute_exact_match(predictions, ground_truth)
        field_accuracy = self._compute_field_accuracy(predictions, ground_truth)
        type_correctness = self._compute_type_correctness(predictions, ground_truth)
        per_tool = self._compute_per_tool_accuracy(predictions, ground_truth)

        return {
            "exact_match_rate": exact_match,
            "field_accuracy": field_accuracy,
            "type_correctness": type_correctness,
            "per_tool_accuracy": per_tool
        }

    def _compare_parameters(self, pred: Dict, truth: Dict) -> float:
        """
        Compare parameter dicts with fuzzy matching.

        Handles:
        - Type coercion (string "5" == int 5)
        - Floating point tolerance
        - String normalization (case, whitespace)
        """
        pass
```

**Tests:**
- `tests/evaluation/metrics/test_tool_selection.py`
- `tests/evaluation/metrics/test_parameter_accuracy.py`
- Test with various prediction scenarios
- Test edge cases (empty predictions, mismatches)
- Test confusion matrix generation

---

#### 2.4 JSON Output System

**Files to Create:**
- `deepfabric/evaluation/outputs/json_reporter.py`
- `deepfabric/evaluation/schemas.py`

**Implementation:**

```python
# deepfabric/evaluation/schemas.py

from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class EvaluationMetadata(BaseModel):
    timestamp: str
    model_name: str
    model_path: str
    eval_dataset: str
    eval_dataset_size: int
    deepfabric_version: str
    evaluation_duration_seconds: float

class MetricSummary(BaseModel):
    overall_score: float
    grade: str
    tool_selection_accuracy: float
    parameter_accuracy: float
    execution_success_rate: float
    response_quality: float

class EvaluationResultSchema(BaseModel):
    evaluation_metadata: EvaluationMetadata
    summary: MetricSummary
    detailed_metrics: Dict[str, Any]
    per_topic_breakdown: Dict[str, Any]
    failure_analysis: Dict[str, Any]
    thresholds: Dict[str, Any]


# deepfabric/evaluation/outputs/json_reporter.py

class JSONReporter:
    """Generate JSON output from evaluation results."""

    def generate(self, results: EvaluationResults, output_path: str):
        """
        Generate comprehensive JSON report.

        Schema matches design document specification.
        """
        schema = EvaluationResultSchema(
            evaluation_metadata=self._build_metadata(results),
            summary=self._build_summary(results),
            detailed_metrics=results.detailed_metrics,
            per_topic_breakdown=results.per_topic_breakdown,
            failure_analysis=results.failure_analysis,
            thresholds=results.threshold_checks
        )

        with open(output_path, 'w') as f:
            json.dump(schema.dict(), f, indent=2)
```

**Key Features:**
- Pydantic schemas for validation
- Matches design document JSON structure exactly
- Pretty-printed with indentation
- Validation ensures no missing fields

---

#### 2.5 CLI Command: `deepfabric evaluate`

**Files to Modify:**
- `deepfabric/cli.py`

**Implementation:**

```python
@cli.command()
@click.option("--model", required=True, help="Path to fine-tuned model")
@click.option("--eval-dataset", required=True, help="Evaluation dataset path")
@click.option("--conversation-type", required=True, help="Conversation type")
@click.option("--reasoning-style", help="Reasoning style (for CoT)")
@click.option("--agent-mode", default="single_turn", help="Agent mode")
@click.option("--output-dir", default="./eval_results", help="Output directory")
@click.option("--format", type=click.Choice(["json", "html", "csv", "all"]), default="all")
@click.option("--threshold", multiple=True, help="Threshold: metric=value")
@click.option("--batch-size", default=1, help="Inference batch size")
@click.option("--max-samples", type=int, help="Limit evaluation samples")
def evaluate(model, eval_dataset, conversation_type, reasoning_style, agent_mode,
             output_dir, format, threshold, batch_size, max_samples):
    """Evaluate fine-tuned model on eval dataset."""

    from deepfabric.evaluation.engine import EvaluationEngine

    # Parse thresholds
    thresholds = {}
    for t in threshold:
        metric, value = t.split("=")
        thresholds[metric] = float(value)

    # Build config
    config = EvaluationConfig(
        conversation_type=conversation_type,
        reasoning_style=reasoning_style,
        agent_mode=agent_mode,
        output_dir=output_dir,
        output_formats=[format] if format != "all" else ["json", "html", "csv"],
        thresholds=thresholds,
        batch_size=batch_size,
        max_samples=max_samples
    )

    # Run evaluation
    console.print(f"[bold]Evaluating model:[/bold] {model}")
    console.print(f"  Dataset: {eval_dataset}")
    console.print(f"  Conversation type: {conversation_type}\n")

    engine = EvaluationEngine(config)

    with Progress() as progress:
        task = progress.add_task("Evaluating...", total=100)
        results = engine.evaluate(model, eval_dataset, output_dir)

    # Display summary
    console.print("\n[bold green]✓ Evaluation Complete![/bold green]\n")

    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="magenta")

    table.add_row("Overall Score", f"{results.summary['overall_score']:.2%}")
    table.add_row("Grade", results.summary['grade'])
    table.add_row("Tool Selection", f"{results.summary['tool_selection_accuracy']:.2%}")
    table.add_row("Parameter Accuracy", f"{results.summary['parameter_accuracy']:.2%}")

    console.print(table)

    # Check thresholds
    if thresholds:
        all_passed = all(results.threshold_checks.values())
        if all_passed:
            console.print("\n[bold green]✓ All thresholds passed![/bold green]")
            sys.exit(0)
        else:
            console.print("\n[bold red]✗ Some thresholds failed![/bold red]")
            sys.exit(1)
```

**Key Features:**
- Rich progress bars during evaluation
- Summary table with color coding
- Threshold validation with exit codes (for CI/CD)
- Links to generated reports

---

### Phase 2 Acceptance Criteria

- [ ] Evaluation engine runs end-to-end
- [ ] Model inference works for HuggingFace models
- [ ] Tool selection metrics calculated correctly
- [ ] Parameter accuracy metrics calculated correctly
- [ ] JSON output matches design document schema
- [ ] CLI command `deepfabric evaluate` is functional
- [ ] Progress tracking works smoothly
- [ ] 90%+ test coverage for Phase 2 components
- [ ] Can evaluate a real fine-tuned model successfully

### Phase 2 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Model loading fails for various formats | High | Start with HF models only, expand later |
| Response parsing fails on edge cases | High | Extensive regex testing, graceful degradation |
| Inference is too slow | Medium | Implement batch processing, GPU support |
| Memory issues with large models | Medium | Implement streaming inference |

---

## 📊 Phase 3: Enhanced Metrics & Reporting

**Duration:** Weeks 9-12
**Status:** ⏸️ Not Started
**Engineer:** TBD

### Objectives

Add advanced metrics, rich HTML reports, CSV exports, and model card generation.

### Deliverables

#### 3.1 Advanced Metric Calculators

**Files to Create:**
- `deepfabric/evaluation/metrics/execution_success.py`
- `deepfabric/evaluation/metrics/response_quality.py`
- `deepfabric/evaluation/metrics/efficiency.py`

**Implementation:**

```python
# deepfabric/evaluation/metrics/execution_success.py

class ExecutionSuccessMetric(Metric):
    """
    Validate tool calls would execute successfully.

    Checks:
    - Required parameters present
    - Parameter types correct
    - Parameter values valid (constraints)
    - JSON well-formed

    Computes:
    - Success rate
    - Error distribution by category
    """

    def compute(self, predictions, ground_truth):
        success_rate = 0.0
        error_distribution = {}

        for pred, truth in zip(predictions, ground_truth):
            errors = self._validate_tool_call(pred, truth.tool_schema)
            if not errors:
                success_rate += 1
            else:
                for error in errors:
                    error_distribution[error.category] = error_distribution.get(error.category, 0) + 1

        success_rate /= len(predictions)

        return {
            "success_rate": success_rate,
            "error_distribution": error_distribution
        }

    def _validate_tool_call(self, pred: Prediction, schema: Dict) -> List[ValidationError]:
        """Validate against JSON schema."""
        pass


# deepfabric/evaluation/metrics/response_quality.py

class ResponseQualityMetric(Metric):
    """
    Evaluate response quality.

    Computes:
    - Answer correctness (BLEU/ROUGE if ground truth available)
    - Reasoning coherence (for chain_of_thought)
    - Response length statistics
    """

    def compute(self, predictions, ground_truth):
        from nltk.translate.bleu_score import sentence_bleu
        from rouge import Rouge

        bleu_scores = []
        rouge_scores = []
        reasoning_scores = []

        for pred, truth in zip(predictions, ground_truth):
            if truth.expected_answer and pred.answer:
                bleu = sentence_bleu([truth.expected_answer.split()], pred.answer.split())
                bleu_scores.append(bleu)

            if pred.reasoning:
                reasoning_score = self._evaluate_reasoning(pred.reasoning)
                reasoning_scores.append(reasoning_score)

        return {
            "answer_correctness": np.mean(bleu_scores) if bleu_scores else None,
            "reasoning_coherence": np.mean(reasoning_scores) if reasoning_scores else None,
            "avg_response_length": np.mean([len(p.raw_output) for p in predictions])
        }


# deepfabric/evaluation/metrics/efficiency.py

class EfficiencyMetric(Metric):
    """
    Track efficiency metrics.

    Computes:
    - Average inference time
    - Average tool calls per query
    - Total tokens generated
    """

    def compute(self, predictions, ground_truth, inference_times):
        return {
            "avg_inference_time_ms": np.mean(inference_times) * 1000,
            "avg_tool_calls_per_query": np.mean([1 if p.tool_name else 0 for p in predictions]),
            "total_tokens_generated": sum([len(p.raw_output.split()) for p in predictions])
        }
```

---

#### 3.2 HTML Report Generator

**Files to Create:**
- `deepfabric/evaluation/outputs/html_reporter.py`
- `deepfabric/evaluation/templates/report.html` (Jinja2 template)

**Implementation:**

```python
# deepfabric/evaluation/outputs/html_reporter.py

class HTMLReporter:
    """Generate interactive HTML report with charts."""

    def __init__(self):
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader("deepfabric/evaluation/templates")
        )

    def generate(self, results: EvaluationResults, output_path: str):
        """
        Generate comprehensive HTML report.

        Sections:
        1. Overview (summary metrics, grade, pass/fail)
        2. Tool Selection Analysis (accuracy, confusion matrix)
        3. Parameter Analysis (field-level accuracy, examples)
        4. Execution Analysis (success rate, error categories)
        5. Topic Breakdown (per-topic performance heatmap)
        6. Failure Analysis (hardest examples, common mistakes)
        7. Recommendations (actionable suggestions)
        """
        template = self.template_env.get_template("report.html")

        html = template.render(
            metadata=results.metadata,
            summary=results.summary,
            metrics=results.detailed_metrics,
            topics=results.per_topic_breakdown,
            failures=results.failure_analysis,
            charts=self._generate_charts(results)
        )

        with open(output_path, 'w') as f:
            f.write(html)

    def _generate_charts(self, results: EvaluationResults) -> Dict[str, str]:
        """Generate chart images (base64-encoded for embedding)."""
        charts = {}

        # Confusion matrix heatmap
        charts['confusion_matrix'] = self._create_confusion_matrix_chart(
            results.detailed_metrics['tool_selection']['confusion_matrix']
        )

        # Metric breakdown bar chart
        charts['metric_breakdown'] = self._create_metric_breakdown_chart(
            results.summary
        )

        # Topic performance radar chart
        charts['topic_heatmap'] = self._create_topic_heatmap(
            results.per_topic_breakdown
        )

        # Error distribution pie chart
        charts['error_distribution'] = self._create_error_pie_chart(
            results.detailed_metrics['execution_metrics']['error_distribution']
        )

        return charts

    def _create_confusion_matrix_chart(self, confusion_matrix: Dict) -> str:
        """Create confusion matrix heatmap using matplotlib."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Convert confusion matrix to DataFrame
        df = pd.DataFrame(confusion_matrix)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues')
        plt.title('Tool Selection Confusion Matrix')
        plt.ylabel('Expected Tool')
        plt.xlabel('Predicted Tool')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        plt.close()

        return f"data:image/png;base64,{image_base64}"
```

**HTML Template Structure:**

```html
<!-- deepfabric/evaluation/templates/report.html -->
<!DOCTYPE html>
<html>
<head>
    <title>DeepFabric Evaluation Report - {{ metadata.model_name }}</title>
    <style>
        /* Modern, clean styling with Tailwind-like aesthetics */
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <h1>Evaluation Report</h1>
            <p>Model: {{ metadata.model_name }}</p>
            <p>Date: {{ metadata.timestamp }}</p>
        </header>

        <!-- Overview Section -->
        <section class="overview">
            <h2>Overview</h2>
            <div class="metric-cards">
                <div class="card {{ 'pass' if summary.grade.startswith('A') else 'warning' }}">
                    <h3>Overall Score</h3>
                    <p class="score">{{ (summary.overall_score * 100)|round(1) }}%</p>
                    <p class="grade">{{ summary.grade }}</p>
                </div>
                <!-- More metric cards -->
            </div>
        </section>

        <!-- Charts Section -->
        <section class="charts">
            <h2>Visual Analysis</h2>
            <div class="chart-grid">
                <div class="chart">
                    <h3>Confusion Matrix</h3>
                    <img src="{{ charts.confusion_matrix }}" alt="Confusion Matrix">
                </div>
                <!-- More charts -->
            </div>
        </section>

        <!-- Failure Analysis Section -->
        <section class="failures">
            <h2>Failure Analysis</h2>
            <table>
                <thead>
                    <tr>
                        <th>Error Type</th>
                        <th>Count</th>
                        <th>Example</th>
                    </tr>
                </thead>
                <tbody>
                    {% for error in failures.most_common_errors %}
                    <tr>
                        <td>{{ error.error_type }}</td>
                        <td>{{ error.count }}</td>
                        <td>{{ error.example_query }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </section>

        <!-- Recommendations Section -->
        <section class="recommendations">
            <h2>Recommendations</h2>
            <ul>
                {% for rec in recommendations %}
                <li>{{ rec }}</li>
                {% endfor %}
            </ul>
        </section>
    </div>
</body>
</html>
```

---

#### 3.3 CSV Export

**Files to Create:**
- `deepfabric/evaluation/outputs/csv_reporter.py`

**Implementation:**

```python
# deepfabric/evaluation/outputs/csv_reporter.py

class CSVReporter:
    """Generate per-example CSV export."""

    def generate(self, results: EvaluationResults, output_path: str):
        """
        Generate detailed per-example results CSV.

        Columns:
        - example_id
        - query
        - expected_tool
        - actual_tool
        - tool_correct
        - param_accuracy
        - executable
        - response_quality
        - topic
        - error_message (if any)
        """
        import csv

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'example_id', 'query', 'expected_tool', 'actual_tool',
                'tool_correct', 'param_accuracy', 'executable',
                'response_quality', 'topic', 'error_message'
            ])

            writer.writeheader()

            for i, (pred, truth) in enumerate(zip(results.predictions, results.ground_truth)):
                writer.writerow({
                    'example_id': i,
                    'query': truth.query,
                    'expected_tool': truth.expected_tool,
                    'actual_tool': pred.tool_name or 'NONE',
                    'tool_correct': pred.tool_name == truth.expected_tool,
                    'param_accuracy': self._compute_param_similarity(pred, truth),
                    'executable': self._is_executable(pred, truth),
                    'response_quality': self._compute_quality(pred, truth),
                    'topic': truth.metadata.get('topic', 'unknown'),
                    'error_message': pred.parse_error or ''
                })
```

---

#### 3.4 Model Card Generator

**Files to Create:**
- `deepfabric/evaluation/outputs/model_card_generator.py`
- `deepfabric/evaluation/templates/model_card.md` (template)

**Implementation:**

```python
# deepfabric/evaluation/outputs/model_card_generator.py

class ModelCardGenerator:
    """Generate model card section from evaluation results."""

    def generate(self, results: EvaluationResults, output_path: str):
        """
        Generate markdown section for model card.

        Sections:
        - Evaluation Results summary
        - Performance Metrics table
        - Strengths (top-performing areas)
        - Areas for Improvement (weaknesses)
        - Recommended Use Cases
        """
        template = """
## Evaluation Results

**Model**: `{{ metadata.model_name }}`
**Evaluated on**: {{ metadata.eval_dataset_size }} synthetic tool-calling examples
**Overall Score**: {{ (summary.overall_score * 100)|round }}/100 ({{ summary.grade }})
**Date**: {{ metadata.timestamp }}

### Performance Metrics

| Metric | Score |
|--------|-------|
| Tool Selection Accuracy | {{ (summary.tool_selection_accuracy * 100)|round }}% |
| Parameter Accuracy | {{ (summary.parameter_accuracy * 100)|round }}% |
| Execution Success Rate | {{ (summary.execution_success_rate * 100)|round }}% |
| Response Quality | {{ (summary.response_quality * 100)|round }}% |

### Strengths
{% for strength in strengths %}
- {{ strength }}
{% endfor %}

### Areas for Improvement
{% for area in improvements %}
- {{ area }}
{% endfor %}

### Recommended Use Cases
{% for use_case in use_cases %}
{{ use_case }}
{% endfor %}

---
*Generated with [DeepFabric](https://github.com/deepfabric/deepfabric) v{{ metadata.deepfabric_version }}*
"""

        from jinja2 import Template
        t = Template(template)

        markdown = t.render(
            metadata=results.metadata,
            summary=results.summary,
            strengths=self._extract_strengths(results),
            improvements=self._extract_improvements(results),
            use_cases=self._generate_use_cases(results)
        )

        with open(output_path, 'w') as f:
            f.write(markdown)

    def _extract_strengths(self, results: EvaluationResults) -> List[str]:
        """Identify top-performing areas."""
        strengths = []

        # Check per-topic breakdown
        for topic, metrics in results.per_topic_breakdown.items():
            if metrics['overall_score'] > 0.90:
                strengths.append(f"Excellent at {topic} ({metrics['overall_score']:.0%} accuracy)")

        # Check parameter type correctness
        if results.detailed_metrics['parameter_accuracy']['type_correctness'] > 0.95:
            strengths.append("Strong parameter type correctness")

        # Check inference speed
        if results.detailed_metrics['efficiency']['avg_inference_time_ms'] < 1000:
            strengths.append("Fast inference time")

        return strengths

    def _extract_improvements(self, results: EvaluationResults) -> List[str]:
        """Identify areas needing improvement."""
        improvements = []

        # Check for weak topics
        for topic, metrics in results.per_topic_breakdown.items():
            if metrics['overall_score'] < 0.80:
                improvements.append(f"{topic} accuracy could be higher ({metrics['overall_score']:.0%})")

        # Check common errors
        for error in results.failure_analysis['most_common_errors'][:3]:
            improvements.append(f"{error['error_type']}: {error['count']} occurrences")

        return improvements

    def _generate_use_cases(self, results: EvaluationResults) -> List[str]:
        """Generate recommended use cases based on performance."""
        use_cases = []

        if results.summary['overall_score'] >= 0.90:
            use_cases.append("✅ Production-ready for general use")
        elif results.summary['overall_score'] >= 0.85:
            use_cases.append("✅ Suitable for production with monitoring")
        else:
            use_cases.append("⚠️ Recommended for testing/development only")

        # Add specific recommendations based on topic performance
        for topic, metrics in results.per_topic_breakdown.items():
            if metrics['overall_score'] >= 0.90:
                use_cases.append(f"✅ Production-ready for {topic}")
            elif metrics['overall_score'] >= 0.85:
                use_cases.append(f"✅ Suitable for {topic}")
            else:
                use_cases.append(f"⚠️ May need fallback logic for {topic}")

        return use_cases
```

---

#### 3.5 Failure Analysis System

**Files to Create:**
- `deepfabric/evaluation/analysis/failure_analyzer.py`

**Implementation:**

```python
# deepfabric/evaluation/analysis/failure_analyzer.py

@dataclass
class FailurePattern:
    """A categorized failure pattern."""

    error_type: str
    count: int
    percentage: float
    examples: List[Dict[str, Any]]
    recommendation: str


class FailureAnalyzer:
    """Analyze failures to identify patterns and recommendations."""

    def analyze(
        self,
        predictions: List[Prediction],
        ground_truth: List[GroundTruth]
    ) -> Dict[str, Any]:
        """
        Analyze all failures and categorize.

        Returns:
        - most_common_errors: Top error types with examples
        - hardest_examples: Samples that failed multiple checks
        - recommendations: Actionable suggestions
        """
        failures = self._identify_failures(predictions, ground_truth)
        patterns = self._categorize_failures(failures)
        hardest = self._find_hardest_examples(failures)
        recommendations = self._generate_recommendations(patterns)

        return {
            "most_common_errors": patterns[:10],
            "hardest_examples": hardest[:5],
            "recommendations": recommendations
        }

    def _categorize_failures(self, failures: List[Tuple]) -> List[FailurePattern]:
        """
        Categorize failures by type.

        Categories:
        - wrong_tool_selected
        - missing_parameter
        - wrong_parameter_value
        - wrong_parameter_type
        - multiple_tool_calls_needed
        - parse_error
        - reasoning_incoherent
        """
        categories = defaultdict(list)

        for pred, truth in failures:
            if pred.tool_name != truth.expected_tool:
                categories['wrong_tool_selected'].append((pred, truth))
            elif pred.tool_parameters:
                if set(truth.expected_parameters.keys()) - set(pred.tool_parameters.keys()):
                    categories['missing_parameter'].append((pred, truth))
                # More categorization logic...

        # Convert to FailurePattern objects with recommendations
        patterns = []
        for category, examples in categories.items():
            patterns.append(FailurePattern(
                error_type=category,
                count=len(examples),
                percentage=len(examples) / len(failures),
                examples=[self._format_example(e) for e in examples[:3]],
                recommendation=self._get_recommendation(category)
            ))

        return sorted(patterns, key=lambda p: p.count, reverse=True)

    def _get_recommendation(self, error_type: str) -> str:
        """Get actionable recommendation for error type."""
        recommendations = {
            'wrong_tool_selected': "Add more diverse training examples for tool selection. Consider balancing tool distribution.",
            'missing_parameter': "Ensure training data includes all required parameters consistently.",
            'wrong_parameter_value': "Add more examples with varied parameter values. Consider data augmentation.",
            'parse_error': "Review output format consistency. May need better prompt engineering.",
        }
        return recommendations.get(error_type, "Review training data quality.")
```

---

### Phase 3 Acceptance Criteria

- [ ] All advanced metrics (execution, quality, efficiency) work correctly
- [ ] HTML report generates with interactive charts
- [ ] CSV export includes all per-example details
- [ ] Model card generator creates publication-ready markdown
- [ ] Failure analysis categorizes errors accurately
- [ ] Charts are visually appealing and informative
- [ ] Reports are useful for debugging models
- [ ] 85%+ test coverage for Phase 3 components

### Phase 3 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Chart generation is slow | Medium | Use matplotlib caching, pre-generate charts |
| HTML report is too large | Low | Implement pagination, lazy loading |
| BLEU/ROUGE dependencies conflict | Low | Make them optional with graceful fallback |

---

## 🔄 Phase 4: Comparison & Integrations

**Duration:** Weeks 13-16
**Status:** ⏸️ Not Started
**Engineer:** TBD

### Objectives

Enable multi-model comparison, time-series tracking, and external platform integrations.

### Deliverables

#### 4.1 Multi-Model Comparison Engine

**Files to Create:**
- `deepfabric/evaluation/comparison/comparator.py`
- `deepfabric/evaluation/outputs/comparison_reporter.py`

**Implementation:**

```python
# deepfabric/evaluation/comparison/comparator.py

@dataclass
class ModelComparison:
    """Comparison results for multiple models."""

    models: Dict[str, EvaluationResults]
    comparison_matrix: pd.DataFrame
    improvement_trajectory: Dict[str, List[float]]
    best_model: str
    recommendation: str


class ModelComparator:
    """Compare multiple models side-by-side."""

    def compare(
        self,
        model_results: Dict[str, EvaluationResults],
        eval_dataset: str
    ) -> ModelComparison:
        """
        Compare multiple models on same dataset.

        Args:
            model_results: Dict mapping model names to evaluation results
            eval_dataset: Dataset used for evaluation (for validation)

        Returns:
            Comprehensive comparison with recommendations
        """
        # Build comparison matrix
        comparison_df = self._build_comparison_matrix(model_results)

        # Analyze improvement trajectory
        trajectory = self._analyze_trajectory(model_results)

        # Identify best model
        best_model = self._identify_best_model(model_results)

        # Generate recommendation
        recommendation = self._generate_recommendation(comparison_df, trajectory)

        return ModelComparison(
            models=model_results,
            comparison_matrix=comparison_df,
            improvement_trajectory=trajectory,
            best_model=best_model,
            recommendation=recommendation
        )

    def _build_comparison_matrix(self, results: Dict) -> pd.DataFrame:
        """Build pandas DataFrame with all metrics for all models."""
        data = {}

        for model_name, result in results.items():
            data[model_name] = {
                'overall_score': result.summary['overall_score'],
                'tool_selection': result.summary['tool_selection_accuracy'],
                'parameter_accuracy': result.summary['parameter_accuracy'],
                'execution_success': result.summary['execution_success_rate'],
                'inference_time_ms': result.detailed_metrics['efficiency']['avg_inference_time_ms'],
            }

        return pd.DataFrame(data).T

    def _analyze_trajectory(self, results: Dict) -> Dict[str, List[float]]:
        """Analyze improvement trajectory if models are ordered chronologically."""
        trajectory = defaultdict(list)

        # If model names contain checkpoint numbers or timestamps, order them
        sorted_models = self._sort_models_chronologically(results.keys())

        for metric in ['overall_score', 'tool_selection_accuracy']:
            for model in sorted_models:
                trajectory[metric].append(results[model].summary[metric])

        return dict(trajectory)

    def _identify_best_model(self, results: Dict) -> str:
        """Identify best model based on overall score."""
        best_model = max(results.keys(), key=lambda k: results[k].summary['overall_score'])
        return best_model

    def _generate_recommendation(self, comparison_df: pd.DataFrame, trajectory: Dict) -> str:
        """Generate human-readable recommendation."""
        best_model = comparison_df['overall_score'].idxmax()
        best_score = comparison_df['overall_score'].max()

        # Check if improvements are plateauing
        if trajectory and len(trajectory['overall_score']) >= 3:
            recent_improvements = np.diff(trajectory['overall_score'][-3:])
            if all(imp < 0.02 for imp in recent_improvements):
                return f"{best_model} shows best performance ({best_score:.2%}), though improvements are plateauing"

        return f"{best_model} shows best performance ({best_score:.2%})"


# deepfabric/evaluation/outputs/comparison_reporter.py

class ComparisonReporter:
    """Generate comparison reports (JSON and HTML)."""

    def generate_json(self, comparison: ModelComparison, output_path: str):
        """Generate JSON comparison report."""
        report = {
            "comparison_metadata": {
                "timestamp": datetime.now().isoformat(),
                "models_compared": len(comparison.models),
                "eval_dataset": "eval.jsonl"  # TODO: get from results
            },
            "models": {
                name: {
                    "overall_score": result.summary['overall_score'],
                    "tool_selection_accuracy": result.summary['tool_selection_accuracy'],
                    # Include all relevant metrics
                }
                for name, result in comparison.models.items()
            },
            "improvement_trajectory": comparison.improvement_trajectory,
            "best_model": comparison.best_model,
            "recommendation": comparison.recommendation
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

    def generate_html(self, comparison: ModelComparison, output_path: str):
        """Generate HTML comparison report with side-by-side charts."""
        # Similar to HTML report but focused on comparison
        # Include:
        # - Side-by-side metric comparison table
        # - Improvement trajectory line chart
        # - Per-metric bar charts
        # - Trade-off analysis (accuracy vs latency)
        pass
```

---

#### 4.2 Time-Series Tracking

**Files to Create:**
- `deepfabric/evaluation/tracking/history.py`

**Implementation:**

```python
# deepfabric/evaluation/tracking/history.py

@dataclass
class EvaluationHistoryEntry:
    """Single evaluation in history."""

    timestamp: str
    checkpoint: str
    overall_score: float
    metrics: Dict[str, float]
    training_steps: Optional[int] = None


class EvaluationHistory:
    """Track evaluation history over time."""

    def __init__(self, model_id: str, storage_dir: str = "./eval_history"):
        self.model_id = model_id
        self.storage_path = Path(storage_dir) / f"{model_id}_history.json"
        self.entries: List[EvaluationHistoryEntry] = []
        self._load()

    def add_evaluation(
        self,
        checkpoint: str,
        results: EvaluationResults,
        training_steps: Optional[int] = None
    ):
        """Add evaluation to history."""
        entry = EvaluationHistoryEntry(
            timestamp=datetime.now().isoformat(),
            checkpoint=checkpoint,
            overall_score=results.summary['overall_score'],
            metrics=results.summary,
            training_steps=training_steps
        )

        self.entries.append(entry)
        self._save()

    def get_trajectory(self, metric: str) -> List[Tuple[str, float]]:
        """Get metric values over time."""
        return [(e.timestamp, e.metrics[metric]) for e in self.entries]

    def plot_trends(self, output_path: str):
        """Generate trend charts."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot overall score over time
        timestamps = [e.timestamp for e in self.entries]
        overall_scores = [e.overall_score for e in self.entries]

        axes[0, 0].plot(range(len(timestamps)), overall_scores, marker='o')
        axes[0, 0].set_title('Overall Score Over Time')
        axes[0, 0].set_xlabel('Evaluation #')
        axes[0, 0].set_ylabel('Score')

        # Plot tool selection accuracy
        tool_scores = [e.metrics['tool_selection_accuracy'] for e in self.entries]
        axes[0, 1].plot(range(len(timestamps)), tool_scores, marker='o', color='orange')
        axes[0, 1].set_title('Tool Selection Accuracy Over Time')

        # More plots...

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    def _load(self):
        """Load history from disk."""
        if self.storage_path.exists():
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.entries = [EvaluationHistoryEntry(**e) for e in data['evaluations']]

    def _save(self):
        """Save history to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump({
                "model_id": self.model_id,
                "evaluations": [asdict(e) for e in self.entries]
            }, f, indent=2)
```

---

#### 4.3 CLI Commands

**Files to Modify:**
- `deepfabric/cli.py`

**Implementation:**

```python
# deepfabric evaluate-compare command

@cli.command("evaluate-compare")
@click.option("--models", multiple=True, required=True, help="Models: name:path")
@click.option("--eval-dataset", required=True, help="Evaluation dataset")
@click.option("--output", required=True, help="Output path for comparison")
@click.option("--format", type=click.Choice(["json", "html", "both"]), default="both")
def evaluate_compare(models, eval_dataset, output, format):
    """Compare multiple models side-by-side."""

    from deepfabric.evaluation.comparison import ModelComparator

    # Parse model specifications (name:path)
    model_specs = {}
    for spec in models:
        name, path = spec.split(":")
        model_specs[name] = path

    console.print(f"[bold]Comparing {len(model_specs)} models:[/bold]")
    for name, path in model_specs.items():
        console.print(f"  • {name}: {path}")

    # Run evaluation for each model
    results = {}
    for name, path in model_specs.items():
        console.print(f"\n[cyan]Evaluating {name}...[/cyan]")
        # Run evaluation (reuse evaluate command logic)
        results[name] = run_evaluation(path, eval_dataset)

    # Compare
    comparator = ModelComparator()
    comparison = comparator.compare(results, eval_dataset)

    # Generate outputs
    if format in ["json", "both"]:
        json_path = output if output.endswith('.json') else output + '.json'
        comparison_reporter.generate_json(comparison, json_path)
        console.print(f"✓ JSON comparison: {json_path}")

    if format in ["html", "both"]:
        html_path = output if output.endswith('.html') else output + '.html'
        comparison_reporter.generate_html(comparison, html_path)
        console.print(f"✓ HTML comparison: {html_path}")

    # Display summary
    console.print(f"\n[bold green]✓ Comparison Complete![/bold green]")
    console.print(f"  Best model: {comparison.best_model}")
    console.print(f"  Recommendation: {comparison.recommendation}")


# deepfabric eval-history command

@cli.command("eval-history")
@click.option("--model-id", required=True, help="Model identifier")
@click.option("--plot/--no-plot", default=False, help="Generate trend charts")
@click.option("--output", help="Output path for chart")
def eval_history(model_id, plot, output):
    """View evaluation history for a model."""

    from deepfabric.evaluation.tracking import EvaluationHistory

    history = EvaluationHistory(model_id)

    if not history.entries:
        console.print(f"[yellow]No evaluation history found for {model_id}[/yellow]")
        return

    # Display history table
    table = Table(title=f"Evaluation History: {model_id}")
    table.add_column("Date", style="cyan")
    table.add_column("Checkpoint", style="magenta")
    table.add_column("Overall Score", style="green")
    table.add_column("Tool Selection", style="blue")

    for entry in history.entries:
        table.add_row(
            entry.timestamp[:10],
            entry.checkpoint,
            f"{entry.overall_score:.2%}",
            f"{entry.metrics['tool_selection_accuracy']:.2%}"
        )

    console.print(table)

    # Plot trends if requested
    if plot:
        output_path = output or f"{model_id}_trends.png"
        history.plot_trends(output_path)
        console.print(f"\n✓ Trend chart saved: {output_path}")
```

---

#### 4.4 External Platform Integrations

**Files to Create:**
- `deepfabric/evaluation/integrations/mlflow_integration.py`
- `deepfabric/evaluation/integrations/wandb_integration.py`
- `deepfabric/evaluation/integrations/huggingface_integration.py`

**Implementation:**

```python
# deepfabric/evaluation/integrations/mlflow_integration.py

class MLflowIntegration:
    """Integrate evaluation with MLflow tracking."""

    def __init__(self, tracking_uri: Optional[str] = None):
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.mlflow = mlflow

    def log_evaluation(self, results: EvaluationResults, experiment_name: str):
        """Log evaluation results to MLflow."""
        with self.mlflow.start_run(run_name=results.metadata['model_name']):
            # Log parameters
            self.mlflow.log_param("model_path", results.metadata['model_path'])
            self.mlflow.log_param("eval_dataset", results.metadata['eval_dataset'])
            self.mlflow.log_param("dataset_size", results.metadata['eval_dataset_size'])

            # Log metrics
            for metric_name, value in results.summary.items():
                if isinstance(value, (int, float)):
                    self.mlflow.log_metric(metric_name, value)

            # Log detailed metrics
            for category, metrics in results.detailed_metrics.items():
                if isinstance(metrics, dict):
                    for metric_name, value in metrics.items():
                        if isinstance(value, (int, float)):
                            self.mlflow.log_metric(f"{category}.{metric_name}", value)

            # Log artifacts
            self.mlflow.log_artifact(results.json_path)
            self.mlflow.log_artifact(results.html_path)

            # Log confusion matrix as figure
            if 'confusion_matrix' in results.detailed_metrics.get('tool_selection', {}):
                fig = self._create_confusion_matrix_figure(
                    results.detailed_metrics['tool_selection']['confusion_matrix']
                )
                self.mlflow.log_figure(fig, "confusion_matrix.png")


# deepfabric/evaluation/integrations/wandb_integration.py

class WandBIntegration:
    """Integrate evaluation with Weights & Biases."""

    def __init__(self, project: str, entity: Optional[str] = None):
        import wandb
        self.wandb = wandb
        self.project = project
        self.entity = entity

    def log_evaluation(self, results: EvaluationResults):
        """Log evaluation to W&B."""
        run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            name=results.metadata['model_name'],
            config={
                "model_path": results.metadata['model_path'],
                "eval_dataset": results.metadata['eval_dataset'],
                "dataset_size": results.metadata['eval_dataset_size']
            }
        )

        # Log summary metrics
        self.wandb.summary.update(results.summary)

        # Log confusion matrix
        if 'confusion_matrix' in results.detailed_metrics.get('tool_selection', {}):
            cm = results.detailed_metrics['tool_selection']['confusion_matrix']
            self.wandb.log({"confusion_matrix": self.wandb.plot.confusion_matrix(
                probs=None,
                y_true=cm['y_true'],
                preds=cm['preds'],
                class_names=cm['labels']
            )})

        # Log per-example results as table
        table = self.wandb.Table(columns=[
            "query", "expected_tool", "predicted_tool", "correct", "param_accuracy"
        ])

        for i, (pred, truth) in enumerate(zip(results.predictions, results.ground_truth)):
            table.add_data(
                truth.query,
                truth.expected_tool,
                pred.tool_name or "NONE",
                pred.tool_name == truth.expected_tool,
                compute_param_accuracy(pred, truth)
            )

        self.wandb.log({"predictions": table})

        # Upload artifacts
        artifact = self.wandb.Artifact(f"eval_{results.metadata['model_name']}", type="evaluation")
        artifact.add_file(results.json_path)
        artifact.add_file(results.html_path)
        run.log_artifact(artifact)

        run.finish()


# deepfabric/evaluation/integrations/huggingface_integration.py

class HuggingFaceIntegration:
    """Integrate evaluation with HuggingFace Hub."""

    def __init__(self, token: Optional[str] = None):
        from huggingface_hub import HfApi
        self.api = HfApi(token=token)

    def push_evaluation(
        self,
        results: EvaluationResults,
        repo_id: str,
        update_model_card: bool = True
    ):
        """
        Upload evaluation results to HuggingFace Hub.

        Actions:
        1. Upload results.json to model repo
        2. Update model card with evaluation section
        3. Add tags for evaluation grade
        """
        # Upload results
        self.api.upload_file(
            path_or_fileobj=results.json_path,
            path_in_repo="evaluation_results.json",
            repo_id=repo_id,
            repo_type="model"
        )

        # Update model card
        if update_model_card:
            # Generate model card section
            model_card_section = generate_model_card_section(results)

            # Append to existing README
            try:
                readme = self.api.hf_hub_download(repo_id, "README.md")
                with open(readme, 'r') as f:
                    existing_content = f.read()

                # Append evaluation section
                updated_content = existing_content + "\n\n" + model_card_section

                with open(readme, 'w') as f:
                    f.write(updated_content)

                # Upload updated README
                self.api.upload_file(
                    path_or_fileobj=readme,
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    repo_type="model"
                )
            except Exception as e:
                console.print(f"[yellow]Warning: Could not update model card: {e}[/yellow]")

        # Add tags
        grade = results.summary['grade']
        tags = [f"deepfabric-eval", f"grade-{grade.split()[0].lower()}"]

        self.api.update_repo_settings(
            repo_id=repo_id,
            repo_type="model",
            tags=tags
        )
```

---

#### 4.5 Training Framework Callbacks

**Files to Create:**
- `deepfabric/evaluation/callbacks/trl_callback.py`

**Implementation:**

```python
# deepfabric/evaluation/callbacks/trl_callback.py

from transformers import TrainerCallback

class DeepFabricEvaluationCallback(TrainerCallback):
    """
    TRL/HuggingFace Trainer callback for periodic evaluation.

    Usage:
        evaluator = DeepFabricEvaluationCallback(
            eval_dataset="eval.jsonl",
            eval_steps=100,
            output_dir="./eval_checkpoints"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            callbacks=[evaluator]
        )
    """

    def __init__(
        self,
        eval_dataset: str,
        eval_steps: int = 100,
        output_dir: str = "./eval_checkpoints",
        config: Optional[EvaluationConfig] = None
    ):
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps
        self.output_dir = output_dir
        self.config = config or EvaluationConfig()
        self.evaluation_history = []

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Run evaluation at specified intervals."""
        if state.global_step % self.eval_steps == 0:
            console.print(f"\n[cyan]Running evaluation at step {state.global_step}...[/cyan]")

            # Save checkpoint
            checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
            model.save_pretrained(checkpoint_dir)

            # Run evaluation
            engine = EvaluationEngine(self.config)
            results = engine.evaluate(
                model_path=checkpoint_dir,
                eval_dataset=self.eval_dataset,
                output_dir=f"{self.output_dir}/eval-{state.global_step}"
            )

            # Store in history
            self.evaluation_history.append({
                "step": state.global_step,
                "overall_score": results.summary['overall_score'],
                "tool_selection_accuracy": results.summary['tool_selection_accuracy']
            })

            # Display summary
            console.print(f"  Overall Score: {results.summary['overall_score']:.2%}")
            console.print(f"  Tool Selection: {results.summary['tool_selection_accuracy']:.2%}\n")

    def on_train_end(self, args, state, control, **kwargs):
        """Generate evaluation trend chart at end of training."""
        if len(self.evaluation_history) > 1:
            self._plot_evaluation_trends()

    def _plot_evaluation_trends(self):
        """Plot evaluation metrics over training."""
        import matplotlib.pyplot as plt

        steps = [e['step'] for e in self.evaluation_history]
        scores = [e['overall_score'] for e in self.evaluation_history]

        plt.figure(figsize=(10, 6))
        plt.plot(steps, scores, marker='o')
        plt.xlabel('Training Step')
        plt.ylabel('Overall Score')
        plt.title('Evaluation Score During Training')
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/evaluation_trend.png", dpi=150)
        plt.close()

        console.print(f"✓ Evaluation trend chart: {self.output_dir}/evaluation_trend.png")
```

---

### Phase 4 Acceptance Criteria

- [ ] Multi-model comparison works correctly
- [ ] Comparison reports are insightful
- [ ] Time-series tracking persists across runs
- [ ] Trend charts visualize improvements
- [ ] MLflow integration logs all metrics and artifacts
- [ ] W&B integration creates rich dashboards
- [ ] HuggingFace Hub integration updates model cards
- [ ] TRL callback evaluates during training
- [ ] All CLI commands work end-to-end
- [ ] 80%+ test coverage for Phase 4 components

### Phase 4 Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| External platform APIs change | Medium | Version pin dependencies, add integration tests |
| Callback slows training significantly | High | Make evaluation async, reduce frequency |
| Time-series storage grows large | Low | Implement retention policies, compression |

---

## 📈 Success Metrics

### Phase 1-2 (MVP)
- [ ] 100 GitHub stars within 1 month of launch
- [ ] 50+ active users (tracked via telemetry opt-in)
- [ ] <5 critical bugs reported
- [ ] Positive feedback on HackerNews/Reddit

### Phase 3-4 (Full Feature Set)
- [ ] 500+ GitHub stars
- [ ] 200+ active users
- [ ] 10+ blog posts/tutorials from community
- [ ] 90%+ user satisfaction (survey)
- [ ] Featured in HuggingFace Newsletter

### Code Quality
- [ ] 85%+ test coverage overall
- [ ] All tests passing in CI
- [ ] No critical security vulnerabilities (Bandit)
- [ ] Ruff linting passes with no errors
- [ ] Type hints coverage >80% (mypy)

---

## 🧪 Testing Strategy

### Unit Tests
- **Target Coverage:** 90%+
- **Focus Areas:**
  - Metric calculations (exact values)
  - Dataset splitting (stratification correctness)
  - Ground truth parsing (all conversation types)
  - Response parsing (various formats)

### Integration Tests
- **Scope:**
  - End-to-end evaluation pipeline
  - CLI commands with real models
  - Output generation (JSON, HTML, CSV)
  - External integrations (MLflow, W&B)

### Fixtures
Create test fixtures for:
- Sample datasets (all conversation types)
- Mock model outputs (various formats)
- Expected metric values
- Sample evaluation configs

### Continuous Integration
```yaml
# .github/workflows/test-evaluation.yml
name: Test Evaluation System

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: uv sync --all-extras
      - name: Run evaluation tests
        run: pytest tests/evaluation/ -v --cov=deepfabric/evaluation
      - name: Check coverage
        run: pytest --cov=deepfabric/evaluation --cov-fail-under=85
```

---

## 📚 Documentation Requirements

### User Documentation
1. **Quickstart Guide**
   - 5-minute tutorial
   - Simple example with dataset splitting and evaluation

2. **CLI Reference**
   - Detailed docs for all commands
   - Examples for common use cases

3. **Configuration Guide**
   - YAML schema documentation
   - All options explained

4. **Integration Guides**
   - MLflow integration
   - W&B integration
   - HuggingFace Hub integration
   - CI/CD pipeline examples

### Developer Documentation
1. **Architecture Overview**
   - Component diagram
   - Data flow explanation

2. **Adding Custom Metrics**
   - Tutorial for extending with custom metrics

3. **API Reference**
   - Auto-generated from docstrings (Sphinx)

---

## 🚀 Release Plan

### v0.4.0 - Phase 1 (MVP Foundation)
- Dataset splitting
- Basic configuration
- Ground truth parsing

### v0.5.0 - Phase 2 (Core Evaluation)
- Evaluation engine
- Basic metrics
- JSON output
- `deepfabric evaluate` command

### v0.6.0 - Phase 3 (Rich Reporting)
- Advanced metrics
- HTML reports with charts
- CSV export
- Model card generation
- Failure analysis

### v0.7.0 - Phase 4 (Integrations & Comparison)
- Multi-model comparison
- Time-series tracking
- MLflow integration
- W&B integration
- HuggingFace Hub integration
- Training framework callbacks

---

## 🎯 Post-Phase 4: Future Enhancements

### Phase 5 (Optional - 6-8 weeks)
- LLM-as-judge for response quality
- Active learning loop (identify hard examples → generate more data)
- Multi-task evaluation
- Interpretability analysis (attention viz)
- Human evaluation integration

**Note:** Cloud SaaS platform is tracked in separate document: `CLOUD_SAAS_DESIGN.md`

---

## 📞 Team & Resources

### Recommended Team
- **Phase 1-2:** 2 engineers (1 senior, 1 mid-level)
- **Phase 3:** 2 engineers
- **Phase 4:** 2 engineers + 0.5 DevOps for integrations

### Skills Required
- Strong Python (type hints, dataclasses, async)
- ML evaluation expertise
- Data visualization (matplotlib, seaborn)
- CLI development (Click, Rich)
- Testing (pytest)
- Documentation (Sphinx, markdown)

### Timeline Assumptions
- Engineers work full-time on evaluation system
- Minimal context switching
- Access to GPUs for testing
- Stakeholder reviews at end of each phase

---

## 📊 Progress Tracking

### Current Status
- **Phase 1:** ⏸️ Not Started (0%)
- **Phase 2:** ⏸️ Not Started (0%)
- **Phase 3:** ⏸️ Not Started (0%)
- **Phase 4:** ⏸️ Not Started (0%)

### Weekly Updates
Document weekly progress in this section:

#### Week 1 (2025-11-04)
- Status: Planning complete
- Next: Begin Phase 1 implementation

---

## ✅ Review & Sign-off

- [ ] Technical design reviewed by team
- [ ] Architecture approved by lead
- [ ] Timeline approved by stakeholders
- [ ] Resources allocated
- [ ] Ready to begin implementation

---

**Document Version:** 1.0
**Last Updated:** 2025-10-31
**Owner:** Engineering Team
**Reviewers:** TBD
