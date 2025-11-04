"""Metrics computation for model evaluation."""

from typing import Any

from pydantic import BaseModel, Field

# Tolerance for numeric comparison
NUMERIC_TOLERANCE = 1e-6


class EvaluationMetrics(BaseModel):
    """Computed evaluation metrics."""

    tool_selection_accuracy: float = Field(
        description="Accuracy of tool selection (0.0-1.0)",
    )
    parameter_accuracy: float = Field(
        description="Accuracy of parameter extraction (0.0-1.0)",
    )
    execution_success_rate: float = Field(
        description="Rate of valid tool calls (0.0-1.0)",
    )
    response_quality: float = Field(
        description="Quality of final response (0.0-1.0)",
    )
    overall_score: float = Field(
        description="Weighted overall score (0.0-1.0)",
    )
    samples_evaluated: int = Field(
        description="Total number of samples evaluated",
    )
    samples_processed: int = Field(
        description="Number of samples processed without system errors",
    )
    processing_errors: int = Field(
        description="Number of samples that failed to process (system errors, timeouts)",
    )


class SampleEvaluation(BaseModel):
    """Evaluation result for a single sample."""

    sample_id: int = Field(description="Sample index")
    query: str = Field(description="Input query")
    expected_tool: str | None = Field(
        default=None,
        description="Expected tool name",
    )
    predicted_tool: str | None = Field(
        default=None,
        description="Predicted tool name",
    )
    expected_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Expected parameters",
    )
    predicted_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Predicted parameters",
    )
    expected_answer: str | None = Field(
        default=None,
        description="Expected final answer",
    )
    predicted_answer: str | None = Field(
        default=None,
        description="Predicted final answer",
    )
    tool_selection_correct: bool = Field(
        description="Whether tool selection was correct",
    )
    parameters_correct: bool = Field(
        description="Whether parameters were correct",
    )
    execution_valid: bool = Field(
        description="Whether the tool call could be executed",
    )
    response_score: float = Field(
        description="Response quality score (0.0-1.0)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if prediction failed",
    )


def compute_tool_selection_accuracy(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute tool selection accuracy.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Accuracy score (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    correct = sum(1 for e in evaluations if e.tool_selection_correct)
    return correct / len(evaluations)


def compute_parameter_accuracy(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute parameter extraction accuracy.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Accuracy score (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    correct = sum(1 for e in evaluations if e.parameters_correct)
    return correct / len(evaluations)


def compute_execution_success_rate(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute execution success rate.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Success rate (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    valid = sum(1 for e in evaluations if e.execution_valid)
    return valid / len(evaluations)


def compute_response_quality(
    evaluations: list[SampleEvaluation],
) -> float:
    """Compute average response quality.

    Args:
        evaluations: List of sample evaluations

    Returns:
        Average quality score (0.0-1.0)
    """
    if not evaluations:
        return 0.0

    total_score = sum(e.response_score for e in evaluations)
    return total_score / len(evaluations)


def compute_overall_score(
    tool_accuracy: float,
    param_accuracy: float,
    exec_success: float,
    response_quality: float,
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted overall score.

    Args:
        tool_accuracy: Tool selection accuracy
        param_accuracy: Parameter accuracy
        exec_success: Execution success rate
        response_quality: Response quality score
        weights: Custom weights for each metric (defaults used if None)

    Returns:
        Weighted overall score (0.0-1.0)
    """
    # Default weights (response_quality excluded for tool-calling mode)
    if weights is None:
        weights = {
            "tool_selection": 0.40,
            "parameter_accuracy": 0.35,
            "execution_success": 0.25,
            "response_quality": 0.00,  # Not used for tool-calling evaluation
        }

    return (
        tool_accuracy * weights.get("tool_selection", 0.0)
        + param_accuracy * weights.get("parameter_accuracy", 0.0)
        + exec_success * weights.get("execution_success", 0.0)
        + response_quality * weights.get("response_quality", 0.0)
    )


def compute_metrics(
    evaluations: list[SampleEvaluation],
    weights: dict[str, float] | None = None,
) -> EvaluationMetrics:
    """Compute all evaluation metrics from sample evaluations.

    Args:
        evaluations: List of sample evaluations
        weights: Custom weights for overall score computation

    Returns:
        EvaluationMetrics with all computed scores
    """
    if not evaluations:
        return EvaluationMetrics(
            tool_selection_accuracy=0.0,
            parameter_accuracy=0.0,
            execution_success_rate=0.0,
            response_quality=0.0,
            overall_score=0.0,
            samples_evaluated=0,
            samples_processed=0,
            processing_errors=0,
        )

    tool_acc = compute_tool_selection_accuracy(evaluations)
    param_acc = compute_parameter_accuracy(evaluations)
    exec_success = compute_execution_success_rate(evaluations)
    resp_quality = compute_response_quality(evaluations)

    overall = compute_overall_score(
        tool_acc,
        param_acc,
        exec_success,
        resp_quality,
        weights,
    )

    # Count processing status (system errors vs successfully processed)
    processed = sum(1 for e in evaluations if e.error is None)
    errors = len(evaluations) - processed

    return EvaluationMetrics(
        tool_selection_accuracy=tool_acc,
        parameter_accuracy=param_acc,
        execution_success_rate=exec_success,
        response_quality=resp_quality,
        overall_score=overall,
        samples_evaluated=len(evaluations),
        samples_processed=processed,
        processing_errors=errors,
    )


def compare_parameters(
    expected: dict[str, Any],
    predicted: dict[str, Any],
) -> bool:
    """Compare expected and predicted parameters.

    Performs fuzzy matching for string values (case-insensitive).

    Args:
        expected: Expected parameters
        predicted: Predicted parameters

    Returns:
        True if parameters match, False otherwise
    """
    if not expected and not predicted:
        return True

    # Check if all expected keys are present
    if set(expected.keys()) != set(predicted.keys()):
        return False

    # Compare values
    for key, expected_val in expected.items():
        predicted_val = predicted.get(key)

        # Handle different types
        if isinstance(expected_val, str) and isinstance(predicted_val, str):
            # Case-insensitive string comparison
            if expected_val.lower().strip() != predicted_val.lower().strip():
                return False
        elif isinstance(expected_val, int | float) and isinstance(predicted_val, int | float):
            # Numeric comparison with small tolerance
            if abs(float(expected_val) - float(predicted_val)) > NUMERIC_TOLERANCE:
                return False
        elif expected_val != predicted_val:
            # Exact match for other types
            return False

    return True


def compute_response_similarity(
    expected: str | None,
    predicted: str | None,
) -> float:
    """Compute similarity between expected and predicted responses.

    Uses simple word overlap for now. Can be enhanced with semantic similarity.

    Args:
        expected: Expected response
        predicted: Predicted response

    Returns:
        Similarity score (0.0-1.0)
    """
    if not expected or not predicted:
        return 0.0 if expected != predicted else 1.0

    # Tokenize and normalize
    expected_words = set(expected.lower().split())
    predicted_words = set(predicted.lower().split())

    # Compute Jaccard similarity
    if not expected_words and not predicted_words:
        return 1.0

    intersection = expected_words & predicted_words
    union = expected_words | predicted_words

    return len(intersection) / len(union) if union else 0.0
