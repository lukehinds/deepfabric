"""Evaluation module for DeepFabric.

This module provides functionality to evaluate fine-tuned models on tool-calling tasks.
"""

from .evaluator import EvaluationResult, Evaluator, EvaluatorConfig
from .evaluators import (
    BaseEvaluator,
    EvaluationContext,
    EvaluatorRegistry,
    EvaluatorResult,
    ToolCallingEvaluator,
)
from .inference import InferenceConfig, ModelResponse, create_inference_backend
from .metrics import EvaluationMetrics, SampleEvaluation, compute_metrics
from .parser import GroundTruth, GroundTruthParser, parse_batch
from .reporters import BaseReporter, CloudReporter, FileReporter, MultiReporter
from .split import SplitConfig, SplitResult, split_dataset, split_to_hf_dataset

__all__ = [
    # Splitting
    "split_dataset",
    "split_to_hf_dataset",
    "SplitConfig",
    "SplitResult",
    # Parsing
    "GroundTruth",
    "GroundTruthParser",
    "parse_batch",
    # Inference
    "InferenceConfig",
    "ModelResponse",
    "create_inference_backend",
    # Metrics
    "EvaluationMetrics",
    "SampleEvaluation",
    "compute_metrics",
    # Evaluator
    "Evaluator",
    "EvaluatorConfig",
    "EvaluationResult",
    # Evaluators
    "BaseEvaluator",
    "EvaluationContext",
    "EvaluatorRegistry",
    "EvaluatorResult",
    "ToolCallingEvaluator",
    # Reporters
    "BaseReporter",
    "FileReporter",
    "CloudReporter",
    "MultiReporter",
]
