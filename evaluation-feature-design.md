# DeepFabric Evaluation Feature Design

## Executive Summary

This document outlines a comprehensive evaluation feature for DeepFabric that enables users to measure and track the effectiveness of fine-tuned models on tool-calling tasks. The evaluation system is designed to integrate seamlessly into ML pipelines, provide actionable metrics, and create valuable artifacts for model comparison and improvement tracking.

---

## Table of Contents

1. [Overview & Goals](#overview--goals)
2. [User Experience Design](#user-experience-design)
3. [ML Pipeline Integration](#ml-pipeline-integration)
4. [Evaluation Metrics & Scoring](#evaluation-metrics--scoring)
5. [Output Artifacts](#output-artifacts)
6. [Storage & Tracking](#storage--tracking)
7. [CLI & Configuration](#cli--configuration)
8. [Implementation Architecture](#implementation-architecture)
9. [Use Cases & Workflows](#use-cases--workflows)
10. [DeepFabric Cloud: SaaS Platform](#deepfabric-cloud-saas-platform)
11. [Future Enhancements](#future-enhancements)

---

## Overview & Goals

### Purpose

The evaluation feature enables users to:
1. **Validate fine-tuned models** against held-out synthetic data
2. **Measure tool-calling accuracy** (correct tool selection, parameter accuracy, execution success)
3. **Track improvements** across training iterations
4. **Compare models** side-by-side using standardized metrics
5. **Generate reports** for model cards, papers, and stakeholder communication
6. **Upload results** to experiment tracking platform (DeepFabric SaaS, MLflow, W&B, HuggingFace Hub)

### Core Principle

**"Evaluate the same way you train"** - Use the synthetic dataset to both fine-tune AND evaluate, ensuring consistency between training distribution and evaluation distribution, use dataset split functionality to create train/eval sets from the same data generation process.

---

## User Experience Design

### Philosophy

The evaluation UX is designed around three core principles:

1. **Zero friction**: Evaluation should be as simple as running an `evaluate` command after training
2. **Rich insights**: Generate comprehensive metrics without requiring ML expertise
3. **Pipeline-first**: Every output is designed for programmatic consumption and CI/CD integration

### Primary User Flows

#### Flow 1: Quick Evaluation (Interactive)

```bash
# Step 1: Generate your dataset normally
deepfabric generate config.yaml

# Step 2: Split existing dataset into train/eval sets
# This can use the hugginface-style stratified split `dataset.train_test_split`
deepfabric split dataset.jsonl \
  --train-output dataset_train.jsonl \
  --eval-output dataset_eval.jsonl \
  --test-size 0.2 \
  --stratify-by topic

# Step 3: Format training data for your framework (if needed)
deepfabric format dataset_train.jsonl \
  --formatter trl_sft_tools \
  --output dataset_train_formatted.jsonl

# Step 4: Fine-tune your model (outside DeepFabric)
# ... your training code here ...

# Step 5: Evaluate the fine-tuned model
deepfabric evaluate \
  --model path/to/fine-tuned-model \
  --eval-dataset dataset_eval.jsonl \
  --conversation-type chain_of_thought \
  --output-dir ./eval_results
```

**User sees:**
- Real-time progress bar during evaluation
- Summary table with key metrics
- Link to detailed HTML report
- Pass/fail status with color coding
- Request for API key to upload results to DeepFabric Cloud (optional)

#### Flow 2: Pipeline Evaluation (Automated)

```bash
# In a training script or CI/CD pipeline
deepfabric evaluate \
  --model path/to/model \
  --eval-dataset eval.jsonl \
  --format json \
  --output eval_results.json \
  --threshold tool_selection_accuracy=0.85

# Exit code 0 if thresholds met, 1 otherwise
```

#### Flow 3: Comparative Evaluation

```bash
# Compare multiple model checkpoints
deepfabric evaluate-compare \
  --models checkpoint-100:./checkpoint-100 checkpoint-500:./checkpoint-500 final:./final-model \
  --eval-dataset eval.jsonl \
  --output comparison_report.html
```

---

## ML Pipeline Integration

### Integration Points

The evaluation feature integrates at three key stages:

```
┌─────────────────┐
│ 1. Dataset Gen  │
│  with Holdout   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 2. Fine-Tuning  │
│  (TRL/Unsloth)  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ 3. Evaluation   │
│  & Reporting    │
└─────────────────┘
```

### Full Pipeline Example

```python
# complete_pipeline.py - End-to-end training with evaluation

from deepfabric import DeepFabricConfig, Dataset
from deepfabric.evaluation import split_dataset, evaluate_model
from trl import SFTTrainer
from transformers import AutoModelForCausalLM

# Step 1: Generate dataset
config = DeepFabricConfig.from_yaml("agent_config.yaml")
dataset = generate_dataset(config)
dataset.save("dataset_full.jsonl")

# Step 2: Split dataset for training and evaluation
train_dataset, eval_dataset = split_dataset(
    "dataset_full.jsonl",
    test_size=0.2,
    stratify_by="topic",  # Ensure balanced topic distribution
    conversation_type=config.data_engine.conversation_type
)

train_dataset.save("train.jsonl")
eval_dataset.save("eval.jsonl")

# Step 3: Format training data for TRL (eval data stays in original format)
from deepfabric.formatters import format_dataset
formatted_train = format_dataset(
    "train.jsonl",
    formatter="trl_sft_tools",
    config=config
)
formatted_train.save("train_formatted.jsonl")

# Step 4: Fine-tune model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_train,  # Use formatted data
)
trainer.train()
trainer.save_model("./fine-tuned-model")

# Step 5: Evaluate (uses original eval.jsonl, not formatted)
eval_results = evaluate_model(
    model_path="./fine-tuned-model",
    eval_dataset="eval.jsonl",  # Original format with ground truth
    conversation_type=config.data_engine.conversation_type,
    agent_mode=config.data_engine.agent_mode,
    metrics=["tool_selection", "parameter_accuracy", "execution_success"],
    output_dir="./eval_results"
)

print(f"Tool Selection Accuracy: {eval_results.tool_selection_accuracy:.2%}")
print(f"Overall Score: {eval_results.overall_score:.2%}")

# Step 6: Generate model card with evaluation results
eval_results.generate_model_card("./model_card.md")
```

### Integration with Training Frameworks

#### TRL (HuggingFace)

```python
from trl import SFTTrainer
from deepfabric.evaluation import DeepFabricEvaluator

# Add evaluation callback during training
evaluator = DeepFabricEvaluator(
    eval_dataset="eval.jsonl",
    eval_steps=100,  # Evaluate every 100 steps
    output_dir="./eval_checkpoints"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    callbacks=[evaluator]
)

trainer.train()  # Automatically evaluates at intervals
```

#### Axolotl

```yaml
# axolotl_config.yaml
base_model: meta-llama/Llama-3.2-1B
datasets:
  - path: train.jsonl
    type: deepfabric

# DeepFabric evaluation configuration
deepfabric_eval:
  enabled: true
  dataset: eval.jsonl
  metrics:
    - tool_selection_accuracy
    - parameter_f1_score
    - execution_success_rate
  eval_steps: 100
  output_dir: ./eval_results
```

### CI/CD Integration

#### GitHub Actions Example

```yaml
# .github/workflows/train-and-eval.yml
name: Train and Evaluate Model

on:
  push:
    branches: [main]

jobs:
  train-eval:
    runs-on: gpu-runner
    steps:
      - uses: actions/checkout@v3

      - name: Generate Dataset
        run: |
          deepfabric generate config.yaml --holdout 0.2

      - name: Fine-tune Model
        run: |
          python train.py --output ./model

      - name: Evaluate Model
        run: |
          deepfabric evaluate \
            --model ./model \
            --eval-dataset dataset_eval.jsonl \
            --output eval_results.json \
            --threshold tool_selection_accuracy=0.85

      - name: Upload Evaluation Report
        uses: actions/upload-artifact@v3
        with:
          name: evaluation-report
          path: eval_results.json

      - name: Comment PR with Results
        uses: actions/github-script@v6
        with:
          script: |
            const results = require('./eval_results.json');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              body: `## Evaluation Results\n\n${results.summary}`
            });
```

---

## Evaluation Metrics & Scoring

### Metric Categories

#### 1. Tool Selection Metrics

**Tool Selection Accuracy**
- **Definition**: Percentage of queries where the model selected the correct tool(s)
- **Calculation**: `correct_tool_selections / total_queries`
- **Critical for**: Understanding if the model understands which tool to use

**Tool Selection Precision/Recall/F1**
- **Definition**: For multi-tool queries, measure partial correctness
- **Use case**: When queries may require multiple tools

```python
{
    "tool_selection_accuracy": 0.87,  # 87% perfect matches
    "tool_selection_precision": 0.92,  # 92% of selected tools were correct
    "tool_selection_recall": 0.85,     # 85% of required tools were selected
    "tool_selection_f1": 0.88
}
```

#### 2. Parameter Accuracy Metrics

**Exact Parameter Match**
- **Definition**: Percentage where all parameters exactly match expected values
- **Calculation**: `exact_parameter_matches / tool_calls`

**Parameter Field Accuracy**
- **Definition**: Per-field accuracy for tool parameters
- **Example**:
```python
{
    "parameter_accuracy": {
        "get_weather": {
            "location": 0.95,      # 95% correct location parameters
            "units": 0.88,         # 88% correct unit specifications
            "overall": 0.91
        },
        "calculate": {
            "expression": 0.82,
            "precision": 0.78,
            "overall": 0.80
        }
    }
}
```

**Parameter Type Correctness**
- **Definition**: Are parameter types correct (string, int, float, etc.)?
- **Critical for**: Type-strict APIs and function calling

#### 3. Execution Success Metrics

**Successful Execution Rate**
- **Definition**: Percentage of tool calls that would execute without errors
- **Validation**: Type checking, required field presence, value constraints

**Error Category Distribution**
```python
{
    "execution_success_rate": 0.83,
    "error_distribution": {
        "missing_required_param": 0.08,
        "invalid_param_type": 0.05,
        "invalid_param_value": 0.03,
        "malformed_json": 0.01
    }
}
```

#### 4. Response Quality Metrics

**Answer Correctness** (for queries with ground truth)
- **Definition**: BLEU/ROUGE score or exact match against expected answer
- **Use case**: When evaluating end-to-end task completion

**Reasoning Quality** (for chain-of-thought)
- **Definition**: Structured evaluation of reasoning steps
- **Metrics**: Reasoning coherence, step completeness, logical flow

#### 5. Latency & Efficiency Metrics

**Average Inference Time**
- **Definition**: Mean time from query to response
- **Critical for**: Production deployment decisions

**Tool Call Efficiency**
- **Definition**: Average number of tool calls per query
- **Optimal**: Minimum calls needed to accomplish task

### Overall Scoring System

**Composite Score Calculation**

```python
overall_score = (
    0.40 * tool_selection_accuracy +
    0.30 * parameter_accuracy +
    0.20 * execution_success_rate +
    0.10 * response_quality
)

# Configurable weights in YAML:
# evaluation:
#   weights:
#     tool_selection: 0.40
#     parameter_accuracy: 0.30
#     execution_success: 0.20
#     response_quality: 0.10
```

**Grade Classification**

```python
def get_grade(overall_score):
    if overall_score >= 0.95: return "A+ (Production Ready)"
    elif overall_score >= 0.90: return "A (Excellent)"
    elif overall_score >= 0.85: return "B+ (Good)"
    elif overall_score >= 0.80: return "B (Acceptable)"
    elif overall_score >= 0.70: return "C (Needs Improvement)"
    else: return "D (Not Ready)"
```

---

## Output Artifacts

### 1. JSON Results File

**Primary programmatic output** for CI/CD and automation.

```json
{
  "evaluation_metadata": {
    "timestamp": "2025-10-27T10:30:00Z",
    "model_name": "my-fine-tuned-llama-1b",
    "model_path": "./fine-tuned-model",
    "eval_dataset": "eval.jsonl",
    "eval_dataset_size": 200,
    "deepfabric_version": "0.4.0",
    "evaluation_duration_seconds": 145.3
  },
  "summary": {
    "overall_score": 0.87,
    "grade": "B+ (Good)",
    "tool_selection_accuracy": 0.89,
    "parameter_accuracy": 0.85,
    "execution_success_rate": 0.91,
    "response_quality": 0.82
  },
  "detailed_metrics": {
    "tool_selection": {
      "accuracy": 0.89,
      "precision": 0.92,
      "recall": 0.87,
      "f1": 0.89,
      "confusion_matrix": {
        "get_weather": {"get_weather": 45, "search_web": 2, "none": 1},
        "calculate": {"calculate": 38, "search_web": 1, "none": 0},
        "search_web": {"search_web": 40, "get_weather": 1, "none": 2}
      }
    },
    "parameter_accuracy": {
      "exact_match_rate": 0.78,
      "field_accuracy": 0.85,
      "type_correctness": 0.94,
      "per_tool_accuracy": {
        "get_weather": 0.88,
        "calculate": 0.82,
        "search_web": 0.85
      }
    },
    "execution_metrics": {
      "success_rate": 0.91,
      "error_distribution": {
        "missing_required_param": 0.04,
        "invalid_param_type": 0.03,
        "invalid_param_value": 0.02
      }
    },
    "response_quality": {
      "answer_correctness": 0.82,
      "avg_response_length": 156,
      "reasoning_coherence": 0.85
    },
    "efficiency": {
      "avg_inference_time_ms": 1247,
      "avg_tool_calls_per_query": 1.2,
      "total_tokens_generated": 31240
    }
  },
  "per_topic_breakdown": {
    "weather_queries": {
      "count": 45,
      "tool_selection_accuracy": 0.93,
      "parameter_accuracy": 0.88,
      "overall_score": 0.90
    },
    "calculations": {
      "count": 40,
      "tool_selection_accuracy": 0.85,
      "parameter_accuracy": 0.82,
      "overall_score": 0.84
    }
  },
  "failure_analysis": {
    "most_common_errors": [
      {
        "error_type": "wrong_tool_selected",
        "count": 12,
        "example_query": "What's 25% of 180?",
        "expected_tool": "calculate",
        "actual_tool": "search_web"
      },
      {
        "error_type": "missing_parameter",
        "count": 8,
        "example": "get_weather called without 'units' parameter"
      }
    ],
    "hardest_examples": [
      {
        "query": "Compare weather in Paris and Tokyo",
        "expected": "get_weather (multiple calls)",
        "actual": "get_weather (single call)",
        "error": "Did not handle multiple locations"
      }
    ]
  },
  "thresholds": {
    "passed": true,
    "checks": [
      {
        "metric": "tool_selection_accuracy",
        "threshold": 0.85,
        "actual": 0.89,
        "passed": true
      },
      {
        "metric": "execution_success_rate",
        "threshold": 0.90,
        "actual": 0.91,
        "passed": true
      }
    ]
  }
}
```

### 2. HTML Report

**Human-readable comprehensive report** with visualizations.

**Features:**
- Executive summary with grade and pass/fail status
- Interactive charts (metric trends, confusion matrix, error distribution)
- Side-by-side comparison of expected vs actual tool calls
- Filterable table of all evaluation examples
- Exportable as PDF for stakeholder presentations

**Report Sections:**
1. **Overview**: Summary metrics, grade, pass/fail
2. **Tool Selection Analysis**: Accuracy, confusion matrix, error patterns
3. **Parameter Analysis**: Field-level accuracy, type errors, examples
4. **Execution Analysis**: Success rate, error categories
5. **Topic Breakdown**: Per-topic performance heatmap
6. **Failure Analysis**: Hardest examples, common mistakes, recommendations
7. **Recommendations**: Actionable suggestions for improvement

### 3. Model Card Enhancement

**Auto-generated model card section** for HuggingFace Hub or documentation.

```markdown
## Evaluation Results

**Model**: `my-fine-tuned-llama-1b`
**Evaluated on**: 200 synthetic tool-calling examples
**Overall Score**: 87/100 (B+)
**Date**: 2025-10-27

### Performance Metrics

| Metric | Score |
|--------|-------|
| Tool Selection Accuracy | 89% |
| Parameter Accuracy | 85% |
| Execution Success Rate | 91% |
| Response Quality | 82% |

### Strengths
- Excellent at weather queries (93% accuracy)
- Strong parameter type correctness (94%)
- Fast inference (avg 1.2s)

### Areas for Improvement
- Calculations accuracy could be higher (85%)
- Occasional confusion between search_web and calculate
- Missing optional parameters in 4% of calls

### Recommended Use Cases
✅ Production-ready for weather and time queries
✅ Suitable for simple calculations
⚠️ May need fallback logic for complex multi-tool scenarios
```

### 4. CSV Export

**Detailed per-example results** for further analysis.

```csv
example_id,query,expected_tool,actual_tool,tool_correct,param_accuracy,executable,response_quality,topic
1,"Weather in Paris?",get_weather,get_weather,true,1.0,true,0.92,weather_queries
2,"What's 10+5?",calculate,calculate,true,0.85,true,0.88,calculations
3,"News about AI",search_web,search_web,true,1.0,true,0.78,web_search
...
```

### 5. Comparison Report (Multi-Model)

When comparing multiple models:

```json
{
  "comparison_metadata": {
    "timestamp": "2025-10-27T11:00:00Z",
    "models_compared": 3,
    "eval_dataset": "eval.jsonl"
  },
  "models": {
    "checkpoint-100": {
      "overall_score": 0.78,
      "tool_selection_accuracy": 0.80,
      "parameter_accuracy": 0.76,
      "training_steps": 100
    },
    "checkpoint-500": {
      "overall_score": 0.85,
      "tool_selection_accuracy": 0.88,
      "parameter_accuracy": 0.83,
      "training_steps": 500
    },
    "final-model": {
      "overall_score": 0.87,
      "tool_selection_accuracy": 0.89,
      "parameter_accuracy": 0.85,
      "training_steps": 1000
    }
  },
  "improvement_trajectory": {
    "tool_selection_accuracy": [0.80, 0.88, 0.89],
    "overall_score": [0.78, 0.85, 0.87]
  },
  "best_model": "final-model",
  "recommendation": "final-model shows best performance, though improvements plateau after 500 steps"
}
```

---

## Storage & Tracking

### Local Storage Structure

```
./eval_results/
├── run_2025-10-27_10-30-00/
│   ├── config.yaml                  # Evaluation configuration
│   ├── results.json                 # Primary results file
│   ├── report.html                  # Human-readable report
│   ├── model_card_section.md        # Model card enhancement
│   ├── detailed_results.csv         # Per-example results
│   ├── failures.jsonl               # Failed examples for analysis
│   └── charts/
│       ├── confusion_matrix.png
│       ├── metric_breakdown.png
│       └── topic_heatmap.png
├── run_2025-10-27_11-00-00/
└── comparison_2025-10-27_12-00-00/
    ├── comparison.json
    ├── comparison_report.html
    └── trend_charts/
```


### API Integration

Integrate with a DeepFabric SaaS platform for centralized tracking, dataset & model registry, and collaboration.

### Experiment Tracking Integration

Support for popular tracking platforms out-of-the-box.

#### MLflow

```python
import mlflow
from deepfabric.evaluation import evaluate_model

# Automatic MLflow logging
with mlflow.start_run():
    mlflow.log_param("model_name", "llama-1b-finetuned")
    mlflow.log_param("eval_dataset", "eval.jsonl")

    eval_results = evaluate_model(
        model_path="./model",
        eval_dataset="eval.jsonl",
        mlflow_tracking=True  # Auto-log all metrics
    )

    # All metrics automatically logged:
    # - mlflow.log_metric("tool_selection_accuracy", 0.89)
    # - mlflow.log_metric("overall_score", 0.87)
    # - mlflow.log_artifact("report.html")
```

#### Weights & Biases

```python
import wandb
from deepfabric.evaluation import evaluate_model

wandb.init(project="model-evaluation")

eval_results = evaluate_model(
    model_path="./model",
    eval_dataset="eval.jsonl",
    wandb_tracking=True
)

# Automatic W&B logging:
# - Summary metrics as wandb.summary
# - Confusion matrix as wandb.plot.confusion_matrix()
# - Per-example results as wandb.Table()
# - Report artifacts
```

#### Hugging Face Hub

```python
from deepfabric.evaluation import evaluate_model

eval_results = evaluate_model(
    model_path="./model",
    eval_dataset="eval.jsonl",
    push_to_hub="username/model-name",
    hf_token=HF_TOKEN
)

# Automatically:
# 1. Uploads results.json to model repo
# 2. Updates model card with evaluation section
# 3. Tags model with evaluation grade
```


## DeepFabric Cloud Integration

**Upload evaluation results to DeepFabric Cloud for centralized tracking and collaboration.**

```bash
deepfabric evaluate \
  --model ./model \
  --eval-dataset eval.jsonl \
  --output-dir ./eval_results \
  --deepfabric-cloud-upload \
  --api-key YOUR_API_KEY
```



### Time-Series Tracking

**Track improvements over time:**

```python
# evaluation_history.json (auto-maintained)
{
  "model_id": "my-agent-model",
  "evaluations": [
    {
      "timestamp": "2025-10-20T10:00:00Z",
      "checkpoint": "checkpoint-100",
      "overall_score": 0.78,
      "training_steps": 100
    },
    {
      "timestamp": "2025-10-23T10:00:00Z",
      "checkpoint": "checkpoint-500",
      "overall_score": 0.85,
      "training_steps": 500
    },
    {
      "timestamp": "2025-10-27T10:00:00Z",
      "checkpoint": "final",
      "overall_score": 0.87,
      "training_steps": 1000
    }
  ]
}
```

**Generate improvement charts:**

```bash
deepfabric eval-history --model-id my-agent-model --plot
```

Outputs trend chart showing score improvements over time.

---

## CLI & Configuration

### CLI Commands

#### `deepfabric split` (New Command)

```bash
# Split existing dataset into train/eval sets
deepfabric split dataset.jsonl \
  --train-output dataset_train.jsonl \
  --eval-output dataset_eval.jsonl \
  --test-size 0.2 \
  --stratify-by topic

# Full options:
deepfabric split DATASET_PATH \
  --train-output PATH                  # Output path for training set
  --eval-output PATH                   # Output path for evaluation set
  --test-size FLOAT                    # Fraction for eval (default: 0.2)
  --stratify-by FIELD                  # Stratify by field (topic, tool, conversation_type)
  --seed INT                           # Random seed for reproducibility
  --shuffle                            # Shuffle before splitting (default: true)
```

**Key Design Points:**
- Works with any existing dataset (from any version/config)
- Preserves conversation structure and metadata
- Stratification ensures balanced representation
- Can be run multiple times with different splits

#### `deepfabric evaluate`

```bash
# Basic evaluation
deepfabric evaluate \
  --model MODEL_PATH \
  --eval-dataset DATASET_PATH \
  --conversation-type chain_of_thought \
  --output-dir DIR

# Full options:
deepfabric evaluate \
  --model MODEL_PATH                              # Path to fine-tuned model
  --eval-dataset DATASET_PATH                     # Evaluation dataset (original format)
  --conversation-type TYPE                        # Type: basic, structured, chain_of_thought
  --reasoning-style STYLE                         # Style: freetext, structured, hybrid
  --agent-mode MODE                               # Mode: single_turn, multi_turn
  --output-dir DIR                                # Output directory
  --format [json|html|csv|all]                    # Output format (default: all)
  --metrics METRIC1,METRIC2                       # Specific metrics to compute
  --threshold METRIC=VALUE                        # Pass/fail thresholds
  --batch-size N                                  # Inference batch size
  --max-samples N                                 # Limit evaluation samples
  --push-to-hub REPO                              # Upload to HF Hub
  --mlflow-tracking                               # Enable MLflow logging
  --wandb-tracking                                # Enable W&B logging
  --include-failures                              # Save failed examples separately
  --deepfabric-cloud-upload                       # Upload results to DeepFabric Cloud
  --api-key API_KEY                               # API key for DeepFabric Cloud
```

**Critical Design Principles:**
- **Eval dataset must be in original DeepFabric format** (not formatted for training)
- **Conversation type must match** the type used during dataset generation
- Evaluation parses model outputs and compares against ground truth
- Supports all conversation types (basic, structured, chain_of_thought)
- Supports all agent modes (single_turn, multi_turn)

#### `deepfabric evaluate-compare`

```bash
# Compare multiple models
deepfabric evaluate-compare \
  --models NAME1:PATH1 NAME2:PATH2 NAME3:PATH3 \
  --eval-dataset DATASET_PATH \
  --output comparison_report.html

# Example:
deepfabric evaluate-compare \
  --models \
    baseline:./checkpoint-100 \
    improved:./checkpoint-500 \
    final:./final-model \
  --eval-dataset eval.jsonl \
  --output ./comparison
```

#### `deepfabric eval-history`

```bash
# View evaluation history for a model
deepfabric eval-history --model-id my-model

# Generate trend charts
deepfabric eval-history --model-id my-model --plot --output history.html
```

### YAML Configuration

**Evaluation configuration (separate file or embedded):**

```yaml
# eval_config.yaml - Evaluation-specific configuration
evaluation:
  # Conversation type (must match dataset generation)
  conversation_type: "chain_of_thought"
  reasoning_style: "hybrid"
  agent_mode: "single_turn"

  # Metrics to compute
  metrics:
    - tool_selection_accuracy
    - parameter_accuracy
    - execution_success_rate
    - response_quality
    - efficiency

  # Pass/fail thresholds
  thresholds:
    tool_selection_accuracy: 0.85
    execution_success_rate: 0.90
    overall_score: 0.80

  # Metric weights for overall score
  weights:
    tool_selection: 0.40
    parameter_accuracy: 0.30
    execution_success: 0.20
    response_quality: 0.10

  # Output configuration
  output:
    dir: "./eval_results"
    formats: ["json", "html", "csv"]
    include_failures: true
    generate_charts: true

  # Tracking integration
  tracking:
    mlflow: true
    wandb: false
    push_to_hub: null
```

**Or embed in main config for convenience:**

```yaml
# config.yaml - Main dataset generation config
dataset_system_prompt: "..."
topic_tree: {...}
data_engine:
  conversation_type: "chain_of_thought"
  reasoning_style: "hybrid"
  agent_mode: "single_turn"
  {...}

dataset:
  save_as: "dataset.jsonl"
  creation:
    num_steps: 100
    batch_size: 10

# Evaluation settings (optional, can override via CLI)
evaluation:
  thresholds:
    tool_selection_accuracy: 0.85
    overall_score: 0.80
```

---

## Implementation Architecture

### Design Philosophy

**Key Principle: Evaluation uses original DeepFabric format, not training format**

```
Dataset Generation → Original Format (with ground truth)
                           ↓
                     ┌─────┴─────┐
                     ↓           ↓
              Training Set   Eval Set
                     ↓           ↓
              Format for     Keep Original
              Training       (with ground truth)
              (TRL, etc)          ↓
                     ↓           ↓
              Fine-Tune     Evaluation
              Model         (parse outputs,
                           compare to ground truth)
```

**Why this design?**

1. **Formatters may lose information**: Training formatters (TRL, Alpaca, etc.) transform data for specific frameworks, potentially discarding metadata needed for evaluation
2. **Ground truth preservation**: Original format contains complete tool schemas, expected parameters, and correct answers
3. **Flexibility**: Same eval dataset can evaluate models trained with different formatters
4. **Conversation type aware**: Evaluation understands different conversation structures (basic, chain_of_thought, agent modes)

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Evaluation Orchestrator                   │
│  (deepfabric.evaluation.EvaluationEngine)                   │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─> Dataset Splitter (deepfabric.evaluation.split_dataset)
             │   - Stratified sampling by topic/tool/conversation_type
             │   - Preserves conversation structure
             │   - Supports all DeepFabric formats
             │
             ├─> Model Inference Runner
             │   - Batch inference from HuggingFace models
             │   - Response parsing (handles different output formats)
             │   - Conversation type aware
             │
             ├─> Ground Truth Parser
             │   - Extracts expected tools, parameters, answers
             │   - Handles all conversation types
             │   - Parses tool schemas from original format
             │
             ├─> Metric Calculators
             │   - ToolSelectionMetric (works for all agent modes)
             │   - ParameterAccuracyMetric (field-level comparison)
             │   - ExecutionSuccessMetric (validation against schemas)
             │   - ResponseQualityMetric (BLEU/ROUGE/LLM-as-judge)
             │
             ├─> Comparison Engine
             │   - Multi-model evaluation
             │   - Trend analysis
             │
             └─> Output Generators
                 - JSONReporter
                 - HTMLReporter
                 - CSVExporter
                 - ModelCardGenerator
```

### Handling Different Conversation Types

**The evaluation system is conversation-type aware:**

```python
# Evaluation adapts to conversation type
def evaluate_sample(sample, model_output, conversation_type, agent_mode):
    if conversation_type == "basic":
        # Simple Q&A evaluation
        return evaluate_basic_conversation(sample, model_output)

    elif conversation_type == "structured":
        # Structured conversation with metadata
        return evaluate_structured_conversation(sample, model_output)

    elif conversation_type == "chain_of_thought":
        # Chain-of-thought with reasoning traces
        metrics = evaluate_cot_conversation(sample, model_output)
        # Additional metrics for reasoning quality
        metrics["reasoning_coherence"] = evaluate_reasoning(model_output)
        return metrics

    # Agent mode variations
    if agent_mode == "single_turn":
        # Single tool call evaluation
        return evaluate_single_turn_agent(sample, model_output)

    elif agent_mode == "multi_turn":
        # Multi-turn conversation with tool calls
        return evaluate_multi_turn_agent(sample, model_output)
```

**Example: Evaluating Different Formats**

```python
# Basic conversation
{
  "messages": [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "4"}
  ]
}
# Evaluation: Check if answer is correct

# Chain-of-thought
{
  "messages": [
    {"role": "user", "content": "What's 2+2?"},
    {"role": "assistant", "content": "<reasoning>Need to add 2+2</reasoning><answer>4</answer>"}
  ]
}
# Evaluation: Check reasoning coherence AND answer correctness

# Single-turn agent
{
  "messages": [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "content": "<tool_call>get_weather(\"Paris\")</tool_call>"},
    {"role": "tool", "content": "15C, sunny"},
    {"role": "assistant", "content": "It's 15C and sunny in Paris"}
  ],
  "ground_truth": {
    "expected_tool": "get_weather",
    "expected_params": {"location": "Paris"}
  }
}
# Evaluation: Tool selection, parameter accuracy, response quality
```

### Key Classes

```python
# deepfabric/evaluation/engine.py

class EvaluationEngine:
    """Main evaluation orchestrator"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.metrics = [
            ToolSelectionMetric(),
            ParameterAccuracyMetric(),
            ExecutionSuccessMetric(),
            ResponseQualityMetric()
        ]

    def evaluate(
        self,
        model_path: str,
        eval_dataset: str,
        output_dir: str
    ) -> EvaluationResults:
        """Run full evaluation pipeline"""
        pass


class ToolSelectionMetric:
    """Evaluate tool selection correctness"""

    def compute(
        self,
        predictions: List[Prediction],
        ground_truth: List[GroundTruth]
    ) -> ToolSelectionResults:
        """Compute accuracy, precision, recall, F1"""
        pass


class EvaluationResults:
    """Container for all evaluation results"""

    def to_json(self) -> str:
        """Export as JSON"""

    def to_html(self) -> str:
        """Generate HTML report"""

    def to_csv(self) -> str:
        """Export detailed CSV"""

    def generate_model_card(self) -> str:
        """Create model card section"""

    def push_to_hub(self, repo_id: str):
        """Upload to HuggingFace Hub"""
```

### Evaluation Flow

```python
# Pseudocode for evaluation flow

def evaluate_model(model_path, eval_dataset, config):
    # 1. Load model and dataset
    model = load_model(model_path)
    dataset = load_dataset(eval_dataset)

    # 2. Run inference
    predictions = []
    for batch in dataset.batches(config.batch_size):
        outputs = model.generate(batch.queries)
        predictions.extend(parse_outputs(outputs))

    # 3. Compute metrics
    results = EvaluationResults()

    for metric in metrics:
        metric_result = metric.compute(predictions, dataset.ground_truth)
        results.add_metric(metric_result)

    # 4. Analyze failures
    failures = identify_failures(predictions, dataset.ground_truth)
    results.failure_analysis = analyze_failure_patterns(failures)

    # 5. Generate outputs
    results.save_json(output_dir / "results.json")
    results.save_html(output_dir / "report.html")
    results.save_csv(output_dir / "detailed_results.csv")

    # 6. Check thresholds
    if config.thresholds:
        passed = results.check_thresholds(config.thresholds)
        results.threshold_check = passed
        return results, 0 if passed else 1

    return results, 0
```

---

## Use Cases & Workflows

### Use Case 1: Model Development Iteration

**Scenario**: Data scientist training a tool-calling agent

**Workflow**:

1. Generate initial dataset with holdout
2. Fine-tune model v1
3. Evaluate → Score: 78/100
4. Analyze failure report → Issue: Confuses 'calculate' with 'search_web'
5. Augment training data with more calculation examples
6. Fine-tune model v2
7. Evaluate → Score: 85/100 ✓
8. Deploy with confidence

**Value**: Iterative improvement with clear feedback loop

### Use Case 2: Model Selection

**Scenario**: Choosing between model architectures

**Workflow**:

1. Generate evaluation dataset once
2. Fine-tune Llama-1B → Evaluate
3. Fine-tune Qwen-1.5B → Evaluate
4. Fine-tune Phi-3-mini → Evaluate
5. Compare results side-by-side
6. Select best model based on metrics + efficiency

**Value**: Standardized comparison across architectures

### Use Case 3: Production Deployment Gate

**Scenario**: CI/CD pipeline for production deployment

**Workflow**:

```yaml
# .github/workflows/deploy.yml
- name: Evaluate Model
  run: |
    deepfabric evaluate \
      --model ./candidate-model \
      --eval-dataset eval.jsonl \
      --threshold tool_selection_accuracy=0.90 \
      --threshold execution_success_rate=0.95 \
      --format json

- name: Deploy if Passed
  if: success()
  run: deploy_to_production.sh
```

**Value**: Automated quality gate preventing bad deployments

### Use Case 4: Research Publication

**Scenario**: Publishing paper on tool-calling agent training

**Workflow**:

1. Generate comprehensive evaluation dataset
2. Evaluate multiple models
3. Export results as tables for paper
4. Generate comparison charts
5. Include model card in supplementary materials

**Value**: Reproducible, publication-ready results

### Use Case 5: Dataset Quality Validation

**Scenario**: Validating synthetic dataset quality

**Workflow**:

1. Generate synthetic dataset
2. Fine-tune small model on 80%
3. Evaluate on held-out 20%
4. If evaluation score is high → Dataset is high quality
5. If evaluation score is low → Investigate dataset issues

**Value**: Confidence in synthetic data quality

---

## DeepFabric Cloud: SaaS Platform

### Vision

**DeepFabric Cloud** is a managed SaaS platform that provides end-to-end model training observability, evaluation tracking, drift detection, and continuous improvement monitoring for agent systems. It transforms the local CLI evaluation experience into a collaborative, team-wide platform with persistent history, real-time alerts, and production monitoring.

### Core Value Proposition

- **Centralized Model Registry**: Track all models, datasets, and evaluation results in one place
- **Drift Detection**: Monitor production performance and detect distribution shifts
- **Continuous Improvement Tracking**: Visualize model improvements over time with trend analysis
- **Team Collaboration**: Share evaluation results, compare models, and collaborate on improvements
- **Production Monitoring**: Real-time observability for deployed agent models
- **Automated Alerts**: Get notified when models degrade or drift from training distribution

---

### Platform Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        DeepFabric Cloud                               │
│                    (cloud.deepfabric.ai)                             │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        v                    v                    v
┌───────────────┐   ┌────────────────┐   ┌──────────────┐
│  Web Dashboard│   │  API Server    │   │  Production  │
│  (React App)  │   │  (FastAPI)     │   │  Monitor SDK │
└───────────────┘   └────────────────┘   └──────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        v                    v                    v
┌───────────────┐   ┌────────────────┐   ┌──────────────┐
│  PostgreSQL   │   │  TimescaleDB   │   │  Object Store│
│  (Metadata)   │   │  (Timeseries)  │   │  (S3/GCS)    │
└───────────────┘   └────────────────┘   └──────────────┘
```

---

### Key Features

#### 1. Model Registry & Versioning

**Automatic Model Tracking**

```python
# In your training script - automatic tracking
from deepfabric import DeepFabricConfig
from deepfabric.cloud import track_training

with track_training(
    project="customer-support-agent",
    experiment="v2-with-reasoning"
):
    # Generate dataset
    dataset = generate_dataset(config)

    # Fine-tune model
    trainer.train()

    # Evaluate - automatically uploaded to cloud
    eval_results = evaluate_model(
        model_path="./model",
        eval_dataset="eval.jsonl",
        cloud_tracking=True  # Auto-upload to DeepFabric Cloud
    )

# Results automatically visible in dashboard at:
# https://cloud.deepfabric.ai/projects/customer-support-agent/experiments/v2-with-reasoning
```

**Dashboard View:**

```
┌─────────────────────────────────────────────────────────────────┐
│ Project: customer-support-agent                                 │
│                                                                 │
│ Experiments (12)                                                │
│ ┌─────────────────────────────────────────────────────────┐   │
│ │ Name              │ Status  │ Score │ Training Time │ Date│   │
│ │───────────────────┼─────────┼───────┼───────────────┼─────│   │
│ │ v1-baseline       │ Complete│ 78%   │ 2.5h          │ Oct20│  │
│ │ v1.1-more-data    │ Complete│ 82%   │ 3.1h          │ Oct22│  │
│ │ v2-with-reasoning │ Running │ --    │ 1.2h (45%)    │ Oct27│  │
│ │ v2-qwen-test      │ Queued  │ --    │ --            │ --   │  │
│ └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│ [+ New Experiment]  [Compare Selected]  [Export Report]        │
└─────────────────────────────────────────────────────────────────┘
```

#### 2. Real-Time Training Dashboard

**Live Training Monitoring**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Experiment: v2-with-reasoning                        [Live]          │
│                                                                       │
│ Training Progress                                                     │
│ ████████████████████░░░░░░░░░░ 45% (450/1000 steps)                 │
│                                                                       │
│ ┌─────────────────────────────────────────────────────────────┐     │
│ │             Training Loss Over Time                          │     │
│ │  3.5│                                                        │     │
│ │     │●                                                       │     │
│ │  3.0│ ●●                                                     │     │
│ │     │   ●●●                                                  │     │
│ │  2.5│      ●●●●                                             │     │
│ │     │          ●●●●●                                        │     │
│ │  2.0│              ●●●●●●●●●●                              │     │
│ │     │                        ●●●●●●●●●●●                   │     │
│ │  1.5│                                  ●●●●●●●●●●●         │     │
│ │     └──────────────────────────────────────────────────────│     │
│ │      0        200       400        600       800      1000  │     │
│ └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│ Checkpoint Evaluations                                                │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ Step │ Tool Accuracy │ Param Accuracy │ Overall Score       │    │
│ │──────┼───────────────┼────────────────┼────────────────────│    │
│ │ 100  │ 75%           │ 70%            │ 72% [View Report]  │    │
│ │ 200  │ 81%           │ 77%            │ 79% [View Report]  │    │
│ │ 300  │ 85%           │ 81%            │ 83% [View Report]  │    │
│ │ 400  │ 87%           │ 84%            │ 85% [View Report]  │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ Current Metrics                                                       │
│ • GPU Utilization: 92%                                               │
│ • Samples/sec: 12.4                                                  │
│ • Est. completion: 1h 15m                                            │
└──────────────────────────────────────────────────────────────────────┘
```

#### 3. Comprehensive Comparison View

**Multi-Model Comparison Dashboard**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Compare Experiments                                                   │
│                                                                       │
│ Selected: v1-baseline, v1.1-more-data, v2-with-reasoning            │
│                                                                       │
│ Overall Performance                                                   │
│ ┌─────────────────────────────────────────────────────────────┐     │
│ │                Score Comparison                              │     │
│ │  100│                                                        │     │
│ │     │                                                        │     │
│ │   90│                                          ██            │     │
│ │     │                              ██          ██            │     │
│ │   80│                  ██          ██          ██            │     │
│ │     │      ██          ██          ██          ██            │     │
│ │   70│      ██          ██          ██          ██            │     │
│ │     └──────────────────────────────────────────────────────│     │
│ │        Overall    Tool Sel   Param Acc  Exec Success      │     │
│ │        [█ v1  █ v1.1  █ v2]                                │     │
│ └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│ Detailed Metrics Table                                                │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ Metric              │ v1-baseline│ v1.1      │ v2           │    │
│ │─────────────────────┼────────────┼───────────┼──────────────│    │
│ │ Overall Score       │ 78%        │ 82% ⬆️4%  │ 87% ⬆️9%     │    │
│ │ Tool Selection      │ 80%        │ 84% ⬆️4%  │ 89% ⬆️9%     │    │
│ │ Parameter Accuracy  │ 76%        │ 80% ⬆️4%  │ 85% ⬆️9%     │    │
│ │ Execution Success   │ 85%        │ 88% ⬆️3%  │ 91% ⬆️6%     │    │
│ │ Inference Time (ms) │ 890        │ 920 ⬇️30  │ 1247 ⬇️357   │    │
│ │ Model Size (GB)     │ 2.4        │ 2.4 ➡️    │ 2.4 ➡️       │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ Key Insights                                                          │
│ • v2 shows 9% improvement in overall score vs baseline               │
│ • All metrics improved, but inference time increased by 40%          │
│ • Recommendation: Consider optimization for production deployment    │
│                                                                       │
│ [Export Comparison] [Generate Report] [Share Link]                   │
└──────────────────────────────────────────────────────────────────────┘
```

#### 4. Improvement Tracking & Trends

**Time-Series Performance Tracking**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Project: customer-support-agent                                       │
│ Performance Over Time (Last 6 Months)                                 │
│                                                                       │
│ Overall Score Trend                                                   │
│ ┌─────────────────────────────────────────────────────────────┐     │
│ │  100│                                               ●        │     │
│ │     │                                          ●──●          │     │
│ │   90│                                     ●──●               │     │
│ │     │                                ●──●                    │     │
│ │   80│                           ●──●                         │     │
│ │     │                      ●──●                              │     │
│ │   70│                 ●──●                                   │     │
│ │     │            ●──●                                        │     │
│ │   60│       ●──●                                             │     │
│ │     └──────────────────────────────────────────────────────│     │
│ │       May   Jun   Jul   Aug   Sep   Oct   Nov              │     │
│ └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│ Key Milestones                                                        │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ Date    │ Event                           │ Impact          │    │
│ │─────────┼─────────────────────────────────┼─────────────────│    │
│ │ May 15  │ Initial model (v0.1)            │ 58% baseline    │    │
│ │ Jun 03  │ Added chain-of-thought          │ +7% reasoning   │    │
│ │ Jul 12  │ Expanded training data (3x)     │ +8% accuracy    │    │
│ │ Aug 20  │ Improved parameter extraction   │ +5% params      │    │
│ │ Sep 18  │ Multi-tool support added        │ +6% complex     │    │
│ │ Oct 27  │ v2 with hybrid reasoning        │ +4% overall     │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ Progress Statistics                                                   │
│ • Total improvement: +29 points (58% → 87%)                          │
│ • Average monthly improvement: +4.8 points                            │
│ • Best single improvement: +8 points (Jul data expansion)            │
│                                                                       │
│ [Export Timeline] [Download Data] [Set Goals]                        │
└──────────────────────────────────────────────────────────────────────┘
```

#### 5. Production Monitoring & Drift Detection

**Real-Time Production Dashboard**

```python
# Install production monitoring SDK
pip install deepfabric-cloud

# Integrate with your production agent
from deepfabric.cloud import ProductionMonitor

monitor = ProductionMonitor(
    api_key=DEEPFABRIC_API_KEY,
    project="customer-support-agent",
    model_version="v2-with-reasoning"
)

# Wrap your agent inference
@monitor.track_inference
def handle_customer_query(query: str) -> dict:
    result = agent.process(query)
    return result

# Automatically tracks:
# - Tool selection distribution
# - Parameter patterns
# - Inference latency
# - Error rates
# - Response quality
```

**Production Dashboard View:**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Production: customer-support-agent v2                    [Live]       │
│                                                                       │
│ Health Status: ⚠️  WARNING - Potential drift detected                │
│                                                                       │
│ Traffic & Performance (Last 24h)                                     │
│ ┌────────────────────┬────────────────────┬────────────────────┐    │
│ │ Total Requests     │ Avg Latency        │ Error Rate         │    │
│ │ 45,234             │ 1.3s               │ 2.1%               │    │
│ │ ⬆️ 12% vs yesterday│ ⬆️ 8% vs baseline  │ ⬆️ 0.5% vs baseline│    │
│ └────────────────────┴────────────────────┴────────────────────┘    │
│                                                                       │
│ Tool Selection Distribution Drift                                     │
│ ┌─────────────────────────────────────────────────────────────┐     │
│ │              Training    │    Production (24h)             │     │
│ │─────────────────────────┼─────────────────────────────────│     │
│ │ get_weather    35% ████ │ get_weather    28% ███ ⚠️        │     │
│ │ calculate      25% ███  │ calculate      31% ████ ⚠️       │     │
│ │ search_web     30% ███  │ search_web     35% ████ ⚠️       │     │
│ │ get_time       10% █    │ get_time        6% █ ⚠️          │     │
│ └─────────────────────────────────────────────────────────────┘     │
│                                                                       │
│ Drift Score: 0.23 (⚠️ Warning threshold: 0.20)                       │
│                                                                       │
│ Alerts (3)                                                            │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ ⚠️  Tool distribution shifted by 23% - investigate cause     │    │
│ │ ⚠️  Latency increased 8% - check infrastructure              │    │
│ │ ℹ️  New query patterns detected in 5% of requests            │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ Recommendations                                                       │
│ • Consider retraining with recent production data                    │
│ • Analyze new query patterns for dataset augmentation                │
│ • Monitor latency - may need model optimization                      │
│                                                                       │
│ [View Full Report] [Trigger Evaluation] [Configure Alerts]           │
└──────────────────────────────────────────────────────────────────────┘
```

**Drift Detection Algorithm:**

```python
# Automated drift detection metrics
drift_metrics = {
    # Tool distribution drift (JS Divergence)
    "tool_distribution_divergence": 0.23,  # ⚠️ Above threshold

    # Parameter distribution drift
    "parameter_distribution_shift": 0.12,  # ✅ Within threshold

    # Response time drift
    "latency_drift_percent": 8.0,  # ⚠️ Above 5% threshold

    # Error rate change
    "error_rate_change": 0.5,  # ℹ️ Monitored but acceptable

    # New query patterns
    "unknown_pattern_percent": 5.0,  # ℹ️ Novel queries detected

    # Overall drift score (weighted composite)
    "overall_drift_score": 0.23  # ⚠️ Warning level
}
```

#### 6. Automated Retraining Triggers

**Smart Retraining Workflow**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Automated Retraining Pipeline                                         │
│                                                                       │
│ Trigger Conditions (Configurable)                                    │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ ☑ Drift score exceeds 0.25 for 7 days                        │    │
│ │ ☑ Error rate increases by >2% for 3 days                     │    │
│ │ ☑ New query patterns exceed 10% of traffic                   │    │
│ │ ☑ Manual trigger by team member                              │    │
│ │ ☐ Schedule: Monthly automatic retraining                     │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ Current Status: ⏸️  Pending Approval                                  │
│                                                                       │
│ Retraining Proposal                                                   │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ Reason: Drift score 0.23 for 5 consecutive days              │    │
│ │                                                               │    │
│ │ Proposed Actions:                                             │    │
│ │ 1. Collect 5,000 production queries from last 7 days         │    │
│ │ 2. Augment training dataset with new patterns                │    │
│ │ 3. Generate 10,000 additional synthetic examples             │    │
│ │ 4. Retrain model with combined dataset                       │    │
│ │ 5. Evaluate on production-like holdout set                   │    │
│ │                                                               │    │
│ │ Estimated Resources:                                          │    │
│ │ • Dataset generation: 2h, $5                                 │    │
│ │ • Training time: 4h on 1x A100                              │    │
│ │ • Total cost: ~$35                                           │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ [✓ Approve & Start] [✗ Dismiss] [⚙️ Configure]                      │
└──────────────────────────────────────────────────────────────────────┘
```

#### 7. Team Collaboration Features

**Project Sharing & Comments**

```
┌──────────────────────────────────────────────────────────────────────┐
│ Experiment: v2-with-reasoning                                         │
│                                                                       │
│ Team Activity                                                         │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ @sarah commented 2 hours ago:                                 │    │
│ │ "Great improvement on tool selection! But latency is up 40%. │    │
│ │  Let's try quantization next iteration."                      │    │
│ │                                                               │    │
│ │ @mike replied 1 hour ago:                                     │    │
│ │ "Agreed. I'll test GPTQ quantization this afternoon."        │    │
│ │                                                               │    │
│ │ @john tagged you in a report 30 min ago:                     │    │
│ │ "Check out the failure analysis - calculation errors up 3%"  │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ Assigned Tasks                                                        │
│ ┌──────────────────────────────────────────────────────────────┐    │
│ │ ☐ Investigate calculation tool errors (@sarah)                │    │
│ │ ☐ Test quantized model (@mike)                               │    │
│ │ ☑ Document evaluation results (@john) - Done                 │    │
│ └──────────────────────────────────────────────────────────────┘    │
│                                                                       │
│ [Add Comment] [Assign Task] [Share Link] [Export to Slack]          │
└──────────────────────────────────────────────────────────────────────┘
```

#### 8. Automated Reporting & Insights

**Weekly Summary Email:**

```
Subject: [DeepFabric] Weekly Summary - customer-support-agent

Hi Team,

Here's your weekly summary for the customer-support-agent project:

📊 PERFORMANCE OVERVIEW
• Production requests: 324,567 (⬆️ 8% vs last week)
• Average score: 87% (➡️ stable)
• Error rate: 2.1% (⬆️ 0.3%)

⚠️ ALERTS & DRIFT
• Drift warning: Tool distribution shifted 23%
• Latency increased by 8% - investigate infrastructure
• New query patterns detected: 5% of traffic

🚀 IMPROVEMENTS
• v2.1 experiment completed: 89% score (⬆️ 2%)
• Best checkpoint identified: step-800
• Recommended for production deployment

📈 TRENDS (30 days)
• Overall improvement: +4 points
• Parameter accuracy: +3%
• Fastest model: v2.1 (1.1s avg)

🎯 RECOMMENDED ACTIONS
1. Deploy v2.1 to production (89% score, lower latency)
2. Collect production queries for dataset augmentation
3. Schedule retraining for next week

View full dashboard: https://cloud.deepfabric.ai/projects/customer-support-agent

---
DeepFabric Cloud
Unsubscribe | Notification Settings
```

---

### Pricing Tiers

#### Free Tier
- **Price**: $0/month
- **Limits**:
  - 1 project
  - 10 experiments/month
  - 1,000 production requests/month monitored
  - 7-day data retention
  - Basic drift detection
- **Best for**: Individual developers, prototyping

#### Pro Tier
- **Price**: $49/month
- **Limits**:
  - 5 projects
  - Unlimited experiments
  - 100,000 production requests/month monitored
  - 90-day data retention
  - Advanced drift detection with alerts
  - Team collaboration (up to 5 members)
  - API access
  - Slack/Discord integration
- **Best for**: Small teams, production deployments

#### Enterprise Tier
- **Price**: Custom (starting at $499/month)
- **Features**:
  - Unlimited projects
  - Unlimited experiments
  - Unlimited production monitoring
  - Unlimited data retention
  - Dedicated support
  - SSO/SAML
  - On-premise deployment option
  - Custom integrations
  - SLA guarantees
- **Best for**: Large organizations, mission-critical systems

---

### Integration & Setup

#### Quick Start

```bash
# Install DeepFabric Cloud CLI
pip install deepfabric-cloud

# Authenticate
deepfabric cloud login

# Create project
deepfabric cloud create-project customer-support-agent

# Set default project
deepfabric cloud use customer-support-agent

# Enable cloud tracking in your code
from deepfabric.cloud import enable_cloud_tracking
enable_cloud_tracking()

# All evaluations now automatically sync to cloud!
```

#### SDK Integration

```python
from deepfabric import DeepFabricConfig
from deepfabric.cloud import CloudTracker

# Initialize cloud tracker
tracker = CloudTracker(
    api_key=DEEPFABRIC_API_KEY,
    project="customer-support-agent",
    experiment_name="v2-with-reasoning"
)

# Track dataset generation
with tracker.track_phase("dataset_generation"):
    dataset = generate_dataset(config)
    tracker.log_artifact("dataset.jsonl", dataset)

# Track training
with tracker.track_phase("training"):
    trainer.train()
    tracker.log_metrics({
        "training_loss": final_loss,
        "training_time": elapsed_time
    })

# Track evaluation (automatically uploads all results)
with tracker.track_phase("evaluation"):
    eval_results = evaluate_model(
        model_path="./model",
        eval_dataset="eval.jsonl",
        cloud_tracking=True
    )

# Mark experiment complete
tracker.complete()
```

#### Production Monitoring SDK

```python
from deepfabric.cloud import ProductionMonitor
from fastapi import FastAPI

app = FastAPI()
monitor = ProductionMonitor(
    api_key=DEEPFABRIC_API_KEY,
    project="customer-support-agent",
    model_version="v2-with-reasoning",
    drift_detection=True,  # Enable automatic drift detection
    alert_threshold=0.20   # Alert when drift > 0.20
)

@app.post("/query")
@monitor.track_request
async def handle_query(query: str):
    # Your agent logic here
    result = agent.process(query)

    # Optionally add custom metrics
    monitor.log_custom_metric("query_complexity", calculate_complexity(query))

    return result

# Monitor will automatically track:
# - Request volume and latency
# - Tool selection distribution
# - Parameter patterns
# - Error rates
# - Drift metrics
```

---

### Data Privacy & Security

#### Security Features

- **Encryption**: All data encrypted at rest (AES-256) and in transit (TLS 1.3)
- **Access Control**: Role-based access control (RBAC) for team members
- **Audit Logs**: Complete audit trail of all actions
- **Data Isolation**: Strict tenant isolation in multi-tenant environment
- **Compliance**: SOC 2 Type II, GDPR compliant
- **Data Residency**: Choose data center region (US, EU, Asia)

#### Privacy Options

```yaml
# Configure what gets tracked
deepfabric_cloud:
  tracking:
    # Track evaluation metrics (required)
    metrics: true

    # Track model artifacts (optional)
    model_artifacts: false  # Don't upload model weights

    # Track dataset samples (optional)
    dataset_samples: false  # Don't upload training data

    # Track production queries (optional)
    production_queries: true  # Upload for drift analysis

    # Anonymize production data
    anonymize_pii: true  # Strip PII before upload

    # Data retention
    retention_days: 90  # Auto-delete after 90 days
```

---

### API Reference

#### REST API

```bash
# Get project overview
GET /api/v1/projects/{project_id}

# List experiments
GET /api/v1/projects/{project_id}/experiments

# Get experiment details
GET /api/v1/experiments/{experiment_id}

# Get evaluation results
GET /api/v1/experiments/{experiment_id}/evaluations

# Compare experiments
POST /api/v1/experiments/compare
{
  "experiment_ids": ["exp1", "exp2", "exp3"],
  "metrics": ["tool_selection_accuracy", "overall_score"]
}

# Get drift analysis
GET /api/v1/production/{model_id}/drift?period=7d

# Trigger retraining
POST /api/v1/projects/{project_id}/retrain
{
  "trigger_reason": "drift_threshold_exceeded",
  "config": {...}
}
```

---

### Benefits of DeepFabric Cloud

#### For Individual Developers

✅ **Centralized History**: Never lose evaluation results again
✅ **Visual Trends**: See your model improvements over time
✅ **Easy Comparison**: Compare experiments side-by-side
✅ **Free Tier**: Start without cost for small projects

#### For Teams

✅ **Collaboration**: Share results, comment, assign tasks
✅ **Knowledge Sharing**: Document learnings and best practices
✅ **Standardization**: Consistent evaluation across team
✅ **Accountability**: Track who trained what and when

#### For Production Systems

✅ **Drift Detection**: Catch performance degradation early
✅ **Automated Alerts**: Get notified of issues immediately
✅ **Root Cause Analysis**: Understand why models fail
✅ **Continuous Monitoring**: Real-time observability
✅ **Smart Retraining**: Automated recommendations for when to retrain

#### For Organizations

✅ **ROI Tracking**: Quantify model improvements over time
✅ **Resource Optimization**: Identify which experiments work
✅ **Risk Management**: Quality gates before production
✅ **Compliance**: Audit trails and data governance
✅ **Cross-Team Learning**: Share insights across projects

---

## Future Enhancements

### Phase 2 Features

1. **Human Evaluation Integration**
   - Upload results to annotation platform
   - Collect human judgments for subset
   - Correlate automatic metrics with human ratings

2. **Active Learning Loop**
   - Identify hardest examples from evaluation
   - Generate more training data targeting weaknesses
   - Re-evaluate after augmentation

3. **Drift Detection**
   - Monitor production inference logs
   - Compare production distribution to eval dataset
   - Alert when distribution shifts

4. **Multi-Task Evaluation**
   - Evaluate across multiple tool sets simultaneously
   - Domain-specific metric configurations
   - Cross-domain transfer analysis

5. **LLM-as-Judge Integration**
   - Use GPT-4 to judge response quality
   - Automated reasoning coherence scoring
   - Natural language feedback generation

6. **Interpretability Analysis**
   - Attention visualization for tool selection
   - Feature importance for parameter prediction
   - Counterfactual analysis

### Integration Wishlist

- **Langfuse**: Real-time production monitoring
- **Arize**: Model performance tracking
- **Evidently AI**: Drift detection
- **Label Studio**: Human evaluation interface

---

## Conclusion

The DeepFabric evaluation feature transforms synthetic dataset generation into a complete training and validation pipeline. By enabling users to:

1. **Hold out evaluation data** during generation
2. **Evaluate fine-tuned models** with comprehensive metrics
3. **Track improvements** over time with experiment tracking
4. **Generate actionable reports** for debugging and communication
5. **Integrate seamlessly** into ML pipelines and CI/CD

DeepFabric becomes not just a dataset generator, but a **complete agent training platform**.

The evaluation system's **zero-friction UX** (as simple as `--holdout 0.2` and `deepfabric evaluate`) combined with **rich, actionable outputs** makes it valuable for every stage of the agent development lifecycle: from initial prototyping to production deployment and continuous improvement.

Most importantly, the evaluation feature **closes the feedback loop**: users can now measure whether their fine-tuning is working, understand why models fail, and systematically improve agent capabilities with confidence.

---

## Appendix: Configuration Examples

### Complete Workflow Example

```bash
# 1. Generate dataset
deepfabric generate agent_config.yaml
# Output: dataset_full.jsonl (1000 samples)

# 2. Split for training and evaluation
deepfabric split dataset_full.jsonl \
  --train-output dataset_train.jsonl \
  --eval-output dataset_eval.jsonl \
  --test-size 0.2 \
  --stratify-by topic \
  --seed 42

# Outputs:
# - dataset_train.jsonl (800 samples, stratified)
# - dataset_eval.jsonl (200 samples, stratified)

# 3. Format ONLY training data for your framework
deepfabric format dataset_train.jsonl \
  --formatter trl_sft_tools \
  --output dataset_train_formatted.jsonl

# 4. Train your model (using formatted training data)
python train.py \
  --train-data dataset_train_formatted.jsonl \
  --output ./fine-tuned-model

# 5. Evaluate (using ORIGINAL eval data, not formatted)
deepfabric evaluate \
  --model ./fine-tuned-model \
  --eval-dataset dataset_eval.jsonl \
  --conversation-type chain_of_thought \
  --reasoning-style hybrid \
  --agent-mode single_turn \
  --threshold tool_selection_accuracy=0.85 \
  --threshold overall_score=0.80 \
  --output-dir ./eval_results
```

### Minimal Evaluation Config

```yaml
# eval_config.yaml
evaluation:
  conversation_type: "chain_of_thought"
  thresholds:
    overall_score: 0.80
```

### Production Evaluation Config

```yaml
# eval_config.yaml
evaluation:
  # Must match dataset generation config
  conversation_type: "chain_of_thought"
  reasoning_style: "hybrid"
  agent_mode: "single_turn"

  # Metrics to compute
  metrics:
    - tool_selection_accuracy
    - parameter_accuracy
    - execution_success_rate
    - response_quality
    - efficiency

  # Pass/fail thresholds for CI/CD
  thresholds:
    tool_selection_accuracy: 0.90
    execution_success_rate: 0.95
    overall_score: 0.85

  # Metric weights
  weights:
    tool_selection: 0.40
    parameter_accuracy: 0.30
    execution_success: 0.20
    response_quality: 0.10

  # Output configuration
  output:
    dir: "./eval_results"
    formats: ["json", "html", "csv"]
    include_failures: true
    generate_charts: true

  # Tracking integration
  tracking:
    mlflow: true
    wandb: true
    push_to_hub: "username/model-name"
```

### Working with Multiple Formatters

```bash
# Generate dataset once
deepfabric generate config.yaml
# Output: dataset_full.jsonl

# Split once
deepfabric split dataset_full.jsonl \
  --train-output dataset_train.jsonl \
  --eval-output dataset_eval.jsonl \
  --test-size 0.2

# Train multiple models with DIFFERENT formatters
# Model 1: TRL format
deepfabric format dataset_train.jsonl -f trl_sft_tools -o train_trl.jsonl
python train.py --data train_trl.jsonl --output model_trl

# Model 2: Alpaca format
deepfabric format dataset_train.jsonl -f alpaca -o train_alpaca.jsonl
python train.py --data train_alpaca.jsonl --output model_alpaca

# Evaluate BOTH models with the SAME eval dataset (original format)
deepfabric evaluate --model model_trl --eval-dataset dataset_eval.jsonl --output eval_trl
deepfabric evaluate --model model_alpaca --eval-dataset dataset_eval.jsonl --output eval_alpaca

# Compare results
deepfabric evaluate-compare \
  --models trl:model_trl alpaca:model_alpaca \
  --eval-dataset dataset_eval.jsonl \
  --output comparison.html
```

---

**Document Version**: 1.1
**Last Updated**: 2025-10-27
**Author**: DeepFabric Team
**Status**: Design Proposal

**Key Changes in v1.1:**
- Evaluation now operates on existing datasets (via `deepfabric split`)
- Training data is formatted, evaluation data stays in original format
- Conversation type awareness throughout evaluation pipeline
- Support for all formatters while maintaining consistent evaluation
- Comprehensive SaaS platform design with drift detection and improvement tracking
