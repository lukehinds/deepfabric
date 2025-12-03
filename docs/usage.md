# DeepFabric Usage Guide

This guide covers the complete workflow for generating synthetic training data, working with datasets, and evaluating fine-tuned models.

## Overview

DeepFabric generates synthetic training data and returns standard HuggingFace `datasets.Dataset` objects. This means you can use all native HuggingFace methods for splitting, saving, uploading, and processing your data.

## Installation

```bash
pip install deepfabric
```

## Dataset Generation

### Using the CLI

The simplest way to generate a dataset:

```bash
export OPENAI_API_KEY="your-api-key"

deepfabric generate \
  --topic-prompt "Python programming fundamentals" \
  --generation-system-prompt "You are a Python expert" \
  --mode tree \
  --depth 3 \
  --degree 3 \
  --num-samples 27 \
  --batch-size 3 \
  --provider openai \
  --model gpt-4o \
  --output-save-as dataset.jsonl
```

### Using the Python API

```python
from deepfabric import DataSetGenerator, Tree

# Generate topic tree
tree = Tree(
    root="Python programming fundamentals",
    provider="openai",
    model_name="gpt-4o",
    tree_depth=3,
    tree_degree=3,
)
tree.build_tree()

# Generate dataset (returns HuggingFace Dataset)
generator = DataSetGenerator(
    instructions="Create practical Python examples",
    generation_system_prompt="You are a Python expert",
    dataset_system_prompt="You are a helpful Python assistant",
    provider="openai",
    model_name="gpt-4o",
)

dataset = generator.create_data(
    num_steps=3,
    batch_size=9,
    topic_model=tree,
)

# dataset is a HuggingFace Dataset object
print(f"Generated {len(dataset)} samples")
```

## Working with Datasets

Since DeepFabric returns HuggingFace `datasets.Dataset` objects, you can use all standard HF methods.

### Saving Datasets

```python
# Save to JSONL
dataset.to_json("my-dataset.jsonl", orient="records", lines=True)

# Save to Parquet (more efficient for large datasets)
dataset.to_parquet("my-dataset.parquet")

# Save to CSV
dataset.to_csv("my-dataset.csv")
```

### Uploading to HuggingFace Hub

```python
# Push directly to Hub
dataset.push_to_hub("your-username/my-dataset")

# With private visibility
dataset.push_to_hub("your-username/my-dataset", private=True)
```

### Loading from HuggingFace Hub

```python
from datasets import load_dataset

# Load from Hub
dataset = load_dataset("your-username/my-dataset", split="train")

# Load specific split
train_ds = load_dataset("your-username/my-dataset", split="train")
```

### Loading from Local Files

```python
from datasets import load_dataset

# Load from JSONL
dataset = load_dataset("json", data_files="my-dataset.jsonl", split="train")

# Load from Parquet
dataset = load_dataset("parquet", data_files="my-dataset.parquet", split="train")
```

## Splitting Datasets

Use native HuggingFace methods for splitting:

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("your-username/my-dataset", split="train")

# Split into train/test
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds = splits["test"]

print(f"Train: {len(train_ds)} samples")
print(f"Eval: {len(eval_ds)} samples")

# Split into train/validation/test
train_test = dataset.train_test_split(test_size=0.2, seed=42)
test_valid = train_test["test"].train_test_split(test_size=0.5, seed=42)

train_ds = train_test["train"]
valid_ds = test_valid["train"]
test_ds = test_valid["test"]
```

## Formatting for Training

Use your tokenizer's `apply_chat_template` for formatting. This is the standard approach used by training frameworks.

### Basic Formatting

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset and tokenizer
dataset = load_dataset("your-username/my-dataset", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def format_example(example):
    # Clean messages (remove None values if present)
    messages = []
    for msg in example["messages"]:
        clean_msg = {k: v for k, v in msg.items() if v is not None}
        messages.append(clean_msg)

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }

formatted_dataset = dataset.map(format_example)
```

### Formatting with Tool Calls

For datasets with tool calls, ensure your tokenizer supports the format:

```python
from transformers import AutoTokenizer

# Qwen models support parallel tool calls
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def format_with_tools(example):
    messages = []
    for msg in example["messages"]:
        clean_msg = {k: v for k, v in msg.items() if v is not None}
        messages.append(clean_msg)

    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    }

formatted_dataset = dataset.map(format_with_tools)
```

### Custom Formatting

For custom formats or when chat templates don't fit your needs:

```python
def custom_format(example):
    text = ""
    for msg in example["messages"]:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            text += f"### System:\n{content}\n\n"
        elif role == "user":
            text += f"### User:\n{content}\n\n"
        elif role == "assistant":
            text += f"### Assistant:\n{content}\n\n"
        elif role == "tool":
            text += f"### Tool Result:\n{content}\n\n"

    return {"text": text}

formatted_dataset = dataset.map(custom_format)
```

## Training Integration

### With TRL SFTTrainer

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# Load and prepare data
dataset = load_dataset("your-username/my-dataset", split="train")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

def format_example(example):
    messages = [{k: v for k, v in msg.items() if v is not None}
                for msg in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted = dataset.map(format_example)

# Split for training
splits = formatted.train_test_split(test_size=0.1, seed=42)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="auto",
    device_map="auto",
)

# Configure training
config = SFTConfig(
    output_dir="./output",
    max_seq_length=2048,
    per_device_train_batch_size=4,
    num_train_epochs=3,
)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=splits["train"],
    eval_dataset=splits["test"],
    args=config,
)
trainer.train()
```

### With Unsloth

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# Load with Unsloth optimizations
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
)

# Load and format dataset
dataset = load_dataset("your-username/my-dataset", split="train")

def format_example(example):
    messages = [{k: v for k, v in msg.items() if v is not None}
                for msg in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted = dataset.map(format_example)
splits = formatted.train_test_split(test_size=0.1, seed=42)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=splits["train"],
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
    ),
)
trainer.train()
```

## Evaluation

DeepFabric includes an evaluation module for assessing fine-tuned models on tool-calling tasks.

### Evaluating with HuggingFace Dataset

```python
from datasets import load_dataset
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

# Load and split dataset
dataset = load_dataset("your-username/my-dataset", split="train")
splits = dataset.train_test_split(test_size=0.1, seed=42)
eval_ds = splits["test"]

# Configure evaluator
config = EvaluatorConfig(
    model_path="./output/checkpoint-final",  # Your fine-tuned model
    inference_config=InferenceConfig(backend="vllm"),
)

# Run evaluation with HF Dataset directly
evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_ds)

# Print results
print(f"Samples Evaluated: {results.metrics.samples_evaluated}")
print(f"Tool Selection Accuracy: {results.metrics.tool_selection_accuracy:.2%}")
print(f"Parameter Accuracy: {results.metrics.parameter_accuracy:.2%}")
print(f"Execution Success Rate: {results.metrics.execution_success_rate:.2%}")
print(f"Overall Score: {results.metrics.overall_score:.2%}")
```

### Evaluating from File

The file-based approach is still supported:

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

# Save eval split to file first
eval_ds.to_json("eval_split.jsonl", orient="records", lines=True)

# Configure with file path
config = EvaluatorConfig(
    model_path="./output/checkpoint-final",
    dataset_path="eval_split.jsonl",
    inference_config=InferenceConfig(backend="vllm"),
    output_path="eval_results.json",
)

# Run evaluation
evaluator = Evaluator(config)
results = evaluator.evaluate()
```

### Evaluation with LoRA Adapters

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    model_path="meta-llama/Llama-3.1-8B-Instruct",  # Base model
    inference_config=InferenceConfig(
        backend="vllm",
        adapter_path="./output/checkpoint-final",  # LoRA adapter
    ),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_ds)
```

### Limiting Evaluation Samples

```python
config = EvaluatorConfig(
    model_path="./output/checkpoint-final",
    inference_config=InferenceConfig(backend="vllm"),
    max_samples=100,  # Only evaluate first 100 samples
)
```

## Complete Workflow Example

Here's a complete example from generation to evaluation:

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from deepfabric import DataSetGenerator, Tree
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

# 1. Generate dataset
tree = Tree(
    root="Building REST APIs with FastAPI",
    provider="openai",
    model_name="gpt-4o",
    tree_depth=3,
    tree_degree=3,
)
tree.build_tree()

generator = DataSetGenerator(
    instructions="Create practical FastAPI examples with tool usage",
    generation_system_prompt="You are a FastAPI expert",
    dataset_system_prompt="You are a helpful API development assistant",
    provider="openai",
    model_name="gpt-4o",
)

dataset = generator.create_data(num_steps=3, batch_size=9, topic_model=tree)

# 2. Save and upload
dataset.to_json("fastapi-dataset.jsonl", orient="records", lines=True)
dataset.push_to_hub("your-username/fastapi-training-data")

# 3. Split for training
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds = splits["test"]

# 4. Format for training
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def format_example(example):
    messages = [{k: v for k, v in msg.items() if v is not None}
                for msg in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted_train = train_ds.map(format_example)

# 5. Train your model (using TRL, Unsloth, etc.)
# ... training code ...

# 6. Evaluate fine-tuned model
config = EvaluatorConfig(
    model_path="./output/checkpoint-final",
    inference_config=InferenceConfig(backend="vllm"),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_ds)

print(f"Overall Score: {results.metrics.overall_score:.2%}")
```

## Supported Providers

| Provider | Environment Variable | Example Model |
|----------|---------------------|---------------|
| OpenAI | `OPENAI_API_KEY` | `gpt-4o`, `gpt-4o-mini` |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.0-flash` |
| Ollama | - (local) | `llama3.1`, `mistral` |
| OpenRouter | `OPENROUTER_API_KEY` | Various models |

## Resources

- [Configuration Reference](../README.md#configuration)
- [Examples](../examples/)
- [API Documentation](https://always-further.github.io/deepfabric/)
- [Discord Community](https://discord.gg/pPcjYzGvbS)
