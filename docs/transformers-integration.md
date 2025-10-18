# HuggingFace Transformers Integration

## Overview

DeepFabric now supports local HuggingFace Transformers models for inference and training. This integration enables:

- **Local model inference** for topic tree and dataset generation
- **Outlines integration** for structured JSON output
- **Direct SFT training pipeline** with TRL
- **LoRA/QLoRA support** for efficient fine-tuning
- **End-to-end workflows** from generation to deployment

## Installation

### Basic Installation (Generation Only)

```bash
pip install deepfabric
```

### With Training Support

```bash
pip install 'deepfabric[training]'
```

This installs:
- `transformers>=4.45.0`
- `accelerate>=0.34.0`
- `trl>=0.11.0`
- `peft>=0.13.0`
- `bitsandbytes>=0.43.0`
- `torch>=2.0.0`

## Usage

### 1. Local Model Inference for Dataset Generation

```python
import asyncio
from deepfabric import Tree, DataSetGenerator

# Generate topic tree with local model
tree = Tree(
    topic_prompt="Advanced Python Programming",
    provider="transformers",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
    depth=3,
    degree=10
)

tree_paths = asyncio.run(tree.build_async())
print(f"Generated {len(tree_paths)} topic paths")

# Generate dataset with local model
generator = DataSetGenerator(
    generation_system_prompt="Create high-quality educational Q&A pairs",
    provider="transformers",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
    conversation_type="cot_structured",
    temperature=0.7
)

dataset = asyncio.run(generator.create_data_async(
    num_steps=100,
    batch_size=5,
    topic_model=tree
))

dataset.save("training_data.jsonl")
print(f"Generated {len(dataset)} training samples")
```

### 2. Fine-Tuning with LoRA

```python
from deepfabric.training import DeepFabricSFTTrainer, SFTTrainingConfig, LoRAConfig

# Configure training with LoRA
config = SFTTrainingConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./finetuned-model",

    # Training hyperparameters
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_ratio=0.03,

    # Optimization
    bf16=True,
    gradient_checkpointing=True,

    # LoRA configuration
    lora=LoRAConfig(
        enabled=True,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear"
    ),

    # Logging
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=3
)

# Train
trainer = DeepFabricSFTTrainer(
    config=config,
    train_dataset=dataset,
    formatting_func="trl_sft_tools"  # Use TRL SFT formatter
)

metrics = trainer.train()
print(f"Training complete. Final loss: {metrics['train_loss']}")

# Save model
trainer.save_model()

# Upload to HuggingFace Hub (optional)
trainer.push_to_hub("username/my-finetuned-model")
```

### 3. QLoRA (4-bit Quantization)

```python
from deepfabric.training import SFTTrainingConfig, LoRAConfig, QuantizationConfig

config = SFTTrainingConfig(
    model_name="meta-llama/Llama-3.1-70B-Instruct",  # Large model!
    output_dir="./qlora-model",
    num_train_epochs=3,
    per_device_train_batch_size=1,  # Smaller batch for memory
    gradient_accumulation_steps=16,

    # LoRA
    lora=LoRAConfig(
        enabled=True,
        r=64,
        lora_alpha=128
    ),

    # 4-bit quantization
    quantization=QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
)

trainer = DeepFabricSFTTrainer(config=config, train_dataset=dataset)
trainer.train()
trainer.save_model()
```

### 4. High-Level Pipeline API

```python
from deepfabric.pipeline import DeepFabricPipeline
from deepfabric.training import SFTTrainingConfig, LoRAConfig

# Create end-to-end pipeline
pipeline = DeepFabricPipeline(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    provider="transformers",
    device="cuda",
    torch_dtype="bfloat16"
)

# Step 1: Generate dataset
dataset = pipeline.generate_dataset(
    topic_prompt="Machine Learning Fundamentals",
    num_samples=1000,
    batch_size=10,
    tree_depth=3,
    tree_degree=10,
    conversation_type="cot_structured"
)

# Step 2: Train model
training_config = SFTTrainingConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    bf16=True,
    lora=LoRAConfig(enabled=True, r=16, lora_alpha=32)
)

metrics = pipeline.train(training_config)

# Step 3: Save and upload
pipeline.save_and_upload(
    output_dir="./final-model",
    dataset_path="./dataset.jsonl",
    hf_repo="username/ml-tutor-model"
)
```

### 5. Mixed API and Local Models

```python
# Generate topics with API (fast, cheap)
tree = Tree(
    topic_prompt="Advanced Topics",
    provider="openai",
    model_name="gpt-4o-mini",
    depth=4,
    degree=15
)
tree_paths = asyncio.run(tree.build_async())

# Generate dataset with local model (more control)
generator = DataSetGenerator(
    generation_system_prompt="Create detailed training examples",
    provider="transformers",
    model_name="meta-llama/Llama-3.1-70B-Instruct",
    device="cuda",
    torch_dtype="bfloat16",
    load_in_4bit=True,  # Use quantization for large model
    conversation_type="cot_hybrid"
)

dataset = asyncio.run(generator.create_data_async(
    num_steps=200,
    batch_size=5,
    topic_model=tree
))
```

## Configuration Options

### TransformersConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | str | required | HuggingFace model ID or local path |
| `device` | str | None | Device (cuda, cpu, auto) |
| `device_map` | str/dict | None | Device mapping strategy |
| `torch_dtype` | str | "auto" | Model precision (float16, bfloat16, float32) |
| `load_in_8bit` | bool | False | Enable 8-bit quantization |
| `load_in_4bit` | bool | False | Enable 4-bit quantization |
| `trust_remote_code` | bool | False | Allow custom model code |
| `use_fast_tokenizer` | bool | True | Use fast tokenizer |
| `chat_template` | str | None | Custom chat template |
| `max_length` | int | 8192 | Maximum sequence length |

### LoRA Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | False | Enable LoRA |
| `r` | int | 16 | LoRA rank (attention dimension) |
| `lora_alpha` | int | 32 | LoRA alpha scaling parameter |
| `lora_dropout` | float | 0.05 | Dropout probability |
| `target_modules` | list/str | "all-linear" | Target modules for LoRA |
| `bias` | str | "none" | Bias training strategy |

### Quantization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `load_in_8bit` | bool | False | 8-bit quantization |
| `load_in_4bit` | bool | False | 4-bit quantization (QLoRA) |
| `bnb_4bit_compute_dtype` | str | "bfloat16" | Compute dtype for 4-bit |
| `bnb_4bit_use_double_quant` | bool | True | Double quantization |
| `bnb_4bit_quant_type` | str | "nf4" | Quantization type (nf4 or fp4) |

## Memory Optimization Tips

### For 8B Models (e.g., Llama-3.1-8B)

```python
# Full precision (requires ~32GB VRAM)
device="cuda", torch_dtype="float32"

# BF16 (requires ~16GB VRAM) - Recommended
device="cuda", torch_dtype="bfloat16"

# 8-bit quantization (requires ~8GB VRAM)
device="cuda", load_in_8bit=True

# 4-bit quantization (requires ~5GB VRAM)
device="cuda", load_in_4bit=True
```

### For 70B Models (e.g., Llama-3.1-70B)

```python
# 4-bit quantization with device mapping (requires ~40GB VRAM)
TransformersConfig(
    model_id="meta-llama/Llama-3.1-70B-Instruct",
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16"
)
```

### Training Memory Optimization

```python
SFTTrainingConfig(
    # Use gradient checkpointing
    gradient_checkpointing=True,

    # Smaller batch size with gradient accumulation
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,

    # Use bfloat16
    bf16=True,

    # Enable LoRA for parameter efficiency
    lora=LoRAConfig(enabled=True, r=16),

    # Use 4-bit quantization
    quantization=QuantizationConfig(load_in_4bit=True)
)
```

## Error Handling

The Transformers provider includes robust error handling:

- **CUDA OOM**: Automatic retry with suggestions for memory optimization
- **Model loading failures**: Clear error messages with dependency checks
- **Generation failures**: Retry logic specific to local inference

## Performance Considerations

- **Batch Processing**: Use larger batch sizes for faster generation
- **Device Placement**: Use `device="cuda"` for GPU acceleration
- **Quantization**: Balance between speed and quality
- **Model Selection**: Smaller models (7-8B) are often sufficient for data generation

## Supported Models

All HuggingFace causal language models are supported, including:

- **Llama 3/3.1**: `meta-llama/Llama-3.1-8B-Instruct`
- **Mistral**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Qwen**: `Qwen/Qwen2.5-7B-Instruct`
- **Phi**: `microsoft/Phi-3-mini-4k-instruct`
- **Gemma**: `google/gemma-2-9b-it`

## Troubleshooting

### Import Error: transformers not found

```bash
pip install 'deepfabric[training]'
```

### CUDA Out of Memory

1. Reduce batch size
2. Enable gradient checkpointing
3. Use quantization (4-bit recommended)
4. Use a smaller model

### Slow Generation

1. Use GPU acceleration (`device="cuda"`)
2. Increase batch size
3. Use bfloat16 precision
4. Consider using API-based models for topic generation

## Examples

See `docs/examples/` for complete end-to-end examples:
- `transformers_basic.py`: Basic local inference
- `transformers_training.py`: Full training pipeline
- `transformers_qlora.py`: QLoRA with large models
- `hybrid_generation.py`: Mix API and local models

## Next Steps

- Explore advanced conversation types (`cot_structured`, `agent_cot_multi_turn`)
- Experiment with different LoRA ranks and alpha values
- Try multi-GPU training with device mapping
- Upload your models to HuggingFace Hub
