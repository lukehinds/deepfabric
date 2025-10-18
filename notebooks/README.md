# DeepFabric Jupyter Notebooks

This directory contains Jupyter notebooks for using DeepFabric in interactive environments like Google Colab, Kaggle, or local Jupyter setups.

## Available Notebooks

### 1. `cuda_dataset_and_training.ipynb` - **Complete Workflow**
Full end-to-end example including:
- ✅ Dataset generation with local LLM on CUDA
- ✅ Fine-tuning with SFT (Supervised Fine-Tuning)
- ✅ LoRA for memory-efficient training
- ✅ Model testing and inference
- ✅ HuggingFace Hub integration

**Best for:**
- Complete training pipelines
- Production workflows
- Learning the full DeepFabric workflow

**Requirements:**
- NVIDIA GPU with CUDA
- ~16GB GPU memory for 7B models (8GB for 3B models)
- 30-60 minutes to complete

---

## Platform-Specific Instructions

### Google Colab

1. **Open the notebook in Colab:**
   - Upload the `.ipynb` file to Google Drive
   - Right-click → Open with → Google Colaboratory
   - Or use: `File → Upload notebook`

2. **Enable GPU:**
   ```
   Runtime → Change runtime type → Hardware accelerator → GPU (T4)
   ```

3. **For better GPUs (Colab Pro):**
   - A100: Best performance, 40GB memory
   - V100: Good performance, 16GB memory

4. **Save models to Google Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   OUTPUT_DIR = '/content/drive/MyDrive/models/my_model'
   ```

5. **Installation tip:**
   - First cell installs dependencies automatically
   - Expect 2-3 minutes for installation

**Colab Limitations:**
- Free tier: 12 hours max session, limited GPU
- Pro: Better GPUs, longer sessions ($10/month)
- GPU quota: Monitor your usage

---

### Kaggle Notebooks

1. **Upload to Kaggle:**
   - Go to kaggle.com/code
   - Click "New Notebook" → "Upload Notebook"
   - Upload the `.ipynb` file

2. **Enable GPU:**
   ```
   Settings (right sidebar) → Accelerator → GPU T4 x2
   ```

3. **GPU quota:**
   - Free: 30 hours/week of GPU time
   - Shared between all notebooks

4. **Save outputs:**
   - Notebooks saved automatically
   - Models in `/kaggle/working/` (9GB limit)
   - Or create Kaggle Dataset for larger models

**Kaggle Benefits:**
- 2x T4 GPUs available
- Persistent dataset storage
- Easy sharing and versioning

---

### Local Jupyter

1. **Install Jupyter:**
   ```bash
   pip install jupyter notebook
   # or
   pip install jupyterlab
   ```

2. **Install DeepFabric:**
   ```bash
   pip install 'deepfabric[training]'
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

4. **Requirements:**
   - NVIDIA GPU with CUDA 11.8+
   - 16GB+ GPU memory recommended
   - PyTorch with CUDA support

---

## Memory Optimization Tips

### If you run out of GPU memory:

#### Option 1: Use a smaller model
```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Instead of 7B
```

#### Option 2: Reduce batch size
```python
BATCH_SIZE = 1  # Instead of 4
training_config = SFTTrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,  # Maintain effective batch size
)
```

#### Option 3: Use 4-bit quantization
```python
from deepfabric.training import QuantizationConfig

training_config = SFTTrainingConfig(
    quantization=QuantizationConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
)
```

#### Option 4: Generate dataset separately
```python
# First: Generate dataset only (doesn't keep model in memory)
dataset = pipeline.generate_dataset(...)
dataset.save("dataset.jsonl")

# Clear memory
del pipeline
gc.collect()
torch.cuda.empty_cache()

# Second: Load dataset and train
from deepfabric.dataset import Dataset
dataset = Dataset.from_jsonl("dataset.jsonl")
# ... train with dataset
```

---

## Common Issues

### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
1. Reduce model size (use 3B instead of 7B)
2. Reduce batch size to 1
3. Enable gradient checkpointing (already enabled in notebooks)
4. Use 4-bit quantization
5. Reduce `num_samples` for dataset generation

---

### No GPU Available
```
CUDA not available
```

**Solutions:**
- **Colab**: Runtime → Change runtime type → GPU
- **Kaggle**: Settings → Accelerator → GPU
- **Local**: Install CUDA toolkit and PyTorch with CUDA

---

### Slow Dataset Generation

Dataset generation can take time:
- **3B model**: ~10-20 seconds per sample
- **7B model**: ~20-40 seconds per sample
- **200 samples**: 1-2 hours

**Tips:**
- Start with `NUM_SAMPLES = 50` for testing
- Use `batch_size=5` for parallel generation
- Monitor GPU usage in Colab: `!nvidia-smi` (run in cell)

---

## Example: Quick Test Run

For a quick test to verify everything works:

```python
# Minimal configuration for fast testing
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
NUM_SAMPLES = 20  # Small dataset
NUM_EPOCHS = 1    # One epoch
BATCH_SIZE = 2

# Should complete in ~10-15 minutes total
```

---

## Advanced Dataset Generation

### Understanding Generation Prompts

DeepFabric uses **generation prompts** to control the quality and style of synthetic data:

```python
# Custom system prompt for better quality
GENERATION_SYSTEM_PROMPT = """You are an expert Python and ML educator.

Create training examples that:
- Cover fundamental and advanced concepts
- Include practical, real-world scenarios
- Demonstrate best practices
- Provide clear explanations with code examples"""

dataset = pipeline.generate_dataset(
    topic_prompt="Python Programming and Machine Learning",
    num_samples=200,

    # Topic tree configuration
    tree_depth=3,      # Depth of topic hierarchy
    tree_degree=5,     # Subtopics per level

    # Generation quality
    generation_system_prompt=GENERATION_SYSTEM_PROMPT,
    temperature=0.8,   # 0.0=deterministic, 2.0=very creative
    max_tokens=2000,   # Max tokens per sample

    # Conversation types
    conversation_type="cot_structured",  # Options:
    # - "basic": Simple Q&A
    # - "cot_structured": Chain-of-thought with structured reasoning
    # - "cot_freetext": Free-form chain-of-thought
    # - "agent_cot_tools": Agent with tool calling

    # Additional guidance
    instructions="Focus on practical examples with working code",
    reasoning_style="logical",  # For CoT: "mathematical", "logical", "general"
)
```

### Two Types of System Prompts

1. **`generation_system_prompt`** - Controls **data generation**
   - Guides the LLM while creating synthetic training data
   - Should specify expertise, quality requirements, focus areas
   - Example: "You are an expert Python educator..."

2. **`dataset_system_prompt`** - Controls **final dataset**
   - Included in the dataset as the system message
   - If not provided, defaults to `generation_system_prompt`
   - Use when you want different behavior during fine-tuning

```python
dataset = pipeline.generate_dataset(
    topic_prompt="...",

    # High-detail prompt for generation
    generation_system_prompt="""You are an expert data scientist creating
    comprehensive training examples covering theory, practice, and edge cases.""",

    # Simple prompt for the fine-tuned model
    dataset_system_prompt="You are a helpful Python programming assistant.",
)
```

### Few-Shot Learning with Examples

Provide example data to guide generation style:

```python
from deepfabric.dataset import Dataset

# Load or create example dataset
examples = Dataset.from_jsonl("high_quality_examples.jsonl")

dataset = pipeline.generate_dataset(
    topic_prompt="...",
    example_data=examples,  # LLM will follow this style
    # num_example_demonstrations=3,  # How many examples per generation
)
```

### Using Different Models

Use different models for topic generation vs. data generation:

```python
dataset = pipeline.generate_dataset(
    topic_prompt="...",

    # Use larger model for topic tree (runs once)
    tree_model_name="meta-llama/Llama-3.1-70B-Instruct",

    # Use smaller model for data generation (runs many times)
    generation_model_name="meta-llama/Llama-3.2-3B-Instruct",
)
```

---

## Advanced: Multi-GPU Training

If you have multiple GPUs (Kaggle's 2x T4 or local multi-GPU):

```python
# DeepFabric automatically uses all available GPUs with device_map="auto"
pipeline = DeepFabricPipeline(
    model_name=MODEL_NAME,
    provider="transformers",
    device="cuda",  # Will use device_map="auto" internally
    dtype="bfloat16",
)
```

For explicit multi-GPU control:
```python
training_config = SFTTrainingConfig(
    model_name=MODEL_NAME,
    model_kwargs={
        "device_map": "auto",  # Automatic multi-GPU
        # or
        "device_map": {"": 0},  # Force single GPU
    },
)
```
---
