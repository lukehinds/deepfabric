<div align="center">
  <h1>DeepFabric</h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.png" />
    <img alt="DeepFabric logo" src="./assets/logo-light.png" style="width:40%;max-width:40%;height:auto;display:block;margin:0 auto;" />
  </picture>
  <h3>Training Model Behavior in Agentic Systems</h3>

  <!-- CTA Buttons -->
  <p>
    <a href="https://github.com/always-further/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
      <img src="https://img.shields.io/badge/Contribute-Good%20First%20Issues-green?style=for-the-badge&logo=github" alt="Good First Issues"/>
    </a>
    &nbsp;
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/badge/Chat-Join%20Discord-7289da?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord"/>
    </a>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
    </a>
    <a href="https://github.com/always-further/deepfabric/actions/workflows/test.yml">
      <img src="https://github.com/always-further/deepfabric/actions/workflows/test.yml/badge.svg" alt="CI Status"/>
    </a>
    <a href="https://pypi.org/project/deepfabric/">
      <img src="https://img.shields.io/pypi/v/deepfabric.svg" alt="PyPI Version"/>
    </a>
    <a href="https://pepy.tech/project/deepfabric">
      <img src="https://static.pepy.tech/badge/deepfabric" alt="Downloads"/>
    </a>
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/discord/1384081906773131274?color=7289da&label=Discord&logo=discord&logoColor=white" alt="Discord"/>
    </a>
  </p>
</div>

**DeepFabric** generates synthetic training data for language models. By combining reasoning traces with tool-calling patterns, it creates high-quality, domain-specific datasets that teach models to think, plan, and act effectively, call tools correctly, and conform to strict schema structures.

What sets DeepFabric apart from other dataset generation tools is its ability to ensure high diversity yet domain-anchored relevance through unique topic graph generation algorithms. This guides sample creation to cover all necessary subtopics while avoiding redundancy, which is where other tools often fall short, resulting in model overfit.

<img src="/assets/df-demo.gif" width="100%" height="100%"/>

Constrained decoding and response validation ensure that generated samples strictly adhere to desired formats, ensuring datasets have exact syntax and structure for use in model training pipelines.

Once your dataset is generated, it can be automatically uploaded to Hugging Face for easy sharing and versioning, and directly imported into popular training frameworks like TRL, Unsloth, and Axolotl. Post-training, DeepFabric's built-in evaluation engine assesses model performance, whereby models prove their capabilities on unseen tasks derived from training splits—covering evaluation-only questions, answers, and tool traces.

## Quickstart

DeepFabric can be used in several ways, as a library, CLI tool, or via YAML configuration. Here's a quick example using the CLI:

```bash
pip install deepfabric
```

```bash
export OPENAI_API_KEY="your-api-key"

deepfabric generate \
  --topic-prompt "Python programming fundamentals" \
  --generation-system-prompt "You are a Python expert" \
  --mode tree \
  --depth 3 \
  --degree 3 \
  --num-samples 9 \
  --batch-size 3 \
  --provider openai \
  --model gpt-4o \
  --output-save-as dataset.jsonl
```

This generates a topic tree and creates 27 unique leaf topics, then generates 27 training samples saved to `dataset.jsonl`, giving you 100% topic coverage.

## Configuration

DeepFabric also uses YAML configuration with three main sections and optional shared LLM defaults:

```yaml
# Optional: Shared LLM defaults (inherited by topics and generation)
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7

# TOPICS: Generate the topic tree/graph
topics:
  prompt: "Building production-ready REST APIs with Python"
  mode: tree                    # tree | graph
  depth: 3
  degree: 3
  save_as: "topics.jsonl"
  # Optional: Override shared LLM settings
  llm:
    model: "gpt-4o-mini"        # Use cheaper model for topics

# GENERATION: Create training samples from topics
generation:
  system_prompt: |
    You are an expert Python backend developer and technical educator.
    Create practical, production-ready code examples with clear explanations.
    Include error handling, type hints, and follow PEP 8 conventions.

  # Additional instructions for sample generation
  instructions: |
    Focus on real-world scenarios developers encounter daily.
    Include both happy path and edge case handling.
    Provide context on when and why to use specific patterns.

  conversation:
    type: chain_of_thought      # basic | chain_of_thought
    reasoning_style: agent      # freetext | agent (for chain_of_thought)
    agent_mode: single_turn     # single_turn | multi_turn (for agent)
  
  # Tool configuration (required for agent modes)
  tools:
    registry_path: "dev-tools.yaml"  # Path to tool definitions
    # available: []             # Specific tools to use (empty = all)
    # custom: []                # Inline tool definitions
    max_per_query: 3            # Maximum tools per query
    strict: true                # Discard samples exceeding max (vs truncate)

    max_retries: 3                # Retries for failed generations
    sample_retries: 2             # Retries for validation failures
    max_tokens: 2000              # Max tokens per generation

  # Optional: Override shared LLM settings
  llm:
    temperature: 0.3            # Lower temp for consistent code

# OUTPUT: Final dataset configuration
output:
  # System prompt that goes INTO the training data
  # This is what the trained model will see as its system message
  system_prompt: |
    You are a helpful Python programming assistant specialized in REST API
    development. You provide clear, production-ready code with explanations.
    Always consider security, error handling, and best practices.

  include_system_message: true  # Whether to include system message in output
  num_samples: 4                 # Total training samples to generate
  batch_size: 3                 # Parallel generation batch size
  save_as: "api-dataset.jsonl"

# Optional: Upload to Hugging Face
huggingface:
  repository: "your-username/api-dataset-training-name"
  tags: ["python", "programming"]
```

Within `dev-tools.yaml`, define tools the model can use during generation, for example:

```yaml
- name: get_commit
  description: "Get details for a commit from a GitHub repository"
  parameters:
    - name: owner
      type: str
      description: "Repository owner"
      required: true
    - name: repo
      type: str
      description: "Repository name"
      required: true
    - name: sha
      type: str
      description: "Commit SHA, branch name, or tag name"
      required: true
    - name: include_diff
      type: bool
      description: "Whether to include file diffs and stats"
      required: false
  returns: "Commit details including author, message, and changed files"

- name: "search_file"
  description: "Search for a keyword in a text file"
  parameters:
    - name: file_path
      type: str
      description: "Path to the text file"
      required: true
    - name: keyword
      type: str
      description: "Keyword to search for"
```

Run with:

```bash
deepfabric generate config.yaml
```

## Generate, Train, Evaluate

DeepFabric returns standard HuggingFace datasets, making it easy to integrate with any training framework.

### 1. Generate Dataset

```bash
deepfabric generate config.yaml --output-save-as dataset.jsonl
```

Or upload to HuggingFace Hub:

```bash
deepfabric upload dataset.jsonl --repo your-username/my-dataset
```

### 2. Load and Split for Training

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load from Hub
dataset = load_dataset("alwaysfurther/deepfabric-generic-tools", split="train")

# Split into train/eval
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds = splits["test"]

# Format using your tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def format_example(example):
    messages = [{k: v for k, v in msg.items() if v is not None}
                for msg in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted_train = train_ds.map(format_example)
```

### 3. Train with TRL or Unsloth

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted_train,
    args=SFTConfig(output_dir="./output", num_train_epochs=3),
)
trainer.train()
```

### 4. Evaluate Your Model

```python
from deepfabric.evaluation import Evaluator, EvaluatorConfig, InferenceConfig

config = EvaluatorConfig(
    inference_config=InferenceConfig(
        model_path="./output/checkpoint-final",  # Local path or HF Hub ID
        backend="transformers",
    ),
)

evaluator = Evaluator(config)
results = evaluator.evaluate(dataset=eval_ds)  # Pass HF Dataset directly

print(f"Tool Selection Accuracy: {results.metrics.tool_selection_accuracy:.2%}")
print(f"Parameter Accuracy: {results.metrics.parameter_accuracy:.2%}")
print(f"Overall Score: {results.metrics.overall_score:.2%}")
```

## Providers

| Provider | Local/Cloud | Best For |
|----------|-------------|----------|
| OpenAI | Cloud | High quality, complex tasks |
| Anthropic | Cloud | Nuanced reasoning |
| Google Gemini | Cloud | Cost-effective at scale |
| Ollama | Local | Privacy, unlimited generation |
| OpenRouter | Cloud | Flexible model choice |

## Resources

- [Documentation](https://always-further.github.io/deepfabric/)
- [Examples](./examples/README.md)
- [Discord](https://discord.gg/pPcjYzGvbS)
- [Issues](https://github.com/always-further/deepfabric/issues)

## Development

```bash
git clone https://github.com/always-further/deepfabric
cd deepfabric
uv sync --all-extras
make test
```

## Analytics

We collect anonymous usage metrics to improve DeepFabric. No personal data, prompts, or API keys are collected.

```bash
# Disable analytics
export ANONYMIZED_TELEMETRY=False
```
