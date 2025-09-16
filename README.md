<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-dark.png">
    <img alt="DeepFabric logo" src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-light.png" width="486px" height="150spx" style="max-width: 100%;">
  </picture>
  <h3>Generate High-Quality Synthetic Datasets at Scale</h3>

  <!-- CTA Buttons -->
  <p>
    <a href="https://github.com/lukehinds/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
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
    <a href="https://github.com/lukehinds/deepfabric/actions/workflows/test.yml">
      <img src="https://github.com/lukehinds/deepfabric/actions/workflows/test.yml/badge.svg" alt="CI Status"/>
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
  <br/>
</div>

**DeepFabric** is a CLI tool and SDK, that leverages large language models to generate high-quality synthetic datasets. It's designed for researchers and developers building teacher-student distillation pipelines, creating evaluation benchmarks for models and agents, or conducting research requiring diverse training data.

The key innovation lies in DeepFabric's graph and tree-based architecture, which uses structured topic nodes as generation seeds. This approach ensures the creation of datasets that are both highly diverse and domain-specific, while minimizing redundancy and duplication across generated samples.

<div align="center">
  <img src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/demo.gif" alt="DeepFabric Demo" width="80%" style="max-width: 700px;">
</div>

## Quickstart

### 1. Install DeepFabric

```bash
pip install deepfabric
```

### 2. Generate Your First Dataset

```bash
# Set your API key (or use Ollama for local generation)
export OPENAI_API_KEY="your-api-key"

# Generate a dataset with a single command
deepfabric generate \
  --mode tree \
  --provider openai \
  --model gpt-4o \
  --depth 3 \
  --degree 3 \
  --num-steps 9 \
  --batch-size 1 \
  --topic-prompt "This history Quantum physics" \
  --generation-system-prompt "You are an expert on academic history, with a specialism in the sciences" \
  --dataset-save-as dataset.jsonl
```

Deepfabric will automatically:
- Generate a hierarchical topic tree (3 levels deep, 3 branches per level)
- Create 9 diverse Q&A pairs across the generated topics
- Save your dataset to `dataset.jsonl`

### 3. Use Your Dataset

Your dataset is ready in the OpenAI standard instruct format (JSONL):

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Can you explain Albert Einstein's contribution to quantum theory?"
    },
    {
      "role": "assistant",
      "content": "Albert Einstein made significant contributions to quantum theory, particularly through his explanation of the photoelectric effect, for which he won the Nobel Prize in 1921. He proposed that light could be thought of as discrete packets of energy called quanta or photons, which could explain how electrons are emitted from metals when exposed to light. This idea was instrumental in the development of quantum mechanics. He later became famous for his skepticism about quantum mechanics probabilistic interpretation, leading to his quote \"God does not play dice with the universe.\""
    }
  ]
}
```

### 4. Use local models.

Generate larger datasets with different models:

```bash
# With a depth of 4 and degree of 4^5 = 1,024
deepfabric generate \
  --provider ollama \
  --model qwen3:32b \
  --depth 4 \
  --degree 5 \
  --num-steps 100 \
  --batch-size 5 \
  --topic-prompt "Machine Learning Fundamentals"
  --generation-system-prompt "You are an expert on Machine Learning and its application in modern technologies" \
  --dataset-save-as dataset.jsonl
```

There are lots more [examples](./examples/README.md) to get you going.

## Key Features

### Topic Trees and Graphs

DeepFabric can generate topics using two approaches:

**Topic Graphs** (Experimental): DAG-based structure allowing cross-connections between topics, ideal for complex domains with interconnected concepts.

**Topic Trees**: Traditional hierarchical structure where each topic branches into subtopics, perfect for well-organized domains.

### Multi-Provider Support

Leverage different LLMs for different tasks. Use GPT-4 for complex topic generation, then switch to a local model like Mixtral for bulk data creation:

```yaml
topic_tree:
  provider: "openai"
  model: "gpt-4"  # High quality for topic structure

data_engine:
  provider: "ollama"
  model: "mixtral"  # Fast and efficient for bulk generation
```

### Automatic Dataset Upload

Push your datasets directly to Hugging Face Hub with automatic dataset cards:

```bash
deepfabric generate config.yaml --hf-repo username/my-dataset --hf-token $HF_TOKEN
```

### Configuration-Based Approach (Recommended)

DeepFabric uses YAML configuration files for maximum flexibility. Here's a complete example:

```yaml
# Main system prompt - used as fallback throughout the pipeline
dataset_system_prompt: "You are a helpful AI assistant providing clear, educational responses."

# Topic Tree Configuration
# Generates a hierarchical topic structure using tree generation
topic_tree:
  topic_prompt: "Python programming fundamentals and best practices"

  # LLM Settings
  provider: "ollama"                    # Options: openai, anthropic, gemini, ollama
  model: "qwen3:0.6b"                    # Change to your preferred model
  temperature: 0.7                      # 0.0 = deterministic, 1.0 = creative

  # Tree Structure
  degree: 2                             # Number of subtopics per node (1-10)
  depth: 2                              # Depth of the tree (1-5)

  # Topic generation prompt (optional - uses dataset_system_prompt if not specified)
  topic_system_prompt: "You are a curriculum designer creating comprehensive programming learning paths. Focus on practical concepts that beginners need to master."

  # Output
  save_as: "python_topics_tree.jsonl"  # Where to save the generated topic tree

# Data Engine Configuration
# Generates the actual training examples
data_engine:
  instructions: "Create clear programming tutorials with working code examples and explanations"

  # LLM Settings (can override main provider/model)
  provider: "ollama"
  model: "qwen3:0.6b"
  temperature: 0.3                      # Lower temperature for more consistent code
  max_retries: 3                        # Number of retries for failed generations

  # Content generation prompt
  generation_system_prompt: "You are a Python programming instructor creating educational content. Provide working code examples, clear explanations, and practical applications."

# Dataset Assembly Configuration
# Controls how the final dataset is created and formatted
dataset:
  creation:
    num_steps: 4                        # Number of training examples to generate
    batch_size: 1                       # Process 3 examples at a time
    sys_msg: true                       # Include system messages in output format

  # Output
  save_as: "python_programming_dataset.jsonl"

# Optional Hugging Face Hub configuration
huggingface:
  # Repository in format "username/dataset-name"
  repository: "your-username/your-dataset-name"
  # Token can also be provided via HF_TOKEN environment variable or --hf-token CLI option
  token: "your-hf-token"
  # Additional tags for the dataset (optional)
  # "deepfabric" and "synthetic" tags are added automatically
  tags:
    - "deepfabric-generated-dataset"
    - "geography"
```

Run using the CLI:

```bash
deepfabric generate config.yaml
```

The CLI supports various options to override configuration values:

```bash
deepfabric generate config.yaml \
  --save-tree output_tree.jsonl \
  --dataset-save-as output_dataset.jsonl \
  --model-name ollama/qwen3:8b \
  --temperature 0.8 \
  --degree 4 \
  --depth 3 \
  --num-steps 10 \
  --batch-size 2 \
  --sys-msg true \  # Control system message inclusion (default: true)
  --hf-repo username/dataset-name \
  --hf-token your-token \
  --hf-tags tag1 --hf-tags tag2
```

### Supported Providers

DeepFabric supports serveral providers. Here are the most common:

**OpenAI**
```yaml
provider: "openai"
model: "gpt-4-turbo-preview"
# Set: export OPENAI_API_KEY="your-key"
```

**Anthropic**
```yaml
provider: "anthropic"
model: "claude-3-opus-20240229"
# Set: export ANTHROPIC_API_KEY="your-key"
```

**Google**
```yaml
provider: "gemini"
model: "gemini-pro"
# Set: export GEMINI_API_KEY="your-key"
```

**Ollama (Local)**
```yaml
provider: "ollama"
model: "qwen3:8b:latest"
# No API key needed
```


## Docs / Examples

For more details, including how to use the SDK, see the [docs!](https://lukehinds.github.io/DeepFabric/)

There are also lots of [examples](./examples/README.md) to get you going.

## Stay Updated

Deepfabric development is moving at a fast pace 🏃‍♂️, for a great way to follow the project and to be instantly notified of new releases, **Star the repo**.

<img src="/assets/star.gif" width="40%" height="40%"/>

## Roadmap

Deepfabric currently outputs to Open AI chat format, we will provide a system where you can easily plug in a post-processing conversion to whatever format is needed. This should allow easy adaption to what ever you need within a training pipeline:

```yaml
formatters:
- name: "alpaca"
  template: "builtin://alpaca.py"
- name: "custom"
  template: "file://./my_format.py"
  config:
    instruction_field: "query"
```

### More Conversations Types

We will be introducing, multi-turn, reasoning, chain-of-thought etc.

### Kaggel Support

Push to Kaggel

## Analytics

We use fully anonymised analytics, to help us improve application performance and stablity. **We never** send Personal identifiable information and **we do not** capture prompts, generated content, API keys etc. We capture model names, numeric parameters (temperature, depth, degree, batch_size), timing and success/failure rates - this then helps us find optimizations or bottlenecks.

Should you wish to opt-out, just set `ANONYMIZED_TELEMETRY=False`.
