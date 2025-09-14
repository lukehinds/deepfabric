<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-dark.png">
    <img alt="Deepfabric logo" src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-light.png" width="486px" height="130px" style="max-width: 100%;">
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

**Deepfabric** is a powerful Python library and CLI for generating synthetic datasets using LLMs. Suitabl for building teacher-student distillation pipelines, creating evaluation datasets for models and agents, or conducting research that requires diverse training data, Deepfabric streamlines the entire process from topic generation to dataset export. It can be used as an individual CLI tool, config driven with YAML, or imported as a library to use within your own ML pipelines"

## See It In Action

<div align="center">
  <img src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/demo.gif" alt="Deepfabric Demo" width="100%" style="max-width: 800px;">
</div>

## Quickstart

Get up and running in under 60 seconds:

### 1. Install Deepfabric

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

That's it! Deepfabric will automatically:
- Generate a hierarchical topic tree (3 levels deep, 3 branches per level)
- Create 9 diverse Q&A pairs across the generated topics
- Save your dataset to `dataset.jsonl`

### 3. Use Your Dataset

Your dataset is ready for fine-tuning in the standard instruct format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Can you explain the significance of the double-slit experiment in quantum physics?"
    },
    {
      "role": "assistant",
      "content": "The double-slit experiment is fundamental in quantum physics as it demonstrates the dual nature of light and particles, showing both wave-like and particle-like properties. When particles such as photons or electrons pass through two slits, they create an interference pattern characteristic of waves, even if sent one at a time. This result challenged classical physics and contributed significantly to the development of quantum mechanics, highlighting the probabilistic nature of quantum states."
    }
  ]
}
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
{
  "messages": [
    {
      "role": "user",
      "content": "Can you explain the role of Niels Bohr in the development of quantum physics?"
    },
    {
      "role": "assistant",
      "content": "Niels Bohr was a pivotal figure in the development of quantum physics, particularly known for introducing the Bohr model of the atom and contributing to the Copenhagen Interpretation of quantum mechanics. His work laid the foundation for understanding atomic structure and quantum theory, emphasizing the idea of complementarity and the probabilistic nature of quantum phenomena."
    }
  ]
}
{
  "messages": [
    {
      "role": "user",
      "content": "Can you tell me about the Stern-Gerlach experiment and its significance in quantum physics?"
    },
    {
      "role": "assistant",
      "content": "The Stern-Gerlach experiment, conducted by Otto Stern and Walther Gerlach in 1922, was crucial in the development of quantum mechanics. It demonstrated that particles such as neutral silver atoms have quantized spin angular momentum. In the experiment, a beam of silver atoms was passed through an inhomogeneous magnetic field, which caused the beam to split according to the spin of the atoms, thus providing evidence for the existence of quantum spin states."
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
  --model qwen3:8b \
  --depth 4 \
  --degree 5 \
  --num-steps 100 \
  --batch-size 5 \
  --topic-prompt "Machine Learning Fundamentals"
  --generation-system-prompt "You are an expert on Machine Learning and its application in modern technologies" \
  --dataset-save-as dataset.jsonl
```

There are lots of [examples](./examples/) to get you going.

## Why Deepfabric?

Deepfabric solves the challenge of creating diverse, high-quality training data at scale. It uses a novel approach of first generating
a topic tree or graph, which results in datasets with a high diversity level and miminmal duplication. Benchmarks using tools such as
great expectations shows that Deepfabric datasets fair well when compared with such as databricks/databricks-dolly-15k.

## I heard that Synthetic Data is inferior to Human curated / labelled data

That used to be the view, but no longer is, especially when it comes to model fine-tuning of SLMs. Views shifted when Deepseek first released r1,
trained largley on synthetics using a process termed 'distilation'. 

Since then many other models have followed suit, including most recently Phi-4, to quote the whitepaper "Synthetic data constitutes the bulk of the training data for phi-4".

## Key Features

### Topic Trees and Graphs

Deepfabric can generate topics using two approaches:

**Topic Trees**: Traditional hierarchical structure where each topic branches into subtopics, perfect for well-organized domains.

**Topic Graphs** (Experimental): DAG-based structure allowing cross-connections between topics, ideal for complex domains with interconnected concepts.

<img src="https://raw.githubusercontent.com/lukehinds/deepfabric/f6ac2717a99b1ae1963aedeb099ad75bb30170e8/assets/graph.svg" width="100%" height="100%"/>

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
deepfabric start config.yaml --hf-repo username/my-dataset --hf-token $HF_TOKEN
```

## Installation

### Requirements

Python 3.11 or higher

### Install from PyPI

```bash
pip install deepfabric
```

### Install from Source

```bash
git clone https://github.com/lukehinds/deepfabric.git
cd deepfabric
pip install -e .
```

### Development Setup

For contributors and developers:

```bash
# Install uv for dependency management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup development environment
git clone https://github.com/lukehinds/deepfabric.git
cd deepfabric
uv sync --all-extras
```

## Usage Guide

### Configuration-Based Approach (Recommended)

Deepfabric uses YAML configuration files for maximum flexibility. Here's a complete example:

```yaml
dataset_system_prompt: "You are a helpful assistant. You provide clear and concise answers to user questions."

topic_tree:
  topic_prompt: "Capital Cities of the World."
  topic_system_prompt: "You are a helpful assistant. You provide clear and concise answers to user questions."
  degree: 3
  depth: 2
  temperature: 0.7
  model_name: "ollama/mistral:latest"
  save_as: "basic_prompt_Tree.jsonl"

data_engine:
  instructions: "Please provide training examples with questions about capital cities."
  generation_system_prompt: "You are a helpful assistant. You provide clear and concise answers to user questions."
  model_name: "ollama/mistral:latest"
  temperature: 0.9
  max_retries: 2

dataset:
  creation:
    num_steps: 5
    batch_size: 1
    model_name: "ollama/mistral:latest"
    sys_msg: true  # Include system message in dataset (default: true)
  save_as: "basic_prompt_dataset.jsonl"

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
deepfabric start config.yaml
```

The CLI supports various options to override configuration values:

```bash
deepfabric start config.yaml \
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

Deepfabric supports all LiteLLM providers. Here are the most common:

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

**Azure OpenAI**
```yaml
provider: "azure"
model: "your-deployment-name"
# Set: export AZURE_API_KEY="your-key"
# Set: export AZURE_API_BASE="your-endpoint"
```

For a complete list of providers and models, see the [LiteLLM documentation](https://docs.litellm.ai/docs/providers/).

### Hugging Face Hub Integration

Share your datasets with the community:

```bash
# Using environment variable
export HF_TOKEN=your-token
deepfabric start config.yaml --hf-repo username/my-dataset

# Or pass token directly
deepfabric start config.yaml \
  --hf-repo username/my-dataset \
  --hf-token your-token \
  --hf-tags "gpt4" --hf-tags "chemistry"
```

Deepfabric automatically creates dataset cards with generation metadata, tags your dataset appropriately, and handles the upload process.

### Programmatic API

For advanced use cases, use Deepfabric as a Python library:

```python
from deepfabric import DataSetGenerator, Tree

tree = Tree(
    topic_prompt="Creative Writing Prompts",
    topic_system_prompt=dataset_system_prompt,
    degree=5,
    depth=4,
    temperature=0.9,
    model_name="ollama/qwen3:8b"
)

engine = DataSetGenerator(
    instructions="Generate creative writing prompts and example responses.",
    generation_system_prompt="You are a creative writing instructor providing writing prompts and example responses.",
    model_name="ollama/qwen3:8b",
    temperature=0.9,
    max_retries=2,
    sys_msg=True,  # Include system message in dataset (default: true)
)
```

### Output Format

Deepfabric generates datasets in the standard conversational format:

```json
{
  "messages": [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "User question..."},
    {"role": "assistant", "content": "Assistant response..."}
  ]
}
```

Control system message inclusion with the `sys_msg` parameter:

```yaml
dataset:
  creation:
    sys_msg: false  # Omit system messages for certain training scenarios
```

## Development

### Running Tests

```bash
make test
# or
uv run pytest
```

### Code Quality

```bash
make format  # Format with black and ruff
make lint    # Run linting checks
make security  # Security analysis with bandit
```

### Building

```bash
make build  # Clean, test, and build package
make all    # Complete workflow
```

## Documentation

Much more in the docs: https://lukehinds.github.io/DeepFabric/

