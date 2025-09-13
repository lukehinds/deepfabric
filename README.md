<div align="center">
  <h1>DeepFabric</h1>
  <h3>Generate High-Quality Synthetic Datasets at Scale</h3>

  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-dark.png">
    <img alt="Deepfabric logo" src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-light.png" width="486px" height="130px" style="max-width: 100%;">
  </picture>

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

**DeepFabric** is a powerful Python library and CLI for generating synthetic datasets using LLMs. Whether you're building teacher-student distillation pipelines, creating evaluation datasets for models and agents, or conducting research that requires diverse training data, DeepFabric streamlines the entire process from topic generation to dataset export, with an option to push direct to Hugging Face.

## See It In Action

<div align="center">
  <img src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/demo.gif" alt="DeepFabric Demo" width="100%" style="max-width: 800px;">
</div>

## Quickstart

Get up and running in under 60 seconds:

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
  --provider openai \
  --model gpt-4o \
  --tree-depth 3 \
  --tree-degree 3 \
  --num-steps 9 \
  --batch-size 1 \
  --topic-prompt "Python Programming Best Practices"
```

That's it! DeepFabric will automatically:
- Generate a hierarchical topic tree (3 levels deep, 3 branches per level)
- Create 9 diverse Q&A pairs across the generated topics
- Save your dataset to `dataset.jsonl`

### 3. Use Your Dataset

Your dataset is ready for fine-tuning in the standard format:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful AI assistant..."},
    {"role": "user", "content": "What is a Python decorator?"},
    {"role": "assistant", "content": "A decorator is a design pattern..."}
  ]
}
```

### 4. Want More Control?

Generate larger datasets with different models:

```bash
# Use Ollama for local generation
deepfabric generate \
  --provider ollama \
  --model llama3 \
  --tree-depth 4 \
  --tree-degree 5 \
  --num-steps 100 \
  --batch-size 5 \
  --topic-prompt "Machine Learning Fundamentals"

# Or use a config file for complex setups
deepfabric start config.yaml
```
Or take a look at [examples](./examples/).

## Why DeepFabric?

DeepFabric solves the challenge of creating diverse, high-quality training data at scale

## I heard that Synthetic Data is inferior to Human curated / labelled data

That used to be the view, but no longer is, well especially when it comes to model training. Views changed when Deepseek first released,
where the Deepseek whitepaper outlined how the model had been exclusively trained with synthetics using a process termed 'distilation'. 

Since then many other models have followed suit, including most recently Phi-4.

**Hierarchical Topic Generation**: Automatically explores related subtopics from a single root prompt, ensuring comprehensive coverage of your domain, with high diversity, no duplication - without losing domain specificity.

**Provider Agnostic**: Works seamlessly with OpenAI, Anthropic, Google, Ollama, and any LiteLLM-supported provider. Mix and match models for different stages of generation.

**Export Anywhere**: Direct integration with Hugging Face Hub, or export to JSONL for use with any training framework.

## Key Features

### Topic Trees and Graphs

DeepFabric can generate topics using two approaches:

**Topic Trees**: Traditional hierarchical structure where each topic branches into subtopics, perfect for well-organized domains.

**Topic Graphs** (Experimental): DAG-based structure allowing cross-connections between topics, ideal for complex domains with interconnected concepts.

<img src="https://raw.githubusercontent.com/lukehinds/deepfabric/f6ac2717a99b1ae1963aedeb099ad75bb30170e8/assets/graph.svg" width="100%" height="100%"/>

Example graph configuration:

```yaml
topic_generator: graph  # Use 'tree' for traditional hierarchy

topic_graph:
  root_prompt: "Modern Software Architecture"
  provider: "ollama"
  model: "llama3"
  graph_degree: 3    # Subtopics per node
  graph_depth: 3     # Graph depth
  save_as: "architecture_graph.json"
```

### Multi-Provider Support

Leverage different LLMs for different tasks. Use GPT-4 for complex topic generation, then switch to a faster model like Mixtral for bulk data creation:

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

DeepFabric uses YAML configuration files for maximum flexibility. Here's a complete example:

```yaml
system_prompt: "You are a helpful assistant. You provide clear and concise answers to user questions."

topic_tree:
  root_prompt: "Capital Cities of the World."
  model_system_prompt: "<system_prompt_placeholder>"
  tree_degree: 3
  tree_depth: 2
  temperature: 0.7
  model_name: "ollama/mistral:latest"
  save_as: "basic_prompt_Tree.jsonl"

data_engine:
  instructions: "Please provide training examples with questions about capital cities."
  system_prompt: "<system_prompt_placeholder>"
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
  --model-name ollama/llama3 \
  --temperature 0.8 \
  --tree-degree 4 \
  --tree-depth 3 \
  --num-steps 10 \
  --batch-size 2 \
  --sys-msg true \  # Control system message inclusion (default: true)
  --hf-repo username/dataset-name \
  --hf-token your-token \
  --hf-tags tag1 --hf-tags tag2
```

### Supported Providers

DeepFabric supports all LiteLLM providers. Here are the most common:

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
model: "llama3:latest"
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

DeepFabric automatically creates dataset cards with generation metadata, tags your dataset appropriately, and handles the upload process.

### Programmatic API

For advanced use cases, use DeepFabric as a Python library:

```python
from deepfabric import DataSetGenerator, Tree

tree = Tree(
    root_prompt="Creative Writing Prompts",
    model_system_prompt=system_prompt,
    tree_degree=5,
    tree_depth=4,
    temperature=0.9,
    model_name="ollama/llama3"
)

engine = DataSetGenerator(
    instructions="Generate creative writing prompts and example responses.",
    system_prompt="You are a creative writing instructor providing writing prompts and example responses.",
    model_name="ollama/llama3",
    temperature=0.9,
    max_retries=2,
    sys_msg=True,  # Include system message in dataset (default: true)
)
```

## Advanced Examples

### Multi-Stage Generation Pipeline

Combine different models for optimal results:

```yaml
# Stage 1: High-quality topic generation with GPT-4
topic_tree:
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  tree_depth: 4

# Stage 2: Fast bulk generation with Mixtral
data_engine:
  provider: "ollama"
  model: "mixtral"
  temperature: 0.9

# Stage 3: Final dataset with efficient model
dataset:
  creation:
    provider: "openai"
    model: "gpt-3.5-turbo"
    num_steps: 100
    batch_size: 5
```

### Domain-Specific Datasets

Create specialized datasets for any domain:

```yaml
# Medical Q&A Dataset
system_prompt: "You are a medical professional providing accurate health information."
topic_tree:
  root_prompt: "Common Medical Conditions and Treatments"
  tree_degree: 5
  tree_depth: 3

# Code Generation Dataset
system_prompt: "You are an expert programmer."
topic_tree:
  root_prompt: "Data Structures and Algorithms in Python"
  tree_degree: 4
  tree_depth: 3

# Creative Writing Dataset
system_prompt: "You are a creative writing instructor."
topic_tree:
  root_prompt: "Science Fiction Story Elements"
  tree_degree: 6
  tree_depth: 2
```

### Output Format

DeepFabric generates datasets in the standard conversational format:

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

## Best Practices

**Start Small**: Begin with a small `num_steps` value to test your configuration before scaling up.

**Temperature Tuning**: Use lower temperatures (0.3-0.7) for factual content, higher (0.7-1.0) for creative tasks.

**Iterative Refinement**: Review initial outputs and adjust your prompts and instructions accordingly.

**Model Selection**: Use more capable models for topic generation, then switch to faster models for bulk data creation.

**Validation**: Always validate a sample of your dataset before using it for training.

## Community and Support

**Discord**: Join our community for discussions and support: [discord.gg/pPcjYzGvbS](https://discord.gg/pPcjYzGvbS)

**Issues**: Report bugs or request features: [GitHub Issues](https://github.com/lukehinds/deepfabric/issues)

**Contributing**: We welcome contributions! Check out our [good first issues](https://github.com/lukehinds/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.

## License

Apache 2.0 - See [LICENSE](LICENSE) for details.

