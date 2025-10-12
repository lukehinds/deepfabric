<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.png">
    <img alt="DeepFabric logo" src="./assets/logo-light.png" width="400px" height="270px" style="max-width: 100%;">
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

**DeepFabric** is a powerful synthetic dataset generation framework that leverages LLMs to create high-quality, diverse training data at scale. Built for ML engineers, researchers, and AI developers, it streamlines the entire dataset creation pipeline from topic generation to model-ready formats.

No more unruly models failing to Tool call or comply with reams of natural language to try and yield structured formats. DeepFabric ensures your models are consistent, well-structured, and ready for fine-tuning or evaluation.
## Key Features

### Core Capabilities
- **Hierarchical Topic Generation**: Tree and graph-based architectures for comprehensive domain coverage
- **Multi-Format Export**: Direct export to popular training formats (no conversion scripts needed)
- **Conversation Templates**: Support for various dialogue patterns and reasoning styles
- **Tool Calling Support**: Generate function-calling and agent interaction datasets
- **Structured Output**: Pydantic & Outlines enforced schemas for consistent, high-quality data
- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, Ollama, and more
- **HuggingFace Integration**: Direct dataset upload with auto-generated cards 

## Supported Output Formats

| Format | Template | Use Case | Framework Compatibility |
|--------|----------|----------|-----------------------|
| **TRL SFT Tools** | `builtin://trl_sft_tools` | Tool calling fine-tuning | HuggingFace TRL SFTTrainer |
| **Alpaca** | `builtin://alpaca.py` | Instruction-following | Stanford Alpaca, LLaMA |
| **ChatML** | `builtin://chatml.py` | Multi-turn conversations | Most chat models |
| **Unsloth** | `builtin://unsloth.py` | Optimized fine-tuning | Unsloth notebooks |
| **GRPO** | `builtin://grpo.py` | Mathematical reasoning | GRPO training |
| **Im Format** | `builtin://im_format.py` | Chat with delimiters | ChatML-compatible models |
| **Tool Calling** | `builtin://tool_calling.py` | Function calling | Agent training |
| **Single Tool Call** | `builtin://single_tool_call.py` | Individual tool calls | Single function execution |
| **XLAM v2** | `builtin://xlam_v2` | Multi-turn tool calling | Salesforce xLAM models |
| **Harmony** | `builtin://harmony.py` | Reasoning with tags | OpenAI gpt-oss |
| **Custom** | `file://your_format.py` | Your requirements | Any framework |

### Custom Format

You can create your own custom output format by implementing a simple Python class with a `format` method using the `deepfabric` library and `BaseFormatter` class. See the [Custom Format Guide](./docs/formatters/custom-formatter-guide.md) for details.

## Conversation Templates

| Template Type | Description | Example Use Case |
|--------------|-------------|------------------|
| **Single-Turn** | Question → Answer | FAQ, classification |
| **Multi-Turn** | Extended dialogues | Chatbots, tutoring |
| **Chain of Thought (CoT)** | Step-by-step reasoning | Math, logic problems |
| **Structured CoT** | Explicit reasoning traces | Educational content |
| **Hybrid CoT** | Mixed reasoning styles | Complex problem-solving |
| **Tool Calling** | Function invocations | Agent interactions |
| **System-Prompted** | With system instructions | Role-playing, personas |

### Template Missing?

If there's a format or feature you'd like to see, please [open an issue](https://github.com/lukehinds/deepfabric/issues/new).

## DeepFabric Pipeline

DeepFabric is designed to work within a modular MLOps pipeline, allowing you to customize each stage of the dataset generation process. The main components are:

- **Topic Generation**: Create a structured topic tree or graph based on a high-level prompt.
- **Data Generation**: Generate training examples for each topic using LLMs.
- **Format Engine**: Convert raw outputs into your desired dataset format.

```mermaid
graph LR
    A[Topic Prompt] --> B[Topic Tree/Graph]
    B --> C[Data Generator]
    C --> D[Format Engine]
    D --> E[Export/Upload]
```

By decoupling these components, you can easily swap out models, prompts, and formats to suit your specific needs, along with version controlling your configurations for reproducibility.

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

### Topic Generation Modes

| Mode | Structure | Use Case | Max Topics |
|------|-----------|----------|------------|
| **Tree** | Hierarchical branching | Well-organized domains | depth^degree |
| **Graph** | DAG with cross-connections | Interconnected concepts | Flexible |
| **Linear** | Sequential topics | Simple lists | User-defined |
| **Custom** | User-provided structure | Specific requirements | Unlimited |

### Provider Support Matrix

| Provider | Models | Best For | Local/Cloud |
|----------|--------|----------|-------------|
| **OpenAI** | GPT-4, GPT-4o, GPT-3.5 | High quality, complex tasks | Cloud |
| **Anthropic** | Claude 3.5 Sonnet, Haiku | Nuanced reasoning | Cloud |
| **Google** | Gemini 2.0, 1.5 | Cost-effective at scale | Cloud |
| **Ollama** | Llama, Mistral, Qwen, etc. | Privacy, unlimited generation | Local |
| **Together** | Open models | Fast inference | Cloud |
| **Groq** | Llama, Mixtral | Ultra-fast generation | Cloud |

## Configuration System

DeepFabric uses a flexible YAML-based configuration with extensive CLI overrides:

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

## Advanced Features

### Chain of Thought (CoT) Generation

| CoT Style | Template Pattern | Best For |
|-----------|-----------------|----------|
| **Free-text** | Natural language steps | Mathematical problems (GSM8K-style) |
| **Structured** | Explicit reasoning traces | Educational content, tutoring |
| **Hybrid** | Mixed reasoning | Complex multi-step problems |

```yaml
# Example: Structured CoT configuration
data_engine:
  conversation_template: "cot_structured"
  cot_style: "mathematical"
  include_reasoning_tags: true
```

### Quality Control Features

- **Deduplication**: Automatic removal of similar samples
- **Validation**: Schema enforcement for all outputs
- **Rate Limiting**: Provider-aware retry with exponential backoff and jitter ([docs](./docs/rate-limiting.md))
- **Progress Monitoring**: Real-time generation statistics

## 📖 Documentation & Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Documentation** | Complete API reference & guides | [docs](https://lukehinds.github.io/deepfabric/) |
| **Examples** | Ready-to-use configurations | [examples/](./examples/README.md) |
| **Discord** | Community support | [Join Discord](https://discord.gg/pPcjYzGvbS) |
| **Issues** | Bug reports & features | [GitHub Issues](https://github.com/lukehinds/deepfabric/issues) |

## Stay Updated

Deepfabric development is moving at a fast pace 🏃‍♂️, for a great way to follow the project and to be instantly notified of new releases, **Star the repo**.

<img src="/assets/star.gif" width="40%" height="40%"/>

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/lukehinds/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22) to get started.

### Development Setup
```bash
git clone https://github.com/lukehinds/deepfabric
cd deepfabric
uv sync --all-extras  # Install with dev dependencies
make test            # Run tests
make format          # Format code
```

## Community & Support

- **Discord**: [Join our community](https://discord.gg/pPcjYzGvbS) for real-time help
- **Issues**: [Report bugs](https://github.com/lukehinds/deepfabric/issues) or request features
- **Discussions**: Share your use cases and datasets

### Who's Using DeepFabric?

If you're using DeepFabric in production or research, we'd love to hear from you! Share your experience in our [Discord](https://discord.gg/pPcjYzGvbS) or open a discussion.

## Use Cases

### Industry Applications
| Use Case | Description | Example Config |
|----------|-------------|----------------|
| **Model Distillation** | Teacher-student training | [distillation.yaml](examples/specialized.yaml) |
| **Evaluation Benchmarks** | Model testing datasets | [benchmark.yaml](examples/advanced.yaml) |
| **Domain Adaptation** | Specialized knowledge | [domain.yaml](examples/specialized.yaml) |
| **Agent Training** | Tool-use & reasoning | [agent.yaml](examples/agent_tool_calling.yaml) |
| **Instruction Tuning** | Task-specific models | [instruct.yaml](examples/unsloth_instruct_config.yaml) |
| **Math Reasoning** | Step-by-step solutions | [math.yaml](examples/grpo_math_config.yaml) |


## Tips for Best Results

1. **Start Small**: Test with `depth=2, degree=3` before scaling up
2. **Mix Models**: Use stronger models for topics, faster ones for generation
3. **Iterate**: Generate small batches and refine prompts based on results
4. **Validate**: Always review a sample before training
5. **Version Control**: Save configurations for reproducibility

### Analytics

We use privacy-respecting analytics to help us improve application performance and stability. We never send Personal identifiable information and we do not capture prompts, generated content, API keys, file names etc.

#### What We Collect
- **Anonymous User ID**: A stable, one-way hash based on your machine characteristics (hostname + MAC address). This helps us understand unique user counts without identifying you. Its impossible to reverse this hash to get your actual machine details and one-way only.
- **Usage Metrics**: Model names, numeric parameters (temperature, depth, degree, batch_size), timing and success/failure rates
- **Developer Flag**: If you set `DEEPFABRIC_DEVELOPER=True`, events are marked to help us filter developer testing from real usage

#### Privacy Guarantees
- No usernames, emails, IP addresses, or personal information
- User ID is cryptographically hashed and cannot be reversed and contains no Personal Identifiable Information
- No prompts, generated datasets, or sensitive data is collected
- All data is used solely for application improvement in regards to performance, stability, and feature usage

#### Control Your Participation
```bash
# Disable all analytics
export ANONYMIZED_TELEMETRY=False

# Mark yourself as a developer (for filtering)
export DEEPFABRIC_DEVELOPER=True
```
