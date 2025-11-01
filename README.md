<div align="center">
  <h1>DeepFabric</h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.png" />
    <img alt="DeepFabric logo" src="./assets/logo-light.png" style="width:70%;max-width:70%;height:auto;display:block;margin:0 auto;" />
  </picture>
  <h3>Training Model Behavior in Agentic Systems</h3>

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

  <div align="center">
   <table>
    <tr>
      <td align="center">
        <strong>🚀 Active Development</strong>
        <br/>
        <sub>🏃‍♂️ We are moving fast, things may break!</sub>
        <br/>
      </td>
    </tr>
  </table>
</div>
</div>

**DeepFabric** is a specialised dataset generation and model fine-tuning framework designed for training small language models (SLMs) to become capable agents. By combining reasoning traces with tool calling patterns, and enforcement of type based structured outputs - DeepFabric enables you to fine-tune models that make intelligent decisions, select appropriate tools, and execute multi-step workflows efficiently and with accuracy.

Built by engineers behind production-grade open source projects used by **NVIDIA, Google, GitHub, and more.**

DeepFabric plans to streamline the entire agent training pipeline and drag AI/ML development into the modern age: By allow the quick production of training ready structured reasoning templates to multi architecture model-ready formats across all major training frameworks. Whether you're building MCP-compatible agents, distilling capabilities into smaller models, or creating specialized tool-calling systems, DeepFabric provides the high-quality, diverse training data you need at scale.

Deepfabric development is moving at a fast pace 🏃‍♂️, for a great way to follow the project and to be instantly notified of new releases, **Star the repo**.

<img src="/assets/star.gif" width="40%" height="40%"/>

## Generation to Training

DeepFabric datasets are ready for immediate training, multi-format capable with no conversion scripts needed, no preprocessing pipelines, just generate and train:

**Supervised Fine-Tuning (SFT)**: Drop DeepFabric datasets directly into HuggingFace TRL's `SFTTrainer` for tool-calling and conversational agent training. The `builtin://trl_sft_tools` formatter outputs the exact structure TRL expects, including function schemas and multi-turn tool interactions.

```python
from trl import SFTTrainer
from datasets import load_dataset

# Load your DeepFabric dataset
dataset = load_dataset("json", data_files="your_deepfabric_dataset.jsonl")

# Train directly with TRL - no preprocessing needed
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    # ... your training config
)
trainer.train()
```

Or if your dataset is already in a different template style, simply re-format on the fly direct within your training notebook.

```python
from trl import SFTTrainer
from datasets import load_dataset

!deepfabric format --repo "org/deepfabric-dataset" -f trl_sft_tools -o dataset.jsonl
dataset = load_dataset("json", data_files="dataset.jsonl", split="train")

# Train directly with TRL - no preprocessing needed
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    # ... your training config
)
trainer.train()
```

**Reinforcement Learning from Process Supervision (GRPO)**: Train models to generate step-by-step reasoning with the `builtin://grpo` formatter. Ideal for mathematical reasoning, complex problem-solving, and transparent decision-making where each reasoning step can be verified and reinforced.

```python
# GRPO-formatted dataset with explicit reasoning traces
dataset = load_dataset("json", data_files="math_reasoning_grpo.jsonl")

# Each example contains structured reasoning steps for RL optimization
# Perfect for training models that show their work
```

**Multi-Framework Support**: The same dataset generation can target Unsloth, Axolotl, or custom frameworks through DeepFabric's formatter system—generate once, experiment with multiple training approaches without regenerating data.

## Key Features

### Core Capabilities
- **Hierarchical Topic Generation**: Tree and graph-based architectures for comprehensive domain coverage
- **Multi-Format Export**: Direct export to popular training formats (no conversion scripts needed)
- **Conversation Templates**: Support for various dialogue patterns and reasoning styles
- **Tool Calling Support**: Generate function-calling and agent interaction datasets
- **Structured Output**: Pydantic & Outlines enforced schemas for consistent, high-quality data
- **Multi-Provider Support**: Works with OpenAI, Anthropic, Google, Ollama, OpenRouter and more
- **HuggingFace Integration**: Direct dataset upload with auto-generated cards

## Why Train SLMs for Agents?

Training smaller, specialized models for agentic tasks offers distinct advantages over relying on large API-based models:

- **Cost Efficiency**: Deploy fine-tuned agents locally instead of paying per API call, dramatically reducing operational costs
- **Privacy & Control**: Keep sensitive data and agent reasoning entirely within your infrastructure
- **Specialized Behavior**: Train models to follow your exact tool-calling patterns, reasoning styles, and domain expertise
- **Format Flexibility**: Generate once, train everywhere—DeepFabric's formatters support TRL, Unsloth, Axolotl, and custom training frameworks
- **MCP Ecosystem Ready**: Create training datasets optimized for Model Context Protocol servers and standardized tool interfaces

DeepFabric's structured approach ensures your training data teaches not just *what* tools to call, but *why* they're selected and *how* to construct parameters—the reasoning traces that transform models into reliable agents.

## Supported Output Formats

| Format | Template | Use Case | Framework Compatibility |
|--------|----------|----------|-----------------------|
| **TRL SFT Tools** | `builtin://trl_sft_tools` | Tool calling fine-tuning | HuggingFace TRL SFTTrainer |
| **XLAM v2** | `builtin://xlam_v2` | Multi-turn tool calling | Salesforce xLAM models |
| **Tool Calling** | `builtin://tool_calling.py` | Function calling | Agent training |
| **Single Tool Call** | `builtin://single_tool_call.py` | Individual tool calls | Single function execution |
| **GRPO** | `builtin://grpo.py` | Mathematical reasoning | GRPO training |
| **Harmony** | `builtin://harmony.py` | Reasoning with tags | OpenAI gpt-oss |
| **Conversations** | `builtin://conversations.py` | Generic conversations format | Unsloth, Axolotl, HF TRL |
| **ChatML** | `builtin://chatml.py` | Multi-turn conversations | Most chat models |
| **Alpaca** | `builtin://alpaca.py` | Instruction-following | Stanford Alpaca, LLaMA |
| **Custom** | `file://your_format.py` | Your requirements | Any framework |

### Custom Format

You can create your own custom output format by implementing a simple Python class with a `format` method using the `deepfabric` library and `BaseFormatter` class. See the [Custom Format Guide](./docs/formatters/custom-formatter-guide.md) for details.

## Conversation Templates

DeepFabric's conversation templates determine how your training data structures reasoning and tool interaction. For agent training, combining reasoning templates (CoT variants) with tool calling creates datasets that teach both *decision-making* and *execution*—the foundation of capable agents.

| Template Type | Description | Agent Training Value |
|--------------|-------------|---------------------|
| **Tool Calling** | Function invocations with reasoning | Teaches tool selection, parameter construction, and execution patterns |
| **Chain of Thought (CoT)** | Step-by-step reasoning | Enables transparent decision-making for complex multi-step tasks |
| **Structured CoT** | Explicit reasoning traces | Provides clear reasoning paths ideal for agent auditing and debugging |
| **Hybrid CoT** | Mixed reasoning styles | Combines intuitive and analytical thinking for adaptive agents |
| **Multi-Turn** | Extended dialogues | Enables context retention and multi-step planning |
| **System-Prompted** | With system instructions | Defines agent personas, constraints, and behavioral guidelines |
| **Single-Turn** | Question → Answer | Direct task completion and classification tasks |

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
  --num-steps 27 \
  --batch-size 1 \
  --topic-prompt "This history Quantum physics" \
  --generation-system-prompt "You are an expert on academic history, with a specialism in the sciences" \
  --dataset-save-as dataset.jsonl
```

Deepfabric will automatically:
- Generate a hierarchical topic tree (3 levels deep, 3 branches per level)
- Create 27 diverse Q&A pairs across the generated topics
- Save your dataset to `dataset.jsonl`

> [!NOTE]  
> Want to generate faster? Increase batch size! For example, if you set `--batch-size 3` and `--num-steps 9` deepfabric will generate 3 entries at a time, while ensuring rate limits of OpenAI are monitored (we use backoff, jitter etc). 

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
| **OpenRouter** | Many Models! | Flexible model choice | Cloud |

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

## Tips for Best Results

1. **Start Small**: Test with `depth=2, degree=3` before scaling up
2. **Mix Models**: Use stronger models for topics, faster ones for generation
3. **Combine Templates**: Mix CoT reasoning with tool calling to teach both decision-making and execution
4. **Validate Tool Patterns**: Ensure tool calls include reasoning about *why* tools are selected and *how* parameters are constructed
5. **Iterate**: Generate small batches and refine prompts based on results
6. **Test Agent Behavior**: Run small-scale training experiments to validate dataset quality before generating at scale
7. **Version Control**: Save configurations for reproducibility and systematic improvement

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
