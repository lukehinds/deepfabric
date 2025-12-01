<div align="center">
  <h1>DeepFabric</h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/logo-light.png" />
    <img alt="DeepFabric logo" src="./assets/logo-light.png" style="width:40%;max-width:40%;height:auto;display:block;margin:0 auto;" />
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
</div>

**DeepFabric** generates synthetic datasets for training small language models as capable agents. It combines reasoning traces with tool-calling patterns and structured outputs to create high-quality training data at scale.

## Quickstart

```bash
pip install deepfabric
```

```bash
export OPENAI_API_KEY="your-api-key"

deepfabric generate \
  --topic-prompt "Python programming fundamentals" \
  --generation-system-prompt "You are a Python expert" \
  --num-samples 27 \
  --output-save-as dataset.jsonl
```

This generates a topic tree and creates 27 Q&A training samples.

## Configuration

DeepFabric uses YAML configuration with three main sections and optional shared LLM defaults:

```yaml
# Optional: Shared LLM defaults (inherited by topics and generation)
llm:
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.7

# TOPICS: Generate the topic tree/graph
topics:
  prompt: "Python programming fundamentals"
  mode: tree                    # tree | graph
  depth: 3
  degree: 3
  save_as: "topics.jsonl"
  # Optional: Override shared LLM settings
  llm:
    model: "gpt-4o-mini"        # Use cheaper model for topics

# GENERATION: Create training samples from topics
generation:
  system_prompt: "You are a Python programming instructor..."
  instructions: "Create clear tutorials with code examples"
  conversation:
    type: basic                 # basic | chain_of_thought
  max_retries: 3
  # Optional: Override shared LLM settings
  llm:
    temperature: 0.3            # Lower temp for consistent code

# OUTPUT: Final dataset configuration
output:
  system_prompt: "You are a helpful Python assistant"
  include_system_message: true
  num_samples: 27
  batch_size: 3
  save_as: "dataset.jsonl"

# Optional: Upload to Hugging Face
huggingface:
  repository: "your-username/dataset-name"
  tags: ["python", "programming"]
```

Run with:

```bash
deepfabric generate config.yaml
```

### CLI Overrides

Override any configuration value from the command line:

```bash
deepfabric generate config.yaml \
  --provider gemini \
  --model gemini-2.0-flash \
  --num-samples 100 \
  --batch-size 5 \
  --depth 4 \
  --degree 5
```

| Flag | Description |
|------|-------------|
| `--provider` | LLM provider (openai, anthropic, gemini, ollama) |
| `--model` | Model name |
| `--temperature` | Generation temperature |
| `--depth` | Topic tree depth |
| `--degree` | Subtopics per node |
| `--num-samples` | Number of samples to generate |
| `--batch-size` | Parallel generation batch size |
| `--topic-prompt` | Starting topic for generation |
| `--topics-save-as` | Save path for topics |
| `--topics-load` | Load existing topics file |
| `--output-save-as` | Save path for dataset |
| `--include-system-message` | Include system message in output |
| `--topic-only` | Generate topics only, skip samples |

## Agent & Tool-Calling Datasets

For training agents with tool-calling capabilities:

```yaml
llm:
  provider: "openai"
  model: "gpt-4o"

topics:
  prompt: "Tasks requiring web search and calculations"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "You are an AI assistant with tool access..."
  conversation:
    type: chain_of_thought
    reasoning_style: agent      # freetext | agent
    agent_mode: single_turn     # single_turn | multi_turn
  tools:
    registry_path: "tools.yaml"
    max_per_query: 3

output:
  num_samples: 50
  save_as: "agent-dataset.jsonl"
  formatters:
    - name: trl
      template: builtin://trl_sft_tools
      output: agent-trl.jsonl
```

## Output Formats

| Format | Template | Use Case |
|--------|----------|----------|
| TRL SFT Tools | `builtin://trl_sft_tools` | HuggingFace TRL SFTTrainer |
| XLAM v2 | `builtin://xlam_v2` | Salesforce xLAM models |
| Tool Calling | `builtin://tool_calling` | Agent training |
| GRPO | `builtin://grpo` | Reinforcement learning |
| Conversations | `builtin://conversations` | Unsloth, Axolotl |
| ChatML | `builtin://chatml` | Multi-turn chat models |
| Alpaca | `builtin://alpaca` | Instruction-following |

## Providers

| Provider | Local/Cloud | Best For |
|----------|-------------|----------|
| OpenAI | Cloud | High quality, complex tasks |
| Anthropic | Cloud | Nuanced reasoning |
| Google Gemini | Cloud | Cost-effective at scale |
| Ollama | Local | Privacy, unlimited generation |
| OpenRouter | Cloud | Flexible model choice |

## Resources

- [Documentation](https://lukehinds.github.io/deepfabric/)
- [Examples](./examples/README.md)
- [Discord](https://discord.gg/pPcjYzGvbS)
- [Issues](https://github.com/lukehinds/deepfabric/issues)

## Development

```bash
git clone https://github.com/lukehinds/deepfabric
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
