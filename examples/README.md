# DeepFabric Examples

This directory contains comprehensive examples demonstrating DeepFabric's capabilities with the new modular configuration system.

## Quick Start

**Fastest way to get started:**

```bash
# Using a YAML config file
deepfabric start examples/configs/01-quickstart.yaml

# Or programmatically with Python
python examples/code/01-quickstart.py
```

## Examples Organization

### 📁 Configuration Examples (`configs/`)

YAML-based configuration files demonstrating different features:

1. **[01-quickstart.yaml](configs/01-quickstart.yaml)** - Simplest possible example
   - Basic conversation generation
   - Minimal configuration required
   - Great starting point for new users

2. **[02-conversation-types.yaml](configs/02-conversation-types.yaml)** - Conversation type options
   - `basic` - Simple Q&A conversations
   - `chain_of_thought` - Reasoning-enabled conversations

3. **[03-reasoning-styles.yaml](configs/03-reasoning-styles.yaml)** - Chain-of-thought reasoning
   - `freetext` - Natural language reasoning
   - `structured` - Step-by-step reasoning traces
   - `hybrid` - Combination of both styles

4. **[04-agent-tools.yaml](configs/04-agent-tools.yaml)** - Agent modes with tool calling
   - `single_turn` - One-shot tool usage
   - `multi_turn` - Multi-turn agent conversations
   - Built-in and custom tools

5. **[05-output-formats.yaml](configs/05-output-formats.yaml)** - Output format options
   - `standard` - Universal conversation format
   - `xlam` - XLAM v2 agent format
   - `openai` - OpenAI fine-tuning format
   - `anthropic` - Anthropic fine-tuning format

6. **[06-complete-pipeline.yaml](configs/06-complete-pipeline.yaml)** - Full workflow
   - Topic tree generation
   - Dataset creation
   - Multiple formatters
   - HuggingFace upload

7. **[custom-tools.yaml](configs/custom-tools.yaml)** - Custom tool definitions
   - Tool registry format
   - Parameter definitions
   - Usage examples

### 💻 Python Examples (`code/`)

Programmatic usage examples:

1. **[01-quickstart.py](code/01-quickstart.py)** - Minimal programmatic example
2. **[02-programmatic-config.py](code/02-programmatic-config.py)** - Building configs in code
3. **[03-custom-tools.py](code/03-custom-tools.py)** - Defining custom tools
4. **[04-complete-pipeline.py](code/04-complete-pipeline.py)** - Full programmatic workflow

## Modular Configuration Architecture

DeepFabric now uses a clean, composable configuration system with orthogonal concerns:

### Conversation Types

Controls the base conversation structure:

- **`basic`** - Simple message exchanges
- **`structured`** - Conversations with metadata fields
- **`chain_of_thought`** - Enables reasoning capabilities

### Reasoning Styles (for `chain_of_thought`)

When using `chain_of_thought`, specify how reasoning should be structured:

- **`freetext`** - Natural language reasoning paragraphs
- **`structured`** - Explicit step-by-step reasoning traces
- **`hybrid`** - Both freetext analysis and structured steps

### Agent Modes (optional)

Enable agent capabilities with tool calling:

- **`single_turn`** - Agent completes task in one interaction
- **`multi_turn`** - Multi-turn conversations with tools

**Note:** Agent modes require `available_tools` to be configured.

### Output Formats

Transform your dataset into specific training formats:

- **`standard`** - Universal conversation format (default)
- **`xlam`** - XLAM v2 format (requires `multi_turn` agent)
- **`openai`** - OpenAI fine-tuning format
- **`anthropic`** - Anthropic fine-tuning format

## Configuration Examples

### Basic Conversation

```yaml
data_engine:
  conversation_type: "basic"
  provider: "openai"
  model: "gpt-4"
```

### Chain of Thought with Structured Reasoning

```yaml
data_engine:
  conversation_type: "chain_of_thought"
  reasoning_style: "structured"
  provider: "openai"
  model: "gpt-4"
```

### Agent with Tools (Single Turn)

```yaml
data_engine:
  conversation_type: "chain_of_thought"
  reasoning_style: "hybrid"
  agent_mode: "single_turn"
  available_tools: ["get_weather", "calculate"]
  provider: "openai"
  model: "gpt-4"
```

### Multi-Turn Agent

```yaml
data_engine:
  conversation_type: "chain_of_thought"
  reasoning_style: "hybrid"
  agent_mode: "multi_turn"
  available_tools: ["get_weather", "search_web"]
  provider: "openai"
  model: "gpt-4"
```

## Common Use Cases

### Training a General-Purpose Assistant

Use `basic` conversation type.

**Example:** `configs/02-conversation-types.yaml` (basic section)

### Training a Reasoning Model

Use `chain_of_thought` with your preferred reasoning style.

**Example:** `configs/03-reasoning-styles.yaml`

### Training a Tool-Using Agent (Single Turn)

Use `chain_of_thought` + `agent_mode: single_turn` + tools.

**Example:** `configs/04-agent-tools.yaml` (single_turn section)

### Training a Multi-Turn Agent

Use `chain_of_thought` + `agent_mode: multi_turn` + tools.

**Example:** `configs/04-agent-tools.yaml` (multi_turn section)

### Formatting for Different Platforms

Use formatters to transform generated data to platform-specific formats (XLAM, OpenAI, Anthropic, etc.).

**Example:** `configs/05-output-formats.yaml`

## Configuration Reference

### Top-Level Structure

```yaml
dataset_system_prompt: "..."  # Overall system prompt for dataset

topic_tree:                   # Topic generation (optional)
  topic_prompt: "..."
  provider: "..."
  model: "..."
  depth: 2
  degree: 3

data_engine:                  # Core generation configuration
  generation_system_prompt: "..."
  conversation_type: "..."    # basic | chain_of_thought
  reasoning_style: "..."      # freetext | structured | hybrid (if CoT)
  agent_mode: "..."           # single_turn | multi_turn (optional)
  available_tools: [...]      # Tools for agent mode
  provider: "..."
  model: "..."

dataset:                      # Dataset configuration
  save_as: "..."
  creation:
    num_steps: 10
    batch_size: 2
  formatters: [...]           # Transform to specific formats (XLAM, OpenAI, etc.)
```

### Validation Rules

The configuration system enforces logical combinations:

1. **`reasoning_style`** can only be set when `conversation_type` is `chain_of_thought`
2. **`agent_mode`** requires `available_tools` to be configured

## Running Examples

### Using YAML Configs

```bash
# Run with a config file
deepfabric start examples/configs/01-quickstart.yaml

# Override specific parameters
deepfabric start examples/configs/01-quickstart.yaml --model gpt-4o --num-steps 20
```

### Using Python

```bash
# Run Python examples directly
python examples/code/01-quickstart.py

# Or with modifications
python examples/code/02-programmatic-config.py
```

## Additional Resources

- **[Main Documentation](https://docs.deepfabric.ai)** - Complete reference
- **[API Reference](https://docs.deepfabric.ai/api)** - Python API docs
- **[Migration Guide](https://docs.deepfabric.ai/migration)** - Upgrading from old configs

## Getting Help

- Create an issue on [GitHub](https://github.com/yourusername/deepfabric/issues)
- Join our [Discord community](https://discord.gg/deepfabric)
- Check the [FAQ](https://docs.deepfabric.ai/faq)

## License

All examples are provided under the same license as DeepFabric (MIT).
