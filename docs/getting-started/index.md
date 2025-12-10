# Getting Started

## Installation

```bash
pip install deepfabric
```

### Development Installation

```bash
git clone https://github.com/lukehinds/deepfabric.git
cd deepfabric
uv sync --all-extras
```

## Provider Setup

Set your API key for your chosen provider:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GEMINI_API_KEY="..."
```

### Ollama (Local)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
ollama serve
```

No API key needed for local models.

## Verify Installation

```bash
deepfabric --help
deepfabric info
```

## Generate Your First Dataset

```bash
deepfabric generate \
  --topic-prompt "Python programming basics" \
  --provider openai \
  --model gpt-4o \
  --num-samples 10 \
  --output-save-as dataset.jsonl
```

This creates a JSONL file with 10 training samples.

## Using a Config File

For more control, create `config.yaml`:

```yaml
topics:
  prompt: "Machine learning fundamentals"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate educational Q&A pairs."
  conversation:
    type: basic
  llm:
    provider: openai
    model: gpt-4o

output:
  system_prompt: "You are a helpful ML tutor."
  num_samples: 20
  save_as: "ml-dataset.jsonl"
```

Then run:

```bash
deepfabric generate config.yaml
```

## Next Steps

- [Dataset Generation](../dataset-generation/index.md) - Types and configuration options
- [Tools](../tools/index.md) - Real tool execution for agent datasets
- [Training](../training/index.md) - Using datasets with TRL/Unsloth
- [Evaluation](../evaluation/index.md) - Testing fine-tuned models
