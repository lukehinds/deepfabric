<style>
  .md-typeset h1,
  .md-content__button {
    display: none;
  }
</style>
<div align="center">
    <p align="center">
        <img src="images/logo-light.png" alt="DeepFabric Logo" width="500"/>
    </p>
  <h3>Synthetic Training Data for Agents</h3>

  <p>
    <a href="https://github.com/lukehinds/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
      <img src="https://img.shields.io/badge/Contribute-Good%20First%20Issues-green?style=for-the-badge&logo=github" alt="Good First Issues"/>
    </a>
    &nbsp;
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/badge/Chat-Join%20Discord-7289da?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord"/>
    </a>
  </p>

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

## Quick Start

```bash
pip install deepfabric
export OPENAI_API_KEY="your-key"

deepfabric generate \
  --topic-prompt "DevOps and Platform Engineering" \
  --generation-system-prompt "You are an expert in DevOps and Platform Engineering" \
  --mode graph \
  --depth 2 \
  --degree 2 \
  --provider openai \
  --model gpt-4o \
  --num-samples 2 \
  --batch-size 1 \
  --conversation-type chain_of_thought \
  --reasoning-style freetext \
  --output-save-as dataset.jsonl \
```

## What DeepFabric Does

1. **Generates topic hierarchies** from a root prompt
2. **Creates training samples** for each topic
3. **Outputs JSONL** compatible with HuggingFace and training frameworks

## Dataset Types

| Type | Description | Use Case |
|------|-------------|----------|
| [Basic](dataset-generation/basic.md) | Simple Q&A pairs | Instruction following |
| [Reasoning](dataset-generation/reasoning.md) | Chain-of-thought traces | Step-by-step problem solving |
| [Agent](dataset-generation/agent.md) | Tool-calling with real execution | Building agents |

## Key Features

**Topic-driven generation** ensures diverse, non-redundant samples. Each training example maps to a specific subtopic, avoiding the repetition common in naive generation.

**Real tool execution** via [Spin](tools/index.md). Agent datasets include actual tool results from isolated WebAssembly sandboxes, not simulated outputs.

**Training integration** with [TRL, Unsloth](training/index.md), and HuggingFace. Use `apply_chat_template` to format for any model.

**Built-in evaluation** for [testing fine-tuned models](evaluation/index.md) on tool-calling tasks with metrics for accuracy and correctness.

## Documentation

- [Getting Started](getting-started/index.md) - Installation and first dataset
- [Dataset Generation](dataset-generation/index.md) - Types and configuration
- [Tools](tools/index.md) - Spin components and MCP integration
- [Training](training/index.md) - Loading datasets and formatting
- [Evaluation](evaluation/index.md) - Testing fine-tuned models
- [CLI Reference](cli/index.md) - Command documentation
