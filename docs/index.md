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
  <h3>Training Model Behavior for Agentic Systems</h3>

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



DeepFabric enables training language models to be capable agents through structured synthetic datasets. By combining reasoning traces with tool calling patterns, DeepFabric creates training data that teaches models both intelligent decision-making and precise execution—at any model parameter scale.

Built around topic-driven generation, DeepFabric uses hierarchical topic trees and experimental graph-based modeling to ensure comprehensive coverage of agent capabilities. The library serves researchers, engineers, and practitioners building Agents, distilling large model behaviors into efficient SLMs, or creating specialized tool-calling systems. Whether you're training for multi-step workflows, generating reasoning datasets, or building domain-specific agent systems, DeepFabric provides the structured, high-quality training data that transforms models into reliable agents.

## Core Capabilities

DeepFabric operates through a three-stage agent training pipeline that transforms a simple prompt into comprehensive, model-ready datasets:

**1. Topic Generation**: Creates either hierarchical tree structures or graph representations of your agent's knowledge domain. This systematic approach ensures comprehensive coverage of agent capabilities—from basic tool operations to complex multi-step reasoning tasks.

**2. Dataset Generation**: Produces training examples that combine structured reasoning with tool calling patterns. Rather than isolated question-answer pairs, the engine generates examples showing decision-making processes, tool selection logic, and parameter construction reasoning. Templates support Chain of Thought variants, multi-turn tool calling (XLAM v2), and MCP-compatible function invocations.

**3. Format Engine**: Packages datasets into model architecture-specific formats for seamless integration into training frameworks. Built-in formatters support HuggingFace TRL and custom training pipelines. This "generate once, train everywhere" approach enables rapid experimentation across model architectures and parameter scales without regenerating data.

The topic modeling foundation sets DeepFabric apart from simple prompt-based generation. Rather than creating isolated examples, the system builds a conceptual map of your agent's domain and generates examples that systematically explore different capabilities. This ensures broader skill coverage and more consistent quality—essential for training reliable agents.

## The Agent Training Paradigm

Training smaller language models to be effective agents requires more than simple instruction-response pairs. DeepFabric's approach centers on three interconnected elements:

**Structured Reasoning Traces**: Chain of Thought templates teach models to break down complex tasks into logical steps, making agent decision-making transparent and auditable. By combining free-text reasoning with structured formats, agents learn both intuitive and analytical problem-solving approaches.

**Tool Calling Patterns**: Specialized formats like TRL SFT Tools and XLAM v2 capture the full lifecycle of tool usage—from identifying the need for a tool, to selecting the appropriate function, to constructing valid parameters. This teaches agents not just to execute tools, but to understand *when* and *why* specific tools are needed.

**Format Flexibility**: DeepFabric's formatter system enables a "generate once, train everywhere" workflow. A single dataset can be exported to TRL, Unsloth, Axolotl, or custom training frameworks, allowing you to experiment with different model architectures and training approaches without regenerating data. This flexibility is particularly valuable when training SLMs across different parameter scales (0.5B to 14B+) to find the optimal balance of capability and efficiency.

This shift from simple supervised learning to structured agentic training enables SLMs to rival larger models on specific tasks while maintaining the advantages of local deployment, cost efficiency, and specialized behavior.

## AI/ML Training Pipeline Integration

DeepFabric datasets integrate directly into modern training frameworks without preprocessing or conversion pipelines, where the models Chat templates are derived from Transformers tokenizers. For example:

```python
from deepfabric import Dataset
from deepfabric.evaluation import split_to_hf_dataset, SplitConfig
from datasets import Dataset as HFDataset
import json

# Load from hub
ds = Dataset.from_hub("lukehinds/lab-equipment-tool-test")
formatted = ds.format(tokenizer=tokenizer)
formatted.save("qwen-formatted.jsonl")
```

## Topic Trees and Graphs

Traditional topic trees provide a hierarchical breakdown of subjects, ideal for domains with clear categorical structures. The experimental topic graph feature extends this concept by allowing cross-connections between topics, creating more realistic representations of complex domains where concepts naturally interconnect.

Both approaches leverage large language models to intelligently expand topics and generate relevant content, but they serve different use cases depending on your domain's structure and complexity requirements.

??? tip "Choosing Between Trees and Graphs"
    Topic trees work well for domains with clear hierarchical relationships, such as academic subjects, product categories, or organizational structures. Topic graphs excel in interconnected domains like research areas, technical concepts, or social phenomena where relationships span multiple categories.

## Getting Started

The fastest path to your first dataset involves three simple steps: installation, configuration, and generation. The [Getting Started](getting-started/index.md) section walks through this process with practical examples that you can run immediately.

For those preferring configuration-driven workflows, DeepFabric's YAML format provides comprehensive control over every aspect of generation. Developers seeking programmatic integration can access the full API through Python classes that mirror the CLI functionality.

## Integration Ecosystem

DeepFabric integrates seamlessly across the agent training ecosystem:

**Training Frameworks**: Formatters support HuggingFace TRL (SFTTrainer, tool calling), Unsloth, Axolotl, and custom frameworks. This enables rapid experimentation with different model architectures and training approaches without regenerating datasets.

**MCP Compatibility**: Generated datasets are designed for training agents that integrate with Model Context Protocol servers, ensuring your models work seamlessly with standardized tool interfaces.

**LLM Providers**: Generation leverages OpenAI, Anthropic, Google Gemini, local Ollama instances, and cloud-based solutions (Together, Groq) to create high-quality training data at scale.

**Publishing & Sharing**: Datasets export directly to Hugging Face Hub with automatic dataset cards and metadata, enabling reproducible research and community collaboration.

The modular CLI design supports complex agent training workflows through commands like `deepfabric validate` for configuration checking, `deepfabric visualize` for topic graph exploration, and `deepfabric upload` for streamlined dataset publishing.

## Next Steps

Begin with the [Installation Guide](getting-started/installation.md) to set up your environment, then follow the [First Dataset](getting-started/first-dataset.md) tutorial to generate your initial synthetic dataset. The [Configuration Guide](guide/configuration.md) provides comprehensive coverage of YAML options, while the [API Reference](api/index.md) documents programmatic usage patterns.