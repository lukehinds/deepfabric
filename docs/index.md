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



DeepFabric transforms the process of creating synthetic datasets for language model training, evaluation, and research. Built around the concept of topic-driven data generation, it provides both hierarchical topic trees and experimental graph-based topic modeling to create diverse, contextually rich training examples.

The library serves researchers, engineers, and practitioners who need high-quality synthetic data for model distillation, agent evaluation, or statistical research. Whether you're generating conversational datasets, creating domain-specific training examples, building evaluation benchmarks, or training agents for systematic tool usage, DeepFabric provides the tools to scale your data generation process while maintaining quality and diversity.

## Core Capabilities

DeepFabric operates through a three-stage pipeline that transforms a simple prompt into a comprehensive dataset. The process begins with topic generation, where the system creates either a hierarchical tree structure or a more complex graph representation of your domain. These topics then feed into the dataset generation engine, which produces contextually appropriate training examples including traditional conversations, Chain of Thought reasoning, and advanced agent tool-calling scenarios. Finally, the system packages everything into standard formats ready for immediate use.

The topic modeling approach sets DeepFabric apart from simple prompt-based generation. Rather than creating isolated examples, the system builds a conceptual map of your domain and generates examples that explore different aspects systematically. This ensures broader coverage and more consistent quality across your dataset.

**Agent Tool-Calling Capabilities**: DeepFabric includes specialized formats for training models in systematic tool usage, capturing not just what tools to use but why they're selected and how parameters are constructed. This enables creation of datasets for training more effective function-calling models and agents that work seamlessly with Model Context Protocol (MCP) servers.

## Topic Trees and Graphs

Traditional topic trees provide a hierarchical breakdown of subjects, ideal for domains with clear categorical structures. The experimental topic graph feature extends this concept by allowing cross-connections between topics, creating more realistic representations of complex domains where concepts naturally interconnect.

Both approaches leverage large language models to intelligently expand topics and generate relevant content, but they serve different use cases depending on your domain's structure and complexity requirements.

??? tip "Choosing Between Trees and Graphs"
    Topic trees work well for domains with clear hierarchical relationships, such as academic subjects, product categories, or organizational structures. Topic graphs excel in interconnected domains like research areas, technical concepts, or social phenomena where relationships span multiple categories.

## Getting Started

The fastest path to your first dataset involves three simple steps: installation, configuration, and generation. The [Getting Started](getting-started/index.md) section walks through this process with practical examples that you can run immediately.

For those preferring configuration-driven workflows, DeepFabric's YAML format provides comprehensive control over every aspect of generation. Developers seeking programmatic integration can access the full API through Python classes that mirror the CLI functionality.

## Integration Ecosystem

DeepFabric integrates seamlessly with the modern machine learning ecosystem, including OpenAI, Anthropic, local Ollama instances, and cloud-based solutions. Generated datasets export directly to Hugging Face Hub with automatic dataset cards and metadata.

The modular CLI design supports complex workflows through commands like `deepfabric validate` for configuration checking, `deepfabric visualize` for topic graph exploration, and `deepfabric upload` for streamlined dataset publishing.

## Next Steps

Begin with the [Installation Guide](getting-started/installation.md) to set up your environment, then follow the [First Dataset](getting-started/first-dataset.md) tutorial to generate your initial synthetic dataset. The [Configuration Guide](guide/configuration.md) provides comprehensive coverage of YAML options, while the [API Reference](api/index.md) documents programmatic usage patterns.