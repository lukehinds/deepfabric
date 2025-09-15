# Getting Started with DeepFabric

DeepFabric transforms simple prompts into comprehensive synthetic datasets through an intuitive three-step process. This guide walks you through installation, basic configuration, and generating your first dataset in under five minutes.

The journey from installation to your first dataset involves setting up the environment, creating a configuration file that defines your generation parameters, and running the generation process. Each step builds naturally toward producing high-quality synthetic data tailored to your specific needs.

## Prerequisites

DeepFabric requires Python 3.11 or higher and works with multple model providers. Whether you prefer OpenAI's GPT models, Anthropic's Claude, Google's Gemini, or local Ollama instances, DeepFabric adapts to your preferred setup.

The system works equally well with cloud-based APIs and local models, making it suitable for both rapid prototyping and production environments where data privacy considerations may require on-premises processing.

## Installation Methods

Choose the installation method that best fits your workflow. The pip installation provides immediate access to stable releases, while the development installation offers access to the latest features and the ability to contribute back to the project.

### Standard Installation

```bash
pip install deepfabric
```

### Development Installation

For those wanting to explore the latest features or contribute to the project, the development installation provides access to the full source code and development tools.

```bash
# Install uv for dependency management
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and set up the development environment
git clone https://github.com/lukehinds/deepfabric.git
cd deepfabric
uv sync --all-extras
```

??? tip "Why uv for Development?"
    The uv package manager provides faster dependency resolution and better reproducibility than traditional pip workflows. It's particularly beneficial when working with the large dependency trees common in machine learning projects.

## Verification

Confirm your installation by checking the available commands and version information:

```bash
deepfabric --help
deepfabric info
```

The `info` command displays version details, available commands, and environment variable requirements, providing a quick overview of your installation status.

## Next Steps

With DeepFabric installed, proceed to [Installation](installation.md) for detailed setup instructions, then move to [First Dataset](first-dataset.md) to generate your initial synthetic dataset. The process takes just a few minutes and demonstrates the core concepts you'll use in more complex scenarios.