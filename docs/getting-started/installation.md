# Installation

DeepFabric offers flexible installation options designed to accommodate different use cases, from quick experimentation to full development environments. The process adapts to your workflow while ensuring you have access to all necessary dependencies.

## Standard Installation

The pip installation provides the most straightforward path to using DeepFabric. This method installs the latest stable release with all core dependencies, suitable for most use cases.

```bash
pip install deepfabric
```

After installation, verify the setup by running:

```bash
deepfabric --version
```

This command confirms that DeepFabric is properly installed and accessible from your command line environment.

## Provider Setup

DeepFabric operates through various language model providers, each requiring specific authentication setup. The system supports any provider comp, giving you flexibility in choosing models based on cost, performance, or privacy requirements.

### OpenAI Configuration

For OpenAI models including GPT-4 and GPT-3.5:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Anthropic Configuration

For Claude models:

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Ollama Setup

For local model execution with Ollama, first install and start the Ollama service:

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull mistral

# Verify the model is available
ollama list
```

Ollama provides privacy-focused local execution without requiring API keys, making it ideal for sensitive data processing or environments with limited internet connectivity.

??? tip "Choosing Your Model Provider"
    OpenAI and Anthropic offer state-of-the-art performance with cloud-based processing. Ollama provides privacy and cost benefits through local execution. Consider your specific requirements for data privacy, cost sensitivity, and performance needs when selecting a provider.

### Google Gemini Configuration

For Google's Gemini models:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### Environment Variable Management

Create a `.env` file in your project directory to manage API keys systematically:

```bash
# .env file
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key
HF_TOKEN=your-huggingface-token
```

Load these variables using:

```bash
set -a && source .env && set +a
```

## Development Installation

The development installation provides access to the latest features, development tools, and the ability to contribute to the project. This approach uses uv for faster dependency management and includes testing frameworks.

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/lukehinds/deepfabric.git
cd deepfabric

# Install all dependencies including development tools
uv sync --all-extras
```

The development installation includes additional tools for code formatting, linting, testing, and documentation generation. These tools maintain code quality and enable contribution to the project.

### Development Verification

Confirm your development setup by running the test suite:

```bash
make test
```

This command executes the full test suite, ensuring your environment is properly configured and all dependencies are correctly installed.

## Hugging Face Integration

If you plan to upload datasets to Hugging Face Hub, configure your authentication token:

```bash
export HF_TOKEN="your-huggingface-token"
```

Alternatively, use the Hugging Face CLI to authenticate:

```bash
pip install huggingface_hub
huggingface-cli login
```

The Hugging Face integration enables automatic dataset uploading with generated dataset cards, streamlining the process of sharing your synthetic datasets with the community.

## Installation Verification

Verify your complete installation by running:

```bash
deepfabric info
```

This command displays your DeepFabric version, available commands, and detected environment variables, confirming that your installation is complete and properly configured.

## Next Steps

With DeepFabric installed and your preferred model provider configured, proceed to [First Dataset](first-dataset.md) to generate your initial synthetic dataset and explore the core functionality.