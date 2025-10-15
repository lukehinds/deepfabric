# Provider Integration

DeepFabric integrates with language model providers, enabling access to virtually any commercial or open-source language model through a unified interface. This abstraction layer provides consistent behavior across different providers while preserving access to provider-specific capabilities and optimizations.

The integration approach supports both simple single-provider workflows and sophisticated multi-provider strategies that optimize different components for cost, performance, or capability requirements.

## Supported Providers

DeepFabric supports all model providers through consistent configuration patterns:

**OpenAI** provides GPT models including GPT-4, GPT-3.5, and specialized variants optimized for different use cases.

**Anthropic** offers Claude models with strong reasoning capabilities and detailed response generation.

**Google** includes Gemini models with multimodal capabilities and competitive performance characteristics.

**Ollama** enables local model execution with privacy benefits and no per-token costs.

**Azure OpenAI** provides enterprise-grade OpenAI model access with additional security and compliance features.

**AWS Bedrock** offers access to multiple model providers through a single enterprise-focused platform.

## Authentication Patterns

Each provider uses environment variables following the `{PROVIDER}_API_KEY` pattern:

```bash
# OpenAI authentication
export OPENAI_API_KEY="your-openai-key"

# Anthropic authentication
export ANTHROPIC_API_KEY="your-anthropic-key"

# Google Gemini authentication
export GEMINI_API_KEY="your-gemini-key"

# Hugging Face (for dataset upload)
export HF_TOKEN="your-hf-token"
```

This consistent pattern simplifies multi-provider workflows and enables easy provider switching through environment configuration.

## Configuration Syntax

Provider configuration uses the `provider/model` format throughout DeepFabric:

```yaml
# OpenAI GPT-4
provider: "openai"
model: "gpt-4"

# Anthropic Claude
provider: "anthropic" 
model: "claude-sonnet-4-5"

# Local Ollama
provider: "ollama"
model: "mistral:latest"

# Google Gemini
provider: "gemini"
model: "gemini-pro"
```

This separation enables clear specification of both the provider infrastructure and the specific model within that provider's catalog.

## Multi-Provider Workflows

DeepFabric supports using different providers for different components, enabling optimization strategies:

```yaml
# Fast, cost-effective topic generation
topic_tree:
  provider: "openai"
  model: "gpt-4-turbo"

# High-quality dataset generation
data_engine:
  provider: "anthropic"
  model: "claude-sonnet-4-5"

# Different model for final dataset creation
dataset:
  creation:
    provider: "ollama"
    model: "mistral:latest"
```

This approach balances cost, speed, and quality by using appropriate models for each stage's specific requirements.

## Provider-Specific Optimizations

Different providers have distinct characteristics that influence optimal usage patterns:

**OpenAI Models** excel at following instructions precisely and provide good balance of speed and quality. GPT-4 offers superior reasoning while GPT-3.5 provides faster, more economical processing.

**Anthropic Claude** demonstrates strong performance on complex reasoning tasks and produces detailed, well-structured responses ideal for educational content generation.

**Ollama Local Models** eliminate per-token costs and provide complete data privacy, making them suitable for sensitive content or high-volume generation with budget constraints.

**Google Gemini** offers competitive performance with potentially lower costs and good instruction-following capabilities.

## Rate Limiting and Error Handling

DeepFabric includes comprehensive error handling for provider-specific issues:

**Automatic Retry Logic** handles temporary failures with exponential backoff, adapting to different providers' reliability characteristics.

**Rate Limit Management** respects provider-specific rate limits through configurable delays and batch size management.

**Fallback Strategies** can be configured to switch providers when primary providers experience issues or reach limits.

**Cost Monitoring** tracks usage patterns to help optimize provider selection for budget-conscious deployments.

## Local Model Setup

Ollama provides completely local model execution with simple setup:

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull and run a model
ollama pull mistral:latest
ollama pull codellama:latest

# Verify model availability
ollama list
```

Local models require no API keys and provide unlimited usage, though they require sufficient local computational resources.

## Provider Selection Strategy

Choose providers based on your specific requirements:

**For Development and Testing** use faster, less expensive models like GPT-3.5 or local Ollama models to iterate quickly on configuration and approach.

**For Production Datasets** consider higher-quality models like GPT-4 or Claude-3-Opus when output quality is paramount.

**For Privacy-Sensitive Content** use local Ollama models to maintain complete control over data processing.

**For Cost Optimization** mix providers strategically, using economical models for topic generation and premium models only for final dataset creation.

## Performance Monitoring

Monitor provider performance through built-in metrics:

**Response Times** vary significantly between providers and models, affecting overall generation time.

**Success Rates** differ based on provider reliability and model capability for your specific use cases.

**Cost Tracking** helps optimize provider selection for budget-conscious projects.

**Quality Assessment** evaluates output quality across different providers to inform selection decisions.

??? tip "Provider Migration Strategy"
    Start with a single provider for simplicity, then experiment with multi-provider approaches as you understand your specific requirements. Keep provider configurations in separate YAML files to enable easy switching for different use cases or cost optimization needs.