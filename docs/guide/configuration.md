# Configuration

DeepFabric's YAML configuration system provides comprehensive control over the synthetic data generation process while maintaining clarity and reproducibility. The configuration structure mirrors the generation pipeline, with distinct sections that operate independently while sharing common parameters through intelligent placeholder substitution.

Understanding the configuration format enables sophisticated customization of the generation process, from simple parameter adjustments to complex multi-stage workflows that leverage different model providers for different components.

## Configuration Structure

The configuration file contains four primary sections, each controlling a different aspect of the generation process:

```yaml
dataset_system_prompt: "Global system prompt available to all components"

topic_tree:
  # Topic tree generation parameters
  save_as: "output_file.jsonl"

data_engine:
  # Dataset generation parameters

dataset:
  creation:
    # Generation execution parameters
  save_as: "dataset.jsonl"

# Optional Hugging Face integration
huggingface:
  repository: "username/dataset-name"
  token: "your-token"
  tags: ["custom", "tags"]
```

This structure separates concerns while enabling parameter sharing through the placeholder system, creating configurations that are both powerful and maintainable.

## System Prompt Integration

The `dataset_system_prompt` field provides a template prompt that can be copied to other sections in your configuration. Users should specify system prompts directly in each section where they want them to ensure clarity and maintainability.

```yaml
dataset_system_prompt: "You are an expert data scientist creating educational content for machine learning practitioners."

topic_tree:
  topic_system_prompt: "You are an expert data scientist creating educational content for machine learning practitioners."

data_engine:
  generation_system_prompt: "You are an expert data scientist creating educational content for machine learning practitioners."
```

The placeholder substitution occurs at runtime, allowing you to modify the global behavior by changing a single line rather than updating multiple sections throughout your configuration.

??? tip "Advanced Prompt Strategies"
    Consider using different system prompts for topic generation versus dataset creation. Topic generation often benefits from broader, exploratory prompts that encourage comprehensive coverage, while dataset generation may require more focused prompts that emphasize specific output formats or quality criteria.

## Topic Tree Configuration

The topic tree section controls the hierarchical expansion of your root prompt into comprehensive topic coverage:

```yaml
topic_tree:
  topic_prompt: "Machine learning fundamentals for data scientists"
  topic_system_prompt: "You are an expert data scientist creating educational content for machine learning practitioners."
  degree: 4      # Subtopics per node
  depth: 3       # Maximum tree depth
  temperature: 0.7    # Generation creativity
  provider: "openai"  # Model provider
  model: "gpt-4"      # Specific model
  save_as: "ml_topics.jsonl"
```

The `degree` parameter controls breadth while `depth` controls depth, allowing you to balance comprehensive coverage with generation time. Higher degree values create more subtopics per level, while greater depth values enable more detailed exploration of each subtopic.

Temperature affects the creativity and diversity of topic generation. Lower values produce more predictable, conventional topics, while higher values encourage more creative and unexpected connections.

## Data Engine Configuration

The data engine transforms topics into actual training examples using configurable templates and generation parameters:

```yaml
data_engine:
  instructions: "Create a practical code example with detailed explanation"
  generation_system_prompt: "You are an expert data scientist creating educational content for machine learning practitioners."
  provider: "anthropic"
  model: "claude-3-sonnet"
  temperature: 0.8
  max_retries: 3
  request_timeout: 30
  default_batch_size: 5
  default_num_examples: 3
```

The `instructions` field guides the generation process, specifying the type and format of content to create. This field accepts detailed specifications about output format, complexity level, target audience, and quality criteria.

We can also supply `max_tokens` for each llm call for the data engine (default is 2000):

```yaml
data_engine:
  instructions: "Create a practical code example with detailed explanation"
  generation_system_prompt: "You are an expert data scientist creating educational content for machine learning practitioners."
  provider: "anthropic"
  model: "claude-3-sonnet"
  temperature: 0.8
  max_tokens: 4096
  max_retries: 3
  request_timeout: 30
  default_batch_size: 5
  default_num_examples: 3
```


Error handling parameters like `max_retries` and `request_timeout` ensure robust operation when working with external API services that may experience temporary issues.

## Dataset Creation Parameters

The dataset section controls the execution of data generation, including batch processing and output formatting:

```yaml
dataset:
  creation:
    num_steps: 100        # Total examples to generate
    batch_size: 5         # Examples per API call
    provider: "ollama"    # Provider for generation
    model: "mistral"      # Model for generation
    sys_msg: true         # Include system messages
  save_as: "training_dataset.jsonl"
```

The `sys_msg` parameter determines whether system prompts are included in the final dataset. Setting this to `true` creates conversational examples with explicit role definitions, useful for training models that need to understand their intended behavior. Setting it to `false` creates simpler user-assistant pairs.

Batch processing parameters balance generation speed with resource consumption. Larger batch sizes increase throughput but require more memory and may hit API rate limits more frequently.

## Provider Integration

DeepFabric supports any multiple providers through consistent configuration patterns. Different components can use different providers, enabling sophisticated workflows that optimize for cost, performance, or specific model capabilities:

```yaml
topic_tree:
  provider: "openai"
  model: "gpt-4-turbo"    # Fast, cost-effective for topic generation

data_engine:
  provider: "anthropic"
  model: "claude-sonnet-4-5"    # High-quality for content generation
```

Provider authentication occurs through environment variables following the pattern `{PROVIDER}_API_KEY`. For example, OpenAI requires `OPENAI_API_KEY` while Anthropic requires `ANTHROPIC_API_KEY`.

??? tip "Provider Selection Strategy"
    Consider using faster, less expensive models for topic generation and higher-quality models for dataset creation. Topic generation benefits from breadth and speed, while dataset creation benefits from depth and quality. This hybrid approach optimizes both cost and output quality.

## Hugging Face Integration

The optional Hugging Face section enables automatic dataset publishing with generated metadata:

```yaml
huggingface:
  repository: "organization/dataset-name"
  token: "hf_your_token_here"  # Optional if using HF_TOKEN env var
  tags:
    - "synthetic"
    - "educational"
    - "machine-learning"
```

The integration automatically generates dataset cards with metadata about the generation process, model providers used, and dataset statistics. The "deepfabric" and "synthetic" tags are added automatically to identify the generation method.

## Configuration Validation

Validate your configuration before running expensive generation processes:

```bash
deepfabric validate your-config.yaml
```

The validation process checks for common issues including missing required fields, parameter compatibility problems, and provider authentication issues. This proactive approach saves time by catching configuration problems before they cause generation failures.

## Environment-Specific Configurations

Maintain separate configurations for different environments or use cases:

```yaml
# development.yaml - Fast iteration
topic_tree:
  depth: 2
  degree: 3
dataset:
  creation:
    num_steps: 10

---

# production.yaml - Comprehensive generation
topic_tree:
  depth: 4
  degree: 5
dataset:
  creation:
    num_steps: 1000
```

This approach enables rapid prototyping during development while maintaining the ability to generate comprehensive datasets for production use.