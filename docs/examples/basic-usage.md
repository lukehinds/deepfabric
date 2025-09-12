# Basic Usage Examples

These examples demonstrate fundamental DeepFabric patterns through practical, runnable configurations. Each example focuses on a specific use case while illustrating core concepts that apply across different domains and scales.

The examples progress from minimal configurations to more sophisticated setups that leverage advanced features. All examples are designed to run successfully with commonly available model providers and produce immediate, useful results.

## Simple Educational Dataset

This example generates a dataset about basic programming concepts using a straightforward hierarchical approach:

```yaml
# programming-basics.yaml
system_prompt: "You are a programming instructor creating educational content for beginners."

topic_tree:
  args:
    root_prompt: "Basic programming concepts for new developers"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 3
    tree_depth: 2
    temperature: 0.7
    provider: "openai"
    model: "gpt-3.5-turbo"
  save_as: "programming_topics.jsonl"

data_engine:
  args:
    instructions: "Create a clear explanation with a simple code example that a beginner could understand and follow."
    system_prompt: "<system_prompt_placeholder>"
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.8
    max_retries: 3

dataset:
  creation:
    num_steps: 25
    batch_size: 3
    provider: "openai"
    model: "gpt-3.5-turbo"
    sys_msg: true
  save_as: "programming_examples.jsonl"
```

Generate the dataset:

```bash
deepfabric generate programming-basics.yaml
```

This configuration creates approximately 25 educational programming examples with clear explanations and beginner-friendly code samples. The moderate temperature values ensure consistency while allowing some creativity in explanations.

## Local Development Setup

For development environments or privacy-sensitive applications, this example uses Ollama for complete local processing:

```yaml
# local-development.yaml
system_prompt: "You are a senior developer creating code examples and explanations for team documentation."

topic_tree:
  args:
    root_prompt: "Software development best practices"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 4
    tree_depth: 2
    temperature: 0.6
    provider: "ollama"
    model: "codellama"
  save_as: "dev_topics.jsonl"

data_engine:
  args:
    instructions: "Provide a practical code example with explanation of why this approach is considered a best practice."
    system_prompt: "<system_prompt_placeholder>"
    provider: "ollama"
    model: "codellama"
    temperature: 0.7
    max_retries: 2

dataset:
  creation:
    num_steps: 40
    batch_size: 2
    provider: "ollama"
    model: "codellama"
    sys_msg: false
  save_as: "dev_practices_dataset.jsonl"
```

This configuration runs entirely on local infrastructure using Ollama, making it suitable for environments with data privacy requirements or limited internet connectivity.

??? tip "Optimizing for Local Models"
    Local models often benefit from slightly lower temperature values and smaller batch sizes compared to cloud APIs. The `sys_msg: false` setting creates simpler training examples that work well with smaller local models.

## Multi-Domain Research Dataset

This example demonstrates generating content across multiple related domains using broader topic exploration:

```yaml
# research-dataset.yaml
system_prompt: "You are a research assistant creating comprehensive educational content for graduate-level study."

topic_tree:
  args:
    root_prompt: "Machine learning and artificial intelligence research areas"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 5
    tree_depth: 3
    temperature: 0.8
    provider: "anthropic"
    model: "claude-3-sonnet"
  save_as: "research_topics.jsonl"

data_engine:
  args:
    instructions: "Create a detailed explanation suitable for graduate students, including current research trends and practical applications."
    system_prompt: "<system_prompt_placeholder>"
    provider: "anthropic"
    model: "claude-3-sonnet"
    temperature: 0.7
    max_retries: 3

dataset:
  creation:
    num_steps: 100
    batch_size: 4
    provider: "anthropic"
    model: "claude-3-sonnet"
    sys_msg: true
  save_as: "ml_research_dataset.jsonl"
```

This configuration creates a comprehensive research dataset with deep topic exploration and high-quality explanations suitable for advanced educational applications.

## Conversation Dataset Generation

This example focuses on creating conversational training data with natural dialogue patterns:

```yaml
# conversation-dataset.yaml
system_prompt: "You are a helpful assistant engaging in natural, informative conversations."

topic_tree:
  args:
    root_prompt: "Everyday topics for natural conversation"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 4
    tree_depth: 2
    temperature: 0.9
    provider: "openai"
    model: "gpt-4"
  save_as: "conversation_topics.jsonl"

data_engine:
  args:
    instructions: "Create a natural conversation exchange that demonstrates helpful, engaging dialogue about this topic."
    system_prompt: "<system_prompt_placeholder>"
    provider: "openai"
    model: "gpt-4"
    temperature: 0.8
    max_retries: 3

dataset:
  creation:
    num_steps: 150
    batch_size: 5
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "conversation_examples.jsonl"
```

The higher temperature values encourage more natural, varied conversational patterns while maintaining helpful and informative content.

## Quick Prototype Configuration

For rapid experimentation and testing, this minimal configuration generates small datasets quickly:

```yaml
# prototype.yaml
system_prompt: "You are creating test examples for development."

topic_tree:
  args:
    root_prompt: "Basic test concepts"
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 2
    tree_depth: 1
    temperature: 0.5
    provider: "openai"
    model: "gpt-3.5-turbo"
  save_as: "test_topics.jsonl"

data_engine:
  args:
    instructions: "Create a simple example for testing purposes."
    system_prompt: "<system_prompt_placeholder>"
    provider: "openai"
    model: "gpt-3.5-turbo"
    temperature: 0.5
    max_retries: 2

dataset:
  creation:
    num_steps: 5
    batch_size: 1
    provider: "openai"
    model: "gpt-3.5-turbo"
    sys_msg: false
  save_as: "test_dataset.jsonl"
```

This configuration completes in under a minute and produces a small dataset suitable for testing downstream processing, format validation, or initial proof-of-concept work.

## Usage Patterns

These basic examples demonstrate key patterns that apply across different use cases:

**Provider Selection**: Choose providers based on your requirements for cost, quality, privacy, and speed. OpenAI provides good general performance, Anthropic excels at detailed explanations, and Ollama offers privacy and cost benefits.

**Parameter Tuning**: Temperature values between 0.6-0.8 work well for most educational content, while conversational datasets benefit from slightly higher values around 0.8-0.9.

**Scale Management**: Start with small `num_steps` values during development and scale up for production datasets. Use batch sizes between 2-5 for most providers to balance speed with rate limiting.

**Output Format**: Use `sys_msg: true` for training data where role awareness is important, and `sys_msg: false` for simpler question-answer pairs.