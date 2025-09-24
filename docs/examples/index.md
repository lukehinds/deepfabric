# Examples

DeepFabric examples demonstrate practical applications across diverse domains and use cases, from educational content generation to specialized research datasets. These examples showcase both YAML configuration approaches and programmatic Python implementations, providing templates you can adapt for your specific requirements.

Each example includes complete configuration files, expected outputs, and detailed explanations of design decisions. The examples progress from simple single-domain datasets to complex multi-stage workflows that leverage advanced features like topic graphs and Hugging Face integration.

## Example Categories

The examples are organized by complexity and application domain to help you find relevant patterns quickly:

**Basic Usage** covers fundamental patterns including simple topic trees, basic dataset generation, and standard output formats. These examples serve as starting points for new users and templates for common use cases.

**Advanced Workflows** demonstrate sophisticated techniques including multi-stage generation, provider mixing, topic graph usage, and quality control patterns. These examples show how to leverage DeepFabric's full capabilities for complex requirements.

**Domain-Specific Examples** provide complete configurations for specific industries or use cases, including educational content, technical documentation, research datasets, and evaluation benchmarks.

## Configuration Patterns

Examples demonstrate key configuration patterns that appear across different use cases:

**Provider Optimization** shows how to use different models for different stages of generation, balancing cost, speed, and quality based on each component's requirements.

**Quality Control** illustrates techniques for ensuring consistent output quality, including temperature management, retry strategies, and validation patterns.

**Scale Management** covers approaches for generating large datasets efficiently, including batch processing optimization and resource management strategies.

## Real-World Applications

The examples reflect actual use cases encountered in production environments:

**Educational Content Generation** creates structured learning materials with appropriate difficulty progression and comprehensive topic coverage.

**Technical Documentation** produces API documentation, code examples, and troubleshooting guides with consistent formatting and accuracy.

**Research Dataset Creation** generates datasets for academic research, including specialized domains and custom evaluation metrics.

**Model Training Datasets** creates conversation datasets, instruction-following examples, and domain-specific training data optimized for language model fine-tuning.

## Example Sections

Detailed examples are organized into focused sections:

[**Basic Usage**](basic-usage.md) - Simple configurations and common patterns
[**Advanced Workflows**](advanced-workflows.md) - Complex multi-stage generation processes
[**Agent Tool-Calling**](agent-tool-calling.md) - Agent reasoning with systematic tool usage
[**Hugging Face Integration**](huggingface-integration.md) - Dataset publishing and sharing patterns

Each section includes complete, runnable examples with detailed explanations of configuration choices, expected outputs, and adaptation guidance for different requirements.

## Adaptation Guidance

Every example includes guidance for adapting the configuration to different domains, scales, or quality requirements. This guidance covers parameter selection rationale, provider choice implications, and common modifications that users typically need to make.

The examples serve as both learning resources and practical templates that you can modify for your specific use cases, reducing development time and ensuring adherence to established best practices.