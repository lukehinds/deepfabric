# Formatter System Overview

The DeepFabric formatter system provides a pluggable post-processing pipeline for transforming datasets into training framework-specific formats. This allows you to generate data once and format it for multiple training frameworks.

## Core Concepts

### What are Formatters?

Formatters are post-processing modules that transform DeepFabric's internal dataset format into specialized formats required by different training frameworks and methodologies:

- **TRL SFT Tools**: HuggingFace TRL SFTTrainer format with OpenAI function calling schema
- **GRPO**: Reasoning traces with working-out tags for mathematical reasoning models
- **Alpaca**: Instruction-following format for supervised fine-tuning
- **ChatML**: Conversation format with role delineation markers (structured or text)
- **Single Tool Call**: Individual tool call format for training tool usage
- **Harmony**: OpenAI Harmony Response Format for gpt-oss models
- **Conversations**: Generic conversations format compatible with Unsloth, Axolotl, HF TRL
- **Custom**: User-defined formatters for specialized use cases

### Architecture

The formatter system consists of three main components:

1. **BaseFormatter**: Abstract interface that all formatters implement
2. **FormatterRegistry**: Loads and manages formatters (built-in and custom)
3. **Dataset Integration**: Applies formatters to datasets with configuration

## Loading Mechanisms

### Built-in Formatters (`builtin://`)

Built-in formatters are provided by DeepFabric and located in `deepfabric.formatters.builtin`:

```yaml
formatters:
- name: "grpo"
  template: "builtin://grpo.py"
  config:
    reasoning_start_tag: "<start_working_out>"
    reasoning_end_tag: "<end_working_out>"
```

### Custom Formatters (`file://`)

Custom formatters are user-defined Python files that implement the BaseFormatter interface:

```yaml
formatters:
- name: "my_custom"
  template: "file://./formatters/my_custom_formatter.py"
  config:
    custom_option: "value"
```

## Configuration Structure

Formatters are configured in your YAML configuration file under the `dataset.formatters` section:

```yaml
dataset:
  creation:
    num_steps: 100
    batch_size: 4
  save_as: "raw_dataset.jsonl"
  formatters:
    - name: "grpo_math"
      template: "builtin://grpo.py"
      config:
        reasoning_start_tag: "<think>"
        reasoning_end_tag: "</think>"
        solution_start_tag: "<answer>"
        solution_end_tag: "</answer>"
      output: "grpo_formatted.jsonl"

    - name: "alpaca_instruct"
      template: "builtin://alpaca.py"
      config:
        instruction_template: "### Instruction:\n{instruction}\n\n### Response:"
      output: "alpaca_formatted.jsonl"
```

### Configuration Fields

- **name**: Unique identifier for the formatter instance
- **template**: Path to the formatter (`builtin://` or `file://`)
- **config**: Formatter-specific configuration options
- **output**: Optional output file path for the formatted dataset

## Workflow

1. **Dataset Generation**: DeepFabric generates the raw dataset using the configured pipeline
2. **Formatter Application**: Each configured formatter processes the raw dataset
3. **Output Generation**: Formatted datasets are saved to specified output files
4. **Validation**: Each formatter validates both input compatibility and output correctness

## Error Handling

The formatter system includes comprehensive error handling:

- **Loading Errors**: Invalid template paths or missing formatter classes
- **Configuration Errors**: Invalid formatter configuration parameters
- **Processing Errors**: Failures during dataset transformation
- **Validation Errors**: Input data incompatible with formatter requirements

## Performance Considerations

- **Caching**: Formatter classes are cached after first load for better performance
- **Parallel Processing**: Multiple formatters can be applied independently
- **Memory Efficiency**: Formatters process datasets without duplicating the source data
- **Validation**: Optional output validation can be disabled for better performance

## Next Steps

- [Built-in Formatter Reference](built-in-reference.md) - Documentation for all included formatters
- [Custom Formatter Guide](custom-formatter-guide.md) - How to create your own formatters
- [API Reference](api-reference.md) - Complete API documentation
