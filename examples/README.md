# DeepFabric Examples

This directory contains comprehensive examples showing how to use DeepFabric both as a library (programmatically) and via YAML configuration files.

## Quick Start

For the fastest way to get started, run:

```bash
python example_minimal_quickstart.py
```

This creates a simple synthetic dataset about Python programming fundamentals.

## Programmatic Examples (Python Library)

### Basic Examples

- **`example_minimal_quickstart.py`** - Minimal example to get you started
- **`example_culinary_database.py`** - Generates a culinary recipe database
- **`example_custom_topics.py`** - Shows how to define your own topic structure instead of auto-generating

### Advanced Topic Generation

- **`example_topic_graph_programmatic.py`** - Uses the new Graph feature with cross-connections between topics
- **`example_load_existing_graph.py`** - Shows how to load and reuse previously generated graphs

### Multi-Model and Pipeline Examples

- **`example_different_models.py`** - Demonstrates using different LLM providers (OpenAI, Anthropic, Google, Ollama)
- **`example_advanced_pipeline.py`** - Complete production pipeline with error handling, validation, and HuggingFace integration

## YAML Configuration Examples

### Basic Configuration

- **`example_basic_prompt.yaml`** - Basic topic tree generation
- **`example_tree_config.yaml`** - Complete tree-based configuration
- **`example_graph_config.yaml`** - Complete graph-based configuration with cross-connections

### Usage

```bash
# Using topic trees (hierarchical)
deepfabric start example_tree_config.yaml

# Using topic graphs (with cross-connections)  
deepfabric start example_graph_config.yaml

# With parameter overrides
deepfabric start example_basic_prompt.yaml --model gpt-4 --temperature 0.8
```

## Key Concepts

### Topic Trees vs Topic Graphs

- **Topic Trees**: Hierarchical structure, no cross-connections between topics
- **Topic Graphs**: Allow cross-connections between topics for more complex relationships

### Configuration Structure

Both trees and graphs support the same core parameters:

```yaml
# For trees
topic_tree:
  topic_prompt: "Your topic here"
  provider: "ollama"              # LLM provider
  model: "qwen3:8b"                 # Model name
  temperature: 0.7                # Generation temperature
  degree: 3                  # Branches per level
  depth: 2                   # Tree depth
  save_as: "output.jsonl"

topic_graph:
  topic_prompt: "Your topic here"
  provider: "ollama"              # LLM provider
  model: "qwen3:8b"                 # Model name
  temperature: 0.7                # Generation temperature
  degree: 3                 # Subtopics per node
  depth: 2                  # Graph depth
  save_as: "output.json"
```

## Model Support

DeepFabric supports multiple LLM providers via LiteLLM:

- **Ollama (Local)**: `ollama/qwen3:8b`, `ollama/mistral`, `ollama/codellama`
- **OpenAI**: `gpt-4`, `gpt-4o-mini`, `gpt-3.5-turbo`
- **Anthropic**: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`, `claude-3-haiku-20240307`
- **Google**: `gemini/gemini-pro`, `gemini/gemini-2.5-flash-lite`
- **Groq**: `groq/qwen3:8b-8b-8192`, `groq/mixtral-8x7b-32768`

Set API keys as environment variables (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).

## Required Parameters

When using DeepFabric programmatically, ensure all required parameters are provided:

### TreeArguments
```python
TreeArguments(
    topic_prompt="...",                    # Required
    model_name="ollama/qwen3:8b",          # Required
    topic_system_prompt="...",           # Required
    degree=3,                       # Required
    depth=2,                        # Required
    temperature=0.7,                     # Required
)
```

### DataSetGeneratorArguments
```python
DataSetGeneratorArguments(
    instructions="...",                   # Required
    generation_system_prompt="...",                 # Required
    model_name="ollama/qwen3:8b",          # Required
    prompt_template=None,                # Required (can be None)
    example_data=None,                   # Required (can be None)
    temperature=0.7,                     # Required
    max_retries=3,                       # Required
    default_batch_size=5,                # Required
    default_num_examples=3,              # Required
    request_timeout=30,                  # Required
    sys_msg=True,                        # Required
)
```

## Tips for Success

1. **Start Simple**: Begin with `example_minimal_quickstart.py` to understand the basics
2. **Use Local Models**: Ollama models are free and don't require API keys
3. **Experiment with Temperature**: Lower values (0.1-0.3) for technical content, higher (0.7-0.9) for creative content
4. **Validate Your Data**: Use the advanced pipeline example as a template for production usage
5. **Save Intermediate Results**: Always save your topic trees/graphs so you can reuse them

## Troubleshooting

- **Missing API Keys**: Set environment variables like `OPENAI_API_KEY`
- **Ollama Connection**: Ensure Ollama is running with `ollama serve`
- **Parameter Errors**: Check that all required parameters are provided
- **Memory Issues**: Reduce batch sizes or use smaller models for large datasets

## Next Steps

After trying these examples:

1. Modify the prompts and topics for your specific use case
2. Experiment with different models and parameters
3. Build your own validation and quality checks
4. Consider uploading to HuggingFace Hub for sharing