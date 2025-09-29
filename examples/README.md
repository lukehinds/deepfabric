# DeepFabric Examples

This directory contains practical examples demonstrating different DeepFabric usage patterns, from basic quickstart to production pipelines.

Note that for a good amount of these, we use qwen3:8b as its a fast model to pull and being low parameter, its also fast. This
makes it useful for showing how things work. It is however not a good candidate for distillation as it has limitations around it the depth of
its training and therefore has limited knowledge compared to the bigger models.

## Example Files Overview

### Python Examples

| File | Complexity | Purpose | Key Features |
|------|------------|---------|--------------|
| **quickstart.py** | Beginner | Minimal working example | Tree generation, dataset creation, ~65 lines |
| **programmatic_usage.py** | Intermediate | Comprehensive patterns | All generation modes, error handling, statistics |
| **integrations.py** | Intermediate | External systems | YAML loading, multi-provider, HF Hub, CLI patterns |
| **production_pipeline.py** | Advanced | Production-ready workflow | Quality validation, monitoring, retry logic, metadata |
| **agent_tool_calling_example.py** | Advanced | Agent tool-calling datasets | Chain of thought reasoning with embedded tool execution traces |

### YAML Configuration Examples

| File | Mode | Purpose | Features |
|------|------|---------|----------|
| **basic.yaml** | Tree | Simple beginner config | Single provider, clear comments, minimal settings |
| **advanced.yaml** | Graph | Multi-provider setup | Cross-connections, different models per stage |
| **specialized.yaml** | Tree | Domain-specific prompts | Medical education example, specialized system prompts |
| **agent_tool_calling_config.yaml** | Tree | Agent tool-calling with rich CoT | Tool reasoning, embedded execution traces, formatter integration |

## Quick Start

1. **Complete beginner**: Start with `quickstart.py` and `basic.yaml`
2. **Exploring features**: Try `programmatic_usage.py` to see all capabilities
3. **Agent tool-calling**: Use `agent_tool_calling_example.py` for chain of thought reasoning with tools
4. **External integrations**: Use `integrations.py` for YAML configs and HF Hub
5. **Production deployment**: Follow `production_pipeline.py` for robust workflows

## Running Examples

### Prerequisites
```bash
# Install with all dependencies
uv sync --all-extras

# Set API keys (choose your provider)
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
# export GEMINI_API_KEY=your-key-here
# For local models: ensure Ollama is running
```

### Python Examples
```bash
# Run any Python example directly
python examples/quickstart.py
python examples/programmatic_usage.py
python examples/agent_tool_calling_example.py
python examples/integrations.py
python examples/production_pipeline.py
```

### YAML Configuration Examples
```bash
# CLI usage with YAML configs
deepfabric start examples/basic.yaml
deepfabric start examples/advanced.yaml --model gpt-4o  # Override model
deepfabric start examples/specialized.yaml
deepfabric start examples/agent_tool_calling_config.yaml
```

## Example Descriptions

### quickstart.py
- **Purpose**: Minimal example to get started quickly
- **What it does**: Creates a 3x2 topic tree about Python programming, generates 6 training samples
- **Best for**: Learning the basic API, testing your setup
- **Output**: `python_quickstart_tree.jsonl`, `python_quickstart_dataset.jsonl`

### programmatic_usage.py
- **Purpose**: Comprehensive demonstration of all DeepFabric features
- **What it does**: Shows tree vs graph generation, error handling, statistics, different output formats
- **Best for**: Understanding all capabilities, choosing the right approach for your project
- **Output**: Multiple files showing different generation modes

### integrations.py
- **Purpose**: External system integrations and production workflows
- **What it does**: YAML config loading, multi-provider setups, HuggingFace Hub integration, CLI patterns
- **Best for**: Production deployments, complex configurations, sharing datasets
- **Output**: Demonstrates configuration patterns and upload workflows

### production_pipeline.py
- **Purpose**: Production-ready pipeline with monitoring and quality control
- **What it does**: Advanced error recovery, quality validation, statistics collection, metadata generation
- **Best for**: Large-scale dataset generation, quality-critical applications
- **Output**: Dataset + comprehensive metadata and statistics

### agent_tool_calling_example.py
- **Purpose**: Agent chain of thought reasoning with tool calling capabilities
- **What it does**: Generates datasets showing step-by-step reasoning for tool selection and usage, then formats them into embedded execution traces
- **Best for**: Training models to reason about tool usage, function calling, and multi-step problem solving
- **Output**: Raw agent reasoning data + formatted embedded execution traces with `<think>`, `<tool_call>`, and `<tool_response>` tags

## Configuration Examples

### basic.yaml
Simple tree-based generation perfect for beginners:
- Single provider (Ollama with local model)
- Clear documentation and comments
- Conservative settings for reliable results

### advanced.yaml
Graph-based generation with multiple providers:
- Gemini for fast topic generation
- OpenAI for high-quality content
- Anthropic for final dataset assembly
- Demonstrates cost/quality optimization

### specialized.yaml
Domain-specific prompts for medical education:
- Specialized system prompts for each pipeline stage
- Higher quality settings for accuracy-critical domains
- Shows prompt engineering for specific use cases

### agent_tool_calling_config.yaml
Agent chain of thought with tool calling and embedded execution traces:
- Uses agent_cot_tools conversation type for structured reasoning
- Includes tool-calling formatter for embedded execution format
- Demonstrates step-by-step reasoning, tool selection, and result interpretation
- Produces training data with `<think>`, `<tool_call>`, and `<tool_response>` tags

## Tips for Success

1. **Start Small**: Begin with `quickstart.py` and low `num_steps` values
2. **Provider Choice**: Use Ollama for development, cloud providers for production
3. **Quality vs Cost**: Mix providers - fast for topics, high-quality for content
4. **Validation**: Always validate your configuration before large runs
5. **Monitoring**: Use the production pipeline pattern for quality control

## Common Patterns

### Basic Pipeline
```python
import asyncio

# 1. Create topic structure
tree = Tree(topic_prompt="Your topic", model_name="provider/model", degree=3, depth=2)

async def build_tree() -> None:
    async for _ in tree.build_async():
        pass

asyncio.run(build_tree())

# 2. Generate dataset
engine = DataSetGenerator(instructions="Your instructions", model_name="provider/model")
dataset = engine.create_data(num_steps=10, topic_model=tree)

# 3. Save results
dataset.save("output.jsonl")
```

### Configuration-Driven
```python
# Load from YAML
config = DeepFabricConfig.from_yaml("config.yaml")

# Use configuration parameters
tree_params = config.get_tree_params()
engine_params = config.get_engine_params()
dataset_config = config.get_dataset_config()
```

### Quality Control
```python
# Validate before generation
if len(tree.tree_paths) < num_steps * batch_size:
    print("Insufficient topic paths - increase tree size or reduce dataset size")

# Filter results
valid_samples = [s for s in dataset.samples if validate_sample(s)]
```

## Troubleshooting

- **"Insufficient topic paths"**: Increase tree `degree`/`depth` or reduce `num_steps`
- **Empty responses**: Check API keys and model availability
- **JSON parsing errors**: Retry with lower `temperature` or different model
- **Rate limits**: Reduce `batch_size` or add delays between requests

## Next Steps

After running these examples:
1. Adapt configurations for your specific domain
2. Experiment with different prompt templates
3. Set up quality validation for your use case
4. Configure CI/CD pipelines for dataset updates
5. Share datasets via HuggingFace Hub integration