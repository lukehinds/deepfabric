# YAML Configuration Guide for Chain of Thought

This guide provides comprehensive documentation for configuring Chain of Thought datasets using YAML files. DeepFabric's YAML configuration system allows you to define topic generation, data generation parameters, and output settings in a declarative format.

## Configuration Structure Overview

```yaml
# High-level structure of CoT configuration
dataset_system_prompt: "System prompt for the assistant"

topic_tree:           # OR topic_graph:
  # Topic generation settings

data_engine:
  # Data generation settings
  # CoT-specific parameters

dataset:
  creation:
    # Generation control parameters
  save_as: "output.jsonl"

huggingface:          # Optional
  # HuggingFace Hub upload settings
```

## Complete Configuration Examples

### Free-text CoT Configuration

```yaml
# free-text-math-reasoning.yaml
# Optimized for mathematical word problems with natural language reasoning

dataset_system_prompt: "You are a helpful math tutor who explains problems step-by-step with clear reasoning."

# Topic generation using Tree structure (hierarchical)
topic_tree:
  topic_prompt: "Elementary and middle school mathematics including arithmetic, basic algebra, geometry, and word problems"

  # LLM settings for topic generation
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7        # Higher creativity for topic diversity

  # Tree structure parameters
  degree: 3               # 3 subtopics per node
  depth: 2                # 2 levels deep

  # Output
  save_as: "math_topics.jsonl"

# Data generation engine
data_engine:
  instructions: "Create clear mathematical word problems that require step-by-step analytical thinking to solve."
  generation_system_prompt: "You are a mathematics educator creating practice problems with detailed reasoning explanations."

  # LLM settings for data generation
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.3        # Lower temperature for consistent reasoning
  max_retries: 4

  # Chain of Thought specific settings
  conversation_type: "cot_freetext"
  reasoning_style: "mathematical"

# Dataset assembly
dataset:
  creation:
    num_steps: 15         # Generate 15 examples
    batch_size: 1         # Process one at a time
    sys_msg: false        # Free-text CoT doesn't use system messages

  save_as: "math_reasoning_freetext.jsonl"

# Optional: Upload to HuggingFace Hub
# huggingface:
#   repository: "username/math-reasoning-cot"
#   tags: ["mathematics", "reasoning", "chain-of-thought", "deepfabric"]
```

### Structured CoT Configuration

```yaml
# structured-cs-education.yaml
# Optimized for computer science educational dialogues

dataset_system_prompt: "You are a computer science instructor who guides students through systematic problem-solving approaches."

# Topic generation using Graph structure (interconnected)
topic_graph:
  topic_prompt: "Computer science fundamentals including algorithms, data structures, programming concepts, and computational thinking"

  # LLM settings
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.6

  # Graph structure (more complex than tree)
  degree: 2
  depth: 3

  # Output
  save_as: "cs_topics_graph.json"

# Data generation with structured conversations
data_engine:
  instructions: "Create educational conversations where students learn computer science concepts through guided discovery and systematic reasoning."
  generation_system_prompt: "You are a CS instructor creating realistic teaching dialogues with explicit reasoning steps."

  # LLM settings
  provider: "openai"
  model: "gpt-4o-mini"  # Consider gpt-4o for more complex conversations
  temperature: 0.4
  max_retries: 3

  # Structured CoT settings
  conversation_type: "cot_structured"
  reasoning_style: "logical"

# Dataset assembly
dataset:
  creation:
    num_steps: 8          # Fewer steps due to complexity
    batch_size: 1
    sys_msg: true         # Include system messages in conversations

  save_as: "cs_education_structured.jsonl"

# HuggingFace upload with rich metadata
huggingface:
  repository: "deepfabric/cs-education-structured-cot"
  tags:
    - "computer-science"
    - "education"
    - "algorithms"
    - "structured-reasoning"
    - "chain-of-thought"
    - "deepfabric"
  description: "Educational computer science conversations with structured reasoning traces"
```

### Hybrid CoT Configuration

```yaml
# hybrid-scientific-reasoning.yaml
# Optimized for complex scientific problems requiring multi-modal reasoning

dataset_system_prompt: "You are a scientific expert who combines intuitive insights with rigorous systematic analysis."

# Topic generation for complex scientific domains
topic_tree:
  topic_prompt: "Advanced scientific problems in physics, chemistry, biology, and interdisciplinary research requiring both intuitive understanding and systematic analysis"

  # Higher-capability model for complex topics
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.5

  # Moderate complexity
  degree: 2
  depth: 2

  save_as: "scientific_topics.jsonl"

# Data generation requiring dual reasoning modes
data_engine:
  instructions: "Create challenging scientific problems that require both intuitive conceptual understanding and systematic step-by-step analysis."
  generation_system_prompt: "You are a research scientist who excels at combining natural scientific insight with rigorous methodical reasoning."

  # Premium model recommended for hybrid reasoning
  provider: "openai"
  model: "gpt-4o"        # Higher capability needed
  temperature: 0.3
  max_retries: 5         # More retries due to complexity

  # Hybrid CoT settings
  conversation_type: "cot_hybrid"
  reasoning_style: "logical"  # Could also be "general" for interdisciplinary

# Dataset assembly (smaller scale due to complexity)
dataset:
  creation:
    num_steps: 5          # Fewer due to high complexity and cost
    batch_size: 1         # Always 1 for hybrid format
    sys_msg: false

  save_as: "scientific_reasoning_hybrid.jsonl"
```

## Parameter Reference

### Global Parameters

#### `dataset_system_prompt`
- **Type**: String
- **Required**: Yes
- **Description**: System prompt that defines the assistant's role and behavior
- **Best Practices**:
  - Keep concise but descriptive
  - Match the reasoning style and domain
  - Emphasize step-by-step thinking for CoT formats

```yaml
# Examples by format
dataset_system_prompt: "You are a helpful teacher who shows step-by-step reasoning."  # Free-text
dataset_system_prompt: "You are an instructor who guides students systematically."    # Structured
dataset_system_prompt: "You are an expert who combines intuition with analysis."     # Hybrid
```

### Topic Generation Parameters

You can use either `topic_tree` (hierarchical) or `topic_graph` (interconnected):

#### `topic_prompt`
- **Type**: String
- **Required**: Yes
- **Description**: Root prompt that defines the domain and scope of topics
- **Best Practices**:
  - Be specific about the domain and complexity level
  - Include examples of the types of problems desired
  - Mention target audience if relevant

```yaml
# Domain-specific examples
topic_prompt: "Elementary mathematics word problems suitable for grades 3-6"
topic_prompt: "Computer science algorithms and data structures for undergraduate level"
topic_prompt: "Advanced physics problems requiring multi-step problem solving"
```

#### `provider` and `model`
- **Type**: String
- **Required**: Yes
- **Description**: LLM provider and model for topic generation
- **Options**:
  - `openai`: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`
  - `anthropic`: `claude-sonnet-4-5`, `claude-3-sonnet`, `claude-3-haiku`
  - `gemini`: `gemini-pro`, `gemini-2.5-flash-lite`
  - `ollama`: `mistral:latest`, `llama3:latest`, etc.

```yaml
# Recommended combinations
provider: "openai"
model: "gpt-4o-mini"     # Good balance of quality and cost

provider: "openai"
model: "gpt-4o"          # Higher quality for complex topics

provider: "anthropic"
model: "claude-3-sonnet" # Alternative high-quality option
```

#### `temperature`
- **Type**: Float (0.0 - 2.0)
- **Default**: 0.7
- **Description**: Controls creativity in topic generation
- **Recommendations**:
  - `0.5-0.7`: Balanced creativity and consistency
  - `0.7-1.0`: More diverse topics
  - `0.2-0.5`: More focused, consistent topics

#### `degree` and `depth`
- **Type**: Integer
- **Description**: Control topic tree/graph structure
- **degree**: Number of subtopics per node
- **depth**: Number of levels in the structure

```yaml
# Small dataset (4-9 topics)
degree: 2
depth: 2

# Medium dataset (8-27 topics)
degree: 3
depth: 2

# Large dataset (16-64 topics)
degree: 4
depth: 3
```

### Data Engine Parameters

#### Core Generation Settings

```yaml
data_engine:
  instructions: "High-level guidance for the type of problems to create"
  generation_system_prompt: "Specific role definition for the data generation model"

  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.3        # Lower than topic generation
  max_retries: 4
```

#### Chain of Thought Specific Settings

##### `conversation_type`
- **Type**: Enum
- **Required**: Yes for CoT
- **Options**:
  - `"cot_freetext"`: Natural language reasoning
  - `"cot_structured"`: Conversations with reasoning traces
  - `"cot_hybrid"`: Both natural and structured reasoning
- **Default**: `"basic"` (non-CoT)

##### `reasoning_style`
- **Type**: Enum
- **Required**: No
- **Options**:
  - `"mathematical"`: Numerical calculations, algebraic reasoning
  - `"logical"`: Premise-conclusion chains, formal logic
  - `"general"`: Flexible reasoning adapting to problem type
- **Default**: `"general"`

```yaml
# Style-specific configurations
conversation_type: "cot_freetext"
reasoning_style: "mathematical"  # For math word problems

conversation_type: "cot_structured"
reasoning_style: "logical"       # For CS algorithms

conversation_type: "cot_hybrid"
reasoning_style: "general"       # For interdisciplinary problems
```

### Dataset Assembly Parameters

```yaml
dataset:
  creation:
    num_steps: 10         # Number of examples to generate
    batch_size: 1         # Samples per batch (recommend 1 for CoT)
    sys_msg: false        # Whether to include system messages

  save_as: "output.jsonl" # Output filename
```

#### `num_steps`
- **Type**: Integer
- **Description**: Number of examples to generate
- **Recommendations by format**:
  - Free-text CoT: 10-50 (efficient generation)
  - Structured CoT: 5-20 (more complex)
  - Hybrid CoT: 3-10 (most complex and expensive)

#### `batch_size`
- **Type**: Integer
- **Default**: 1
- **Description**: Number of samples to generate per batch
- **Recommendation**: Always use 1 for CoT formats for best quality

#### `sys_msg`
- **Type**: Boolean
- **Description**: Whether to include system messages in the final dataset
- **Recommendations**:
  - Free-text CoT: `false` (no conversations)
  - Structured CoT: `true` (part of conversation)
  - Hybrid CoT: `false` (standalone reasoning)

### HuggingFace Integration

```yaml
huggingface:
  repository: "username/dataset-name"
  description: "Dataset description for the Hub"
  tags:
    - "chain-of-thought"
    - "reasoning"
    - "mathematics"  # Domain-specific tags
    - "deepfabric"
  private: false            # Set to true for private repos
  license: "mit"            # Optional license specification
```

## Configuration Best Practices

### Model Selection Guidelines

#### For Topic Generation
- **Simple domains**: `gpt-4o-mini`, `claude-3-haiku`
- **Complex domains**: `gpt-4o`, `claude-3-sonnet`
- **Interdisciplinary**: `gpt-4o`, `claude-sonnet-4-5`

#### For Data Generation
- **Free-text CoT**: `gpt-4o-mini` sufficient
- **Structured CoT**: `gpt-4o-mini` or `gpt-4o`
- **Hybrid CoT**: `gpt-4o` recommended

### Temperature Guidelines

```yaml
# Conservative approach (more consistent)
topic_tree:
  temperature: 0.6
data_engine:
  temperature: 0.2

# Balanced approach (recommended)
topic_tree:
  temperature: 0.7
data_engine:
  temperature: 0.3

# Creative approach (more diverse)
topic_tree:
  temperature: 0.8
data_engine:
  temperature: 0.4
```

### Cost Optimization

```yaml
# Budget-friendly configuration
provider: "openai"
model: "gpt-4o-mini"      # Most cost-effective
conversation_type: "cot_freetext"  # Simplest format
num_steps: 10             # Moderate dataset size
batch_size: 1
temperature: 0.3          # Reduce need for retries
```

```yaml
# Quality-focused configuration
provider: "openai"
model: "gpt-4o"           # Highest quality
conversation_type: "cot_hybrid"    # Most comprehensive
num_steps: 5              # Smaller but higher quality
temperature: 0.2          # Very consistent
max_retries: 5            # Ensure success
```

## Environment Variables

Set required API keys:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google Gemini
export GOOGLE_API_KEY="..."

# HuggingFace (for uploads)
export HUGGINGFACE_API_TOKEN="hf_..."
```

## Validation and Testing

### Configuration Validation
```bash
# Validate your configuration before running
deepfabric validate my-config.yaml
```

### Test Runs
```bash
# Small test run to verify settings
num_steps: 2              # Override in config for testing
```

### Monitoring Generation
```bash
# Run with verbose output to monitor progress
deepfabric generate config.yaml --verbose
```

## Common Configuration Patterns

### Academic Research Dataset
```yaml
# For research papers and academic use
dataset_system_prompt: "You are a research assistant who provides rigorous scientific reasoning."

topic_tree:
  topic_prompt: "Advanced research problems in [domain] requiring systematic investigation"
  model: "gpt-4o"
  temperature: 0.5

data_engine:
  conversation_type: "cot_hybrid"
  reasoning_style: "logical"
  model: "gpt-4o"
  temperature: 0.2

dataset:
  creation:
    num_steps: 25
```

### Educational Content
```yaml
# For tutoring and educational applications
dataset_system_prompt: "You are a patient teacher who explains concepts clearly."

topic_tree:
  topic_prompt: "Educational problems suitable for [grade level] students"
  temperature: 0.7

data_engine:
  conversation_type: "cot_structured"
  reasoning_style: "general"
  temperature: 0.4

dataset:
  creation:
    sys_msg: true  # Include teaching context
```

### Industry Training Data
```yaml
# For commercial applications
dataset_system_prompt: "You are a professional expert who provides practical solutions."

topic_tree:
  topic_prompt: "Real-world problems in [industry domain] requiring professional expertise"

data_engine:
  conversation_type: "cot_freetext"  # Efficient for large datasets
  reasoning_style: "general"

dataset:
  creation:
    num_steps: 100  # Large-scale generation
```

## Troubleshooting Configuration Issues

### Common Errors

#### "Invalid conversation_type"
```yaml
# Incorrect
conversation_type: "cot-freetext"  # Underscore, not hyphen

# Correct
conversation_type: "cot_freetext"
```

#### "Provider API key not found"
```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
```

#### "Schema validation failed"
```yaml
# Ensure all required fields are present
dataset_system_prompt: "Required field"
topic_tree:  # Either topic_tree OR topic_graph required
  topic_prompt: "Required field"
data_engine:
  conversation_type: "Required for CoT"
```

### Performance Issues

#### Slow generation
- Reduce `num_steps` for testing
- Use smaller models (`gpt-4o-mini`)
- Increase `batch_size` (with caution for CoT)

#### Poor quality output
- Increase model capability (`gpt-4o`)
- Lower `temperature` (0.2-0.3)
- Adjust prompts for clarity
- Use appropriate `reasoning_style`

#### High costs
- Use `gpt-4o-mini` instead of `gpt-4o`
- Choose `cot_freetext` over `cot_hybrid`
- Reduce `num_steps`
- Optimize `temperature` to reduce retries

## Next Steps

- **Python API Configuration**: → [Python API Guide](python-api.md)
- **Math Reasoning Tutorial**: → [Math Reasoning Tutorial](../tutorials/math-reasoning.md)
- **Advanced Reasoning Styles**: → [Reasoning Styles Guide](../advanced/reasoning-styles.md)
- **Schema Reference**: → [Schema Reference](../reference/schemas.md)