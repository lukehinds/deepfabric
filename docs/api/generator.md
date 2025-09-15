# DataSetGenerator API

The DataSetGenerator class transforms topic structures into practical training examples through configurable templates, quality control mechanisms, and batch processing. This API provides comprehensive control over the dataset creation process from topic selection through content validation.

## DataSetGenerator Configuration

Dataset generation configuration is passed directly to the DataSetGenerator constructor:

```python
from deepfabric import DataSetGenerator

generator = DataSetGenerator(
    instructions="Create detailed explanations with practical examples for intermediate learners.",
    generation_system_prompt="You are an expert instructor creating educational content.",
    model_name="openai/gpt-4",
    temperature=0.8,
    max_retries=3,
    request_timeout=30,
    default_batch_size=5,
    default_num_examples=3
)
```

### Parameters

**instructions** (str): Core guidance for content generation specifying format, complexity, target audience, and quality expectations.

**generation_system_prompt** (str): System prompt providing behavioral context for the generation model.

**model** (str): model specification in `provider/model` format.

**provider** (str): provider name , e.g `openai`, `anthropic`.

**temperature** (float): Controls creativity and diversity in content generation. Range 0.0-2.0, typically 0.7-0.9.

**max_retries** (int): Number of retry attempts for failed generation requests.

**request_timeout** (int): Maximum seconds to wait for API responses.

**default_batch_size** (int): Default number of examples to generate per API call.

**default_num_examples** (int): Default number of examples to generate when not specified.

## DataSetGenerator Class

The DataSetGenerator class orchestrates the conversion from topics to training examples:

```python
from deepfabric import DataSetGenerator, Tree

# Create generator
generator = DataSetGenerator(
    instructions="Create detailed educational content",
    generation_system_prompt="You are an expert instructor",
    model_name="openai/gpt-4",
    temperature=0.8
)

# Generate dataset from topic model
dataset = generator.create_data(
    num_steps=100,
    batch_size=5,
    topic_model=tree,
    model_name="anthropic/claude-3-opus",
    sys_msg=True
)
```

### Core Methods

#### create_data()

Primary method for generating complete datasets:

```python
dataset = generator.create_data(
    num_steps=100,              # Total examples to generate
    batch_size=5,               # Examples per API call
    topic_model=topic_model,    # Tree or Graph instance
    model_name=None,            # Override model (optional)
    sys_msg=True                # Include system messages
)
```

**Parameters:**

- **num_steps** (int): Total number of training examples to generate
- **batch_size** (int): Number of examples processed in each API call
- **topic_model** (Tree | Graph): Source of topics for generation
- **model_name** (str, optional): Override the configured model
- **sys_msg** (bool): Include system prompts in output format

**Returns:** Dataset instance containing generated training examples

#### create_batch()

Generate a single batch of examples for fine-grained control:

```python
batch = generator.create_batch(
    topics=selected_topics,
    batch_size=3,
    model_name="openai/gpt-3.5-turbo"
)
```

Enables custom topic selection and incremental dataset building.

#### validate_configuration()

Check generator configuration for common issues:

```python
issues = generator.validate_configuration()
if issues:
    for issue in issues:
        print(f"Configuration issue: {issue}")
```

Returns list of configuration problems that might affect generation quality or reliability.

### Template System

The generator uses a flexible template system for content creation:

#### Default Templates

Built-in templates handle common use cases:

```python
# Instructional content
generator.set_template("instruction", """
Create a clear explanation of {topic} suitable for {audience_level}.
Include practical examples and key concepts.
""")

# Conversational format
generator.set_template("conversation", """
Generate a natural conversation about {topic} that demonstrates
helpful, informative dialogue.
""")
```

#### Custom Templates

Define domain-specific templates:

```python
custom_template = """
Create a {content_type} about {topic} that includes:
1. Clear definition
2. Practical code example
3. Common pitfalls to avoid
4. Best practices

Target audience: {audience}
Complexity level: {complexity}
"""

generator.set_custom_template(custom_template)
```

#### Template Variables

Templates support variable substitution:

- **{topic}**: Current topic being processed
- **{context}**: Additional context from topic hierarchy
- **{examples}**: Related examples from the domain
- **{audience}**: Target audience specification
- **{complexity}**: Desired complexity level

### Quality Control

Multiple quality control mechanisms ensure consistent output:

#### Content Filtering

Apply filters to generated content:

```python
def quality_filter(content, topic, metadata):
    # Check content quality criteria
    if len(content) < 100:
        return False, "Content too short"
    if "inappropriate" in content.lower():
        return False, "Inappropriate content detected"
    return True, "Acceptable"

generator.add_content_filter(quality_filter)
```

#### Retry Strategies

Configure retry behavior for failed generations:

```python
generator.set_retry_strategy(
    max_retries=5,
    backoff_multiplier=2.0,
    max_backoff_seconds=60,
    retry_on_errors=["timeout", "rate_limit", "json_parse_error"]
)
```

#### Statistical Monitoring

Monitor generation statistics in real-time:

```python
generator.enable_monitoring(verbose=True)
dataset = generator.create_data(...)

stats = generator.get_generation_stats()
print(f"Success rate: {stats.success_rate:.2%}")
print(f"Average retry count: {stats.avg_retries:.1f}")
print(f"Error breakdown: {stats.error_categories}")
```

### Advanced Usage

#### Topic Sampling Strategies

Control how topics are selected from the topic model:

```python
# Sequential sampling (default)
dataset = generator.create_data(topic_model=tree, sampling_strategy="sequential")

# Random sampling with replacement
dataset = generator.create_data(topic_model=tree, sampling_strategy="random")

# Balanced sampling across tree branches
dataset = generator.create_data(topic_model=tree, sampling_strategy="balanced")

# Custom sampling function
def custom_sampler(topic_model, count):
    # Implement domain-specific sampling logic
    return selected_topics

dataset = generator.create_data(topic_model=tree, topic_sampler=custom_sampler)
```

#### Multi-Provider Generation

Use different models for different types of content:

```python
# High-quality generator for complex topics
complex_generator = DataSetGenerator(
    instructions="Create advanced technical content",
    model_name="anthropic/claude-3-opus",
    temperature=0.7
)
complex_topics = tree.get_topics_at_depth(3)
complex_dataset = complex_generator.create_data(
    topics=complex_topics
)

# Faster generator for simple topics
simple_generator = DataSetGenerator(
    instructions="Create basic explanations",
    model_name="openai/gpt-3.5-turbo",
    temperature=0.8
)
simple_topics = tree.get_topics_at_depth(1)
simple_dataset = simple_generator.create_data(
    topics=simple_topics
)
```

#### Incremental Generation

Build datasets incrementally with progress tracking:

```python
dataset = Dataset()
total_steps = 1000
batch_size = 10

for i in range(0, total_steps, batch_size):
    batch = generator.create_batch(
        topics=topic_model.sample_topics(batch_size),
        batch_size=batch_size
    )
    dataset.extend(batch)
    
    # Save progress periodically
    if i % 100 == 0:
        dataset.save(f"checkpoint_{i}.jsonl")
        print(f"Progress: {i}/{total_steps} examples")
```

### Error Handling

Comprehensive error handling for robust operation:

```python
from deepfabric import DataSetGeneratorError, ModelError, ValidationError

try:
    generator = DataSetGenerator(
        instructions="Create educational content",
        model_name="openai/gpt-4"
    )
    dataset = generator.create_data(topic_model=tree, num_steps=100)
except ModelError as e:
    print(f"Model API issue: {e}")
    # Implement fallback strategy
except ValidationError as e:
    print(f"Generated content validation failed: {e}")
except DataSetGeneratorError as e:
    print(f"Generation process error: {e}")
```

### Performance Optimization

Optimize generation performance through parameter tuning:

```python
# Optimize for throughput
generator.optimize_for_throughput(
    large_batch_sizes=True,
    parallel_requests=True,
    aggressive_timeouts=False
)

# Optimize for reliability
generator.optimize_for_reliability(
    conservative_batch_sizes=True,
    extended_timeouts=True,
    maximum_retries=True
)
```