# Python API for Agent Tool-Calling

The DeepFabric Python API provides programmatic access to agent tool-calling dataset generation with full control over configuration, tool definitions, and generation parameters.

## Basic Usage

### Single-Turn Agent Dataset
```python
import asyncio
from deepfabric import DataSetGenerator
from deepfabric.dataset import Dataset
from deepfabric.tree import Tree

async def generate_agent_dataset():
    # Create topic tree
    tree = Tree(
        topic_prompt="Real-world scenarios requiring tool usage",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=3,
        depth=2,
        temperature=0.7
    )

    topics = await tree.generate()

    # Create agent dataset generator
    generator = DataSetGenerator(
        generation_system_prompt="You excel at systematic tool reasoning.",
        provider="openai",
        model_name="gpt-4o-mini",
        conversation_type="agent_cot_tools",
        available_tools=["get_weather", "search_web", "calculator"],
        max_tools_per_query=2,
        topics=topics
    )

    # Generate samples
    samples = await generator.generate()

    # Create and save dataset
    dataset = Dataset.from_list(samples)
    dataset.save("agent_dataset.jsonl")

    return dataset

# Run the generation
dataset = asyncio.run(generate_agent_dataset())
print(f"Generated {len(dataset)} agent samples")
```

### Multi-Turn Agent Dataset
```python
async def generate_multi_turn_dataset():
    generator = DataSetGenerator(
        generation_system_prompt="Create multi-turn conversations with progressive tool usage.",
        provider="openai",
        model_name="gpt-4o",
        conversation_type="agent_cot_multi_turn",
        available_tools=["get_weather", "book_restaurant", "calculator"],
        max_tools_per_query=3,
        temperature=0.8
    )

    samples = await generator.generate()
    dataset = Dataset.from_list(samples)
    dataset.save("multi_turn_agent.jsonl")

    return dataset
```

## Advanced Configuration

### Custom Tool Integration
```python
from deepfabric.schemas import ToolDefinition, ToolParameter
from deepfabric.tools.defaults import get_default_tools

# Define custom tools using Pydantic models
custom_booking_tool = ToolDefinition(
    name="book_restaurant",
    description="Book a restaurant reservation",
    parameters=[
        ToolParameter(
            name="restaurant",
            type="str",
            description="Restaurant name",
            required=True
        ),
        ToolParameter(
            name="party_size",
            type="int",
            description="Number of people",
            required=True
        ),
        ToolParameter(
            name="date",
            type="str",
            description="Reservation date",
            required=True
        ),
        ToolParameter(
            name="time",
            type="str",
            description="Preferred time",
            required=True
        )
    ],
    returns="Reservation confirmation with details",
    category="booking"
)

# Create generator with custom tools
generator = DataSetGenerator(
    generation_system_prompt="You are an AI agent with restaurant booking capabilities.",
    provider="openai",
    model_name="gpt-4o",
    conversation_type="agent_cot_tools",
    available_tools=["get_weather", "book_restaurant"],  # Mix default and custom
    custom_tools=[custom_booking_tool.model_dump()],  # Custom tools as dicts
    max_tools_per_query=2
)
```

### Loading Tools from Files
```python
from deepfabric.tools.loader import load_tools_from_file, get_available_tools

# Load tools from YAML file
custom_tools = load_tools_from_file("custom_tools.yaml")

# Get available tools (defaults + customs)
all_tools = get_available_tools(
    available_tools=["get_weather", "book_restaurant"],
    custom_tool_registry=custom_tools
)

generator = DataSetGenerator(
    generation_system_prompt="You have access to specialized tools.",
    provider="openai",
    model_name="gpt-4o",
    conversation_type="agent_cot_tools",
    tool_registry_path="custom_tools.yaml",
    available_tools=["get_weather", "book_restaurant", "analyze_stock"]
)
```

## Configuration Classes

### DataSetGeneratorConfig
```python
from deepfabric.generator import DataSetGeneratorConfig

# Create configuration object
config = DataSetGeneratorConfig(
    generation_system_prompt="You excel at tool reasoning.",
    provider="openai",
    model_name="gpt-4o",
    conversation_type="agent_cot_tools",
    reasoning_style="general",
    available_tools=["get_weather", "calculator"],
    max_tools_per_query=2,
    temperature=0.8,
    max_retries=3
)

# Use configuration with generator
generator = DataSetGenerator(**config.model_dump())
```

### From YAML Configuration
```python
from deepfabric.config import DeepFabricConfig

# Load from YAML file
config = DeepFabricConfig.from_yaml("agent_config.yaml")

# Extract generator parameters
engine_params = config.get_engine_params()

# Create generator
generator = DataSetGenerator(**engine_params)
```

## Provider Configuration

### OpenAI Configuration
```python
generator = DataSetGenerator(
    provider="openai",
    model_name="gpt-4o",  # or gpt-4o-mini, gpt-4-turbo
    temperature=0.8,
    max_retries=3,
    # OpenAI-specific parameters
    timeout=60,
    # API key from environment: OPENAI_API_KEY
)
```

### Anthropic Configuration
```python
generator = DataSetGenerator(
    provider="anthropic",
    model_name="claude-3-5-sonnet-20241022",
    temperature=0.7,
    max_retries=2,
    # API key from environment: ANTHROPIC_API_KEY
)
```

### Local/Ollama Configuration
```python
generator = DataSetGenerator(
    provider="ollama",
    model_name="llama3.1:8b",
    temperature=0.6,
    max_retries=5,
    # No API key required for local models
)
```

## Batch Generation

### Large Dataset Generation
```python
async def generate_large_dataset():
    generator = DataSetGenerator(
        generation_system_prompt="Create diverse tool usage scenarios.",
        provider="openai",
        model_name="gpt-4o-mini",  # Cost-effective for large datasets
        conversation_type="agent_cot_tools",
        available_tools=["get_weather", "search_web", "calculator"],
        max_tools_per_query=2
    )

    # Generate in batches
    all_samples = []
    batch_size = 10
    total_batches = 5

    for i in range(total_batches):
        print(f"Generating batch {i+1}/{total_batches}")

        batch_samples = await generator.generate()
        all_samples.extend(batch_samples)

        # Optional: save intermediate results
        if i % 2 == 0:  # Save every 2 batches
            temp_dataset = Dataset.from_list(all_samples)
            temp_dataset.save(f"agent_dataset_batch_{i}.jsonl")

    # Final dataset
    dataset = Dataset.from_list(all_samples)
    dataset.save("agent_dataset_complete.jsonl")

    return dataset
```

### Parallel Generation
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_generation():
    # Create multiple generators for different scenarios
    generators = [
        DataSetGenerator(
            generation_system_prompt="Focus on weather-related tool usage.",
            conversation_type="agent_cot_tools",
            available_tools=["get_weather", "search_web"],
            topics=weather_topics
        ),
        DataSetGenerator(
            generation_system_prompt="Focus on calculation and analysis.",
            conversation_type="agent_cot_tools",
            available_tools=["calculator", "analyze_stock"],
            topics=calculation_topics
        ),
        DataSetGenerator(
            generation_system_prompt="Focus on booking and reservations.",
            conversation_type="agent_cot_tools",
            available_tools=["book_restaurant", "search_web"],
            topics=booking_topics
        )
    ]

    # Generate in parallel
    tasks = [gen.generate() for gen in generators]
    results = await asyncio.gather(*tasks)

    # Combine results
    all_samples = []
    for samples in results:
        all_samples.extend(samples)

    return Dataset.from_list(all_samples)
```

## Quality Control

### Validation and Filtering
```python
def validate_agent_sample(sample):
    """Validate agent CoT sample quality."""
    required_fields = ["question", "tool_planning", "tool_executions", "final_answer"]

    # Check required fields
    if not all(field in sample for field in required_fields):
        return False

    # Check tool usage
    if len(sample["tool_executions"]) == 0:
        return False

    # Check reasoning quality
    if len(sample["tool_planning"]) == 0:
        return False

    return True

async def generate_validated_dataset():
    generator = DataSetGenerator(
        generation_system_prompt="Create high-quality agent reasoning examples.",
        conversation_type="agent_cot_tools",
        available_tools=["get_weather", "calculator", "search_web"],
        max_tools_per_query=3
    )

    valid_samples = []
    attempts = 0
    max_attempts = 100

    while len(valid_samples) < 50 and attempts < max_attempts:
        samples = await generator.generate()

        for sample in samples:
            if validate_agent_sample(sample):
                valid_samples.append(sample)

        attempts += 1
        print(f"Valid samples: {len(valid_samples)}, Attempts: {attempts}")

    return Dataset.from_list(valid_samples)
```

## Output Formatting

### Apply Formatters Programmatically
```python
# Generate raw dataset
dataset = await generate_agent_dataset()

# Apply formatters
formatter_configs = [
    {
        "name": "tool_calling",
        "template": "builtin://tool_calling",
        "output": "agent_tool_calling.jsonl",
        "config": {
            "system_prompt": "You are a function calling AI model.",
            "include_tools_in_system": True,
            "thinking_format": "<think>{reasoning}</think>",
            "tool_call_format": "<tool_call>\n{tool_call}\n</tool_call>",
            "tool_response_format": "<tool_response>\n{tool_output}\n</tool_response>"
        }
    }
]

formatted_datasets = dataset.apply_formatters(formatter_configs)
tool_calling_dataset = formatted_datasets["tool_calling"]

print(f"Original samples: {len(dataset)}")
print(f"Formatted samples: {len(tool_calling_dataset)}")
```

## Error Handling

### Robust Generation with Retry Logic
```python
async def robust_generation():
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            generator = DataSetGenerator(
                generation_system_prompt="Create agent tool usage examples.",
                provider="openai",
                model_name="gpt-4o-mini",
                conversation_type="agent_cot_tools",
                available_tools=["get_weather", "calculator"],
                max_retries=2  # Per-sample retries
            )

            samples = await generator.generate()

            if len(samples) > 0:
                return Dataset.from_list(samples)

        except Exception as e:
            retry_count += 1
            print(f"Generation failed (attempt {retry_count}): {e}")

            if retry_count < max_retries:
                print(f"Retrying in 5 seconds...")
                await asyncio.sleep(5)
            else:
                print("Max retries exceeded")
                raise

    return None
```

## Integration Examples

### Complete Production Workflow
```python
async def production_agent_dataset():
    """Complete production workflow for agent dataset generation."""

    # 1. Generate topic tree
    tree = Tree(
        topic_prompt="Professional scenarios requiring intelligent tool usage",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=4,
        depth=3,
        temperature=0.7
    )

    topics = await tree.generate()
    print(f"Generated {len(topics)} topics")

    # 2. Load custom tools
    custom_tools = load_tools_from_file("production_tools.yaml")

    # 3. Generate agent dataset
    generator = DataSetGenerator(
        generation_system_prompt="Create realistic professional agent scenarios.",
        provider="openai",
        model_name="gpt-4o",
        conversation_type="agent_cot_tools",
        available_tools=["get_weather", "search_web", "book_restaurant", "analyze_stock"],
        custom_tools=[tool.model_dump() for tool in custom_tools.tools],
        max_tools_per_query=3,
        temperature=0.8,
        topics=topics
    )

    samples = await generator.generate()

    # 4. Create and validate dataset
    dataset = Dataset.from_list(samples)

    # 5. Apply multiple formatters
    formatters = [
        {
            "name": "tool_calling",
            "template": "builtin://tool_calling",
            "output": "production_tool_calling.jsonl"
        },
        {
            "name": "conversation",
            "template": "builtin://conversation",
            "output": "production_conversation.jsonl"
        }
    ]

    formatted_datasets = dataset.apply_formatters(formatters)

    # 6. Save results
    dataset.save("production_agent_raw.jsonl")

    return {
        "raw_dataset": dataset,
        "formatted_datasets": formatted_datasets,
        "topics": topics
    }

# Run production workflow
results = asyncio.run(production_agent_dataset())
print("Production dataset generation complete!")
```

This API provides full programmatic control over agent tool-calling dataset generation, enabling sophisticated workflows and integration with existing ML pipelines.