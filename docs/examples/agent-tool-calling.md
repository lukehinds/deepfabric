# Agent Tool-Calling Examples

This section provides complete, runnable examples for generating agent tool-calling datasets. These examples demonstrate both programmatic and configuration-driven approaches to creating training data that teaches models systematic tool usage and reasoning.

## Overview

Agent tool-calling datasets train models to:
- **Reason systematically** about tool selection
- **Construct parameters** accurately from user input
- **Explain their thinking** at each step
- **Synthesize results** into helpful responses
- **Maintain context** across conversation turns

## Quick Start Example

### Basic Agent Tool-Calling

The simplest way to get started with agent tool-calling datasets:

**Configuration File** (`agent-quickstart.yaml`):
```yaml
dataset_system_prompt: "You are an AI assistant that explains your reasoning when using tools."

topic_tree:
  topic_prompt: "Daily tasks requiring tool usage: weather, calculations, searches"
  provider: "openai"
  model: "gpt-4o-mini"
  depth: 2
  degree: 3

data_engine:
  generation_system_prompt: "Focus on WHY tools are selected and HOW parameters are constructed."
  provider: "openai"
  model: "gpt-4o-mini"
  conversation_type: "agent_cot_tools"
  available_tools: ["get_weather", "calculator", "search_web"]
  max_tools_per_query: 2

dataset:
  creation:
    num_steps: 10
    batch_size: 2
  save_as: "agent_quickstart.jsonl"
```

**Generate the dataset:**
```bash
deepfabric start agent-quickstart.yaml
```

**Expected output structure:**
```json
{
  "question": "What's the weather in Tokyo and what's 20% of the temperature?",
  "available_tools": [...],
  "initial_analysis": "User wants weather data for Tokyo and a percentage calculation.",
  "tool_planning": [
    {
      "step_number": 1,
      "reasoning": "Need current weather for Tokyo to get temperature",
      "selected_tool": {...},
      "parameter_reasoning": {"location": "Tokyo specified by user"},
      "expected_result": "Current weather conditions including temperature"
    }
  ],
  "tool_executions": [...],
  "result_synthesis": "Combined weather data with percentage calculation",
  "final_answer": "Tokyo is currently 25°C with sunny skies. 20% of 25°C is 5°C."
}
```

## Complete Examples

### 1. Professional Scenarios with Custom Tools

**Use case**: Business productivity scenarios with booking and analysis tools.

**Custom tools definition** (`business_tools.yaml`):
```yaml
tools:
  - name: "book_meeting_room"
    description: "Reserve a meeting room"
    parameters:
      - name: "room"
        type: "str"
        description: "Room name or number"
        required: true
      - name: "duration"
        type: "int"
        description: "Duration in minutes"
        required: true
      - name: "date"
        type: "str"
        description: "Date in YYYY-MM-DD format"
        required: true
    returns: "Meeting room reservation confirmation"
    category: "productivity"

  - name: "analyze_sales_data"
    description: "Analyze sales performance metrics"
    parameters:
      - name: "period"
        type: "str"
        description: "Time period (week, month, quarter)"
        required: true
      - name: "region"
        type: "str"
        description: "Geographic region"
        required: false
        default: "all"
    returns: "Sales analysis report with key metrics"
    category: "analytics"
```

**Configuration** (`business_agent.yaml`):
```yaml
dataset_system_prompt: |
  You are a business productivity AI assistant with access to meeting, scheduling,
  and analytics tools. Always explain your reasoning when selecting tools.

topic_tree:
  topic_prompt: "Business productivity scenarios: meeting coordination, sales analysis, resource planning"
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
  depth: 3
  degree: 3

data_engine:
  generation_system_prompt: |
    Create realistic business scenarios requiring systematic tool usage.
    Focus on professional workflows and decision-making processes.

  provider: "openai"
  model: "gpt-4o"
  temperature: 0.8
  conversation_type: "agent_cot_tools"

  # Mix default and custom tools
  available_tools:
    - "get_weather"
    - "calculator"
    - "book_meeting_room"
    - "analyze_sales_data"

  tool_registry_path: "business_tools.yaml"
  max_tools_per_query: 3

dataset:
  creation:
    num_steps: 25
    batch_size: 5
    sys_msg: false
  save_as: "business_agent_dataset.jsonl"

  # Apply formatters for different training formats
  formatters:
    - name: "tool_calling"
      template: "builtin://tool_calling"
      output: "business_agent_formatted.jsonl"
      config:
        system_prompt: "You are a business productivity function calling AI."
        include_tools_in_system: true
```

**Generate:**
```bash
deepfabric start business_agent.yaml
```

### 2. Multi-Turn Conversational Agent

**Use case**: Progressive information gathering and context-aware tool usage.

**Configuration** (`conversational_agent.yaml`):
```yaml
dataset_system_prompt: "You are a conversational AI that maintains context and uses tools progressively."

topic_tree:
  topic_prompt: "Multi-step assistance scenarios: travel planning, event coordination, research tasks"
  provider: "openai"
  model: "gpt-4o-mini"
  depth: 3
  degree: 4

data_engine:
  generation_system_prompt: |
    Create realistic multi-turn conversations where the agent gathers information
    progressively and adapts tool usage based on user responses.

  provider: "openai"
  model: "gpt-4o"
  conversation_type: "agent_cot_multi_turn"

  available_tools:
    - "get_weather"
    - "search_web"
    - "book_restaurant"
    - "calculator"
    - "get_directions"

  max_tools_per_query: 2
  temperature: 0.8

dataset:
  creation:
    num_steps: 15
    batch_size: 3
    sys_msg: true  # Multi-turn supports system messages
  save_as: "conversational_agent.jsonl"
```

**Expected multi-turn output:**
```json
{
  "messages": [
    {"role": "user", "content": "I need help planning a dinner"},
    {"role": "assistant", "content": "I'd be happy to help! What type of cuisine are you interested in, and how many people?"},
    {"role": "user", "content": "Italian food for 4 people tomorrow in Boston"},
    {"role": "assistant", "content": "Great! Let me search for Italian restaurants in Boston..."},
    {"role": "user", "content": "What's the weather like? Should we consider outdoor seating?"}
  ],
  "tool_planning_trace": [
    {
      "step_number": 1,
      "reasoning": "User wants restaurant help but gave incomplete info - need more details",
      "selected_tool": null
    },
    {
      "step_number": 2,
      "reasoning": "Now have location, cuisine, party size - can search for restaurants",
      "selected_tool": {...}
    },
    {
      "step_number": 3,
      "reasoning": "User asking about weather for outdoor seating decision",
      "selected_tool": {...}
    }
  ],
  "tool_execution_trace": [...],
  "reasoning_summary": "Progressive information gathering leading to restaurant recommendation with weather consideration"
}
```

### 3. Programmatic Generation with Custom Logic

**Use case**: Full programmatic control with custom validation and filtering.

**Python script** (`advanced_agent_generation.py`):
```python
import asyncio
from deepfabric import DataSetGenerator
from deepfabric.dataset import Dataset
from deepfabric.tree import Tree
from deepfabric.schemas import ToolDefinition, ToolParameter

async def generate_advanced_agent_dataset():
    """Generate agent dataset with custom tools and validation."""

    # Define domain-specific custom tools
    fitness_tool = ToolDefinition(
        name="create_workout_plan",
        description="Generate personalized workout plan",
        parameters=[
            ToolParameter(
                name="fitness_level",
                type="str",
                description="beginner/intermediate/advanced",
                required=True
            ),
            ToolParameter(
                name="goals",
                type="list",
                description="List of fitness goals",
                required=True
            ),
            ToolParameter(
                name="duration",
                type="int",
                description="Workout duration in minutes",
                required=False,
                default=30
            )
        ],
        returns="Personalized workout plan with exercises",
        category="fitness"
    )

    nutrition_tool = ToolDefinition(
        name="analyze_nutrition",
        description="Analyze nutritional content of foods",
        parameters=[
            ToolParameter(
                name="foods",
                type="list",
                description="List of foods to analyze",
                required=True
            ),
            ToolParameter(
                name="portion_sizes",
                type="list",
                description="Portion sizes for each food",
                required=False,
                default=[]
            )
        ],
        returns="Nutritional analysis with calories and macros",
        category="health"
    )

    # Create topic tree for fitness domain
    tree = Tree(
        topic_prompt="Fitness and wellness scenarios: workout planning, nutrition analysis, health tracking",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=4,
        depth=3,
        temperature=0.7
    )

    topics = await tree.generate()
    print(f"Generated {len(topics)} fitness topics")

    # Create generator with custom tools
    generator = DataSetGenerator(
        generation_system_prompt="You are a fitness and wellness AI with specialized tools for workout and nutrition planning.",
        provider="openai",
        model_name="gpt-4o",
        conversation_type="agent_cot_tools",
        available_tools=[
            "calculator",
            "search_web",
            "create_workout_plan",
            "analyze_nutrition"
        ],
        custom_tools=[
            fitness_tool.model_dump(),
            nutrition_tool.model_dump()
        ],
        max_tools_per_query=3,
        temperature=0.8,
        topics=topics
    )

    # Generate samples with validation
    valid_samples = []
    total_attempts = 0
    max_attempts = 50

    while len(valid_samples) < 20 and total_attempts < max_attempts:
        batch_samples = await generator.generate()

        for sample in batch_samples:
            if validate_fitness_sample(sample):
                valid_samples.append(sample)
                print(f"Valid sample {len(valid_samples)}: {sample['question'][:50]}...")

        total_attempts += 1

    # Create final dataset
    dataset = Dataset.from_list(valid_samples)
    dataset.save("fitness_agent_dataset.jsonl")

    # Apply fitness-specific formatting
    formatter_config = {
        "name": "fitness_tool_calling",
        "template": "builtin://tool_calling",
        "output": "fitness_agent_formatted.jsonl",
        "config": {
            "system_prompt": "You are a fitness AI with workout and nutrition tools.",
            "include_tools_in_system": True
        }
    }

    formatted = dataset.apply_formatters([formatter_config])

    print(f"Generated {len(dataset)} validated fitness agent samples")
    return dataset

def validate_fitness_sample(sample):
    """Validate fitness-specific sample quality."""
    # Check tool usage is fitness-related
    tool_names = [exec["function"] for exec in sample.get("tool_executions", [])]
    fitness_tools = ["create_workout_plan", "analyze_nutrition", "calculator"]

    has_fitness_tool = any(tool in fitness_tools for tool in tool_names)
    has_reasoning = len(sample.get("tool_planning", [])) > 0
    has_question = "fitness" in sample.get("question", "").lower() or "workout" in sample.get("question", "").lower()

    return has_fitness_tool and has_reasoning and has_question

# Run the generation
if __name__ == "__main__":
    dataset = asyncio.run(generate_advanced_agent_dataset())
```

## Output Format Examples

### Tool-Calling Format Output

When using the `builtin://tool_calling` formatter, samples are converted to function calling format:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a function calling AI model. You have access to the following functions:\n\n<tools>\nget_weather(location: str) → Weather data\ncalculator(expression: str) → Calculation result\n</tools>"
    },
    {
      "role": "user",
      "content": "What's the weather in Paris and what's 15% of the temperature?"
    },
    {
      "role": "assistant",
      "content": "<think>User wants weather for Paris and a percentage calculation. I'll get the weather first, then calculate 15% of the temperature.</think>\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "<tool_response>\nParis: 18°C, partly cloudy, 65% humidity\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "<tool_call>\n{\"name\": \"calculator\", \"arguments\": {\"expression\": \"18 * 0.15\"}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "<tool_response>\n2.7\n</tool_response>"
    },
    {
      "role": "assistant",
      "content": "The current weather in Paris is 18°C with partly cloudy conditions and 65% humidity. 15% of the current temperature (18°C) equals 2.7°C."
    }
  ]
}
```

## Best Practices

### Topic Selection
- **Be specific** about tool usage requirements in topic prompts
- **Include domain context** that naturally requires tools
- **Mix complexity levels** for varied training scenarios

### Tool Configuration
- **Start with defaults** then add domain-specific custom tools
- **Limit tools per query** to maintain focus (2-4 tools maximum)
- **Group related tools** by category for better organization

### Quality Control
- **Validate tool usage** - ensure samples actually use tools meaningfully
- **Check reasoning quality** - tool planning should be logical and detailed
- **Review parameter construction** - arguments should be based on user input

### Production Considerations
- **Use cost-effective models** (gpt-4o-mini) for large datasets
- **Batch processing** for efficiency
- **Incremental validation** to catch issues early
- **Multiple formatters** for different training needs

These examples provide complete, production-ready templates for generating high-quality agent tool-calling datasets that effectively train models in systematic tool usage and reasoning.