# Getting Started with Agent Tool-Calling

This tutorial will guide you through creating your first agent tool-calling dataset, from basic setup to generating and validating your results.

## Prerequisites

- DeepFabric installed (`pip install deepfabric`)
- API key for your chosen provider (OpenAI, Anthropic, etc.)
- Basic understanding of JSON/YAML formats

## Step 1: Basic Setup

Let's start with a simple agent tool-calling dataset that demonstrates weather queries.

### Create Configuration File

Create a file called `agent-getting-started.yaml`:

```yaml
# Basic Agent Tool-Calling Configuration
dataset_system_prompt: "You are an AI assistant with access to various tools. Always explain your reasoning when selecting and using tools."

topic_tree:
  topic_prompt: "Daily tasks requiring tool usage: weather checks, simple calculations, web searches"
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
  depth: 2
  degree: 3

data_engine:
  generation_system_prompt: "You excel at reasoning about tool selection and parameter construction. Create realistic scenarios showing step-by-step tool usage."
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.8
  conversation_type: "agent_cot_tools"
  available_tools:
    - "get_weather"
    - "calculator"
    - "search_web"
  max_tools_per_query: 2
  max_retries: 3

dataset:
  creation:
    num_steps: 5  # Start small for testing
    batch_size: 1
    sys_msg: false
  save_as: "my_first_agent_dataset.jsonl"
```

### Set Environment Variables

Set your API key:

```bash
# For OpenAI
export OPENAI_API_KEY="your-api-key-here"

# Or for Anthropic
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Step 2: Generate Your First Dataset

### Using CLI (Recommended)

```bash
deepfabric start agent-getting-started.yaml
```

You should see output like:
```
🚀 Starting generation with agent-getting-started.yaml
📊 Generating topic tree...
✓ Generated 12 topics
🤖 Generating agent tool-calling samples...
✓ Generated 5 agent samples
💾 Dataset saved to my_first_agent_dataset.jsonl
```

### Using Python

Create `generate_agent.py`:

```python
import asyncio
from deepfabric.config import DeepFabricConfig
from deepfabric.tree import Tree
from deepfabric.generator import DataSetGenerator
from deepfabric.dataset import Dataset

async def main():
    # Load configuration
    config = DeepFabricConfig.from_yaml("agent-getting-started.yaml")

    # Generate topics
    tree_params = config.get_topic_tree_params()
    tree = Tree(tree_params)
    topics = await tree.generate()
    print(f"Generated {len(topics)} topics")

    # Generate agent dataset
    engine_params = config.get_engine_params()
    engine_params["topics"] = topics

    generator = DataSetGenerator(engine_params)
    samples = await generator.generate()
    print(f"Generated {len(samples)} agent samples")

    # Save dataset
    dataset = Dataset.from_list(samples)
    dataset.save("my_first_agent_dataset.jsonl")
    print("Dataset saved!")

# Run generation
asyncio.run(main())
```

Then run:
```bash
python generate_agent.py
```

## Step 3: Examine Your Results

### View Sample Structure

Let's look at what was generated:

```bash
# View the first sample
head -n 1 my_first_agent_dataset.jsonl | jq .
```

You should see a structure like:

```json
{
  "question": "What's the weather like in Seattle and what's 15% of the temperature in Fahrenheit?",
  "available_tools": [
    {
      "name": "get_weather",
      "description": "Get current weather conditions for a location",
      "parameters": [...],
      "returns": "Weather data including temperature and conditions",
      "category": "information"
    },
    {
      "name": "calculator",
      "description": "Perform mathematical calculations",
      "parameters": [...],
      "returns": "Calculation result",
      "category": "computation"
    }
  ],
  "initial_analysis": "The user wants two things: current weather information for Seattle, and a calculation of 15% of the temperature value. I'll need to use the weather tool first to get the temperature, then use the calculator to compute 15% of that value.",
  "tool_planning": [
    {
      "step_number": 1,
      "reasoning": "Need to get current weather for Seattle to obtain temperature data",
      "selected_tool": {...},
      "parameter_reasoning": {
        "location": "Seattle is explicitly specified by the user"
      },
      "expected_result": "Current weather conditions including temperature in Fahrenheit"
    },
    {
      "step_number": 2,
      "reasoning": "Need to calculate 15% of the temperature value obtained from weather data",
      "selected_tool": {...},
      "parameter_reasoning": {
        "expression": "Will use the temperature from step 1 multiplied by 0.15"
      },
      "expected_result": "15% of the temperature value"
    }
  ],
  "tool_executions": [
    {
      "function": "get_weather",
      "arguments": {"location": "Seattle"},
      "reasoning": "Getting weather data for Seattle as planned",
      "result": "Seattle: 65°F, partly cloudy, 60% humidity, light wind from northwest"
    },
    {
      "function": "calculator",
      "arguments": {"expression": "65 * 0.15"},
      "reasoning": "Calculating 15% of 65°F temperature",
      "result": "9.75"
    }
  ],
  "result_synthesis": "I obtained the current weather for Seattle (65°F, partly cloudy) and calculated that 15% of the temperature (65°F) equals 9.75. This provides both pieces of information the user requested.",
  "final_answer": "The current weather in Seattle is 65°F and partly cloudy with 60% humidity and light winds from the northwest. 15% of the current temperature (65°F) is 9.75."
}
```

### Key Components to Notice

1. **Complete reasoning trace**: From initial analysis to final answer
2. **Tool planning**: Step-by-step reasoning for tool selection
3. **Parameter reasoning**: Explanation of how arguments were constructed
4. **Tool executions**: Actual function calls with realistic results
5. **Result synthesis**: How tool outputs combine to answer the question

## Step 4: Understanding the Output

### Reasoning Quality Indicators

Good agent samples should have:

✅ **Clear initial analysis** - Understanding of what's needed
✅ **Logical tool planning** - Step-by-step reasoning for tool selection
✅ **Realistic tool usage** - Tools that would actually solve the problem
✅ **Parameter reasoning** - Clear explanation of argument construction
✅ **Result synthesis** - How tool outputs address the original question

### Common Patterns You'll See

**Single Tool Usage:**
```json
{
  "question": "What's the weather in Paris?",
  "tool_planning": [
    {
      "reasoning": "User wants current weather for Paris - need weather tool",
      "selected_tool": "get_weather"
    }
  ]
}
```

**Multi-Tool Workflows:**
```json
{
  "question": "Search for Italian restaurants in NYC and calculate tip for $75",
  "tool_planning": [
    {
      "step_number": 1,
      "reasoning": "Search for Italian restaurants in NYC",
      "selected_tool": "search_web"
    },
    {
      "step_number": 2,
      "reasoning": "Calculate tip amount for $75 bill",
      "selected_tool": "calculator"
    }
  ]
}
```

## Step 5: Expand and Customize

### Add More Tools

Edit your configuration to include more tools:

```yaml
data_engine:
  available_tools:
    - "get_weather"
    - "calculator"
    - "search_web"
    - "get_time"          # Add time queries
    - "get_current_time"  # Add current time
  max_tools_per_query: 3  # Allow more tools per query
```

### Custom Topics

Make topics more specific:

```yaml
topic_tree:
  topic_prompt: "Professional scenarios: business calculations, market research, scheduling coordination, travel planning"
```

### Increase Dataset Size

Generate more samples:

```yaml
dataset:
  creation:
    num_steps: 25    # Generate 25 samples
    batch_size: 5    # Process 5 at a time
```

## Step 6: Quality Validation

### Check Sample Quality

Use this Python script to validate your dataset:

```python
import json

def validate_sample(sample):
    """Check if agent sample has good quality."""
    issues = []

    # Check required fields
    required = ["question", "tool_planning", "tool_executions", "final_answer"]
    for field in required:
        if field not in sample:
            issues.append(f"Missing {field}")

    # Check tool usage
    if len(sample.get("tool_executions", [])) == 0:
        issues.append("No tools were executed")

    # Check reasoning quality
    planning = sample.get("tool_planning", [])
    if len(planning) == 0:
        issues.append("No tool planning found")

    # Check parameter reasoning
    for step in planning:
        if "parameter_reasoning" not in step:
            issues.append("Missing parameter reasoning")

    return issues

# Validate dataset
with open("my_first_agent_dataset.jsonl", "r") as f:
    samples = [json.loads(line) for line in f]

for i, sample in enumerate(samples):
    issues = validate_sample(sample)
    if issues:
        print(f"Sample {i}: {', '.join(issues)}")
    else:
        print(f"Sample {i}: ✓ Good quality")
```

### Sample Statistics

Check your dataset statistics:

```python
import json
from collections import Counter

with open("my_first_agent_dataset.jsonl", "r") as f:
    samples = [json.loads(line) for line in f]

# Tool usage statistics
tool_usage = Counter()
for sample in samples:
    for execution in sample.get("tool_executions", []):
        tool_usage[execution["function"]] += 1

print("Tool usage statistics:")
for tool, count in tool_usage.most_common():
    print(f"  {tool}: {count} times")

# Tools per query
tools_per_query = [len(sample.get("tool_executions", [])) for sample in samples]
avg_tools = sum(tools_per_query) / len(tools_per_query)
print(f"\nAverage tools per query: {avg_tools:.1f}")
print(f"Tool usage range: {min(tools_per_query)}-{max(tools_per_query)}")
```

## Step 7: Next Steps

### Try Multi-Turn Agent CoT

Edit your config to use multi-turn:

```yaml
data_engine:
  conversation_type: "agent_cot_multi_turn"
dataset:
  creation:
    sys_msg: true  # Multi-turn can use system messages
```

### Add Custom Tools

Create `custom_tools.yaml`:

```yaml
tools:
  - name: "book_restaurant"
    description: "Book a restaurant reservation"
    parameters:
      - name: "restaurant"
        type: "str"
        description: "Restaurant name"
        required: true
      - name: "party_size"
        type: "int"
        description: "Number of people"
        required: true
    returns: "Reservation confirmation"
    category: "booking"
```

Then reference it in your config:

```yaml
data_engine:
  tool_registry_path: "custom_tools.yaml"
  available_tools:
    - "get_weather"
    - "book_restaurant"  # Your custom tool
```

### Apply Formatters

Add formatting for different training formats:

```yaml
dataset:
  formatters:
    - name: "tool_calling"
      template: "builtin://tool_calling"
      output: "agent_tool_calling_format.jsonl"
      config:
        system_prompt: "You are a function calling AI model."
        include_tools_in_system: true
```

## Common Issues and Solutions

### Issue: No tool executions generated
**Solution**: Ensure your topic prompts naturally require tool usage:
```yaml
topic_tree:
  topic_prompt: "Tasks requiring weather data, calculations, or web searches"
```

### Issue: Poor reasoning quality
**Solution**: Improve system prompts:
```yaml
data_engine:
  generation_system_prompt: "Focus on WHY tools are selected and HOW parameters are determined. Provide detailed reasoning at each step."
```

### Issue: Unrealistic tool results
**Solution**: This is expected - the model generates realistic-looking results for training purposes.

## Congratulations!

You've successfully created your first agent tool-calling dataset! This dataset teaches models to:

- **Reason systematically** about tool selection
- **Construct parameters** based on user input
- **Explain their thinking** at each step
- **Synthesize results** into helpful responses

Next, try [multi-turn conversations](../multi-turn.md) for more advanced scenarios.