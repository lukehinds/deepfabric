# Single-Turn Agent CoT (`agent_cot_tools`)

Single-turn agent Chain of Thought datasets focus on training models to handle complete tasks requiring tool usage in a single interaction. This format captures the full reasoning process from initial analysis through tool execution to final synthesis.

## When to Use Single-Turn Agent CoT

Single-turn agent CoT is ideal for:
- **Complete task resolution** where the user's request can be fully addressed in one response
- **Tool reasoning training** where you want models to learn systematic tool selection
- **Parameter construction** training for building accurate function calls
- **Result synthesis** where multiple tool outputs need to be combined into coherent answers

## Format Structure

The `AgentCoTWithTools` schema captures six key components:

```json
{
  "question": "User's question or request",
  "available_tools": [list of tool definitions],
  "initial_analysis": "Understanding of what's needed",
  "tool_planning": [reasoning steps for tool selection],
  "tool_executions": [actual tool calls and results],
  "result_synthesis": "How results combine to answer question",
  "final_answer": "Complete response to user"
}
```

## Detailed Schema Breakdown

### `question` (string)
The user's original question or request that requires tool usage.

**Examples:**
- `"What's the weather in London and Paris?"`
- `"Calculate the tip for a $45.67 dinner and find a good restaurant nearby"`
- `"Book a table for 4 at an Italian restaurant tomorrow at 7 PM"`

### `available_tools` (list)
Complete tool definitions using Pydantic schemas. These are the tools the agent can choose from.

```json
"available_tools": [
  {
    "name": "get_weather",
    "description": "Get current weather conditions for a location",
    "parameters": [
      {
        "name": "location",
        "type": "str",
        "description": "City name or location",
        "required": true
      }
    ],
    "returns": "Weather data including temperature and conditions",
    "category": "information"
  }
]
```

### `initial_analysis` (string)
The agent's understanding of what the user needs and what information or actions are required.

**Examples:**
- `"User wants weather information for two cities. I need to retrieve current weather data for both London and Paris and present a comparison."`
- `"This requires two tasks: calculating an 18-20% tip on $45.67, and finding restaurant recommendations near the user's location."`

### `tool_planning` (list)
Step-by-step reasoning for tool selection and parameter construction. Each step includes:

```json
{
  "step_number": 1,
  "reasoning": "Why this tool is needed at this point",
  "selected_tool": {tool definition},
  "parameter_reasoning": {
    "param_name": "explanation of how this parameter value was determined"
  },
  "expected_result": "What the tool should return and how it helps"
}
```

**Example:**
```json
{
  "step_number": 1,
  "reasoning": "Need current weather data for London as specified in the user's request",
  "selected_tool": {
    "name": "get_weather",
    "description": "Get current weather conditions for a location",
    // ... full tool definition
  },
  "parameter_reasoning": {
    "location": "User explicitly mentioned London as first city"
  },
  "expected_result": "Current weather conditions for London including temperature, conditions, and other relevant data"
}
```

### `tool_executions` (list)
Actual function calls made with their results. Each execution includes:

```json
{
  "function": "tool_name",
  "arguments": {parameter dictionary},
  "reasoning": "Brief explanation of why executing now",
  "result": "The actual result returned from the tool"
}
```

**Example:**
```json
{
  "function": "get_weather",
  "arguments": {"location": "London"},
  "reasoning": "Executing weather lookup for London as planned",
  "result": "London: 15°C, overcast, 80% humidity, light rain expected"
}
```

### `result_synthesis` (string)
Explanation of how the tool results are combined and interpreted to address the original question.

**Example:**
- `"Both weather reports show cooler temperatures with London having rain while Paris is clearer. Combined this data to provide a comprehensive comparison for the user's planning purposes."`

### `final_answer` (string)
The complete, natural response to the user incorporating all tool results.

**Example:**
- `"Here's the current weather for both cities: London is 15°C with overcast skies and light rain expected (80% humidity), while Paris is warmer at 18°C with partly cloudy conditions (65% humidity). Paris has better weather today if you're planning outdoor activities."`

## Configuration Example

### YAML Configuration
```yaml
dataset_system_prompt: "You are an AI assistant that excels at systematic tool usage and reasoning."

data_engine:
  generation_system_prompt: "Generate realistic scenarios requiring tool usage with detailed reasoning traces."
  provider: "openai"
  model: "gpt-4o-mini"
  conversation_type: "agent_cot_tools"
  available_tools:
    - "get_weather"
    - "search_web"
    - "calculator"
    - "book_restaurant"
  max_tools_per_query: 3

dataset:
  creation:
    num_steps: 20
    batch_size: 5
    sys_msg: false  # Agent format doesn't use system messages
  save_as: "single_turn_agent.jsonl"
```

### Python API
```python
from deepfabric import DataSetGenerator
from deepfabric.tools.defaults import get_default_tools

engine = DataSetGenerator(
    generation_system_prompt="You excel at tool reasoning and selection.",
    provider="openai",
    model_name="gpt-4o-mini",
    conversation_type="agent_cot_tools",
    available_tools=["get_weather", "search_web", "calculator"],
    max_tools_per_query=2,
)

dataset = engine.create_data(
    num_steps=10,
    batch_size=2,
    sys_msg=False
)
```

## Best Practices

### Tool Planning Quality
- **Be specific** about why each tool is selected
- **Explain parameter construction** based on user input
- **Set clear expectations** for what each tool should return

### Tool Execution Realism
- **Match planning to execution** - tools used should align with the plan
- **Provide realistic results** that tools would actually return
- **Include error cases** where tools might not have complete information

### Result Synthesis
- **Address the original question** directly
- **Integrate all tool results** meaningfully
- **Provide actionable information** when possible

## Common Use Cases

### Weather and Location Services
```json
{
  "question": "What's the weather like in Tokyo and should I bring an umbrella?",
  "tool_planning": [
    {
      "reasoning": "Need current weather conditions for Tokyo to assess umbrella necessity",
      "selected_tool": {...weather_tool...},
      "parameter_reasoning": {"location": "Tokyo specified by user"},
      "expected_result": "Current weather including precipitation probability"
    }
  ]
}
```

### Calculation and Information Lookup
```json
{
  "question": "If I invest $5000 at 7% annual interest for 3 years, how much will I have?",
  "tool_planning": [
    {
      "reasoning": "Need to calculate compound interest with given parameters",
      "selected_tool": {...calculator_tool...},
      "parameter_reasoning": {
        "expression": "Compound interest formula: P(1+r)^t where P=5000, r=0.07, t=3"
      },
      "expected_result": "Final investment value after 3 years"
    }
  ]
}
```

### Multi-Tool Workflows
```json
{
  "question": "Find a good Italian restaurant in NYC and calculate the tip for a $85 dinner",
  "tool_planning": [
    {
      "step_number": 1,
      "reasoning": "Need to search for Italian restaurants in New York City",
      "selected_tool": {...search_tool...}
    },
    {
      "step_number": 2,
      "reasoning": "Need to calculate appropriate tip amount for $85 bill",
      "selected_tool": {...calculator_tool...}
    }
  ]
}
```

## Training Benefits

Models trained on single-turn agent CoT data demonstrate:
- **Improved tool selection** accuracy across different domains
- **Better parameter construction** with fewer errors
- **Enhanced reasoning transparency** that users can follow
- **More efficient tool usage** with reduced unnecessary calls
- **Better error handling** when tools return unexpected results

This format creates training data that teaches models not just to use tools, but to **think systematically** about tool usage in a way that's both effective and explainable.