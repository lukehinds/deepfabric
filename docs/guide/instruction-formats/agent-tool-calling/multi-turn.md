# Multi-Turn Agent CoT (`agent_cot_multi_turn`)

Multi-turn agent Chain of Thought datasets train models to maintain context and reasoning across multiple conversation exchanges while using tools progressively. This format captures how intelligent agents build upon previous interactions and adapt their tool usage based on evolving user needs.

## When to Use Multi-Turn Agent CoT

Multi-turn agent CoT is ideal for:
- **Conversational tool usage** where context from previous turns influences current decisions
- **Progressive information gathering** where each user response adds constraints or details
- **Complex workflows** that naturally span multiple interactions
- **Context-aware assistance** where the agent remembers previous tool results

## Key Differences from Single-Turn

| Aspect | Single-Turn | Multi-Turn |
|--------|-------------|------------|
| **Scope** | Complete task in one response | Evolving task across conversation |
| **Context** | Self-contained | Builds on previous exchanges |
| **Tool Usage** | Planned upfront | Adapts based on user feedback |
| **Complexity** | Linear reasoning | Non-linear, responsive reasoning |

## Format Structure

The `AgentCoTMultiTurn` schema captures the complete conversational flow:

```json
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "tool_planning_trace": [
    {reasoning steps across all turns}
  ],
  "tool_execution_trace": [
    {all tool executions throughout conversation}
  ],
  "reasoning_summary": "Overall strategy across the conversation"
}
```

## Detailed Schema Breakdown

### `messages` (list)
Complete conversation history showing natural flow between user and assistant.

**Example:**
```json
"messages": [
  {"role": "user", "content": "What's the weather like today?"},
  {"role": "assistant", "content": "I'd be happy to check the weather for you! However, I need to know your location first. Which city would you like me to check?"},
  {"role": "user", "content": "I'm in San Francisco"},
  {"role": "assistant", "content": "Let me get the current weather for San Francisco..."},
  {"role": "user", "content": "Is it good for a picnic?"},
  {"role": "assistant", "content": "Based on the weather I just retrieved, let me give you a recommendation for picnic conditions..."}
]
```

### `tool_planning_trace` (list)
Reasoning steps that span across conversation turns, showing how planning evolves:

```json
"tool_planning_trace": [
  {
    "step_number": 1,
    "reasoning": "User wants weather but didn't specify location - need to ask for clarification",
    "selected_tool": null,
    "parameter_reasoning": {},
    "expected_result": "User will provide location information"
  },
  {
    "step_number": 2,
    "reasoning": "Now have location (San Francisco) - can get weather data",
    "selected_tool": {weather_tool_definition},
    "parameter_reasoning": {"location": "San Francisco provided by user in second message"},
    "expected_result": "Current weather conditions for San Francisco"
  },
  {
    "step_number": 3,
    "reasoning": "User asking about picnic suitability - need to interpret weather data for outdoor activity",
    "selected_tool": null,
    "parameter_reasoning": {},
    "expected_result": "Analysis of weather suitability for picnic activities"
  }
]
```

### `tool_execution_trace` (list)
All tool calls made throughout the conversation with their context:

```json
"tool_execution_trace": [
  {
    "function": "get_weather",
    "arguments": {"location": "San Francisco"},
    "reasoning": "Getting current weather for San Francisco as requested",
    "result": "San Francisco: 22°C, sunny, light breeze from west, 0% chance of rain",
    "turn_number": 2
  }
]
```

### `reasoning_summary` (string)
High-level description of the reasoning strategy across the entire conversation:

**Example:**
- `"Progressive information gathering: started with incomplete query, gathered location through conversation, retrieved weather data, then interpreted results for specific use case (picnic suitability). Demonstrates context maintenance and adaptive tool usage."`

## Multi-Turn Patterns

### Progressive Information Gathering
```json
{
  "messages": [
    {"role": "user", "content": "Book me a restaurant reservation"},
    {"role": "assistant", "content": "I'd be happy to help with that! I'll need a few details: What type of cuisine do you prefer, how many people, what date and time, and which city?"},
    {"role": "user", "content": "Italian food for 2 people tomorrow at 7 PM in Boston"},
    {"role": "assistant", "content": "Perfect! Let me search for available Italian restaurants in Boston for tomorrow evening..."},
    {"role": "user", "content": "Something upscale but not too expensive"},
    {"role": "assistant", "content": "I'll refine the search to focus on mid-to-upscale Italian restaurants..."}
  ],
  "tool_planning_trace": [
    {
      "step_number": 1,
      "reasoning": "User wants restaurant reservation but provided incomplete information - need cuisine, party size, date/time, location",
      "selected_tool": null
    },
    {
      "step_number": 2,
      "reasoning": "Now have all basic info - can search for Italian restaurants in Boston",
      "selected_tool": {search_tool},
      "parameter_reasoning": {"query": "Italian restaurants Boston reservations tomorrow 7 PM"}
    },
    {
      "step_number": 3,
      "reasoning": "User added preference for upscale but not expensive - need to filter results",
      "selected_tool": {search_tool},
      "parameter_reasoning": {"query": "mid-range upscale Italian restaurants Boston", "price_range": "$$-$$$"}
    }
  ]
}
```

### Context-Dependent Tool Usage
```json
{
  "messages": [
    {"role": "user", "content": "What's 15% of 247?"},
    {"role": "assistant", "content": "Let me calculate that for you..."},
    {"role": "user", "content": "And what about 20%?"},
    {"role": "assistant", "content": "I'll calculate 20% of 247 as well..."},
    {"role": "user", "content": "Which one should I use for a tip?"}
  ],
  "tool_planning_trace": [
    {
      "step_number": 1,
      "reasoning": "Direct calculation request for 15% of 247",
      "selected_tool": {calculator_tool}
    },
    {
      "step_number": 2,
      "reasoning": "Follow-up calculation for 20% of same number (247) - context maintained",
      "selected_tool": {calculator_tool}
    },
    {
      "step_number": 3,
      "reasoning": "User asking for tip advice - need to provide guidance based on previous calculations",
      "selected_tool": null,
      "expected_result": "Tip recommendation based on context and social norms"
    }
  ]
}
```

### Iterative Refinement
```json
{
  "messages": [
    {"role": "user", "content": "Find flights to Europe"},
    {"role": "assistant", "content": "I'd be happy to help find flights to Europe! Which specific city or country are you interested in, and what are your departure details?"},
    {"role": "user", "content": "Paris, leaving from New York next month"},
    {"role": "assistant", "content": "Let me search for flights from New York to Paris for next month..."},
    {"role": "user", "content": "Actually, I prefer direct flights under $800"},
    {"role": "assistant", "content": "I'll refine the search to show only direct flights under $800..."}
  ]
}
```

## Configuration Example

### YAML Configuration
```yaml
dataset_system_prompt: "You are a conversational AI assistant that maintains context and uses tools progressively across multiple turns."

data_engine:
  generation_system_prompt: "Generate realistic multi-turn conversations showing progressive tool usage and context maintenance."
  provider: "openai"
  model: "gpt-4o-mini"
  conversation_type: "agent_cot_multi_turn"
  available_tools:
    - "get_weather"
    - "search_web"
    - "book_restaurant"
    - "calculator"
  max_tools_per_query: 2

dataset:
  creation:
    num_steps: 15
    batch_size: 3
    sys_msg: true  # Multi-turn supports system messages
  save_as: "multi_turn_agent.jsonl"
```

### Python API
```python
from deepfabric import DataSetGenerator

engine = DataSetGenerator(
    generation_system_prompt="Create multi-turn conversations with progressive tool usage.",
    provider="openai",
    model_name="gpt-4o-mini",
    conversation_type="agent_cot_multi_turn",
    available_tools=["get_weather", "search_web", "book_restaurant"],
    max_tools_per_query=2,
)

dataset = engine.create_data(
    num_steps=10,
    batch_size=2,
    sys_msg=True
)
```

## Best Practices

### Context Maintenance
- **Reference previous exchanges** in reasoning steps
- **Build upon prior information** rather than starting fresh
- **Acknowledge user clarifications** explicitly

### Progressive Tool Usage
- **Start with available information** and identify gaps
- **Use follow-up questions** to gather missing details
- **Adapt tool selection** based on new user input

### Natural Conversation Flow
- **Maintain conversational tone** throughout exchanges
- **Ask clarifying questions** when information is incomplete
- **Provide interim updates** during multi-step processes

## Training Benefits

Models trained on multi-turn agent CoT data demonstrate:
- **Better context retention** across conversation turns
- **More natural information gathering** through clarifying questions
- **Adaptive tool usage** based on evolving user needs
- **Improved conversation management** for complex workflows
- **Enhanced user experience** through progressive assistance

## Common Conversation Patterns

### Information Gathering → Tool Usage → Follow-up
1. User makes vague request
2. Assistant asks for specifics
3. User provides details
4. Assistant uses tools with complete information
5. User asks follow-up questions
6. Assistant adapts based on tool results and context

### Iterative Refinement
1. User makes broad request
2. Assistant provides initial results using tools
3. User refines requirements
4. Assistant adjusts tool usage and filters results
5. Process continues until user is satisfied

### Context-Building Workflows
1. Each turn adds information to the context
2. Tool usage becomes more targeted with each exchange
3. Final responses synthesize all previous interactions
4. Complex workflows completed through natural conversation

This format creates training data for agents that can engage in sophisticated, context-aware conversations while using tools intelligently and progressively.