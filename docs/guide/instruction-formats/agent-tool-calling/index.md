# Agent Tool-Calling Instruction Formats

Agent tool-calling datasets are designed for training language models to effectively reason about and use external tools. Unlike traditional conversation datasets, agent tool-calling formats capture the complete reasoning process: **why** tools are selected, **how** parameters are constructed, and **what** results mean in context.

## Why Agent Tool-Calling Datasets?

### Advanced Reasoning with Tools
Traditional tool-calling datasets often show final function calls without the reasoning process. Agent CoT datasets teach models to:
- **Analyze user intent** and identify what information or actions are needed
- **Select appropriate tools** from available options with clear reasoning
- **Construct parameters** systematically based on user input and context
- **Interpret tool results** and synthesize them into helpful responses
- **Chain multiple tools** together for complex multi-step tasks

### Enhanced Model Training for MCP Integration
Models trained on agent tool-calling data perform significantly better with:
- **Model Context Protocol (MCP)** server integration
- **Function calling APIs** across different providers
- **Multi-step workflows** requiring tool chaining
- **Parameter validation** and error handling
- **Tool selection** from large tool catalogs

### Structured vs Natural Language Approaches
Agent tool-calling datasets leverage **structured generation** with Pydantic schemas rather than relying solely on natural language prompts. This provides:
- **Type-safe tool definitions** with automatic validation
- **Consistent parameter formats** across different tools
- **Reusable tool libraries** that can be shared across projects
- **Better training efficiency** through structured reasoning traces

## Agent CoT Format Types

DeepFabric supports two specialized agent tool-calling formats:

### 1. Single-Turn Agent CoT (`agent_cot_tools`)
**Best for**: Training models to handle complete tasks with tool usage in a single interaction.

```json
{
  "question": "What's the weather like in Paris and how does it compare to New York?",
  "available_tools": [...],
  "initial_analysis": "I need to get weather data for two cities and compare them.",
  "tool_planning": [
    {
      "step_number": 1,
      "reasoning": "Need current weather for Paris",
      "selected_tool": {...},
      "parameter_reasoning": {"location": "Paris specified by user"},
      "expected_result": "Current weather conditions in Paris"
    },
    {
      "step_number": 2,
      "reasoning": "Need current weather for New York to compare",
      "selected_tool": {...},
      "parameter_reasoning": {"location": "New York for comparison"},
      "expected_result": "Current weather conditions in New York"
    }
  ],
  "tool_executions": [
    {
      "function_name": "get_weather",
      "arguments": {"location": "Paris"},
      "reasoning": "Getting Paris weather as requested",
      "result": "Paris: 18°C, partly cloudy, 60% humidity"
    },
    {
      "function_name": "get_weather",
      "arguments": {"location": "New York"},
      "reasoning": "Getting NYC weather for comparison",
      "result": "New York: 22°C, sunny, 45% humidity"
    }
  ],
  "result_synthesis": "Compared both cities' weather data to provide comprehensive answer",
  "final_answer": "Paris is currently 18°C and partly cloudy with 60% humidity, while New York is warmer at 22°C with sunny skies and lower humidity at 45%. New York has better weather today."
}
```

### 2. Multi-Turn Agent CoT (`agent_cot_multi_turn`)
**Best for**: Training conversational agents that maintain context and reasoning across multiple exchanges.

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "content": "Let me check the current weather in Paris for you..."},
    {"role": "user", "content": "How about New York?"},
    {"role": "assistant", "content": "I'll get the New York weather to compare with Paris..."},
    {"role": "user", "content": "Which city is better for outdoor activities today?"}
  ],
  "tool_planning_trace": [...],
  "tool_execution_trace": [...],
  "reasoning_summary": "Progressive weather comparison leading to activity recommendation"
}
```

## Key Differences from Traditional CoT

| Aspect | Traditional CoT | Agent Tool-Calling CoT |
|--------|----------------|----------------------|
| **Focus** | Problem reasoning | Tool reasoning + problem solving |
| **Structure** | Natural language steps | Structured tool planning + execution |
| **Tools** | Hardcoded in prompts | Dynamic, user-definable schemas |
| **Validation** | Text-based | Type-safe with Pydantic |
| **Reusability** | Low (prompt-specific) | High (schema-based) |
| **Training Efficiency** | Moderate | Higher (structured traces) |

## Quick Start Guide

### 1. Choose Your Agent Format

| Format | Use Case | Best For |
|--------|----------|----------|
| **Single-Turn** | Complete task resolution | Tool reasoning, parameter construction |
| **Multi-Turn** | Conversational tool usage | Context maintenance, progressive reasoning |

### 2. Basic Configuration

```yaml
# agent-tool-calling.yaml
dataset_system_prompt: "You are an AI assistant with access to various tools. Always explain your reasoning when selecting and using tools."

topic_tree:
  topic_prompt: "Real-world scenarios requiring tool usage"
  provider: "openai"
  model: "gpt-4o-mini"
  degree: 3
  depth: 2

data_engine:
  generation_system_prompt: "You excel at reasoning about tool selection and parameter construction."
  provider: "openai"
  model: "gpt-4o-mini"
  conversation_type: "agent_cot_tools"  # or agent_cot_multi_turn
  available_tools: ["get_weather", "search_web", "calculator"]
  max_tools_per_query: 2

dataset:
  creation:
    num_steps: 10
    batch_size: 2
  save_as: "agent_tool_dataset.jsonl"
```

### 3. Generate Your Dataset

```bash
# Using CLI
deepfabric start agent-tool-calling.yaml

# Using Python
python examples/agent_tool_calling.py
```

## What's Next?

### **Format Deep Dives**
- [Single-Turn Agent CoT](single-turn.md) - Complete task resolution with tools
- [Multi-Turn Agent CoT](multi-turn.md) - Conversational tool usage

### **Tutorials**
- [Getting Started](tutorials/getting-started.md) - First agent tool-calling dataset

### **Configuration**
- [YAML Configuration](configuration/yaml-config.md) - Agent-specific parameters
- [Python API](configuration/python-api.md) - Programmatic usage

---

*Agent tool-calling datasets bridge the gap between reasoning and action, creating training data that teaches models not just **what** to do, but **how to think** about doing it.*