# Single Tool Call Formatter Guide

The Single Tool Call formatter transforms agent reasoning datasets into a conversational format where each tool call is handled in its own message exchange. This format is ideal for training models to make single, focused tool calls with clear reasoning.

## Overview

Unlike the standard tool calling formatter that embeds multiple tool calls within thinking traces, the Single Tool Call formatter creates separate message exchanges for each tool call. This provides cleaner boundaries between tool invocations and makes it easier to train models on sequential tool usage patterns.

## Key Features

- **Individual Tool Calls**: Each tool call gets its own assistant message
- **Reasoning Prefixes**: Optional natural language explanations before tool calls
- **JSON Tool Responses**: Consistent JSON formatting for tool outputs
- **System Tool Integration**: Automatic inclusion of tool definitions in system messages
- **Action Generation**: Smart action descriptions based on tool type and parameters

## Quick Start

### 1. Basic Configuration

```yaml
formatters:
  - name: "single_tool_format"
    template: "builtin://single_tool_call"
    output: "single_tool_dataset.jsonl"
    config:
      system_prompt: "You are a helpful assistant with access to functions."
      include_tools_in_system: true
      include_reasoning_prefix: true
```

### 2. Generate Dataset

```bash
deepfabric generate config.yaml
```

### 3. Example Output

The formatter transforms this input:
```json
{
  "question": "What's the weather in Paris and time in Tokyo?",
  "tool_used": "get_weather",
  "tool_input": "{\"location\": \"Paris\"}",
  "tool_output": "15°C, partly cloudy",
  "answer": "Weather and time information provided."
}
```

Into this output:
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to the following functions. Use them if required:\n\n{\"functions\": [{\"name\": \"get_weather\", \"description\": \"Get weather information\", \"parameters\": {...}}]}"
    },
    {
      "role": "user",
      "content": "What's the weather in Paris and time in Tokyo?"
    },
    {
      "role": "assistant",
      "content": "I'll check the weather in Paris for you.\n\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Paris\"}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"temperature\": \"15°C\", \"conditions\": \"Partly cloudy\"}"
    },
    {
      "role": "assistant",
      "content": "Weather and time information provided."
    }
  ]
}
```

## Configuration Options

### Core Settings

| Option | Default | Description |
|--------|---------|-------------|
| `system_prompt` | "You are a helpful assistant..." | System message explaining tool capabilities |
| `include_tools_in_system` | `true` | Whether to include tool definitions in system message |
| `include_reasoning_prefix` | `true` | Whether to add explanatory text before tool calls |

### Formatting Settings

| Option | Default | Description |
|--------|---------|-------------|
| `reasoning_prefix_template` | "I'll {action} for you." | Template for reasoning text ({action} is replaced) |
| `tool_call_format` | "&lt;tool_call&gt;\n{tool_call}\n&lt;/tool_call&gt;" | XML format for tool calls |
| `tool_response_as_json` | `true` | Format tool responses as JSON objects |

## Advanced Usage

### Custom Reasoning Prefixes

```yaml
config:
  include_reasoning_prefix: true
  reasoning_prefix_template: "Let me {action} to help you."
```

### Disable Reasoning Prefixes

```yaml
config:
  include_reasoning_prefix: false
```

### Custom Tool Call Format

```yaml
config:
  tool_call_format: "FUNCTION_CALL: {tool_call}"
```

### Plain Text Tool Responses

```yaml
config:
  tool_response_as_json: false
```

## Generated Action Descriptions

The formatter automatically generates contextual action descriptions:

| Tool Name | Input Parameters | Generated Action |
|-----------|------------------|------------------|
| `get_weather` | `{"location": "Paris"}` | "check the weather in Paris" |
| `get_time` | `{"timezone": "Asia/Tokyo"}` | "check the time in Asia/Tokyo" |
| `calculator` | `{"expression": "2+2"}` | "perform the calculation" |
| `unknown_tool` | Any | "use the unknown_tool tool" |

## Use Cases

### Sequential Tool Usage Training
Perfect for training models to make one tool call at a time in a clear sequence.

### API Integration Patterns
Matches real-world API calling patterns where each function call is a separate request/response cycle.

### Debugging and Tracing
Easier to trace individual tool executions and identify issues.

### Clear Boundaries
When you need explicit separation between different tool invocations.

## Example Configurations

### Minimal Configuration
```yaml
formatters:
  - name: "simple_tools"
    template: "builtin://single_tool_call"
    output: "tools.jsonl"
```

### Advanced Configuration
```yaml
formatters:
  - name: "advanced_tools"
    template: "builtin://single_tool_call"
    output: "advanced_tools.jsonl"
    config:
      system_prompt: "You are an expert assistant with access to specialized tools."
      include_tools_in_system: true
      include_reasoning_prefix: true
      reasoning_prefix_template: "To help with this, I'll {action}."
      tool_call_format: "<function_call>\n{tool_call}\n</function_call>"
      tool_response_as_json: true
```

### No Reasoning Configuration
```yaml
formatters:
  - name: "direct_tools"
    template: "builtin://single_tool_call"
    output: "direct_tools.jsonl"
    config:
      include_reasoning_prefix: false
      tool_response_as_json: false
```

## Integration with Agent Datasets

The Single Tool Call formatter works seamlessly with agent reasoning datasets generated using:

- `conversation_type: "agent_cot_tools"`
- Available tools configuration
- Multi-step reasoning samples

See the [Agent Tool Calling Guide](../guide/instruction-formats/agent-tool-calling/index.md) for more details on generating compatible datasets.

## Comparison with Other Formatters

| Formatter | Tool Calls per Message | Reasoning Style | Best For |
|-----------|------------------------|-----------------|----------|
| **Single Tool Call** | One | Separate reasoning prefix | Sequential tool usage |
| **Tool Calling** | Multiple | Embedded thinking traces | Complex multi-tool scenarios |
| **Harmony** | Multiple | Channel-based reasoning | gpt-oss models with channels |

## Troubleshooting

### Common Issues

**Missing Tool Definitions**
- Ensure `include_tools_in_system: true` in config
- Check that `available_tools` is specified in data engine config

**No Reasoning Prefixes**
- Set `include_reasoning_prefix: true`
- Customize with `reasoning_prefix_template`

**Tool Responses Not JSON**
- Enable with `tool_response_as_json: true`
- Check tool output format in source data

**Invalid Tool Call Format**
- Verify `tool_call_format` template syntax
- Ensure `{tool_call}` placeholder is present

### Validation

The formatter validates input samples require:
- `question` field
- `tool_used` field
- Either `answer` or `final_answer` field

Invalid samples are skipped with a warning.