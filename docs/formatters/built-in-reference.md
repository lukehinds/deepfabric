# Built-in Formatter Reference

DeepFabric includes several built-in formatters for popular training frameworks and methodologies. This document provides comprehensive reference for all built-in formatters.

## Conversations Formatter

**Template**: `builtin://conversations.py`
**Use Case**: Generic conversations format for multiple training frameworks

### Description

The Conversations formatter transforms datasets into the standard conversations format with role/content pairs. This format is compatible with multiple training frameworks including Unsloth, Axolotl, and HuggingFace TRL, making it a versatile choice for conversational AI training.

### Configuration Options

```yaml
config:
  include_system: false                      # Default: false
  system_message: "Custom system message"    # Default: None
  roles_map:                                # Default: shown below
    user: "user"
    assistant: "assistant"
    system: "system"
```

### Input Formats Supported

- **Messages**: Chat format with role/content pairs
- **Q&A**: Question and answer fields
- **Instruction**: Instruction/input/output format
- **Direct**: User/assistant fields
- **Generic**: Any format with extractable conversation patterns

### Output Format

```json
{
  "conversations": [
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its simplicity and readability."}
  ]
}
```

### Example Configuration

```yaml
formatters:
- name: "conversations_training"
  template: "builtin://conversations.py"
  config:
    include_system: false  # Most frameworks apply system messages via chat templates
    roles_map:
      user: "user"
      assistant: "assistant"
  output: "conversations_dataset.jsonl"
```

### Framework Compatibility

**Unsloth:**

```python
dataset = load_dataset("your-username/your-dataset", split="train")
dataset = standardize_data_formats(dataset) # Placeholder for data standardization
dataset = dataset.map(formatting_prompts_func, batched=True) # Placeholder for formatting function
```

**Axolotl:**

```yaml
datasets:
  - path: your-username/your-dataset
    type: conversation
```

**HuggingFace TRL:**

```python
from trl import SFTTrainer
trainer = SFTTrainer(
    dataset=dataset,
    dataset_text_field="conversations"
)
```

---

## GRPO Formatter

**Template**: `builtin://grpo.py`
**Use Case**: Mathematical reasoning model training with GRPO (Generalized Reward-based Policy Optimization)

### Description

The GRPO formatter transforms datasets for mathematical reasoning training, wrapping reasoning processes in configurable tags and ensuring numerical answers are extractable for reward functions.

### Configuration Options

```yaml
config:
  reasoning_start_tag: "<start_working_out>"  # Default: "<start_working_out>"
  reasoning_end_tag: "<end_working_out>"      # Default: "<end_working_out>"
  solution_start_tag: "<SOLUTION>"            # Default: "<SOLUTION>"
  solution_end_tag: "</SOLUTION>"             # Default: "</SOLUTION>"
  system_prompt: "Custom system prompt..."    # Default: Auto-generated
  validate_numerical: true                    # Default: true
```

### Input Formats Supported

- **Messages**: Chat format with system/user/assistant roles
- **Q&A**: Question and answer fields with optional reasoning
- **Chain of Thought**: Questions with reasoning traces
- **Generic**: Any format with identifiable question/answer patterns

### Output Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION> and </SOLUTION>."
    },
    {
      "role": "user",
      "content": "What is 2 + 2?"
    },
    {
      "role": "assistant",
      "content": "<start_working_out>I need to add 2 and 2. This is basic addition.<end_working_out><SOLUTION>4</SOLUTION>"
    }
  ]
}
```

### Example Configuration

```yaml
formatters:
- name: "grpo_math"
  template: "builtin://grpo.py"
  config:
    reasoning_start_tag: "<think>"
    reasoning_end_tag: "</think>"
    solution_start_tag: "<answer>"
    solution_end_tag: "</answer>"
    validate_numerical: true
  output: "grpo_dataset.jsonl"
```

---

## Alpaca Formatter

**Template**: `builtin://alpaca.py`
**Use Case**: Instruction-following fine-tuning with the Stanford Alpaca format

### Description

The Alpaca formatter transforms datasets into the standard instruction-following format used by Stanford Alpaca and many other instruction-tuning projects.

### Configuration Options

```yaml
config:
  instruction_field: "instruction"           # Default: "instruction"
  input_field: "input"                      # Default: "input"
  output_field: "output"                    # Default: "output"
  include_empty_input: true                 # Default: true
  instruction_template: "Custom template"   # Default: None
```

### Input Formats Supported

- **Messages**: Chat format (system → instruction, user → input, assistant → output)
- **Direct**: Already has instruction/input/output fields
- **Q&A**: Question/answer pairs with optional context
- **Generic**: Any format with instruction-like patterns

### Output Format

```json
{
  "instruction": "Solve this math problem:",
  "input": "What is 15 + 27?",
  "output": "To solve 15 + 27, I'll add the numbers: 15 + 27 = 42"
}
```

### Example Configuration

```yaml
formatters:
- name: "alpaca_instruct"
  template: "builtin://alpaca.py"
  config:
    instruction_template: "### Instruction:\n{instruction}\n\n### Response:"
    include_empty_input: false
  output: "alpaca_dataset.jsonl"
```

---

## Harmony Formatter

**Template**: `builtin://harmony.py`
**Use Case**: OpenAI Harmony format for gpt-oss models with channels and TypeScript-style tool definitions

### Description

The Harmony formatter transforms datasets into the OpenAI Harmony Response Format, which is designed for the gpt-oss open-source models. It features a sophisticated role hierarchy, channel-based message organization (final, analysis, commentary), and TypeScript-style function definitions for tool calling.

### Configuration Options

```yaml
config:
  start_token: "<|start|>"                      # Default: "<|start|>"
  end_token: "<|end|>"                          # Default: "<|end|>"
  message_token: "<|message|>"                  # Default: "<|message|>"
  output_format: "text"                         # Default: "text" (or "structured")
  default_channel: "final"                      # Default: "final" (analysis/commentary/final)
  include_developer_role: false                 # Default: false
  developer_instructions: "Custom instructions" # Default: None
  system_message: "You are ChatGPT..."         # Default: "You are ChatGPT, a large language model trained by OpenAI."
  reasoning_level: "high"                       # Default: "high" (none/low/medium/high)
  knowledge_cutoff: "2024-01"                  # Default: "2024-01"
  current_date: "2024-03-15"                   # Default: None (optional, for deterministic output)
  include_metadata: true                        # Default: true
  tool_namespace: "functions"                   # Default: "functions"
```

### Role Hierarchy

The Harmony format enforces a strict role hierarchy (highest to lowest priority):

1. **system** - System instructions and metadata
2. **developer** - Developer instructions and tool definitions
3. **user** - User messages
4. **assistant** - Model responses with channel support
5. **tool** - Tool responses

### Channels

Assistant messages can be assigned to different channels:

- **final**: User-facing responses (default)
- **analysis**: Internal chain-of-thought reasoning (not safe for user display)
- **commentary**: Function tool calls and preambles

### Input Formats Supported

- **Messages**: Chat format with role/content pairs and optional tool calls
- **Q&A**: Question/answer pairs with optional chain_of_thought
- **Instruction**: Instruction/output patterns
- **Generic**: Any format with extractable conversation patterns

### Output Formats

**Text Format** (`output_format: "text"`):

```text
<|start|>system<|message|>
You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-01
Current date: 2024-03-15
Reasoning: high
# Valid channels: analysis, commentary, final
<|end|>
<|start|>developer<|message|>
# Instructions
Always provide detailed explanations

# Tools
namespace functions {
  type get_weather = (_: { location: string, unit?: "celsius" | "fahrenheit" }) => any;
}
<|end|>
<|start|>user<|message|>
What's the weather in London?
<|end|>
<|start|>assistant<|channel|>analysis<|message|>
I need to check the weather in London using the weather tool.
<|end|>
<|start|>assistant<|channel|>commentary<|recipient|>functions.get_weather<|message|>
{"location": "London", "unit": "celsius"}
<|end|>
<|start|>tool<|message|>
{"temperature": 18, "condition": "cloudy"}
<|end|>
<|start|>assistant<|channel|>final<|message|>
The weather in London is currently 18°C with cloudy conditions.
<|end|>
```

**Structured Format** (`output_format: "structured"`):

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are ChatGPT, a large language model...\nKnowledge cutoff: 2024-01\nReasoning: high\n# Valid channels: analysis, commentary, final",
      "channel": null,
      "recipient": null
    },
    {
      "role": "developer",
      "content": "# Instructions\nAlways provide detailed explanations\n\n# Tools\nnamespace functions {\n  type get_weather = (_: { location: string, unit?: \"celsius\" | \"fahrenheit\" }) => any;\n}",
      "channel": null,
      "recipient": null
    },
    {
      "role": "user",
      "content": "What's the weather in London?",
      "channel": null,
      "recipient": null
    },
    {
      "role": "assistant",
      "content": "I need to check the weather in London using the weather tool.",
      "channel": "analysis",
      "recipient": null
    },
    {
      "role": "assistant",
      "content": "{\"location\": \"London\", \"unit\": \"celsius\"}",
      "channel": "commentary",
      "recipient": "functions.get_weather"
    },
    {
      "role": "tool",
      "content": "{\"temperature\": 18, \"condition\": \"cloudy\"}",
      "channel": null,
      "recipient": null
    },
    {
      "role": "assistant",
      "content": "The weather in London is currently 18°C with cloudy conditions.",
      "channel": "final",
      "recipient": null
    }
  ]
}
```

### Tool Definitions

Tools are defined using TypeScript-style type syntax in the developer message:

```typescript
namespace functions {
  type calculator = (_: {
    operation: "add" | "subtract" | "multiply" | "divide",
    a: number,
    b: number
  }) => any;

  type web_search = (_: {
    query: string,
    limit?: number
  }) => any;
}
```

### Example Configurations

**Basic Chat Configuration**:

```yaml
formatters:
- name: "harmony_chat"
  template: "builtin://harmony.py"
  config:
    output_format: "text"
    default_channel: "final"
    include_metadata: true
  output: "harmony_chat.jsonl"
```

**Advanced Configuration with Tools**:

```yaml
formatters:
- name: "harmony_tools"
  template: "builtin://harmony.py"
  config:
    output_format: "text"
    include_developer_role: true
    developer_instructions: |
      You are an expert assistant with access to various tools.
      Always think through your approach before using tools.
    reasoning_level: "high"
    default_channel: "final"
    tool_namespace: "functions"
    current_date: "2024-03-15"  # For deterministic output
  output: "harmony_tools.jsonl"
```

**Chain-of-Thought Configuration**:

```yaml
formatters:
- name: "harmony_cot"
  template: "builtin://harmony.py"
  config:
    output_format: "structured"
    default_channel: "analysis"  # Default to analysis channel for reasoning
    reasoning_level: "high"
    include_metadata: true
  output: "harmony_cot.jsonl"
```

### Special Features

1. **Multiple Tool Calls**: Handles multiple tool calls in a single message by creating separate messages for each tool call
2. **Deterministic Output**: Use `current_date` config to ensure reproducible outputs (no dynamic timestamps)
3. **Tool Name Validation**: Skips tools without names to prevent namespace conflicts
4. **Flexible Channels**: Automatically assigns channels based on message content (reasoning → analysis, tool calls → commentary)

---

## Single Tool Call Formatter

**Template**: `builtin://single_tool_call.py`
**Use Case**: Individual tool calling format where each tool call is in its own message exchange

### Description

The Single Tool Call formatter transforms agent reasoning datasets into a conversational format where each tool call is handled in its own message exchange, rather than embedding multiple tool calls within a single response. This format is ideal for training models to make single, focused tool calls with clear reasoning prefixes.

### Configuration Options

```yaml
config:
  system_prompt: "You are a helpful assistant..."            # Default: "You are a helpful assistant with access to the following functions. Use them if required:"
  include_tools_in_system: true                             # Default: true
  include_reasoning_prefix: true                             # Default: true
  reasoning_prefix_template: "I'll {action} for you."       # Default: "I'll {action} for you."
  tool_call_format: "<tool_call>\n{tool_call}\n</tool_call>" # Default: "<tool_call>\n{tool_call}\n</tool_call>"
  tool_response_as_json: true                               # Default: true
```

### Input Formats Supported

- **Agent CoT Tools**: Agent reasoning with tool usage from `agent_cot_tools` conversation type
- **Simple Agent CoT**: Simple agent chain-of-thought with tool selection and multi-tool support
- **Hybrid Agent CoT**: Hybrid agent reasoning with tool traces and multi-tool support

### Output Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant with access to the following functions. Use them if required:\n\n{\"functions\": [{\"name\": \"get_weather\", \"description\": \"Get weather information\", \"parameters\": {...}}]}"
    },
    {
      "role": "user",
      "content": "What's the weather in Paris and the time in Tokyo?"
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
      "content": "Now let me check the time in Tokyo.\n\n<tool_call>\n{\"name\": \"get_time\", \"arguments\": {\"timezone\": \"Asia/Tokyo\"}}\n</tool_call>"
    },
    {
      "role": "tool",
      "content": "{\"time\": \"2024-01-15 22:30:00\", \"timezone\": \"JST\"}"
    },
    {
      "role": "assistant",
      "content": "The weather in Paris is currently 15°C and partly cloudy. The current time in Tokyo is 10:30 PM JST."
    }
  ]
}
```

### Key Features

1. **Individual Tool Calls**: Each tool call is in its own assistant message, followed by its tool response
2. **Reasoning Prefixes**: Optional natural language explanations before each tool call
3. **Tool Action Generation**: Automatically generates contextual action descriptions (e.g., "check the weather in Paris")
4. **JSON Tool Responses**: Formats tool outputs as valid JSON for consistency
5. **System Tool Integration**: Includes available tools in the system message with proper function definitions

### Example Configuration

```yaml
formatters:
- name: "single_tool_format"
  template: "builtin://single_tool_call.py"
  config:
    system_prompt: "You are a helpful assistant with access to the following functions. Use them if required:"
    include_tools_in_system: true
    include_reasoning_prefix: true
    reasoning_prefix_template: "I'll {action} for you."
    tool_call_format: "<tool_call>\n{tool_call}\n</tool_call>"
    tool_response_as_json: true
  output: "single_tool_call_dataset.jsonl"
```

### Advanced Configuration

**Disable Reasoning Prefixes**:

```yaml
config:
  include_reasoning_prefix: false
  tool_call_format: "<tool_call>\n{tool_call}\n</tool_call>"
```

**Custom Tool Call Format**:

```yaml
config:
  tool_call_format: "FUNCTION_CALL: {tool_call}"
  reasoning_prefix_template: "Let me {action}."
```

**Plain Text Tool Responses**:

```yaml
config:
  tool_response_as_json: false
```

### Use Cases

- **Single Function Training**: Training models to make one tool call at a time
- **Sequential Tool Usage**: Teaching models to chain tool calls in separate messages
- **Clear Tool Boundaries**: When you need explicit separation between tool calls and responses
- **Function Call Debugging**: Easier to trace individual tool executions
- **API Integration**: Matches many real-world API calling patterns

---

## TRL SFT Tools Formatter

**Template**: `builtin://trl_sft_tools`
**Use Case**: HuggingFace TRL SFTTrainer tool/function calling fine-tuning

### Description

The TRL SFT Tools formatter transforms DeepFabric agent reasoning datasets into the format required by HuggingFace TRL's SFTTrainer for tool calling fine-tuning. It converts DeepFabric's internal `available_tools` field to the `tools` field in OpenAI function calling schema format, which is the standard format expected by TRL and other modern training frameworks.

This formatter is specifically designed for training models with tool/function calling capabilities using supervised fine-tuning.

### Configuration Options

```yaml
config:
  include_system_prompt: true                    # Default: true
  system_prompt_override: null                   # Default: null (uses original)
  validate_tool_schemas: true                    # Default: true
  remove_available_tools_field: false            # Default: false
```

### Input Formats Supported

- **Agent CoT Tools**: Agent reasoning with tool usage from `agent_cot_tools` conversation type
- **Agent CoT Hybrid**: Hybrid agent CoT with structured reasoning
- **Agent CoT Multi-Turn**: Multi-turn agent conversations with tools
- **XLAM Multi-Turn**: XLAM 2.0 format multi-turn tool calling
- **Any format with messages and available_tools fields**

### Output Format

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant with access to various tools..."
    },
    {
      "role": "user",
      "content": "What's the weather in Paris?"
    },
    {
      "role": "assistant",
      "content": "Let me check the weather for you..."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "City name"
            },
            "unit": {
              "type": "string",
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    }
  ]
}
```

### Key Features

1. **OpenAI Schema Conversion**: Automatically converts DeepFabric tool definitions to OpenAI function calling schema
2. **Type Mapping**: Maps DeepFabric types (str, int, float, bool, list, dict) to JSON Schema types
3. **Required Parameter Handling**: Properly distinguishes required vs optional parameters
4. **Schema Validation**: Optional validation of tool schemas for correctness
5. **System Prompt Control**: Override or preserve system prompts as needed

### Example Configuration

```yaml
formatters:
- name: "trl_sft"
  template: "builtin://trl_sft_tools"
  output: "trl_sft_dataset.jsonl"
  config:
    include_system_prompt: true
    validate_tool_schemas: true
    remove_available_tools_field: false
```

### Advanced Configuration

**Override System Prompt for Training**:

```yaml
config:
  include_system_prompt: true
  system_prompt_override: |
    You are a function calling AI model. You are provided with function signatures
    within <tools></tools> XML tags. You may call one or more functions to assist
    with the user query.
```

**Clean Output (Remove Original Tools Field)**:

```yaml
config:
  remove_available_tools_field: true
```

**Disable Schema Validation (Performance)**:

```yaml
config:
  validate_tool_schemas: false
```

### Quick Start with HuggingFace Hub

Download and format datasets directly from the HuggingFace Hub:

```bash
# Download from Hub and format to TRL SFT Tools format
deepfabric format --repo lukehinds/smol-test-sample --formatter trl

# Specify a different split (default is 'train')
deepfabric format --repo username/dataset-name --formatter trl --split validation

# Custom output path
deepfabric format --repo org/agent-dataset --formatter trl -o trl_formatted.jsonl
```

This workflow is ideal for:

- Converting existing agent/tool datasets to TRL format
- Reformatting community datasets for your training pipeline
- Experimenting with different formatters on public datasets

### Usage with TRL SFTTrainer

After formatting your dataset, use it directly with TRL's SFTTrainer:

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your formatted dataset
dataset = load_dataset("json", data_files="trl_sft_dataset.jsonl", split="train")

# Setup model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Configure training
training_args = SFTConfig(
    output_dir="./tool_calling_model",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
)

# Train with tool calling support
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

### Type Mapping Reference

| DeepFabric Type | JSON Schema Type |
|-----------------|------------------|
| `str`           | `string`         |
| `int`           | `integer`        |
| `float`         | `number`         |
| `bool`          | `boolean`        |
| `list`          | `array`          |
| `dict`          | `object`         |

### References

- [HuggingFace TRL SFTTrainer Tool Calling Docs](https://huggingface.co/docs/trl/en/sft_trainer#tool-calling-with-sft)
- [OpenAI Function Calling Format](https://platform.openai.com/docs/guides/function-calling)

---

## ChatML Formatter

**Template**: `builtin://chatml.py`
**Use Case**: Conversation format with clear role delineation using ChatML markup

### Description

The ChatML formatter creates standardized conversation formats with special tokens for role boundaries, compatible with many modern chat-based training frameworks.

### Configuration Options

```yaml
config:
  start_token: "<|im_start|>"                    # Default: "<|im_start|>"
  end_token: "<|im_end|>"                        # Default: "<|im_end|>"
  output_format: "structured"                    # Default: "structured" (or "text")
  default_system_message: "You are helpful..."   # Default: "You are a helpful assistant."
  require_system_message: false                  # Default: false
```

### Input Formats Supported

- **Messages**: Direct chat format
- **Q&A**: Question/answer pairs
- **Instruction-Response**: Instruction-following patterns
- **Generic**: Any conversational patterns

### Output Formats

**Structured Format** (`output_format: "structured"`):

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help you today?"}
  ]
}
```

**Text Format** (`output_format: "text"`):

```json
{
  "text": "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\nHello!\n<|im_end|>\n<|im_start|>assistant\nHi there! How can I help you today?\n<|im_end|>"
}
```

### Example Configuration

```yaml
formatters:
- name: "chatml_chat"
  template: "builtin://chatml.py"
  config:
    output_format: "text"
    require_system_message: true
    default_system_message: "You are a helpful AI assistant specialized in mathematics."
  output: "chatml_dataset.jsonl"
```

---

## Choosing the Right Formatter

### For Mathematical Reasoning Training

- **GRPO**: When training models to show step-by-step reasoning with extractable answers
- **Harmony**: For models that need to show internal reasoning (analysis channel) separate from final answers
- **Alpaca**: For instruction-following with math problems
- **ChatML**: For conversational math tutoring scenarios

### For General Instruction Following

- **Alpaca**: Standard instruction-following format
- **ChatML**: When you need conversation context and role clarity
- **Harmony**: For gpt-oss models with developer instructions and role hierarchy
- **Conversations**: For Unsloth, Axolotl, or HF TRL training with conversations format

### For Chat and Dialogue

- **Harmony**: Advanced format with channels, tool support, and role hierarchy for gpt-oss models
- **ChatML**: ChatML-compatible format with `<|im_start|>/<|im_end|>` delimiters
- **Conversations**: Standard conversations format for multiple frameworks
- **Alpaca**: Single-turn instruction-response pairs

### For Tool/Function Calling

- **OpenAI Schema**: HuggingFace TRL SFTTrainer using the OpenAI function schema (recommended for modern training workflows)
- **Single Tool Call**: Individual tool call format with each call in its own message exchange
- **Tool Calling**: Embedded tool calling with thinking traces and multiple tools per response
- **Harmony**: TypeScript-style function definitions with channels for tool calls and responses
- **Custom formatters**: For specific tool calling conventions

### For Custom Requirements

Create a [custom formatter](custom-formatter-guide.md) that inherits from BaseFormatter.

## Validation and Error Handling

All built-in formatters include:

- **Input Validation**: Checks if the input data is compatible
- **Output Validation**: Ensures the formatted output meets requirements
- **Error Messages**: Clear error descriptions for debugging
- **Graceful Degradation**: Handles edge cases without crashing

## Performance Notes

- Built-in formatters are optimized for both speed and memory efficiency
- Large datasets are processed in streaming fashion when possible
- Validation can be disabled for better performance in production
- Formatter instances are cached for repeated use
