# Built-in Formatter Reference

DeepFabric includes several built-in formatters for popular training frameworks and methodologies. This document provides comprehensive reference for all built-in formatters.

## Im Format Formatter

**Template**: `builtin://im_format.py`
**Use Case**: ChatML-compatible training with `<|im_start|>` and `<|im_end|>` delimiters

### Description

The Im Format formatter transforms datasets into the format used by models that expect conversation delimiters with `<|im_start|>` and `<|im_end|>` tokens. This format is widely used for chat models and is compatible with ChatML and similar conversation formats.

### Configuration Options

```yaml
config:
  include_system: true                       # Default: false
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

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is Python?<|im_end|>
<|im_start|>assistant
Python is a high-level, interpreted programming language known for its simplicity and readability.<|im_end|>
```

### Example Configuration

```yaml
formatters:
- name: "chatml_training"
  template: "builtin://im_format.py"
  config:
    include_system: true
    system_message: |
      You are an expert programming assistant.
      Provide clear, accurate, and practical answers.
    roles_map:
      user: "user"
      assistant: "assistant"
      system: "system"
  output: "chatml_dataset.jsonl"
```

---

## Unsloth Formatter

**Template**: `builtin://unsloth.py`
**Use Case**: Training with Unsloth framework using conversations format

### Description

The Unsloth formatter transforms datasets into the conversations format expected by Unsloth training notebooks. This enables seamless integration with Unsloth's training pipeline and chat templates.

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
- name: "unsloth_training"
  template: "builtin://unsloth.py"
  config:
    include_system: false  # Unsloth applies system messages via chat templates
    roles_map:
      user: "user"
      assistant: "assistant"
  output: "unsloth_dataset.jsonl"
```

### Integration with Unsloth Notebooks

After formatting with this formatter and uploading to HuggingFace Hub, use directly in Unsloth notebooks:

```python
# Replace the default dataset
dataset = load_dataset("your-username/your-dataset", split="train")

# The rest of the notebook works unchanged
dataset = standardize_data_formats(dataset)
dataset = dataset.map(formatting_prompts_func, batched=True)
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
- **Alpaca**: For instruction-following with math problems
- **ChatML**: For conversational math tutoring scenarios

### For General Instruction Following
- **Alpaca**: Standard instruction-following format
- **ChatML**: When you need conversation context and role clarity
- **Unsloth**: When using Unsloth training notebooks with conversations format

### For Chat and Dialogue
- **ChatML**: Structured conversations with multiple turns
- **Im Format**: ChatML-compatible format with `<|im_start|>/<|im_end|>` delimiters
- **Unsloth**: Conversations format for Unsloth framework integration
- **Alpaca**: Single-turn instruction-response pairs

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