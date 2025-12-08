# Basic Datasets

Basic datasets generate simple question-answer pairs without reasoning traces or tool calls.

## When to Use

- General instruction-following tasks
- Domain-specific Q&A (e.g., customer support, FAQs)
- Models that don't need to show reasoning
- Quick dataset generation with minimal configuration

## Configuration

```yaml
topics:
  prompt: "Python programming fundamentals"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate clear, educational Q&A pairs."
  instructions: "Create diverse questions with detailed answers."

  conversation:
    type: basic

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  system_prompt: |
    You are a helpful assistant.
  num_samples: 10
  save_as: "dataset.jsonl"
```

The key setting is `conversation.type: basic`.

## Output Format

Basic datasets produce standard chat-format JSONL:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "What are Python's numeric data types?"
    },
    {
      "role": "assistant",
      "content": "Python has three built-in numeric types: integers (int), floating-point numbers (float), and complex numbers (complex)..."
    }
  ]
}
```

## CLI Usage

Generate a basic dataset from the command line:

```bash
deepfabric generate config.yaml
```

Or with inline options:

```bash
deepfabric generate \
  --topic-prompt "Machine learning basics" \
  --conversation-type basic \
  --num-samples 50 \
  --provider openai \
  --model gpt-4o \
  --output-save-as ml-dataset.jsonl
```

## Tips

**Topic depth and degree** control dataset diversity. A tree with `depth: 3` and `degree: 3` produces 40 unique topics (1 + 3 + 9 + 27).

**System prompts** differ between generation and output:
- `generation.system_prompt` - Instructions for the LLM generating examples
- `output.system_prompt` - The system message included in training data

**Batch size** affects generation speed. Higher values (`batch_size: 5`) parallelize requests but may hit rate limits.
