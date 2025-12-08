# Reasoning Datasets

Reasoning datasets include chain-of-thought traces that show how the model arrives at its answer.

## When to Use

- Training models to explain their thinking
- Math, logic, or multi-step problems
- Tasks requiring transparent decision-making
- Improving model reliability through explicit reasoning

## Configuration

```yaml
topics:
  prompt: "Mathematical problem solving"
  mode: tree
  depth: 2
  degree: 3

generation:
  system_prompt: "Generate problems with step-by-step solutions."
  instructions: "Show clear reasoning before the final answer."

  conversation:
    type: chain_of_thought
    reasoning_style: freetext

  llm:
    provider: "openai"
    model: "gpt-4o"

output:
  system_prompt: |
    You are an AI that explains its reasoning step-by-step.
  num_samples: 10
  save_as: "reasoning-dataset.jsonl"
```

Key settings:
- `conversation.type: chain_of_thought`
- `conversation.reasoning_style: freetext`

## Output Format

Reasoning datasets include a `reasoning` field alongside messages:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an AI that explains its reasoning."
    },
    {
      "role": "user",
      "content": "Is a student with a 92 grade and 2 activities eligible for a scholarship that requires 90+ grade OR 3+ activities?"
    },
    {
      "role": "assistant",
      "content": "Yes, the student is eligible..."
    }
  ],
  "reasoning": {
    "style": "freetext",
    "content": "The scholarship requires EITHER a grade of 90+ OR 3+ activities. The student has a 92 grade, which exceeds the 90 threshold. Since the 'or' condition only needs one criterion to be true, the student qualifies based on their grade alone, regardless of having only 2 activities."
  }
}
```

## Reasoning Styles

DeepFabric supports one reasoning style for chain-of-thought:

| Style | Description |
|-------|-------------|
| `freetext` | Natural language explanation of thought process |

The `freetext` style produces readable, conversational reasoning traces.

## CLI Usage

```bash
deepfabric generate \
  --topic-prompt "Logic puzzles" \
  --conversation-type chain_of_thought \
  --reasoning-style freetext \
  --num-samples 20 \
  --output-save-as logic-dataset.jsonl
```

## Training Considerations

**Reasoning placement**: The `reasoning` field is stored separately from messages. During training, you can:
- Include reasoning in the assistant's response (visible to users)
- Use it as auxiliary training signal only
- Format it as a special token section (e.g., `<thinking>...</thinking>`)

**Model size matters**: Smaller models (< 3B parameters) may struggle to generate consistent reasoning. Consider using larger teacher models for generation.
