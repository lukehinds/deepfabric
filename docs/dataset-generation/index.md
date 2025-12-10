# Dataset Generation

DeepFabric generates four types of synthetic training datasets, each designed for different model capabilities.

## Dataset Types

| Type | Purpose | Use Case |
|------|---------|----------|
| **Basic** | Simple Q&A pairs | General instruction following |
| **Reasoning** | Chain-of-thought traces | Step-by-step problem solving |
| **Agent (Single-Turn)** | Tool calls in one response | Simple tool use |
| **Agent (Multi-Turn)** | Extended tool conversations | Complex multi-step tasks |

## Generation Pipeline

All dataset types follow the same three-stage pipeline:

1. **Topic Generation** - Creates a tree or graph of subtopics from your root prompt
2. **Sample Generation** - Produces training examples for each topic
3. **Output** - Saves to JSONL with optional HuggingFace upload

## Quick Comparison

```yaml
# Basic: Simple Q&A
conversation:
  type: basic

# Reasoning: Chain-of-thought
conversation:
  type: chain_of_thought
  reasoning_style: freetext

# Agent Single-Turn: One-shot tool use
conversation:
  type: chain_of_thought
  reasoning_style: agent
  agent_mode: single_turn

# Agent Multi-Turn: Extended tool conversations
conversation:
  type: chain_of_thought
  reasoning_style: agent
  agent_mode: multi_turn
```

## Choosing a Dataset Type

**Basic datasets** work for general instruction-following tasks. The model learns to answer questions directly without explicit reasoning.

**Reasoning datasets** teach models to think before answering. The output includes a `reasoning` field with the model's thought process, useful for training models that explain their logic.

**Agent datasets** train tool-calling capabilities. Single-turn generates complete tool workflows in one response. Multi-turn creates extended conversations with multiple tool calls and observations, following a ReAct-style pattern.

## Next Steps

- [Basic Datasets](basic.md) - Simple Q&A generation
- [Reasoning Datasets](reasoning.md) - Chain-of-thought training data
- [Agent Datasets](agent.md) - Tool-calling datasets
- [Configuration Reference](configuration.md) - Full YAML options
