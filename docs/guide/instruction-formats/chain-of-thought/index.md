# Chain of Thought Instruction Formats

```python
import asyncio

def consume_tree(tree):
    async def _run():
        async for _ in tree.build_async():
            pass
    asyncio.run(_run())

def consume_graph(graph):
    async def _run():
        async for _ in graph.build_async():
            pass
    asyncio.run(_run())
```

Chain of Thought (CoT) datasets represent a significant advancement in training language models to exhibit transparent, step-by-step reasoning. Unlike traditional
conversation datasets that focus on final answers, CoT formats capture the *thinking process* that leads to solutions, enabling models to learn systematic problem-solving approaches.
## Why Chain of Thought Datasets?

### Enhanced Reasoning Capabilities
Traditional training data often presents solutions without showing the work. CoT datasets teach models to:
- Break down complex problems into manageable steps
- Show their reasoning process explicitly
- Build logical connections between concepts
- Verify their own thinking through structured steps

### Improved Transparency and Trust
Models trained on CoT data provide:
- **Explainable AI**: Users can follow the model's reasoning process
- **Error detection**: Faulty reasoning steps become visible and correctable
- **Educational value**: Students can learn from the model's problem-solving approach
- **Verification**: Experts can validate reasoning quality, not just final answers

### Better Performance on Complex Tasks
Research shows CoT training significantly improves performance on:
- Mathematical problem solving (GSM8K, MATH datasets)
- Logical reasoning tasks
- Multi-step code generation and debugging
- Scientific hypothesis formation and testing

## CoT Format Types in DeepFabric

DeepFabric supports five distinct Chain of Thought formats, including specialized agent formats for tool-calling scenarios:

### Traditional CoT Formats

The classic Chain of Thought formats focus on reasoning without external tools:

### 1. Free-text CoT
**Best for**: Natural language reasoning, math problems, general problem-solving

```json
{
  "question": "A baker has 24 cupcakes. If she puts 6 cupcakes in each box, how many boxes will she need?",
  "chain_of_thought": "I need to find how many groups of 6 can be made from 24 cupcakes. This is a division problem: 24 ÷ 6. Let me work this out: 6 × 1 = 6, 6 × 2 = 12, 6 × 3 = 18, 6 × 4 = 24. So 24 ÷ 6 = 4.",
  "final_answer": "4 boxes"
}
```

### 2. Structured CoT
**Best for**: Educational dialogues, tutoring scenarios, conversational learning

```json
{
  "messages": [
    {"role": "user", "content": "How do I solve 2x + 5 = 13?"},
    {"role": "assistant", "content": "Let's solve this step by step. First, we need to isolate the variable x..."}
  ],
  "reasoning_trace": [
    {"step_number": 1, "thought": "Identify this as a linear equation", "action": "classify"},
    {"step_number": 2, "thought": "Subtract 5 from both sides", "action": "calculate"}
  ],
  "final_answer": "x = 4"
}
```

### 3. Hybrid CoT
**Best for**: Complex problems requiring both natural and structured reasoning

```json
{
  "question": "Explain how quicksort works and analyze its time complexity",
  "chain_of_thought": "Quicksort is a divide-and-conquer algorithm that works by selecting a pivot element and partitioning the array around it...",
  "reasoning_trace": [
    {"step_number": 1, "thought": "Choose a pivot element", "action": "select"},
    {"step_number": 2, "thought": "Partition array around pivot", "action": "partition"},
    {"step_number": 3, "thought": "Recursively sort subarrays", "action": "recurse"}
  ],
  "final_answer": "Average case: O(n log n), Worst case: O(n²)"
}
```

### Agent CoT Formats

Advanced Chain of Thought formats that include external tool usage and reasoning:

### 4. Agent CoT with Tools (`agent_cot_tools`)
**Best for**: Training models to reason about tool selection and usage

```json
{
  "question": "What's the weather in Paris and what's 15% of the temperature?",
  "available_tools": [{"name": "get_weather", ...}, {"name": "calculator", ...}],
  "initial_analysis": "Need weather data for Paris, then calculate 15% of temperature",
  "tool_planning": [
    {
      "step_number": 1,
      "reasoning": "Get current weather for Paris to obtain temperature",
      "selected_tool": {"name": "get_weather", ...},
      "parameter_reasoning": {"location": "Paris specified by user"},
      "expected_result": "Current weather including temperature"
    }
  ],
  "tool_executions": [
    {
      "function": "get_weather",
      "arguments": {"location": "Paris"},
      "reasoning": "Getting Paris weather as planned",
      "result": "Paris: 18°C, partly cloudy"
    }
  ],
  "result_synthesis": "Combined weather data with calculation",
  "final_answer": "Paris is 18°C and partly cloudy. 15% of 18°C is 2.7°C."
}
```

### 5. Multi-Turn Agent CoT (`agent_cot_multi_turn`)
**Best for**: Conversational agents with progressive tool usage

```json
{
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": "I need to know your location first."},
    {"role": "user", "content": "Paris"},
    {"role": "assistant", "content": "Let me check Paris weather..."}
  ],
  "tool_planning_trace": [...],
  "tool_execution_trace": [...],
  "reasoning_summary": "Progressive information gathering and tool usage"
}
```

## Quick Start Guide

Get started with Chain of Thought datasets in under 5 minutes:

### 1. Choose Your Format

| Format | Use Case | Complexity | Best For |
|--------|----------|------------|----------|
| **Free-text** | Simple reasoning problems | Low | Math, logic, general Q&A |
| **Structured** | Educational conversations | Medium | Tutoring, step-by-step learning |
| **Hybrid** | Complex explanations | High | Algorithm analysis, proofs |
| **Agent CoT Tools** | Tool reasoning & selection | High | Function calling, tool usage training |
| **Agent Multi-Turn** | Conversational tool usage | Very High | Multi-turn agents, progressive reasoning |

### 2. Basic Configuration

Create a YAML configuration file:

```yaml
# cot-quickstart.yaml
dataset_system_prompt: "You are a helpful teacher who shows step-by-step reasoning."

topic_tree:
  topic_prompt: "Elementary mathematics and basic problem solving"
  provider: "openai"
  model: "gpt-4o-mini"
  degree: 2
  depth: 2

data_engine:
  provider: "openai"
  model: "gpt-4o-mini"
  conversation_type: "cot_freetext"  # or cot_structured, cot_hybrid
  reasoning_style: "mathematical"   # or logical, general

dataset:
  creation:
    num_steps: 5
    batch_size: 1
  save_as: "my_cot_dataset.jsonl"
```

### 3. Generate Your Dataset

```bash
# Using CLI
deepfabric generate cot-quickstart.yaml

# Or using Python
python -c "
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree

tree = Tree(topic_prompt='Math problems', provider='openai', model_name='gpt-4o-mini')
async for event in tree.build_async():
    if event['event'] == 'build_complete':
        print(f'Built {event[\"total_paths\"]} topics')

engine = DataSetGenerator(
    provider='openai',
    model_name='gpt-4o-mini',
    conversation_type='cot_freetext',
    reasoning_style='mathematical'
)

dataset = engine.create_data(num_steps=5, topic_model=tree)
dataset.save('quickstart_cot.jsonl')
"
```

### 4. Verify Your Results

```bash
# Check your dataset
head -n 1 my_cot_dataset.jsonl | jq .

# Should show a properly formatted CoT sample with:
# - question: The problem to solve
# - chain_of_thought: Step-by-step reasoning
# - final_answer: The solution
```

## What's Next?

### **Configuration**
- [YAML Configuration Guide](configuration/yaml-config.md) - Detailed parameter explanations
- [Python API Guide](configuration/python-api.md) - Programmatic usage patterns

### **Format Deep Dives**
- [Free-text CoT](formats/free-text.md) - Natural language reasoning
- [Structured CoT](formats/structured.md) - Conversation-based learning
- [Hybrid CoT](formats/hybrid.md) - Complex multi-modal reasoning
- [Agent Tool-Calling CoT](../agent-tool-calling/index.md) - Tool reasoning and function calling

### **Step-by-Step Tutorials**
- [Math Reasoning Dataset](tutorials/math-reasoning.md) - GSM8K-style problems

### **Advanced Topics**
- [Reasoning Styles](advanced/reasoning-styles.md) - Mathematical vs logical vs general

### **Reference**
- [Schema Reference](reference/schemas.md) - Complete format specifications
- [Troubleshooting](reference/troubleshooting.md) - Common issues and solutions

---
