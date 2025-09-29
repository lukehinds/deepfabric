# Free-text Chain of Thought Format

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

The free-text Chain of Thought format is the simplest and most widely-used CoT structure, perfect for problems that benefit from natural language reasoning. Inspired by datasets like GSM8K, this format presents a question, shows the step-by-step thinking process in natural language, and provides a final answer.
## When to Use Free-text CoT

### Ideal Use Cases
- **Mathematical word problems**: Grade school through high school math
- **Logic puzzles**: Reasoning through constraints and deductions
- **General problem-solving**: Everyday reasoning tasks
- **Reading comprehension**: Analyzing passages and drawing conclusions
- **Simple scientific problems**: Basic physics, chemistry calculations

### Strengths
- **Natural reasoning flow**: Mirrors how humans think through problems
- **Easy to understand**: Readable by humans without technical knowledge
- **Flexible structure**: Can handle varied reasoning patterns
- **Token efficient**: Compact representation of reasoning

### Limitations
- **Less structured**: Harder to validate individual reasoning steps
- **No explicit step tracking**: Can't easily verify intermediate calculations
- **Limited metadata**: No action classification or step numbering

## Schema Specification

```python
class FreeTextCoT(BaseModel):
    """Chain of Thought dataset with natural language reasoning."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    final_answer: str = Field(description="The definitive answer to the question")
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ | The problem statement or question to be solved |
| `chain_of_thought` | string | ✅ | Step-by-step reasoning in natural language |
| `final_answer` | string | ✅ | The final solution or answer |

## Examples by Domain

### Mathematics - Basic Arithmetic

```json
{
  "question": "Sarah has 15 stickers. She gives 4 stickers to her brother and 3 stickers to her sister. How many stickers does Sarah have left?",
  "chain_of_thought": "Let me work through this step by step. Sarah starts with 15 stickers. She gives away 4 stickers to her brother, so she has 15 - 4 = 11 stickers left. Then she gives 3 stickers to her sister, so she has 11 - 3 = 8 stickers remaining.",
  "final_answer": "8 stickers"
}
```

### Mathematics - Algebra

```json
{
  "question": "Solve for x: 3x + 7 = 22",
  "chain_of_thought": "I need to isolate x on one side of the equation. First, I'll subtract 7 from both sides: 3x + 7 - 7 = 22 - 7, which gives me 3x = 15. Now I'll divide both sides by 3: 3x ÷ 3 = 15 ÷ 3, so x = 5. Let me check: 3(5) + 7 = 15 + 7 = 22 ✓",
  "final_answer": "x = 5"
}
```

### Logic - Reasoning Puzzle

```json
{
  "question": "In a race, Alice finished before Bob, Bob finished before Charlie, and Charlie finished before David. If there were only these four runners, what was Alice's finishing position?",
  "chain_of_thought": "Let me work out the finishing order based on the given information. Alice finished before Bob, so Alice is ahead of Bob. Bob finished before Charlie, so the order so far is: Alice, Bob, Charlie. Charlie finished before David, so David is last. The complete finishing order is: Alice (1st), Bob (2nd), Charlie (3rd), David (4th).",
  "final_answer": "1st place"
}
```

### Science - Physics

```json
{
  "question": "A ball is thrown upward with an initial velocity of 20 m/s. How high will it go? (Use g = 10 m/s²)",
  "chain_of_thought": "I need to find the maximum height of the ball. At maximum height, the velocity becomes zero. I can use the kinematic equation: v² = u² + 2as, where v = final velocity (0 m/s), u = initial velocity (20 m/s), a = acceleration (-g = -10 m/s²), and s = displacement (height). Substituting: 0² = 20² + 2(-10)s. This gives me: 0 = 400 - 20s. Solving for s: 20s = 400, so s = 20 meters.",
  "final_answer": "20 meters"
}
```

### Reading Comprehension

```json
{
  "question": "Based on the passage: 'The ancient library of Alexandria was one of the largest and most significant libraries of the ancient world. It was part of the larger research institution called the Mouseion.' What was the relationship between the library and the Mouseion?",
  "chain_of_thought": "Let me carefully read the passage to understand the relationship. The passage states that the library of Alexandria 'was part of the larger research institution called the Mouseion.' This means the library was a component or section within the Mouseion, not a separate entity. The Mouseion was the larger organization, and the library was one part of it.",
  "final_answer": "The library was part of (or a component within) the larger Mouseion research institution."
}
```

## Configuration for Free-text CoT

### YAML Configuration

```yaml
# free-text-cot.yaml
dataset_system_prompt: "You are a helpful teacher who explains step-by-step reasoning clearly and naturally."

topic_tree:
  topic_prompt: "Elementary and middle school mathematics, basic science, and logic problems"
  provider: "openai"
  model: "gpt-4o-mini"
  degree: 3
  depth: 2
  temperature: 0.7

data_engine:
  instructions: "Create clear problems that require step-by-step thinking to solve."
  generation_system_prompt: "You are an educator creating practice problems with detailed reasoning."

  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.3

  # Free-text CoT specific settings
  conversation_type: "cot_freetext"
  reasoning_style: "mathematical"  # or "logical" or "general"

dataset:
  creation:
    num_steps: 10
    batch_size: 1
    sys_msg: false  # Free-text CoT doesn't use system messages
  save_as: "math_reasoning_freetext.jsonl"
```

### Python API

```python
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree

# Create topic tree
tree = Tree(
    topic_prompt="Mathematical word problems and logical reasoning",
    provider="openai",
    model_name="gpt-4o-mini",
    degree=2,
    depth=2,
    temperature=0.7
)

# Build tree
async for event in tree.build_async():
    if event['event'] == 'build_complete':
        print(f"Built {event['total_paths']} topic paths")

# Create generator
generator = DataSetGenerator(
    instructions="Create problems requiring analytical thinking.",
    generation_system_prompt="You are a math tutor creating practice problems.",
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.3,
    conversation_type="cot_freetext",
    reasoning_style="mathematical"
)

# Generate dataset
dataset = generator.create_data(
    num_steps=10,
    batch_size=1,
    topic_model=tree,
    sys_msg=False
)

# Save results
dataset.save("freetext_cot_dataset.jsonl")
print(f"Generated {len(dataset.samples)} reasoning examples")
```

## Best Practices

### Reasoning Quality Guidelines

**✅ Good Chain of Thought:**
- Shows clear logical progression
- Explains the reasoning behind each step
- Uses appropriate mathematical or logical language
- Verifies the answer when possible
- Breaks complex problems into manageable parts

**❌ Poor Chain of Thought:**
- Jumps to conclusions without explanation
- Skips important intermediate steps
- Uses vague or unclear language
- Makes calculation errors without verification
- Rushes through complex reasoning

### Example Comparison

**Good Example:**
```json
{
  "question": "What is 15% of 240?",
  "chain_of_thought": "To find 15% of 240, I need to multiply 240 by 0.15. I can break this down: 15% = 15/100 = 0.15. So I need to calculate 240 × 0.15. I can do this as: 240 × 0.15 = 240 × (0.1 + 0.05) = (240 × 0.1) + (240 × 0.05) = 24 + 12 = 36.",
  "final_answer": "36"
}
```

**Poor Example:**
```json
{
  "question": "What is 15% of 240?",
  "chain_of_thought": "15% of 240 is 36.",
  "final_answer": "36"
}
```

### Reasoning Style Impact

The `reasoning_style` parameter affects the generated reasoning patterns:

**Mathematical Style:**
- Focuses on calculations and numerical reasoning
- Shows algebraic manipulations step-by-step
- Emphasizes verification through checking

**Logical Style:**
- Emphasizes premise-conclusion relationships
- Uses formal logical structure
- Highlights reasoning patterns and inference rules

**General Style:**
- Flexible approach adapting to problem type
- Natural language explanations
- Common-sense reasoning patterns

## Quality Validation

### Automated Checks
- **Length validation**: Chain of thought should be substantial (>50 characters)
- **Answer consistency**: Final answer should be supported by reasoning
- **Format compliance**: Must follow exact schema structure

### Human Evaluation Criteria
- **Correctness**: Is the reasoning mathematically/logically sound?
- **Completeness**: Are all necessary steps shown?
- **Clarity**: Would a student understand the explanation?
- **Efficiency**: Is the reasoning path reasonably direct?

## Common Issues and Solutions

### Issue: Reasoning Too Brief
**Problem**: Chain of thought lacks detail
**Solution**: Adjust temperature (0.2-0.4) and emphasize "step-by-step" in prompts

### Issue: Incorrect Calculations
**Problem**: Mathematical errors in reasoning
**Solution**: Add verification steps in prompts, use "double-check" instructions

### Issue: Inconsistent Format
**Problem**: Answers don't match expected format
**Solution**: Use structured output with Outlines (automatic in DeepFabric)

### Issue: Repetitive Problems
**Problem**: Similar questions generated repeatedly
**Solution**: Increase topic tree diversity (higher degree/depth values)

## Performance Optimization

### Token Efficiency
- Free-text CoT is typically the most token-efficient format
- Average reasoning length: 100-300 tokens
- Optimal batch size: 1-3 samples per batch

### Generation Speed
- Faster than structured formats due to simpler schema
- Recommended for high-volume generation
- Works well with all provider types (OpenAI, Anthropic, local models)

## Next Steps

- **Try Structured CoT**: For conversational reasoning → [Structured CoT Guide](structured.md)
- **Explore Hybrid CoT**: For complex multi-modal reasoning → [Hybrid CoT Guide](hybrid.md)
- **Advanced Configuration**: → [Configuration Guide](../configuration/yaml-config.md)
- **Domain-Specific Tutorials**: → [Math Reasoning Tutorial](../tutorials/math-reasoning.md)