# Structured Chain of Thought Format

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

The structured Chain of Thought format combines the natural flow of conversations with explicit reasoning traces, making it ideal for educational dialogues and tutoring scenarios. This format captures both the conversational interaction and the underlying thought process of the assistant.
## When to Use Structured CoT

### Ideal Use Cases
- **Educational tutoring**: Step-by-step learning conversations
- **Code explanation**: Algorithm walkthroughs with reasoning
- **Scientific instruction**: Hypothesis formation and testing
- **Interactive problem-solving**: Guided discovery learning
- **Technical support**: Diagnostic reasoning with explanations

### Strengths
- **Dual representation**: Both conversation and reasoning structure
- **Educational value**: Shows both what is said and why
- **Trackable steps**: Numbered reasoning progression
- **Rich metadata**: Action classification and step numbering
- **Natural interaction**: Maintains conversational flow

### Limitations
- **More complex**: Requires both conversation and reasoning skills
- **Token overhead**: Larger than free-text due to dual structure
- **Coordination challenge**: Keeping conversation and reasoning aligned

## Schema Specification

```python
class StructuredCoT(BaseModel):
    """Chain of Thought dataset with structured reasoning trace."""

    messages: list[ChatMessage] = Field(description="Conversation messages", min_length=1)
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")

class ChatMessage(BaseModel):
    """A single message in a conversation."""
    role: str = Field(description="The role of the message sender")
    content: str = Field(description="The content of the message")

class ReasoningStep(BaseModel):
    """A single step in a chain of reasoning."""
    step_number: int = Field(description="The step number in the reasoning chain")
    thought: str = Field(description="The reasoning or thought for this step")
    action: str = Field(description="Any action taken as part of this reasoning step")
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `messages` | array | ✅ | Conversation between user and assistant |
| `reasoning_trace` | array | ✅ | Structured breakdown of assistant's reasoning |
| `final_answer` | string | ✅ | Definitive answer or conclusion |

#### Message Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | ✅ | "user", "assistant", or "system" |
| `content` | string | ✅ | The message text |

#### Reasoning Step Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_number` | integer | ✅ | Sequential step number (1, 2, 3...) |
| `thought` | string | ✅ | The reasoning or analysis for this step |
| `action` | string | ✅ | Classification of the action taken |

## Examples by Domain

### Mathematics - Algebra Tutorial

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful math tutor who guides students through problems step-by-step."
    },
    {
      "role": "user",
      "content": "I'm struggling with solving 2x + 8 = 20. Can you help me?"
    },
    {
      "role": "assistant",
      "content": "I'd be happy to help! Let's solve this step by step. First, what do you think our goal is when solving for x?"
    },
    {
      "role": "user",
      "content": "We want to get x by itself on one side?"
    },
    {
      "role": "assistant",
      "content": "Exactly! We want to isolate x. Let's start by getting rid of the +8. What operation would cancel out adding 8?"
    },
    {
      "role": "user",
      "content": "Subtracting 8?"
    },
    {
      "role": "assistant",
      "content": "Perfect! So let's subtract 8 from both sides: 2x + 8 - 8 = 20 - 8, which gives us 2x = 12. Now what do we do to get x by itself?"
    },
    {
      "role": "user",
      "content": "Divide both sides by 2?"
    },
    {
      "role": "assistant",
      "content": "Excellent! 2x ÷ 2 = 12 ÷ 2, so x = 6. Let's verify: 2(6) + 8 = 12 + 8 = 20 ✓"
    }
  ],
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Student needs help with linear equation 2x + 8 = 20. I should guide them through the process rather than just solve it.",
      "action": "assess_problem"
    },
    {
      "step_number": 2,
      "thought": "I'll start by confirming they understand the goal: isolating the variable x.",
      "action": "clarify_objective"
    },
    {
      "step_number": 3,
      "thought": "They understand the goal. Now I'll guide them to identify the first step: eliminating the constant term.",
      "action": "guide_step"
    },
    {
      "step_number": 4,
      "thought": "They correctly identified subtraction. I'll show the calculation: 2x + 8 - 8 = 20 - 8 → 2x = 12.",
      "action": "demonstrate"
    },
    {
      "step_number": 5,
      "thought": "Now they need to eliminate the coefficient. I'll guide them to division.",
      "action": "guide_step"
    },
    {
      "step_number": 6,
      "thought": "They got it right. I'll complete the solution and verify the answer to reinforce good practices.",
      "action": "verify_solution"
    }
  ],
  "final_answer": "x = 6"
}
```

### Computer Science - Algorithm Explanation

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a computer science instructor who explains algorithms clearly with step-by-step reasoning."
    },
    {
      "role": "user",
      "content": "Can you explain how binary search works? I understand it's faster than linear search but don't get how."
    },
    {
      "role": "assistant",
      "content": "Great question! Binary search is indeed much faster. The key insight is that it works on sorted arrays and eliminates half the remaining elements with each comparison. Let me walk you through it."
    },
    {
      "role": "user",
      "content": "Okay, so we need a sorted array first?"
    },
    {
      "role": "assistant",
      "content": "Exactly! Let's say we're searching for the number 7 in [1, 3, 5, 7, 9, 11, 13]. First, we look at the middle element: 7. Lucky us - we found it immediately!"
    },
    {
      "role": "user",
      "content": "That was too easy! What if we were looking for 11?"
    },
    {
      "role": "assistant",
      "content": "Good example! Looking for 11: We start with the middle (7). Since 11 > 7, we know 11 must be in the right half [9, 11, 13]. Now we check the middle of this subset: 11. Found it in just 2 steps instead of checking each element!"
    }
  ],
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Student wants to understand binary search efficiency. I should explain the core concept: eliminating half the search space each time.",
      "action": "explain_concept"
    },
    {
      "step_number": 2,
      "thought": "They're asking about the sorted array prerequisite. This is crucial for binary search to work.",
      "action": "confirm_prerequisite"
    },
    {
      "step_number": 3,
      "thought": "I'll use a concrete example with array [1,3,5,7,9,11,13] searching for 7. Starting with the obvious case.",
      "action": "demonstrate_simple"
    },
    {
      "step_number": 4,
      "thought": "They want a more interesting example. Searching for 11 will show the elimination process more clearly.",
      "action": "demonstrate_complex"
    }
  ],
  "final_answer": "Binary search works by repeatedly dividing the search space in half, comparing the target with the middle element, and eliminating the half that can't contain the target. This gives O(log n) time complexity instead of O(n) for linear search."
}
```

### Science - Physics Problem Solving

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a physics tutor who helps students work through problems systematically."
    },
    {
      "role": "user",
      "content": "A car accelerates from rest to 60 mph in 8 seconds. What's its acceleration?"
    },
    {
      "role": "assistant",
      "content": "Good problem! Before we calculate, let's make sure we have consistent units. What units should we use for acceleration?"
    },
    {
      "role": "user",
      "content": "Meters per second squared?"
    },
    {
      "role": "assistant",
      "content": "Perfect! So we need to convert 60 mph to m/s first. Do you know the conversion?"
    },
    {
      "role": "user",
      "content": "Not off the top of my head..."
    },
    {
      "role": "assistant",
      "content": "No problem! 60 mph = 60 × (1609.34 m/mile) × (1 hour/3600 s) ≈ 26.8 m/s. Now we can use a = (v_f - v_i)/t. Since the car starts from rest, v_i = 0."
    },
    {
      "role": "user",
      "content": "So a = 26.8/8 = 3.35 m/s²?"
    },
    {
      "role": "assistant",
      "content": "Exactly right! The acceleration is 3.35 m/s². This means the car's velocity increases by 3.35 m/s every second during the acceleration phase."
    }
  ],
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Student gave a physics problem with mixed units (mph and seconds). I need to address unit consistency first.",
      "action": "identify_units"
    },
    {
      "step_number": 2,
      "thought": "They correctly identified m/s² as the standard unit for acceleration. Now I need to guide them through unit conversion.",
      "action": "guide_conversion"
    },
    {
      "step_number": 3,
      "thought": "They don't know the mph to m/s conversion. I'll provide it and set up the acceleration formula.",
      "action": "provide_conversion"
    },
    {
      "step_number": 4,
      "thought": "They correctly applied a = (v_f - v_i)/t with the converted values. I'll confirm and add physical interpretation.",
      "action": "verify_and_interpret"
    }
  ],
  "final_answer": "The car's acceleration is 3.35 m/s²"
}
```

## Configuration for Structured CoT

### YAML Configuration

```yaml
# structured-cot.yaml
dataset_system_prompt: "You are a helpful instructor who guides students through systematic problem-solving."

topic_tree:
  topic_prompt: "Educational topics in mathematics, science, and computer science"
  provider: "openai"
  model: "gpt-4o-mini"
  degree: 2
  depth: 2
  temperature: 0.6

data_engine:
  instructions: "Create educational conversations showing systematic problem-solving with clear reasoning steps."
  generation_system_prompt: "You are an educator creating realistic teaching dialogues with explicit reasoning."

  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.4

  # Structured CoT specific settings
  conversation_type: "cot_structured"
  reasoning_style: "logical"  # or "mathematical" or "general"

dataset:
  creation:
    num_steps: 8
    batch_size: 1
    sys_msg: true  # Include system messages in conversations
  save_as: "structured_cot_dataset.jsonl"
```

### Python API

```python
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree

# Create topic tree for educational content
tree = Tree(
    topic_prompt="Educational problem-solving in STEM subjects",
    provider="openai",
    model_name="gpt-4o-mini",
    degree=2,
    depth=2,
    temperature=0.6
)

# Build tree
async for event in tree.build_async():
    if event['event'] == 'build_complete':
        print(f"Built {event['total_paths']} educational topics")

# Create structured CoT generator
generator = DataSetGenerator(
    instructions="Create educational dialogues with systematic reasoning.",
    generation_system_prompt="You are a tutor creating step-by-step learning conversations.",
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.4,
    conversation_type="cot_structured",
    reasoning_style="logical"
)

# Generate dataset with system messages
dataset = generator.create_data(
    num_steps=8,
    batch_size=1,
    topic_model=tree,
    sys_msg=True  # Important for structured conversations
)

# Save and report
dataset.save("structured_cot_education.jsonl")
print(f"Generated {len(dataset.samples)} educational conversations")

# Show sample structure
if dataset.samples:
    sample = dataset.samples[0]
    print(f"Messages: {len(sample['messages'])}")
    print(f"Reasoning steps: {len(sample['reasoning_trace'])}")
```

## Action Classifications

The `action` field in reasoning steps helps categorize the type of thinking being done:

### Educational Actions
- `assess_problem`: Understanding the student's issue
- `clarify_objective`: Explaining the goal or target
- `guide_step`: Leading the student through a step
- `demonstrate`: Showing a calculation or example
- `verify_solution`: Checking the answer

### Analytical Actions
- `analyze`: Breaking down the problem
- `classify`: Categorizing the problem type
- `calculate`: Performing mathematical operations
- `compare`: Contrasting different approaches
- `synthesize`: Combining information

### Interactive Actions
- `question`: Asking the student something
- `confirm`: Verifying student understanding
- `correct`: Addressing misconceptions
- `encourage`: Providing positive reinforcement
- `summarize`: Recapping key points

## Best Practices

### Conversation Quality

**✅ Good Conversation Flow:**
- Natural dialogue between user and assistant
- Assistant asks guiding questions
- Student responses show engagement
- Progressive difficulty building
- Clear explanations with examples

**❌ Poor Conversation Flow:**
- One-sided information dump
- No student interaction
- Unclear or confusing explanations
- Too advanced too quickly
- Missing verification steps

### Reasoning Trace Quality

**✅ Good Reasoning Trace:**
- Each step corresponds to conversation elements
- Clear thought progression
- Appropriate action classifications
- Strategic teaching decisions explained
- Shows both content and pedagogical reasoning

**❌ Poor Reasoning Trace:**
- Steps don't match conversation
- Vague or generic thoughts
- Incorrect action classifications
- Missing key reasoning elements
- No educational strategy visible

### Alignment Best Practices

1. **Message-Step Correspondence**: Each assistant message should have corresponding reasoning steps
2. **Logical Progression**: Step numbers should follow logical sequence
3. **Action Accuracy**: Use appropriate action classifications
4. **Educational Focus**: Reasoning should show teaching strategy, not just content

## Common Issues and Solutions

### Issue: Misaligned Conversations and Reasoning
**Problem**: Reasoning trace doesn't match conversation flow
**Solution**: Ensure each assistant response has corresponding reasoning steps explaining the pedagogical choices

### Issue: Poor Action Classification
**Problem**: Generic or incorrect action labels
**Solution**: Use specific, meaningful action verbs that describe the educational strategy

### Issue: Shallow Reasoning
**Problem**: Reasoning steps lack depth or insight
**Solution**: Focus on *why* the assistant made specific teaching choices, not just what they said

### Issue: Unnatural Conversations
**Problem**: Dialogue feels scripted or artificial
**Solution**: Include realistic student responses, mistakes, and follow-up questions

## Quality Validation

### Automated Checks
- **Message count**: Should have multiple turns (typically 4-12 messages)
- **Reasoning alignment**: Number of reasoning steps should correlate with assistant messages
- **Role validation**: Proper role assignments in messages
- **Step numbering**: Sequential, starting from 1

### Human Evaluation Criteria
- **Educational effectiveness**: Would this conversation help a student learn?
- **Reasoning transparency**: Do the reasoning steps reveal the teaching strategy?
- **Natural flow**: Does the conversation feel realistic?
- **Content accuracy**: Are the subject matter facts correct?

## Performance Characteristics

### Token Usage
- Highest token usage of the three CoT formats
- Average sample size: 800-1500 tokens
- Recommended batch size: 1 sample per batch

### Generation Complexity
- Most complex format requiring dual skill sets
- Benefits from higher-capability models (GPT-4, Claude-3)
- Longer generation times due to complexity

### Provider Compatibility
- Works well with OpenAI (GPT-4, GPT-4o)
- Compatible with Anthropic Claude models
- May struggle with smaller local models

## Next Steps

- **Explore Hybrid CoT**: For complex multi-modal reasoning → [Hybrid CoT Guide](hybrid.md)
- **Compare with Free-text**: For simpler reasoning → [Free-text CoT Guide](free-text.md)
- **Math Reasoning Tutorial**: → [Math Reasoning Tutorial](../tutorials/math-reasoning.md)
- **Advanced Configuration**: → [Reasoning Styles Guide](../advanced/reasoning-styles.md)