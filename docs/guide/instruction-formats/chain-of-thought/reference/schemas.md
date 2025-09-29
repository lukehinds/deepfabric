# Chain of Thought Schema Reference

This reference documents the complete Pydantic schemas used for Chain of Thought formats in DeepFabric. All schemas leverage Outlines for structured generation, ensuring model outputs strictly conform to these specifications.

## Core Schema Classes

### Base Message Schema

```python
class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: str = Field(description="The role of the message sender")
    content: str = Field(description="The content of the message")
```

**Field Details**:
- `role`: Must be one of `"user"`, `"assistant"`, `"system"`, or `"tool"`
- `content`: The actual message text content

**Validation Rules**:
- Both fields are required
- `role` must be from allowed set
- `content` must be non-empty string

### Reasoning Step Schema

```python
class ReasoningStep(BaseModel):
    """A single step in a chain of reasoning."""

    step_number: int = Field(description="The step number in the reasoning chain")
    thought: str = Field(description="The reasoning or thought for this step")
    action: str = Field(description="Any action taken as part of this reasoning step")
```

**Field Details**:
- `step_number`: Sequential integer starting from 1
- `thought`: The actual reasoning content for this step
- `action`: Classification of the reasoning action (see [Action Classifications](#action-classifications))

**Validation Rules**:
- `step_number` must be positive integer
- `thought` must be non-empty string
- `action` must be non-empty string (changed from optional for OpenAI compatibility)

## Chain of Thought Format Schemas

### Free-text CoT Schema

```python
class FreeTextCoT(BaseModel):
    """Chain of Thought dataset with natural language reasoning."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    final_answer: str = Field(description="The definitive answer to the question")
```

**Use Case**: Mathematical word problems, logic puzzles, general Q&A

**Example JSON**:
```json
{
  "question": "Sarah has 24 stickers. She gives 8 to her friend and buys 15 more. How many stickers does she have now?",
  "chain_of_thought": "Sarah starts with 24 stickers. She gives away 8, so she has 24 - 8 = 16 stickers left. Then she buys 15 more, so her total is 16 + 15 = 31 stickers.",
  "final_answer": "31 stickers"
}
```

**Validation Rules**:
- All fields required and non-empty
- `question` should be a clear problem statement
- `chain_of_thought` should show reasoning process
- `final_answer` should be a definitive conclusion

### Structured CoT Schema

```python
class StructuredCoT(BaseModel):
    """Chain of Thought dataset with structured reasoning trace."""

    messages: list[ChatMessage] = Field(description="Conversation messages", min_length=1)
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")
```

**Use Case**: Educational dialogues, tutoring scenarios, conversational learning

**Example JSON**:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful math tutor."},
    {"role": "user", "content": "How do I solve 2x + 5 = 13?"},
    {"role": "assistant", "content": "Let's solve this step by step. What do you think we should do first?"},
    {"role": "user", "content": "Subtract 5 from both sides?"},
    {"role": "assistant", "content": "Exactly! So we get 2x = 8. Now what?"},
    {"role": "user", "content": "Divide by 2?"},
    {"role": "assistant", "content": "Perfect! So x = 4. Let's verify: 2(4) + 5 = 13 ✓"}
  ],
  "reasoning_trace": [
    {"step_number": 1, "thought": "Student needs guidance on solving linear equations", "action": "assess_problem"},
    {"step_number": 2, "thought": "Guide them to isolate the variable term first", "action": "guide_step"},
    {"step_number": 3, "thought": "Confirm their correct approach and continue", "action": "confirm"},
    {"step_number": 4, "thought": "Guide them to the final step", "action": "guide_step"},
    {"step_number": 5, "thought": "Verify the solution to reinforce good practices", "action": "verify_solution"}
  ],
  "final_answer": "x = 4"
}
```

**Validation Rules**:
- `messages` must have at least one message
- Each message must have valid `role` and `content`
- `reasoning_trace` must have at least one step
- Steps should be sequentially numbered starting from 1
- `final_answer` must be provided

### Hybrid CoT Schema

```python
class HybridCoT(BaseModel):
    """Chain of Thought dataset with both free-text and structured reasoning."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")
```

**Use Case**: Complex problems requiring both intuitive and systematic reasoning

**Example JSON**:
```json
{
  "question": "Explain how bubble sort works and analyze its time complexity.",
  "chain_of_thought": "Bubble sort works by repeatedly stepping through the list, comparing adjacent elements and swapping them if they're in the wrong order. The pass through the list is repeated until the list is sorted. The name comes from the way smaller elements 'bubble' to the top of the list. In terms of time complexity, we need to consider that in the worst case, we need to make n-1 passes through the array, and in each pass, we compare up to n-1 pairs of adjacent elements. This gives us roughly n² comparisons, making the time complexity O(n²).",
  "reasoning_trace": [
    {"step_number": 1, "thought": "Explain the basic mechanism of bubble sort", "action": "explain_algorithm"},
    {"step_number": 2, "thought": "Describe the comparison and swapping process", "action": "detail_process"},
    {"step_number": 3, "thought": "Explain why it's called 'bubble' sort", "action": "provide_intuition"},
    {"step_number": 4, "thought": "Analyze worst-case scenario for time complexity", "action": "analyze_complexity"},
    {"step_number": 5, "thought": "Calculate the mathematical relationship", "action": "calculate"},
    {"step_number": 6, "thought": "State the final time complexity result", "action": "conclude"}
  ],
  "final_answer": "Bubble sort repeatedly compares and swaps adjacent elements until the list is sorted. Time complexity: O(n²) due to nested iteration through the array."
}
```

**Validation Rules**:
- All fields required and non-empty
- `chain_of_thought` provides intuitive explanation
- `reasoning_trace` provides systematic breakdown
- Both reasoning modes should be consistent and complementary
- `final_answer` should synthesize both approaches

## Schema Validation Details

### Automatic Validation with Outlines

DeepFabric uses Outlines to ensure generated content strictly conforms to schemas:

```python
# During generation, this happens automatically:
conversation = self.llm_client.generate(
    prompt=prompt,
    schema=FreeTextCoT,  # Pydantic schema enforces structure
    max_retries=self.config.max_retries,
    temperature=self.config.temperature,
)

# Result is guaranteed to be valid FreeTextCoT instance
assert isinstance(conversation, FreeTextCoT)
sample = conversation.model_dump()  # Convert to dict for dataset
```

### Manual Validation

For samples loaded from files or other sources:

```python
from deepfabric.schemas import FreeTextCoT, StructuredCoT, HybridCoT
from pydantic import ValidationError

def validate_cot_sample(sample: dict, format_type: str) -> bool:
    """Validate a sample against the appropriate CoT schema."""

    schema_map = {
        "cot_freetext": FreeTextCoT,
        "cot_structured": StructuredCoT,
        "cot_hybrid": HybridCoT
    }

    schema_class = schema_map.get(format_type)
    if not schema_class:
        return False

    try:
        schema_class.model_validate(sample)
        return True
    except ValidationError as e:
        print(f"Validation error: {e}")
        return False

# Usage
sample = {"question": "...", "chain_of_thought": "...", "final_answer": "..."}
is_valid = validate_cot_sample(sample, "cot_freetext")
```

### Dataset-Level Validation

The `Dataset` class provides simplified validation that checks for required fields:

```python
from deepfabric.dataset import Dataset

# Simplified validation (used internally)
def validate_sample(sample: dict) -> bool:
    """Check for presence of required fields for any CoT format."""

    # Check for different format patterns
    formats = [
        ["question", "chain_of_thought", "final_answer"],  # Free-text
        ["messages", "reasoning_trace", "final_answer"],   # Structured
        ["question", "chain_of_thought", "reasoning_trace", "final_answer"],  # Hybrid
        ["messages"]  # Basic conversation
    ]

    return any(all(key in sample for key in format_keys) for format_keys in formats)
```

## Action Classifications

The `action` field in `ReasoningStep` uses these common classifications:

### Educational Actions
| Action | Description | Use Case |
|--------|-------------|----------|
| `assess_problem` | Understanding the problem or student's issue | Beginning of tutoring |
| `clarify_objective` | Explaining the goal or target | Setting direction |
| `guide_step` | Leading through a specific step | Step-by-step instruction |
| `demonstrate` | Showing a calculation or example | Concrete examples |
| `verify_solution` | Checking the answer | Ensuring correctness |

### Analytical Actions
| Action | Description | Use Case |
|--------|-------------|----------|
| `analyze` | Breaking down the problem | Problem decomposition |
| `classify` | Categorizing the problem type | Pattern recognition |
| `calculate` | Performing mathematical operations | Numerical work |
| `compare` | Contrasting different approaches | Method evaluation |
| `synthesize` | Combining information | Integration |

### Logical Actions
| Action | Description | Use Case |
|--------|-------------|----------|
| `identify_premise` | Stating given conditions | Formal reasoning |
| `apply_rule` | Using logical principles | Rule-based reasoning |
| `derive_conclusion` | Reaching logical result | Deductive reasoning |
| `check_consistency` | Verifying logical validity | Quality assurance |

### Domain-Specific Actions
| Action | Description | Use Case |
|--------|-------------|----------|
| `explain_algorithm` | Describing how an algorithm works | CS education |
| `analyze_complexity` | Examining computational complexity | Algorithm analysis |
| `prove_correctness` | Demonstrating algorithm correctness | Formal verification |
| `optimize_solution` | Improving efficiency | Performance tuning |

## Schema Evolution and Compatibility

### Version Compatibility

DeepFabric schemas follow semantic versioning principles:

- **Patch versions** (1.0.1): Bug fixes, no schema changes
- **Minor versions** (1.1.0): Backward-compatible additions
- **Major versions** (2.0.0): Breaking schema changes

### Custom Schema Extensions

For domain-specific needs, you can extend the base schemas:

```python
from deepfabric.schemas import FreeTextCoT

class MathCoT(FreeTextCoT):
    """Extended CoT schema for mathematics with additional metadata."""

    difficulty_level: int = Field(ge=1, le=10, description="Problem difficulty (1-10)")
    topic_area: str = Field(description="Mathematical topic (e.g., 'algebra', 'geometry')")
    grade_level: str = Field(description="Target grade level")

    # Additional validation
    @validator('chain_of_thought')
    def must_contain_calculation(cls, v):
        if not any(char in v for char in '=+-×÷'):
            raise ValueError('Mathematical reasoning must contain calculations')
        return v

# Usage with custom schema
generator = DataSetGenerator(
    conversation_type="cot_freetext",
    reasoning_style="mathematical",
    # Note: Custom schemas require additional integration work
)
```

## JSON Schema Export

For integration with other tools, you can export JSON schemas:

```python
from deepfabric.schemas import FreeTextCoT, StructuredCoT, HybridCoT

# Export JSON schemas
schemas = {
    "freetext": FreeTextCoT.model_json_schema(),
    "structured": StructuredCoT.model_json_schema(),
    "hybrid": HybridCoT.model_json_schema()
}

# Save to file
import json
with open("cot_schemas.json", "w") as f:
    json.dump(schemas, f, indent=2)

# Example output structure
"""
{
  "freetext": {
    "type": "object",
    "properties": {
      "question": {"type": "string", "description": "..."},
      "chain_of_thought": {"type": "string", "description": "..."},
      "final_answer": {"type": "string", "description": "..."}
    },
    "required": ["question", "chain_of_thought", "final_answer"]
  }
}
"""
```

## Common Schema Issues and Solutions

### Issue: Missing Required Fields

```python
# Validation error example
{
  "question": "What is 2+2?",
  "chain_of_thought": "2 plus 2 equals 4",
  # Missing "final_answer" field
}

# Error: ValidationError: field required (type=value_error.missing)
```

**Solution**: Ensure all required fields are present and non-empty.

### Issue: Incorrect Field Types

```python
# Validation error example
{
  "question": "What is 2+2?",
  "chain_of_thought": 123,  # Should be string, not integer
  "final_answer": "4"
}
```

**Solution**: Check field types match schema definitions.

### Issue: Empty Reasoning Trace

```python
# Validation error example
{
  "messages": [...],
  "reasoning_trace": [],  # Empty array not allowed (min_length=1)
  "final_answer": "Answer"
}
```

**Solution**: Ensure reasoning_trace has at least one step.

### Issue: Sequential Step Numbers

```python
# Potential issue
{
  "reasoning_trace": [
    {"step_number": 1, "thought": "...", "action": "..."},
    {"step_number": 3, "thought": "...", "action": "..."},  # Skipped 2
    {"step_number": 2, "thought": "...", "action": "..."}   # Out of order
  ]
}
```

**Solution**: While not enforced by schema, ensure step numbers are sequential for clarity.

## Best Practices

### Schema Design Principles
1. **Required fields only**: Make fields optional only when truly optional
2. **Clear descriptions**: Field descriptions guide model generation
3. **Appropriate constraints**: Use min_length, validators for quality
4. **Consistent naming**: Follow established conventions

### Generation Optimization
1. **Simple schemas first**: Start with free-text, progress to complex
2. **Provider compatibility**: Test schemas with your chosen LLM provider
3. **Validation feedback**: Use validation errors to improve prompts

### Quality Assurance
1. **Automated validation**: Always validate generated samples
2. **Manual spot checks**: Review samples for logical consistency
3. **Schema evolution**: Plan for future schema enhancements
