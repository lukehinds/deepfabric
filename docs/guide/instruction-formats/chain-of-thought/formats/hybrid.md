# Hybrid Chain of Thought Format

The hybrid Chain of Thought format combines the natural expressiveness of free-text reasoning with the structured precision of step-by-step traces. This format is ideal
for complex problems that benefit from both intuitive explanation and systematic decomposition. It is also an effective way of reducing over-fit risks that lots of
structured reasoning text can provide.

## When to Use Hybrid CoT

### Ideal Use Cases
- **Algorithm analysis**: Explaining code with both intuition and formal steps
- **Mathematical proofs**: Natural explanation plus rigorous logical structure
- **Scientific reasoning**: Hypothesis formation with systematic testing
- **Complex problem-solving**: Multi-faceted problems requiring different reasoning types
- **Research methodology**: Combining intuitive insights with methodical analysis

### Strengths
- **Dual reasoning modes**: Both natural and structured approaches
- **Maximum expressiveness**: Can handle any type of reasoning pattern
- **Rich metadata**: Step-by-step breakdown with action classification
- **Flexible structure**: Adapts to problem complexity
- **Comprehensive coverage**: Captures both high-level strategy and detailed steps

### Limitations
- **Highest complexity**: Requires coordination of multiple reasoning types
- **Token intensive**: Largest format due to dual representation
- **Generation challenge**: Most difficult for models to produce consistently
- **Validation complexity**: Multiple layers to verify for quality

## Schema Specification

```python
class HybridCoT(BaseModel):
    """Chain of Thought dataset with both free-text and structured reasoning."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")

class ReasoningStep(BaseModel):
    """A single step in a chain of reasoning."""
    step_number: int = Field(description="The step number in the reasoning chain")
    thought: str = Field(description="The reasoning or thought for this step")
    action: str = Field(description="Any action taken as part of this reasoning step")
```

### Field Descriptions

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | ✅ | The problem statement or question to be solved |
| `chain_of_thought` | string | ✅ | Natural language reasoning explanation |
| `reasoning_trace` | array | ✅ | Structured breakdown of reasoning steps |
| `final_answer` | string | ✅ | The definitive answer or conclusion |

#### Reasoning Step Fields
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `step_number` | integer | ✅ | Sequential step number (1, 2, 3...) |
| `thought` | string | ✅ | The specific reasoning or analysis for this step |
| `action` | string | ✅ | Classification of the type of reasoning action |

## Examples by Domain

### Algorithm Analysis - Quicksort

```json
{
  "question": "Explain how the quicksort algorithm works and analyze its time complexity.",
  "chain_of_thought": "Quicksort is a divide-and-conquer algorithm that sorts arrays efficiently by partitioning. The key insight is choosing a 'pivot' element and rearranging the array so all elements smaller than the pivot come before it, and all larger elements come after it. We then recursively apply the same process to the sub-arrays on either side of the pivot. The beauty of quicksort lies in its average-case performance of O(n log n), achieved because each partitioning step roughly halves the problem size. However, the worst-case scenario occurs when we consistently pick the smallest or largest element as the pivot, leading to O(n²) performance. In practice, random pivot selection or median-of-three pivot selection helps avoid worst-case behavior.",
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Identify quicksort as a divide-and-conquer algorithm and explain the core concept of partitioning around a pivot.",
      "action": "classify_algorithm"
    },
    {
      "step_number": 2,
      "thought": "Describe the partitioning process: elements smaller than pivot go left, larger go right.",
      "action": "explain_mechanism"
    },
    {
      "step_number": 3,
      "thought": "Explain the recursive nature: apply quicksort to both sub-arrays created by partitioning.",
      "action": "describe_recursion"
    },
    {
      "step_number": 4,
      "thought": "Analyze average-case time complexity: each level of recursion processes n elements, and there are log n levels on average.",
      "action": "analyze_average_case"
    },
    {
      "step_number": 5,
      "thought": "Analyze worst-case scenario: when pivot is always the minimum or maximum, leading to n levels of recursion.",
      "action": "analyze_worst_case"
    },
    {
      "step_number": 6,
      "thought": "Mention practical optimizations like random pivot selection to avoid worst-case behavior.",
      "action": "suggest_optimizations"
    }
  ],
  "final_answer": "Quicksort works by partitioning arrays around a pivot element and recursively sorting the sub-arrays. Time complexity: Average case O(n log n), Worst case O(n²). The algorithm's efficiency comes from the divide-and-conquer approach that typically halves the problem size at each step."
}
```

### Mathematical Proof - Triangle Inequality

```json
{
  "question": "Prove the triangle inequality: for any triangle with sides a, b, and c, the sum of any two sides must be greater than the third side.",
  "chain_of_thought": "The triangle inequality is fundamental to understanding geometric relationships. Intuitively, it makes sense: if you're trying to get from point A to point C, the direct path (one side of the triangle) must be shorter than any indirect path that goes through point B (the sum of the other two sides). To prove this rigorously, we can use the fact that the shortest distance between two points is a straight line. Consider triangle ABC with sides a (opposite to A), b (opposite to B), and c (opposite to C). We need to prove that a + b > c, b + c > a, and a + c > b. The key insight is that if any of these inequalities were violated, we couldn't form a valid triangle - the three points would be collinear or one point would be unreachable.",
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Establish the geometric intuition: the direct path between two points is shorter than any indirect path.",
      "action": "build_intuition"
    },
    {
      "step_number": 2,
      "thought": "Define the triangle ABC with sides a, b, c opposite to vertices A, B, C respectively.",
      "action": "define_notation"
    },
    {
      "step_number": 3,
      "thought": "State what needs to be proven: three inequalities a+b>c, b+c>a, and a+c>b must all hold.",
      "action": "state_theorem"
    },
    {
      "step_number": 4,
      "thought": "Consider the contrapositive: if a+b ≤ c, then points A, B, C cannot form a proper triangle.",
      "action": "proof_by_contradiction"
    },
    {
      "step_number": 5,
      "thought": "Apply the principle that the straight-line distance between two points is minimal.",
      "action": "apply_geometric_principle"
    },
    {
      "step_number": 6,
      "thought": "Conclude that all three inequalities must hold for a valid triangle to exist.",
      "action": "complete_proof"
    }
  ],
  "final_answer": "The triangle inequality holds because the direct distance between any two vertices of a triangle must be less than the path that goes through the third vertex. Mathematically: a + b > c, b + c > a, and a + c > b for any valid triangle."
}
```

### Scientific Reasoning - Chemical Equilibrium

```json
{
  "question": "Explain how Le Châtelier's principle predicts the effect of adding more reactant to a chemical equilibrium.",
  "chain_of_thought": "Le Châtelier's principle states that when a system at equilibrium is subjected to a stress, the system will shift in a direction that relieves that stress. When we add more reactant to an equilibrium system, we're increasing the concentration of one of the species on the left side of the equilibrium equation. The system responds by consuming some of this excess reactant, which means the forward reaction rate increases. This continues until a new equilibrium is established with higher concentrations of products and a somewhat higher concentration of reactants than the original equilibrium, but lower than immediately after the addition. The key insight is that the system doesn't return to the original state - it finds a new equilibrium position that partially counteracts the disturbance.",
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "State Le Châtelier's principle as the governing concept for predicting equilibrium shifts.",
      "action": "cite_principle"
    },
    {
      "step_number": 2,
      "thought": "Identify the stress: adding more reactant increases the concentration of species on the left side.",
      "action": "identify_disturbance"
    },
    {
      "step_number": 3,
      "thought": "Predict the system's response: forward reaction rate increases to consume excess reactant.",
      "action": "predict_response"
    },
    {
      "step_number": 4,
      "thought": "Explain the kinetics: higher reactant concentration leads to more frequent productive collisions.",
      "action": "explain_mechanism"
    },
    {
      "step_number": 5,
      "thought": "Describe the new equilibrium state: higher product concentrations, reactant concentration between original and post-addition levels.",
      "action": "describe_outcome"
    },
    {
      "step_number": 6,
      "thought": "Emphasize that the system reaches a new equilibrium, not a return to the original state.",
      "action": "clarify_concept"
    }
  ],
  "final_answer": "According to Le Châtelier's principle, adding more reactant shifts the equilibrium toward products. The system responds by increasing the forward reaction rate until a new equilibrium is established with higher product concentrations and partially restored reactant levels."
}
```

### Computer Science - Database Optimization

```json
{
  "question": "Explain how database indexing improves query performance and when it might hurt performance.",
  "chain_of_thought": "Database indexing is like creating a specialized lookup table that points to the actual data locations, similar to an index in a book. When you create an index on a column, the database builds a sorted data structure (often a B-tree) that allows for much faster searching. Instead of scanning every row (O(n) complexity), the database can use the index to jump directly to relevant rows (O(log n) complexity). However, indexes aren't free - they require additional storage space and must be maintained whenever the underlying data changes. Every INSERT, UPDATE, or DELETE operation must also update all relevant indexes, which can slow down write operations. The key is finding the right balance: index frequently queried columns, but avoid over-indexing tables that have heavy write workloads.",
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Explain the concept of indexing using the book index analogy for intuitive understanding.",
      "action": "build_analogy"
    },
    {
      "step_number": 2,
      "thought": "Describe the technical implementation: indexes are typically B-tree data structures that maintain sorted order.",
      "action": "explain_implementation"
    },
    {
      "step_number": 3,
      "thought": "Quantify the performance benefit: reduce search complexity from O(n) to O(log n).",
      "action": "analyze_complexity"
    },
    {
      "step_number": 4,
      "thought": "Identify the costs: additional storage space required for index structures.",
      "action": "identify_storage_cost"
    },
    {
      "step_number": 5,
      "thought": "Explain maintenance overhead: indexes must be updated on every data modification operation.",
      "action": "explain_maintenance_cost"
    },
    {
      "step_number": 6,
      "thought": "Provide guidance on when to use indexes: balance read performance gains against write performance costs.",
      "action": "provide_guidelines"
    }
  ],
  "final_answer": "Database indexes improve query performance by providing O(log n) lookup time instead of O(n) table scans. However, they consume additional storage and slow down write operations due to maintenance overhead. Use indexes for frequently queried columns, but avoid over-indexing write-heavy tables."
}
```

## Configuration for Hybrid CoT

### YAML Configuration

```yaml
# hybrid-cot.yaml
dataset_system_prompt: "You are an expert educator who explains complex topics with both intuitive insights and systematic analysis."

topic_tree:
  topic_prompt: "Complex problems in computer science, mathematics, and science requiring multi-faceted reasoning"
  provider: "openai"
  model: "gpt-4o"  # Recommend higher-capability model for hybrid reasoning
  degree: 2
  depth: 2
  temperature: 0.5

data_engine:
  instructions: "Create challenging problems that require both intuitive understanding and systematic analysis."
  generation_system_prompt: "You are an expert who combines natural explanation with rigorous step-by-step reasoning."

  provider: "openai"
  model: "gpt-4o"  # Higher capability needed for dual reasoning modes
  temperature: 0.3

  # Hybrid CoT specific settings
  conversation_type: "cot_hybrid"
  reasoning_style: "logical"  # Can be "mathematical", "logical", or "general"

dataset:
  creation:
    num_steps: 6  # Fewer steps due to complexity
    batch_size: 1  # Always use batch_size=1 for hybrid format
    sys_msg: false  # Hybrid CoT doesn't typically use system messages
  save_as: "hybrid_cot_dataset.jsonl"
```

### Python API

```python
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree

# Create topic tree for complex reasoning problems
tree = Tree(
    topic_prompt="Advanced problems requiring both intuitive and systematic reasoning",
    provider="openai",
    model_name="gpt-4o",  # Use higher-capability model
    degree=2,
    depth=2,
    temperature=0.5
)

# Build tree
for event in tree.build():
    if event['event'] == 'build_complete':
        print(f"Built {event['total_paths']} complex topic paths")

# Create hybrid CoT generator
generator = DataSetGenerator(
    instructions="Create complex problems requiring multi-faceted reasoning.",
    generation_system_prompt="You are an expert who combines intuitive insights with systematic analysis.",
    provider="openai",
    model_name="gpt-4o",
    temperature=0.3,
    conversation_type="cot_hybrid",
    reasoning_style="logical"
)

# Generate dataset (smaller batches due to complexity)
dataset = generator.create_data(
    num_steps=6,
    batch_size=1,
    topic_model=tree,
    sys_msg=False
)

# Save and analyze
dataset.save("hybrid_reasoning_dataset.jsonl")
print(f"Generated {len(dataset.samples)} hybrid reasoning examples")

# Analyze sample complexity
if dataset.samples:
    sample = dataset.samples[0]
    print(f"Chain of thought length: {len(sample['chain_of_thought'])} characters")
    print(f"Reasoning steps: {len(sample['reasoning_trace'])}")
    print(f"Average step length: {sum(len(step['thought']) for step in sample['reasoning_trace']) / len(sample['reasoning_trace']):.0f} chars")
```

## Action Classifications for Hybrid CoT

Hybrid CoT reasoning steps use more sophisticated action classifications:

### Analytical Actions
- `classify_algorithm`: Categorizing computational approaches
- `analyze_complexity`: Examining algorithmic or mathematical complexity
- `decompose_problem`: Breaking down into sub-problems
- `synthesize_solution`: Combining multiple approaches

### Explanatory Actions
- `build_intuition`: Developing conceptual understanding
- `build_analogy`: Using metaphors or comparisons
- `explain_mechanism`: Describing how something works
- `provide_context`: Giving background information

### Logical Actions
- `state_theorem`: Presenting formal statements
- `proof_by_contradiction`: Using indirect proof methods
- `apply_principle`: Using established rules or laws
- `verify_logic`: Checking reasoning validity

### Strategic Actions
- `identify_constraints`: Recognizing limitations or requirements
- `suggest_optimizations`: Proposing improvements
- `evaluate_tradeoffs`: Analyzing competing factors
- `provide_guidelines`: Offering decision criteria

## Best Practices

### Balancing Free-text and Structured Elements

**✅ Good Balance:**
- Chain of thought provides intuitive narrative
- Reasoning trace breaks down systematic steps
- Both elements complement, don't duplicate
- Natural flow between explanation types

**❌ Poor Balance:**
- Chain of thought just repeats reasoning steps
- Reasoning trace adds no structured value
- Inconsistent information between elements
- Artificial separation of reasoning modes

### Quality Guidelines

**Excellent Hybrid CoT Example:**
- Rich, intuitive explanation in chain_of_thought
- Systematic breakdown in reasoning_trace
- Appropriate action classifications
- Seamless integration of reasoning modes
- Addresses both "what" and "why"

**Poor Hybrid CoT Example:**
- Redundant information between sections
- Shallow reasoning in either section
- Generic or incorrect action classifications
- Disconnected explanation modes
- Missing key insights or steps

## Advanced Patterns

### Multi-Domain Reasoning
When problems span multiple domains (e.g., computational biology), use:
- Domain-specific action classifications
- Cross-domain connection explanations
- Appropriate reasoning style for each domain

### Proof and Algorithm Combination
For problems involving both mathematical proofs and algorithmic analysis:
- Mathematical reasoning for correctness proofs
- Computational analysis for efficiency
- Clear separation of concerns in reasoning trace

### Research Methodology
For complex research problems:
- Hypothesis formation in chain_of_thought
- Systematic testing in reasoning_trace
- Literature integration and synthesis

## Performance Considerations

### Model Requirements
- **Recommended**: GPT-4, GPT-4o, Claude-3 Opus
- **Minimum**: GPT-3.5-turbo may struggle with consistency
- **Local models**: Generally not recommended for hybrid format

### Token Usage
- Highest token usage: 1000-2500 tokens per sample
- Most expensive format to generate
- Longest generation times

### Quality vs. Quantity Tradeoffs
- Generate fewer, higher-quality samples
- Focus on complex, valuable problems
- Use for specialized domains requiring deep reasoning

## Validation and Quality Control

### Automated Checks
- **Length validation**: Both chain_of_thought and reasoning_trace should be substantial
- **Consistency check**: Information alignment between reasoning modes
- **Step completeness**: Reasoning trace should cover all major points from chain_of_thought
- **Action appropriateness**: Verify action classifications make sense

### Human Evaluation
- **Reasoning quality**: Are both explanation modes effective?
- **Integration**: Do the reasoning modes work together well?
- **Accuracy**: Is the domain knowledge correct?
- **Completeness**: Are all aspects of the problem addressed?

## Troubleshooting Common Issues

### Issue: Redundant Reasoning
**Problem**: Chain of thought and reasoning trace repeat the same information
**Solution**: Guide models to use chain_of_thought for intuition and reasoning_trace for systematic breakdown

### Issue: Inconsistent Information
**Problem**: Conflicting details between reasoning modes
**Solution**: Emphasize consistency in prompts, use verification steps

### Issue: Poor Action Classification
**Problem**: Generic or inappropriate action labels
**Solution**: Provide clear action taxonomy in prompts, use domain-specific actions

### Issue: Unbalanced Complexity
**Problem**: One reasoning mode much simpler than the other
**Solution**: Ensure prompts emphasize the value of both reasoning approaches

## Next Steps

- **Master Simpler Formats**: Start with [Free-text CoT](free-text.md) or [Structured CoT](structured.md)
- **Explore Reasoning Styles**: → [Reasoning Styles Guide](../advanced/reasoning-styles.md)
- **Math Reasoning Tutorial**: → [Math Reasoning Tutorial](../tutorials/math-reasoning.md)
- **Schema Reference**: → [Schema Reference](../reference/schemas.md)