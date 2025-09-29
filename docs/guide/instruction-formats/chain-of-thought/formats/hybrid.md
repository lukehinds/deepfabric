# Hybrid Chain of Thought Format

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

## Examples by Domain

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

### Physics - Quantum Entanglement

```json
{
  "question": "How can two particles remain entangled over vast distances, and what implications does this have for information transfer?",
  "chain_of_thought": "Quantum entanglement is a phenomenon where two particles become linked, such that the state of one particle instantly influences the state of the other, regardless of the distance separating them. This raises questions about the nature of information transfer and challenges classical notions of locality and causality.",
  "reasoning_trace": [
    {
      "step_number": 1,
      "thought": "Understand the concept of quantum entanglement.",
      "action": "Research the basics of quantum entanglement, including how particles become entangled and the implications of this entanglement."
    },
    {
      "step_number": 2,
      "thought": "Explore the principle of non-locality in quantum mechanics.",
      "action": "Investigate how entangled particles exhibit non-local behavior, meaning their states are correlated even when separated by large distances."
    },
    {
      "step_number": 3,
      "thought": "Consider the implications for information transfer.",
      "action": "Analyze whether entanglement allows for faster-than-light communication or if it simply reflects a deeper quantum correlation."
    },
    {
      "step_number": 4,
      "thought": "Evaluate the role of measurement in entanglement.",
      "action": "Examine how measuring one particle affects the state of the other and the implications for information transfer."
    },
    {
      "step_number": 5,
      "thought": "Synthesize findings to address the question.",
      "action": "Combine insights from previous steps to form a coherent explanation of how entanglement affects information transfer and its broader implications."
    }
  ],
  "final_answer": "While quantum entanglement suggests a form of instantaneous correlation between particles, it does not allow for faster-than-light communication. The phenomenon challenges classical ideas of locality but adheres to the principles of quantum mechanics, where information transfer is constrained by the no-communication theorem."
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
async for event in tree.build_async():
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

## Action Field Guidelines for Hybrid CoT

The `action` field in `ReasoningStep` is a free-form string that describes what action is being taken in that reasoning step. While not enforced by the schema, here are suggested patterns for different reasoning styles:

**Note**: These are documentation suggestions only. The `action` field accepts any string value and there's no validation against these specific values in the code.

### Analytical Actions (Suggested)
- `"classify_algorithm"`: When categorizing computational approaches
- `"analyze_complexity"`: When examining algorithmic or mathematical complexity
- `"decompose_problem"`: When breaking down into sub-problems
- `"synthesize_solution"`: When combining multiple approaches

### Explanatory Actions (Suggested)
- `"build_intuition"`: When developing conceptual understanding
- `"build_analogy"`: When using metaphors or comparisons
- `"explain_mechanism"`: When describing how something works
- `"provide_context"`: When giving background information

### Logical Actions (Suggested)
- `"state_theorem"`: When presenting formal statements
- `"proof_by_contradiction"`: When using indirect proof methods
- `"apply_principle"`: When using established rules or laws
- `"verify_logic"`: When checking reasoning validity

### Strategic Actions (Suggested)
- `"identify_constraints"`: When recognizing limitations or requirements
- `"suggest_optimizations"`: When proposing improvements
- `"evaluate_tradeoffs"`: When analyzing competing factors
- `"provide_guidelines"`: When offering decision criteria

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
