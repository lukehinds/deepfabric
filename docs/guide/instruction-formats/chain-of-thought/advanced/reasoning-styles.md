# Reasoning Styles Guide

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

DeepFabric's Chain of Thought implementation supports three distinct reasoning styles that optimize the generated reasoning patterns for different domains and problem types. Understanding when and how to use each style is crucial for creating high-quality CoT datasets.
## Overview of Reasoning Styles

| Style | Best For | Characteristics | Example Domains |
|-------|----------|-----------------|-----------------|
| **Mathematical** | Numerical problems | Step-by-step calculations, verification | Math, physics, chemistry |
| **Logical** | Structured reasoning | Premise-conclusion chains, formal logic | CS algorithms, proofs, analysis |
| **General** | Mixed domains | Flexible, adaptive reasoning | General Q&A, interdisciplinary problems |

## Mathematical Reasoning Style

### When to Use
- **Primary use cases**: Math word problems, physics calculations, financial problems
- **Problem characteristics**: Involve numerical computation, measurements, quantitative analysis
- **Learning objectives**: Teaching systematic calculation methods and verification

### Reasoning Characteristics

Mathematical reasoning emphasizes:
- **Sequential calculations**: Each step builds on the previous
- **Explicit arithmetic**: Shows mathematical operations clearly
- **Unit tracking**: Maintains dimensional analysis
- **Verification steps**: Double-checks calculations when possible
- **Precision focus**: Exact values and proper rounding

### Example: Mathematical Style

```json
{
  "question": "A rectangular garden is 15 feet long and 8 feet wide. If fencing costs $4.50 per foot, how much will it cost to fence the entire perimeter?",
  "chain_of_thought": "To find the cost of fencing, I need to calculate the perimeter first. For a rectangle, perimeter = 2 × (length + width). Length = 15 feet, width = 8 feet. Perimeter = 2 × (15 + 8) = 2 × 23 = 46 feet. Now I can calculate the total cost: Cost = perimeter × price per foot = 46 feet × $4.50/foot = $207.00. Let me verify: 46 × $4.50 = $207.00 ✓",
  "final_answer": "$207.00"
}
```

**Key features of mathematical style**:
- Shows formula: `perimeter = 2 × (length + width)`
- Explicit substitution: `2 × (15 + 8)`
- Step-by-step calculation: `2 × 23 = 46`
- Unit tracking: `46 feet × $4.50/foot`
- Verification: `46 × $4.50 = $207.00 ✓`

### Configuration for Mathematical Style

```yaml
# Optimized for mathematical reasoning
data_engine:
  conversation_type: "cot_freetext"
  reasoning_style: "mathematical"
  temperature: 0.2          # Lower for calculation consistency

  generation_system_prompt: "You are a mathematics tutor who shows detailed step-by-step calculations with proper mathematical notation and verification."
```

```python
# Python API configuration
generator = DataSetGenerator(
    generation_system_prompt="You are a math teacher who demonstrates careful calculation steps with verification.",
    conversation_type="cot_freetext",
    reasoning_style="mathematical",
    temperature=0.2  # Consistent calculations
)
```

## Logical Reasoning Style

### When to Use
- **Primary use cases**: Algorithm analysis, logical proofs, decision trees, systematic analysis
- **Problem characteristics**: Require formal reasoning, rule application, structured argumentation
- **Learning objectives**: Teaching systematic thinking and logical progression

### Reasoning Characteristics

Logical reasoning emphasizes:
- **Premise identification**: Clear statement of given conditions
- **Rule application**: Explicit use of logical principles
- **Systematic progression**: Following formal reasoning patterns
- **Conclusion validation**: Ensuring logical consistency
- **Structure awareness**: Organizing thoughts hierarchically

### Example: Logical Style

```json
{
  "question": "In a binary search tree, if we want to find the minimum element, what is the optimal strategy and what is the time complexity?",
  "chain_of_thought": "To find the minimum element in a binary search tree, I need to apply the BST property: for any node, all elements in the left subtree are smaller than the node's value. Given this property, the minimum element must be the leftmost node in the tree. The optimal strategy is: 1) Start at the root node, 2) While the current node has a left child, move to the left child, 3) When we reach a node with no left child, that node contains the minimum value. For time complexity analysis: In each step, we move one level down the tree, and we only traverse the left path. The maximum number of steps equals the height of the tree. Therefore, time complexity is O(h) where h is the height of the tree, which is O(log n) for a balanced BST and O(n) for a degenerate tree.",
  "final_answer": "Follow the left path from root to the leftmost node. Time complexity: O(h) where h is tree height."
}
```

**Key features of logical style**:
- Property identification: "BST property: all elements in left subtree are smaller"
- Systematic steps: "1) Start at root, 2) Move left, 3) Stop when no left child"
- Logical deduction: "minimum element must be the leftmost node"
- Complexity analysis: "O(h) where h is height"
- Formal reasoning structure

### Configuration for Logical Style

```yaml
# Optimized for logical reasoning
data_engine:
  conversation_type: "cot_structured"  # Often pairs well with structured format
  reasoning_style: "logical"
  temperature: 0.3          # Balanced consistency and flexibility

  generation_system_prompt: "You are a computer science instructor who demonstrates systematic logical reasoning with clear premise-conclusion chains."
```

```python
# Python API with logical reasoning
generator = DataSetGenerator(
    generation_system_prompt="You are a CS professor who teaches systematic problem-solving with formal logical reasoning.",
    conversation_type="cot_structured",
    reasoning_style="logical",
    temperature=0.3
)
```

## General Reasoning Style

### When to Use
- **Primary use cases**: Interdisciplinary problems, creative problem-solving, general Q&A
- **Problem characteristics**: Mixed domains, require adaptive thinking, no single dominant reasoning pattern
- **Learning objectives**: Teaching flexible thinking and domain adaptation

### Reasoning Characteristics

General reasoning emphasizes:
- **Adaptive approach**: Changes reasoning style based on problem type
- **Contextual awareness**: Considers domain-specific factors
- **Flexible structure**: Not bound to mathematical or logical templates
- **Natural flow**: Follows intuitive reasoning patterns
- **Integrated thinking**: Combines multiple reasoning approaches

### Example: General Style

```json
{
  "question": "A company is deciding between two marketing strategies: Strategy A costs $50,000 upfront with a 70% success rate, while Strategy B costs $80,000 upfront with a 90% success rate. Both strategies, if successful, generate $200,000 in revenue. Which strategy should the company choose?",
  "chain_of_thought": "This is a decision problem that involves both mathematical calculation and business judgment. Let me calculate the expected value for each strategy. Strategy A: Expected revenue = $200,000 × 0.70 = $140,000. Net expected value = $140,000 - $50,000 = $90,000. Strategy B: Expected revenue = $200,000 × 0.90 = $180,000. Net expected value = $180,000 - $80,000 = $100,000. From a purely mathematical perspective, Strategy B has a higher expected value ($100,000 vs $90,000). However, we should also consider risk tolerance and cash flow. Strategy A requires less upfront investment and might be better for a cash-constrained company. Strategy B has higher expected returns but also higher upfront risk. The choice depends on the company's risk profile and available capital.",
  "final_answer": "Strategy B has higher expected value ($100,000 vs $90,000), but the choice depends on the company's risk tolerance and cash flow situation."
}
```

**Key features of general style**:
- Multi-faceted analysis: combines math and business considerations
- Adaptive reasoning: switches between calculation and qualitative analysis
- Context awareness: considers company's financial situation
- Balanced conclusion: acknowledges multiple factors
- Natural language flow: not constrained by formal templates

### Configuration for General Style

```yaml
# Optimized for general reasoning
data_engine:
  conversation_type: "cot_hybrid"    # Often works well with hybrid format
  reasoning_style: "general"
  temperature: 0.4          # Higher creativity for adaptive thinking

  generation_system_prompt: "You are an expert who adapts your reasoning approach to the specific problem, combining mathematical, logical, and intuitive thinking as needed."
```

```python
# Python API with general reasoning
generator = DataSetGenerator(
    generation_system_prompt="You are a problem-solving expert who uses the most appropriate reasoning approach for each unique situation.",
    conversation_type="cot_hybrid",
    reasoning_style="general",
    temperature=0.4  # More flexibility
)
```

## Comparative Analysis

### Style Selection Decision Tree

```
Is your problem primarily...

├─ Numerical/Quantitative?
│  ├─ Calculations, measurements, formulas
│  └─ → Use MATHEMATICAL style
│
├─ Logical/Structural?
│  ├─ Algorithms, proofs, systematic analysis
│  └─ → Use LOGICAL style
│
└─ Mixed/Interdisciplinary?
   ├─ Multiple domains, creative problem-solving
   └─ → Use GENERAL style
```

## Advanced Usage Patterns

### Mixing Styles Within a Dataset

For comprehensive datasets, you can generate samples with different reasoning styles:

```python
def create_mixed_style_dataset():
    """Create dataset with multiple reasoning styles."""

    from deepfabric import DataSetGenerator
    from deepfabric.tree import Tree

    # Define style-specific configurations
    styles = {
        "mathematical": {
            "topic_prompt": "Mathematical word problems requiring calculations",
            "reasoning_style": "mathematical",
            "temperature": 0.2,
            "samples": 20
        },
        "logical": {
            "topic_prompt": "Algorithm and logical reasoning problems",
            "reasoning_style": "logical",
            "temperature": 0.3,
            "samples": 15
        },
        "general": {
            "topic_prompt": "Interdisciplinary problems requiring adaptive thinking",
            "reasoning_style": "general",
            "temperature": 0.4,
            "samples": 15
        }
    }

    all_samples = []

    for style_name, config in styles.items():
        print(f"Generating {style_name} reasoning samples...")

        # Create style-specific topic tree
        tree = Tree(
            topic_prompt=config["topic_prompt"],
            provider="openai",
            model_name="gpt-4o-mini",
            degree=3,
            depth=2,
            temperature=config["temperature"]
        )

        async for event in tree.build_async():
            if event['event'] == 'build_complete':
                print(f"  Topics: {event['total_paths']}")

        # Generate with style-specific settings
        generator = DataSetGenerator(
            provider="openai",
            model_name="gpt-4o-mini",
            conversation_type="cot_freetext",
            reasoning_style=config["reasoning_style"],
            temperature=config["temperature"]
        )

        dataset = generator.create_data(
            num_steps=config["samples"],
            batch_size=1,
            topic_model=tree,
            sys_msg=False
        )

        # Tag samples with reasoning style
        for sample in dataset.samples:
            sample["reasoning_style"] = style_name

        all_samples.extend(dataset.samples)
        print(f"  Generated {len(dataset.samples)} samples")

    # Save combined dataset
    from deepfabric.dataset import Dataset
    combined = Dataset()
    combined.samples = all_samples
    combined.save("mixed_reasoning_styles.jsonl")

    print(f"\nTotal samples: {len(all_samples)}")

    # Analyze style distribution
    from collections import Counter
    style_counts = Counter(sample["reasoning_style"] for sample in all_samples)
    for style, count in style_counts.items():
        print(f"  {style}: {count} samples")

# Usage
create_mixed_style_dataset()
```

### Dynamic Style Selection

For advanced use cases, you can implement dynamic style selection based on problem characteristics:

```python
def select_reasoning_style(problem_text: str) -> str:
    """Automatically select reasoning style based on problem content."""

    problem_lower = problem_text.lower()

    # Mathematical indicators
    math_keywords = ['calculate', 'cost', 'price', 'area', 'volume', 'distance', 'speed', 'time', 'percent', 'fraction']
    math_symbols = any(char in problem_text for char in '+-×÷=<>%$')

    if any(keyword in problem_lower for keyword in math_keywords) or math_symbols:
        return "mathematical"

    # Logical indicators
    logic_keywords = ['algorithm', 'if', 'then', 'prove', 'logic', 'tree', 'graph', 'sort', 'search', 'complexity']
    cs_terms = ['array', 'list', 'node', 'function', 'recursive', 'iterate']

    if any(keyword in problem_lower for keyword in logic_keywords + cs_terms):
        return "logical"

    # Default to general for mixed or unclear domains
    return "general"

# Example usage
problems = [
    "Calculate the area of a rectangle with length 5m and width 3m",  # → mathematical
    "Explain how binary search works and analyze its time complexity",  # → logical
    "Should a company invest in renewable energy given environmental and economic factors?"  # → general
]

for problem in problems:
    style = select_reasoning_style(problem)
    print(f"Problem: {problem[:50]}...")
    print(f"Recommended style: {style}\n")
```

## Style-Specific Prompt Engineering

### Mathematical Style Prompts

```python
# Enhanced prompts for mathematical reasoning
mathematical_prompts = {
    "generation_system_prompt": """You are a mathematics teacher who demonstrates problems with:
    - Clear step-by-step calculations
    - Proper mathematical notation
    - Unit tracking and dimensional analysis
    - Verification steps when possible
    - Exact arithmetic with explanations""",

    "instructions": """Create mathematical word problems that require:
    - Multi-step numerical calculations
    - Real-world applications of mathematical concepts
    - Clear quantitative relationships
    - Opportunities for verification and checking"""
}
```

### Logical Style Prompts

```python
# Enhanced prompts for logical reasoning
logical_prompts = {
    "generation_system_prompt": """You are a computer science instructor who demonstrates:
    - Systematic logical progression
    - Clear premise-conclusion relationships
    - Formal reasoning patterns
    - Structured analysis methods
    - Rule-based decision making""",

    "instructions": """Create problems that require:
    - Systematic logical analysis
    - Algorithm understanding and application
    - Formal reasoning and proof techniques
    - Structured problem decomposition"""
}
```

### General Style Prompts

```python
# Enhanced prompts for general reasoning
general_prompts = {
    "generation_system_prompt": """You are an expert problem solver who:
    - Adapts reasoning approach to the problem domain
    - Combines multiple types of thinking (mathematical, logical, creative)
    - Considers contextual factors and real-world constraints
    - Uses natural, intuitive reasoning patterns
    - Balances analytical and creative thinking""",

    "instructions": """Create diverse problems that require:
    - Adaptive thinking across different domains
    - Integration of multiple perspectives
    - Consideration of practical constraints
    - Creative and flexible problem-solving approaches"""
}
```

## Quality Evaluation by Style

### Style-Specific Quality Metrics

```python
def evaluate_reasoning_style_quality(sample: dict, expected_style: str) -> dict:
    """Evaluate how well a sample matches its intended reasoning style."""

    reasoning = sample.get('chain_of_thought', '').lower()

    scores = {
        'mathematical': 0.0,
        'logical': 0.0,
        'general': 0.0
    }

    # Mathematical style indicators
    math_indicators = [
        any(char in reasoning for char in '+-×÷='),  # Mathematical symbols
        any(word in reasoning for word in ['calculate', 'multiply', 'divide', 'add', 'subtract']),
        'verify' in reasoning or 'check' in reasoning,  # Verification
        any(word in reasoning for word in ['step', 'first', 'then', 'next']),  # Sequential steps
    ]
    scores['mathematical'] = sum(math_indicators) / len(math_indicators)

    # Logical style indicators
    logic_indicators = [
        any(word in reasoning for word in ['therefore', 'because', 'since', 'given']),  # Logical connectors
        any(word in reasoning for word in ['property', 'rule', 'principle', 'definition']),  # Formal concepts
        any(word in reasoning for word in ['if', 'then', 'when', 'condition']),  # Conditional logic
        reasoning.count('.') >= 3,  # Structured sentences
    ]
    scores['logical'] = sum(logic_indicators) / len(logic_indicators)

    # General style indicators
    general_indicators = [
        any(word in reasoning for word in ['consider', 'factor', 'aspect', 'perspective']),  # Multi-faceted thinking
        any(word in reasoning for word in ['however', 'but', 'although', 'while']),  # Balanced reasoning
        any(word in reasoning for word in ['context', 'situation', 'case', 'scenario']),  # Contextual awareness
        len(reasoning.split()) > 50,  # Substantial explanation
    ]
    scores['general'] = sum(general_indicators) / len(general_indicators)

    return {
        'scores': scores,
        'predicted_style': max(scores, key=scores.get),
        'confidence': max(scores.values()),
        'matches_expected': max(scores, key=scores.get) == expected_style
    }

# Usage
sample = {
    'chain_of_thought': 'To calculate the area, I need to multiply length × width. Area = 5m × 3m = 15m². Let me verify: 5 × 3 = 15 ✓'
}
evaluation = evaluate_reasoning_style_quality(sample, 'mathematical')
print(f"Predicted style: {evaluation['predicted_style']}")
print(f"Confidence: {evaluation['confidence']:.2f}")
print(f"Matches expected: {evaluation['matches_expected']}")
```

## Troubleshooting Style Issues

### Common Problems and Solutions

#### Problem: Mathematical style not showing calculations

**Symptoms**: Reasoning lacks explicit arithmetic steps
**Solution**: Adjust prompts and temperature

```python
# More explicit mathematical prompting
generator = DataSetGenerator(
    generation_system_prompt="You are a math teacher who ALWAYS shows every calculation step with arithmetic operations clearly visible (like 15 + 8 = 23).",
    reasoning_style="mathematical",
    temperature=0.1  # Very low for calculation consistency
)
```

#### Problem: Logical style too informal

**Symptoms**: Reasoning lacks formal structure and logical connectors
**Solution**: Emphasize systematic reasoning

```python
# More formal logical prompting
generator = DataSetGenerator(
    generation_system_prompt="You are a computer science professor who demonstrates formal logical reasoning with clear premises, systematic analysis, and structured conclusions.",
    reasoning_style="logical",
    temperature=0.2
)
```

#### Problem: General style too vague

**Symptoms**: Reasoning lacks depth and specific insights
**Solution**: Encourage multi-faceted analysis

```python
# More comprehensive general reasoning
generator = DataSetGenerator(
    generation_system_prompt="You are an expert analyst who examines problems from multiple angles, considering quantitative factors, logical relationships, and practical constraints.",
    reasoning_style="general",
    temperature=0.35
)
```

## Conclusion

Reasoning styles are a powerful tool for creating domain-appropriate Chain of Thought datasets. By selecting the right style for your use case:

- **Mathematical style** excels at numerical problems requiring step-by-step calculations
- **Logical style** shines for algorithmic and systematic reasoning tasks
- **General style** adapts flexibly to interdisciplinary and creative problems

Understanding these distinctions and applying them appropriately will significantly improve the quality and effectiveness of your CoT datasets for training reasoning-capable language models.

## Next Steps

- **Experiment with styles**: Try different styles on the same problem type to see the differences
- **Mixed datasets**: Create datasets combining multiple reasoning styles for comprehensive training
- **Custom styles**: Consider extending DeepFabric to support domain-specific reasoning patterns
- **Evaluation**: Develop metrics to assess reasoning quality within each style
- **Model training**: Fine-tune models on style-specific datasets and compare performance