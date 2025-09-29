# Tutorial: Building a Math Reasoning Dataset

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

This comprehensive tutorial walks you through creating a high-quality mathematical reasoning dataset using DeepFabric's Chain of Thought capabilities. We'll build a GSM8K-style dataset focused on elementary and middle school math problems.
## Tutorial Overview

**What you'll build**: A dataset of 50 mathematical word problems with step-by-step reasoning
**Format**: Free-text Chain of Thought
**Domain**: Elementary/middle school mathematics
**Time required**: 30-45 minutes
**Cost estimate**: ~$2-5 using GPT-4o-mini

## Prerequisites

- DeepFabric installed (`pip install deepfabric`)
- OpenAI API key set as environment variable
- Basic familiarity with YAML or Python

```bash
# Verify setup
echo $OPENAI_API_KEY
deepfabric --version
```

## Step 1: Domain Analysis and Planning

### Understanding Math Reasoning Requirements

Mathematical reasoning datasets need:
- **Clear problem statements**: Unambiguous questions with sufficient information
- **Step-by-step solutions**: Explicit reasoning showing mathematical work
- **Accurate calculations**: Error-free arithmetic and logical progression
- **Natural language flow**: Readable explanations that teach the process
- **Diverse problem types**: Varied scenarios to avoid over-fitting

### Target Problem Types

We'll generate problems covering:
- **Basic arithmetic**: Addition, subtraction, multiplication, division with word problems
- **Multi-step problems**: Requiring 2-4 calculation steps
- **Real-world scenarios**: Money, measurements, time, age problems
- **Proportional reasoning**: Ratios, rates, unit conversions
- **Basic geometry**: Area, perimeter, simple volume

## Step 2: Topic Structure Design

### Creating a Hierarchical Topic Tree

```yaml
# math-reasoning-config.yaml
dataset_system_prompt: "You are a helpful math tutor who explains problems step-by-step with clear, logical reasoning."

topic_tree:
  topic_prompt: "Elementary and middle school mathematics word problems covering arithmetic, basic algebra, geometry, and real-world applications. Include problems involving money, measurements, time, age, ratios, and multi-step calculations suitable for grades 3-8."

  # LLM configuration for topic generation
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7        # Higher creativity for diverse topics

  # Tree structure for comprehensive coverage
  degree: 4               # 4 subtopics per node for good variety
  depth: 2                # 2 levels = 16 total topic paths

  save_as: "math_topics.jsonl"
```

Let's understand this configuration:
- **degree: 4, depth: 2** = 4^2 = 16 topic paths
- **temperature: 0.7** = Good balance of creativity and consistency
- **gpt-4o-mini** = Cost-effective while maintaining quality

## Step 3: Data Generation Configuration

```yaml
# Continue in math-reasoning-config.yaml

data_engine:
  instructions: "Create clear mathematical word problems that require analytical thinking and step-by-step problem solving. Focus on real-world scenarios that are relatable to students."

  generation_system_prompt: "You are a mathematics educator creating practice problems. Generate problems that require multiple steps to solve and show clear mathematical reasoning in the solution process."

  # LLM settings optimized for math reasoning
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.3        # Lower temperature for consistent reasoning
  max_retries: 4

  # Chain of Thought configuration
  conversation_type: "cot_freetext"
  reasoning_style: "mathematical"

dataset:
  creation:
    num_steps: 50         # Generate 50 examples
    batch_size: 1         # One at a time for quality
    sys_msg: false        # Free-text CoT doesn't use system messages

  save_as: "math_reasoning_dataset.jsonl"
```

### Configuration Rationale

- **temperature: 0.3** for generation vs **0.7** for topics creates good diversity in problems while maintaining reasoning consistency
- **reasoning_style: "mathematical"** optimizes prompts for numerical reasoning and calculations
- **num_steps: 50** provides a substantial dataset for training while keeping costs reasonable

## Step 4: Generation Process

### Method 1: Using YAML Configuration

```bash
# Generate the dataset
deepfabric generate math-reasoning-config.yaml

# Monitor progress (in another terminal if desired)
tail -f math_reasoning_dataset.jsonl | wc -l
```

### Method 2: Using Python for Better Control

```python
# math_reasoning_generator.py
import json
import time
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree
from deepfabric.dataset import Dataset

def main():
    print("Math Reasoning Dataset Generator")
    print("=" * 50)

    # Step 1: Create topic tree
    print("\nStep 1: Generating topic structure...")
    tree = Tree(
        topic_prompt="Elementary and middle school mathematics word problems covering arithmetic, basic algebra, geometry, and real-world applications. Include problems involving money, measurements, time, age, ratios, and multi-step calculations suitable for grades 3-8.",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=4,
        depth=2,
        temperature=0.7
    )

    # Build tree with progress tracking
    topic_count = 0
    async for event in tree.build_async():
        if event['event'] == 'depth_start':
            print(f"  Building depth {event['depth']}...")
        elif event['event'] == 'build_complete':
            topic_count = event['total_paths']
            print(f"  Generated {topic_count} math topic paths")

    # Step 2: Create dataset generator
    print("\nStep 2: Configuring problem generator...")
    generator = DataSetGenerator(
        instructions="Create clear mathematical word problems that require analytical thinking and step-by-step problem solving. Focus on real-world scenarios that are relatable to students.",
        generation_system_prompt="You are a mathematics educator creating practice problems. Generate problems that require multiple steps to solve and show clear mathematical reasoning in the solution process.",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.3,
        conversation_type="cot_freetext",
        reasoning_style="mathematical"
    )

    # Step 3: Generate dataset with monitoring
    print("\nStep 3: Generating math problems...")
    start_time = time.time()

    dataset = None
    for event in generator.create_data_with_events(
        num_steps=50,
        batch_size=1,
        topic_model=tree,
        sys_msg=False
    ):
        if isinstance(event, dict):
            if event.get('event') == 'step_start':
                print(f"  Generating problem {event['step']}/50...")
            elif event.get('event') == 'step_complete':
                elapsed = time.time() - start_time
                rate = event['step'] / elapsed * 60  # problems per minute
                print(f"    ✓ Problem {event['step']}: {event['samples_generated']} samples ({rate:.1f}/min)")
            elif event.get('event') == 'generation_complete':
                total_time = time.time() - start_time
                print(f"\nGeneration complete!")
                print(f"   Total time: {total_time:.1f} seconds")
                print(f"   Total samples: {event['total_samples']}")
                print(f"   Average: {total_time/event['total_samples']:.1f}s per problem")
        else:
            dataset = event

    # Step 4: Validate and save
    print("\nStep 4: Validating and saving dataset...")
    if dataset and len(dataset.samples) > 0:
        dataset.save("math_reasoning_dataset.jsonl")
        print(f"   Saved {len(dataset.samples)} problems to math_reasoning_dataset.jsonl")

        # Quick quality check
        sample = dataset.samples[0]
        print(f"\nSample problem preview:")
        print(f"   Question: {sample['question'][:100]}...")
        print(f"   Reasoning length: {len(sample['chain_of_thought'])} characters")
        print(f"   Answer: {sample['final_answer']}")
    else:
        print("   No samples generated - check configuration and API key")

if __name__ == "__main__":
    main()
```

## Step 5: Quality Analysis and Validation

### Automated Quality Checks

```python
# quality_analysis.py
import json
import statistics
from collections import Counter

def analyze_dataset(filename: str):
    """Comprehensive quality analysis of math reasoning dataset."""

    print(f"Analyzing {filename}")
    print("=" * 50)

    # Load dataset
    samples = []
    with open(filename, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Basic Statistics:")
    print(f"   Total samples: {len(samples)}")

    # Length analysis
    question_lengths = [len(sample['question']) for sample in samples]
    reasoning_lengths = [len(sample['chain_of_thought']) for sample in samples]
    answer_lengths = [len(sample['final_answer']) for sample in samples]

    print(f"   Question length: {statistics.mean(question_lengths):.0f} ± {statistics.stdev(question_lengths):.0f} chars")
    print(f"   Reasoning length: {statistics.mean(reasoning_lengths):.0f} ± {statistics.stdev(reasoning_lengths):.0f} chars")
    print(f"   Answer length: {statistics.mean(answer_lengths):.0f} ± {statistics.stdev(answer_lengths):.0f} chars")

    # Content analysis
    print(f"\nContent Analysis:")

    # Mathematical operations
    operations = Counter()
    for sample in samples:
        reasoning = sample['chain_of_thought'].lower()
        if '+' in reasoning or 'add' in reasoning or 'sum' in reasoning:
            operations['addition'] += 1
        if '-' in reasoning or 'subtract' in reasoning or 'minus' in reasoning:
            operations['subtraction'] += 1
        if '×' in reasoning or '*' in reasoning or 'multiply' in reasoning or 'times' in reasoning:
            operations['multiplication'] += 1
        if '÷' in reasoning or '/' in reasoning or 'divide' in reasoning:
            operations['division'] += 1

    print("   Mathematical operations detected:")
    for op, count in operations.most_common():
        print(f"     {op}: {count} problems ({count/len(samples)*100:.1f}%)")

    # Step indicators
    step_indicators = 0
    calculation_indicators = 0
    for sample in samples:
        reasoning = sample['chain_of_thought'].lower()
        if any(word in reasoning for word in ['step', 'first', 'then', 'next', 'finally']):
            step_indicators += 1
        if any(char in reasoning for char in '=+-×÷'):
            calculation_indicators += 1

    print(f"   Step-by-step indicators: {step_indicators} problems ({step_indicators/len(samples)*100:.1f}%)")
    print(f"   Mathematical symbols: {calculation_indicators} problems ({calculation_indicators/len(samples)*100:.1f}%)")

    # Quality flags
    print(f"\n⚠️  Quality Flags:")
    short_reasoning = sum(1 for l in reasoning_lengths if l < 100)
    long_reasoning = sum(1 for l in reasoning_lengths if l > 800)
    empty_answers = sum(1 for sample in samples if len(sample['final_answer'].strip()) == 0)

    print(f"   Short reasoning (<100 chars): {short_reasoning} ({short_reasoning/len(samples)*100:.1f}%)")
    print(f"   Long reasoning (>800 chars): {long_reasoning} ({long_reasoning/len(samples)*100:.1f}%)")
    print(f"   Empty answers: {empty_answers}")

    # Sample examples by category
    print(f"\nSample Problems by Category:")

    # Find examples of different problem types
    categories = {
        'money': ['dollar', 'cent', 'price', 'cost', 'buy', 'sell'],
        'time': ['hour', 'minute', 'day', 'week', 'month'],
        'measurement': ['meter', 'feet', 'inch', 'pound', 'gram', 'liter'],
        'age': ['age', 'old', 'older', 'younger', 'born'],
        'geometry': ['area', 'perimeter', 'rectangle', 'circle', 'triangle']
    }

    for category, keywords in categories.items():
        examples = []
        for sample in samples:
            question = sample['question'].lower()
            if any(keyword in question for keyword in keywords):
                examples.append(sample)

        if examples:
            example = examples[0]
            print(f"\n   {category.title()} problem example:")
            print(f"     Q: {example['question'][:120]}...")
            print(f"     A: {example['final_answer']}")

# Usage
analyze_dataset("math_reasoning_dataset.jsonl")
```

### Sample Quality Validation

Let's examine what good vs. poor quality samples look like:

#### ✓ High-Quality Example

```json
{
  "question": "Sarah is saving money to buy a bicycle that costs $180. She has already saved $45. If she saves $15 each week, how many weeks will it take her to have enough money?",
  "chain_of_thought": "I need to find how many more weeks Sarah needs to save. First, let me calculate how much more money she needs: $180 - $45 = $135. She saves $15 each week, so I need to divide the remaining amount by her weekly savings: $135 ÷ $15 = 9. Let me verify: $45 (already saved) + $15 × 9 weeks = $45 + $135 = $180. ✓",
  "final_answer": "9 weeks"
}
```

**Why this is high quality**:
- Clear, realistic problem scenario
- Shows all calculation steps
- Includes verification
- Natural language explanation
- Correct mathematical reasoning

#### ✗ Poor-Quality Example

```json
{
  "question": "How much is 25 + 37?",
  "chain_of_thought": "25 + 37 = 62",
  "final_answer": "62"
}
```

**Why this is poor quality**:
- Too simple, not a word problem
- No step-by-step reasoning shown
- Missing real-world context
- Doesn't demonstrate analytical thinking

## Step 6: Dataset Refinement and Filtering

### Quality Filtering Script

```python
# filter_quality.py
import json
from deepfabric.dataset import Dataset

def quality_score(sample: dict) -> float:
    """Calculate quality score for a math reasoning sample."""
    score = 0.0

    # Question quality (30% of score)
    question = sample['question']
    if len(question) >= 50:  # Substantial problem
        score += 0.15
    if any(word in question.lower() for word in ['buy', 'cost', 'save', 'spend', 'earn', 'age', 'time', 'distance', 'area']):  # Real-world context
        score += 0.15

    # Reasoning quality (50% of score)
    reasoning = sample['chain_of_thought']
    if len(reasoning) >= 100:  # Sufficient explanation
        score += 0.15
    if any(word in reasoning.lower() for word in ['first', 'then', 'next', 'step', 'calculate', 'find']):  # Step indicators
        score += 0.15
    if any(char in reasoning for char in '=+-×÷'):  # Mathematical operations shown
        score += 0.10
    if 'verify' in reasoning.lower() or '✓' in reasoning:  # Verification step
        score += 0.10

    # Answer quality (20% of score)
    answer = sample['final_answer']
    if len(answer.strip()) > 0 and len(answer) <= 50:  # Non-empty, concise answer
        score += 0.20

    return min(score, 1.0)

def filter_high_quality_samples(input_file: str, output_file: str, threshold: float = 0.75):
    """Filter dataset to keep only high-quality samples."""

    # Load original dataset
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Score and filter
    high_quality = []
    quality_scores = []

    for sample in samples:
        score = quality_score(sample)
        quality_scores.append(score)
        if score >= threshold:
            high_quality.append(sample)

    print(f"Quality Filtering Results:")
    print(f"  Original samples: {len(samples)}")
    print(f"  Average quality score: {sum(quality_scores)/len(quality_scores):.2f}")
    print(f"  High quality (≥{threshold}): {len(high_quality)}")
    print(f"  Retention rate: {len(high_quality)/len(samples)*100:.1f}%")

    # Save filtered dataset
    with open(output_file, 'w') as f:
        for sample in high_quality:
            f.write(json.dumps(sample) + '\n')

    return high_quality

# Usage
high_quality_samples = filter_high_quality_samples(
    "math_reasoning_dataset.jsonl",
    "math_reasoning_filtered.jsonl",
    threshold=0.75
)
```

## Step 7: Advanced Enhancements

### Domain-Specific Improvements

If you want to focus on specific mathematical domains:

```python
# domain_specific_generation.py
def generate_domain_specific_datasets():
    """Generate separate datasets for different math domains."""

    domains = {
        "arithmetic": {
            "prompt": "Basic arithmetic word problems involving addition, subtraction, multiplication, and division with whole numbers and decimals",
            "style": "mathematical",
            "samples": 20
        },
        "geometry": {
            "prompt": "Elementary geometry problems involving area, perimeter, volume, and basic shapes",
            "style": "mathematical",
            "samples": 15
        },
        "fractions": {
            "prompt": "Fraction problems including addition, subtraction, multiplication, division, and word problems with fractions",
            "style": "mathematical",
            "samples": 15
        },
        "word_problems": {
            "prompt": "Real-world word problems involving money, time, measurement, and multi-step reasoning",
            "style": "general",
            "samples": 20
        }
    }

    for domain, config in domains.items():
        print(f"\nGenerating {domain} dataset...")

        tree = Tree(
            topic_prompt=config["prompt"],
            provider="openai",
            model_name="gpt-4o-mini",
            degree=3,
            depth=2,
            temperature=0.7
        )

        async for event in tree.build_async():
            if event['event'] == 'build_complete':
                print(f"  Topics: {event['total_paths']}")

        generator = DataSetGenerator(
            instructions=f"Create {domain} problems suitable for elementary/middle school students.",
            generation_system_prompt=f"You are a math teacher specializing in {domain}.",
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.3,
            conversation_type="cot_freetext",
            reasoning_style=config["style"]
        )

        dataset = generator.create_data(
            num_steps=config["samples"],
            batch_size=1,
            topic_model=tree,
            sys_msg=False
        )

        filename = f"math_{domain}_cot.jsonl"
        dataset.save(filename)
        print(f"  Saved {len(dataset.samples)} problems to {filename}")

# Usage
generate_domain_specific_datasets()
```

## Step 8: Integration and Next Steps

### Combining Multiple Datasets

```python
# combine_datasets.py
def combine_math_datasets():
    """Combine multiple math datasets into a comprehensive collection."""

    from deepfabric.dataset import Dataset

    # Load all datasets
    filenames = [
        "math_arithmetic_cot.jsonl",
        "math_geometry_cot.jsonl",
        "math_fractions_cot.jsonl",
        "math_word_problems_cot.jsonl"
    ]

    combined_dataset = Dataset()

    for filename in filenames:
        try:
            domain_dataset = Dataset.from_jsonl(filename)
            print(f"Loaded {len(domain_dataset.samples)} samples from {filename}")

            # Add samples to combined dataset
            failed, descriptions = combined_dataset.add_samples(domain_dataset.samples)
            if failed:
                print(f"  Warning: {len(failed)} samples failed validation")

        except FileNotFoundError:
            print(f"  Skipping {filename} (not found)")

    # Save combined dataset
    combined_dataset.save("math_reasoning_complete.jsonl")
    print(f"\nCombined dataset: {len(combined_dataset.samples)} total samples")

    return combined_dataset

# Usage
final_dataset = combine_math_datasets()
```

### Preparing for Model Training

```python
# prepare_for_training.py
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_training_data():
    """Prepare the math reasoning dataset for model training."""

    # Load the dataset
    samples = []
    with open("math_reasoning_complete.jsonl", 'r') as f:
        for line in f:
            samples.append(json.loads(line))

    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(samples)

    # Create training splits
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    print(f"Dataset splits:")
    print(f"  Training: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")

    # Save splits
    train_df.to_json("train_math_reasoning.jsonl", orient="records", lines=True)
    val_df.to_json("val_math_reasoning.jsonl", orient="records", lines=True)
    test_df.to_json("test_math_reasoning.jsonl", orient="records", lines=True)

    # Create instruction-following format
    def format_for_instruction_tuning(row):
        return {
            "instruction": "Solve this math problem step by step, showing your reasoning.",
            "input": row['question'],
            "output": f"{row['chain_of_thought']}\n\nFinal answer: {row['final_answer']}"
        }

    instruction_format = train_df.apply(format_for_instruction_tuning, axis=1).tolist()

    with open("math_reasoning_instruction_format.jsonl", 'w') as f:
        for sample in instruction_format:
            f.write(json.dumps(sample) + '\n')

    print(f"Created instruction-tuning format with {len(instruction_format)} samples")

# Usage
prepare_training_data()
```

## Troubleshooting Common Issues

### Issue: Poor Problem Diversity

**Symptoms**: Similar problems generated repeatedly
**Solution**: Increase topic tree diversity

```yaml
# Increase diversity
topic_tree:
  degree: 5      # More subtopics
  depth: 3       # Deeper tree
  temperature: 0.8  # More creative topics
```

### Issue: Reasoning Too Brief

**Symptoms**: Chain of thought lacks detail
**Solution**: Adjust prompts and temperature

```yaml
data_engine:
  instructions: "Create problems requiring detailed multi-step reasoning. Each step should be clearly explained with mathematical work shown."
  temperature: 0.2  # More consistent, detailed reasoning
```

### Issue: Mathematical Errors

**Symptoms**: Incorrect calculations in reasoning
**Solution**: Add verification emphasis

```yaml
data_engine:
  generation_system_prompt: "You are a careful math teacher who always double-checks calculations and shows verification steps."
```

## Cost and Performance Metrics

Based on our testing with GPT-4o-mini:

| Metric | Value |
|--------|-------|
| **Cost per sample** | ~$0.08-0.12 |
| **Generation time** | ~15-25 seconds per sample |
| **Success rate** | ~95% with retries |
| **Average quality score** | 0.78/1.0 |

**Total cost for 50 samples**: $4-6
**Total time**: 15-20 minutes

## Conclusion and Next Steps

You've successfully created a comprehensive mathematical reasoning dataset! Here's what you've accomplished:

✓ **Generated 50+ high-quality math problems** with step-by-step reasoning
✓ **Implemented quality validation** and filtering processes
✓ **Created domain-specific datasets** for different math topics
✓ **Prepared data for model training** with proper splits and formatting

### Recommended Next Steps

1. **Scale up**: Generate larger datasets (200-500 samples) for production use
2. **Model training**: Fine-tune a language model on your reasoning dataset
3. **Evaluation**: Test model performance on math reasoning benchmarks
4. **Domain expansion**: Create datasets for algebra, calculus, statistics
5. **Multi-language**: Generate problems in different languages

### Advanced Extensions

- **Structured CoT**: Try conversation-based math tutoring datasets
- **Hybrid CoT**: Combine natural reasoning with formal mathematical notation
- **Visual elements**: Include problems with diagrams and geometric figures
- **Difficulty progression**: Create graduated difficulty levels
- **Assessment integration**: Build evaluation metrics and rubrics

The mathematical reasoning dataset you've created forms a solid foundation for training models that can think through problems systematically and explain their reasoning clearly - essential capabilities for educational AI applications.