# Python API Configuration Guide for Chain of Thought

This guide demonstrates how to configure and generate Chain of Thought datasets using DeepFabric's Python API. The programmatic approach offers greater flexibility, real-time monitoring, and integration with existing Python workflows.

## Quick Start Example

> **Note**: All examples assume an async context. Wrap snippets with `asyncio.run(...)` or integrate with your existing event loop when using `build_async()` methods.

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

```python
import asyncio
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree
from deepfabric.dataset import Dataset

# 1. Create topic structure
tree = Tree(
    topic_prompt="Elementary mathematics word problems",
    provider="openai",
    model_name="gpt-4o-mini",
    degree=3,
    depth=2,
    temperature=0.7
)

# 2. Build topic tree with progress monitoring
print("Building topic tree...")

async def build_tree() -> None:
    async for event in tree.build_async():
        if event['event'] == 'build_complete':
            print(f"Generated {event['total_paths']} topic paths")

asyncio.run(build_tree())

# 3. Create CoT generator
generator = DataSetGenerator(
    instructions="Create clear math problems requiring step-by-step thinking.",
    generation_system_prompt="You are a math tutor creating practice problems.",
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.3,
    conversation_type="cot_freetext",
    reasoning_style="mathematical"
)

# 4. Generate dataset with event monitoring
dataset = asyncio.run(generator.create_data_async(
    num_steps=10,
    batch_size=1,
    topic_model=tree,
    sys_msg=False
))

# 5. Save and validate
dataset.save("math_reasoning.jsonl")
print(f"Generated {len(dataset.samples)} CoT examples")
```

## Core Classes and Configuration

### Tree vs Graph Topic Generation

#### Tree (Hierarchical Topics)
```python
from deepfabric.tree import Tree

# Hierarchical topic structure
tree = Tree(
    topic_prompt="Computer science algorithms and data structures",
    provider="openai",
    model_name="gpt-4o-mini",
    degree=2,          # 2 subtopics per node
    depth=3,           # 3 levels deep
    temperature=0.6,
    max_retries=3
)

# Build with progress tracking
async for event in tree.build_async():
    if event['event'] == 'depth_start':
        print(f"Starting depth {event['depth']}")
    elif event['event'] == 'build_complete':
        print(f"Tree complete: {event['total_paths']} paths")
```

#### Graph (Interconnected Topics)
```python
from deepfabric.graph import Graph

# More complex interconnected topics
graph = Graph(
    topic_prompt="Interdisciplinary scientific problems",
    provider="openai",
    model_name="gpt-4o",
    degree=2,
    depth=2,
    temperature=0.5
)

# Build with node tracking
async for event in graph.build_async():
    if event['event'] == 'node_expanded':
        print(f"Expanded: {event['node_topic']}")
    elif event['event'] == 'build_complete':
        print(f"Graph complete: {event['nodes_count']} nodes")
```

### DataSetGenerator Configuration

```python
from deepfabric import DataSetGenerator

# Basic configuration
generator = DataSetGenerator(
    # Content guidance
    instructions="High-level guidance for problem creation",
    generation_system_prompt="Specific role for the generation model",

    # LLM settings
    provider="openai",
    model_name="gpt-4o-mini",
    temperature=0.3,
    max_retries=4,
    request_timeout=60,

    # CoT-specific settings
    conversation_type="cot_freetext",    # Required for CoT
    reasoning_style="mathematical",      # Optional: "mathematical", "logical", "general"

    # System message control
    sys_msg=False  # Set during generation, not here
)
```

## Format-Specific Examples

### Free-text Chain of Thought

```python
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree

# Optimized for natural language reasoning
def create_freetext_cot_dataset():
    # Topic generation
    tree = Tree(
        topic_prompt="Mathematical word problems for middle school students",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=3,
        depth=2,
        temperature=0.7
    )

    # Build topics
    topics_created = 0
    async for event in tree.build_async():
        if event['event'] == 'build_complete':
            topics_created = event['total_paths']
            print(f"Created {topics_created} math topics")

    # Data generation
    generator = DataSetGenerator(
        instructions="Create word problems that require multi-step reasoning to solve.",
        generation_system_prompt="You are a mathematics educator creating practice problems with detailed step-by-step solutions.",

        # Efficient model for free-text
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.3,

        # Free-text CoT settings
        conversation_type="cot_freetext",
        reasoning_style="mathematical"
    )

    # Generate with event monitoring
    print("Generating free-text CoT dataset...")
    dataset = None
    for event in generator.create_data_with_events(
        num_steps=15,
        batch_size=1,
        topic_model=tree,
        sys_msg=False
    ):
        if isinstance(event, dict):
            if event.get('event') == 'step_complete':
                print(f"Step {event['step']}: {event['samples_generated']} samples")
        else:
            dataset = event  # Final result

    return dataset

# Usage
dataset = create_freetext_cot_dataset()
dataset.save("freetext_math_reasoning.jsonl")
```

### Structured Chain of Thought

```python
from deepfabric import DataSetGenerator
from deepfabric.graph import Graph

def create_structured_cot_dataset():
    # Complex topic graph for educational dialogues
    graph = Graph(
        topic_prompt="Computer science education topics including algorithms, data structures, and programming concepts",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=2,
        depth=3,
        temperature=0.6
    )

    # Build graph
    async for event in graph.build_async():
        if event['event'] == 'build_complete':
            print(f"Created graph with {event['nodes_count']} nodes")

    # Structured conversation generator
    generator = DataSetGenerator(
        instructions="Create educational conversations where students learn through guided discovery.",
        generation_system_prompt="You are a computer science instructor creating realistic teaching dialogues with systematic reasoning.",

        # Higher capability for complex conversations
        provider="openai",
        model_name="gpt-4o",  # Consider upgrading for better conversations
        temperature=0.4,

        # Structured CoT settings
        conversation_type="cot_structured",
        reasoning_style="logical"
    )

    # Generate conversations
    dataset = asyncio.run(generator.create_data_async(
        num_steps=8,        # Fewer due to complexity
        batch_size=1,
        topic_model=graph,
        sys_msg=True        # Include system messages in conversations
    ))

    return dataset

# Usage with validation
dataset = create_structured_cot_dataset()

# Validate conversation structure
for i, sample in enumerate(dataset.samples[:3]):
    print(f"\nSample {i+1}:")
    print(f"  Messages: {len(sample['messages'])}")
    print(f"  Reasoning steps: {len(sample['reasoning_trace'])}")
    print(f"  Has system message: {'system' in [msg['role'] for msg in sample['messages']]}")

dataset.save("structured_cs_education.jsonl")
```

### Hybrid Chain of Thought

```python
from deepfabric import DataSetGenerator
from deepfabric.tree import Tree

def create_hybrid_cot_dataset():
    # Advanced topics requiring dual reasoning
    tree = Tree(
        topic_prompt="Complex scientific and mathematical problems requiring both intuitive insights and systematic analysis",
        provider="openai",
        model_name="gpt-4o",  # Premium model for complex topics
        degree=2,
        depth=2,
        temperature=0.5
    )

    # Build topics
    async for event in tree.build_async():
        if event['event'] == 'build_complete':
            print(f"Generated {event['total_paths']} complex topics")

    # Hybrid reasoning generator
    generator = DataSetGenerator(
        instructions="Create challenging problems that require both conceptual understanding and systematic step-by-step analysis.",
        generation_system_prompt="You are an expert who excels at combining intuitive scientific insights with rigorous methodical reasoning.",

        # Premium model required for hybrid reasoning
        provider="openai",
        model_name="gpt-4o",
        temperature=0.3,
        max_retries=5,      # More retries due to complexity

        # Hybrid CoT settings
        conversation_type="cot_hybrid",
        reasoning_style="logical"
    )

    # Generate with careful monitoring
    dataset = None
    total_tokens = 0

    for event in generator.create_data_with_events(
        num_steps=5,        # Fewer samples due to cost and complexity
        batch_size=1,
        topic_model=tree,
        sys_msg=False
    ):
        if isinstance(event, dict):
            if event.get('event') == 'step_complete':
                print(f"Generated step {event['step']}: {event['samples_generated']} samples")
            elif event.get('event') == 'generation_complete':
                print(f"Total samples: {event['total_samples']}")
        else:
            dataset = event

    return dataset

# Usage with cost monitoring
dataset = create_hybrid_cot_dataset()

# Analyze sample complexity
if dataset.samples:
    sample = dataset.samples[0]
    cot_length = len(sample['chain_of_thought'])
    trace_length = len(sample['reasoning_trace'])
    avg_step_length = sum(len(step['thought']) for step in sample['reasoning_trace']) / trace_length

    print(f"\nSample complexity analysis:")
    print(f"  Chain of thought: {cot_length} characters")
    print(f"  Reasoning steps: {trace_length}")
    print(f"  Avg step length: {avg_step_length:.0f} characters")

dataset.save("hybrid_scientific_reasoning.jsonl")
```

## Advanced Configuration Patterns

### Event-Driven Generation with Monitoring

```python
import time
from datetime import datetime

def generate_with_monitoring(generator, **kwargs):
    """Generate dataset with comprehensive monitoring."""

    start_time = time.time()
    generation_log = []

    print(f"Starting generation at {datetime.now().strftime('%H:%M:%S')}")

    for event in generator.create_data_with_events(**kwargs):
        if isinstance(event, dict):
            # Log all events
            event['timestamp'] = datetime.now().isoformat()
            generation_log.append(event)

            # Real-time progress updates
            if event.get('event') == 'generation_start':
                print(f"Target: {event['total_samples']} samples")

            elif event.get('event') == 'step_start':
                print(f"Step {event['step']}/{event['total_steps']} starting...")

            elif event.get('event') == 'step_complete':
                elapsed = time.time() - start_time
                print(f"Step {event['step']}: {event['samples_generated']} samples ({elapsed:.1f}s)")

            elif event.get('event') == 'step_failed':
                print(f"Step {event['step']} failed: {event['message']}")

            elif event.get('event') == 'generation_complete':
                total_time = time.time() - start_time
                print(f"Complete: {event['total_samples']} samples in {total_time:.1f}s")

        else:
            # Final dataset
            dataset = event

            # Save generation log
            import json
            with open('generation_log.json', 'w') as f:
                json.dump(generation_log, f, indent=2)

            return dataset

# Usage
generator = DataSetGenerator(
    provider="openai",
    model_name="gpt-4o-mini",
    conversation_type="cot_freetext",
    reasoning_style="mathematical"
)

dataset = generate_with_monitoring(
    generator,
    num_steps=10,
    batch_size=1,
    topic_model=tree,
    sys_msg=False
)
```

### Dynamic Configuration Based on Domain

```python
def create_domain_specific_generator(domain: str):
    """Create optimized generator based on domain."""

    domain_configs = {
        "mathematics": {
            "reasoning_style": "mathematical",
            "temperature": 0.2,
            "model": "gpt-4o-mini",
            "instructions": "Create mathematical problems requiring step-by-step calculation.",
            "sys_prompt": "You are a mathematics tutor who shows detailed work."
        },

        "computer_science": {
            "reasoning_style": "logical",
            "temperature": 0.3,
            "model": "gpt-4o",
            "instructions": "Create programming and algorithm problems requiring systematic analysis.",
            "sys_prompt": "You are a CS instructor who explains systematic problem-solving."
        },

        "science": {
            "reasoning_style": "general",
            "temperature": 0.4,
            "model": "gpt-4o",
            "instructions": "Create scientific problems requiring hypothesis formation and testing.",
            "sys_prompt": "You are a scientist who combines intuition with rigorous analysis."
        }
    }

    config = domain_configs.get(domain, domain_configs["mathematics"])

    return DataSetGenerator(
        instructions=config["instructions"],
        generation_system_prompt=config["sys_prompt"],
        provider="openai",
        model_name=config["model"],
        temperature=config["temperature"],
        conversation_type="cot_freetext",
        reasoning_style=config["reasoning_style"]
    )

# Usage
math_generator = create_domain_specific_generator("mathematics")
cs_generator = create_domain_specific_generator("computer_science")
science_generator = create_domain_specific_generator("science")
```

### Batch Processing with Error Recovery

```python
import os
from typing import List, Dict, Any

def robust_batch_generation(
    topics: List[str],
    conversation_type: str = "cot_freetext",
    samples_per_topic: int = 5
) -> Dict[str, Any]:
    """Generate datasets for multiple topics with error recovery."""

    results = {
        "successful": [],
        "failed": [],
        "datasets": []
    }

    for i, topic in enumerate(topics):
        print(f"\nProcessing topic {i+1}/{len(topics)}: {topic}")

        try:
            # Create topic-specific tree
            tree = Tree(
                topic_prompt=f"Problems related to: {topic}",
                provider="openai",
                model_name="gpt-4o-mini",
                degree=2,
                depth=2,
                temperature=0.7
            )

            # Build tree with timeout protection
            tree_built = False
            async for event in tree.build_async():
                if event['event'] == 'build_complete':
                    tree_built = True
                    print(f"  Tree: {event['total_paths']} paths")
                    break

            if not tree_built:
                raise Exception("Tree building failed")

            # Create generator
            generator = DataSetGenerator(
                instructions=f"Create problems about {topic} requiring step-by-step reasoning.",
                generation_system_prompt="You are an expert educator creating practice problems.",
                provider="openai",
                model_name="gpt-4o-mini",
                temperature=0.3,
                conversation_type=conversation_type,
                reasoning_style="general"
            )

            # Generate dataset
            dataset = asyncio.run(generator.create_data_async(
                num_steps=samples_per_topic,
                batch_size=1,
                topic_model=tree,
                sys_msg=False
            ))

            # Save topic-specific dataset
            filename = f"dataset_{topic.replace(' ', '_').lower()}.jsonl"
            dataset.save(filename)

            results["successful"].append(topic)
            results["datasets"].append({
                "topic": topic,
                "filename": filename,
                "samples": len(dataset.samples)
            })

            print(f"  Generated {len(dataset.samples)} samples -> {filename}")

        except Exception as e:
            print(f"  Failed: {str(e)}")
            results["failed"].append({"topic": topic, "error": str(e)})

    # Summary
    print(f"\nBatch Summary:")
    print(f"  Successful: {len(results['successful'])}")
    print(f"  Failed: {len(results['failed'])}")
    print(f"  Total samples: {sum(d['samples'] for d in results['datasets'])}")

    return results

# Usage
topics = [
    "linear algebra",
    "basic calculus",
    "probability theory",
    "combinatorics"
]

results = robust_batch_generation(topics, "cot_freetext", 8)
```

### Quality Validation and Filtering

```python
from deepfabric.dataset import Dataset

def validate_and_filter_dataset(dataset: Dataset, quality_threshold: float = 0.8) -> Dataset:
    """Validate CoT samples and filter low-quality entries."""

    def quality_score(sample: dict) -> float:
        """Calculate quality score for a CoT sample."""
        score = 0.0

        # Check required fields
        if "question" in sample and len(sample["question"]) > 20:
            score += 0.2

        if "chain_of_thought" in sample:
            cot = sample["chain_of_thought"]
            # Length check
            if 50 <= len(cot) <= 1000:
                score += 0.3
            # Step indicators
            if any(word in cot.lower() for word in ["step", "first", "then", "next", "finally"]):
                score += 0.2
            # Calculation indicators
            if any(char in cot for char in "=+-×÷"):
                score += 0.1

        if "final_answer" in sample and len(sample["final_answer"]) > 0:
            score += 0.2

        # Additional checks for structured/hybrid formats
        if "reasoning_trace" in sample:
            trace = sample["reasoning_trace"]
            if isinstance(trace, list) and len(trace) >= 2:
                score += 0.2
                # Check step progression
                step_numbers = [step.get("step_number", 0) for step in trace]
                if step_numbers == list(range(1, len(step_numbers) + 1)):
                    score += 0.1

        return min(score, 1.0)

    # Score all samples
    scored_samples = []
    for sample in dataset.samples:
        score = quality_score(sample)
        scored_samples.append((sample, score))

    # Filter by threshold
    high_quality = [sample for sample, score in scored_samples if score >= quality_threshold]

    print(f"Quality filtering results:")
    print(f"  Original samples: {len(dataset.samples)}")
    print(f"  High quality (≥{quality_threshold}): {len(high_quality)}")
    print(f"  Filtered out: {len(dataset.samples) - len(high_quality)}")

    # Create new dataset with high-quality samples
    filtered_dataset = Dataset()
    filtered_dataset.samples = high_quality

    return filtered_dataset

# Usage
dataset = create_freetext_cot_dataset()
filtered_dataset = validate_and_filter_dataset(dataset, quality_threshold=0.7)
filtered_dataset.save("high_quality_cot.jsonl")
```

## Integration Patterns

### With Machine Learning Pipelines

```python
import pandas as pd
from sklearn.model_selection import train_test_split

def create_training_pipeline():
    """Create CoT dataset and prepare for ML training."""

    # Generate CoT dataset
    dataset = create_freetext_cot_dataset()

    # Convert to DataFrame for analysis
    df = pd.DataFrame(dataset.samples)

    # Basic statistics
    print("Dataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Avg question length: {df['question'].str.len().mean():.0f} chars")
    print(f"  Avg reasoning length: {df['chain_of_thought'].str.len().mean():.0f} chars")

    # Split for training/validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save splits
    train_df.to_json("train_cot.jsonl", orient="records", lines=True)
    val_df.to_json("val_cot.jsonl", orient="records", lines=True)

    return train_df, val_df

# Usage
train_data, val_data = create_training_pipeline()
```

### With LangChain Integration

```python
from langchain.schema import HumanMessage, AIMessage
from typing import List

def convert_to_langchain_format(dataset: Dataset) -> List[List[object]]:
    """Convert CoT dataset to LangChain message format."""

    langchain_conversations = []

    for sample in dataset.samples:
        if "messages" in sample:
            # Structured CoT with conversations
            messages = []
            for msg in sample["messages"]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
            langchain_conversations.append(messages)

        else:
            # Free-text or Hybrid CoT - create simple Q&A
            question = sample.get("question", "")
            reasoning = sample.get("chain_of_thought", "")
            answer = sample.get("final_answer", "")

            full_response = f"{reasoning}\n\nFinal answer: {answer}"

            conversation = [
                HumanMessage(content=question),
                AIMessage(content=full_response)
            ]
            langchain_conversations.append(conversation)

    return langchain_conversations

# Usage
dataset = create_structured_cot_dataset()
langchain_data = convert_to_langchain_format(dataset)
```

## Error Handling and Debugging

### Common Issues and Solutions

```python
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('deepfabric')

def debug_generation_issues():
    """Common debugging patterns for CoT generation."""

    try:
        generator = DataSetGenerator(
            provider="openai",
            model_name="gpt-4o-mini",
            conversation_type="cot_freetext",
            reasoning_style="mathematical"
        )

        # Test with minimal configuration
        dataset = asyncio.run(generator.create_data_async(
            num_steps=1,  # Start small
            batch_size=1,
            topic_model=None,  # Test without topics first
            sys_msg=False
        ))

        print("Basic generation works")

    except Exception as e:
        print(f"Generation failed: {e}")

        # Common fixes
        if "API key" in str(e):
            print("Set OPENAI_API_KEY environment variable")
        elif "schema" in str(e):
            print("Check conversation_type is valid CoT format")
        elif "timeout" in str(e):
            print("Increase request_timeout parameter")

# Usage
debug_generation_issues()
```

## Performance Optimization

### Async Generation for Scale

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_generation(topics: List[str], max_workers: int = 3):
    """Generate multiple datasets in parallel."""

    def generate_single_dataset(topic: str):
        tree = Tree(
            topic_prompt=f"Problems about {topic}",
            provider="openai",
            model_name="gpt-4o-mini"
        )

        async for event in tree.build_async():
            if event['event'] == 'build_complete':
                break

        generator = DataSetGenerator(
            provider="openai",
            model_name="gpt-4o-mini",
            conversation_type="cot_freetext",
            reasoning_style="general"
        )

        return asyncio.run(generator.create_data_async(num_steps=5, topic_model=tree))

    # Run in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(executor, generate_single_dataset, topic)
            for topic in topics
        ]

        results = await asyncio.gather(*tasks)

    return results

# Usage
topics = ["algebra", "geometry", "statistics"]
datasets = asyncio.run(parallel_generation(topics))
```

## Next Steps

- **YAML Configuration**: → [YAML Config Guide](yaml-config.md)
- **Math Reasoning Tutorial**: → [Math Reasoning Tutorial](../tutorials/math-reasoning.md)
- **Advanced Reasoning Styles**: → [Reasoning Styles Guide](../advanced/reasoning-styles.md)
- **Schema Reference**: → [Schema Reference](../reference/schemas.md)