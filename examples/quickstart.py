"""
DeepFabric Quickstart Example

The simplest possible example to get started with DeepFabric.
Creates a small synthetic dataset about Python programming in ~20 lines.
"""

import asyncio
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator
from deepfabric.dataset import Dataset
from deepfabric.tree import Tree


def main():
    """Generate a simple dataset about Python programming."""

    # Step 1: Create a topic tree
    tree = Tree(
        topic_prompt="Python programming fundamentals",
        provider="ollama",
        model_name="qwen3:8b",  # Change to your preferred model
        degree=2,  # 2 branches per level
        depth=2,   # 2 levels deep
        temperature=0.7,
    )

    # Build the tree
    print("Building topic tree...")

    async def _build_tree() -> None:
        async for event in tree.build_async():
            if event["event"] == "build_complete":
                print(f"✅ Tree built with {event['total_paths']} topic paths")

    asyncio.run(_build_tree())

    # Step 2: Create dataset generator
    engine = DataSetGenerator(
        instructions="Create a Python code example with explanation",
        generation_system_prompt="You are a Python programming instructor.",
        provider="ollama",
        model_name="qwen3:8b",
        temperature=0.7,
    )

    # Step 3: Generate the dataset
    print("Generating dataset...")
    dataset = engine.create_data(
        num_steps=4,      # Generate 4 examples
        batch_size=1,     # One at a time
        topic_model=tree, # Use our topic tree
    )

    # Validate dataset was created
    if dataset is None:
        msg = "Dataset generation failed"
        raise RuntimeError(msg)
    if not isinstance(dataset, Dataset):
        msg = f"Expected Dataset object, got {type(dataset)}"
        raise TypeError(msg)

    # Step 4: Save the dataset
    dataset.save("python_examples.jsonl")

    print(f"✅ Generated {len(dataset.samples)} training examples")
    print("📁 Saved to python_examples.jsonl")

if __name__ == "__main__":
    main()
