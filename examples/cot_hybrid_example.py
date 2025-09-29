"""
DeepFabric Chain of Thought Example - Hybrid Format
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
    """Generate a hybrid Chain of Thought reasoning dataset."""

    # Step 1: Create a topic tree for science problems
    tree = Tree(
        topic_prompt="Scientific reasoning, physics problems, and analytical thinking challenges",
        provider="openai",
        model_name="gpt-4o",
        degree=2,
        depth=3,
        temperature=0.5,
    )

    # Build the tree
    print("Building science reasoning topic tree...")

    async def _build_tree() -> None:
        async for event in tree.build_async():
            if event["event"] == "build_complete":
                print(f"Tree built with {event['total_paths']} topic paths")

    asyncio.run(_build_tree())

    # Step 2: Create hybrid CoT dataset generator
    engine = DataSetGenerator(
        instructions="Create scientific and analytical problems that demonstrate both natural reasoning and systematic problem-solving.",
        generation_system_prompt="You are a science educator who excels at explaining problems through both intuitive reasoning and systematic analysis.",
        provider="openai",
        model_name="gpt-4o",
        temperature=0.2,
        conversation_type="cot_hybrid",  # Use hybrid CoT format
        reasoning_style="general",
    )

    # Step 3: Generate the hybrid CoT dataset
    print("Generating hybrid CoT dataset...")
    dataset = engine.create_data(
        num_steps=4,      # Generate 4 examples
        batch_size=1,     # One at a time
        topic_model=tree, # Use our topic tree
        sys_msg=False,    # Hybrid format doesn't use conversation messages
    )

    # Validate dataset was created
    if dataset is None:
        msg = "Dataset generation failed"
        raise RuntimeError(msg)
    if not isinstance(dataset, Dataset):
        msg = f"Expected Dataset object, got {type(dataset)}"
        raise TypeError(msg)

    # Step 4: Save the dataset
    dataset.save("science_reasoning_hybrid_cot.jsonl")

    print(f"Generated {len(dataset.samples)} hybrid CoT training examples")
    print("Saved to science_reasoning_hybrid_cot.jsonl")

    # Show a sample
    if dataset.samples:
        print("\nSample hybrid CoT entry:")
        sample = dataset.samples[0]
        print(f"Question: {sample.get('question', 'N/A')}")
        print(f"Natural reasoning: {sample.get('chain_of_thought', 'N/A')[:100]}...")
        print(f"Structured steps: {len(sample.get('reasoning_trace', []))}")
        print(f"Final answer: {sample.get('final_answer', 'N/A')}")


if __name__ == "__main__":
    main()
