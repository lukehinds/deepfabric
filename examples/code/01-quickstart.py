"""
Quickstart Example - Minimal DeepFabric usage

This example shows the simplest way to generate a dataset programmatically.
For more complex examples, see the other files in this directory or use YAML configs.

The recommended way to use DeepFabric is via YAML configuration:
  deepfabric generate examples/configs/01-quickstart.yaml
"""

import asyncio

from deepfabric import Dataset, DataSetGenerator, Tree
from deepfabric.topic_manager import handle_tree_events


def main():
    """Generate a simple Q&A dataset about Python programming."""

    print("=" * 80)
    print("DeepFabric Quickstart Example")
    print("=" * 80)
    print("\n[1/3] Creating and building topic tree...")

    # Step 1: Create and build a topic tree
    tree = Tree(
        topic_prompt="Python programming fundamentals",
        topic_system_prompt="Generate diverse programming topics",
        provider="openai",
        model_name="gpt-4o-mini",
        depth=2,  # 2 levels deep
        degree=2,  # 3 subtopics per topic
        temperature=0.8,
    )

    # Build the tree (generates the topic hierarchy)
    handle_tree_events(tree)

    # Step 2: Create a data generator and generate samples
    print("\n[2/3] Generating dataset...")

    generator = DataSetGenerator(
        instructions="Generate clear Q&A pairs about Python programming",
        generation_system_prompt="You are a helpful Python programming tutor",
        dataset_system_prompt="You are a helpful AI assistant for learning Python",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_retries=3,
        sys_msg=True,
    )

    # Generate the dataset
    async def generate():
        generator_async = generator.create_data_with_events_async(
            num_steps=2,
            batch_size=2,
            topic_model=tree,
        )

        # Collect the final dataset from the async generator
        final_dataset = None
        async for event in generator_async:
            if isinstance(event, Dataset):
                final_dataset = event
            elif isinstance(event, dict) and event.get("event") == "generation_start":
                print(f"  Generating {event['num_steps']} samples...")
            elif isinstance(event, dict) and event.get("event") == "generation_complete":
                print(f"  ✓ Generated {event['total_samples']} samples")

        return final_dataset

    dataset = asyncio.run(generate())

    # Step 3: Save the dataset
    print("\n[3/3] Saving dataset...")
    save_path = "quickstart_dataset.jsonl"

    if dataset is None:
        print("Error: Failed to generate dataset")
        return

    dataset.save(save_path)

    print("\n" + "=" * 80)
    print(f"✓ Successfully generated {len(dataset.samples)} samples")
    print(f"✓ Dataset saved to: {save_path}")
    print("=" * 80)
    print("\nFor more advanced features (formatters, uploads, etc.),")
    print("use YAML configs: deepfabric generate <config.yaml>")


if __name__ == "__main__":
    main()
