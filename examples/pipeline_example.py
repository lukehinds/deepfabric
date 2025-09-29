"""
DeepFabric Production Pipeline Example

Complete pipeline demonstrating:
- Quality validation and filtering
- Statistics and monitoring
- Error recovery and retry logic
- Output organization and metadata
- Performance optimization
"""

import asyncio
import json
import os
import sys

from typing import Any

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import (
    Dataset,
    DataSetGenerator,
    Tree,
)

# Uncomment to use HuggingFace integration
# from deepfabric.hf_hub import HFUploader  # noqa: ERA001


class CustomDatasetValidator:
    """Custom validator for dataset quality checks."""

    @staticmethod
    def validate_sample(sample: dict[str, Any]) -> bool:
        """Validate a single sample meets quality criteria."""
        if "messages" not in sample:
            return False

        messages = sample["messages"]
        MIN_MESSAGES = 2  # Minimum number of messages required  # noqa: N806
        if len(messages) < MIN_MESSAGES:
            return False

        # Check message roles
        roles = [msg.get("role") for msg in messages]
        if "user" not in roles or "assistant" not in roles:
            return False

        # Check minimum content length
        MIN_CONTENT_LENGTH = 10  # noqa: N806
        return all(len(msg.get("content", "")) >= MIN_CONTENT_LENGTH for msg in messages)

    @staticmethod
    def get_statistics(dataset: Dataset) -> dict[str, Any]:
        """Get statistics about the dataset."""
        total_tokens = 0
        message_counts = []

        for sample in dataset.samples:
            messages = sample.get("messages", [])
            message_counts.append(len(messages))
            for msg in messages:
                # Rough token estimation (actual tokenization would be model-specific)
                total_tokens += len(msg.get("content", "").split())

        return {
            "total_samples": len(dataset.samples),
            "avg_messages_per_sample": sum(message_counts) / len(message_counts)
            if message_counts
            else 0,
            "estimated_total_tokens": total_tokens,
            "avg_tokens_per_sample": total_tokens / len(dataset.samples) if dataset.samples else 0,
        }


def create_filtered_dataset(
    topic_tree: Tree, engine: DataSetGenerator, num_steps: int, min_quality_threshold: float = 0.8
) -> Dataset:
    """Create a dataset with quality filtering."""

    # Generate initial dataset
    print(f"Generating {num_steps} samples...")
    dataset = engine.create_data(
        num_steps=num_steps,
        batch_size=1,  # Use batch size of 1 to match num_steps exactly
        topic_model=topic_tree,
    )

    # Ensure we have a valid Dataset object
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, got {type(dataset)}")  # noqa: TRY003

    # Validate and filter samples
    validator = CustomDatasetValidator()
    valid_samples = []
    invalid_count = 0

    for sample in dataset.samples:
        if validator.validate_sample(sample):
            valid_samples.append(sample)
        else:
            invalid_count += 1

    # Create filtered dataset
    filtered_dataset = Dataset()
    filtered_dataset.samples = valid_samples

    quality_ratio = len(valid_samples) / len(dataset.samples) if dataset.samples else 0
    print(f"Quality check: {len(valid_samples)}/{len(dataset.samples)} samples passed validation")
    print(f"Quality ratio: {quality_ratio:.2%}")

    if quality_ratio < min_quality_threshold:
        print(f"⚠️  Warning: Quality ratio below threshold ({min_quality_threshold:.0%})")

    return filtered_dataset


def main():
    """Run the complete pipeline."""

    # Configuration
    ROOT_TOPIC = "Advanced Python Design Patterns and Architecture"  # noqa: N806
    PROVIDER = "openai"  # Change to your provider  # noqa: N806
    MODEL_NAME = "openai/gpt-4o"  # Change to your model  # noqa: N806
    OUTPUT_DIR = "output"  # noqa: N806

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Create and build topic tree
    print("=" * 50)
    print("Step 1: Building Topic Tree")
    print("=" * 50)

    tree = Tree(
        topic_prompt=ROOT_TOPIC,
        model_name=MODEL_NAME,
        topic_system_prompt="You are an expert in software architecture and Python.",
        degree=4,
        depth=2,
        temperature=0.7,
    )

    try:
        # Build the tree using the generator pattern
        async def _build_tree() -> None:
            async for event in tree.build_async():
                if event["event"] == "build_complete":
                    print(f" Topic tree created with {event['total_paths']} paths")

        asyncio.run(_build_tree())

        tree.save(os.path.join(OUTPUT_DIR, "python_patterns_tree.jsonl"))
    except Exception as e:
        print(f"❌ Error building tree: {e}")
        return

    # Step 2: Create data engine with custom settings
    print("\n" + "=" * 50)
    print("Step 2: Configuring Data Engine")
    print("=" * 50)

    engine = DataSetGenerator(
        instructions="""Create an advanced Python tutorial that includes:
                      - Design pattern explanation and use cases
                      - Complete implementation with type hints
                      - Unit tests using pytest
                      - Performance considerations
                      - Real-world application example
                      - Comparison with alternative approaches""",
        generation_system_prompt="""You are a senior Python developer and architect.
                       Your code follows PEP 8, uses type hints, and includes
                       comprehensive docstrings. You prioritize clean, maintainable code.""",
        model_name=MODEL_NAME,
        provider=PROVIDER,
        prompt_template=None,
        example_data=None,
        temperature=0.3,
        max_retries=5,
        default_batch_size=3,
        default_num_examples=3,
        request_timeout=30,
        sys_msg=True,
    )

    # Debug: Check what dataset_system_prompt is being used
    print(f"Dataset system prompt: {engine.dataset_system_prompt}")

    # Step 3: Generate and validate dataset
    print("\n" + "=" * 50)
    print("Step 3: Generating Dataset")
    print("=" * 50)

    try:
        dataset = create_filtered_dataset(
            topic_tree=tree, engine=engine, num_steps=15, min_quality_threshold=0.7
        )
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        return

    # Step 4: Get statistics
    print("\n" + "=" * 50)
    print("Step 4: Dataset Statistics")
    print("=" * 50)

    validator = CustomDatasetValidator()
    stats = validator.get_statistics(dataset)

    for key, value in stats.items():
        print(f"  {key}: {value:,.2f}" if isinstance(value, float) else f"  {key}: {value}")

    # Step 5: Save dataset
    print("\n" + "=" * 50)
    print("Step 5: Saving Dataset")
    print("=" * 50)

    output_path = os.path.join(OUTPUT_DIR, "python_patterns_dataset.jsonl")
    dataset.save(output_path)
    print(f" Dataset saved to {output_path}")

    # Step 6: Optional - Upload to HuggingFace Hub
    # Uncomment and configure if you want to use this
    """
    print("\n" + "=" * 50)
    print("Step 6: Uploading to HuggingFace Hub")
    print("=" * 50)

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        uploader = HFUploader(hf_token)
        result = uploader.push_to_hub(
            repo_name="username/python-patterns-dataset",
            dataset_path=output_path,
            tags=["python", "design-patterns", "synthetic", "education"]
        )
        if result["status"] == "success":
            print(f" {result['message']}")
        else:
            print(f"❌ {result['message']}")
    else:
        print("⚠️  HF_TOKEN not found, skipping upload")
    """

    # Step 7: Save metadata
    print("\n" + "=" * 50)
    print("Step 7: Saving Metadata")
    print("=" * 50)

    metadata = {
        "root_topic": ROOT_TOPIC,
        "model": MODEL_NAME,
        "tree_config": {
            "degree": tree.degree,
            "depth": tree.depth,
        },
        "statistics": stats,
        "failed_samples": len(engine.failed_samples),
    }

    with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(" Pipeline completed successfully!")
    print(f"\nOutput files in '{OUTPUT_DIR}/':")
    print("  - python_patterns_tree.jsonl (topic tree)")
    print("  - python_patterns_dataset.jsonl (training dataset)")
    print("  - metadata.json (generation metadata)")


if __name__ == "__main__":
    main()
