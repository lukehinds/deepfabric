"""
DeepFabric Quick Start - Programmatic Usage

Minimal example showing how to use DeepFabric as a Python library.
"""

import asyncio

from deepfabric import Dataset, DataSetGenerator, Tree
from deepfabric.formatters import FormatterRegistry


async def quick_start():
    """Generate a simple dataset programmatically."""

    # 1. Configure and generate topic tree
    tree = Tree(
        topic_prompt="Python programming basics",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        depth=2,
        degree=2,
    )

    # build_async returns an async generator; collect items into a list
    topics = [t async for t in tree.build_async()]
    print(f"✓ Generated {len(topics)} topics")

    # 2. Configure dataset generator
    generator = DataSetGenerator(
        # Required fields
        generation_system_prompt="Generate educational Python Q&A.",
        instructions="Create clear questions and detailed answers about Python programming.",
        # Optional dataset-level system prompt
        dataset_system_prompt="You are a helpful AI assistant for learning Python programming.",
        provider="openai",
        model_name="gpt-4o-mini",
        # Conversation style
        conversation_type="chain_of_thought",  # or "basic"
        reasoning_style="freetext",  # or "structured", "hybrid"
        # Optional: Agent mode for tool calling
        # agent_mode="single_turn",  # or "multi_turn"
        # available_tools=["get_weather", "calculate"],
        # Generation settings
        temperature=0.7,
        max_retries=3,
        sys_msg=True,
    )

    # 3. Generate dataset
    dataset = await generator.create_data_async(
        num_steps=2,  # Number of generation batches
        batch_size=2,  # Samples per batch
    )

    # 4. Save to file
    dataset.save("quickstart_dataset.jsonl")
    print(f"✓ Saved {len(dataset)} samples to quickstart_dataset.jsonl")

    # 5. Optional: Apply formatter
    registry = FormatterRegistry()
    formatter = registry.load_formatter(
        template="builtin://chatml",
        config={"output_format": "text"}
    )

    formatted_data = []
    for sample in dataset.samples:
        result = formatter.format(sample)
        if result:
            formatted_data.append(result)

    formatted_dataset = Dataset.from_list(formatted_data)
    formatted_dataset.save("quickstart_chatml.jsonl")
    print("✓ Saved formatted dataset to quickstart_chatml.jsonl")


if __name__ == "__main__":
    asyncio.run(quick_start())
