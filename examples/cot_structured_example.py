"""
DeepFabric Chain of Thought Example - Structured Format

Generates conversation-based datasets with structured reasoning traces.
"""

import asyncio
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator
from deepfabric.dataset import Dataset
from deepfabric.graph import Graph


def main():
    """Generate a structured Chain of Thought reasoning dataset."""

    # Step 1: Create a topic graph for CS problems
    graph = Graph(
        topic_prompt="Computer science algorithms, data structures, and logical problem solving",
        provider="openai",
        model_name="gpt-4o-mini",
        degree=2,
        depth=2,
        temperature=0.6,
    )

    # Build the graph
    print("Building CS reasoning topic graph...")

    async def _build_graph() -> None:
        async for event in graph.build_async():
            if event["event"] == "build_complete":
                print(f"✅ Graph built with {event['nodes_count']} nodes")

    asyncio.run(_build_graph())

    # Step 2: Create structured CoT dataset generator
    engine = DataSetGenerator(
        instructions="Create programming and algorithmic problems that require systematic thinking with clear reasoning steps.",
        generation_system_prompt="You are a computer science instructor creating educational dialogues with systematic problem-solving approaches.",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.4,
        conversation_type="cot_structured",  # Use structured CoT format
        reasoning_style="logical",
    )

    # Step 3: Generate the structured CoT dataset
    print("Generating structured CoT dataset...")
    dataset = engine.create_data(
        num_steps=4,      # Generate 4 examples
        batch_size=1,     # One at a time
        topic_model=graph, # Use our topic graph
        sys_msg=True,     # Include system messages in conversations
    )

    # Validate dataset was created
    if dataset is None:
        msg = "Dataset generation failed"
        raise RuntimeError(msg)
    if not isinstance(dataset, Dataset):
        msg = f"Expected Dataset object, got {type(dataset)}"
        raise TypeError(msg)

    # Step 4: Save the dataset
    dataset.save("cs_reasoning_structured_cot.jsonl")

    print(f"Generated {len(dataset.samples)} structured CoT training examples")
    print("Saved to cs_reasoning_structured_cot.jsonl")

    # Show a sample
    if dataset.samples:
        print("\n Sample structured CoT entry:")
        sample = dataset.samples[0]
        print(f"Messages: {len(sample.get('messages', []))} conversation turns")
        print(f"Reasoning steps: {len(sample.get('reasoning_trace', []))}")
        print(f"Final answer: {sample.get('final_answer', 'N/A')[:100]}...")


if __name__ == "__main__":
    main()
