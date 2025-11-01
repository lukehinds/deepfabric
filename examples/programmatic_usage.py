"""
DeepFabric Programmatic API Examples

This file demonstrates how to use DeepFabric as a Python library
instead of via YAML configs and CLI.
"""

import asyncio

from pathlib import Path

from deepfabric import (
    Dataset,
    DataSetGenerator,
    DataSetGeneratorConfig,
    Tree,
    TreeConfig,
)
from deepfabric.formatters import FormatterRegistry

# ==============================================================================
# EXAMPLE 1: Basic Q&A Dataset Generation
# ==============================================================================


async def example_basic_qa():
    """Generate a simple Q&A dataset programmatically."""
    print("\n=== Example 1: Basic Q&A ===\n")

    # Step 1: Generate topic tree
    tree = Tree(
        topic_prompt="Python programming fundamentals",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        depth=2,
        degree=2,
    )

    topics = [t async for t in tree.build_async()]
    print(f"✓ Generated {len(topics)} topics")

    # Step 2: Generate dataset
    generator = DataSetGenerator(
        generation_system_prompt="Generate clear, educational Q&A pairs about Python.",
        dataset_system_prompt="You are a helpful AI assistant for learning Python programming.",
        instructions="Create diverse questions and detailed answers.",
        conversation_type="basic",  # Simple Q&A
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_retries=3,
        sys_msg=True,
    )

    # Generate 4 samples (2 steps × 2 batch_size)
    dataset = await generator.create_data_async(
        num_steps=2,
        batch_size=2,
    )

    # Save to file
    output_path = Path("programmatic_basic_qa.jsonl")
    dataset.save(str(output_path))
    print(f"✓ Saved {len(dataset)} samples to {output_path}")

    return dataset


# ==============================================================================
# EXAMPLE 2: Chain-of-Thought with Freetext Reasoning
# ==============================================================================


async def example_cot_freetext():
    """Generate chain-of-thought dataset with natural language reasoning."""
    print("\n=== Example 2: Chain-of-Thought (Freetext) ===\n")

    # Generate topics
    tree_config = TreeConfig(
        topic_prompt="Math word problems requiring reasoning",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        depth=2,
        degree=2,
    )

    tree = Tree(tree_config)
    topics = await tree.generate()

    # Generate dataset with chain-of-thought
    generator_config = DataSetGeneratorConfig(
        generation_system_prompt="Generate math problems with clear reasoning.",
        dataset_system_prompt="You are a helpful AI assistant for learning math.",
        instructions="Create problems showing step-by-step thinking.",
        conversation_type="chain_of_thought",  # Enable reasoning
        reasoning_style="freetext",  # Natural language reasoning
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_retries=3,
        sys_msg=True,
    )

    generator = DataSetGenerator(generator_config)
    dataset = await generator.generate(topics=topics, num_steps=2, batch_size=2)

    output_path = Path("programmatic_cot_freetext.jsonl")
    dataset.save(str(output_path))
    print(f"Saved {len(dataset)} samples to {output_path}")

    return dataset


# ==============================================================================
# EXAMPLE 3: Chain-of-Thought with Structured Reasoning
# ==============================================================================


async def example_cot_structured():
    """Generate chain-of-thought with explicit step-by-step traces."""
    print("\n=== Example 3: Chain-of-Thought (Structured) ===\n")

    tree_config = TreeConfig(
        topic_prompt="Logical reasoning and problem-solving",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        depth=2,
        degree=2,
    )

    tree = Tree(tree_config)
    topics = await tree.generate()

    generator_config = DataSetGeneratorConfig(
        generation_system_prompt="Generate problems with structured reasoning.",
        dataset_system_prompt="You are a helpful AI assistant for logical reasoning.",
        instructions="Create problems with clear step-by-step solutions.",
        conversation_type="chain_of_thought",
        reasoning_style="structured",  # Explicit step-by-step
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_retries=3,
        sys_msg=True,
    )

    generator = DataSetGenerator(generator_config)
    dataset = await generator.generate(topics=topics, num_steps=2, batch_size=2)

    output_path = Path("programmatic_cot_structured.jsonl")
    dataset.save(str(output_path))
    print(f"Saved {len(dataset)} samples to {output_path}")

    return dataset


# ==============================================================================
# EXAMPLE 4: Single-Turn Agent with Tools
# ==============================================================================


async def example_single_turn_agent():
    """Generate single-turn agent conversations with tool calling."""
    print("\n=== Example 4: Single-Turn Agent ===\n")

    tree_config = TreeConfig(
        topic_prompt="Tasks requiring weather, calculations, and web search",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        depth=2,
        degree=2,
    )

    tree = Tree(tree_config)
    topics = await tree.generate()

    generator_config = DataSetGeneratorConfig(
        generation_system_prompt="Generate tool-using agent examples.",
        instructions="Create realistic scenarios requiring tool usage.",
        dataset_system_prompt="You are a helpful AI assistant for task completion.",
        conversation_type="chain_of_thought",
        reasoning_style="structured",
        agent_mode="single_turn",  # Single-turn agent
        available_tools=[  # Built-in tools
            "get_weather",
            "calculate",
            "search_web",
            "get_time",
        ],
        max_tools_per_query=3,
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        max_retries=3,
        sys_msg=True,
    )

    generator = DataSetGenerator(generator_config)
    dataset = await generator.generate(topics=topics, num_steps=2, batch_size=2)

    output_path = Path("programmatic_single_turn_agent.jsonl")
    dataset.save(str(output_path))
    print(f"Saved {len(dataset)} samples to {output_path}")

    return dataset


# ==============================================================================
# EXAMPLE 5: Multi-Turn Agent Conversations
# ==============================================================================


async def example_multi_turn_agent():
    """Generate multi-turn agent conversations."""
    print("\n=== Example 5: Multi-Turn Agent ===\n")

    tree_config = TreeConfig(
        topic_prompt="Complex multi-step tasks requiring multiple tools",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        depth=2,
        degree=2,
    )

    tree = Tree(tree_config)
    topics = await tree.generate()

    generator_config = DataSetGeneratorConfig(
        generation_system_prompt="Generate multi-turn agent conversations.",
        instructions="Create complex scenarios with multiple interaction turns.",
        dataset_system_prompt="You are a helpful AI assistant for complex tasks.",
        conversation_type="chain_of_thought",
        reasoning_style="hybrid",  # Both freetext and structured
        agent_mode="multi_turn",  # Multi-turn agent
        min_turns=2,  # Minimum conversation turns
        max_turns=6,  # Maximum conversation turns
        available_tools=[
            "get_weather",
            "calculate",
            "search_web",
            "get_time",
        ],
        max_tools_per_query=5,
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        max_retries=3,
        sys_msg=True,
    )

    generator = DataSetGenerator(generator_config)
    dataset = await generator.generate(topics=topics, num_steps=2, batch_size=1)

    output_path = Path("programmatic_multi_turn_agent.jsonl")
    dataset.save(str(output_path))
    print(f"Saved {len(dataset)} samples to {output_path}")

    return dataset


# ==============================================================================
# EXAMPLE 6: Custom Tools
# ==============================================================================


async def example_custom_tools():
    """Generate agent conversations with custom tools."""
    print("\n=== Example 6: Custom Tools ===\n")

    tree_config = TreeConfig(
        topic_prompt="Database queries and data analysis tasks",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        depth=2,
        degree=2,
    )

    tree = Tree(tree_config)
    topics = await tree.generate()

    # Define custom tools
    custom_tools = [
        {
            "name": "database_query",
            "description": "Execute a SQL query on a database",
            "parameters": [
                {
                    "name": "query",
                    "type": "str",
                    "description": "SQL query to execute",
                    "required": True,
                },
                {
                    "name": "database",
                    "type": "str",
                    "description": "Database name",
                    "required": True,
                },
            ],
            "returns": "Query results as JSON",
        },
        {
            "name": "analyze_csv",
            "description": "Analyze a CSV file and return statistics",
            "parameters": [
                {
                    "name": "file_path",
                    "type": "str",
                    "description": "Path to CSV file",
                    "required": True,
                },
                {
                    "name": "operations",
                    "type": "list",
                    "description": "Statistical operations: mean, median, std, etc.",
                    "required": False,
                },
            ],
            "returns": "Statistical analysis results",
        },
    ]

    generator_config = DataSetGeneratorConfig(
        generation_system_prompt="Generate data analysis agent examples.",
        instructions="Create scenarios using database and CSV tools.",
        dataset_system_prompt="You are a helpful AI assistant for data analysis tasks.",
        conversation_type="chain_of_thought",
        reasoning_style="structured",
        agent_mode="single_turn",
        available_tools=["calculate"],  # Mix built-in and custom
        custom_tools=custom_tools,  # Add custom tools
        max_tools_per_query=3,
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.8,
        max_retries=3,
        sys_msg=True,
    )

    generator = DataSetGenerator(generator_config)
    dataset = await generator.generate(topics=topics, num_steps=2, batch_size=2)

    output_path = Path("programmatic_custom_tools.jsonl")
    dataset.save(str(output_path))
    print(f"Saved {len(dataset)} samples to {output_path}")

    return dataset


# ==============================================================================
# EXAMPLE 7: Applying Formatters Programmatically
# ==============================================================================


async def example_formatters():
    """Generate dataset and apply multiple formatters."""
    print("\n=== Example 7: Formatters ===\n")

    # Generate a basic dataset
    tree_config = TreeConfig(
        topic_prompt="Customer support scenarios",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        depth=2,
        degree=2,
    )

    tree = Tree(tree_config)
    topics = await tree.generate()

    generator_config = DataSetGeneratorConfig(
        generation_system_prompt="Generate customer support conversations.",
        dataset_system_prompt="You are a helpful AI assistant for customer support.",
        instructions="Create realistic support scenarios.",
        conversation_type="basic",
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_retries=3,
        sys_msg=True,
    )

    generator = DataSetGenerator(generator_config)
    dataset = await generator.generate(topics=topics, num_steps=2, batch_size=2)

    # Save raw dataset
    raw_path = Path("programmatic_raw.jsonl")
    dataset.save(str(raw_path))
    print(f"Saved raw dataset to {raw_path}")

    registry = FormatterRegistry()

    # Format to ChatML (text mode)
    chatml_formatter = registry.get_formatter(
        "chatml",
        config={
            "output_format": "text",
            "normalize_whitespace": True,
        },
    )

    chatml_output = []
    for sample in dataset.samples:
        formatted = chatml_formatter.format(sample)
        if formatted:
            chatml_output.append(formatted)

    # Save ChatML formatted
    chatml_dataset = Dataset.from_list(chatml_output)
    chatml_path = Path("programmatic_chatml.jsonl")
    chatml_dataset.save(str(chatml_path))
    print(f"Saved ChatML formatted to {chatml_path}")

    # Format to Alpaca
    alpaca_formatter = registry.load_formatter(
        template="builtin://alpaca",
        config={}
    )

    alpaca_output = []
    for sample in dataset.samples:
        formatted = alpaca_formatter.format(sample)
        if formatted:
            alpaca_output.append(formatted)

    alpaca_dataset = Dataset.from_list(alpaca_output)
    alpaca_path = Path("programmatic_alpaca.jsonl")
    alpaca_dataset.save(str(alpaca_path))
    print(f"Saved Alpaca formatted to {alpaca_path}")

    return dataset


# ==============================================================================
# Main Entry Point
# ==============================================================================


async def main():
    """Run all examples."""
    print("=" * 70)
    print("DeepFabric Programmatic API Examples")
    print("=" * 70)

    # Run examples (comment out any you don't want to run)
    await example_basic_qa()
    await example_cot_freetext()
    await example_cot_structured()
    await example_single_turn_agent()
    await example_multi_turn_agent()
    await example_custom_tools()
    await example_formatters()

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
