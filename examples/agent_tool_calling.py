#!/usr/bin/env python3
"""
Agent Tool-Calling Dataset Generation Example

This script demonstrates how to generate agent CoT datasets that teach models:
- How to reason about tool selection
- How to construct tool parameters
- How to synthesize tool results into answers

Two approaches shown:
1. Programmatic configuration (main function)
2. YAML configuration file (see agent_tool_calling.yaml)
"""

import asyncio
import os

from deepfabric.config import DeepFabricConfig
from deepfabric.dataset import Dataset
from deepfabric.generator import DataSetGenerator
from deepfabric.tree import Tree


def generate_agent_tool_calling_dataset():
    """Generate an agent tool-calling dataset programmatically."""

    # Constants
    max_content_preview = 200

    print("🚀 Starting Agent Tool-Calling Dataset Generation")
    print("=" * 60)

    # Configuration
    config = {
        "dataset_system_prompt": """You are an intelligent AI assistant with access to various tools.
When presented with a task, analyze what's needed, select appropriate tools,
execute them with proper parameters, and provide clear answers.""",

        "topic_tree": {
            "topic_prompt": "Real-world scenarios requiring tool usage",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "depth": 2,
            "degree": 3,
        },

        "data_engine": {
            "generation_system_prompt": """You are an intelligent AI assistant with access to various tools.
When presented with a task, analyze what's needed, select appropriate tools,
execute them with proper parameters, and provide clear answers.""",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.8,
            "conversation_type": "agent_cot_tools",
            "available_tools": ["web_search", "calculator", "weather"],
            "max_tools_per_query": 3,
            "max_retries": 3
        },

        "dataset": {
            "creation": {
                "num_steps": 20,
                "batch_size": 5,
                "sys_msg": True
            },
            "save_as": "agent_tool_calling_raw.jsonl",
            "formatters": [
                {
                    "name": "tool_calling_embedded",
                    "template": "builtin://tool_calling",
                    "output": "agent_tool_calling_formatted.jsonl",
                    "config": {
                        "system_prompt": "You are a function calling AI model. Use tools when needed.",
                        "include_tools_in_system": True,
                        "thinking_format": "<think>{reasoning}</think>",
                        "tool_call_format": "<tool_call>\n{tool_call}\n</tool_call>",
                        "tool_response_format": "<tool_response>\n{tool_output}\n</tool_response>"
                    }
                }
            ]
        }
    }

    try:
        # Step 1: Generate topic tree
        print("📊 Generating topic tree...")
        df_config = DeepFabricConfig(**config)
        tree_params = df_config.get_topic_tree_params()

        tree = Tree(**tree_params)

        async def _build_tree() -> None:
            async for _ in tree.build_async():
                pass

        asyncio.run(_build_tree())
        print(f"✓ Generated {len(tree.tree_paths)} topics")

        # Step 2: Generate dataset with agent tool usage
        print("\n🤖 Generating agent tool-calling samples...")
        engine_params = df_config.get_engine_params()

        generator = DataSetGenerator(**engine_params)
        result = generator.create_data(
            num_steps=df_config.dataset.creation.num_steps,
            batch_size=df_config.dataset.creation.batch_size,
            sys_msg=df_config.dataset.creation.sys_msg,
            topic_model=tree
        )
        # The create_data method returns a Dataset object
        dataset = result if isinstance(result, Dataset) else Dataset()
        print(f"✓ Generated {len(dataset.samples) if hasattr(dataset, 'samples') else 0} agent samples")

        # Step 3: Save raw format
        print("\n💾 Saving dataset...")
        raw_output = "agent_tool_calling_raw.jsonl"
        dataset.save(raw_output)
        print(f"✓ Saved raw dataset to {raw_output}")

        # Step 4: Apply tool-calling formatter
        print("\n🔧 Applying tool-calling formatter...")
        formatter_configs = df_config.get_formatter_configs()

        if formatter_configs:
            formatted_datasets = dataset.apply_formatters(formatter_configs)
            formatted_dataset = formatted_datasets["tool_calling_embedded"]

            print(f"✓ Applied formatter, created {len(formatted_dataset)} formatted samples")
            print("✓ Saved formatted dataset to agent_tool_calling_formatted.jsonl")

            # Step 5: Show example formatted output
            print("\n📋 Sample formatted output:")
            print("-" * 40)
            if len(formatted_dataset) > 0:
                example = formatted_dataset[0]
                for i, message in enumerate(example["messages"]):
                    role = message["role"]
                    content = message["content"][:max_content_preview] + "..." if len(message["content"]) > max_content_preview else message["content"]
                    print(f"Message {i+1} ({role}): {content}")
                    print()

        print("Dataset generation complete!")
        print("Files created:")
        print(f"   - {raw_output} (raw agent reasoning)")
        print("   - agent_tool_calling_formatted.jsonl (embedded execution format)")

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise


def generate_from_yaml_config():
    """Generate using YAML configuration file."""
    config_path = os.path.join(os.path.dirname(__file__), "agent_tool_calling.yaml")

    if not os.path.exists(config_path):
        print(f"YAML config not found: {config_path}")
        return

    print("\n" + "="*60)
    print("🔧 YAML Configuration Approach")
    print("="*60)

    # Load from YAML
    config = DeepFabricConfig.from_yaml(config_path)

    # Generate topics
    tree_params = config.get_topic_tree_params()
    tree = Tree(**tree_params)

    async def _build_tree() -> None:
        async for _ in tree.build_async():
            pass

    asyncio.run(_build_tree())
    print(f"✓ Generated {len(tree.tree_paths)} topics from YAML config")

    # Generate dataset
    engine_params = config.get_engine_params()

    generator = DataSetGenerator(**engine_params)
    result = generator.create_data(
        num_steps=config.dataset.creation.num_steps,
        batch_size=config.dataset.creation.batch_size,
        sys_msg=config.dataset.creation.sys_msg,
        topic_model=tree
    )
    # The create_data method returns a Dataset object
    dataset = result if isinstance(result, Dataset) else Dataset()
    dataset.save("agent_yaml_dataset.jsonl")
    print(f"✓ Generated {len(dataset.samples) if hasattr(dataset, 'samples') else 0} samples and saved to agent_yaml_dataset.jsonl")


if __name__ == "__main__":
    print("Running Agent Tool-Calling Dataset Generation Examples")
    print("="*60)

    # Run programmatic approach
    generate_agent_tool_calling_dataset()

    # Run YAML approach
    generate_from_yaml_config()
