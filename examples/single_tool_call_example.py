#!/usr/bin/env python3
"""
Single Tool Call Format Example

This example demonstrates how to use the single_tool_call formatter
to generate datasets where each tool call is in its own message exchange.
"""

import asyncio
import json

from deepfabric.config import DeepFabricConfig
from deepfabric.generator import DataSetGenerator
from deepfabric.tree import Tree


def demonstrate_single_tool_call_format():
    """Generate a dataset with single tool call format."""

    print("🚀 Generating Dataset with Single Tool Call Format")
    print("=" * 60)

    # Configuration
    config = {
        "dataset_system_prompt": """You are an intelligent AI assistant with access to various tools.
When presented with a task, analyze what's needed, select appropriate tools,
execute them with proper parameters, and provide clear answers.""",

        "topic_tree": {
            "topic_prompt": "Tasks requiring multiple tool calls",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.7,
            "depth": 2,
            "degree": 3,
        },

        "data_engine": {
            "generation_system_prompt": """You are an AI assistant that uses tools to help users.""",
            "provider": "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.8,
            "conversation_type": "agent_cot_tools",
            "available_tools": ["get_weather", "get_time", "calculator", "web_search"],
            "max_tools_per_query": 2,
            "max_retries": 3
        },

        "dataset": {
            "creation": {
                "num_steps": 5,
                "batch_size": 2,
                "sys_msg": True
            },
            "save_as": "raw_dataset.jsonl",
            "formatters": [
                {
                    "name": "single_tool_format",
                    "template": "builtin://single_tool_call",
                    "output": "single_tool_call_dataset.jsonl",
                    "config": {
                        "system_prompt": "You are a helpful assistant with access to the following functions. Use them if required:",
                        "include_tools_in_system": True,
                        "include_reasoning_prefix": True,
                        "reasoning_prefix_template": "I'll {action} for you.",
                        "tool_call_format": "<tool_call>\n{tool_call}\n</tool_call>",
                        "tool_response_as_json": True
                    }
                }
            ]
        }
    }

    try:
        # Step 1: Generate topic tree
        print("📊 Generating topics...")
        df_config = DeepFabricConfig(**config)
        tree_params = df_config.get_topic_tree_params()

        tree = Tree(**tree_params)

        async def _build_tree() -> None:
            async for _ in tree.build_async():
                pass

        asyncio.run(_build_tree())
        print(f"✓ Generated {len(tree.tree_paths)} topics")

        # Step 2: Generate dataset
        print("\n🤖 Generating samples...")
        engine_params = df_config.get_engine_params()

        generator = DataSetGenerator(**engine_params)
        dataset = generator.create_data(
            num_steps=df_config.dataset.creation.num_steps,
            batch_size=df_config.dataset.creation.batch_size,
            sys_msg=df_config.dataset.creation.sys_msg,
            topic_model=tree
        )

        # Save raw dataset
        dataset.save("raw_dataset.jsonl")
        print(f"✓ Generated {len(dataset.samples) if hasattr(dataset, 'samples') else 0} samples")

        # Step 3: Apply formatter
        print("\n🔧 Applying single tool call formatter...")
        formatter_configs = df_config.get_formatter_configs()

        if formatter_configs:
            formatted_datasets = dataset.apply_formatters(formatter_configs)
            formatted_dataset = formatted_datasets["single_tool_format"]

            print(f"✓ Formatted {len(formatted_dataset)} samples")
            print("✓ Saved to single_tool_call_dataset.jsonl")

            # Show example output
            if len(formatted_dataset) > 0:
                print("\n📋 Example formatted output:")
                print("-" * 40)
                example = formatted_dataset[0]
                print(json.dumps(example, indent=2))

                print("\n📄 Message flow:")
                for i, message in enumerate(example["messages"]):
                    role = message["role"]
                    content = message["content"]
                    max_preview_length = 100
                    if len(content) > max_preview_length:
                        content = content[:max_preview_length] + "..."
                    print(f"{i+1}. [{role}]: {content}")

        print("\n✅ Dataset generation complete!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    demonstrate_single_tool_call_format()
