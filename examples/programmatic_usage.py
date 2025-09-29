"""
DeepFabric Programmatic Usage Examples

Demonstrates all programmatic patterns:
- Tree vs Graph comparison
- Custom topic creation
- Progress monitoring and event handling
- Loading/saving existing models
- Error handling best practices
"""

import asyncio
import json
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator
from deepfabric.dataset import Dataset
from deepfabric.graph import Graph
from deepfabric.tree import Tree


def example_tree_vs_graph():
    """Compare Tree and Graph generation side-by-side."""
    print("=" * 60)
    print("Example 1: Tree vs Graph Comparison")
    print("=" * 60)

    # Tree: Hierarchical structure
    print("\n🌳 Building Tree (hierarchical)...")
    tree = Tree(
        topic_prompt="Web Development Technologies",
        provider="ollama",
        model_name="qwen3:8b",
        degree=2,
        depth=2,
        temperature=0.7,
    )

    tree_events: list[dict] = []

    async def _build_tree() -> None:
        async for event in tree.build_async():
            tree_events.append(event)
            if event["event"] == "build_complete":
                print(f"   Tree: {event['total_paths']} paths")

    asyncio.run(_build_tree())

    # Graph: Interconnected structure
    print("\n🕸️  Building Graph (interconnected)...")
    graph = Graph(
        topic_prompt="Web Development Technologies",
        provider="ollama",
        model_name="qwen3:8b",
        degree=2,
        depth=2,
        temperature=0.7,
    )

    graph_events: list[dict] = []

    async def _build_graph() -> None:
        async for event in graph.build_async():
            graph_events.append(event)
            if event["event"] == "build_complete":
                print(f"   Graph: {event['nodes_count']} nodes")

    asyncio.run(_build_graph())

    # Compare paths
    tree_paths = tree.get_all_paths()
    graph_paths = graph.get_all_paths()

    print("\n📊 Comparison:")
    print(f"   Tree paths:  {len(tree_paths)}")
    print(f"   Graph paths: {len(graph_paths)}")
    print(f"   Graph creates {'more' if len(graph_paths) > len(tree_paths) else 'fewer'} paths due to cross-connections")

    # Save both
    tree.save("web_dev_tree.jsonl")
    graph.save("web_dev_graph.json")
    print("   💾 Saved: web_dev_tree.jsonl, web_dev_graph.json")

    return tree, graph


def example_custom_topics():
    """Demonstrate manual topic creation."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Topic Creation")
    print("=" * 60)

    # Create tree with manual topics
    tree = Tree(
        topic_prompt="Data Science",
        provider="ollama",
        model_name="qwen3:8b",
        degree=3,
        depth=2,
        temperature=0.7,
    )

    # Define custom topic paths
    custom_topics = [
        {"path": ["Data Science", "Machine Learning", "Supervised Learning"]},
        {"path": ["Data Science", "Machine Learning", "Unsupervised Learning"]},
        {"path": ["Data Science", "Deep Learning", "Neural Networks"]},
        {"path": ["Data Science", "Data Engineering", "ETL Pipelines"]},
        {"path": ["Data Science", "Statistics", "Hypothesis Testing"]},
    ]

    # Load custom topics
    tree.from_dict_list(custom_topics)
    tree.save("custom_data_science_tree.jsonl")

    print(f"📝 Created tree with {len(custom_topics)} custom topics:")
    for topic in custom_topics:
        print(f"   - {' → '.join(topic['path'])}")

    return tree


def example_progress_monitoring():
    """Demonstrate detailed progress monitoring."""
    print("\n" + "=" * 60)
    print("Example 3: Progress Monitoring & Event Handling")
    print("=" * 60)

    tree = Tree(
        topic_prompt="Software Architecture Patterns",
        provider="ollama",
        model_name="qwen3:8b",
        degree=3,
        depth=2,
        temperature=0.7,
    )

    # Track all event types
    events_by_type = {}
    failed_count = 0

    print("🔄 Building with detailed monitoring...")

    async def _monitor_build() -> None:
        nonlocal failed_count
        async for event in tree.build_async():
            event_type = event["event"]

            # Track events
            events_by_type.setdefault(event_type, []).append(event)

            # Handle different event types
            if event_type == "build_start":
                print(f"   🚀 Started: {event['model_name']}, depth={event['depth']}")
            elif event_type == "subtopics_generated":
                status = "✅" if event["success"] else "❌"
                if not event["success"]:
                    failed_count += 1
                print(f"   {status} Generated {event['count']} subtopics")
            elif event_type == "build_complete":
                print(
                    f"   🎉 Complete: {event['total_paths']} paths, {event['failed_generations']} failures"
                )

    asyncio.run(_monitor_build())

    # Print event summary
    print("\n📈 Event Summary:")
    for event_type, events in events_by_type.items():
        print(f"   {event_type}: {len(events)} occurrences")

    return tree


def example_load_existing():
    """Demonstrate loading and extending existing models."""
    print("\n" + "=" * 60)
    print("Example 4: Loading & Extending Existing Models")
    print("=" * 60)

    # Load the graph we created earlier
    try:
        print("📂 Loading existing graph...")
        graph_params = {
            "topic_prompt": "Web Development Technologies",
            "provider": "ollama",
            "model_name": "qwen3:8b",
            "temperature": 0.7,
            "degree": 2,
            "depth": 2,
        }

        loaded_graph = Graph.from_json("web_dev_graph.json", graph_params)
        print(f"   ✅ Loaded graph with {len(loaded_graph.nodes)} nodes")

        # Extend the graph
        print("🔧 Extending graph...")
        new_node = loaded_graph.add_node("Progressive Web Apps")
        loaded_graph.add_edge(loaded_graph.root.id, new_node.id)

        # Save extended version
        loaded_graph.save("web_dev_graph_extended.json")
        print(f"   💾 Extended graph saved with {len(loaded_graph.nodes)} nodes")

    except FileNotFoundError:
        print("   ⚠️  Graph file not found, skipping load example")
        return None
    else:
        return loaded_graph


def example_error_handling():
    """Demonstrate proper error handling patterns."""
    print("\n" + "=" * 60)
    print("Example 5: Error Handling Best Practices")
    print("=" * 60)

    try:
        # Use a tree with limited paths
        tree = Tree(
            topic_prompt="Small Topic",
            provider="ollama",
            model_name="qwen3:8b",
            degree=1,  # Very small tree
            depth=1,
            temperature=0.7,
        )

        async def _build_small_tree() -> None:
            async for event in tree.build_async():
                if event["event"] == "build_complete":
                    total_paths = event["total_paths"]
                    print(f"🌳 Built tree with {total_paths} paths")

        asyncio.run(_build_small_tree())

        # Create engine
        engine = DataSetGenerator(
            instructions="Create examples",
            generation_system_prompt="You are a helpful assistant.",
            provider="ollama",
            model_name="qwen3:8b",
            temperature=0.7,
        )

        # Try to generate more samples than paths available
        print("⚠️  Attempting to generate more samples than available paths...")
        try:
            dataset = engine.create_data(
                num_steps=10,  # This will likely exceed available paths
                batch_size=1,
                topic_model=tree,
            )

            if dataset is None:
                print("   ❌ Dataset generation returned None")
                return

            if not isinstance(dataset, Dataset):
                print(f"   ❌ Expected Dataset, got {type(dataset)}")
                return

            print(f"   ✅ Successfully generated {len(dataset.samples)} samples")

        except Exception as e:
            print(f"   ❌ Generation failed as expected: {e}")
            print("   💡 This demonstrates path validation working correctly")

    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


def example_dataset_generation():
    """Generate dataset using one of our topic models."""
    print("\n" + "=" * 60)
    print("Example 6: Dataset Generation")
    print("=" * 60)

    # Load custom topics tree
    try:
        with open("custom_data_science_tree.jsonl") as f:
            topic_data = [json.loads(line) for line in f]

        tree = Tree(
            topic_prompt="Data Science",
            provider="ollama",
            model_name="qwen3:8b",
            degree=3,
            depth=2,
            temperature=0.7,
        )
        tree.from_dict_list(topic_data)

        # Create engine
        engine = DataSetGenerator(
            instructions="Create comprehensive tutorials with code examples",
            generation_system_prompt="You are a data science educator creating practical tutorials.",
            provider="ollama",
            model_name="qwen3:8b",
            temperature=0.3,
        )

        # Generate dataset
        print("🔄 Generating dataset...")
        dataset = engine.create_data(
            num_steps=3,  # Generate 3 examples
            batch_size=1,
            topic_model=tree,
        )

        if dataset and isinstance(dataset, Dataset):
            dataset.save("data_science_tutorials.jsonl")
            print(f"✅ Generated {len(dataset.samples)} tutorials")
            print("📁 Saved to data_science_tutorials.jsonl")
        else:
            print("❌ Dataset generation failed")

    except FileNotFoundError:
        print("⚠️  Custom topics file not found, skipping dataset generation")


def main():
    """Run all programmatic usage examples."""
    print("🚀 DeepFabric Programmatic Usage Examples")
    print("=" * 60)

    try:
        # Run all examples
        tree, graph = example_tree_vs_graph()
        _custom_tree = example_custom_topics()
        _progress_tree = example_progress_monitoring()
        _loaded_graph = example_load_existing()
        example_error_handling()
        example_dataset_generation()

        print("\n" + "=" * 60)
        print("🎉 All examples completed successfully!")
        print("=" * 60)
        print("📁 Files created:")
        print("   - web_dev_tree.jsonl")
        print("   - web_dev_graph.json")
        print("   - custom_data_science_tree.jsonl")
        print("   - web_dev_graph_extended.json (if load example ran)")
        print("   - data_science_tutorials.jsonl (if dataset example ran)")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
