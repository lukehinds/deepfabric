"""
Example showing how to load and reuse an existing topic graph.
This is useful when you want to generate more data from a previously created graph.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from promptwright import DataEngine, EngineArguments
from promptwright.topic_graph import TopicGraph, TopicGraphArguments

# Load an existing graph from a JSON file
# First, let's create a graph to demonstrate (in practice, you'd already have this)
print("Creating initial graph for demonstration...")
initial_graph = TopicGraph(
    args=TopicGraphArguments(
        root_prompt="Web Development Technologies",
        model_name="ollama/llama3",
        temperature=0.6,
        graph_degree=2,
        graph_depth=2,
    )
)

# Build and save the initial graph
initial_graph.build()
initial_graph.save("web_dev_graph.json")
print(f"Initial graph created with {len(initial_graph.nodes)} nodes\n")

# Now demonstrate loading the saved graph
print("Loading existing graph from file...")
loaded_graph = TopicGraph.from_json(
    "web_dev_graph.json",
    args=TopicGraphArguments(
        root_prompt="Web Development Technologies",  # Must match original
        model_name="ollama/llama3",  # Can be different from original
        temperature=0.6,
        graph_degree=2,
        graph_depth=2,
    )
)

print(f"Loaded graph with {len(loaded_graph.nodes)} nodes")
print(f"Root topic: {loaded_graph.root.topic}")

# Print all paths in the loaded graph
print("\nPaths in loaded graph:")
for i, path in enumerate(loaded_graph.get_all_paths(), 1):
    print(f"  {i}. {' → '.join(path)}")

# Create a new data engine with different parameters
engine = DataEngine(
    args=EngineArguments(
        instructions="""Create a practical coding tutorial that includes:
                       - A real-world problem to solve
                       - Complete, runnable code example
                       - Explanation of key concepts
                       - Best practices and optimization tips
                       - Common mistakes to avoid""",
        system_prompt="You are a web development expert creating practical tutorials.",
        model_name="ollama/llama3",
        prompt_template=None,
        example_data=None,
        temperature=0.4,  # Different temperature for varied output
        max_retries=3,
        default_batch_size=5,
        default_num_examples=3,
        request_timeout=30,
        sys_msg=True,
    )
)

# Generate a new dataset from the loaded graph
print("\nGenerating new dataset from loaded graph...")
dataset = engine.create_data(
    num_steps=5,            # Generate fewer examples this time
    batch_size=1,           # One at a time
    topic_model=loaded_graph,
)

# Save with a different name
dataset.save("web_dev_tutorials_v2.jsonl")
print(f"✅ Generated {len(dataset.samples)} new tutorials from existing graph")

# Optional: Extend the loaded graph with more nodes
print("\nExtending the loaded graph...")
# You could add more nodes manually if needed
new_node = loaded_graph.add_node("Progressive Web Apps")
loaded_graph.add_edge(loaded_graph.root.id, new_node.id)

# Save the extended graph
loaded_graph.save("web_dev_graph_extended.json")
print(f"Extended graph saved with {len(loaded_graph.nodes)} nodes")
