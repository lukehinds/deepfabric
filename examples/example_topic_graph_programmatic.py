"""
Example of using DeepFabric's Graph programmatically to create
a knowledge graph with cross-connections between topics.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator, DataSetGeneratorArguments
from deepfabric.graph import Graph, GraphArguments

# Define the system prompt for consistency
system_prompt = """You are an expert in computer science and software engineering.
Your responses should be technically accurate, practical, and include real-world examples."""

# Create a topic graph with cross-connections
print("Building topic graph...")
graph = Graph(
    args=GraphArguments(
        root_prompt="Modern Software Architecture and Design Patterns",
        model_name="ollama/llama3",  # Change to your preferred model
        temperature=0.7,
        graph_degree=3,  # 3 subtopics per node
        graph_depth=2,   # 2 levels deep
    )
)

# Build the graph (this will make LLM calls)
graph.build()

# Save the graph structure
graph.save("software_architecture_graph.json")
print(f"Graph saved with {len(graph.nodes)} nodes")

# Optional: Visualize the graph (requires mermaid-py)
try:
    graph.visualize("software_architecture_graph")
    print("Graph visualization saved to software_architecture_graph.svg")
except Exception as e:
    print(f"Visualization skipped: {e}")

# Check for cycles in the graph
if graph.has_cycle():
    print("Warning: Graph contains cycles")
else:
    print("Graph is acyclic")

# Create a data engine for generating training data
engine = DataSetGenerator(
    args=DataSetGeneratorArguments(
        instructions="""Create detailed technical explanations that include:
                       - Core concepts and principles
                       - Implementation examples in multiple languages
                       - Best practices and anti-patterns
                       - Performance considerations
                       - Real-world use cases
                       - Common pitfalls and how to avoid them""",
        system_prompt=system_prompt,
        model_name="ollama/llama3",
        prompt_template=None,
        example_data=None,
        temperature=0.2,  # Lower temperature for more focused technical content
        max_retries=3,
        default_batch_size=5,
        default_num_examples=3,
        request_timeout=30,
        sys_msg=True,
    )
)

# Generate dataset using the graph
print("\nGenerating dataset from graph...")
dataset = engine.create_data(
    num_steps=10,       # Generate 10 examples
    batch_size=2,       # Process 2 at a time
    topic_model=graph,  # Use the graph as topic source
)

# Save the dataset
dataset.save("software_architecture_dataset.jsonl")
print(f"Dataset saved with {len(dataset.samples)} samples")

# Print some statistics
print("\nGraph Statistics:")
print(f"  Total nodes: {len(graph.nodes)}")
print(f"  Total paths: {len(graph.get_all_paths())}")
if graph.failed_generations:
    print(f"  Failed generations: {len(graph.failed_generations)}")
