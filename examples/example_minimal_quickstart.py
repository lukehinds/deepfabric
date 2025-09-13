"""
Minimal example to get started with DeepFabric.
This creates a small synthetic dataset about Python programming.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator, DataSetGeneratorArguments, Tree, TreeArguments

# Step 1: Create a topic tree
tree = Tree(
    args=TreeArguments(
        root_prompt="Python programming fundamentals",
        model_name="ollama/llama3",  # Change to your model
        model_system_prompt="You are a Python programming instructor.",
        tree_degree=3,  # 3 branches per level
        tree_depth=2,  # 2 levels deep
        temperature=0.7,
    )
)

# Build the tree structure
tree.build()
tree.save("python_topics.jsonl")

# Step 2: Create a data engine
engine = DataSetGenerator(
    args=DataSetGeneratorArguments(
        instructions="Create a Python code example with explanation",
        system_prompt="You are a Python programming instructor.",
        model_name="ollama/llama3",
        prompt_template=None,
        example_data=None,
        temperature=0.7,
        max_retries=3,
        default_batch_size=5,
        default_num_examples=3,
        request_timeout=30,
        sys_msg=True,
    )
)

# Step 3: Generate the dataset
dataset = engine.create_data(
    num_steps=5,  # Generate 5 examples
    batch_size=1,  # One at a time
    topic_model=tree,  # Use our topic tree
)

# Step 4: Save the dataset
dataset.save("python_examples.jsonl")

print(f"Generated {len(dataset.samples)} training examples")
print("Saved to python_examples.jsonl")
