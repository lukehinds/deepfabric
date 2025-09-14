"""
Example showing how to use custom topics instead of auto-generated ones.
This gives you full control over the topics used for dataset generation.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator, Tree

# Create a topic tree with a simple root
tree = Tree(
    topic_prompt="Data Science",
    model_name="ollama/qwen3:8b",
    topic_system_prompt="You are a data science expert.",
    degree=3,
    depth=2,
    temperature=0.7,
)

custom_topics = [
    {"path": ["Data Science", "Machine Learning", "Supervised Learning", "Classification"]},
    {"path": ["Data Science", "Machine Learning", "Supervised Learning", "Regression"]},
    {"path": ["Data Science", "Machine Learning", "Unsupervised Learning", "Clustering"]},
    {
        "path": [
            "Data Science",
            "Machine Learning",
            "Unsupervised Learning",
            "Dimensionality Reduction",
        ]
    },
    {"path": ["Data Science", "Deep Learning", "Neural Networks", "CNNs"]},
    {"path": ["Data Science", "Deep Learning", "Neural Networks", "RNNs"]},
    {"path": ["Data Science", "Deep Learning", "Neural Networks", "Transformers"]},
    {"path": ["Data Science", "Data Engineering", "ETL Pipelines", "Apache Spark"]},
    {"path": ["Data Science", "Data Engineering", "ETL Pipelines", "Apache Airflow"]},
    {"path": ["Data Science", "Data Engineering", "Data Warehousing", "Snowflake"]},
]

# Load the custom topics into the tree
tree.from_dict_list(custom_topics)

# Save the custom tree if needed
tree.save("custom_data_science_tree.jsonl")

# Create an engine with specific instructions for data science content
engine = DataSetGenerator(
    instructions="""Create a comprehensive tutorial that includes:
                   - Theoretical explanation of the concept
                   - Mathematical formulation (if applicable)
                   - Python code implementation using popular libraries
                   - A practical example with sample data
                   - Common use cases and applications
                   - Pros and cons of the approach""",
    generation_system_prompt="""You are a data science educator with expertise in
                    machine learning, deep learning, and data engineering.
                    Provide clear, practical examples using Python.""",
    model_name="ollama/qwen3:8b",
    prompt_template=None,
    example_data=None,
    temperature=0.3,  # Lower temperature for more consistent technical content
    max_retries=3,
    default_batch_size=5,
    default_num_examples=3,
    request_timeout=30,
    sys_msg=True,
)

# Generate dataset from custom topics
dataset = engine.create_data(
    num_steps=len(custom_topics),  # One example per topic
    batch_size=2,  # Process 2 at a timeGenerate 3 examples per topic
    topic_model=tree,
    sys_msg=True,  # Include system messages in the dataset
)

# Save the final dataset
dataset.save("data_science_tutorials.jsonl")

print(f"Generated {len(dataset.samples)} data science tutorials")
print(f"Failed samples: {len(engine.failed_samples)}")

# Print the topics that were used
print("\nTopics covered:")
for topic in custom_topics:
    print(f"  - {' > '.join(topic['path'])}")
