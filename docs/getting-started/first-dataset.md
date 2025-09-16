# Your First Dataset

Creating your first synthetic dataset with DeepFabric takes just a few minutes and demonstrates the core concepts you'll use in more sophisticated scenarios. This walkthrough generates a practical dataset about Python programming fundamentals using a simple configuration file.

The process showcases DeepFabric's three-stage pipeline: topic tree generation, dataset creation, and output formatting. You'll see how a single root prompt expands into multiple related topics, which then generate diverse training examples.

## Configuration File

Create a file named `python-tutorial.yaml` with the following configuration:

```yaml
dataset_system_prompt: "You are a Python programming instructor providing clear, educational content for intermediate developers."

topic_tree:
  topic_prompt: "Python programming fundamentals"
  topic_system_prompt: "You are a Python programming instructor providing clear, educational content for intermediate developers."
  degree: 3
  depth: 2
  temperature: 0.7
  provider: "ollama"
  model: "mistral"
  save_as: "python_topics.jsonl"

data_engine:
  instructions: "Create a Python code example with detailed explanation suitable for intermediate developers."
  generation_system_prompt: "You are a Python programming instructor providing clear, educational content for intermediate developers."
  provider: "ollama"
  model: "mistral"
  temperature: 0.8
  max_retries: 3

dataset:
  creation:
    num_steps: 10
    batch_size: 2
    provider: "ollama"
    model: "mistral"
    sys_msg: true
  save_as: "python_tutorial_dataset.jsonl"
```

This configuration demonstrates several key concepts. System prompts are specified directly in each section where they're needed. The tree structure creates a hierarchical breakdown of Python concepts, while the data engine generates practical examples based on these topics.

??? tip "Understanding the Configuration Structure"
    The configuration separates concerns between topic generation and dataset creation. This separation allows you to experiment with different generation parameters without rebuilding the topic structure, saving time during iterative development.

## Generate the Dataset

Run the generation process using the CLI:

```bash
deepfabric generate python-tutorial.yaml
```

The generation process displays real-time progress information, showing topic expansion, dataset creation steps, and any errors that occur during processing. The entire process typically completes in 2-5 minutes depending on your model provider and configuration parameters.

## Understanding the Output

DeepFabric produces two primary outputs from this configuration:

**Topic Tree (`python_topics.jsonl`)**: Contains the hierarchical breakdown of your root prompt into specific subtopics. Each line represents a topic node with its path from the root prompt.

**Training Dataset (`python_tutorial_dataset.jsonl`)**: Contains the actual training examples in conversation format, ready for use with language model training frameworks.

### Sample Topic Structure

The generated topic tree might include entries like:

```json
{"path": ["Python programming fundamentals", "Data Structures", "Lists and Arrays"]}
{"path": ["Python programming fundamentals", "Control Flow", "Loop Constructs"]}
{"path": ["Python programming fundamentals", "Object-Oriented Programming", "Class Inheritance"]}
```

Each path represents a specific focus area that the dataset generator will use to create targeted training examples.

### Sample Training Examples

The dataset contains conversational examples following this structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a Python programming instructor providing clear, educational content for intermediate developers."
    },
    {
      "role": "user",
      "content": "Can you explain list comprehensions in Python with a practical example?"
    },
    {
      "role": "assistant",
      "content": "List comprehensions provide a concise way to create lists in Python. Here's how they work: [detailed explanation with code examples]"
    }
  ]
}
```

The `sys_msg: true` setting includes the system prompt in each training example, providing context for the model about its role and expected behavior.

## Customizing Your Generation

Modify the configuration to explore different approaches:

**Increase Diversity**: Raise `temperature` values to generate more creative and varied content, though this may reduce consistency.

**Expand Coverage**: Increase `degree` and `depth` to create more comprehensive topic coverage, though this extends generation time.

**Scale Production**: Increase `num_steps` and `batch_size` for larger datasets, balancing generation speed with resource consumption.

**Alternative Providers**: Change the provider and model parameters to experiment with different language models and their characteristics.

## Validation and Quality Control

Validate your configuration before running expensive generation processes:

```bash
deepfabric validate python-tutorial.yaml
```

This command checks your configuration for common issues, missing required fields, and parameter compatibility problems, helping you catch problems before investing time in generation.

## Next Steps

With your first dataset successfully generated, explore the [Configuration Guide](../guide/index.md) to understand advanced configuration options, or review the [CLI Reference](../cli/index.md) to discover additional commands for dataset management and visualization.

The generated dataset is ready for immediate use in training frameworks, evaluation pipelines, or research experiments. The JSONL format integrates seamlessly with popular machine learning libraries and Hugging Face datasets.