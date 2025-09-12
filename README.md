<div align="center">
  <h1>DeepFabric - Synthetic Dataset Generation</h1>
  <h3>Model Distillation, Agent / Model Evaluations, and Statistical Research</h3>
  
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-dark.png">
    <img alt="Deepfabric logo" src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-light.png" width="800px" style="max-width: 100%;">
  </picture>
  
  <!-- CTA Buttons -->
  <p>
    <a href="https://github.com/lukehinds/deepfabric/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22">
      <img src="https://img.shields.io/badge/Contribute-Good%20First%20Issues-green?style=for-the-badge&logo=github" alt="Good First Issues"/>
    </a>
    &nbsp;
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/badge/Chat-Join%20Discord-7289da?style=for-the-badge&logo=discord&logoColor=white" alt="Join Discord"/>
    </a>
  </p>

  <!-- Badges -->
  <p>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/>
    </a>
    <a href="https://github.com/lukehinds/deepfabric/actions/workflows/test.yml">
      <img src="https://github.com/lukehinds/deepfabric/actions/workflows/test.yml/badge.svg" alt="CI Status"/>
    </a>
    <a href="https://pypi.org/project/deepfabric/">
      <img src="https://img.shields.io/pypi/v/deepfabric.svg" alt="PyPI Version"/>
    </a>
    <a href="https://pepy.tech/project/deepfabric">
      <img src="https://static.pepy.tech/badge/deepfabric" alt="Downloads"/>
    </a>
    <a href="https://discord.gg/pPcjYzGvbS">
      <img src="https://img.shields.io/discord/1384081906773131274?color=7289da&label=Discord&logo=discord&logoColor=white" alt="Discord"/>
    </a>
  </p>
  <br/>
</div>

Generate complex Graph or Tree seeded Synthetic Datasets with DeepFabric (formerly known as promptwright)

DeepFabric is a library / CLI that offers a flexible and easy-to-use set of interfaces, enabling
users the ability to generate prompt led synthetic datasets. This makes it suitable for a wide range
of applications, from Teacher-Student distillation, generation of Model / Agent Evals, or general datasets
used for research purposes.

## Topic Graphs (Experimental)

<img src="https://raw.githubusercontent.com/lukehinds/deepfabric/f6ac2717a99b1ae1963aedeb099ad75bb30170e8/assets/graph.svg" width="100%" height="100%"/>

DeepFabric now includes an experimental **Topic Graph** feature that extends beyond traditional hierarchical topic trees to support **cross-connections** between topics. 

The Topic Graph uses a directed acyclic graph (DAG) in place of the Topic Tree. It allows for more complex and realistic relationships between topics,
where a topic can have multiple parent topics and more connection density. This system is introduced as an experimental feature, designed to co-exist with the current `Tree` implementation,
allowing for a gradual transition and comparative analysis.

### Usage

**YAML Configuration:**
```yaml
# Enable graph mode
topic_generator: graph

topic_graph:
  args:
    root_prompt: "Modern Software Architecture"
    provider: "ollama"
    model: "llama3"
    temperature: 0.7
    graph_degree: 3    # Subtopics per node
    graph_depth: 3     # Graph depth
  save_as: "software_graph.json"
```

**Programmatic Usage:**
```python
from deepfabric.topic_graph import Graph, GraphArguments

graph = Graph(
    args=GraphArguments(
        root_prompt="Machine Learning Fundamentals",
        model_name="ollama/llama3",
        temperature=0.7,
        graph_degree=3,
        graph_depth=2,
    )
)

graph.build()
graph.save("ml_graph.json")

# Optional: Generate visualization
graph.visualize("ml_graph")  # Creates ml_graph.svg
```

## Getting Started

### Prerequisites

- Python 3.11+
- uv (for dependency management)
- (Optional) Hugging Face account and API token for dataset upload

### Installation

#### pip

You can install DeepFabric using pip:

```bash
pip install deepfabric
```

#### Development Installation

To install the prerequisites, you can use the following commands:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install deepfabric and its dependencies
git clone https://github.com/lukehinds/deepfabric.git
cd deepfabric
uv sync --all-extras
```

### Usage

DeepFabric offers two ways to define and run your generation tasks:

#### 1. Using YAML Configuration (Recommended)

Create a YAML file defining your generation task:

```yaml
system_prompt: "You are a helpful assistant. You provide clear and concise answers to user questions."

topic_tree:
  args:
    root_prompt: "Capital Cities of the World."
    model_system_prompt: "<system_prompt_placeholder>"
    tree_degree: 3
    tree_depth: 2
    temperature: 0.7
    model_name: "ollama/mistral:latest"
  save_as: "basic_prompt_Tree.jsonl"

data_engine:
  args:
    instructions: "Please provide training examples with questions about capital cities."
    system_prompt: "<system_prompt_placeholder>"
    model_name: "ollama/mistral:latest"
    temperature: 0.9
    max_retries: 2

dataset:
  creation:
    num_steps: 5
    batch_size: 1
    model_name: "ollama/mistral:latest"
    sys_msg: true  # Include system message in dataset (default: true)
  save_as: "basic_prompt_dataset.jsonl"

# Optional Hugging Face Hub configuration
huggingface:
  # Repository in format "username/dataset-name"
  repository: "your-username/your-dataset-name"
  # Token can also be provided via HF_TOKEN environment variable or --hf-token CLI option
  token: "your-hf-token"
  # Additional tags for the dataset (optional)
  # "deepfabric" and "synthetic" tags are added automatically
  tags:
    - "deepfabric-generated-dataset"
    - "geography"
```

Run using the CLI:

```bash
deepfabric start config.yaml
```

The CLI supports various options to override configuration values:

```bash
deepfabric start config.yaml \
  --save-tree output_tree.jsonl \
  --dataset-save-as output_dataset.jsonl \
  --model-name ollama/llama3 \
  --temperature 0.8 \
  --tree-degree 4 \
  --tree-depth 3 \
  --num-steps 10 \
  --batch-size 2 \
  --sys-msg true \  # Control system message inclusion (default: true)
  --hf-repo username/dataset-name \
  --hf-token your-token \
  --hf-tags tag1 --hf-tags tag2
```

#### Provider Integration

DeepFabric uses LiteLLM to interface with LLM providers. You can specify the
provider in the provider, model section in your config or code:

```yaml
provider: "openai"  # LLM provider
    model: "gpt-4-1106-preview"  # Model name
```

Choose any of the listed providers [here](https://docs.litellm.ai/docs/providers/)
and following the same naming convention.

e.g.

The LiteLLM convention for Google Gemini would is:

```python
from litellm import completion
import os

os.environ['GEMINI_API_KEY'] = ""
response = completion(
    model="gemini/gemini-pro", 
    messages=[{"role": "user", "content": "write code for saying hi from LiteLLM"}]
)
```

In DeepFabric, you would specify the provider as `gemini` and the model as `gemini-pro`.

```yaml
provider: "gemini"  # LLM provider
    model: "gemini-pro"  # Model name
```

For Ollama, you would specify the provider as `ollama` and the model as `mistral`
and so on.

```yaml
provider: "ollama"  # LLM provider
    model: "mistral:latest"  # Model name
```

##### API Keys

You can set the API key for the provider in the environment variable. The key
should be set as `PROVIDER_API_KEY`. For example, for OpenAI, you would set the
API key as `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY
```

Again, refer to the [LiteLLM documentation](https://docs.litellm.ai/docs/providers/)
for more information on setting up the API keys.

#### Hugging Face Hub Integration

DeepFabric supports automatic dataset upload to the Hugging Face Hub with the following features:

1. **Dataset Upload**: Upload your generated dataset directly to Hugging Face Hub
2. **Dataset Cards**: Automatically creates and updates dataset cards
3. **Automatic Tags**: Adds "deepfabric" and "synthetic" tags automatically
4. **Custom Tags**: Support for additional custom tags
5. **Flexible Authentication**: HF token can be provided via:
   - CLI option: `--hf-token your-token`
   - Environment variable: `export HF_TOKEN=your-token`
   - YAML configuration: `huggingface.token`

Example using environment variable:
```bash
export HF_TOKEN=your-token
deepfabric start config.yaml --hf-repo username/dataset-name
```

Or pass it in as a CLI option:
```bash
deepfabric start config.yaml --hf-repo username/dataset-name --hf-token your-token
```

#### 2. Using Python Code

You can also create generation tasks programmatically using Python code. There
are several examples in the `examples` directory that demonstrate this approach.

Example Python usage:

```python
from deepfabric import DataSetGenerator, DataSetGeneratorArguments, Tree, TreeArguments

tree = Tree(
    args=TreeArguments(
        root_prompt="Creative Writing Prompts",
        model_system_prompt=system_prompt,
        tree_degree=5,
        tree_depth=4,
        temperature=0.9,
        model_name="ollama/llama3"
    )
)

engine = DataSetGenerator(
    args=DataSetGeneratorArguments(
        instructions="Generate creative writing prompts and example responses.",
        system_prompt="You are a creative writing instructor providing writing prompts and example responses.",
        model_name="ollama/llama3",
        temperature=0.9,
        max_retries=2,
        sys_msg=True,  # Include system message in dataset (default: true)
    )
)
```

### Development

The project uses Poetry for dependency management. Here are some common development commands:

```bash
# Install dependencies including development dependencies
make install

# Format code
make format

# Run linting
make lint

# Run tests
make test

# Run security checks
make security

# Build the package
make build

# Run all checks and build
make all
```

### Prompt Output Examples

With sys_msg=true (default):
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are tasked with designing an immersive virtual reality experience that transports users to a fantastical world of wonder."
    },
    {
      "role": "user",
      "content": "Create a descriptive passage about a character discovering their hidden talents."
    },
    {
      "role": "assistant",
      "content": "As she stared at the canvas, Emma's fingers hovered above the paintbrushes, as if hesitant to unleash the colors that had been locked within her. The strokes began with bold abandon, swirling blues and greens merging into a mesmerizing dance of light and shadow. With each passing moment, she felt herself becoming the art – her very essence seeping onto the canvas like watercolors in a spring storm. The world around her melted away, leaving only the vibrant symphony of color and creation."
    }
  ]
}
```

With sys_msg=false:
```json
{
  "messages": [
    {
      "role": "user",
      "content": "Create a descriptive passage about a character discovering their hidden talents."
    },
    {
      "role": "assistant",
      "content": "As she stared at the canvas, Emma's fingers hovered above the paintbrushes, as if hesitant to unleash the colors that had been locked within her. The strokes began with bold abandon, swirling blues and greens merging into a mesmerizing dance of light and shadow. With each passing moment, she felt herself becoming the art – her very essence seeping onto the canvas like watercolors in a spring storm. The world around her melted away, leaving only the vibrant symphony of color and creation."
    }
  ]
}
```


## Unpredictable Behavior

The library is designed to generate synthetic data based on the prompts and instructions
provided. The quality of the generated data is dependent on the quality of the prompts
and the model used. The library does not guarantee the quality of the generated data.

Large Language Models can sometimes generate unpredictable or inappropriate
content and the authors of this library are not responsible for the content
generated by the models. We recommend reviewing the generated data before using it
in any production environment.

Large Language Models also have the potential to fail to stick with the behavior
defined by the prompt around JSON formatting, and may generate invalid JSON. This
is a known issue with the underlying model and not the library. We handle these
errors by retrying the generation process and filtering out invalid JSON. The 
failure rate is low, but it can happen. We report on each failure within a final
summary.

## Promptwright

The project was renamed and also happened in sync with a large refactor. The last
release of what we promptwright was made at v1.50 which is now under the branch `archive`

I will make sure any security issues or nasty bugs are backported, but I won't be pushing
to pypi under the old promptwright name anymore.

## Contributing

If something here could be improved, please open an issue or submit a pull request.

### License

This project is licensed under the Apache 2 License. See the `LICENSE` file for more details.
