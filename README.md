<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-dark.png">
    <img alt="Deepfabric logo" src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/logo-light.png" width="486px" height="150spx" style="max-width: 100%;">
  </picture>
  <h3>Generate High-Quality Synthetic Datasets at Scale</h3>

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

**Deepfabric** is an SDK and CLI tool that leverages large language models to generate high-quality synthetic datasets. It's designed for researchers and developers building teacher-student distillation pipelines, creating evaluation benchmarks for models and agents, or conducting research requiring diverse training data.

The key innovation lies in Deepfabric's graph and tree-based architecture, which uses structured topic nodes as generation seeds. This approach ensures the creation of datasets that are both highly diverse and domain-specific, while minimizing redundancy and duplication across generated samples.

<div align="center">
  <img src="https://raw.githubusercontent.com/lukehinds/deepfabric/main/assets/demo.gif" alt="Deepfabric Demo" width="80%" style="max-width: 700px;">
</div>

## Quickstart

Get up and running in under 60 seconds:

### 1. Install Deepfabric

```bash
pip install deepfabric
```

### 2. Generate Your First Dataset

```bash
# Set your API key (or use Ollama for local generation)
export OPENAI_API_KEY="your-api-key"

# Generate a dataset with a single command
deepfabric generate \
  --mode tree \
  --provider openai \
  --model gpt-4o \
  --depth 3 \
  --degree 3 \
  --num-steps 9 \
  --batch-size 1 \
  --topic-prompt "This history Quantum physics" \
  --generation-system-prompt "You are an expert on academic history, with a specialism in the sciences" \
  --dataset-save-as dataset.jsonl
```

Deepfabric will automatically:
- Generate a hierarchical topic tree (3 levels deep, 3 branches per level)
- Create 9 diverse Q&A pairs across the generated topics
- Save your dataset to `dataset.jsonl`

### 3. Use Your Dataset

Your dataset is ready for fine-tuning in the standard instruct format:

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Can you explain the significance of the double-slit experiment in quantum physics?"
    },
    {
      "role": "assistant",
      "content": "The double-slit experiment is fundamental in quantum physics as it demonstrates the dual nature of light and particles, showing both wave-like and particle-like properties. When particles such as photons or electrons pass through two slits, they create an interference pattern characteristic of waves, even if sent one at a time. This result challenged classical physics and contributed significantly to the development of quantum mechanics, highlighting the probabilistic nature of quantum states."
    }
  ]
}
{
  "messages": [
    {
      "role": "user",
      "content": "Can you explain Albert Einstein's contribution to quantum theory?"
    },
    {
      "role": "assistant",
      "content": "Albert Einstein made significant contributions to quantum theory, particularly through his explanation of the photoelectric effect, for which he won the Nobel Prize in 1921. He proposed that light could be thought of as discrete packets of energy called quanta or photons, which could explain how electrons are emitted from metals when exposed to light. This idea was instrumental in the development of quantum mechanics. He later became famous for his skepticism about quantum mechanics probabilistic interpretation, leading to his quote \"God does not play dice with the universe.\""
    }
  ]
}
```

### 4. Use local models.

Generate larger datasets with different models:

```bash
# With a depth of 4 and degree of 4^5 = 1,024
deepfabric generate \
  --provider ollama \
  --model qwen3:8b \
  --depth 4 \
  --degree 5 \
  --num-steps 100 \
  --batch-size 5 \
  --topic-prompt "Machine Learning Fundamentals"
  --generation-system-prompt "You are an expert on Machine Learning and its application in modern technologies" \
  --dataset-save-as dataset.jsonl
```

## Key Features

### Topic Trees and Graphs

Deepfabric can generate topics using two approaches:

**Topic Graphs** (Experimental): DAG-based structure allowing cross-connections between topics, ideal for complex domains with interconnected concepts.

**Topic Trees**: Traditional hierarchical structure where each topic branches into subtopics, perfect for well-organized domains.

### Multi-Provider Support

Leverage different LLMs for different tasks. Use GPT-4 for complex topic generation, then switch to a local model like Mixtral for bulk data creation:

```yaml
topic_tree:
  provider: "openai"
  model: "gpt-4"  # High quality for topic structure

data_engine:
  provider: "ollama"
  model: "mixtral"  # Fast and efficient for bulk generation
```

### Automatic Dataset Upload

Push your datasets directly to Hugging Face Hub with automatic dataset cards:

```bash
deepfabric generate config.yaml --hf-repo username/my-dataset --hf-token $HF_TOKEN
```

## Docs / Examples

For more details, including how to use the SDK, see the [docs!](https://lukehinds.github.io/DeepFabric/)

There are also lots of [examples](./examples/) to get you going.


## Stay Updated

Deepfabric development is moving at a fast pace 🏃‍♂️, for a great way to follow the project and to be instantly notified of new releases, **Star the repo**.

<img src="/assets/star.gif" width="40%" height="40%"/>

## FAQ

## I heard that Synthetic Data is inferior to Human curated / labelled data

That used to be the view, but no longer is, especially when it comes to model fine-tuning of SLMs. Views shifted when Deepseek first released r1,
trained largley on synthetics using a process termed 'distilation'. 

Since then many other models have followed suit, including most recently Phi-4, to quote the whitepaper "Synthetic data constitutes the bulk of the training data for phi-4".

## Roadmap

### Outlines

Introduce Outlines as an LLM replacement, and make use of its structured ouput support

### Conversation Framework

Deepfabric currently outputs to Open AI chat format, we will provide a system where you can easily plug in a post-processing conversion to whatever format is needed. This should allow easy adaption to what ever you need within a training pipeline:

```yaml
formatters:
- name: "alpaca"
  template: "builtin://alpaca.py"
- name: "custom"
  template: "file://./my_format.py"
  config:
    instruction_field: "query"
```


