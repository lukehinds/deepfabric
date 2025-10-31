"""
Complete Pipeline Example

This example demonstrates a full end-to-end workflow using DeepFabric's API.

RECOMMENDED: Use YAML configs with the CLI for production workflows:
  deepfabric generate examples/configs/06-complete-pipeline.yaml

This file shows how to use the API directly if needed.
"""

from deepfabric import DataSetGenerator
from deepfabric.config import DeepFabricConfig
from deepfabric.dataset_manager import create_dataset, save_dataset
from deepfabric.topic_manager import load_or_build_topic_model


def run_from_yaml_config():
    """
    Complete pipeline using YAML configuration.

    This is the RECOMMENDED approach for production use.
    """

    print("=" * 80)
    print("Complete Pipeline - YAML Config Approach (Recommended)")
    print("=" * 80)

    # Step 1: Load configuration from YAML
    print("\n[1/4] Loading configuration from YAML...")
    config = DeepFabricConfig.from_yaml("examples/configs/06-complete-pipeline.yaml")

    print("  ✓ Config loaded")
    print(f"    Topic tree depth: {config.topic_tree.depth if config.topic_tree else 'N/A'}")
    print(f"    Dataset size: {config.dataset.creation.num_steps}")
    print(f"    Formatters: {len(config.dataset.formatters)}")

    # Step 2: Build or load topic model
    print("\n[2/4] Building topic model...")
    topic_model = load_or_build_topic_model(config=config)
    print("  ✓ Topic model ready")

    # Step 3: Create data generator and generate dataset
    print("\n[3/4] Generating dataset...")

    engine_params = config.get_engine_params()
    generator = DataSetGenerator(**engine_params)

    dataset = create_dataset(
        engine=generator,
        topic_model=topic_model,
        config=config,
    )

    print(f"  ✓ Dataset generated: {len(dataset.samples)} samples")

    # Step 4: Save dataset (with formatters and uploads)
    print("\n[4/4] Saving dataset and applying formatters...")
    save_dataset(dataset, config.dataset.save_as, config)

    print("\n" + "=" * 80)
    print("✓ Pipeline complete!")
    print(f"  Raw dataset: {config.dataset.save_as}")
    if config.dataset.formatters:
        for formatter in config.dataset.formatters:
            print(f"  Formatted: {formatter.output}")
    print("=" * 80)


def demonstrate_cli_approach():
    """Show the simplest approach: using the CLI."""

    print("\n" + "=" * 80)
    print("SIMPLEST APPROACH: Use the CLI")
    print("=" * 80)

    cli_instructions = """
The easiest and most powerful way to use DeepFabric is via the CLI:

1. BASIC USAGE:
   deepfabric generate examples/configs/06-complete-pipeline.yaml

2. WITH PARAMETER OVERRIDES:
   deepfabric generate examples/configs/06-complete-pipeline.yaml \\
     --num-steps 100 \\
     --model gpt-4o \\
     --temperature 0.8

3. SKIP TOPIC GENERATION (use existing):
   deepfabric generate config.yaml --load-tree topics.jsonl

4. GENERATE TOPICS ONLY:
   deepfabric generate config.yaml --topic-only

BENEFITS:
- Automatic progress display with TUI
- Error handling and retries
- Formatters applied automatically
- Automatic upload to HuggingFace/Kaggle
- Parameter validation
- Debug mode available (--debug)

For complex workflows, the CLI is more reliable than direct API usage.
"""

    print(cli_instructions)


def show_api_overview():
    """Show the structure of the DeepFabric API."""

    print("\n" + "=" * 80)
    print("DeepFabric API Overview")
    print("=" * 80)

    api_overview = """
CORE COMPONENTS:

1. Topic Generation:
   - Tree: Hierarchical topic tree for diverse examples
   - Graph: Topic graph for interconnected concepts
   - TopicModel: Abstract interface for both

2. Dataset Generation:
   - DataSetGenerator: Creates training examples from topics
   - Supports: basic, structured, chain_of_thought conversations
   - Agent modes: single_turn, multi_turn (with tools)

3. Dataset Management:
   - Dataset: Container for training examples
   - Formatters: Transform to different output formats
   - Validation: Schema validation and quality checks

4. Configuration:
   - DeepFabricConfig: Main configuration object
   - Load from YAML or create programmatically
   - Type-safe with Pydantic validation

5. Utilities:
   - topic_manager: Build/load topic models with progress
   - dataset_manager: Create/save datasets with formatters
   - HFUploader: Upload to HuggingFace Hub
   - KaggleUploader: Upload to Kaggle

TYPICAL WORKFLOW (API):

```python
from deepfabric import DataSetGenerator
from deepfabric.config import DeepFabricConfig
from deepfabric.topic_manager import load_or_build_topic_model
from deepfabric.dataset_manager import create_dataset, save_dataset

# 1. Load config
config = DeepFabricConfig.from_yaml("config.yaml")

# 2. Build topic model
topic_model = load_or_build_topic_model(config=config)

# 3. Create generator
engine_params = config.get_engine_params()
generator = DataSetGenerator(**engine_params)

# 4. Generate dataset
dataset = create_dataset(
    engine=generator,
    topic_model=topic_model,
    config=config,
)

# 5. Save with formatters
save_dataset(dataset, config.dataset.save_as, config)
```

TYPICAL WORKFLOW (CLI):

```bash
# Create YAML config file
deepfabric generate my_config.yaml
```

Much simpler! Use CLI for production.
"""

    print(api_overview)


def main():
    """Main demonstration."""

    # Show the CLI approach first (recommended)
    demonstrate_cli_approach()

    # Show API overview
    show_api_overview()

    print("\n" + "=" * 80)
    print("Would you like to run the complete pipeline?")
    print("Uncomment the line below to execute:")
    print("=" * 80)

    # Uncomment to actually run the pipeline:
    # run_from_yaml_config()

    print("\nTo run this example:")
    print("1. Edit this file and uncomment the run_from_yaml_config() call above")
    print("2. OR run: deepfabric generate examples/configs/06-complete-pipeline.yaml")
    print("\nUsing the CLI (option 2) is recommended!")


if __name__ == "__main__":
    main()
