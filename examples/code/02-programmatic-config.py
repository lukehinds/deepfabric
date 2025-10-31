"""
Programmatic Configuration Example

This example demonstrates using YAML configuration files programmatically
and how to override specific parameters.

The recommended workflow is:
1. Create a YAML config file
2. Load it with DeepFabricConfig.from_yaml()
3. Optionally override parameters via CLI or in code
4. Use the CLI's generate command
"""

from deepfabric.config import DeepFabricConfig


def demonstrate_yaml_loading():
    """Show how to load and inspect YAML configuration."""

    print("=" * 80)
    print("Loading Configuration from YAML")
    print("=" * 80)

    # Load configuration from YAML file
    config = DeepFabricConfig.from_yaml("examples/configs/01-quickstart.yaml")

    print("\nConfiguration loaded successfully!")
    if config.dataset_system_prompt:
        print(f"  Dataset system prompt: {config.dataset_system_prompt[:50]}...")
    print(f"  Data engine provider: {config.data_engine.provider}")
    print(f"  Data engine model: {config.data_engine.model}")
    print(f"  Conversation type: {config.data_engine.conversation_type}")

    if config.topic_tree:
        print("\n  Topic tree:")
        print(f"    Depth: {config.topic_tree.depth}")
        print(f"    Degree: {config.topic_tree.degree}")
        print(f"    Provider: {config.topic_tree.provider}")

    if config.dataset:
        print("\n  Dataset:")
        print(f"    Num steps: {config.dataset.creation.num_steps}")
        print(f"    Batch size: {config.dataset.creation.batch_size}")
        print(f"    Output: {config.dataset.save_as}")

    if config.dataset and config.dataset.formatters:
        print(f"\n  Formatters ({len(config.dataset.formatters)}):")
        for fmt in config.dataset.formatters:
            print(f"    - {fmt.name}: {fmt.template}")


def show_config_parameters():
    """Show how to extract parameters for use with the API."""

    print("\n" + "=" * 80)
    print("Extracting Parameters from Config")
    print("=" * 80)

    config = DeepFabricConfig.from_yaml("examples/configs/06-complete-pipeline.yaml")

    # Get tree parameters
    if config.topic_tree:
        tree_params = config.get_topic_tree_params()
        print("\nTree Parameters:")
        for key, value in tree_params.items():
            if isinstance(value, str) and len(value) > 50:  # noqa: PLR2004
                print(f"  {key}: {value[:50]}...")
            else:
                print(f"  {key}: {value}")

    # Get engine parameters
    engine_params = config.get_engine_params()
    print("\nEngine Parameters:")
    for key, value in engine_params.items():
        if isinstance(value, str) and len(value) > 50:  # noqa: PLR2004
            print(f"  {key}: {value[:50]}...")
        else:
            print(f"  {key}: {value}")

    # Get dataset configuration
    dataset_config = config.get_dataset_config()
    print("\nDataset Configuration:")
    print(f"  save_as: {dataset_config['save_as']}")
    print(f"  num_steps: {dataset_config['creation']['num_steps']}")
    print(f"  batch_size: {dataset_config['creation']['batch_size']}")


def demonstrate_config_usage():
    """Show the recommended way to use configs."""

    print("\n" + "=" * 80)
    print("Recommended Usage Pattern")
    print("=" * 80)

    print("""
The recommended way to use DeepFabric is via the CLI with YAML configs:

1. CREATE A YAML CONFIG FILE:
   See examples/configs/ for templates

2. RUN GENERATION:
   deepfabric generate my_config.yaml

3. OVERRIDE PARAMETERS (optional):
   deepfabric generate my_config.yaml \\
     --num-steps 100 \\
     --model gpt-4o \\
     --temperature 0.8

4. PROGRAMMATIC USAGE:
   If you need to use the API directly:

   ```python
   from deepfabric import Tree, DataSetGenerator
   from deepfabric.config import DeepFabricConfig
   from deepfabric.topic_manager import handle_tree_events, load_or_build_topic_model
   from deepfabric.dataset_manager import create_dataset, save_dataset

   # Load config
   config = DeepFabricConfig.from_yaml("config.yaml")

   # Build topic model
   topic_model = load_or_build_topic_model(config=config)

   # Create generator
   engine_params = config.get_engine_params()
   generator = DataSetGenerator(**engine_params)

   # Generate dataset
   dataset = create_dataset(
       engine=generator,
       topic_model=topic_model,
       config=config,
   )

   # Save with formatters and uploads
   save_dataset(dataset, config.dataset.save_as, config)
   ```

For simple scripts, see 01-quickstart.py for direct API usage.
For complete workflows, use YAML configs with the CLI.
    """)


def main():
    """Main demonstration."""

    # Show how to load YAML configs
    demonstrate_yaml_loading()

    # Show how to extract parameters
    show_config_parameters()

    # Show recommended patterns
    demonstrate_config_usage()


if __name__ == "__main__":
    main()
