"""
DeepFabric Integrations Example

Demonstrates external integrations:
- YAML configuration loading
- HuggingFace Hub upload
- Multiple provider examples
- CLI integration patterns
"""

import os
import sys
import tempfile

import yaml

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric.config import DeepFabricConfig
from deepfabric.dataset import Dataset


def example_yaml_configuration():
    """Demonstrate YAML configuration loading and usage."""
    print("=" * 60)
    print("Example 1: YAML Configuration Loading")
    print("=" * 60)

    # Create a sample YAML configuration
    sample_config = {
        "dataset_system_prompt": "You are a helpful programming instructor.",
        "topic_tree": {
            "topic_prompt": "Python programming concepts",
            "provider": "ollama",
            "model_name": "qwen3:8b",
            "temperature": 0.7,
            "degree": 2,
            "depth": 2,
            "save_as": "python_tree.jsonl"
        },
        "data_engine": {
            "instructions": "Create clear programming examples",
            "generation_system_prompt": "You are a Python expert creating educational content.",
            "provider": "ollama",
            "model_name": "qwen3:8b",
            "temperature": 0.3,
            "max_retries": 3
        },
        "dataset": {
            "creation": {
                "num_steps": 3,
                "batch_size": 1,
                "provider": "ollama",
                "model_name": "qwen3:8b",
                "sys_msg": True
            },
            "save_as": "python_dataset.jsonl"
        }
    }

    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_config, f, default_flow_style=False)
        config_path = f.name

    try:
        print(f"📝 Created sample config: {config_path}")

        # Load configuration
        config = DeepFabricConfig.from_yaml(config_path)
        print("✅ Successfully loaded YAML configuration")

        # Show configuration details
        if hasattr(config, 'dataset_system_prompt') and config.dataset_system_prompt:
            print(f"   Dataset system prompt: {config.dataset_system_prompt[:50]}...")

        tree_config = config.get_topic_tree_params()
        if tree_config:
            degree = tree_config.get('degree', 'not set')
            depth = tree_config.get('depth', 'not set')
            temp = tree_config.get('temperature', 'not set')
            print(f"   Tree config: {degree}x{depth} @ {temp}°")

        engine_config = config.get_engine_params()
        print(f"   Engine provider: {engine_config.get('provider', 'not set')}")

        dataset_config = config.get_dataset_config()
        if dataset_config and 'creation' in dataset_config:
            print(f"   Dataset steps: {dataset_config['creation'].get('num_steps', 'not set')}")

        return config

    finally:
        # Clean up temporary file
        os.unlink(config_path)


def example_multiple_providers():
    """Demonstrate using different providers for different stages."""
    print("\n" + "=" * 60)
    print("Example 2: Multiple Provider Configuration")
    print("=" * 60)

    configs = {
        "openai_config": {
            "provider": "openai",
            "model": "gpt-4o",
            "use_case": "High-quality content generation"
        },
        "anthropic_config": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-5",
            "use_case": "Complex reasoning tasks"
        },
        "ollama_config": {
            "provider": "ollama",
            "model": "qwen3:8b",
            "use_case": "Local development and testing"
        },
        "gemini_config": {
            "provider": "gemini",
            "model": "gemini-2.5-flash-lite",
            "use_case": "Fast, cost-effective generation"
        }
    }

    print("🔧 Available Provider Configurations:")
    for _name, config in configs.items():
        print(f"   {config['provider']:10} | {config['model']:20} | {config['use_case']}")

    # Example of mixed provider configuration
    mixed_config = {
        "dataset_system_prompt": "You are an expert software engineer.",
        "topic_tree": {
            "topic_prompt": "Software design patterns",
            "provider": "gemini",  # Fast for topic generation
            "model_name": "gemini-2.5-flash-lite",
            "temperature": 0.7,
            "degree": 3,
            "depth": 2,
        },
        "data_engine": {
            "instructions": "Create detailed explanations with code examples",
            "provider": "anthropic",  # High quality for content
            "model_name": "claude-sonnet-4-5",
            "temperature": 0.3,
        },
        "dataset": {
            "creation": {
                "provider": "openai",  # Reliable for final dataset
                "model_name": "gpt-4o",
                "num_steps": 2,
                "batch_size": 1,
                "sys_msg": True
            }
        }
    }

    print("\n💡 Example Mixed Provider Strategy:")
    print(f"   Topic Generation: {mixed_config['topic_tree']['provider']} (fast)")
    print(f"   Content Creation: {mixed_config['data_engine']['provider']} (high quality)")
    print(f"   Dataset Assembly: {mixed_config['dataset']['creation']['provider']} (reliable)")


def example_huggingface_integration():
    """Demonstrate HuggingFace Hub integration."""
    print("\n" + "=" * 60)
    print("Example 3: HuggingFace Hub Integration")
    print("=" * 60)

    # Check for HF token
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        print("⚠️  HF_TOKEN environment variable not set")
        print("   Set it with: export HF_TOKEN=your_token_here")
        print("   Skipping actual upload, showing configuration...")

        # Show configuration example
        hf_config = {
            "huggingface": {
                "repository": "username/dataset-name",
                "token": "hf_your_token_here",  # Or use HF_TOKEN env var
                "tags": [
                    "synthetic",
                    "programming",
                    "education",
                    "deepfabric"
                ]
            }
        }

        print("📝 HuggingFace configuration example:")
        print(yaml.dump(hf_config, default_flow_style=False))
        return

    # If we have a token, demonstrate the upload process
    try:
        from deepfabric.hf_hub import HFUploader  # noqa: PLC0415

        print(f"🔑 Found HF_TOKEN: {hf_token[:10]}...")

        # Create a small sample dataset for upload
        sample_dataset = Dataset()
        sample_data = [
            {
                "messages": [
                    {"role": "user", "content": "What is a design pattern?"},
                    {"role": "assistant", "content": "A design pattern is a reusable solution to a commonly occurring problem in software design."}
                ]
            }
        ]
        sample_dataset.samples = sample_data
        sample_dataset.save("sample_upload_dataset.jsonl")

        print("📁 Created sample dataset for upload")

        # Initialize uploader
        _uploader = HFUploader(hf_token)

        # Note: Uncomment the following lines to perform actual upload
        # result = uploader.push_to_hub(
        #     repo_name="your-username/test-deepfabric-dataset",
        #     dataset_path="sample_upload_dataset.jsonl",
        #     tags=["test", "synthetic", "deepfabric"]
        # )
        #
        # if result["status"] == "success":
        #     print(f"✅ {result['message']}")
        # else:
        #     print(f"❌ {result['message']}")

        print("💡 Upload code ready (commented out to prevent accidental uploads)")
        print("   Uncomment the upload section to perform actual upload")

    except ImportError:
        print("⚠️  HuggingFace hub dependencies not available")
    except Exception as e:
        print(f"❌ Error with HuggingFace integration: {e}")


def example_cli_integration():
    """Demonstrate CLI integration patterns."""
    print("\n" + "=" * 60)
    print("Example 4: CLI Integration Patterns")
    print("=" * 60)

    print("🖥️  CLI Usage Examples:")

    cli_examples = [
        {
            "purpose": "Basic generation from config",
            "command": "deepfabric generate config.yaml"
        },
        {
            "purpose": "Override model in config",
            "command": "deepfabric generate config.yaml --model gpt-4o"
        },
        {
            "purpose": "Generate without config file",
            "command": 'deepfabric generate --topic-prompt "Python basics" --provider ollama --model qwen3:8b'
        },
        {
            "purpose": "Graph mode with custom parameters",
            "command": "deepfabric generate config.yaml --mode graph --depth 3 --degree 4"
        },
        {
            "purpose": "Upload to HuggingFace after generation",
            "command": "deepfabric upload dataset.jsonl --repo username/dataset-name"
        },
        {
            "purpose": "Validate configuration before running",
            "command": "deepfabric validate config.yaml"
        }
    ]

    for i, example in enumerate(cli_examples, 1):
        print(f"\n   {i}. {example['purpose']}:")
        print(f"      {example['command']}")

    print("\n📖 CLI Configuration Tips:")
    print("   • Use YAML configs for complex setups")
    print("   • Override specific parameters with --flags")
    print("   • Set API keys via environment variables")
    print("   • Validate configs before running expensive operations")


def example_environment_setup():
    """Show environment variable patterns."""
    print("\n" + "=" * 60)
    print("Example 5: Environment Variables")
    print("=" * 60)

    env_vars = {
        "OPENAI_API_KEY": "OpenAI API access",
        "ANTHROPIC_API_KEY": "Anthropic/Claude API access",
        "GEMINI_API_KEY": "Google Gemini API access",
        "HF_TOKEN": "HuggingFace Hub upload access"
    }

    print("🔧 Required Environment Variables:")
    for var, description in env_vars.items():
        value = os.getenv(var)
        status = "✅ Set" if value else "❌ Not set"
        print(f"   {var:20} | {description:30} | {status}")

    print("\n💡 Setup Examples:")
    print("   # Bash/Zsh")
    print("   export OPENAI_API_KEY=sk-your-key-here")
    print("   export HF_TOKEN=hf_your-token-here")
    print()
    print("   # Or create a .env file:")
    print("   echo 'OPENAI_API_KEY=sk-your-key-here' >> .env")
    print("   echo 'HF_TOKEN=hf_your-token-here' >> .env")


def main():
    """Run all integration examples."""
    print("🚀 DeepFabric Integrations Examples")
    print("=" * 60)

    try:
        _config = example_yaml_configuration()
        example_multiple_providers()
        example_huggingface_integration()
        example_cli_integration()
        example_environment_setup()

        print("\n" + "=" * 60)
        print("🎉 All integration examples completed!")
        print("=" * 60)
        print("📚 Key Takeaways:")
        print("   • YAML configs provide flexible configuration")
        print("   • Mix providers for optimal cost/quality balance")
        print("   • Environment variables secure API keys")
        print("   • CLI supports both config and parameter override")
        print("   • HuggingFace integration enables easy sharing")

    except Exception as e:
        print(f"\n❌ Error running integration examples: {e}")
        raise


if __name__ == "__main__":
    main()
