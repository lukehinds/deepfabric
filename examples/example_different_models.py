"""
Example showing how to use different LLM providers and models with DeepFabric.
Demonstrates using OpenAI, Anthropic, Google, and local models.
"""

import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepfabric import DataSetGenerator, DataSetGeneratorArguments, Tree, TreeArguments


def example_with_ollama():
    """Example using Ollama (local models)."""
    print("=" * 50)
    print("Example: Ollama (Local)")
    print("=" * 50)

    tree = Tree(
        args=TreeArguments(
            root_prompt="Machine Learning Algorithms",
            model_name="ollama/llama3",  # or ollama/mistral, ollama/codellama, etc.
            model_system_prompt="You are an expert in machine learning.",
            temperature=0.7,
            tree_degree=3,
            tree_depth=2,
        )
    )

    # Note: Ollama must be running locally
    # Install: https://ollama.ai
    # Run model: ollama run llama3

    tree.build()
    print(f"✅ Generated tree with {len(tree.get_all_paths())} paths using Ollama\n")
    return tree


def example_with_openai():
    """Example using OpenAI GPT models."""
    print("=" * 50)
    print("Example: OpenAI")
    print("=" * 50)

    # Requires: export OPENAI_API_KEY="your-key"
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not set, skipping OpenAI example\n")
        return None

    tree = Tree(
        args=TreeArguments(
            root_prompt="Natural Language Processing Techniques",
            model_name="gpt-4o-mini",  # or gpt-4, gpt-3.5-turbo
            model_system_prompt="You are an expert in natural language processing.",
            temperature=0.8,
            tree_degree=4,
            tree_depth=2,
        )
    )

    tree.build()
    print(f"✅ Generated tree with {len(tree.get_all_paths())} paths using OpenAI\n")
    return tree


def example_with_anthropic():
    """Example using Anthropic Claude models."""
    print("=" * 50)
    print("Example: Anthropic Claude")
    print("=" * 50)

    # Requires: export ANTHROPIC_API_KEY="your-key"
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  ANTHROPIC_API_KEY not set, skipping Anthropic example\n")
        return None

    tree = Tree(
        args=TreeArguments(
            root_prompt="Distributed Systems Architecture",
            model_name="claude-3-haiku-20240307",  # or claude-3-opus-20240229, claude-3-sonnet-20240229
            model_system_prompt="You are an expert in distributed systems.",
            temperature=0.6,
            tree_degree=3,
            tree_depth=2,
        )
    )

    tree.build()
    print(f"✅ Generated tree with {len(tree.get_all_paths())} paths using Claude\n")
    return tree


def example_with_google():
    """Example using Google Gemini models."""
    print("=" * 50)
    print("Example: Google Gemini")
    print("=" * 50)

    # Requires: export GEMINI_API_KEY="your-key"
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  GEMINI_API_KEY not set, skipping Gemini example\n")
        return None

    tree = Tree(
        args=TreeArguments(
            root_prompt="Cloud Computing Services",
            model_name="gemini/gemini-pro",  # or gemini/gemini-2.5-flash-lite
            model_system_prompt="You are an expert in cloud computing.",
            temperature=0.7,
            tree_degree=3,
            tree_depth=2,
        )
    )

    tree.build()
    print(f"✅ Generated tree with {len(tree.get_all_paths())} paths using Gemini\n")
    return tree


def example_mixed_models():
    """Example using different models for tree and data generation."""
    print("=" * 50)
    print("Example: Mixed Models (Tree + Data)")
    print("=" * 50)

    # Use a fast, cheap model for tree generation
    tree = Tree(
        args=TreeArguments(
            root_prompt="Database Technologies",
            model_name="ollama/mistral",  # Fast local model for tree
            model_system_prompt="You are an expert in database technologies.",
            temperature=0.8,
            tree_degree=3,
            tree_depth=2,
        )
    )

    print("Building tree with Mistral...")
    tree.build()
    tree.save("database_tree.jsonl")

    # Use a more powerful model for content generation
    engine = DataSetGenerator(
        args=DataSetGeneratorArguments(
            instructions="""Create a detailed technical explanation including:
                          - Core concepts
                          - SQL examples
                          - Performance tips
                          - Use cases""",
            system_prompt="You are a database expert.",
            model_name="ollama/llama3",  # More capable model for content
            prompt_template=None,
            example_data=None,
            temperature=0.3,  # Lower temperature for technical accuracy
            max_retries=3,
            default_batch_size=5,
            default_num_examples=3,
            request_timeout=30,
            sys_msg=True,
        )
    )

    print("Generating dataset with Llama 3...")
    dataset = engine.create_data(
        num_steps=5,
        batch_size=1,
        topic_model=tree,
    )

    dataset.save("database_dataset.jsonl")
    print(f"✅ Generated dataset with {len(dataset.samples)} samples using mixed models\n")


def main():
    """Run examples with different providers."""

    print("\nDeepFabric Multi-Model Examples")
    print("=" * 50)
    print("\nNote: Each provider requires appropriate API keys set as environment variables.")
    print("See each function for specific requirements.\n")

    # Run examples based on available API keys

    # Local model (always available if Ollama is installed)
    try:
        example_with_ollama()
    except Exception as e:
        print(f"❌ Ollama example failed: {e}\n")

    # Cloud providers (require API keys)
    try:
        example_with_openai()
    except Exception as e:
        print(f"❌ OpenAI example failed: {e}\n")

    try:
        example_with_anthropic()
    except Exception as e:
        print(f"❌ Anthropic example failed: {e}\n")

    try:
        example_with_google()
    except Exception as e:
        print(f"❌ Google example failed: {e}\n")

    # Mixed model example
    try:
        example_mixed_models()
    except Exception as e:
        print(f"❌ Mixed models example failed: {e}\n")

    print("\n" + "=" * 50)
    print("Supported Model Formats:")
    print("=" * 50)
    print("""
    Ollama:     ollama/<model-name>
                Examples: ollama/llama3, ollama/mistral, ollama/codellama

    OpenAI:     gpt-4, gpt-4o-mini, gpt-3.5-turbo
                (uses OpenAI format by default)

    Anthropic:  claude-3-opus-20240229
                claude-3-sonnet-20240229
                claude-3-haiku-20240307

    Google:     gemini/gemini-pro
                gemini/gemini-2.5-flash-lite

    Groq:       groq/llama3-8b-8192
                groq/mixtral-8x7b-32768

    Together:   together_ai/meta-llama/Llama-3-70b-chat-hf

    See LiteLLM docs for full list: https://docs.litellm.ai/docs/providers
    """)


if __name__ == "__main__":
    main()
