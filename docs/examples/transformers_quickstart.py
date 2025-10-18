"""Quickstart example for HuggingFace Transformers integration with DeepFabric.

This example demonstrates:
1. Using a local Transformers model for topic generation
2. Generating a synthetic dataset
3. Fine-tuning the model with LoRA
4. Saving the results

Requirements:
    pip install 'deepfabric[training]'
"""

import asyncio

from deepfabric import DataSetGenerator, Tree
from deepfabric.training import DeepFabricSFTTrainer, LoRAConfig, SFTTrainingConfig


async def main():
    """Run complete workflow with local Transformers model."""

    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # noqa: N806
    TOPIC = "Python Programming Best Practices"  # noqa: N806
    NUM_SAMPLES = 100  # noqa: N806

    print("=" * 80)
    print("DeepFabric Transformers Integration - Quickstart")
    print("=" * 80)

    # Step 1: Generate Topic Tree
    print("\n[Step 1/4] Generating topic tree with local model...")
    print(f"Model: {MODEL_NAME}")
    print(f"Topic: {TOPIC}")

    tree = Tree(
        topic_prompt=TOPIC,
        provider="transformers",
        model_name=MODEL_NAME,
        device="cuda",  # Change to "cpu" if no GPU available
        torch_dtype="bfloat16",  # Use bfloat16 for efficiency
        depth=2,  # Shallow tree for quick demo
        degree=5,  # 5 subtopics per node
    )

    # Build tree (async)
    async for event in tree.build_async():
        if event.get("event") == "build_complete":
            total_paths = event.get("total_paths", 0)
            print(f"✓ Topic tree built: {total_paths} paths generated")

    # Step 2: Generate Dataset
    print(f"\n[Step 2/4] Generating {NUM_SAMPLES} training samples...")

    generator = DataSetGenerator(
        generation_system_prompt="Create high-quality Python programming Q&A pairs",
        provider="transformers",
        model_name=MODEL_NAME,
        device="cuda",
        torch_dtype="bfloat16",
        conversation_type="basic",  # Simple Q&A format
        temperature=0.7,
    )

    # Calculate batch parameters
    batch_size = 5
    num_steps = (NUM_SAMPLES + batch_size - 1) // batch_size

    dataset = await generator.create_data_async(
        num_steps=num_steps,
        batch_size=batch_size,
        topic_model=tree,
    )

    print(f"✓ Dataset generated: {len(dataset)} samples")

    # Save dataset
    dataset.save("python_qa_dataset.jsonl")
    print("✓ Dataset saved to: python_qa_dataset.jsonl")

    # Step 3: Fine-tune Model (Optional - comment out if you don't want to train)
    print("\n[Step 3/4] Fine-tuning model with LoRA...")

    training_config = SFTTrainingConfig(
        model_name=MODEL_NAME,
        output_dir="./python-tutor-model",
        # Training parameters (minimal for demo)
        num_train_epochs=1,  # Just 1 epoch for quick demo
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        # Memory optimization
        bf16=True,
        gradient_checkpointing=True,
        # LoRA configuration
        lora=LoRAConfig(enabled=True, r=8, lora_alpha=16, lora_dropout=0.05),
        # Logging
        logging_steps=5,
        save_strategy="epoch",
    )

    trainer = DeepFabricSFTTrainer(
        config=training_config, train_dataset=dataset, formatting_func="default"
    )

    metrics = trainer.train()
    print(f"✓ Training complete. Loss: {metrics.get('train_loss', 'N/A')}")

    # Step 4: Save Model
    print("\n[Step 4/4] Saving fine-tuned model...")
    trainer.save_model()
    print("✓ Model saved to: ./python-tutor-model")

    # Summary
    print("\n" + "=" * 80)
    print("Complete! Summary:")
    print(f"  - Topic paths generated: {len(tree.get_all_paths())}")
    print(f"  - Training samples: {len(dataset)}")
    print("  - Model saved to: ./python-tutor-model")
    print("  - Dataset saved to: python_qa_dataset.jsonl")
    print("=" * 80)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
