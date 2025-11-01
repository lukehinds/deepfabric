"""
Example: Fine-tuning workflow using pre-loaded tokenizer

This example demonstrates how to use DeepFabric's Dataset.format() method
with a pre-loaded tokenizer for efficient fine-tuning workflows.

Benefits:
- Reuse the same tokenizer instance for formatting and training
- Avoid redundant tokenizer downloads/loading
- Ensure consistency across your pipeline
- Cleaner, more maintainable code
"""

from datasets import Dataset as HFDataset
from transformers import AutoTokenizer, TrainingArguments

from deepfabric import Dataset

# Constants
PREVIEW_LENGTH = 200


def main():
    print("="*80)
    print("DeepFabric + Transformers Fine-Tuning Workflow")
    print("="*80)
    print()

    # ============================================================================
    # Step 1: Load model and tokenizer
    # ============================================================================
    print("📦 Step 1: Loading model and tokenizer...")
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # Uncomment to load model for actual training:
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     dtype="auto",
    #     device_map="auto"
    # )

    print(f"   ✓ Tokenizer: {tokenizer.__class__.__name__}")
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")
    print()

    # ============================================================================
    # Step 2: Load DeepFabric dataset
    # ============================================================================
    print("📂 Step 2: Loading DeepFabric dataset...")

    # Option A: From local JSONL file
    dataset = Dataset.from_jsonl("dataset.jsonl")

    # Option B: From HuggingFace Hub
    # dataset = Dataset.from_hub("your-username/your-dataset")

    print(f"   ✓ Loaded {len(dataset)} samples")
    print()

    # ============================================================================
    # Step 3: Format dataset using YOUR pre-loaded tokenizer
    # ============================================================================
    print("🎨 Step 3: Formatting with chat template...")

    # 🎯 KEY FEATURE: Pass your existing tokenizer!
    # No redundant loading - uses the same instance you'll train with
    formatted = dataset.format(tokenizer=tokenizer)

    print(f"   ✓ Formatted {len(formatted)} samples")
    print()

    # Preview first sample
    if formatted.samples:
        sample = formatted.samples[0]["text"]
        preview = sample[:PREVIEW_LENGTH] + "..." if len(sample) > PREVIEW_LENGTH else sample
        print("   📝 Sample preview:")
        print("   " + "-"*76)
        for line in preview.split('\n'):
            print(f"   {line}")
        print("   " + "-"*76)
        print()

    # ============================================================================
    # Step 4: Convert to HuggingFace Dataset format
    # ============================================================================
    print("🔄 Step 4: Converting to HuggingFace Dataset...")

    hf_dataset = HFDataset.from_dict({
        "text": [sample["text"] for sample in formatted.samples]
    })

    print(f"   ✓ Created HF Dataset with {len(hf_dataset)} samples")
    print()

    # ============================================================================
    # Step 5: Tokenize for training
    # ============================================================================
    print("🔢 Step 5: Tokenizing for training...")

    def tokenize_fn(examples):
        """Tokenize text samples for causal language modeling."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors=None
        )

    tokenized_dataset = hf_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing samples"
    )

    print(f"   ✓ Tokenized {len(tokenized_dataset)} samples")
    print()

    # ============================================================================
    # Step 6: Setup training (optional)
    # ============================================================================
    print("⚙️  Step 6: Training setup...")

    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-5,
        logging_steps=10,
        save_strategy="epoch",
        # Add more args as needed
    )

    print("   ✓ Training args configured")
    print(f"   • Batch size: {training_args.per_device_train_batch_size}")
    print(f"   • Epochs: {training_args.num_train_epochs}")
    print(f"   • Learning rate: {training_args.learning_rate}")
    print()

    # Uncomment to start training:
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_dataset,
    #     tokenizer=tokenizer
    # )
    # trainer.train()

    print("="*80)
    print("✨ Setup complete! Ready for training.")
    print("="*80)
    print()
    print("Key takeaways:")
    print("  1. ✅ Single tokenizer instance used throughout")
    print("  2. ✅ No redundant tokenizer loading")
    print("  3. ✅ Guaranteed consistency between formatting and training")
    print("  4. ✅ Clean, efficient workflow")
    print()
    print("Alternative API (auto-loads tokenizer):")
    print("  formatted = dataset.format(target_model='Qwen/Qwen2.5-1.5B-Instruct')")
    print()

if __name__ == "__main__":
    main()
