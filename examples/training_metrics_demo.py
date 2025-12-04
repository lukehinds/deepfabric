#!/usr/bin/env python3
"""Demo script for DeepFabric training metrics logging.

This script trains a tiny model and sends metrics to a local server.

Usage:
    # Terminal 1: Start the mock server
    python examples/mock_metrics_server.py

    # Terminal 2: Run the training demo
    DEEPFABRIC_API_URL=http://localhost:8888 python examples/training_metrics_demo.py
"""

from __future__ import annotations

import os
import shutil

# Configure API URL (set before importing deepfabric)
os.environ.setdefault("DEEPFABRIC_API_URL", "http://localhost:8888")
# Optionally set API key via env var (or be prompted interactively)
os.environ.setdefault("DEEPFABRIC_API_KEY", "test-api-key-12345")

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from deepfabric.training import DeepFabricCallback


def main():
    print("=" * 60)
    print("  DeepFabric Training Metrics Demo")
    print("=" * 60)
    print()
    print(f"API URL: {os.environ.get('DEEPFABRIC_API_URL')}")
    print(f"API Key: {os.environ.get('DEEPFABRIC_API_KEY', '')[:10]}...")
    print()

    # Use tiny model for fast testing (~500KB)
    model_name = "sshleifer/tiny-gpt2"

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Model parameters: {model.num_parameters():,}")
    print()

    # Create tiny synthetic dataset
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language.",
        "Deep learning models require large amounts of data.",
        "Natural language processing enables computers to understand text.",
        "Neural networks are inspired by the human brain.",
        "Training large models requires significant compute resources.",
        "Fine-tuning adapts pre-trained models to specific tasks.",
    ] * 10  # 80 samples total

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=64,
            padding="max_length",
        )

    dataset = Dataset.from_dict({"text": texts})
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    print(f"Dataset size: {len(tokenized_dataset)} samples")

    # Training arguments - very short training for demo
    training_args = TrainingArguments(
        output_dir="./demo_output",
        num_train_epochs=2,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        warmup_steps=5,
        learning_rate=5e-4,
        logging_steps=5,  # Log every 5 steps
        save_strategy="no",  # Don't save checkpoints
        report_to=[],  # Disable default reporters
        use_cpu=True,  # Use CPU for simplicity
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Add DeepFabric callback for metrics logging
    trainer.add_callback(DeepFabricCallback(trainer))

    # Check what callbacks are registered
    print("\nRegistered callbacks:")
    for cb in trainer.callback_handler.callbacks:
        print(f"  - {type(cb).__name__}")
    print()

    # Train!
    print("=" * 60)
    print("  Starting training...")
    print("  Watch your mock server for incoming metrics!")
    print("=" * 60)
    print()

    trainer.train()

    print()
    print("=" * 60)
    print("  Training complete!")
    print("=" * 60)

    # Clean up
    shutil.rmtree("./demo_output", ignore_errors=True)


if __name__ == "__main__":
    main()
