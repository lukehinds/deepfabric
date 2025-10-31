"""
Comparison: Auto-loading vs. Pre-loaded Tokenizer

This example compares the two approaches for formatting datasets:
1. Auto-loading (simpler, good for quick scripts)
2. Pre-loaded (efficient, ideal for fine-tuning workflows)
"""

import time

from transformers import AutoTokenizer

from deepfabric import Dataset


def approach_1_auto_loading():
    """
    Approach 1: Auto-loading tokenizer (current workflow)

    Pros:
    - Simplest API - just pass model ID
    - Good for quick scripts and one-off formatting
    - No need to import transformers

    Cons:
    - Loads tokenizer internally (redundant if you need it elsewhere)
    - Less control over tokenizer configuration
    """
    print("="*80)
    print("Approach 1: Auto-Loading Tokenizer")
    print("="*80)
    print()

    dataset = Dataset.from_jsonl("dataset.jsonl")

    print("Code:")
    print("  formatted = dataset.format(target_model='Qwen/Qwen2.5-1.5B-Instruct')")
    print()

    start = time.time()
    formatted = dataset.format(target_model="Qwen/Qwen2.5-1.5B-Instruct")
    elapsed = time.time() - start

    print(f"✓ Formatted {len(formatted)} samples in {elapsed:.2f}s")
    print()

    return formatted, elapsed


def approach_2_preloaded_tokenizer():
    """
    Approach 2: Pre-loaded tokenizer (new workflow)

    Pros:
    - Reuse tokenizer you already have
    - More efficient for fine-tuning pipelines
    - Full control over tokenizer settings
    - Can use custom tokenizers

    Cons:
    - Requires importing transformers
    - Slightly more verbose
    """
    print("="*80)
    print("Approach 2: Pre-loaded Tokenizer")
    print("="*80)
    print()

    dataset = Dataset.from_jsonl("dataset.jsonl")

    print("Code:")
    print("  tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')")
    print("  formatted = dataset.format(tokenizer=tokenizer)")
    print()

    # Load tokenizer separately (simulating fine-tuning workflow)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    start = time.time()
    formatted = dataset.format(tokenizer=tokenizer)
    elapsed = time.time() - start

    print(f"✓ Formatted {len(formatted)} samples in {elapsed:.2f}s")
    print()

    return formatted, elapsed, tokenizer


def main():
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "DeepFabric Tokenizer API Comparison" + " "*23 + "║")
    print("╚" + "="*78 + "╝")
    print()

    # Test both approaches
    formatted1, time1 = approach_1_auto_loading()
    formatted2, time2, tokenizer = approach_2_preloaded_tokenizer()

    # Comparison
    print("="*80)
    print("Comparison")
    print("="*80)
    print()

    print(f"Approach 1 (auto-load):  {time1:.2f}s")
    print(f"Approach 2 (pre-loaded): {time2:.2f}s")
    print()

    # Verify outputs are identical
    sample1 = formatted1.samples[0]["text"]
    sample2 = formatted2.samples[0]["text"]

    if sample1 == sample2:
        print("✅ Both approaches produce identical output")
    else:
        print("❌ Outputs differ!")

    print()

    # Use case recommendations
    print("="*80)
    print("When to use each approach?")
    print("="*80)
    print()

    print("🎯 Use Approach 1 (auto-load) when:")
    print("   • Writing quick scripts or notebooks")
    print("   • You don't need the tokenizer elsewhere")
    print("   • Simplicity is priority")
    print()

    print("🚀 Use Approach 2 (pre-loaded) when:")
    print("   • Building fine-tuning pipelines")
    print("   • You already have a tokenizer loaded")
    print("   • Formatting multiple datasets with same tokenizer")
    print("   • Using custom/modified tokenizers")
    print("   • Memory efficiency matters")
    print()

    # Show typical fine-tuning workflow
    print("="*80)
    print("Typical Fine-Tuning Workflow (Approach 2)")
    print("="*80)
    print()
    print("from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer")
    print("from deepfabric import Dataset")
    print()
    print("# 1. Load model & tokenizer")
    print("tokenizer = AutoTokenizer.from_pretrained('model-id')")
    print("model = AutoModelForCausalLM.from_pretrained('model-id')")
    print()
    print("# 2. Format your dataset with SAME tokenizer")
    print("dataset = Dataset.from_jsonl('data.jsonl')")
    print("formatted = dataset.format(tokenizer=tokenizer)  # ← Reuses tokenizer!")
    print()
    print("# 3. Convert & tokenize")
    print("hf_dataset = convert_to_hf(formatted)")
    print("tokenized = hf_dataset.map(lambda x: tokenizer(x['text']))")
    print()
    print("# 4. Train")
    print("trainer = Trainer(model=model, train_dataset=tokenized, tokenizer=tokenizer)")
    print("trainer.train()")
    print()
    print("✨ One tokenizer instance throughout the entire pipeline!")
    print()


if __name__ == "__main__":
    main()
