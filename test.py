from deepfabric import Dataset

dataset = Dataset.from_hub("lukehinds/huggingface-dataset")

# Format for ANY model
formatted = dataset.format(target_model="Qwen/Qwen2.5-7B-Instruct")
formatted.save("qwen-formatted.jsonl")

# Works with Llama, Mistral, Gemma, etc.
granite = dataset.format(target_model="ibm-granite/granite-4.0-h-1b")
granite.save("granite-formatted.jsonl")
