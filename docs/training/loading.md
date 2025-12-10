# Loading Datasets

DeepFabric outputs standard JSONL files compatible with HuggingFace datasets.

## From Local File

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="dataset.jsonl", split="train")
```

## From HuggingFace Hub

Upload first:

```bash
deepfabric upload dataset.jsonl --repo your-username/my-dataset
```

Then load:

```python
dataset = load_dataset("your-username/my-dataset", split="train")
```

## Train/Test Split

```python
splits = dataset.train_test_split(test_size=0.1, seed=42)
train_ds = splits["train"]
eval_ds = splits["test"]
```

## Accessing Fields

```python
for sample in dataset:
    messages = sample["messages"]
    reasoning = sample.get("reasoning")
    tools = sample.get("tools")

    # Messages structure
    for msg in messages:
        role = msg["role"]      # system, user, assistant, tool
        content = msg["content"]
        tool_calls = msg.get("tool_calls")
```

## Filtering

Keep only samples with tool calls:

```python
with_tools = dataset.filter(lambda x: x.get("tools") is not None)
```

Filter by reasoning style:

```python
agent_samples = dataset.filter(
    lambda x: x.get("reasoning", {}).get("style") == "agent"
)
```

## Shuffling

```python
shuffled = dataset.shuffle(seed=42)
```

## Streaming Large Datasets

For datasets that don't fit in memory:

```python
dataset = load_dataset(
    "your-username/large-dataset",
    split="train",
    streaming=True
)

for sample in dataset:
    # Process one at a time
    process(sample)
```

## Combining Datasets

Merge multiple DeepFabric datasets:

```python
from datasets import concatenate_datasets

basic = load_dataset("user/basic-ds", split="train")
agent = load_dataset("user/agent-ds", split="train")

combined = concatenate_datasets([basic, agent])
combined = combined.shuffle(seed=42)
```

## Inspection

```python
# Dataset info
print(dataset)
# Dataset({features: ['messages', 'reasoning', 'tools', ...], num_rows: 1000})

# First sample
print(dataset[0])

# Column names
print(dataset.column_names)
# ['messages', 'reasoning', 'tools', 'metadata', ...]
```
