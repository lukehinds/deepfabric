# Training Frameworks

Integration patterns for TRL and Unsloth.

## TRL (Transformers Reinforcement Learning)

### Basic SFT

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Load and format dataset
dataset = load_dataset("your-username/my-dataset", split="train")

def format_sample(example):
    messages = [{k: v for k, v in m.items() if v is not None}
                for m in example["messages"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

formatted = dataset.map(format_sample)

# Train
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    args=SFTConfig(
        output_dir="./output",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=100,
    ),
)
trainer.train()
```

### With Tool Calling

Include tools in the chat template:

```python
def format_with_tools(example):
    messages = [{k: v for k, v in m.items() if v is not None}
                for m in example["messages"]]
    tools = example.get("tools")

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}
```

## Unsloth

Unsloth provides faster training with lower memory usage.

### Basic Setup

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```

### Training

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=formatted,
    args=SFTConfig(
        output_dir="./output",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
    ),
)
trainer.train()
```

### Saving

```python
# Save LoRA adapter
model.save_pretrained("./lora-adapter")

# Merge and save full model
model.save_pretrained_merged("./merged-model", tokenizer)
```

## Training Tips

**Batch size**: Start small (2-4) and increase if memory allows. Use gradient accumulation for effective larger batches.

**Learning rate**: 2e-5 for full fine-tuning, 2e-4 for LoRA.

**Epochs**: 1-3 epochs is usually sufficient. More can cause overfitting on small datasets.

**Evaluation**: Hold out 10% of data for validation. Monitor loss to detect overfitting.

**Mixed precision**: Use bf16 if supported, otherwise fp16.

## Evaluation During Training

```python
from transformers import TrainerCallback

class EvalCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        print(f"Eval loss: {metrics.get('eval_loss'):.4f}")

trainer = SFTTrainer(
    ...,
    eval_dataset=eval_ds,
    callbacks=[EvalCallback()],
    args=SFTConfig(
        ...,
        eval_strategy="steps",
        eval_steps=100,
    ),
)
```
