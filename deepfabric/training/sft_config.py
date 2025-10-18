"""Configuration for supervised fine-tuning with TRL."""

import logging

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class LoRAConfig(BaseModel):
    """Configuration for LoRA (Low-Rank Adaptation) fine-tuning.

    LoRA enables parameter-efficient fine-tuning by training low-rank
    decomposition matrices instead of the full model weights.
    """

    enabled: bool = Field(default=False, description="Enable LoRA fine-tuning")

    r: int = Field(
        default=16,
        ge=1,
        le=256,
        description="LoRA attention dimension (rank)",
    )

    lora_alpha: int = Field(
        default=32,
        ge=1,
        le=512,
        description="LoRA alpha parameter for scaling",
    )

    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LoRA layers",
    )

    target_modules: list[str] | Literal["all-linear"] = Field(
        default="all-linear",
        description="Target modules for LoRA (e.g., ['q_proj', 'v_proj'] or 'all-linear')",
    )

    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Bias training strategy",
    )

    task_type: str = Field(
        default="CAUSAL_LM",
        description="Task type for PEFT (usually CAUSAL_LM)",
    )


class QuantizationConfig(BaseModel):
    """Configuration for model quantization."""

    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit precision (requires bitsandbytes)",
    )

    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit precision (QLoRA, requires bitsandbytes)",
    )

    bnb_4bit_compute_dtype: Literal["float16", "bfloat16"] | None = Field(
        default="bfloat16",
        description="Compute dtype for 4-bit quantization",
    )

    bnb_4bit_use_double_quant: bool = Field(
        default=True,
        description="Use double quantization for 4-bit",
    )

    bnb_4bit_quant_type: Literal["fp4", "nf4"] = Field(
        default="nf4",
        description="Quantization type for 4-bit (nf4 recommended)",
    )

    @model_validator(mode="after")
    def validate_quantization(self) -> "QuantizationConfig":
        """Ensure only one quantization method is enabled."""
        if self.load_in_8bit and self.load_in_4bit:
            msg = "Cannot enable both 8-bit and 4-bit quantization"
            raise ValueError(msg)
        return self


class SFTTrainingConfig(BaseModel):
    """Configuration for Supervised Fine-Tuning with TRL's SFTTrainer.

    This configuration wraps HuggingFace's TrainingArguments and adds
    SFT-specific parameters for training models with DeepFabric datasets.

    Example:
        ```python
        config = SFTTrainingConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            lora=LoRAConfig(enabled=True, r=16, lora_alpha=32)
        )
        ```
    """

    # Model configuration
    model_name: str = Field(
        ...,
        min_length=1,
        description="HuggingFace model identifier or local path",
    )

    # Training output
    output_dir: str = Field(
        ...,
        min_length=1,
        description="Directory to save model checkpoints and outputs",
    )

    # Training hyperparameters
    num_train_epochs: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of training epochs",
    )

    per_device_train_batch_size: int = Field(
        default=4,
        ge=1,
        le=128,
        description="Training batch size per device",
    )

    per_device_eval_batch_size: int | None = Field(
        default=None,
        ge=1,
        le=128,
        description="Evaluation batch size per device (defaults to train batch size)",
    )

    gradient_accumulation_steps: int = Field(
        default=1,
        ge=1,
        le=512,
        description="Number of gradient accumulation steps",
    )

    learning_rate: float = Field(
        default=2e-5,
        ge=1e-7,
        le=1e-2,
        description="Learning rate for optimizer",
    )

    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Weight decay for regularization",
    )

    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of warmup steps for learning rate scheduler",
    )

    warmup_ratio: float = Field(
        default=0.03,
        ge=0.0,
        le=1.0,
        description="Ratio of warmup steps to total steps",
    )

    max_grad_norm: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum gradient norm for clipping",
    )

    # Optimization
    optim: str = Field(
        default="adamw_torch",
        description="Optimizer to use (adamw_torch, adamw_8bit, paged_adamw_8bit, etc.)",
    )

    lr_scheduler_type: str = Field(
        default="cosine",
        description="Learning rate scheduler type (linear, cosine, constant, etc.)",
    )

    # Checkpointing and logging
    save_strategy: Literal["no", "epoch", "steps"] = Field(
        default="epoch",
        description="Checkpoint save strategy",
    )

    save_steps: int = Field(
        default=500,
        ge=1,
        description="Save checkpoint every X steps (if save_strategy='steps')",
    )

    save_total_limit: int | None = Field(
        default=3,
        ge=1,
        description="Maximum number of checkpoints to keep",
    )

    logging_steps: int = Field(
        default=10,
        ge=1,
        description="Log metrics every X steps",
    )

    logging_strategy: Literal["no", "epoch", "steps"] = Field(
        default="steps",
        description="Logging strategy",
    )

    # Evaluation
    evaluation_strategy: Literal["no", "epoch", "steps"] = Field(
        default="no",
        description="Evaluation strategy during training",
    )

    eval_steps: int | None = Field(
        default=None,
        ge=1,
        description="Evaluate every X steps (if evaluation_strategy='steps')",
    )

    # Device and memory
    bf16: bool = Field(
        default=False,
        description="Use bfloat16 precision",
    )

    fp16: bool = Field(
        default=False,
        description="Use float16 precision",
    )

    gradient_checkpointing: bool = Field(
        default=False,
        description="Enable gradient checkpointing to save memory",
    )

    # SFT-specific configuration
    max_seq_length: int = Field(
        default=2048,
        ge=128,
        le=131072,
        description="Maximum sequence length for packing",
    )

    packing: bool = Field(
        default=False,
        description="Pack multiple samples into single sequences",
    )

    # Dataset configuration
    dataset_text_field: str = Field(
        default="text",
        description="Field name for text data in dataset",
    )

    # LoRA and Quantization
    lora: LoRAConfig = Field(
        default_factory=LoRAConfig,
        description="LoRA configuration for parameter-efficient fine-tuning",
    )

    quantization: QuantizationConfig = Field(
        default_factory=QuantizationConfig,
        description="Model quantization configuration",
    )

    # Hub integration
    push_to_hub: bool = Field(
        default=False,
        description="Push model to HuggingFace Hub after training",
    )

    hub_model_id: str | None = Field(
        default=None,
        description="HuggingFace Hub model ID for pushing",
    )

    hub_strategy: Literal["end", "checkpoint", "every_save", "all_checkpoints"] = Field(
        default="end",
        description="Strategy for pushing to Hub",
    )

    # Additional training arguments
    additional_training_args: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional arguments passed to TrainingArguments",
    )

    # Model loading kwargs
    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional kwargs for model loading",
    )

    @model_validator(mode="after")
    def validate_precision(self) -> "SFTTrainingConfig":
        """Ensure only one precision type is enabled."""
        if self.bf16 and self.fp16:
            msg = "Cannot enable both bf16 and fp16"
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def validate_lora_and_quantization(self) -> "SFTTrainingConfig":
        """Validate LoRA and quantization compatibility."""
        if self.lora.enabled and (self.quantization.load_in_8bit or self.quantization.load_in_4bit):
            logger.info("Using QLoRA (LoRA + Quantization)")
        return self

    def to_training_arguments(self) -> dict[str, Any]:
        """Convert to TrainingArguments-compatible dict.

        Returns:
            Dictionary of arguments for HuggingFace TrainingArguments
        """
        args = {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "max_grad_norm": self.max_grad_norm,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "save_total_limit": self.save_total_limit,
            "logging_steps": self.logging_steps,
            "logging_strategy": self.logging_strategy,
            "eval_strategy": self.evaluation_strategy,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "push_to_hub": self.push_to_hub,
            "hub_strategy": self.hub_strategy,
        }

        # Add optional parameters
        if self.per_device_eval_batch_size:
            args["per_device_eval_batch_size"] = self.per_device_eval_batch_size

        if self.eval_steps:
            args["eval_steps"] = self.eval_steps

        if self.hub_model_id:
            args["hub_model_id"] = self.hub_model_id

        # Merge additional training arguments
        args.update(self.additional_training_args)

        return args
