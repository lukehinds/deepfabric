"""DeepFabric SFT Trainer for supervised fine-tuning with TRL."""

import logging

from collections.abc import Callable
from typing import Any

from ..dataset import Dataset
from ..exceptions import DeepFabricError
from .sft_config import SFTTrainingConfig

logger = logging.getLogger(__name__)


class DeepFabricSFTTrainer:
    """Trainer for supervised fine-tuning using TRL's SFTTrainer.

    This class wraps HuggingFace TRL's SFTTrainer to provide seamless
    integration with DeepFabric datasets. It handles:
    - Dataset formatting and preparation
    - Model and tokenizer loading
    - LoRA/QLoRA configuration
    - Training execution
    - Model saving and Hub upload

    Example:
        ```python
        from deepfabric import DataSetGenerator, Tree
        from deepfabric.training import DeepFabricSFTTrainer, SFTTrainingConfig

        # Generate dataset
        tree = Tree(topic_prompt="Python", provider="openai", model_name="gpt-4o-mini")
        tree_paths = await tree.build_async()

        generator = DataSetGenerator(
            generation_system_prompt="Create Q&A pairs",
            provider="openai",
            model_name="gpt-4o-mini"
        )
        dataset = await generator.create_data_async(
            num_steps=100,
            batch_size=5,
            topic_model=tree
        )

        # Train model
        config = SFTTrainingConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            lora=LoRAConfig(enabled=True, r=16)
        )

        trainer = DeepFabricSFTTrainer(config=config, train_dataset=dataset)
        trainer.train()
        trainer.save_model()
        ```
    """

    def __init__(
        self,
        config: SFTTrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        formatting_func: str | Callable | None = "default",
    ):
        """Initialize the SFT trainer.

        Args:
            config: Training configuration
            train_dataset: DeepFabric Dataset for training
            eval_dataset: Optional Dataset for evaluation
            formatting_func: Function to format samples for training:
                - "default": Use messages field directly
                - "trl_sft_tools": Apply TRL SFT tools formatter
                - callable: Custom formatting function
                - None: No formatting (use raw data)

        Raises:
            DeepFabricError: If dependencies are missing or initialization fails
        """
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.formatting_func_name = formatting_func

        # Validate dependencies
        self._validate_dependencies()

        # Initialize trainer components
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.peft_config = None

        logger.info(
            "Initializing DeepFabricSFTTrainer with model: %s",
            self.config.model_name,
        )

        # Setup model, tokenizer, and trainer
        self._setup_model_and_tokenizer()
        self._prepare_datasets()
        self._setup_trainer()

    def _validate_dependencies(self):
        """Validate that required training dependencies are installed."""
        try:
            import transformers  # noqa: F401, PLC0415
        except ImportError as e:
            msg = (
                "transformers library not found. Install with: "
                "pip install 'deepfabric[training]' or pip install transformers>=4.45.0"
            )
            raise DeepFabricError(msg) from e

        try:
            import trl  # noqa: F401, PLC0415
        except ImportError as e:
            msg = (
                "trl library not found. Install with: "
                "pip install 'deepfabric[training]' or pip install trl>=0.11.0"
            )
            raise DeepFabricError(msg) from e

        try:
            import torch  # noqa: F401, PLC0415
        except ImportError as e:
            msg = (
                "torch library not found. Install with: "
                "pip install 'deepfabric[training]' or pip install torch>=2.0.0"
            )
            raise DeepFabricError(msg) from e

        # Check for PEFT if LoRA is enabled
        if self.config.lora.enabled:
            try:
                import peft  # noqa: F401, PLC0415
            except ImportError as e:
                msg = (
                    "peft library not found (required for LoRA). Install with: "
                    "pip install 'deepfabric[training]' or pip install peft>=0.13.0"
                )
                raise DeepFabricError(msg) from e

        # Check for bitsandbytes if quantization is enabled
        if self.config.quantization.load_in_8bit or self.config.quantization.load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401, PLC0415
            except ImportError as e:
                msg = (
                    "bitsandbytes library not found (required for quantization). "
                    "Install with: pip install 'deepfabric[training]' or pip install bitsandbytes>=0.43.0"
                )
                raise DeepFabricError(msg) from e

    def _setup_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate configuration."""
        try:
            import torch  # noqa: PLC0415

            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            # Prepare model loading kwargs
            model_kwargs = {
                **self.config.model_kwargs,
            }

            # Handle quantization
            if self.config.quantization.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("Loading model in 8-bit quantized format")
            elif self.config.quantization.load_in_4bit:
                from transformers import BitsAndBytesConfig  # noqa: PLC0415

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=self.config.quantization.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.quantization.bnb_4bit_use_double_quant,
                    bnb_4bit_compute_dtype=(
                        torch.bfloat16
                        if self.config.quantization.bnb_4bit_compute_dtype == "bfloat16"
                        else torch.float16
                    ),
                )
                model_kwargs["quantization_config"] = bnb_config
                logger.info("Loading model in 4-bit quantized format (QLoRA)")

            # Handle precision
            if self.config.bf16:
                model_kwargs["dtype"] = torch.bfloat16
            elif self.config.fp16:
                model_kwargs["dtype"] = torch.float16

            # Load model
            logger.info("Loading model: %s", self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                **model_kwargs,
            )

            # Enable gradient checkpointing if configured
            if self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                logger.info("Enabled gradient checkpointing")

            # Load tokenizer
            logger.info("Loading tokenizer: %s", self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Setup LoRA if enabled
            if self.config.lora.enabled:
                self._setup_lora()

            logger.info("Successfully loaded model and tokenizer")

        except Exception as e:
            msg = f"Failed to setup model and tokenizer: {str(e)}"
            raise DeepFabricError(msg) from e

    def _setup_lora(self):
        """Configure LoRA for parameter-efficient fine-tuning."""
        try:
            from peft import LoraConfig  # noqa: PLC0415

            lora_config = LoraConfig(
                r=self.config.lora.r,
                lora_alpha=self.config.lora.lora_alpha,
                lora_dropout=self.config.lora.lora_dropout,
                target_modules=self.config.lora.target_modules,
                bias=self.config.lora.bias,
                task_type=self.config.lora.task_type,
            )

            self.peft_config = lora_config
            logger.info(
                "Configured LoRA with r=%d, alpha=%d, dropout=%.2f",
                self.config.lora.r,
                self.config.lora.lora_alpha,
                self.config.lora.lora_dropout,
            )

        except Exception as e:
            msg = f"Failed to setup LoRA: {str(e)}"
            raise DeepFabricError(msg) from e

    def _prepare_datasets(self):
        """Prepare datasets for training."""
        try:
            # Apply formatting if specified
            if self.formatting_func_name and self.formatting_func_name != "default":
                if self.formatting_func_name == "trl_sft_tools":
                    # Use TRL SFT tools formatter
                    formatted_datasets = self.train_dataset.apply_formatters(
                        [
                            {
                                "name": "trl_sft_tools",
                                "template": "builtin://trl_sft_tools",
                                "config": {},
                            }
                        ]
                    )
                    self.train_dataset = formatted_datasets["trl_sft_tools"]
                    logger.info("Applied TRL SFT tools formatter to training dataset")

                    if self.eval_dataset:
                        formatted_eval_datasets = self.eval_dataset.apply_formatters(
                            [
                                {
                                    "name": "trl_sft_tools",
                                    "template": "builtin://trl_sft_tools",
                                    "config": {},
                                }
                            ]
                        )
                        self.eval_dataset = formatted_eval_datasets["trl_sft_tools"]
                        logger.info("Applied TRL SFT tools formatter to eval dataset")

                elif callable(self.formatting_func_name):
                    # Custom formatting function
                    logger.info("Using custom formatting function")
                    # Formatting will be handled by SFTTrainer
                else:
                    logger.warning(
                        "Unknown formatting function: %s, using default",
                        self.formatting_func_name,
                    )

            # Convert to HuggingFace Dataset format
            from datasets import Dataset as HFDataset  # noqa: PLC0415

            self.hf_train_dataset = HFDataset.from_list(self.train_dataset.samples)
            logger.info(
                "Converted training dataset to HuggingFace format: %d samples",
                len(self.hf_train_dataset),
            )

            if self.eval_dataset:
                self.hf_eval_dataset = HFDataset.from_list(self.eval_dataset.samples)
                logger.info(
                    "Converted eval dataset to HuggingFace format: %d samples",
                    len(self.hf_eval_dataset),
                )
            else:
                self.hf_eval_dataset = None

        except Exception as e:
            msg = f"Failed to prepare datasets: {str(e)}"
            raise DeepFabricError(msg) from e

    def _setup_trainer(self):
        """Initialize the TRL SFTTrainer."""
        try:
            from transformers import TrainingArguments  # noqa: PLC0415
            from trl import SFTTrainer  # noqa: PLC0415

            # Create TrainingArguments
            training_args = TrainingArguments(**self.config.to_training_arguments())

            # Prepare SFTTrainer arguments
            sft_args = {
                "model": self.model,
                "args": training_args,
                "train_dataset": self.hf_train_dataset,
                "tokenizer": self.tokenizer,
                "max_seq_length": self.config.max_seq_length,
                "packing": self.config.packing,
            }

            # Add LoRA config if enabled
            if self.peft_config:
                sft_args["peft_config"] = self.peft_config

            # Add eval dataset if provided
            if self.hf_eval_dataset is not None:
                sft_args["eval_dataset"] = self.hf_eval_dataset

            # Add formatting function if custom
            if callable(self.formatting_func_name):
                sft_args["formatting_func"] = self.formatting_func_name

            # Create SFTTrainer
            self.trainer = SFTTrainer(**sft_args)

            logger.info("Successfully initialized SFTTrainer")

        except Exception as e:
            msg = f"Failed to setup trainer: {str(e)}"
            raise DeepFabricError(msg) from e

    def train(self) -> dict[str, Any]:
        """Execute the training loop.

        Returns:
            Training metrics dictionary

        Raises:
            DeepFabricError: If training fails
        """
        if self.trainer is None:
            msg = "Trainer not initialized. Call _setup_trainer() first."
            raise DeepFabricError(msg)

        try:
            logger.info("Starting training...")
            train_result = self.trainer.train()
        except Exception as e:
            msg = f"Training failed: {str(e)}"
            raise DeepFabricError(msg) from e
        else:
            logger.info("Training completed successfully")
            return train_result.metrics

    def save_model(self, output_dir: str | None = None):
        """Save the trained model and tokenizer.

        Args:
            output_dir: Directory to save the model (defaults to config.output_dir)

        Raises:
            DeepFabricError: If saving fails
        """
        if self.trainer is None:
            msg = "Trainer not initialized"
            raise DeepFabricError(msg)

        save_dir = output_dir or self.config.output_dir

        try:
            logger.info("Saving model to: %s", save_dir)
            self.trainer.save_model(save_dir)

            # Save tokenizer separately if available
            if self.tokenizer is not None:
                logger.info("Saving tokenizer to: %s", save_dir)
                try:
                    self.tokenizer.save_pretrained(save_dir)
                    logger.info("Model and tokenizer saved successfully")
                except Exception as e:
                    # Tokenizer save failed but model was saved; log a warning instead of raising
                    logger.warning("Failed to save tokenizer: %s", str(e))
                    logger.info("Model saved successfully (tokenizer not saved)")
            else:
                logger.warning("No tokenizer available to save")
                logger.info("Model saved successfully (no tokenizer to save)")

        except Exception as e:
            msg = f"Failed to save model: {str(e)}"
            raise DeepFabricError(msg) from e

    def push_to_hub(self, repo_id: str | None = None, **kwargs):
        """Push the trained model to HuggingFace Hub.

        Args:
            repo_id: Repository ID on HuggingFace Hub (defaults to config.hub_model_id)
            **kwargs: Additional arguments passed to push_to_hub

        Raises:
            DeepFabricError: If push fails
        """
        if self.trainer is None:
            msg = "Trainer not initialized"
            raise DeepFabricError(msg)

        hub_id = repo_id or self.config.hub_model_id
        if not hub_id:
            msg = "No hub_model_id specified"
            raise DeepFabricError(msg)

        try:
            logger.info("Pushing model to HuggingFace Hub: %s", hub_id)
            self.trainer.push_to_hub(repo_id=hub_id, **kwargs)

            logger.info("Model pushed to Hub successfully")

        except Exception as e:
            msg = f"Failed to push to Hub: {str(e)}"
            raise DeepFabricError(msg) from e

    def evaluate(self) -> dict[str, Any]:
        """Evaluate the model on the evaluation dataset.

        Returns:
            Evaluation metrics dictionary

        Raises:
            DeepFabricError: If evaluation fails or no eval dataset
        """
        if self.trainer is None:
            msg = "Trainer not initialized"
            raise DeepFabricError(msg)

        if self.hf_eval_dataset is None:
            msg = "No evaluation dataset provided"
            raise DeepFabricError(msg)

        try:
            logger.info("Evaluating model...")
            eval_metrics = self.trainer.evaluate()
        except Exception as e:
            msg = f"Evaluation failed: {str(e)}"
            raise DeepFabricError(msg) from e
        else:
            logger.info("Evaluation completed")
            return eval_metrics

    def __repr__(self) -> str:
        """Return string representation of the trainer."""
        return (
            f"DeepFabricSFTTrainer("
            f"model={self.config.model_name}, "
            f"train_samples={len(self.train_dataset)}, "
            f"lora={'enabled' if self.config.lora.enabled else 'disabled'})"
        )
