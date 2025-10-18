"""Unit tests for training configuration classes."""

import pytest

from deepfabric.training.sft_config import LoRAConfig, QuantizationConfig, SFTTrainingConfig


class TestLoRAConfig:
    """Tests for LoRAConfig."""

    def test_default_config(self):
        """Test default LoRA configuration."""
        config = LoRAConfig()

        assert config.enabled is False
        assert config.r == 16  # noqa: PLR2004
        assert config.lora_alpha == 32  # noqa: PLR2004
        assert config.lora_dropout == 0.05  # noqa: PLR2004
        assert config.target_modules == "all-linear"
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"

    def test_custom_config(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(
            enabled=True,
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="all",
        )

        assert config.enabled is True
        assert config.r == 32  # noqa: PLR2004
        assert config.lora_alpha == 64  # noqa: PLR2004
        assert config.lora_dropout == 0.1  # noqa: PLR2004
        assert config.target_modules == ["q_proj", "v_proj"]
        assert config.bias == "all"

    def test_invalid_r(self):
        """Test that invalid r raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error  # noqa: B017
            LoRAConfig(r=0)

        with pytest.raises(Exception):  # noqa: B017
            LoRAConfig(r=300)

    def test_invalid_dropout(self):
        """Test that invalid dropout raises validation error."""
        with pytest.raises(Exception):  # noqa: B017
            LoRAConfig(lora_dropout=-0.1)

        with pytest.raises(Exception):  # noqa: B017
            LoRAConfig(lora_dropout=1.5)


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_config(self):
        """Test default quantization configuration."""
        config = QuantizationConfig()

        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.bnb_4bit_compute_dtype == "bfloat16"
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_quant_type == "nf4"

    def test_8bit_config(self):
        """Test 8-bit quantization configuration."""
        config = QuantizationConfig(load_in_8bit=True)

        assert config.load_in_8bit is True
        assert config.load_in_4bit is False

    def test_4bit_config(self):
        """Test 4-bit quantization configuration."""
        config = QuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="fp4",
        )

        assert config.load_in_4bit is True
        assert config.load_in_8bit is False
        assert config.bnb_4bit_compute_dtype == "float16"
        assert config.bnb_4bit_quant_type == "fp4"

    def test_both_quantization_raises_error(self):
        """Test that enabling both 8-bit and 4-bit raises validation error."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            QuantizationConfig(load_in_8bit=True, load_in_4bit=True)


class TestSFTTrainingConfig:
    """Tests for SFTTrainingConfig."""

    def test_minimal_config(self):
        """Test minimal required configuration."""
        config = SFTTrainingConfig(model_name="test-model", output_dir="./output")

        assert config.model_name == "test-model"
        assert config.output_dir == "./output"
        assert config.num_train_epochs == 3  # noqa: PLR2004
        assert config.per_device_train_batch_size == 4  # noqa: PLR2004
        assert config.learning_rate == 2e-5  # noqa: PLR2004

    def test_custom_training_config(self):
        """Test custom training configuration."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            num_train_epochs=5,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=4,
            learning_rate=1e-5,
            weight_decay=0.1,
            max_seq_length=4096,
        )

        assert config.num_train_epochs == 5  # noqa: PLR2004
        assert config.per_device_train_batch_size == 8  # noqa: PLR2004
        assert config.gradient_accumulation_steps == 4  # noqa: PLR2004
        assert config.learning_rate == 1e-5  # noqa: PLR2004
        assert config.weight_decay == 0.1  # noqa: PLR2004
        assert config.max_seq_length == 4096  # noqa: PLR2004

    def test_lora_config(self):
        """Test LoRA configuration integration."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            lora=LoRAConfig(enabled=True, r=32, lora_alpha=64),
        )

        assert config.lora.enabled is True
        assert config.lora.r == 32  # noqa: PLR2004
        assert config.lora.lora_alpha == 64  # noqa: PLR2004

    def test_quantization_config(self):
        """Test quantization configuration integration."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            quantization=QuantizationConfig(load_in_4bit=True),
        )

        assert config.quantization.load_in_4bit is True
        assert config.quantization.load_in_8bit is False

    def test_both_precision_raises_error(self):
        """Test that enabling both bf16 and fp16 raises validation error."""
        with pytest.raises(ValueError, match="Cannot enable both"):
            SFTTrainingConfig(model_name="test-model", output_dir="./output", bf16=True, fp16=True)

    def test_to_training_arguments(self):
        """Test conversion to TrainingArguments dict."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-5,
            bf16=True,
        )

        args = config.to_training_arguments()

        assert args["output_dir"] == "./output"
        assert args["num_train_epochs"] == 3  # noqa: PLR2004
        assert args["per_device_train_batch_size"] == 4  # noqa: PLR2004
        assert args["learning_rate"] == 2e-5  # noqa: PLR2004
        assert args["bf16"] is True
        assert "model_name" not in args  # Should not include model_name

    def test_hub_config(self):
        """Test HuggingFace Hub configuration."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            push_to_hub=True,
            hub_model_id="username/model",
            hub_strategy="checkpoint",
        )

        assert config.push_to_hub is True
        assert config.hub_model_id == "username/model"
        assert config.hub_strategy == "checkpoint"

        args = config.to_training_arguments()
        assert args["push_to_hub"] is True
        assert args["hub_model_id"] == "username/model"

    def test_additional_training_args(self):
        """Test additional training arguments are merged."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            additional_training_args={"report_to": "wandb", "run_name": "test-run"},
        )

        args = config.to_training_arguments()
        assert args["report_to"] == "wandb"
        assert args["run_name"] == "test-run"

    def test_validation_invalid_epochs(self):
        """Test validation for invalid num_train_epochs."""
        with pytest.raises(Exception):  # Pydantic validation error  # noqa: B017
            SFTTrainingConfig(model_name="test-model", output_dir="./output", num_train_epochs=0)

    def test_validation_invalid_learning_rate(self):
        """Test validation for invalid learning_rate."""
        with pytest.raises(Exception):  # noqa: B017
            SFTTrainingConfig(model_name="test-model", output_dir="./output", learning_rate=0.0)

    def test_qlora_config(self):
        """Test QLoRA (LoRA + 4-bit quantization) configuration."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            lora=LoRAConfig(enabled=True, r=64, lora_alpha=128),
            quantization=QuantizationConfig(load_in_4bit=True),
        )

        assert config.lora.enabled is True
        assert config.quantization.load_in_4bit is True

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing configuration."""
        config = SFTTrainingConfig(
            model_name="test-model", output_dir="./output", gradient_checkpointing=True
        )

        assert config.gradient_checkpointing is True

        args = config.to_training_arguments()
        assert args["gradient_checkpointing"] is True

    def test_evaluation_config(self):
        """Test evaluation configuration."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./output",
            evaluation_strategy="steps",
            eval_steps=100,
            per_device_eval_batch_size=8,
        )

        assert config.evaluation_strategy == "steps"
        assert config.eval_steps == 100  # noqa: PLR2004
        assert config.per_device_eval_batch_size == 8  # noqa: PLR2004

        args = config.to_training_arguments()
        assert args["eval_strategy"] == "steps"
        assert args["eval_steps"] == 100  # noqa: PLR2004
        assert args["per_device_eval_batch_size"] == 8  # noqa: PLR2004
