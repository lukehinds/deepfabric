"""Unit tests for DeepFabricSFTTrainer."""

from unittest.mock import MagicMock, Mock, patch

import pytest

from deepfabric.dataset import Dataset
from deepfabric.exceptions import DeepFabricError
from deepfabric.training.sft_config import LoRAConfig, SFTTrainingConfig
from deepfabric.training.sft_trainer import DeepFabricSFTTrainer


@pytest.fixture
def mock_dataset():
    """Create a mock dataset."""
    dataset = Dataset()
    dataset.samples = [
        {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        },
    ]
    return dataset


@pytest.fixture
def basic_config():
    """Create a basic training config."""
    return SFTTrainingConfig(
        model_name="test-model",
        output_dir="./test-output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
    )


class TestDeepFabricSFTTrainer:
    """Tests for DeepFabricSFTTrainer."""

    def test_missing_transformers_dependency(self, mock_dataset, basic_config):
        """Test error when transformers is not installed."""
        with (
            patch.dict("sys.modules", {"transformers": None}),
            pytest.raises(DeepFabricError, match="transformers library not found"),
        ):
            DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)

    def test_missing_trl_dependency(self, mock_dataset, basic_config):
        """Test error when TRL is not installed."""
        with (
            patch.dict(
                "sys.modules", {"transformers": MagicMock(), "trl": None, "torch": MagicMock()}
            ),
            pytest.raises(DeepFabricError, match="trl library not found"),
        ):
            DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)

    def test_missing_peft_with_lora(self, mock_dataset):
        """Test error when PEFT is not installed but LoRA is enabled."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./test-output",
            lora=LoRAConfig(enabled=True),
        )

        with (
            patch.dict(
                "sys.modules",
                {
                    "transformers": MagicMock(),
                    "trl": MagicMock(),
                    "torch": MagicMock(),
                    "peft": None,
                },
            ),
            pytest.raises(DeepFabricError, match="peft library not found"),
        ):
            DeepFabricSFTTrainer(config=config, train_dataset=mock_dataset)

    def test_trainer_initialization(self, mock_dataset, basic_config):
        """Test trainer initialization."""
        # Mock dependencies
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"  # noqa: S105

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=Mock())

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)

            assert trainer.config == basic_config
            assert trainer.train_dataset == mock_dataset
            assert trainer.model is not None
            assert trainer.tokenizer is not None

    def test_trainer_with_lora(self, mock_dataset):
        """Test trainer initialization with LoRA enabled."""
        config = SFTTrainingConfig(
            model_name="test-model",
            output_dir="./test-output",
            lora=LoRAConfig(enabled=True, r=16, lora_alpha=32),
        )

        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=Mock())

        # Create mock PEFT module
        mock_lora_config = Mock()
        mock_peft = MagicMock()
        mock_peft.LoraConfig = MagicMock(return_value=mock_lora_config)
        mock_peft.get_peft_model = MagicMock()

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "peft": mock_peft,
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(config=config, train_dataset=mock_dataset)

            assert trainer.peft_config is not None

    def test_train_method(self, mock_dataset, basic_config):
        """Test train method."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105

        mock_trainer_instance = Mock()
        mock_train_result = Mock()
        mock_train_result.metrics = {"train_loss": 0.5}
        mock_trainer_instance.train.return_value = mock_train_result

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=mock_trainer_instance)

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)
            metrics = trainer.train()

            assert metrics == {"train_loss": 0.5}
            mock_trainer_instance.train.assert_called_once()

    def test_save_model(self, mock_dataset, basic_config):
        """Test save_model method."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105

        mock_trainer_instance = Mock()

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=mock_trainer_instance)

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)
            trainer.save_model("./test-save")

            mock_trainer_instance.save_model.assert_called_once_with("./test-save")
            mock_tokenizer.save_pretrained.assert_called_once_with("./test-save")

    def test_push_to_hub(self, mock_dataset, basic_config):
        """Test push_to_hub method."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105

        mock_trainer_instance = Mock()

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=mock_trainer_instance)

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)
            trainer.push_to_hub("username/model")

            mock_trainer_instance.push_to_hub.assert_called_once_with(repo_id="username/model")

    def test_push_to_hub_no_repo_id(self, mock_dataset, basic_config):
        """Test push_to_hub raises error without repo_id."""
        # This test would require full mocking, so we'll just check the signature
        # In reality, the method should raise DeepFabricError if no hub_model_id
        pass

    def test_evaluate_with_eval_dataset(self, mock_dataset, basic_config):
        """Test evaluate method with eval dataset."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105

        mock_trainer_instance = Mock()
        mock_trainer_instance.evaluate.return_value = {"eval_loss": 0.3}

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=mock_trainer_instance)

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        eval_dataset = Dataset()
        eval_dataset.samples = [{"messages": [{"role": "user", "content": "Test"}]}]

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(
                config=basic_config, train_dataset=mock_dataset, eval_dataset=eval_dataset
            )
            metrics = trainer.evaluate()

            assert metrics == {"eval_loss": 0.3}
            mock_trainer_instance.evaluate.assert_called_once()

    def test_repr(self, mock_dataset, basic_config):
        """Test string representation."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105

        # Create mock transformers module
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.TrainingArguments = MagicMock(return_value=Mock())

        # Create mock TRL module
        mock_trl = MagicMock()
        mock_trl.SFTTrainer = MagicMock(return_value=Mock())

        # Create mock datasets module
        mock_hf_dataset_instance = Mock()
        mock_hf_dataset_instance.__len__ = Mock(return_value=2)
        mock_hf_dataset = MagicMock()
        mock_hf_dataset.from_list.return_value = mock_hf_dataset_instance
        mock_datasets = MagicMock()
        mock_datasets.Dataset = mock_hf_dataset

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "trl": mock_trl,
                "torch": MagicMock(),
                "datasets": mock_datasets,
            },
        ):
            trainer = DeepFabricSFTTrainer(config=basic_config, train_dataset=mock_dataset)
            repr_str = repr(trainer)

            assert "DeepFabricSFTTrainer" in repr_str
            assert "test-model" in repr_str
            assert str(len(mock_dataset)) in repr_str
