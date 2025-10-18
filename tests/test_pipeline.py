"""Unit tests for DeepFabricPipeline."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from deepfabric.exceptions import DeepFabricError
from deepfabric.pipeline import DeepFabricPipeline


class TestDeepFabricPipeline:
    """Tests for DeepFabricPipeline."""

    def test_initialization(self):
        """Test pipeline initialization."""
        pipeline = DeepFabricPipeline(
            model_name="test-model", provider="openai", api_key="test-key"
        )

        assert pipeline.model_name == "test-model"
        assert pipeline.provider == "openai"
        assert pipeline.provider_kwargs == {"api_key": "test-key"}
        assert pipeline.tree is None
        assert pipeline.dataset is None
        assert pipeline.trainer is None

    def test_initialization_with_transformers(self):
        """Test pipeline initialization with transformers provider."""
        pipeline = DeepFabricPipeline(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            provider="transformers",
            device="cuda",
            dtype="bfloat16",
        )

        assert pipeline.model_name == "meta-llama/Llama-3.1-8B-Instruct"
        assert pipeline.provider == "transformers"
        assert pipeline.provider_kwargs["device"] == "cuda"
        assert pipeline.provider_kwargs["dtype"] == "bfloat16"

    @pytest.mark.asyncio
    @patch("deepfabric.pipeline.Tree")
    @patch("deepfabric.pipeline.DataSetGenerator")
    async def test_generate_dataset_async(self, mock_generator_cls, mock_tree_cls):
        """Test async dataset generation."""
        # Mock Tree
        mock_tree = Mock()
        mock_tree.get_all_paths.return_value = [["topic1", "subtopic1"], ["topic2", "subtopic2"]]

        async def mock_build():
            yield {"event": "build_start"}
            yield {"event": "build_complete", "total_paths": 2}

        mock_tree.build_async = mock_build
        mock_tree_cls.return_value = mock_tree

        # Mock DataSetGenerator
        mock_generator = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)
        mock_generator.create_data_async = AsyncMock(return_value=mock_dataset)
        mock_generator_cls.return_value = mock_generator

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")

        dataset = await pipeline.generate_dataset_async(
            topic_prompt="Test Topic",
            num_samples=10,
            batch_size=5,
            tree_depth=2,
            tree_degree=5,
        )

        assert dataset is mock_dataset
        assert pipeline.tree is mock_tree
        assert pipeline.dataset is mock_dataset

        # Verify Tree was created with correct parameters
        mock_tree_cls.assert_called_once()
        call_kwargs = mock_tree_cls.call_args[1]
        assert call_kwargs["topic_prompt"] == "Test Topic"
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["depth"] == 2  # noqa: PLR2004
        assert call_kwargs["degree"] == 5  # noqa: PLR2004

        # Verify DataSetGenerator was created
        mock_generator_cls.assert_called_once()
        gen_kwargs = mock_generator_cls.call_args[1]
        assert gen_kwargs["provider"] == "openai"

    @patch("deepfabric.pipeline.Tree")
    @patch("deepfabric.pipeline.DataSetGenerator")
    def test_generate_dataset_sync(self, mock_generator_cls, mock_tree_cls):
        """Test sync dataset generation wrapper."""
        # Mock Tree
        mock_tree = Mock()
        mock_tree.get_all_paths.return_value = [["topic1"]]

        async def mock_build():
            yield {"event": "build_complete", "total_paths": 1}

        mock_tree.build_async = mock_build
        mock_tree_cls.return_value = mock_tree

        # Mock DataSetGenerator
        mock_generator = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_generator.create_data_async = AsyncMock(return_value=mock_dataset)
        mock_generator_cls.return_value = mock_generator

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")

        dataset = pipeline.generate_dataset(
            topic_prompt="Test Topic",
            num_samples=5,
        )

        assert dataset is mock_dataset

    @patch("deepfabric.training.sft_trainer.DeepFabricSFTTrainer")
    def test_train_without_dataset(self, mock_trainer_cls):  # noqa: ARG002
        """Test that training without dataset raises error."""
        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")

        with pytest.raises(DeepFabricError, match="No dataset available"):
            pipeline.train(Mock())

    def test_train_with_dataset(self):
        """Test training with dataset."""
        # Create mock dataset
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=10)

        # Create mock trainer
        mock_trainer = Mock()
        mock_trainer.train.return_value = {"train_loss": 0.5}

        # Create mock DeepFabricSFTTrainer class
        mock_trainer_cls = MagicMock(return_value=mock_trainer)

        # Create mock training module
        mock_training = MagicMock()
        mock_training.DeepFabricSFTTrainer = mock_trainer_cls

        # Create mock config
        mock_config = Mock()

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.dataset = mock_dataset

        # Mock the training module import
        with patch.dict("sys.modules", {"deepfabric.training": mock_training}):
            metrics = pipeline.train(mock_config, formatting_func="default")

        assert metrics == {"train_loss": 0.5}
        mock_trainer_cls.assert_called_once()
        mock_trainer.train.assert_called_once()

    def test_train_missing_dependencies(self):
        """Test training with missing dependencies."""
        mock_dataset = Mock()
        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.dataset = mock_dataset

        # Mock the training module to not exist
        with (
            patch.dict("sys.modules", {"deepfabric.training": None}),
            pytest.raises(DeepFabricError, match="Training dependencies not installed"),
        ):
            pipeline.train(Mock())

    def test_save_and_upload_dataset_only(self):
        """Test saving dataset without training."""
        mock_dataset = Mock()

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.dataset = mock_dataset

        pipeline.save_and_upload(dataset_path="./test.jsonl")

        mock_dataset.save.assert_called_once_with("./test.jsonl")

    def test_save_and_upload_model_only(self):
        """Test saving trained model."""
        mock_trainer = Mock()

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.trainer = mock_trainer

        pipeline.save_and_upload(output_dir="./output")

        mock_trainer.save_model.assert_called_once_with("./output")

    def test_save_and_upload_with_hub(self):
        """Test uploading to HuggingFace Hub."""
        mock_trainer = Mock()

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.trainer = mock_trainer

        pipeline.save_and_upload(hf_repo="username/model")

        mock_trainer.push_to_hub.assert_called_once_with("username/model")

    def test_save_and_upload_all(self):
        """Test saving dataset and model together."""
        mock_dataset = Mock()
        mock_trainer = Mock()

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.dataset = mock_dataset
        pipeline.trainer = mock_trainer

        pipeline.save_and_upload(
            output_dir="./output",
            dataset_path="./dataset.jsonl",
            hf_repo="username/model",
        )

        mock_dataset.save.assert_called_once_with("./dataset.jsonl")
        mock_trainer.save_model.assert_called_once_with("./output")
        mock_trainer.push_to_hub.assert_called_once_with("username/model")

    def test_repr(self):
        """Test string representation."""
        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")

        repr_str = repr(pipeline)

        assert "DeepFabricPipeline" in repr_str
        assert "test-model" in repr_str
        assert "openai" in repr_str
        assert "dataset_samples=0" in repr_str
        assert "trained=no" in repr_str

    def test_repr_with_data(self):
        """Test string representation with dataset."""
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=100)

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")
        pipeline.dataset = mock_dataset
        pipeline.trainer = Mock()

        repr_str = repr(pipeline)

        assert "dataset_samples=100" in repr_str
        assert "trained=yes" in repr_str

    @pytest.mark.asyncio
    @patch("deepfabric.pipeline.Tree")
    @patch("deepfabric.pipeline.DataSetGenerator")
    async def test_generate_dataset_with_different_models(self, mock_generator_cls, mock_tree_cls):
        """Test using different models for tree and generation."""
        # Mock Tree
        mock_tree = Mock()
        mock_tree.get_all_paths.return_value = [["topic"]]

        async def mock_build():
            yield {"event": "build_complete", "total_paths": 1}

        mock_tree.build_async = mock_build
        mock_tree_cls.return_value = mock_tree

        # Mock Generator
        mock_generator = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_generator.create_data_async = AsyncMock(return_value=mock_dataset)
        mock_generator_cls.return_value = mock_generator

        pipeline = DeepFabricPipeline(model_name="default-model", provider="openai")

        await pipeline.generate_dataset_async(
            topic_prompt="Test",
            tree_model_name="tree-model",
            generation_model_name="gen-model",
            num_samples=5,
        )

        # Verify different models were used
        tree_call = mock_tree_cls.call_args[1]
        gen_call = mock_generator_cls.call_args[1]

        assert tree_call["model_name"] == "tree-model"
        assert gen_call["model_name"] == "gen-model"

    @pytest.mark.asyncio
    @patch("deepfabric.pipeline.Tree")
    @patch("deepfabric.pipeline.DataSetGenerator")
    async def test_generate_dataset_custom_system_prompt(self, mock_generator_cls, mock_tree_cls):
        """Test custom generation system prompt."""
        mock_tree = Mock()
        mock_tree.get_all_paths.return_value = [["topic"]]

        async def mock_build():
            yield {"event": "build_complete", "total_paths": 1}

        mock_tree.build_async = mock_build
        mock_tree_cls.return_value = mock_tree

        mock_generator = Mock()
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=5)
        mock_generator.create_data_async = AsyncMock(return_value=mock_dataset)
        mock_generator_cls.return_value = mock_generator

        pipeline = DeepFabricPipeline(model_name="test-model", provider="openai")

        await pipeline.generate_dataset_async(
            topic_prompt="Test",
            generation_system_prompt="Custom prompt",
            num_samples=5,
        )

        gen_call = mock_generator_cls.call_args[1]
        assert gen_call["generation_system_prompt"] == "Custom prompt"
