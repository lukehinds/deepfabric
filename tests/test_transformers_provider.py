"""Unit tests for HuggingFace Transformers provider."""

from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest

from deepfabric.exceptions import DataSetGeneratorError
from deepfabric.llm.transformers_provider import (
    TransformersConfig,
    TransformersProvider,
    make_transformers_model,
)


class TestTransformersConfig:
    """Tests for TransformersConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TransformersConfig(model_id="test-model")

        assert config.model_id == "test-model"
        assert config.device is None
        assert config.torch_dtype == "auto"
        assert config.load_in_8bit is False
        assert config.load_in_4bit is False
        assert config.trust_remote_code is False
        assert config.use_fast_tokenizer is True
        assert config.max_length == 8192  # noqa: PLR2004

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TransformersConfig(
            model_id="test-model",
            device="cuda",
            torch_dtype="bfloat16",
            load_in_4bit=True,
            trust_remote_code=True,
            max_length=4096,
        )

        assert config.device == "cuda"
        assert config.torch_dtype == "bfloat16"
        assert config.load_in_4bit is True
        assert config.trust_remote_code is True

    def test_invalid_torch_dtype(self):
        """Test that invalid torch_dtype raises validation error."""
        with pytest.raises(Exception):  # Pydantic validation error  # noqa: B017
            TransformersConfig(model_id="test-model", torch_dtype=cast(Any, "invalid"))
        with pytest.raises(Exception):  # Pydantic validation error  # noqa: B017
            TransformersConfig(model_id="test-model", torch_dtype=cast(Any, "invalid"))

    def test_model_kwargs(self):
        """Test model_kwargs are stored correctly."""
        config = TransformersConfig(
            model_id="test-model",
            model_kwargs={"attn_implementation": "flash_attention_2"},
        )

        assert config.model_kwargs["attn_implementation"] == "flash_attention_2"


class TestTransformersProvider:
    """Tests for TransformersProvider."""

    def test_provider_initialization(self):
        """Test provider initialization."""
        # Create mocks
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"  # noqa: S105
        mock_tokenizer.chat_template = None

        # Mock transformers classes
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock outlines - need to mock the Transformers class
        mock_outlines_transformers_cls = MagicMock(return_value=Mock())
        mock_outlines_models = MagicMock()
        mock_outlines_models.Transformers = mock_outlines_transformers_cls

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "torch": MagicMock(),
                "outlines": MagicMock(),
                "outlines.models": mock_outlines_models,
            },
        ):
            provider = TransformersProvider("test-model")

            assert provider.model_id == "test-model"
            assert provider.model is not None
            assert provider.tokenizer is not None
            mock_transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
            mock_transformers.AutoTokenizer.from_pretrained.assert_called_once()

    def test_provider_missing_transformers(self):
        """Test error when transformers is not installed."""
        with (
            patch.dict("sys.modules", {"transformers": None}),
            pytest.raises(DataSetGeneratorError, match="transformers library not found"),
        ):
            TransformersProvider("test-model")

    def test_provider_missing_torch(self):
        """Test error when torch is not installed."""
        with (
            patch.dict("sys.modules", {"torch": None, "transformers": MagicMock()}),
            pytest.raises(DataSetGeneratorError, match="torch library not found"),
        ):
            TransformersProvider("test-model")

    def test_provider_with_quantization(self):
        """Test provider with quantization settings."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105
        mock_tokenizer.chat_template = "test"

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock outlines
        mock_outlines_transformers_cls = MagicMock(return_value=Mock())
        mock_outlines_models = MagicMock()
        mock_outlines_models.Transformers = mock_outlines_transformers_cls

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "torch": MagicMock(),
                "outlines": MagicMock(),
                "outlines.models": mock_outlines_models,
            },
        ):
            config = TransformersConfig(model_id="test-model", load_in_4bit=True)
            _provider = TransformersProvider("test-model", config=config)

            # Check that load_in_4bit was passed to model loading
            call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args[1]
            assert call_kwargs.get("load_in_4bit") is True

    def test_get_outlines_model(self):
        """Test getting Outlines model."""
        mock_outlines_model = Mock()
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105
        mock_tokenizer.chat_template = "test"

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock outlines
        mock_outlines_transformers_cls = MagicMock(return_value=mock_outlines_model)
        mock_outlines_models = MagicMock()
        mock_outlines_models.Transformers = mock_outlines_transformers_cls

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "torch": MagicMock(),
                "outlines": MagicMock(),
                "outlines.models": mock_outlines_models,
            },
        ):
            provider = TransformersProvider("test-model")
            outlines_model = provider.get_outlines_model()

            assert outlines_model is mock_outlines_model

    def test_provider_unload(self):
        """Test provider resource cleanup."""
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "<pad>"  # noqa: S105
        mock_tokenizer.chat_template = "test"

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer

        # Mock outlines
        mock_outlines_transformers_cls = MagicMock(return_value=Mock())
        mock_outlines_models = MagicMock()
        mock_outlines_models.Transformers = mock_outlines_transformers_cls

        with patch.dict(
            "sys.modules",
            {
                "transformers": mock_transformers,
                "torch": MagicMock(),
                "gc": MagicMock(),
                "outlines": MagicMock(),
                "outlines.models": mock_outlines_models,
            },
        ):
            provider = TransformersProvider("test-model")
            provider.unload()

            assert provider.model is None
            assert provider.tokenizer is None
            assert provider.outlines_model is None


class TestMakeTransformersModel:
    """Tests for make_transformers_model factory function."""

    @patch("deepfabric.llm.transformers_provider.TransformersProvider")
    def test_make_transformers_model(self, mock_provider_cls):
        """Test factory function creates provider correctly."""
        mock_provider = Mock()
        mock_outlines_model = Mock()
        mock_provider.get_outlines_model.return_value = mock_outlines_model
        mock_provider_cls.return_value = mock_provider

        result = make_transformers_model("test-model", device="cuda")

        mock_provider_cls.assert_called_once_with("test-model", device="cuda")
        assert result is mock_outlines_model

    @patch("deepfabric.llm.transformers_provider.TransformersProvider")
    def test_make_transformers_model_with_kwargs(self, mock_provider_cls):
        """Test factory function passes kwargs correctly."""
        mock_provider = Mock()
        mock_outlines_model = Mock()
        mock_provider.get_outlines_model.return_value = mock_outlines_model
        mock_provider_cls.return_value = mock_provider

        make_transformers_model(
            "test-model", device="cuda", torch_dtype="bfloat16", load_in_4bit=True
        )

        mock_provider_cls.assert_called_once_with(
            "test-model", device="cuda", torch_dtype="bfloat16", load_in_4bit=True
        )
