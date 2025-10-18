"""Unit tests for Transformers rate limit configuration."""

import pytest

from deepfabric.llm.rate_limit_config import (
    BackoffStrategy,
    TransformersRateLimitConfig,
    create_rate_limit_config,
    get_default_rate_limit_config,
)


class TestTransformersRateLimitConfig:
    """Tests for TransformersRateLimitConfig."""

    def test_default_config(self):
        """Test default configuration for transformers provider."""
        config = TransformersRateLimitConfig()

        assert config.max_retries == 2  # noqa: PLR2004
        assert config.base_delay == 1.0  # noqa: PLR2004
        assert config.max_delay == 10.0  # noqa: PLR2004
        assert config.backoff_strategy == BackoffStrategy.LINEAR  # noqa: PLR2004
        assert config.jitter is False  # noqa: PLR2004
        assert config.respect_retry_after is False  # noqa: PLR2004
        assert len(config.retry_on_status_codes) == 0  # noqa: PLR2004
        assert "cuda" in config.retry_on_exceptions
        assert "out of memory" in config.retry_on_exceptions
        assert "generation" in config.retry_on_exceptions

    def test_custom_config(self):
        """Test custom configuration values."""
        config = TransformersRateLimitConfig(
            max_retries=3,
            base_delay=2.0,
            max_delay=20.0,
        )

        assert config.max_retries == 3  # noqa: PLR2004
        assert config.base_delay == 2.0  # noqa: PLR2004
        assert config.max_delay == 20.0  # noqa: PLR2004

    def test_hardware_specific_exceptions(self):
        """Test that hardware-specific exceptions are configured."""
        config = TransformersRateLimitConfig()

        # Should include CUDA and memory-related errors
        exceptions_lower = [e.lower() for e in config.retry_on_exceptions]
        assert "cuda" in exceptions_lower
        assert "out of memory" in exceptions_lower
        assert "generation" in exceptions_lower

    def test_no_http_status_codes(self):
        """Test that HTTP status codes are not used for local inference."""
        config = TransformersRateLimitConfig()

        assert len(config.retry_on_status_codes) == 0

    def test_linear_backoff_strategy(self):
        """Test that linear backoff is default for transformers."""
        config = TransformersRateLimitConfig()

        assert config.backoff_strategy == BackoffStrategy.LINEAR

    def test_no_jitter(self):
        """Test that jitter is disabled for transformers."""
        config = TransformersRateLimitConfig()

        assert config.jitter is False

    def test_validation_max_retries(self):
        """Test validation for max_retries."""
        with pytest.raises(Exception):  # Pydantic validation error  # noqa: B017
            TransformersRateLimitConfig(max_retries=-1)

        with pytest.raises(Exception):  # noqa: B017
            TransformersRateLimitConfig(max_retries=10)

    def test_validation_delays(self):
        """Test validation for delay parameters."""
        with pytest.raises(Exception):  # noqa: B017
            TransformersRateLimitConfig(base_delay=0.0)

        with pytest.raises(Exception):  # noqa: B017
            TransformersRateLimitConfig(base_delay=15.0)

        with pytest.raises(Exception):  # noqa: B017
            TransformersRateLimitConfig(max_delay=70.0)


class TestGetDefaultRateLimitConfig:
    """Tests for get_default_rate_limit_config function."""

    def test_get_transformers_config(self):
        """Test getting default config for transformers provider."""
        config = get_default_rate_limit_config("transformers")

        assert isinstance(config, TransformersRateLimitConfig)
        assert config.max_retries == 2  # noqa: PLR2004

    def test_get_other_provider_configs(self):
        """Test that other providers still work."""
        openai_config = get_default_rate_limit_config("openai")
        anthropic_config = get_default_rate_limit_config("anthropic")
        gemini_config = get_default_rate_limit_config("gemini")
        ollama_config = get_default_rate_limit_config("ollama")

        assert openai_config.__class__.__name__ == "OpenAIRateLimitConfig"
        assert anthropic_config.__class__.__name__ == "AnthropicRateLimitConfig"
        assert gemini_config.__class__.__name__ == "GeminiRateLimitConfig"
        assert ollama_config.__class__.__name__ == "OllamaRateLimitConfig"

    def test_unknown_provider(self):
        """Test fallback for unknown provider."""
        config = get_default_rate_limit_config("unknown")

        assert config.__class__.__name__ == "RateLimitConfig"


class TestCreateRateLimitConfig:
    """Tests for create_rate_limit_config function."""

    def test_create_transformers_config(self):
        """Test creating transformers config with custom parameters."""
        config = create_rate_limit_config("transformers", {"max_retries": 3, "base_delay": 2.0})

        assert isinstance(config, TransformersRateLimitConfig)
        assert config.max_retries == 3  # noqa: PLR2004
        assert config.base_delay == 2.0  # noqa: PLR2004

    def test_create_with_none_dict(self):
        """Test creating config with None dict returns defaults."""
        config = create_rate_limit_config("transformers", None)

        assert isinstance(config, TransformersRateLimitConfig)
        assert config.max_retries == 2  # noqa: PLR2004

    def test_create_other_providers(self):
        """Test creating configs for other providers."""
        openai_config = create_rate_limit_config("openai", {"max_retries": 10})

        assert openai_config.__class__.__name__ == "OpenAIRateLimitConfig"
        assert openai_config.max_retries == 10  # noqa: PLR2004


class TestTransformersVsOtherProviders:
    """Tests comparing transformers config with other providers."""

    def test_fewer_retries_than_apis(self):
        """Test that transformers has fewer retries than API providers."""
        transformers_config = get_default_rate_limit_config("transformers")
        openai_config = get_default_rate_limit_config("openai")

        assert transformers_config.max_retries < openai_config.max_retries

    def test_different_backoff_strategy(self):
        """Test that transformers uses different backoff strategy."""
        transformers_config = get_default_rate_limit_config("transformers")
        openai_config = get_default_rate_limit_config("openai")

        # Transformers uses LINEAR, APIs use EXPONENTIAL_JITTER
        assert transformers_config.backoff_strategy != openai_config.backoff_strategy
        assert transformers_config.backoff_strategy == BackoffStrategy.LINEAR

    def test_no_jitter_vs_apis(self):
        """Test that transformers doesn't use jitter unlike APIs."""
        transformers_config = get_default_rate_limit_config("transformers")
        openai_config = get_default_rate_limit_config("openai")

        assert transformers_config.jitter is False
        assert openai_config.jitter is True
