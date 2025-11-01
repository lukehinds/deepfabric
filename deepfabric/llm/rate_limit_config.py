"""Rate limiting configuration models for different LLM providers."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class BackoffStrategy(str, Enum):
    """Backoff strategy for retry attempts."""

    EXPONENTIAL = "exponential"
    EXPONENTIAL_JITTER = "exponential_jitter"
    LINEAR = "linear"
    CONSTANT = "constant"


class RateLimitConfig(BaseModel):
    """Base configuration for rate limiting and retry behavior.

    This provides sensible defaults that work across all providers,
    with provider-specific subclasses adding specialized behavior.
    """

    max_retries: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum number of retry attempts",
    )
    base_delay: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Base delay in seconds before first retry",
    )
    max_delay: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum delay in seconds between retries",
    )
    backoff_strategy: BackoffStrategy = Field(
        default=BackoffStrategy.EXPONENTIAL_JITTER,
        description="Strategy for calculating retry delays",
    )
    exponential_base: float = Field(
        default=2.0,
        ge=1.1,
        le=10.0,
        description="Base multiplier for exponential backoff",
    )
    jitter: bool = Field(
        default=True,
        description="Add randomization to delays to prevent thundering herd",
    )
    respect_retry_after: bool = Field(
        default=True,
        description="Respect retry-after headers from provider responses",
    )

    # HTTP status codes that should trigger retry
    retry_on_status_codes: set[int] = Field(
        default_factory=lambda: {429, 500, 502, 503, 504},
        description="HTTP status codes that trigger retry",
    )

    # Exception types/messages that should trigger retry
    retry_on_exceptions: list[str] = Field(
        default_factory=lambda: ["timeout", "connection", "network"],
        description="Exception keywords that trigger retry (case-insensitive)",
    )

    @field_validator("max_delay")
    @classmethod
    def validate_max_delay(cls, v: float, info: "ValidationInfo") -> float:
        """Ensure max_delay is greater than base_delay."""
        if "base_delay" in info.data and v < info.data["base_delay"]:
            msg = "max_delay must be greater than or equal to base_delay"
            raise ValueError(msg)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return self.model_dump(mode="json")


class OpenAIRateLimitConfig(RateLimitConfig):
    """OpenAI-specific rate limit configuration.

    OpenAI provides detailed rate limit headers (x-ratelimit-*) and
    retry-after headers. This config enables monitoring and preemptive
    backoff based on remaining capacity.
    """

    check_headers: bool = Field(
        default=True,
        description="Monitor x-ratelimit-* headers for capacity tracking",
    )
    preemptive_backoff: bool = Field(
        default=False,
        description="Back off preemptively when remaining capacity is low",
    )
    preemptive_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=0.5,
        description="Threshold (0-1) for preemptive backoff (e.g., 0.1 = 10% remaining)",
    )


class AnthropicRateLimitConfig(RateLimitConfig):
    """Anthropic Claude-specific rate limit configuration.

    Anthropic uses a token bucket algorithm with separate limits for
    requests per minute (RPM), input tokens per minute (ITPM), and
    output tokens per minute (OTPM). Rate limits vary by model and tier.
    """

    check_headers: bool = Field(
        default=True,
        description="Monitor anthropic-ratelimit-* headers",
    )
    token_bucket_aware: bool = Field(
        default=True,
        description="Account for token bucket continuous replenishment",
    )
    gradual_rampup: bool = Field(
        default=True,
        description="Enable gradual traffic ramp-up for new workloads",
    )


class GeminiRateLimitConfig(RateLimitConfig):
    """Google Gemini-specific rate limit configuration.

    Gemini has RPM, TPM, and RPD (requests per day) limits. Daily quotas
    reset at midnight Pacific time. No explicit retry-after header, so
    more conservative backoff is used. Rate limit errors include detailed
    quota violation information.
    """

    base_delay: float = Field(
        default=2.0,
        ge=0.5,
        le=60.0,
        description="Higher default delay for Gemini (no retry-after header)",
    )
    max_delay: float = Field(
        default=120.0,
        ge=5.0,
        le=600.0,
        description="Longer max delay for daily quota exhaustion",
    )
    parse_quota_details: bool = Field(
        default=True,
        description="Extract quota metric details from RESOURCE_EXHAUSTED errors",
    )
    daily_quota_aware: bool = Field(
        default=True,
        description="Recognize daily quota exhaustion vs per-minute limits",
    )


class OllamaRateLimitConfig(RateLimitConfig):
    """Ollama-specific rate limit configuration.

    Ollama is typically run locally, so rate limiting is less common.
    This config uses minimal retries, primarily for connection issues.
    """

    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Minimal retries for local Ollama server",
    )
    base_delay: float = Field(
        default=0.5,
        ge=0.1,
        le=5.0,
        description="Short delay for local server retry",
    )
    max_delay: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Short max delay for local operations",
    )
    retry_on_status_codes: set[int] = Field(
        default_factory=lambda: {500, 502, 503, 504},
        description="Primarily retry server errors (429 unlikely for local)",
    )


class OpenRouterRateLimitConfig(RateLimitConfig):
    """OpenRouter-specific rate limit configuration.

    OpenRouter aggregates multiple LLM providers and uses OpenAI-compatible API.
    It uses credit-based quotas with model-specific rate limits. Different models
    have different RPM limits, and free model variants have daily limits.
    Returns 402 Payment Required when account balance is negative.
    """

    retry_on_status_codes: set[int] = Field(
        default_factory=lambda: {402, 429, 500, 502, 503, 504},
        description="HTTP status codes that trigger retry (includes 402 for payment issues)",
    )
    check_credits: bool = Field(
        default=False,
        description="Monitor credit balance via /api/v1/key endpoint",
    )


def get_default_rate_limit_config(provider: str) -> RateLimitConfig:
    """Get the default rate limit configuration for a provider.

    Args:
        provider: Provider name (openai, anthropic, gemini, ollama, openrouter)

    Returns:
        Provider-specific rate limit configuration with sensible defaults
    """
    configs = {
        "openai": OpenAIRateLimitConfig(),
        "anthropic": AnthropicRateLimitConfig(),
        "gemini": GeminiRateLimitConfig(),
        "ollama": OllamaRateLimitConfig(),
        "openrouter": OpenRouterRateLimitConfig(),
    }
    return configs.get(provider, RateLimitConfig())


def create_rate_limit_config(
    provider: str,
    config_dict: dict[str, Any] | None = None,
) -> RateLimitConfig:
    """Create a rate limit configuration from a dictionary.

    Args:
        provider: Provider name (openai, anthropic, gemini, ollama, openrouter)
        config_dict: Configuration parameters as dictionary

    Returns:
        Provider-specific rate limit configuration

    Raises:
        ValueError: If configuration validation fails
    """
    if config_dict is None:
        return get_default_rate_limit_config(provider)

    config_classes = {
        "openai": OpenAIRateLimitConfig,
        "anthropic": AnthropicRateLimitConfig,
        "gemini": GeminiRateLimitConfig,
        "ollama": OllamaRateLimitConfig,
        "openrouter": OpenRouterRateLimitConfig,
    }

    config_class = config_classes.get(provider, RateLimitConfig)
    return config_class(**config_dict)
