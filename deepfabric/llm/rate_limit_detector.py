"""Rate limit detection and error parsing for different LLM providers."""

import logging

from dataclasses import dataclass, field
from typing import Any

import anthropic
import openai

logger = logging.getLogger(__name__)


@dataclass
class QuotaInfo:
    """Information extracted from a rate limit error."""

    is_rate_limit: bool = False
    quota_type: str | None = None
    limit_value: int | None = None
    retry_after: float | None = None
    daily_quota_exhausted: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"QuotaInfo(is_rate_limit={self.is_rate_limit}"]
        if self.quota_type:
            parts.append(f"quota_type={self.quota_type}")
        if self.retry_after:
            parts.append(f"retry_after={self.retry_after}s")
        if self.daily_quota_exhausted:
            parts.append("daily_quota_exhausted=True")
        return ", ".join(parts) + ")"


class RateLimitDetector:
    """Detect and parse rate limit errors from different LLM providers."""

    @staticmethod
    def is_rate_limit_error(exception: Exception, provider: str) -> bool:
        """Check if an exception represents a rate limit error.

        Args:
            exception: The exception to check
            provider: Provider name (openai, anthropic, gemini, ollama)

        Returns:
            True if the exception is a rate limit error
        """
        if provider == "openai":
            return isinstance(exception, openai.RateLimitError)

        if provider == "anthropic":
            return isinstance(exception, anthropic.RateLimitError)

        if provider == "gemini":
            error_str = str(exception)
            return "429" in error_str and "RESOURCE_EXHAUSTED" in error_str

        if provider == "ollama":
            # Ollama uses OpenAI-compatible API, but rate limits are unlikely
            return isinstance(exception, openai.RateLimitError)

        # Generic fallback: check for 429 in error message
        error_str = str(exception).lower()
        return "429" in error_str or "rate limit" in error_str

    @staticmethod
    def is_retryable_error(exception: Exception, provider: str) -> bool:
        """Check if an exception should trigger a retry.

        This includes rate limits, timeouts, and server errors.

        Args:
            exception: The exception to check
            provider: Provider name

        Returns:
            True if the error is retryable
        """
        # Check if it's a rate limit error first
        if RateLimitDetector.is_rate_limit_error(exception, provider):
            return True

        # Check for server errors and transient failures
        error_str = str(exception).lower()
        retryable_keywords = [
            "timeout",
            "connection",
            "network",
            "503",
            "502",
            "500",
            "504",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        ]

        return any(keyword in error_str for keyword in retryable_keywords)

    @staticmethod
    def extract_quota_info(exception: Exception, provider: str) -> QuotaInfo:
        """Extract detailed quota information from a rate limit error.

        Args:
            exception: The rate limit exception
            provider: Provider name

        Returns:
            QuotaInfo object with parsed details
        """
        if provider == "openai":
            return RateLimitDetector._parse_openai_error(exception)
        if provider == "anthropic":
            return RateLimitDetector._parse_anthropic_error(exception)
        if provider == "gemini":
            return RateLimitDetector._parse_gemini_error(exception)
        if provider == "ollama":
            return RateLimitDetector._parse_openai_error(exception)

        # Generic fallback
        return QuotaInfo(is_rate_limit=RateLimitDetector.is_rate_limit_error(exception, provider))

    @staticmethod
    def _parse_openai_error(exception: Exception) -> QuotaInfo:
        """Parse OpenAI rate limit error for quota details.

        OpenAI provides detailed headers:
        - x-ratelimit-limit-requests
        - x-ratelimit-remaining-requests
        - retry-after
        """
        quota_info = QuotaInfo()

        if not isinstance(exception, openai.RateLimitError):
            return quota_info

        quota_info.is_rate_limit = True

        # Try to extract retry-after from the exception
        try:
            if hasattr(exception, "response") and exception.response:
                headers = exception.response.headers
                if "retry-after" in headers:
                    quota_info.retry_after = float(headers["retry-after"])

                # Check for remaining capacity in headers
                if "x-ratelimit-remaining-requests" in headers:
                    remaining = int(headers["x-ratelimit-remaining-requests"])
                    quota_info.details["remaining_requests"] = remaining

                if "x-ratelimit-limit-requests" in headers:
                    limit = int(headers["x-ratelimit-limit-requests"])
                    quota_info.limit_value = limit
                    quota_info.quota_type = "requests"

        except (AttributeError, ValueError, KeyError) as e:
            logger.debug("Could not parse OpenAI rate limit headers: %s", e)

        # Parse error message for quota vs rate limit distinction
        error_msg = str(exception).lower()
        if "quota" in error_msg:
            quota_info.daily_quota_exhausted = True
            quota_info.quota_type = "quota"

        return quota_info

    @staticmethod
    def _parse_anthropic_error(exception: Exception) -> QuotaInfo:
        """Parse Anthropic rate limit error for quota details.

        Anthropic provides:
        - retry-after header
        - anthropic-ratelimit-requests-remaining
        - anthropic-ratelimit-tokens-remaining
        """
        quota_info = QuotaInfo()

        if not isinstance(exception, anthropic.RateLimitError):
            return quota_info

        quota_info.is_rate_limit = True

        try:
            if hasattr(exception, "response") and exception.response:
                headers = exception.response.headers
                if "retry-after" in headers:
                    quota_info.retry_after = float(headers["retry-after"])

                # Extract remaining capacity
                if "anthropic-ratelimit-requests-remaining" in headers:
                    remaining = int(headers["anthropic-ratelimit-requests-remaining"])
                    quota_info.details["remaining_requests"] = remaining

                if "anthropic-ratelimit-tokens-remaining" in headers:
                    remaining_tokens = int(headers["anthropic-ratelimit-tokens-remaining"])
                    quota_info.details["remaining_tokens"] = remaining_tokens

        except (AttributeError, ValueError, KeyError) as e:
            logger.debug("Could not parse Anthropic rate limit headers: %s", e)

        # Determine quota type from error message
        error_msg = str(exception).lower()
        if "request" in error_msg:
            quota_info.quota_type = "requests_per_minute"
        elif "token" in error_msg:
            quota_info.quota_type = "tokens_per_minute"

        return quota_info

    @staticmethod
    def _parse_gemini_error(exception: Exception) -> QuotaInfo:
        """Parse Gemini RESOURCE_EXHAUSTED error for quota details.

        Gemini errors include detailed quota violation information:
        - quotaMetric (e.g., generate_requests_per_model_per_day)
        - quotaId
        - No explicit retry-after header
        """
        quota_info = QuotaInfo()

        error_str = str(exception)
        if "429" not in error_str or "RESOURCE_EXHAUSTED" not in error_str:
            return quota_info

        quota_info.is_rate_limit = True

        # Try to parse the error response JSON if available
        try:
            # Look for quota metric in error string
            if "quotaMetric" in error_str:
                # Extract quota metric type
                if "per_day" in error_str:
                    quota_info.quota_type = "requests_per_day"
                    quota_info.daily_quota_exhausted = True
                elif "per_minute" in error_str:
                    quota_info.quota_type = "requests_per_minute"

            # Check for limit value in error message
            if "limit: 0" in error_str:
                quota_info.limit_value = 0
                quota_info.details["limit_exceeded"] = True

            # Extract specific quota details if present
            if "generate_requests_per_model_per_day" in error_str:
                quota_info.details["metric"] = "generate_requests_per_model_per_day"
            elif "generate_requests_per_model_per_minute" in error_str:
                quota_info.details["metric"] = "generate_requests_per_model_per_minute"

        except Exception as e:  # noqa: BLE001
            logger.debug("Could not parse Gemini quota details: %s", e)

        # Gemini doesn't provide retry-after header, use None
        quota_info.retry_after = None

        return quota_info

    @staticmethod
    def should_fail_fast(quota_info: QuotaInfo) -> bool:
        """Determine if we should fail fast instead of retrying.

        Some quota errors are not worth retrying:
        - Daily quota exhausted (won't reset for hours)
        - Quota limit is 0 (account not set up properly)

        Args:
            quota_info: Parsed quota information

        Returns:
            True if we should fail fast and not retry
        """
        # Daily quota exhausted - won't reset until midnight
        if quota_info.daily_quota_exhausted:
            return True

        # Quota limit is 0 - account issue, not transient
        return quota_info.limit_value == 0
