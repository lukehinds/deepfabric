"""Tests for error codes functionality."""

# ruff: noqa: PLR2004

import re

from dataclasses import FrozenInstanceError

import pytest

from deepfabric.error_codes import (
    ALL_ERROR_CODES,
    DF_A01,
    DF_A02,
    DF_A03,
    DF_N01,
    DF_N02,
    DF_N03,
    DF_P01,
    DF_P02,
    DF_P03,
    DF_P04,
    DF_R01,
    DF_R02,
    DF_R03,
    DF_R04,
    DF_T01,
    DF_T02,
    DF_T03,
    DF_X01,
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
    ErrorSeverity,
    classify_error,
    get_classifier,
)


class TestErrorCode:
    """Tests for ErrorCode dataclass."""

    def test_error_code_attributes(self):
        """Test error code has required attributes."""
        assert DF_R01.code == "DF-R01"
        assert DF_R01.short_message == "Rate limit (RPM)"
        assert DF_R01.category == ErrorCategory.RATE_LIMIT
        assert DF_R01.severity == ErrorSeverity.SAMPLE

    def test_format_event_without_detail(self):
        """Test formatting error for TUI display without detail."""
        result = DF_R01.format_event()
        assert result == "DF-R01 Rate limit (RPM)"

    def test_format_event_with_detail(self):
        """Test formatting error for TUI display with detail."""
        result = DF_R01.format_event("retry 3s")
        assert result == "DF-R01 Rate limit (RPM) - retry 3s"

    def test_format_full_without_detail(self):
        """Test full format without detail."""
        result = DF_R01.format_full()
        assert result == "[DF-R01] Rate limit (RPM)"

    def test_format_full_with_detail(self):
        """Test full format with detail."""
        result = DF_R01.format_full("quota exceeded")
        assert result == "[DF-R01] Rate limit (RPM): quota exceeded"

    def test_error_code_is_frozen(self):
        """Test that error codes are immutable."""
        with pytest.raises(FrozenInstanceError):
            DF_R01.code = "DF-R99"


class TestClassifiedError:
    """Tests for ClassifiedError dataclass."""

    def test_to_event_basic(self):
        """Test basic event formatting."""
        classified = ClassifiedError(error_code=DF_R01)
        assert classified.to_event() == "DF-R01 Rate limit (RPM)"

    def test_to_event_with_detail(self):
        """Test event formatting with detail."""
        classified = ClassifiedError(error_code=DF_R01, detail="gemini")
        assert classified.to_event() == "DF-R01 Rate limit (RPM) - gemini"

    def test_to_event_with_retry_after(self):
        """Test event formatting with retry_after takes precedence."""
        classified = ClassifiedError(
            error_code=DF_R01,
            detail="ignored",
            retry_after=5.0,
        )
        assert classified.to_event() == "DF-R01 Rate limit (RPM) - retry 5s"

    def test_classified_error_stores_original(self):
        """Test that original error is stored."""
        classified = ClassifiedError(
            error_code=DF_X01,
            original_error="Something went wrong",
        )
        assert classified.original_error == "Something went wrong"


class TestErrorClassifier:
    """Tests for ErrorClassifier."""

    def test_classifier_with_provider(self):
        """Test classifier stores provider."""
        classifier = ErrorClassifier(provider="gemini")
        assert classifier.provider == "gemini"

    def test_classifier_without_provider(self):
        """Test classifier works without provider."""
        classifier = ErrorClassifier()
        assert classifier.provider is None


class TestRateLimitClassification:
    """Tests for rate limit error classification."""

    def test_classify_rate_limit_429(self):
        """Test classification of 429 error."""
        result = classify_error("Error 429: Too many requests")
        assert result.error_code.category == ErrorCategory.RATE_LIMIT

    def test_classify_rate_limit_rpm(self):
        """Test classification of RPM rate limit."""
        result = classify_error("Rate limit exceeded: requests per_minute quota")
        assert result.error_code == DF_R01

    def test_classify_rate_limit_daily(self):
        """Test classification of daily quota."""
        result = classify_error("RESOURCE_EXHAUSTED: per_day quota exceeded")
        assert result.error_code == DF_R02

    def test_classify_rate_limit_daily_from_context(self):
        """Test classification with daily_quota_exhausted in context."""
        result = classify_error(
            "Rate limit error",
            context={"is_rate_limit": True, "daily_quota_exhausted": True},
        )
        assert result.error_code == DF_R02

    def test_classify_rate_limit_tokens(self):
        """Test classification of token limit."""
        result = classify_error("Rate limit: token quota exceeded")
        assert result.error_code == DF_R03

    def test_classify_rate_limit_generic(self):
        """Test classification of generic rate limit."""
        result = classify_error(
            "Some error",
            context={"is_rate_limit": True},
        )
        assert result.error_code == DF_R04

    def test_retry_after_preserved(self):
        """Test that retry_after is preserved from context."""
        result = classify_error(
            "Rate limit error",
            context={"is_rate_limit": True, "retry_after": 10.5},
        )
        assert result.retry_after == 10.5


class TestAuthErrorClassification:
    """Tests for authentication error classification."""

    def test_classify_auth_error_api_key(self):
        """Test classification of API key error."""
        result = classify_error("Invalid api_key provided")
        assert result.error_code == DF_A01

    def test_classify_auth_error_unauthorized(self):
        """Test classification of unauthorized error."""
        result = classify_error("401 Unauthorized")
        assert result.error_code == DF_A01

    def test_classify_auth_error_permission(self):
        """Test classification of permission denied."""
        result = classify_error("Permission denied for this resource")
        assert result.error_code == DF_A01

    def test_classify_model_not_found(self):
        """Test classification of model not found."""
        result = classify_error("Model 'gpt-5' not found (404)")
        assert result.error_code == DF_A02

    def test_classify_generic_api_error(self):
        """Test classification of generic API error."""
        result = classify_error("500 Internal server error from API")
        assert result.error_code == DF_A03


class TestNetworkErrorClassification:
    """Tests for network error classification."""

    def test_classify_connection_error(self):
        """Test classification of connection error."""
        result = classify_error("Connection refused")
        assert result.error_code == DF_N01

    def test_classify_timeout_error(self):
        """Test classification of timeout error."""
        result = classify_error("Request timed out after 30s")
        assert result.error_code == DF_N02

    def test_classify_service_unavailable_503(self):
        """Test classification of 503 error."""
        result = classify_error("503 Service Unavailable")
        assert result.error_code == DF_N03

    def test_classify_bad_gateway_502(self):
        """Test classification of 502 error."""
        result = classify_error("502 Bad Gateway")
        assert result.error_code == DF_N03


class TestParseErrorClassification:
    """Tests for parse error classification."""

    def test_classify_json_parse_error(self):
        """Test classification of JSON parse error."""
        result = classify_error("Failed to parse JSON response")
        assert result.error_code == DF_P01

    def test_classify_schema_validation_error(self):
        """Test classification of schema validation error."""
        result = classify_error("Schema validation failed")
        assert result.error_code == DF_P02

    def test_classify_empty_response(self):
        """Test classification of empty response."""
        result = classify_error("Empty response from LLM")
        assert result.error_code == DF_P03

    def test_classify_empty_from_context(self):
        """Test classification with error_type context."""
        result = classify_error(
            "Generation failed",
            context={"error_type": "empty_responses"},
        )
        assert result.error_code == DF_P03

    def test_classify_malformed_response(self):
        """Test classification of malformed response."""
        result = classify_error(
            "Invalid format received",
            context={"error_type": "json_parsing_errors"},
        )
        assert result.error_code == DF_P01


class TestToolErrorClassification:
    """Tests for tool error classification."""

    def test_classify_tool_validation_error(self):
        """Test classification of tool validation error."""
        result = classify_error("Tool call format invalid")
        assert result.error_code == DF_T01

    def test_classify_tool_limit_exceeded(self):
        """Test classification of tool limit exceeded."""
        result = classify_error("Sample exceeds limit of max_tools_per_query")
        assert result.error_code == DF_T02

    def test_classify_no_tool_execution(self):
        """Test classification of no tool execution in agent mode."""
        result = classify_error("Agent mode requires at least one tool execution")
        assert result.error_code == DF_T03

    def test_classify_tool_from_context(self):
        """Test tool error classification from context."""
        result = classify_error(
            "Generation failed",
            context={"error_type": "tool_error"},
        )
        assert result.error_code == DF_T01


class TestUnknownErrorClassification:
    """Tests for unknown error classification."""

    def test_classify_unknown_error(self):
        """Test classification of completely unknown error."""
        result = classify_error("Something completely unexpected happened")
        assert result.error_code == DF_X01

    def test_unknown_error_truncates_detail(self):
        """Test that long error messages are truncated in detail."""
        long_error = "x" * 100
        result = classify_error(long_error)
        assert result.error_code == DF_X01
        assert len(result.detail) == 50  # ERROR_DETAIL_MAX_LENGTH


class TestErrorCodeRegistry:
    """Tests for error code registry."""

    def test_all_error_codes_populated(self):
        """Test that ALL_ERROR_CODES contains all codes."""
        assert len(ALL_ERROR_CODES) == 18  # Total number of error codes

    def test_all_codes_have_required_fields(self):
        """Test that all error codes have required fields."""
        for code, error in ALL_ERROR_CODES.items():
            assert error.code == code
            assert error.short_message
            assert error.description
            assert error.category in ErrorCategory
            assert error.severity in ErrorSeverity

    def test_code_format_valid(self):
        """Test that all codes follow DF-XNN format."""
        pattern = r"^DF-[RANPTX]\d{2}$"
        for code in ALL_ERROR_CODES:
            assert re.match(pattern, code), f"Invalid code format: {code}"


class TestGetClassifier:
    """Tests for get_classifier function."""

    def test_get_classifier_with_provider(self):
        """Test getting classifier with specific provider."""
        classifier = get_classifier("openai")
        assert classifier.provider == "openai"

    def test_get_classifier_without_provider_returns_singleton(self):
        """Test that default classifier is reused."""
        c1 = get_classifier()
        c2 = get_classifier()
        assert c1 is c2

    def test_get_classifier_with_provider_creates_new(self):
        """Test that provider-specific classifier is new instance."""
        c1 = get_classifier("openai")
        c2 = get_classifier("gemini")
        assert c1 is not c2


class TestClassifyErrorFunction:
    """Tests for classify_error convenience function."""

    def test_classify_error_with_exception(self):
        """Test classifying an actual exception."""
        exc = ValueError("Rate limit exceeded")
        result = classify_error(exc)
        assert result.error_code.category == ErrorCategory.RATE_LIMIT

    def test_classify_error_with_string(self):
        """Test classifying a string error."""
        result = classify_error("Connection timeout")
        assert result.error_code == DF_N02

    def test_classify_error_with_provider(self):
        """Test classifying with provider context."""
        result = classify_error("Error", provider="gemini")
        assert result.error_code == DF_X01  # Unknown, but provider was set

    def test_classify_error_with_context(self):
        """Test classifying with additional context."""
        result = classify_error(
            "Error",
            context={"is_rate_limit": True, "quota_type": "tokens"},
        )
        assert result.error_code == DF_R03


class TestErrorSeverity:
    """Tests for error severity classification."""

    def test_fatal_errors(self):
        """Test that fatal errors are correctly marked."""
        assert DF_A01.severity == ErrorSeverity.FATAL  # Auth failed
        assert DF_A02.severity == ErrorSeverity.FATAL  # Model not found

    def test_sample_errors(self):
        """Test that sample errors are correctly marked."""
        sample_errors = [
            DF_R01,
            DF_R02,
            DF_R03,
            DF_R04,  # Rate limits
            DF_A03,  # Generic API
            DF_N01,
            DF_N02,
            DF_N03,  # Network
            DF_P01,
            DF_P02,
            DF_P03,
            DF_P04,  # Parse
            DF_T01,
            DF_T02,
            DF_T03,  # Tool
            DF_X01,  # Unknown
        ]
        for error in sample_errors:
            assert error.severity == ErrorSeverity.SAMPLE, f"{error.code} should be SAMPLE"


class TestErrorCategory:
    """Tests for error category classification."""

    def test_rate_limit_category(self):
        """Test rate limit errors are in correct category."""
        rate_limit_codes = [DF_R01, DF_R02, DF_R03, DF_R04]
        for error in rate_limit_codes:
            assert error.category == ErrorCategory.RATE_LIMIT

    def test_auth_category(self):
        """Test auth errors are in correct category."""
        auth_codes = [DF_A01, DF_A02, DF_A03]
        for error in auth_codes:
            assert error.category == ErrorCategory.AUTH_API

    def test_network_category(self):
        """Test network errors are in correct category."""
        network_codes = [DF_N01, DF_N02, DF_N03]
        for error in network_codes:
            assert error.category == ErrorCategory.NETWORK

    def test_parse_category(self):
        """Test parse errors are in correct category."""
        parse_codes = [DF_P01, DF_P02, DF_P03, DF_P04]
        for error in parse_codes:
            assert error.category == ErrorCategory.PARSE

    def test_tool_category(self):
        """Test tool errors are in correct category."""
        tool_codes = [DF_T01, DF_T02, DF_T03]
        for error in tool_codes:
            assert error.category == ErrorCategory.TOOL

    def test_unknown_category(self):
        """Test unknown errors are in correct category."""
        assert DF_X01.category == ErrorCategory.UNKNOWN
