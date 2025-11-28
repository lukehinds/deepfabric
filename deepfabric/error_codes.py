"""DeepFabric error codes for standardized error reporting.

Error codes provide consistent, documentable error identification across the CLI and TUI.
Each code maps to a short message suitable for display and a longer description for docs.

Error Code Format: DF-XNN
- DF: DeepFabric prefix
- X: Category letter (R=Rate limit, A=Auth/API, N=Network, P=Parse, T=Tool, X=Unknown)
- NN: Number within category

Sample-level errors occur during generation and allow processing to continue.
Fatal errors cause the CLI to exit immediately.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

# Constants for error classification
ERROR_DETAIL_MAX_LENGTH = 50


class ErrorCategory(str, Enum):
    """Error category for grouping related errors."""

    RATE_LIMIT = "rate_limit"
    AUTH_API = "auth_api"
    NETWORK = "network"
    PARSE = "parse"
    TOOL = "tool"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    """Whether an error is recoverable or fatal."""

    SAMPLE = "sample"  # Per-sample error, generation continues
    FATAL = "fatal"  # CLI exits


@dataclass(frozen=True)
class ErrorCode:
    """Definition of a DeepFabric error code."""

    code: str
    short_message: str
    description: str
    category: ErrorCategory
    severity: ErrorSeverity

    def format_event(self, detail: str | None = None) -> str:
        """Format error for TUI Events panel display.

        Args:
            detail: Optional short detail (e.g., retry time, quota type)

        Returns:
            Formatted string like "DF-R01 Rate limit (RPM)" or "DF-R01 Rate limit - retry 3s"
        """
        if detail:
            return f"{self.code} {self.short_message} - {detail}"
        return f"{self.code} {self.short_message}"

    def format_full(self, detail: str | None = None) -> str:
        """Format error for detailed output (debug mode, logs).

        Args:
            detail: Optional detail about the specific error

        Returns:
            Formatted string with code, message, and description
        """
        base = f"[{self.code}] {self.short_message}"
        if detail:
            base += f": {detail}"
        return base


# =============================================================================
# Error Code Definitions
# =============================================================================

# Rate Limit Errors (DF-R0x)
DF_R01 = ErrorCode(
    code="DF-R01",
    short_message="Rate limit (RPM)",
    description="Requests per minute limit exceeded. The provider is throttling requests.",
    category=ErrorCategory.RATE_LIMIT,
    severity=ErrorSeverity.SAMPLE,
)

DF_R02 = ErrorCode(
    code="DF-R02",
    short_message="Rate limit (daily)",
    description="Daily quota exhausted. Resets at midnight (provider timezone).",
    category=ErrorCategory.RATE_LIMIT,
    severity=ErrorSeverity.SAMPLE,
)

DF_R03 = ErrorCode(
    code="DF-R03",
    short_message="Rate limit (tokens)",
    description="Token per minute limit exceeded.",
    category=ErrorCategory.RATE_LIMIT,
    severity=ErrorSeverity.SAMPLE,
)

DF_R04 = ErrorCode(
    code="DF-R04",
    short_message="Rate limit",
    description="Generic rate limit error from provider.",
    category=ErrorCategory.RATE_LIMIT,
    severity=ErrorSeverity.SAMPLE,
)

# Auth/API Errors (DF-A0x)
DF_A01 = ErrorCode(
    code="DF-A01",
    short_message="Auth failed",
    description="Authentication failed. Check your API key environment variable.",
    category=ErrorCategory.AUTH_API,
    severity=ErrorSeverity.FATAL,
)

DF_A02 = ErrorCode(
    code="DF-A02",
    short_message="Model not found",
    description="The specified model does not exist or is not accessible.",
    category=ErrorCategory.AUTH_API,
    severity=ErrorSeverity.FATAL,
)

DF_A03 = ErrorCode(
    code="DF-A03",
    short_message="API error",
    description="Generic API error from the provider.",
    category=ErrorCategory.AUTH_API,
    severity=ErrorSeverity.SAMPLE,
)

# Network Errors (DF-N0x)
DF_N01 = ErrorCode(
    code="DF-N01",
    short_message="Network error",
    description="Connection failed. Check your internet connection.",
    category=ErrorCategory.NETWORK,
    severity=ErrorSeverity.SAMPLE,
)

DF_N02 = ErrorCode(
    code="DF-N02",
    short_message="Timeout",
    description="Request timed out waiting for provider response.",
    category=ErrorCategory.NETWORK,
    severity=ErrorSeverity.SAMPLE,
)

DF_N03 = ErrorCode(
    code="DF-N03",
    short_message="Service unavailable",
    description="Provider service temporarily unavailable (503/502).",
    category=ErrorCategory.NETWORK,
    severity=ErrorSeverity.SAMPLE,
)

# Parse Errors (DF-P0x)
DF_P01 = ErrorCode(
    code="DF-P01",
    short_message="JSON parse error",
    description="Failed to parse JSON from LLM response.",
    category=ErrorCategory.PARSE,
    severity=ErrorSeverity.SAMPLE,
)

DF_P02 = ErrorCode(
    code="DF-P02",
    short_message="Schema validation",
    description="Response does not match expected schema structure.",
    category=ErrorCategory.PARSE,
    severity=ErrorSeverity.SAMPLE,
)

DF_P03 = ErrorCode(
    code="DF-P03",
    short_message="Empty response",
    description="LLM returned an empty or whitespace-only response.",
    category=ErrorCategory.PARSE,
    severity=ErrorSeverity.SAMPLE,
)

DF_P04 = ErrorCode(
    code="DF-P04",
    short_message="Malformed response",
    description="Response structure is malformed or incomplete.",
    category=ErrorCategory.PARSE,
    severity=ErrorSeverity.SAMPLE,
)

# Tool Errors (DF-T0x)
DF_T01 = ErrorCode(
    code="DF-T01",
    short_message="Tool validation",
    description="Tool call format is invalid or missing required fields.",
    category=ErrorCategory.TOOL,
    severity=ErrorSeverity.SAMPLE,
)

DF_T02 = ErrorCode(
    code="DF-T02",
    short_message="Tool limit exceeded",
    description="Sample exceeded maximum tool calls per query.",
    category=ErrorCategory.TOOL,
    severity=ErrorSeverity.SAMPLE,
)

DF_T03 = ErrorCode(
    code="DF-T03",
    short_message="No tool execution",
    description="Agent mode requires at least one tool execution.",
    category=ErrorCategory.TOOL,
    severity=ErrorSeverity.SAMPLE,
)

# Unknown Errors (DF-X0x)
DF_X01 = ErrorCode(
    code="DF-X01",
    short_message="Unknown error",
    description="An unexpected error occurred.",
    category=ErrorCategory.UNKNOWN,
    severity=ErrorSeverity.SAMPLE,
)


# =============================================================================
# Error Code Registry
# =============================================================================

ALL_ERROR_CODES: dict[str, ErrorCode] = {
    "DF-R01": DF_R01,
    "DF-R02": DF_R02,
    "DF-R03": DF_R03,
    "DF-R04": DF_R04,
    "DF-A01": DF_A01,
    "DF-A02": DF_A02,
    "DF-A03": DF_A03,
    "DF-N01": DF_N01,
    "DF-N02": DF_N02,
    "DF-N03": DF_N03,
    "DF-P01": DF_P01,
    "DF-P02": DF_P02,
    "DF-P03": DF_P03,
    "DF-P04": DF_P04,
    "DF-T01": DF_T01,
    "DF-T02": DF_T02,
    "DF-T03": DF_T03,
    "DF-X01": DF_X01,
}


@dataclass
class ClassifiedError:
    """Result of classifying an error."""

    error_code: ErrorCode
    detail: str | None = None
    original_error: str | None = None
    retry_after: float | None = None

    def to_event(self) -> str:
        """Format for TUI Events panel."""
        if self.retry_after:
            return self.error_code.format_event(f"retry {self.retry_after:.0f}s")
        return self.error_code.format_event(self.detail)


class ErrorClassifier:
    """Classifies exceptions and error strings into DeepFabric error codes."""

    def __init__(self, provider: str | None = None):
        """Initialize classifier.

        Args:
            provider: LLM provider name for provider-specific classification
        """
        self.provider = provider

    def classify(  # noqa: PLR0911
        self,
        error: Exception | str,
        context: dict[str, Any] | None = None,
    ) -> ClassifiedError:
        """Classify an error into a DeepFabric error code.

        Args:
            error: The exception or error string to classify
            context: Optional context (e.g., quota_info from rate limit detector)

        Returns:
            ClassifiedError with appropriate error code and details
        """
        error_str = str(error).lower()
        context = context or {}

        # Check for rate limit errors first (most common during generation)
        if self._is_rate_limit(error_str, context):
            return self._classify_rate_limit(error_str, context, error)

        # Check for authentication errors
        if self._is_auth_error(error_str):
            return ClassifiedError(
                error_code=DF_A01,
                original_error=str(error),
            )

        # Check for model not found
        if self._is_model_not_found(error_str):
            return ClassifiedError(
                error_code=DF_A02,
                original_error=str(error),
            )

        # Check for network/connection errors
        if self._is_network_error(error_str):
            return self._classify_network_error(error_str, error)

        # Check for parse/schema errors
        if self._is_parse_error(error_str, context):
            return self._classify_parse_error(error_str, context, error)

        # Check for tool errors
        if self._is_tool_error(error_str, context):
            return self._classify_tool_error(error_str, context, error)

        # Check for generic API errors
        if self._is_api_error(error_str):
            return ClassifiedError(
                error_code=DF_A03,
                original_error=str(error),
            )

        # Unknown error
        error_detail = str(error)
        if len(error_detail) > ERROR_DETAIL_MAX_LENGTH:
            error_detail = error_detail[:ERROR_DETAIL_MAX_LENGTH]
        return ClassifiedError(
            error_code=DF_X01,
            detail=error_detail,
            original_error=str(error),
        )

    def _is_rate_limit(self, error_str: str, context: dict[str, Any]) -> bool:
        """Check if error is a rate limit error."""
        rate_limit_indicators = [
            "rate limit",
            "rate_limit",
            "ratelimit",
            "429",
            "resource_exhausted",
            "quota",
            "too many requests",
        ]
        if any(ind in error_str for ind in rate_limit_indicators):
            return True
        return context.get("is_rate_limit", False)

    def _classify_rate_limit(
        self, error_str: str, context: dict[str, Any], error: Exception | str
    ) -> ClassifiedError:
        """Classify rate limit error into specific type."""
        retry_after = context.get("retry_after")
        original = str(error)

        # Daily quota exhausted
        if context.get("daily_quota_exhausted") or "per_day" in error_str:
            return ClassifiedError(
                error_code=DF_R02,
                detail="daily quota",
                retry_after=retry_after,
                original_error=original,
            )

        # Token limit
        quota_type = context.get("quota_type", "")
        if "token" in error_str or "token" in quota_type:
            return ClassifiedError(
                error_code=DF_R03,
                retry_after=retry_after,
                original_error=original,
            )

        # RPM (requests per minute) - most common
        if "per_minute" in error_str or "rpm" in error_str:
            return ClassifiedError(
                error_code=DF_R01,
                retry_after=retry_after,
                original_error=original,
            )

        # Generic rate limit
        return ClassifiedError(
            error_code=DF_R04,
            retry_after=retry_after,
            original_error=original,
        )

    def _is_auth_error(self, error_str: str) -> bool:
        """Check if error is authentication-related."""
        auth_indicators = [
            "authentication",
            "unauthorized",
            "api_key",
            "api key",
            "invalid key",
            "permission denied",
            "403",
            "401",
        ]
        return any(ind in error_str for ind in auth_indicators)

    def _is_model_not_found(self, error_str: str) -> bool:
        """Check if error is model not found."""
        return ("not found" in error_str or "404" in error_str) and "model" in error_str

    def _is_network_error(self, error_str: str) -> bool:
        """Check if error is network-related."""
        network_indicators = [
            "connection",
            "network",
            "timeout",
            "timed out",
            "503",
            "502",
            "504",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
        ]
        return any(ind in error_str for ind in network_indicators)

    def _classify_network_error(self, error_str: str, error: Exception | str) -> ClassifiedError:
        """Classify network error into specific type."""
        if "timeout" in error_str or "timed out" in error_str:
            return ClassifiedError(
                error_code=DF_N02,
                original_error=str(error),
            )

        if any(code in error_str for code in ["503", "502", "504", "service unavailable"]):
            return ClassifiedError(
                error_code=DF_N03,
                original_error=str(error),
            )

        return ClassifiedError(
            error_code=DF_N01,
            original_error=str(error),
        )

    def _is_parse_error(self, error_str: str, context: dict[str, Any]) -> bool:
        """Check if error is a parsing/schema error."""
        parse_indicators = [
            "json",
            "parse",
            "schema",
            "validation",
            "empty",
            "malformed",
            "invalid format",
        ]
        if any(ind in error_str for ind in parse_indicators):
            return True
        return context.get("error_type") in [
            "json_parsing_errors",
            "invalid_schema",
            "empty_responses",
        ]

    def _classify_parse_error(
        self, error_str: str, context: dict[str, Any], error: Exception | str
    ) -> ClassifiedError:
        """Classify parse error into specific type."""
        error_type = context.get("error_type", "")

        if "empty" in error_str or error_type == "empty_responses":
            return ClassifiedError(
                error_code=DF_P03,
                original_error=str(error),
            )

        if "schema" in error_str or error_type == "invalid_schema":
            return ClassifiedError(
                error_code=DF_P02,
                original_error=str(error),
            )

        if "json" in error_str or "parse" in error_str or error_type == "json_parsing_errors":
            return ClassifiedError(
                error_code=DF_P01,
                original_error=str(error),
            )

        return ClassifiedError(
            error_code=DF_P04,
            original_error=str(error),
        )

    def _is_tool_error(self, error_str: str, context: dict[str, Any]) -> bool:
        """Check if error is tool-related."""
        tool_indicators = ["tool", "execution", "agent mode"]
        if any(ind in error_str for ind in tool_indicators):
            return True
        return context.get("error_type") == "tool_error"

    def _classify_tool_error(
        self,
        error_str: str,
        context: dict[str, Any],  # noqa: ARG002
        error: Exception | str,
    ) -> ClassifiedError:
        """Classify tool error into specific type."""
        if "exceeds limit" in error_str or "max_tools" in error_str:
            return ClassifiedError(
                error_code=DF_T02,
                original_error=str(error),
            )

        if "requires at least one" in error_str or "no tool" in error_str:
            return ClassifiedError(
                error_code=DF_T03,
                original_error=str(error),
            )

        return ClassifiedError(
            error_code=DF_T01,
            original_error=str(error),
        )

    def _is_api_error(self, error_str: str) -> bool:
        """Check if error is a generic API error."""
        api_indicators = ["api error", "api_error", "500", "internal server error"]
        return any(ind in error_str for ind in api_indicators)


# Module-level classifier instance for convenience
_default_classifier: ErrorClassifier | None = None


def get_classifier(provider: str | None = None) -> ErrorClassifier:
    """Get an error classifier instance.

    Args:
        provider: Optional provider name for provider-specific classification

    Returns:
        ErrorClassifier instance
    """
    global _default_classifier  # noqa: PLW0603
    if provider:
        return ErrorClassifier(provider)
    if _default_classifier is None:
        _default_classifier = ErrorClassifier()
    return _default_classifier


def classify_error(
    error: Exception | str,
    provider: str | None = None,
    context: dict[str, Any] | None = None,
) -> ClassifiedError:
    """Convenience function to classify an error.

    Args:
        error: The exception or error string to classify
        provider: Optional provider name
        context: Optional context dictionary

    Returns:
        ClassifiedError with appropriate error code
    """
    classifier = get_classifier(provider)
    return classifier.classify(error, context)
