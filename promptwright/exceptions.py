class PromptWrightError(Exception):
    """Base exception class for PromptWright."""

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ConfigurationError(PromptWrightError):
    """Raised when there is an error in configuration."""

    pass


class ValidationError(PromptWrightError):
    """Raised when data validation fails."""

    pass


class ModelError(PromptWrightError):
    """Raised when there is an error with LLM model operations."""

    pass


class TopicTreeError(PromptWrightError):
    """Raised when there is an error in topic tree operations."""

    pass


class DataEngineError(PromptWrightError):
    """Raised when there is an error in data engine operations."""

    pass


class DatasetError(PromptWrightError):
    """Raised when there is an error in dataset operations."""

    pass


class HubUploadError(PromptWrightError):
    """Raised when there is an error uploading to Hugging Face Hub."""

    pass


class JSONParsingError(ValidationError):
    """Raised when JSON parsing fails."""

    pass


class APIError(ModelError):
    """Raised when API calls fail."""

    pass


class RetryExhaustedError(ModelError):
    """Raised when maximum retries are exceeded."""

    pass
