"""LLM client implementation using Outlines for structured generation."""

import os

from typing import Any

import anthropic
import openai
import outlines

from google import genai

from ..exceptions import DataSetGeneratorError
from .errors import handle_provider_error


def _raise_api_key_error(env_var: str) -> None:
    """Raise an error for missing API key."""
    msg = f"{env_var} environment variable not set"
    raise DataSetGeneratorError(msg)


def _raise_unsupported_provider_error(provider: str) -> None:
    """Raise an error for unsupported provider."""
    msg = f"Unsupported provider: {provider}"
    raise DataSetGeneratorError(msg)


def _raise_generation_error(max_retries: int, error: Exception) -> None:
    """Raise an error for generation failure."""
    msg = f"Failed to generate output after {max_retries} attempts: {error!s}"
    raise DataSetGeneratorError(msg) from error


def make_outlines_model(provider: str, model_name: str, **kwargs) -> Any:
    """Create an Outlines model for the specified provider and model.

    Args:
        provider: Provider name (openai, anthropic, gemini, ollama)
        model_name: Model identifier
        **kwargs: Additional parameters passed to the client

    Returns:
        Outlines model instance

    Raises:
        DataSetGeneratorError: If provider is unsupported or configuration fails
    """
    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                _raise_api_key_error("OPENAI_API_KEY")

            client = openai.OpenAI(api_key=api_key, **kwargs)
            return outlines.from_openai(client, model_name)

        if provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                _raise_api_key_error("ANTHROPIC_API_KEY")

            client = anthropic.Anthropic(api_key=api_key, **kwargs)
            return outlines.from_anthropic(client, model_name)

        if provider == "gemini":
            api_key = None
            for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                val = os.getenv(name)
                if val:
                    api_key = val
                    break
            if not api_key:
                _raise_api_key_error("GOOGLE_API_KEY or GEMINI_API_KEY")

            client = genai.Client(api_key=api_key)
            return outlines.from_gemini(client, model_name, **kwargs)

        if provider == "ollama":
            # Use OpenAI-compatible endpoint for Ollama
            base_url = kwargs.get("base_url", "http://localhost:11434/v1")
            client = openai.OpenAI(
                base_url=base_url,
                api_key="ollama",  # Dummy key for Ollama
                **{k: v for k, v in kwargs.items() if k != "base_url"},
            )
            return outlines.from_openai(client, model_name)

        _raise_unsupported_provider_error(provider)

    except DataSetGeneratorError:
        # Re-raise our own errors (like missing API keys)
        raise
    except Exception as e:
        # Use the organized error handler
        raise handle_provider_error(e, provider, model_name) from e


class LLMClient:
    """Wrapper for Outlines models with retry logic and error handling."""

    def __init__(self, provider: str, model_name: str, **kwargs):
        """Initialize LLM client.

        Args:
            provider: Provider name
            model_name: Model identifier
            **kwargs: Additional client configuration
        """
        self.provider = provider
        self.model_name = model_name
        self.model: Any = make_outlines_model(provider, model_name, **kwargs)
        if self.model is None:
            msg = f"Failed to create model for {provider}/{model_name}"
            raise DataSetGeneratorError(msg)

    def generate(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs) -> Any:
        """Generate structured output using the provided schema.

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts
            **kwargs: Additional generation parameters

        Returns:
            Generated output matching the schema

        Raises:
            DataSetGeneratorError: If generation fails after all retries
        """
        last_error: Exception | None = None

        # Convert provider-specific parameters
        kwargs = self._convert_generation_params(**kwargs)

        for attempt in range(max_retries):
            try:
                # Generate JSON string with Outlines using the schema as output type
                json_output = self.model(prompt, schema, **kwargs)

                # Parse and validate the JSON response with Pydantic
                return schema.model_validate_json(json_output)

            except Exception as e:
                last_error = e
                if attempt == max_retries - 1:
                    break

        if last_error is not None:
            _raise_generation_error(max_retries, last_error)
        msg = "Failed to generate output: no attempts were made"
        raise DataSetGeneratorError(msg)

    def _convert_generation_params(self, **kwargs) -> dict:
        """Convert generic parameters to provider-specific ones."""
        # Convert max_tokens to max_output_tokens for Gemini
        if self.provider == "gemini" and "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        return kwargs

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider}, model={self.model_name})"
