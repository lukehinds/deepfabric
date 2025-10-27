"""LLM client implementation using Outlines for structured generation."""

import asyncio
import os

from typing import Any

import anthropic
import openai
import outlines

from google import genai
from pydantic import BaseModel

from ..exceptions import DataSetGeneratorError
from .errors import handle_provider_error
from .rate_limit_config import RateLimitConfig, create_rate_limit_config
from .retry_handler import RetryHandler, retry_with_backoff, retry_with_backoff_async


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


def _strip_additional_properties(schema_dict: dict) -> dict:
    """
    Recursively remove additionalProperties from JSON schema.

    Gemini doesn't support additionalProperties field in JSON schemas.
    This function strips it out from the schema and all nested definitions.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict without additionalProperties
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # Remove additionalProperties from current level
    schema_dict.pop("additionalProperties", None)

    # Recursively process nested structures
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _strip_additional_properties(def_schema)

    # Process properties
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            schema_dict["properties"][prop_name] = _strip_additional_properties(prop_schema)

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _strip_additional_properties(schema_dict["items"])

    return schema_dict


def _create_gemini_compatible_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """
    Create a Gemini-compatible version of a Pydantic schema.

    Gemini doesn't support additionalProperties. This function creates a wrapper
    that generates schemas without this field.

    Args:
        schema: Original Pydantic model

    Returns:
        Wrapper model that generates Gemini-compatible schemas
    """

    # Create a new model class that overrides model_json_schema
    class GeminiCompatModel(schema):  # type: ignore[misc,valid-type]
        @classmethod
        def model_json_schema(cls, **kwargs):
            # Get the original schema
            original_schema = super().model_json_schema(**kwargs)
            # Strip additionalProperties
            return _strip_additional_properties(original_schema)

    # Set name and docstring
    GeminiCompatModel.__name__ = f"{schema.__name__}GeminiCompat"
    GeminiCompatModel.__doc__ = schema.__doc__

    return GeminiCompatModel


def _add_additional_properties_false(schema_dict: dict) -> dict:
    """
    Recursively add additionalProperties: false to all object types in JSON schema.

    OpenAI's structured outputs require additionalProperties to be explicitly set to false
    for all object types to ensure strict schema validation. However, OpenAI does NOT support
    truly dynamic objects (dict[str, Any]) that have no defined properties.

    For such fields, we remove them from the required array and convert them to strings.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict with additionalProperties: false for all objects
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # Handle objects with defined properties vs dynamic objects (dict[str, Any])
    if schema_dict.get("type") == "object":
        if "properties" in schema_dict and schema_dict.get("properties"):
            # Object with defined properties: set additionalProperties: false
            schema_dict["additionalProperties"] = False

            # Convert dynamic dict fields to strings
            # OpenAI doesn't support objects with no defined properties
            # This includes dict[str, Any], dict[str, str], etc.
            properties = schema_dict["properties"]
            for field_name, field_schema in properties.items():
                # Check if this is a dict type (object with no defined properties)
                # These have type="object", no "properties", but may have additionalProperties
                if (
                    field_schema.get("type") == "object"
                    and not field_schema.get("properties")
                    and "additionalProperties" in field_schema
                ):
                    # Convert to string type (JSON-encoded)
                    properties[field_name] = {
                        "type": "string",
                        "description": f"{field_schema.get('description', '')} (JSON-encoded object string)",
                        "title": field_schema.get("title", field_name),
                    }

            # Ensure all properties are in required array (OpenAI requirement)
            if "required" not in schema_dict:
                schema_dict["required"] = list(properties.keys())
            else:
                # Make sure all properties are in required
                existing_required = set(schema_dict["required"])
                all_props = set(properties.keys())
                missing = all_props - existing_required
                if missing:
                    schema_dict["required"] = schema_dict["required"] + list(missing)
        else:
            # Object with no properties (dict[str, Any]): remove additionalProperties
            schema_dict.pop("additionalProperties", None)

    # Recursively process nested structures
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _add_additional_properties_false(def_schema)

    # Process properties
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            # If a property has $ref, it cannot have other keywords (OpenAI requirement)
            if "$ref" in prop_schema and len(prop_schema) > 1:
                # Keep only the $ref
                schema_dict["properties"][prop_name] = {"$ref": prop_schema["$ref"]}
            else:
                schema_dict["properties"][prop_name] = _add_additional_properties_false(prop_schema)

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _add_additional_properties_false(schema_dict["items"])

    # Process anyOf, allOf, oneOf
    for key in ["anyOf", "allOf", "oneOf"]:
        if key in schema_dict:
            schema_dict[key] = [_add_additional_properties_false(s) for s in schema_dict[key]]

    return schema_dict


def _create_openai_compatible_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """
    Create an OpenAI-compatible version of a Pydantic schema.

    OpenAI's structured outputs require:
    1. All object types must have additionalProperties: false
    2. All properties must be in the required array (no optional fields with defaults)

    Args:
        schema: Original Pydantic model

    Returns:
        Wrapper model that generates OpenAI-compatible schemas
    """

    # Create a new model class that overrides model_json_schema
    class OpenAICompatModel(schema):  # type: ignore[misc,valid-type]
        @classmethod
        def model_json_schema(cls, **kwargs):
            # Get the original schema
            original_schema = super().model_json_schema(**kwargs)
            # Add additionalProperties: false to all object types
            return _add_additional_properties_false(original_schema)

    # Set name and docstring
    OpenAICompatModel.__name__ = f"{schema.__name__}OpenAICompat"
    OpenAICompatModel.__doc__ = schema.__doc__

    return OpenAICompatModel


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


def make_async_outlines_model(provider: str, model_name: str, **kwargs) -> Any | None:
    """Create an async Outlines model when the provider supports it.

    Returns ``None`` for providers without async-capable clients.
    """

    try:
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                _raise_api_key_error("OPENAI_API_KEY")

            client = openai.AsyncOpenAI(api_key=api_key, **kwargs)
            return outlines.from_openai(client, model_name)

        if provider == "ollama":
            base_url = kwargs.get("base_url", "http://localhost:11434/v1")
            client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key="ollama",
                **{k: v for k, v in kwargs.items() if k != "base_url"},
            )
            return outlines.from_openai(client, model_name)

    except DataSetGeneratorError:
        raise
    except Exception as e:
        raise handle_provider_error(e, provider, model_name) from e

    # Outlines does not currently expose async structured generation wrappers
    # for the remaining providers. Fallback to synchronous execution later.
    return None


class LLMClient:
    """Wrapper for Outlines models with retry logic and error handling."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        rate_limit_config: RateLimitConfig | dict | None = None,
        **kwargs,
    ):
        """Initialize LLM client.

        Args:
            provider: Provider name
            model_name: Model identifier
            rate_limit_config: Rate limiting configuration (None uses provider defaults)
            **kwargs: Additional client configuration
        """
        self.provider = provider
        self.model_name = model_name

        # Initialize rate limiting
        if isinstance(rate_limit_config, dict):
            self.rate_limit_config = create_rate_limit_config(provider, rate_limit_config)
        elif rate_limit_config is None:
            # Use provider-specific defaults
            from .rate_limit_config import (  # noqa: PLC0415
                get_default_rate_limit_config,
            )

            self.rate_limit_config = get_default_rate_limit_config(provider)
        else:
            self.rate_limit_config = rate_limit_config

        self.retry_handler = RetryHandler(self.rate_limit_config, provider)

        self.model: Any = make_outlines_model(provider, model_name, **kwargs)
        self.async_model: Any | None = make_async_outlines_model(provider, model_name, **kwargs)
        if self.model is None:
            msg = f"Failed to create model for {provider}/{model_name}"
            raise DataSetGeneratorError(msg)

    def generate(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs) -> Any:  # noqa: ARG002
        """Generate structured output using the provided schema.

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts (deprecated, use rate_limit_config)
            **kwargs: Additional generation parameters

        Returns:
            Generated output matching the schema

        Raises:
            DataSetGeneratorError: If generation fails after all retries

        Note:
            The max_retries parameter is deprecated. Use rate_limit_config in __init__ instead.
            If provided, it will be ignored in favor of the configured retry handler.
        """
        return self._generate_with_retry(prompt, schema, **kwargs)

    @retry_with_backoff
    def _generate_with_retry(self, prompt: str, schema: Any, **kwargs) -> Any:
        """Internal method that performs actual generation with retry wrapper.

        This method is decorated with retry_with_backoff to handle rate limits
        and transient errors automatically.
        """
        # Convert provider-specific parameters
        kwargs = self._convert_generation_params(**kwargs)

        # Apply provider-specific schema transformations
        generation_schema = schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if self.provider == "gemini":
                # Gemini doesn't support additionalProperties
                generation_schema = _create_gemini_compatible_schema(schema)
            elif self.provider in ("openai", "ollama"):
                # OpenAI (and Ollama via OpenAI API) requires additionalProperties: false
                generation_schema = _create_openai_compatible_schema(schema)

        # Generate JSON string with Outlines using the schema as output type
        json_output = self.model(prompt, generation_schema, **kwargs)

        # Parse and validate the JSON response with the ORIGINAL schema
        # This ensures we still get proper validation
        return schema.model_validate_json(json_output)

    async def generate_async(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs) -> Any:  # noqa: ARG002
        """Asynchronously generate structured output using provider async clients.

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts (deprecated, use rate_limit_config)
            **kwargs: Additional generation parameters

        Returns:
            Generated output matching the schema

        Raises:
            DataSetGeneratorError: If generation fails after all retries

        Note:
            The max_retries parameter is deprecated. Use rate_limit_config in __init__ instead.
            If provided, it will be ignored in favor of the configured retry handler.
        """
        if self.async_model is None:
            # Fallback to running the synchronous path in a worker thread
            return await asyncio.to_thread(self.generate, prompt, schema, **kwargs)

        return await self._generate_async_with_retry(prompt, schema, **kwargs)

    @retry_with_backoff_async
    async def _generate_async_with_retry(self, prompt: str, schema: Any, **kwargs) -> Any:
        """Internal async method that performs actual generation with retry wrapper.

        This method is decorated with retry_with_backoff_async to handle rate limits
        and transient errors automatically.
        """
        kwargs = self._convert_generation_params(**kwargs)

        # Apply provider-specific schema transformations
        generation_schema = schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            if self.provider == "gemini":
                # Gemini doesn't support additionalProperties
                generation_schema = _create_gemini_compatible_schema(schema)
            elif self.provider in ("openai", "ollama"):
                # OpenAI (and Ollama via OpenAI API) requires additionalProperties: false
                generation_schema = _create_openai_compatible_schema(schema)

        json_output = await self.async_model(prompt, generation_schema, **kwargs)

        # Validate with original schema to ensure proper validation
        return schema.model_validate_json(json_output)

    def _convert_generation_params(self, **kwargs) -> dict:
        """Convert generic parameters to provider-specific ones."""
        # Convert max_tokens to max_output_tokens for Gemini
        if self.provider == "gemini" and "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        return kwargs

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider}, model={self.model_name})"
