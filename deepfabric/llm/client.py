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
    Recursively remove additionalProperties from JSON schema and handle dict[str, Any] fields.

    Gemini doesn't support:
    1. additionalProperties field in JSON schemas
    2. Objects with no properties defined (e.g., dict[str, Any])

    Fields like dict[str, Any] have additionalProperties: true and no properties defined.
    Gemini requires that object-type fields must have properties, so we exclude these
    fields from the schema entirely.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict without additionalProperties and dict[str, Any] fields
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # For Gemini, identify and remove dict[str, Any] fields
    # These have additionalProperties: true and no properties, which Gemini rejects
    if "properties" in schema_dict:
        properties_to_remove = []
        for prop_name, prop_schema in schema_dict["properties"].items():
            # Combine isinstance check with specific conditions to avoid nested ifs
            if isinstance(prop_schema, dict) and prop_schema.get("additionalProperties") is True:
                # Remove fields with additionalProperties: true (e.g., dict[str, Any])
                # Gemini requires objects to have properties defined
                properties_to_remove.append(prop_name)
            elif isinstance(prop_schema, dict) and (
                prop_schema.get("type") == "object"
                and "properties" not in prop_schema
                and "$ref" not in prop_schema
            ):
                # Also remove objects with no properties and no explicit additionalProperties
                properties_to_remove.append(prop_name)
            elif isinstance(prop_schema, dict) and "anyOf" in prop_schema:
                # Check if anyOf contains an object with no properties (e.g., dict[str, Any] | None)
                # Remove the entire field if any variant is an incompatible object
                for variant in prop_schema["anyOf"]:
                    if isinstance(variant, dict) and (
                        variant.get("type") == "object"
                        and "properties" not in variant
                        and "$ref" not in variant
                    ):
                        # This field has an object variant with no properties - remove it
                        properties_to_remove.append(prop_name)
                        break

        # Remove incompatible properties from the schema
        for prop_name in properties_to_remove:
            del schema_dict["properties"][prop_name]

        # Update required array to exclude removed properties
        if "required" in schema_dict:
            schema_dict["required"] = [
                r for r in schema_dict["required"] if r not in properties_to_remove
            ]

    # Remove additionalProperties from current level
    schema_dict.pop("additionalProperties", None)

    # Recursively process nested structures
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _strip_additional_properties(def_schema)

    # Process properties recursively
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            schema_dict["properties"][prop_name] = _strip_additional_properties(prop_schema)

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _strip_additional_properties(schema_dict["items"])

    # Process union types (anyOf, oneOf, allOf)
    for union_key in ["anyOf", "oneOf", "allOf"]:
        if union_key in schema_dict:
            schema_dict[union_key] = [
                _strip_additional_properties(variant) for variant in schema_dict[union_key]
            ]

    return schema_dict


def _create_gemini_compatible_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """
    Create a Gemini-compatible version of a Pydantic schema.

    Gemini doesn't support:
    1. additionalProperties field in JSON schemas
    2. Objects with no properties defined (e.g., dict[str, Any])

    This function creates a wrapper that generates schemas meeting these requirements
    by removing incompatible fields entirely.

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


def _ensure_openai_strict_mode_compliance(schema_dict: dict) -> dict:
    """
    Ensure schema complies with OpenAI's strict mode requirements.

    OpenAI's strict mode requires:
    1. For objects, 'additionalProperties' must be explicitly set to false
    2. ALL properties must be in the 'required' array (no optional fields allowed)
    3. No fields with additionalProperties: true (incompatible with strict mode)

    Fields like dict[str, Any] have additionalProperties: true and cannot be
    represented in strict mode, so they are excluded from the schema entirely.

    Args:
        schema_dict: JSON schema dictionary

    Returns:
        Modified schema dict meeting OpenAI strict mode requirements
    """
    if not isinstance(schema_dict, dict):
        return schema_dict

    # For OpenAI strict mode, identify and remove dict[str, Any] fields
    # These have additionalProperties: true which is incompatible with strict mode
    if "properties" in schema_dict:
        properties_to_remove = []
        for prop_name, prop_schema in schema_dict["properties"].items():
            # Check for direct additionalProperties: true
            if isinstance(prop_schema, dict) and prop_schema.get("additionalProperties") is True:
                # Remove fields with additionalProperties: true (e.g., dict[str, Any])
                properties_to_remove.append(prop_name)
            # Check for anyOf containing object variants with additionalProperties: true
            elif isinstance(prop_schema, dict) and "anyOf" in prop_schema:
                for variant in prop_schema["anyOf"]:
                    if isinstance(variant, dict) and variant.get("additionalProperties") is True:
                        # This anyOf contains an incompatible object variant - remove entire field
                        properties_to_remove.append(prop_name)
                        break

        # Remove incompatible properties from the schema
        for prop_name in properties_to_remove:
            del schema_dict["properties"][prop_name]

        # Update required array to exclude removed properties
        if "required" in schema_dict:
            schema_dict["required"] = [
                r for r in schema_dict["required"] if r not in properties_to_remove
            ]

        # After removing incompatible fields, ensure ALL remaining properties are required
        # OpenAI strict mode doesn't allow optional fields
        property_keys = list(schema_dict["properties"].keys())
        schema_dict["required"] = property_keys
        schema_dict["additionalProperties"] = False

    # For all objects (including those without properties), set additionalProperties to false
    if schema_dict.get("type") == "object":
        schema_dict["additionalProperties"] = False

    # Recursively process nested structures
    if "$defs" in schema_dict:
        for def_name, def_schema in schema_dict["$defs"].items():
            schema_dict["$defs"][def_name] = _ensure_openai_strict_mode_compliance(def_schema)

    # Process properties recursively
    if "properties" in schema_dict:
        for prop_name, prop_schema in schema_dict["properties"].items():
            schema_dict["properties"][prop_name] = _ensure_openai_strict_mode_compliance(
                prop_schema
            )

    # Process items (for arrays)
    if "items" in schema_dict:
        schema_dict["items"] = _ensure_openai_strict_mode_compliance(schema_dict["items"])

    return schema_dict


def _create_openai_compatible_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """
    Create an OpenAI-compatible version of a Pydantic schema.

    OpenAI's strict mode requires that all objects have 'additionalProperties: false'.
    This function ensures the schema meets those requirements while preserving
    Pydantic's correct handling of required vs optional fields.

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
            # Ensure OpenAI strict mode compliance
            return _ensure_openai_strict_mode_compliance(original_schema)

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

        # For Gemini, use compatible schema without additionalProperties
        # For OpenAI, ensure all properties are in required array
        generation_schema = schema
        if self.provider == "gemini" and isinstance(schema, type) and issubclass(schema, BaseModel):
            generation_schema = _create_gemini_compatible_schema(schema)
        elif (
            self.provider == "openai" and isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
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

        # For Gemini, use compatible schema without additionalProperties
        # For OpenAI, ensure all properties are in required array
        generation_schema = schema
        if self.provider == "gemini" and isinstance(schema, type) and issubclass(schema, BaseModel):
            generation_schema = _create_gemini_compatible_schema(schema)
        elif (
            self.provider == "openai" and isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            generation_schema = _create_openai_compatible_schema(schema)

        # Ensure we have an async model; if not, fall back to running the sync path
        async_model = self.async_model
        if async_model is None:
            return await asyncio.to_thread(self.generate, prompt, schema, **kwargs)

        # Call the async model (guaranteed non-None by check above)
        json_output = await async_model(prompt, generation_schema, **kwargs)
        # Validate with original schema to ensure proper validation
        return schema.model_validate_json(json_output)

    async def generate_async_stream(self, prompt: str, schema: Any, max_retries: int = 3, **kwargs):  # noqa: ARG002
        """Asynchronously generate structured output with streaming text chunks.

        This method streams the LLM's output text as it's generated, then returns
        the final parsed Pydantic model. It yields tuples of (chunk, result) where:
        - During streaming: (text_chunk, None)
        - When complete: (None, final_pydantic_model)

        Args:
            prompt: Input prompt
            schema: Pydantic model or other schema type
            max_retries: Maximum number of retry attempts (deprecated, use rate_limit_config)
            **kwargs: Additional generation parameters

        Yields:
            tuple[str | None, Any | None]:
                - (chunk, None) during streaming
                - (None, model) when generation is complete

        Raises:
            DataSetGeneratorError: If generation fails after all retries

        Note:
            The max_retries parameter is deprecated. Use rate_limit_config in __init__ instead.

        Example:
            >>> async for chunk, result in client.generate_async_stream(prompt, MyModel):
            ...     if chunk:
            ...         print(chunk, end='', flush=True)  # Display streaming text
            ...     if result:
            ...         return result  # Final parsed model
        """
        # Call streaming generation directly (retry decorator doesn't work with generators)
        kwargs = self._convert_generation_params(**kwargs)

        # Apply provider-specific schema transformations
        generation_schema = schema
        if self.provider == "gemini" and isinstance(schema, type) and issubclass(schema, BaseModel):
            generation_schema = _create_gemini_compatible_schema(schema)
        elif (
            self.provider == "openai" and isinstance(schema, type) and issubclass(schema, BaseModel)
        ):
            generation_schema = _create_openai_compatible_schema(schema)

        # Check if model supports streaming
        async_model = self.async_model or self.model
        if not hasattr(async_model, "generate_stream"):
            # Fallback: no streaming support, yield entire result at once
            result = await self.generate_async(prompt, schema, max_retries, **kwargs)
            yield (None, result)
            return

        # Stream generation
        accumulated_text = []
        try:
            # For sync models used in async context
            if self.async_model is None:
                # Run streaming generation in thread pool
                from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

                def _sync_stream():
                    return list(self.model.generate_stream(prompt, generation_schema, **kwargs))

                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_sync_stream)
                    # Wait for completion and get all chunks
                    chunks = await asyncio.wrap_future(future)
                    for chunk in chunks:
                        accumulated_text.append(chunk)
                        yield (chunk, None)
            else:
                # True async streaming
                stream = self.async_model.generate_stream(prompt, generation_schema, **kwargs)
                async for chunk in stream:
                    accumulated_text.append(chunk)
                    yield (chunk, None)

            # Parse accumulated JSON with original schema
            full_text = "".join(accumulated_text)
            result = schema.model_validate_json(full_text)
            yield (None, result)

        except Exception as e:
            # Wrap and raise error
            raise DataSetGeneratorError(f"Streaming generation failed: {e}") from e

    def _convert_generation_params(self, **kwargs) -> dict:
        """Convert generic parameters to provider-specific ones."""
        # Convert max_tokens to max_output_tokens for Gemini
        if self.provider == "gemini" and "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")

        return kwargs

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider}, model={self.model_name})"
