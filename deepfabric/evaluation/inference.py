"""Model inference interfaces and implementations for evaluation."""

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, Field

from ..schemas import ToolDefinition


class InferenceConfig(BaseModel):
    """Configuration for model inference."""

    model_path: str = Field(
        description="Path to model (local path or HuggingFace Hub ID)",
    )
    adapter_path: str | None = Field(
        default=None,
        description="Path to PEFT/LoRA adapter (if using adapter-based fine-tuning)",
    )
    backend: Literal["transformers", "ollama"] = Field(
        default="transformers",
        description="Inference backend to use",
    )
    use_unsloth: bool = Field(
        default=False,
        description="Use Unsloth for loading adapter (for adapters trained with Unsloth)",
    )
    max_seq_length: int = Field(
        default=2048,
        ge=1,
        description="Maximum sequence length for Unsloth models",
    )
    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit quantization (for Unsloth)",
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    max_tokens: int = Field(
        default=2048,
        ge=1,
        description="Maximum tokens to generate",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling top-p",
    )
    device: str | None = Field(
        default=None,
        description="Device to use (cuda, cpu, etc.). None for auto-detection",
    )
    batch_size: int = Field(
        default=1,
        ge=1,
        description="Batch size for inference",
    )


class ModelResponse(BaseModel):
    """Model inference response."""

    content: str = Field(description="Generated text content")
    tool_call: dict | None = Field(
        default=None,
        description="Parsed tool call if present",
    )
    raw_output: str = Field(description="Raw model output before parsing")
    finish_reason: str | None = Field(
        default=None,
        description="Reason for completion (stop, length, etc.)",
    )


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""

    def __init__(self, config: InferenceConfig):
        """Initialize inference backend.

        Args:
            config: Inference configuration
        """
        self.config = config

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        """Generate response from model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of available tools for function calling

        Returns:
            ModelResponse with generated content and parsed tool calls
        """

    @abstractmethod
    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        tools: list[ToolDefinition] | None = None,
    ) -> list[ModelResponse]:
        """Generate responses for a batch of message sequences.

        Args:
            batch_messages: List of message sequences
            tools: Optional list of available tools for function calling

        Returns:
            List of ModelResponse objects
        """

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources (GPU memory, etc.)."""


def create_inference_backend(config: InferenceConfig) -> InferenceBackend:
    """Factory function to create inference backend.

    Args:
        config: Inference configuration

    Returns:
        Initialized InferenceBackend instance

    Raises:
        ValueError: If backend type is not supported
    """
    if config.backend == "transformers":
        from .backends.transformers_backend import TransformersBackend  # noqa: PLC0415

        return TransformersBackend(config)
    if config.backend == "ollama":
        from .backends.ollama_backend import OllamaBackend  # noqa: PLC0415

        return OllamaBackend(config)

    msg = f"Unsupported backend: {config.backend}"
    raise ValueError(msg)
