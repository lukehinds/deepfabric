"""HuggingFace Transformers provider for local model inference with Outlines integration."""

import contextlib
import logging

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..exceptions import DataSetGeneratorError

logger = logging.getLogger(__name__)


class TransformersConfig(BaseModel):
    """Configuration for HuggingFace Transformers provider.

    This configuration enables loading and running models locally using the
    transformers library with Outlines integration for structured generation.
    """

    model_id: str = Field(..., description="HuggingFace model identifier or local path")

    device: str | None = Field(
        default=None,
        description="Device to load model on (cuda, cpu, or None for auto)",
    )

    device_map: str | dict[str, Any] | None = Field(
        default=None,
        description="Device map for model layers (auto, balanced, sequential, or custom dict)",
    )

    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] | None = Field(
        default="auto",
        description="Torch dtype for model weights",
    )

    load_in_8bit: bool = Field(
        default=False,
        description="Load model in 8-bit quantized format (requires bitsandbytes)",
    )

    load_in_4bit: bool = Field(
        default=False,
        description="Load model in 4-bit quantized format (requires bitsandbytes)",
    )

    trust_remote_code: bool = Field(
        default=False,
        description="Allow execution of custom code from model repository",
    )

    use_fast_tokenizer: bool = Field(
        default=True,
        description="Use fast tokenizer implementation when available",
    )

    model_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to model initialization",
    )

    tokenizer_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments passed to tokenizer initialization",
    )

    chat_template: str | None = Field(
        default=None,
        description="Custom chat template (None uses model's default or ChatML fallback)",
    )

    max_length: int = Field(
        default=8192,
        ge=512,
        le=131072,
        description="Maximum sequence length for model context",
    )


class TransformersProvider:
    """Provider for HuggingFace Transformers models with Outlines structured generation.

    This provider enables using local HuggingFace models for inference within
    DeepFabric's data generation pipeline. It integrates with Outlines to ensure
    structured JSON output that matches Pydantic schemas.

    Key Features:
    - Local model inference (no API calls)
    - Quantization support (8-bit, 4-bit)
    - Device management (CPU, CUDA, multi-GPU)
    - Chat template support
    - Outlines integration for structured generation

    Example:
        ```python
        from deepfabric import Tree

        tree = Tree(
            topic_prompt="Machine Learning",
            provider="transformers",
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            device="cuda",
            torch_dtype="bfloat16"
        )
        ```
    """

    def __init__(self, model_id: str, config: TransformersConfig | None = None, **kwargs):
        """Initialize the Transformers provider.

        Args:
            model_id: HuggingFace model identifier or local path
            config: TransformersConfig instance (optional)
            **kwargs: Additional configuration parameters passed to TransformersConfig

        Raises:
            DataSetGeneratorError: If model loading fails or dependencies are missing
        """
        # Validate dependencies
        self._validate_dependencies()

        # Merge config and kwargs
        if config is None:
            config = TransformersConfig(model_id=model_id, **kwargs)
        else:
            # Override config with any provided kwargs
            config_dict = config.model_dump()
            config_dict.update(kwargs)
            config = TransformersConfig(**config_dict)

        self.config = config
        self.model_id = config.model_id

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.outlines_model = None

        logger.info(
            "Initializing Transformers provider with model: %s on device: %s",
            self.model_id,
            self.config.device or "auto",
        )

        self._load_model_and_tokenizer()
        self._setup_outlines()

    def _validate_dependencies(self):
        """Validate that required dependencies are installed."""
        try:
            import transformers  # noqa: F401, PLC0415
        except ImportError as e:
            msg = (
                "transformers library not found. Install with: "
                "pip install 'deepfabric[training]' or pip install transformers>=4.45.0"
            )
            raise DataSetGeneratorError(msg) from e

        try:
            import torch  # noqa: F401, PLC0415
        except ImportError as e:
            msg = (
                "torch library not found. Install with: "
                "pip install 'deepfabric[training]' or pip install torch>=2.0.0"
            )
            raise DataSetGeneratorError(msg) from e

    def _load_model_and_tokenizer(self):
        """Load the model and tokenizer from HuggingFace."""
        try:
            import torch  # noqa: PLC0415

            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

        except ImportError as e:
            msg = "Failed to import transformers or torch"
            raise DataSetGeneratorError(msg) from e

        try:
            # Prepare model loading kwargs
            model_kwargs = {
                "trust_remote_code": self.config.trust_remote_code,
                **self.config.model_kwargs,
            }

            # Handle dtype
            if self.config.torch_dtype:
                if self.config.torch_dtype == "auto":
                    model_kwargs["torch_dtype"] = "auto"
                elif self.config.torch_dtype == "float16":
                    model_kwargs["torch_dtype"] = torch.float16
                elif self.config.torch_dtype == "bfloat16":
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif self.config.torch_dtype == "float32":
                    model_kwargs["torch_dtype"] = torch.float32

            # Handle device placement
            if self.config.device_map:
                model_kwargs["device_map"] = self.config.device_map
            elif self.config.device:
                model_kwargs["device_map"] = self.config.device

            # Handle quantization
            if self.config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
                logger.info("Loading model in 8-bit quantized format")
            elif self.config.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
                logger.info("Loading model in 4-bit quantized format")

            # Load model
            logger.info("Loading model %s with kwargs: %s", self.model_id, model_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

            # Load tokenizer
            tokenizer_kwargs = {
                "use_fast": self.config.use_fast_tokenizer,
                "trust_remote_code": self.config.trust_remote_code,
                **self.config.tokenizer_kwargs,
            }

            logger.info("Loading tokenizer for %s", self.model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                **tokenizer_kwargs,
            )

            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Set pad_token to eos_token")

            # Apply chat template if provided
            if self.config.chat_template:
                self.tokenizer.chat_template = self.config.chat_template
                logger.info("Applied custom chat template")
            elif (
                not hasattr(self.tokenizer, "chat_template") or self.tokenizer.chat_template is None
            ):
                # Fallback to ChatML template
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
                )
                logger.info("Applied default ChatML chat template")

            logger.info(
                "Successfully loaded model and tokenizer for %s",
                self.model_id,
            )

        except Exception as e:
            msg = f"Failed to load model {self.model_id}: {str(e)}"
            raise DataSetGeneratorError(msg) from e

    def _setup_outlines(self):
        """Setup Outlines model wrapper for structured generation."""
        try:
            # Import Transformers class from outlines.models
            from outlines.models import Transformers  # noqa: PLC0415

            # Create Outlines model from transformers model and tokenizer
            self.outlines_model = Transformers(
                self.model,
                self.tokenizer,
            )

            logger.info("Successfully created Outlines model wrapper for structured generation")

        except ImportError as e:
            msg = "Outlines library not found. This should not happen as it's a core dependency."
            raise DataSetGeneratorError(msg) from e
        except Exception as e:
            msg = f"Failed to setup Outlines integration: {str(e)}"
            raise DataSetGeneratorError(msg) from e

    def get_outlines_model(self) -> Any:
        """Get the Outlines model for structured generation.

        Returns:
            Outlines model instance that can be used with outlines.generate

        Raises:
            DataSetGeneratorError: If model is not initialized
        """
        if self.outlines_model is None:
            msg = "Outlines model not initialized. Call _setup_outlines() first."
            raise DataSetGeneratorError(msg)
        return self.outlines_model

    def unload(self):
        """Unload the model from memory to free resources."""
        import gc  # noqa: PLC0415

        try:
            import torch  # noqa: PLC0415
        except ImportError:
            logger.warning("torch not available for cleanup")
            return

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.outlines_model is not None:
            del self.outlines_model
            self.outlines_model = None

        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Unloaded model and freed resources")

    def __del__(self):
        """Cleanup resources when provider is destroyed."""
        with contextlib.suppress(Exception):
            self.unload()
            pass  # Ignore errors during cleanup

    def __repr__(self) -> str:
        """Return string representation of the provider."""
        return (
            f"TransformersProvider(model_id={self.model_id}, "
            f"device={self.config.device or 'auto'}, "
            f"dtype={self.config.torch_dtype})"
        )


def make_transformers_model(model_name: str, **kwargs) -> Any:
    """Factory function to create a Transformers Outlines model.

    This function is called by the LLMClient when provider="transformers".

    Args:
        model_name: HuggingFace model identifier
        **kwargs: Additional configuration parameters

    Returns:
        Outlines model instance ready for structured generation

    Example:
        ```python
        model = make_transformers_model(
            "meta-llama/Llama-3.1-8B-Instruct",
            device="cuda",
            torch_dtype="bfloat16"
        )
        ```
    """
    provider = TransformersProvider(model_name, **kwargs)
    return provider.get_outlines_model()
