"""
Model capability detection from tokenizer configuration.

Detects model features like reasoning support, tool calling, and special token handling.
"""

import json
import logging

from typing import Any

from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)


class TokenizerConfig:
    """
    Wrapper for tokenizer_config.json with parsed capabilities.
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize with tokenizer config dictionary.

        Args:
            config: Tokenizer configuration from tokenizer_config.json
        """
        self.config = config
        self._special_tokens_cache: dict[str, str] | None = None
        self._added_tokens_cache: list[str] | None = None

    @classmethod
    def from_model_id(cls, model_id: str) -> "TokenizerConfig":
        """
        Load tokenizer config from HuggingFace model ID.

        Args:
            model_id: HuggingFace model ID (e.g., "google/gemma-7b-it")

        Returns:
            TokenizerConfig instance

        Raises:
            Exception: If config cannot be downloaded
        """
        try:
            config_path = hf_hub_download(  #  nosec
                repo_id=model_id, filename="tokenizer_config.json", repo_type="model"
            )
            with open(config_path) as f:
                config = json.load(f)
            logger.info(f"Loaded tokenizer config for {model_id}")
            return cls(config)
        except Exception:
            logger.exception(f"Failed to load tokenizer config for {model_id}")
            raise

    @property
    def chat_template(self) -> str | None:
        """Get the chat template (Jinja2 string)."""
        return self.config.get("chat_template")

    @property
    def bos_token(self) -> str | None:
        """Get the beginning-of-sequence token."""
        # Can be string or dict with "content" key
        token = self.config.get("bos_token")
        if isinstance(token, dict):
            return token.get("content")
        return token

    @property
    def eos_token(self) -> str | None:
        """Get the end-of-sequence token."""
        token = self.config.get("eos_token")
        if isinstance(token, dict):
            return token.get("content")
        return token

    @property
    def pad_token(self) -> str | None:
        """Get the padding token."""
        token = self.config.get("pad_token")
        if isinstance(token, dict):
            return token.get("content")
        return token

    @property
    def unk_token(self) -> str | None:
        """Get the unknown token."""
        token = self.config.get("unk_token")
        if isinstance(token, dict):
            return token.get("content")
        return token

    @property
    def add_bos_token(self) -> bool:
        """Whether to automatically add BOS token."""
        return self.config.get("add_bos_token", False)

    @property
    def add_eos_token(self) -> bool:
        """Whether to automatically add EOS token."""
        return self.config.get("add_eos_token", False)

    @property
    def model_max_length(self) -> int | None:
        """
        Get maximum sequence length the model can handle.

        Returns:
            Max length in tokens, or None if not specified
        """
        return self.config.get("model_max_length")

    @property
    def padding_side(self) -> str:
        """
        Get padding side (left or right).

        Returns:
            "left" or "right", defaults to "right"
        """
        return self.config.get("padding_side", "right")

    @property
    def tokenizer_class(self) -> str | None:
        """Get the tokenizer class name."""
        return self.config.get("tokenizer_class")

    @property
    def clean_up_tokenization_spaces(self) -> bool:
        """Whether to clean up tokenization spaces."""
        return self.config.get("clean_up_tokenization_spaces", True)

    @property
    def special_tokens(self) -> dict[str, str]:
        """
        Get all special tokens.

        Returns:
            Dictionary mapping token names to token strings
        """
        if self._special_tokens_cache is not None:
            return self._special_tokens_cache

        tokens = {}

        # Standard special tokens
        if self.bos_token:
            tokens["bos"] = self.bos_token
        if self.eos_token:
            tokens["eos"] = self.eos_token
        if self.pad_token:
            tokens["pad"] = self.pad_token
        if self.unk_token:
            tokens["unk"] = self.unk_token

        # Additional tokens from config
        for key in ["cls_token", "sep_token", "mask_token"]:
            token = self.config.get(key)
            if token:
                if isinstance(token, dict):
                    tokens[key.replace("_token", "")] = token.get("content")
                else:
                    tokens[key.replace("_token", "")] = token

        self._special_tokens_cache = tokens
        return tokens

    @property
    def added_tokens(self) -> list[str]:
        """
        Get list of all added tokens (including special tokens).

        Returns:
            List of token strings
        """
        if self._added_tokens_cache is not None:
            return self._added_tokens_cache

        tokens = []
        added_tokens_decoder = self.config.get("added_tokens_decoder", {})

        for _token_id, token_info in added_tokens_decoder.items():
            if isinstance(token_info, dict):
                content = token_info.get("content")
                if content:
                    tokens.append(content)

        self._added_tokens_cache = tokens
        return tokens

    def has_token(self, token: str) -> bool:
        """
        Check if a specific token exists in added tokens.

        Args:
            token: Token string to check (e.g., "<think>", "<tool_call>")

        Returns:
            True if token exists, False otherwise
        """
        return token in self.added_tokens


class CapabilityDetector:
    """
    Detect model capabilities from tokenizer configuration.

    Detects:
    - Reasoning support (think tags, etc.)
    - Tool calling support
    - Special token handling
    """

    def __init__(self, tokenizer_config: TokenizerConfig, model_mappings: dict | None = None):
        """
        Initialize capability detector.

        Args:
            tokenizer_config: TokenizerConfig instance
            model_mappings: Optional model mappings with capability indicators
        """
        self.tokenizer_config = tokenizer_config
        self.model_mappings = model_mappings or {}

    def detect_all(self) -> dict[str, Any]:
        """
        Detect all capabilities.

        Returns:
            Dictionary with detected capabilities:
            {
                "reasoning": {"native_support": bool, "tags": list[str], ...},
                "tools": {"native_support": bool, "format": str, ...},
                "special_tokens": {"bos": str, "eos": str, ...},
                "token_handling": {"add_bos": bool, "add_eos": bool},
                "fine_tuning": {"max_length": int, "padding_side": str, ...}
            }
        """
        return {
            "reasoning": self.detect_reasoning(),
            "tools": self.detect_tools(),
            "special_tokens": self.tokenizer_config.special_tokens,
            "token_handling": {
                "add_bos_token": self.tokenizer_config.add_bos_token,
                "add_eos_token": self.tokenizer_config.add_eos_token,
            },
            "fine_tuning": {
                "model_max_length": self.tokenizer_config.model_max_length,
                "padding_side": self.tokenizer_config.padding_side,
                "tokenizer_class": self.tokenizer_config.tokenizer_class,
                "clean_up_tokenization_spaces": self.tokenizer_config.clean_up_tokenization_spaces,
            },
            "has_chat_template": self.tokenizer_config.chat_template is not None,
        }

    def detect_reasoning(self) -> dict[str, Any]:
        """
        Detect reasoning capability.

        Returns:
            Dictionary with reasoning configuration:
            {
                "native_support": bool,
                "start_tag": str | None,
                "end_tag": str | None,
                "detected_tags": list[str]
            }
        """
        # Get potential reasoning token indicators
        indicators = self._get_reasoning_indicators()

        # Check which indicators exist in tokenizer
        detected = [token for token in indicators if self.tokenizer_config.has_token(token)]

        if not detected:
            return {
                "native_support": False,
                "start_tag": None,
                "end_tag": None,
                "detected_tags": [],
                "inject_mode": "inline",
            }

        # Use the first detected tag pair
        primary_tag = detected[0]

        # Infer end tag (most follow <tag> / </tag> pattern)
        if primary_tag.startswith("<") and not primary_tag.startswith("</"):
            end_tag = primary_tag.replace("<", "</", 1)
        else:
            end_tag = None

        # Verify end tag exists
        if end_tag and not self.tokenizer_config.has_token(end_tag):
            end_tag = None

        return {
            "native_support": True,
            "start_tag": primary_tag,
            "end_tag": end_tag,
            "detected_tags": detected,
            "inject_mode": "native",
        }

    def detect_tools(self) -> dict[str, Any]:
        """
        Detect tool calling capability.

        Returns:
            Dictionary with tool configuration:
            {
                "native_support": bool,
                "format": str,  # "xml", "native", or "unknown"
                "start_tag": str | None,
                "end_tag": str | None,
                "detected_tags": list[str]
            }
        """
        # Get potential tool calling token indicators
        indicators = self._get_tool_indicators()

        # Check which indicators exist in tokenizer
        detected = [token for token in indicators if self.tokenizer_config.has_token(token)]

        if not detected:
            # No XML tags, assume native tool_calls format
            return {
                "native_support": False,  # Can't confirm without testing
                "format": "native",
                "start_tag": None,
                "end_tag": None,
                "detected_tags": [],
            }

        # Determine format based on detected tags
        primary_tag = detected[0]
        format_type = "xml"

        # Infer end tag
        if primary_tag.startswith("<") and not primary_tag.startswith("</"):
            end_tag = primary_tag.replace("<", "</", 1)
        else:
            end_tag = None

        # Verify end tag exists
        if end_tag and not self.tokenizer_config.has_token(end_tag):
            end_tag = None

        return {
            "native_support": True,
            "format": format_type,
            "start_tag": primary_tag,
            "end_tag": end_tag,
            "detected_tags": detected,
        }

    def _get_reasoning_indicators(self) -> list[str]:
        """
        Get list of potential reasoning token indicators.

        Returns:
            List of token strings to check
        """
        # Built-in indicators
        default_indicators = [
            "<think>",
            "<thinking>",
            "<thought>",
            "<reasoning>",
            "<musing>",
            "<contemplation>",
            "<scratch>",
            "<scratchpad>",
        ]

        # Add indicators from model mappings if available
        custom_indicators = []
        if self.model_mappings:
            detection = self.model_mappings.get("capability_detection", {})
            reasoning_config = detection.get("reasoning", {})
            custom_indicators = reasoning_config.get("token_indicators", [])

        return custom_indicators + default_indicators

    def _get_tool_indicators(self) -> list[str]:
        """
        Get list of potential tool calling token indicators.

        Returns:
            List of token strings to check
        """
        # Built-in indicators
        default_indicators = [
            "<tool_call>",
            "<tool_response>",
            "<function_call>",
            "<function>",
            "<invoke>",
            "<call>",
        ]

        # Add indicators from model mappings if available
        custom_indicators = []
        if self.model_mappings:
            detection = self.model_mappings.get("capability_detection", {})
            tool_config = detection.get("tools", {})
            custom_indicators = tool_config.get("token_indicators", [])

        return custom_indicators + default_indicators


def detect_capabilities(model_id: str, model_mappings: dict | None = None) -> dict[str, Any]:
    """
    Convenience function to detect all capabilities for a model.

    Args:
        model_id: HuggingFace model ID
        model_mappings: Optional model mappings configuration

    Returns:
        Dictionary with detected capabilities
    """
    tokenizer_config = TokenizerConfig.from_model_id(model_id)
    detector = CapabilityDetector(tokenizer_config, model_mappings)
    return detector.detect_all()
