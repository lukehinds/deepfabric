"""
HuggingFace Chat Template Formatter.

Universal formatter that uses model-specific chat templates from HuggingFace.
"""

import json
import logging

from pathlib import Path
from typing import Any

from ..schemas import Conversation, ToolExecution
from .capability_detection import CapabilityDetector, TokenizerConfig
from .model_mappings import ModelMappingLoader

logger = logging.getLogger(__name__)


class HFChatTemplateFormatter:
    """
    Universal formatter using HuggingFace chat templates.

    Automatically detects model capabilities and applies correct formatting.
    Supports two modes:
    - Full mode: Uses transformers library (recommended)
    - Fallback mode: Manual Jinja2 rendering (lightweight)
    """

    def __init__(
        self,
        model_id: str,
        model_config_path: str | Path | None = None,
        use_transformers: bool = True,
    ):
        """
        Initialize the formatter.

        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            model_config_path: Optional path to model mappings YAML
            use_transformers: Whether to use transformers library (fallback to manual if False)
        """
        self.model_id = model_id
        self.use_transformers = use_transformers

        # Load model mappings
        self.mapping_loader = ModelMappingLoader(model_config_path)
        self.model_config = self.mapping_loader.resolve(model_id)

        # Load tokenizer config and detect capabilities
        self.tokenizer_config = TokenizerConfig.from_model_id(model_id)
        detector = CapabilityDetector(self.tokenizer_config, self.mapping_loader.config)
        self.capabilities = detector.detect_all()

        # Merge detected capabilities with model config (model config takes priority)
        self._merge_capabilities()

        # Initialize tokenizer (transformers or manual)
        self.tokenizer = None
        self.template = None
        self._init_tokenizer()

        logger.info(f"Initialized HF formatter for {model_id}")
        logger.debug(f"Capabilities: {self.capabilities}")
        logger.debug(f"Model config: {self.model_config}")

    def _merge_capabilities(self):
        """
        Merge detected capabilities with model config.

        Strategy:
        - Start with detected capabilities (from tokenizer analysis)
        - Overlay model config, but only for keys that differ from defaults
        - This allows capability detection to work, while custom configs can override
        """
        # Get defaults to identify which config values are custom vs default
        defaults = self.mapping_loader.defaults

        # Reasoning
        detected_reasoning = self.capabilities["reasoning"]
        config_reasoning = self.model_config.get("reasoning", {})
        default_reasoning = defaults.get("reasoning", {})

        # Merge: detected + (config - defaults)
        merged_reasoning = detected_reasoning.copy()
        for key, value in config_reasoning.items():
            # Only apply config value if it differs from default (i.e., explicitly set)
            if key not in default_reasoning or value != default_reasoning[key]:
                merged_reasoning[key] = value

        self.capabilities["reasoning"] = merged_reasoning
        self.model_config["reasoning"] = merged_reasoning

        # Tools
        detected_tools = self.capabilities["tools"]
        config_tools = self.model_config.get("tools", {})
        default_tools = defaults.get("tools", {})

        merged_tools = detected_tools.copy()
        for key, value in config_tools.items():
            if key not in default_tools or value != default_tools[key]:
                merged_tools[key] = value

        self.capabilities["tools"] = merged_tools
        self.model_config["tools"] = merged_tools

    def _init_tokenizer(self):
        """Initialize tokenizer (transformers or manual mode)."""
        if self.use_transformers:
            try:
                from transformers import AutoTokenizer  # noqa: PLC0415

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)  #  nosec
            except ImportError:
                logger.warning("transformers not installed, falling back to manual mode")
                self.use_transformers = False
            except Exception as e:
                logger.warning(
                    f"Failed to load tokenizer via transformers: {e}, falling back to manual mode"
                )
                self.use_transformers = False
            else:
                logger.info(f"Loaded tokenizer via transformers for {self.model_id}")
                return

        # Fallback: manual Jinja2 mode
        try:
            from jinja2 import Template  # noqa: PLC0415

            if not self.tokenizer_config.chat_template:
                raise ValueError(f"No chat template found for {self.model_id}")

            self.template = Template(self.tokenizer_config.chat_template)
            logger.info(f"Using manual Jinja2 mode for {self.model_id}")
        except ImportError as e:
            raise ImportError(
                "Neither transformers nor jinja2 is installed. "
                "Install with: pip install transformers  or  pip install jinja2"
            ) from e

    def format(self, conversation: Conversation, **kwargs) -> str:
        """
        Format a conversation using the model's chat template.

        Args:
            conversation: DeepFabric Conversation object
            **kwargs: Additional arguments for apply_chat_template

        Returns:
            Formatted string ready for training

        Raises:
            ValueError: If conversation cannot be formatted
        """
        # Stage 1: Convert to standard messages format
        standard_format = self._convert_to_standard(conversation)

        # Stage 2: Apply chat template
        return self._apply_template(
            standard_format["messages"], tools=standard_format.get("tools"), **kwargs
        )

    def _convert_to_standard(self, conversation: Conversation) -> dict[str, Any]:
        """
        Convert DeepFabric Conversation to standard HF message format.

        Args:
            conversation: DeepFabric Conversation object

        Returns:
            Dictionary with "messages" and optionally "tools"
        """
        messages = []
        tools = None

        # Extract tools if present
        if conversation.tool_context and conversation.tool_context.available_tools:
            tools = [tool.to_openai() for tool in conversation.tool_context.available_tools]

        # Process messages
        for msg in conversation.messages:
            if msg.role in ["system", "user"]:
                messages.append({"role": msg.role, "content": msg.content})

        # Handle tool calling
        if conversation.tool_context and conversation.tool_context.executions:
            tool_messages = self._format_tool_calls(conversation.tool_context.executions)
            messages.extend(tool_messages)

        # Handle final assistant message with reasoning
        final_assistant = self._build_final_assistant_message(conversation)
        if final_assistant:
            messages.append(final_assistant)

        return {"messages": messages, "tools": tools}

    def _format_tool_calls(self, executions: list[ToolExecution]) -> list[dict]:
        """
        Format tool executions as messages.

        Args:
            executions: List of ToolExecution objects

        Returns:
            List of message dictionaries
        """
        messages = []
        tool_config = self.model_config.get("tools", {})
        tool_format = tool_config.get("format", "native")

        if tool_format == "native":
            # Native tool_calls format
            tool_calls = []
            for execution in executions:
                try:
                    arguments = json.loads(execution.arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse tool arguments: {execution.arguments}")
                    continue

                tool_calls.append(
                    {
                        "type": "function",
                        "function": {"name": execution.function_name, "arguments": arguments},
                    }
                )

            if tool_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

            # Add tool responses
            for execution in executions:
                messages.append({"role": "tool", "content": execution.result})

        elif tool_format == "xml":
            # XML format (e.g., Qwen)
            # The chat template will handle wrapping in <tool_call> tags
            # We just provide the native format and let the template handle it
            tool_calls = []
            for execution in executions:
                try:
                    arguments = json.loads(execution.arguments)
                except json.JSONDecodeError:
                    continue

                tool_calls.append(
                    {
                        "type": "function",
                        "function": {"name": execution.function_name, "arguments": arguments},
                    }
                )

            if tool_calls:
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

            # Add tool responses
            for execution in executions:
                messages.append({"role": "tool", "content": execution.result})

        return messages

    def _build_final_assistant_message(self, conversation: Conversation) -> dict | None:
        """
        Build final assistant message with reasoning if present.

        Args:
            conversation: DeepFabric Conversation object

        Returns:
            Message dictionary or None
        """
        reasoning_config = self.model_config.get("reasoning", {})
        inject_mode = reasoning_config.get("inject_mode", "inline")
        native_support = reasoning_config.get("native_support", False)

        # Get final answer content
        if conversation.final_answer:
            content = conversation.final_answer
        else:
            # Use last assistant message
            assistant_messages = [msg for msg in conversation.messages if msg.role == "assistant"]
            if assistant_messages:
                content = assistant_messages[-1].content
            else:
                return None

        # Build message dictionary
        message = {"role": "assistant", "content": content}

        # Handle reasoning
        if conversation.reasoning:
            reasoning_text = self._format_reasoning(conversation.reasoning)

            if native_support and inject_mode == "native":
                # Pass reasoning as separate field for chat template to handle
                # (e.g., Qwen Thinking models use reasoning_content field)
                message["reasoning_content"] = reasoning_text
            elif inject_mode == "native":
                # Inject tags manually into content
                start_tag = reasoning_config.get("start_tag", "<think>")
                end_tag = reasoning_config.get("end_tag", "</think>")
                message["content"] = f"{start_tag}\n{reasoning_text}\n{end_tag}\n\n{content}"
            else:
                # Inline mode - prepend reasoning to content
                prefix = reasoning_config.get("prefix", "")
                separator = reasoning_config.get("separator", "\n\n")
                if prefix:
                    message["content"] = f"{prefix}{reasoning_text}{separator}{content}"
                else:
                    message["content"] = f"{reasoning_text}{separator}{content}"

        return message

    def _format_reasoning(self, reasoning) -> str:
        """
        Format reasoning trace to text.

        Args:
            reasoning: ReasoningTrace object

        Returns:
            Formatted reasoning text
        """
        if reasoning.style == "freetext":
            return reasoning.content if isinstance(reasoning.content, str) else ""

        if reasoning.style in ("structured", "hybrid") and isinstance(reasoning.content, list):
            # Format structured steps
            parts = []
            for step in reasoning.content:
                if hasattr(step, "thought"):
                    thought = step.thought
                    action = getattr(step, "action", None)
                    if action:
                        parts.append(f"{thought} → {action}")
                    else:
                        parts.append(thought)
            return " ".join(parts)

        return ""

    def _apply_template(
        self, messages: list[dict], tools: list[dict] | None = None, **kwargs
    ) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: List of message dictionaries
            tools: Optional list of tool definitions
            **kwargs: Additional arguments

        Returns:
            Formatted string
        """
        if self.use_transformers and self.tokenizer:
            # Use transformers library
            try:
                result = self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=kwargs.get("add_generation_prompt", False),
                    **kwargs,
                )
            except Exception:
                logger.exception("Failed to apply chat template via transformers")
                raise
            else:
                return result

        elif self.template:
            try:
                result = self.template.render(
                    messages=messages,
                    tools=tools,
                    add_generation_prompt=kwargs.get("add_generation_prompt", False),
                    bos_token=self.tokenizer_config.bos_token or "",
                    eos_token=self.tokenizer_config.eos_token or "",
                )
            except Exception:
                logger.exception("Failed to render Jinja2 template")
                raise
            else:
                return result
        else:
            raise RuntimeError("No tokenizer or template available")

    def get_capabilities(self) -> dict[str, Any]:
        """
        Get detected capabilities for this model.

        Returns:
            Dictionary with all detected capabilities
        """
        return self.capabilities.copy()

    def get_fine_tuning_metadata(self) -> dict[str, Any]:
        """
        Get metadata useful for fine-tuning.

        Returns:
            Dictionary with fine-tuning-relevant information
        """
        return {
            "model_id": self.model_id,
            "model_max_length": self.capabilities["fine_tuning"]["model_max_length"],
            "padding_side": self.capabilities["fine_tuning"]["padding_side"],
            "tokenizer_class": self.capabilities["fine_tuning"]["tokenizer_class"],
            "special_tokens": self.capabilities["special_tokens"],
            "has_reasoning_support": self.capabilities["reasoning"]["native_support"],
            "has_tool_support": self.capabilities["tools"]["native_support"],
        }
