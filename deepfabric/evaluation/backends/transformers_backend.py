import json
import logging
import re

from contextlib import suppress
from typing import Any

import torch

from pydantic import BaseModel, ValidationError, field_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...schemas import ToolDefinition
from ..inference import InferenceBackend, InferenceConfig, ModelResponse

logger = logging.getLogger(__name__)


def _extract_json_object(text: str, start_pos: int = 0) -> str | None:
    """Extract a complete JSON object from text, handling nested braces.

    Args:
        text: Text containing JSON
        start_pos: Starting position to search from

    Returns:
        Extracted JSON string or None if not found
    """
    # Find first opening brace
    start = text.find("{", start_pos)
    if start == -1:
        return None

    # Track brace depth to find matching closing brace
    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        char = text[i]

        # Handle string escaping
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue

        # Track if we're inside a string
        if char == '"':
            in_string = not in_string
            continue

        # Only count braces outside of strings
        if not in_string:
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    # Found matching closing brace
                    return text[start : i + 1]

    return None


class ToolCall(BaseModel):
    """Parsed tool call in OpenAI format."""

    name: str
    arguments: dict[str, Any] | str

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments_string(cls, v: Any) -> dict[str, Any]:
        """Parse arguments if they're a JSON string.

        Raises:
            ValueError: If arguments is a string but not valid JSON
        """
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in arguments field: {e}") from e
        return v


class TransformersBackend(InferenceBackend):
    """Inference backend using HuggingFace Transformers."""

    def __init__(self, config: InferenceConfig):
        """Initialize Transformers backend.

        Args:
            config: Inference configuration
        """
        super().__init__(config)

        # Determine device
        if config.device:
            self.device = config.device
        # Auto-detect best available device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        # Determine dtype based on device
        if self.device == "cuda":
            dtype = torch.float16
            device_map = "auto"
        elif self.device == "mps":
            dtype = torch.float32  # MPS works best with float32
            device_map = None
        else:
            dtype = torch.float32
            device_map = None

        self.loaded_with_unsloth = False
        # Load with Unsloth if requested and adapter is provided
        if config.use_unsloth and config.adapter_path:
            try:
                from unsloth import FastLanguageModel  # type: ignore # noqa: PLC0415

                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.adapter_path,
                    max_seq_length=config.max_seq_length,
                    dtype=dtype,
                    load_in_4bit=config.load_in_4bit,
                )
                FastLanguageModel.for_inference(self.model)
                self.loaded_with_unsloth = True
            except ImportError:
                logger.warning("Unsloth not installed, falling back to standard PEFT")
            except Exception as e:
                logger.warning("Unsloth loading failed (%s), falling back to standard PEFT", e)

        # Standard transformers/PEFT loading
        if not self.loaded_with_unsloth:
            self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)  #  nosec

            self.model = AutoModelForCausalLM.from_pretrained(  # nosec
                config.model_path,
                device_map=device_map,
                dtype=dtype,
            )

            # Load PEFT adapter if provided
            if config.adapter_path:
                from peft import PeftModel  # noqa: PLC0415

                self.model = PeftModel.from_pretrained(self.model, config.adapter_path)

            # Move to device if not using device_map
            if self.device in ("cpu", "mps"):
                self.model.to(self.device)  # type: ignore[arg-type]

            # Enable optimizations for faster inference
            # Compile model for better performance (PyTorch 2.0+)
            with suppress(Exception):
                # Use reduce-overhead mode for better latency on smaller batches
                self.model = torch.compile(self.model, mode="reduce-overhead")  # type: ignore[assignment]

        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

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
        # Format messages using chat template
        prompt = self._format_prompt(messages, tools)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate with optimizations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Performance optimizations
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding (faster than beam search)
            )

        # Decode output
        generated_ids = outputs[0][inputs.input_ids.shape[1] :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse tool call if present
        tool_call = self._parse_tool_call(generated_text) if tools else None

        return ModelResponse(
            content=generated_text,
            tool_call=tool_call,
            raw_output=generated_text,
            finish_reason="stop",
        )

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
        # Format all prompts
        prompts = [self._format_prompt(msgs, tools) for msgs in batch_messages]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        # Generate batch with optimizations
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Performance optimizations
                use_cache=True,  # Enable KV cache for faster generation
                num_beams=1,  # Greedy decoding (faster than beam search)
            )

        # Decode outputs
        responses = []
        for i, output_ids in enumerate(outputs):
            # Extract generated portion (skip input tokens)
            generated_ids = output_ids[inputs.input_ids[i].shape[0] :]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse tool call if present
            tool_call = self._parse_tool_call(generated_text) if tools else None

            responses.append(
                ModelResponse(
                    content=generated_text,
                    tool_call=tool_call,
                    raw_output=generated_text,
                    finish_reason="stop",
                )
            )

        return responses

    def cleanup(self) -> None:
        """Clean up GPU memory."""
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _format_prompt(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolDefinition] | None = None,
    ) -> str:
        """Format messages into a prompt string.

        Args:
            messages: List of message dicts
            tools: Optional list of tools

        Returns:
            Formatted prompt string
        """
        # Try to use chat template with tools support (modern approach)
        if hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template:
            try:
                # Convert tools to OpenAI format for chat template compatibility
                tools_param = None
                if tools:
                    tools_param = [tool.to_openai() for tool in tools]

                # Try with tools parameter (for models with native tool support)
                return self.tokenizer.apply_chat_template(
                    messages,
                    tools=tools_param,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except (TypeError, KeyError):
                # Model's chat template doesn't support tools parameter
                # Try without tools parameter
                try:
                    return self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:  # noqa: S110
                    # Fallback to manual formatting
                    pass  # nosec

        # Manual formatting fallback (for models without chat templates)
        prompt_parts = []

        # Add tools if present
        if tools:
            tools_str = "Available tools:\n"
            for tool in tools:
                tools_str += f"- {tool.name}: {tool.description}\n"
                params_list = [p.model_dump() for p in tool.parameters]
                tools_str += f"  Parameters: {json.dumps(params_list)}\n"
            prompt_parts.append(tools_str)

        # Add messages
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"{role.upper()}: {content}")

        prompt_parts.append("ASSISTANT:")
        return "\n\n".join(prompt_parts)

    def _parse_tool_call(self, text: str) -> dict | None:
        """Parse tool call from generated text.

        Supports OpenAI-standard tool call formats:
        - JSON: {"name": "func", "arguments": {...}}
        - XML: <tool_call>{"name": "func", "arguments": {...}}</tool_call>

        Args:
            text: Generated text

        Returns:
            Dict with 'name' and 'arguments' if tool call found, None otherwise
        """
        # Try XML format first (common in chat templates)
        xml_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if xml_match:
            try:
                data = json.loads(xml_match.group(1).strip())
                tool_call = ToolCall.model_validate(data)
                return tool_call.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass  # Expected parsing or validation error

        # Try JSON format - extract complete JSON object with proper nesting
        json_str = _extract_json_object(text)
        if json_str:
            try:
                data = json.loads(json_str)
                # Verify it has required fields before validating
                if "name" in data and "arguments" in data:
                    tool_call = ToolCall.model_validate(data)
                    return tool_call.model_dump()
            except (json.JSONDecodeError, ValidationError):
                pass  # Expected parsing or validation error

        return None
