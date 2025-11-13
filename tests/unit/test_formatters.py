"""
Tests for the DeepFabric formatter system.

This module tests:
- BaseFormatter interface
- FormatterRegistry functionality
- Built-in formatters (Alpaca, ChatML, Harmony)
- Dataset integration
- Error handling
"""

import os
import tempfile

from typing import Any

import pytest

from deepfabric.dataset import Dataset
from deepfabric.formatters.base import BaseFormatter, FormatterError
from deepfabric.formatters.builtin.alpaca import AlpacaFormatter
from deepfabric.formatters.builtin.chatml import ChatmlFormatter
from deepfabric.formatters.builtin.harmony import HarmonyFormatter
from deepfabric.formatters.builtin.openai import (
    OpenAISchemaConfig,
    OpenAISchemaFormatter,
)
from deepfabric.formatters.builtin.tool_calling import ToolCallingFormatter
from deepfabric.formatters.registry import FormatterRegistry
from deepfabric.schemas import (
    ChatMessage,
    Conversation,
    ReasoningTrace,
    ToolContext,
    ToolDefinition,
    ToolExecution,
    ToolParameter,
    ToolRegistry,
)

FORMATTED_LENGTH = 3


class TestBaseFormatter:
    """Test the BaseFormatter abstract interface."""

    def test_base_formatter_is_abstract(self):
        """Test that BaseFormatter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseFormatter()  # type: ignore

    def test_base_formatter_methods_exist(self):
        """Test that BaseFormatter has required abstract methods."""
        assert hasattr(BaseFormatter, "format")
        assert hasattr(BaseFormatter, "validate")
        assert hasattr(BaseFormatter, "get_description")
        assert hasattr(BaseFormatter, "get_supported_formats")


class MockFormatter(BaseFormatter):
    """Mock formatter for testing purposes."""

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.format_called = False
        self.validate_called = False

    def format(self, dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self.format_called = True
        return [{"formatted": True, "original": sample} for sample in dataset]

    def validate(self, entry: dict[str, Any]) -> bool:
        self.validate_called = True
        return "test_field" in entry

    def get_description(self) -> str:
        return "Mock formatter for testing"

    def get_supported_formats(self) -> list[str]:
        return ["test_format"]


class TestFormatterRegistry:
    """Test the FormatterRegistry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = FormatterRegistry()

    def test_load_builtin_alpaca_formatter(self):
        """Test loading the built-in Alpaca formatter."""
        formatter = self.registry.load_formatter("builtin://alpaca.py")
        assert isinstance(formatter, AlpacaFormatter)

    def test_load_builtin_chatml_formatter(self):
        """Test loading the built-in ChatML formatter."""
        formatter = self.registry.load_formatter("builtin://chatml.py")
        assert isinstance(formatter, ChatmlFormatter)

    def test_load_nonexistent_builtin_formatter(self):
        """Test loading a non-existent built-in formatter."""
        with pytest.raises(FormatterError, match="Built-in formatter 'nonexistent' not found"):
            self.registry.load_formatter("builtin://nonexistent.py")

    def test_load_custom_formatter_from_file(self):
        """Test loading a custom formatter from file."""
        # Create a temporary formatter file
        formatter_code = """
from deepfabric.formatters.base import BaseFormatter

class CustomFormatter(BaseFormatter):
    def _format_single_sample(self, sample):
        return {"custom": True}
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(formatter_code)
            temp_path = f.name

        try:
            formatter = self.registry.load_formatter(f"file://{temp_path}")
            assert isinstance(formatter, BaseFormatter)

            # Test the formatter works
            result = formatter.format([{"test": "data"}])
            assert len(result.samples) == 1
            assert getattr(result.samples[0], "custom") is True  # noqa: B009
        finally:
            os.unlink(temp_path)

    def test_load_invalid_file_path(self):
        """Test loading formatter from non-existent file."""
        with pytest.raises(FormatterError, match="Formatter file not found"):
            self.registry.load_formatter("file://./nonexistent.py")

    def test_load_invalid_template_format(self):
        """Test loading with invalid template format."""
        with pytest.raises(FormatterError, match="Invalid template format"):
            self.registry.load_formatter("invalid://template")

    def test_formatter_caching(self):
        """Test that formatters are cached."""
        formatter1 = self.registry.load_formatter("builtin://alpaca.py")
        formatter2 = self.registry.load_formatter("builtin://alpaca.py")

        # Should be different instances but same class
        assert isinstance(formatter1, type(formatter2))
        assert formatter1 is not formatter2

    def test_clear_cache(self):
        """Test clearing the formatter cache."""
        self.registry.load_formatter("builtin://alpaca.py")
        assert len(self.registry._cache) > 0

        self.registry.clear_cache()
        assert len(self.registry._cache) == 0

    def test_list_builtin_formatters(self):
        """Test listing available built-in formatters."""
        formatters = self.registry.list_builtin_formatters()
        assert isinstance(formatters, list)
        assert "alpaca" in formatters
        assert "chatml" in formatters

    def test_formatter_with_config(self):
        """Test loading openai formatter with custom configuration."""
        config = {
            "include_system_prompt": False,
            "validate_tool_schemas": False,
            "remove_available_tools_field": True,
            "parallel_tool_calls": False,
        }
        formatter = self.registry.load_formatter("builtin://openai.py", config)
        assert isinstance(formatter, OpenAISchemaFormatter)

        # Verify config was loaded correctly
        assert isinstance(formatter._config_model, OpenAISchemaConfig)
        assert formatter._config_model.include_system_prompt is False
        assert formatter._config_model.validate_tool_schemas is False
        assert formatter._config_model.remove_available_tools_field is True
        assert formatter._config_model.parallel_tool_calls is False


class TestAlpacaFormatter:
    """Test the Alpaca formatter specifically."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = AlpacaFormatter()

    def test_format_messages_sample(self):
        """Test formatting a messages sample to Alpaca format."""
        sample = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Explain photosynthesis."},
                {"role": "assistant", "content": "Photosynthesis is the process..."},
            ]
        }

        result = self.formatter.format([sample])
        assert len(result) == 1

        formatted = result[0]
        assert hasattr(formatted, "instruction")
        assert hasattr(formatted, "output")
        assert formatted.instruction == "You are helpful."
        assert formatted.output == "Photosynthesis is the process..."

    def test_format_qa_sample(self):
        """Test formatting a Q&A sample to Alpaca format."""
        sample = {"question": "What is the capital of France?", "answer": "Paris"}

        result = self.formatter.format([sample])
        assert len(result) == 1

        formatted = result[0]
        assert formatted.instruction == "What is the capital of France?"
        assert formatted.output == "Paris"

    def test_custom_instruction_template(self):
        """Test Alpaca formatter with custom instruction template."""
        config = {"instruction_template": "Task: {instruction}"}
        formatter = AlpacaFormatter(config)

        sample = {"instruction": "Solve this", "output": "Answer"}
        result = formatter.format([sample])

        assert result[0].instruction == "Task: Solve this"

    def test_include_empty_input(self):
        """Test include_empty_input configuration."""
        # Test with include_empty_input=True (default)
        sample = {"instruction": "Test", "output": "Answer"}
        result = self.formatter.format([sample])
        assert hasattr(result[0], "input")

        # Test with include_empty_input=False
        formatter = AlpacaFormatter({"include_empty_input": False})
        result = formatter.format([sample])
        assert not hasattr(result[0], "input")

    def test_validate_output(self):
        """Test output validation."""
        valid_output = {
            "instruction": "Test instruction",
            "output": "Test output",
            "input": "Test input",
        }
        assert self.formatter.validate_output(valid_output)

        invalid_output = {"instruction": "Test"}  # Missing output
        assert not self.formatter.validate_output(invalid_output)


class TestChatmlFormatter:
    """Test the ChatML formatter specifically."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ChatmlFormatter()

    def test_structured_output_format(self):
        """Test ChatML formatter with structured output."""
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        result = self.formatter.format([sample])
        assert len(result) == 1

        formatted = result[0]
        assert hasattr(formatted, "messages")
        assert len(formatted.messages) == 2  # noqa: PLR2004

    def test_text_output_format(self):
        """Test ChatML formatter with text output."""
        config = {"output_format": "text"}
        formatter = ChatmlFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        result = formatter.format([sample])
        assert len(result) == 1

        formatted = result[0]
        assert hasattr(formatted, "text")
        assert "<|im_start|>" in formatted.text
        assert "<|im_end|>" in formatted.text

    def test_custom_tokens(self):
        """Test ChatML formatter with custom tokens."""
        config = {"start_token": "<start>", "end_token": "<end>", "output_format": "text"}
        formatter = ChatmlFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        }
        result = formatter.format([sample])

        text = result[0].text
        assert "<start>" in text
        assert "<end>" in text

    def test_require_system_message(self):
        """Test requiring system message."""
        config = {"require_system_message": True}
        formatter = ChatmlFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        result = formatter.format([sample])
        messages = result[0].messages

        # Should have system message added
        assert any(msg["role"] == "system" for msg in messages)

    def test_validate_output_structured(self):
        """Test output validation for structured format."""
        valid_output = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        assert self.formatter.validate_output(valid_output)

    def test_validate_output_text(self):
        """Test output validation for text format."""
        formatter = ChatmlFormatter({"output_format": "text"})

        valid_output = {"text": "<|im_start|>user\nHello\n<|im_end|>"}
        assert formatter.validate_output(valid_output)

        invalid_output = {"text": "No ChatML tokens"}
        assert not formatter.validate_output(invalid_output)


class TestDatasetIntegration:
    """Test formatter integration with Dataset class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dataset = Dataset()
        self.test_samples = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "The answer is 4."},
                ]
            },
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]
        self.dataset.samples = self.test_samples

    def test_apply_single_formatter(self):
        """Test applying a single formatter to dataset."""
        formatter_configs = [{"name": "alpaca", "template": "builtin://alpaca.py", "config": {}}]

        result = self.dataset.apply_formatters(formatter_configs)

        assert "alpaca" in result
        assert isinstance(result["alpaca"], Dataset)
        assert len(result["alpaca"].samples) == 2  # noqa: PLR2004

    def test_apply_multiple_formatters(self):
        """Test applying multiple formatters to dataset."""
        formatter_configs = [
            {"name": "chatml", "template": "builtin://chatml.py", "config": {}},
            {"name": "alpaca", "template": "builtin://alpaca.py", "config": {}},
        ]

        result = self.dataset.apply_formatters(formatter_configs)

        assert "chatml" in result
        assert "alpaca" in result
        assert isinstance(result["chatml"], Dataset)
        assert isinstance(result["alpaca"], Dataset)

    def test_apply_formatter_with_output_file(self):
        """Test applying formatter with output file specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output.jsonl")

            formatter_configs = [
                {
                    "name": "alpaca",
                    "template": "builtin://alpaca.py",
                    "config": {},
                    "output": output_path,
                }
            ]

            result = self.dataset.apply_formatters(formatter_configs)

            # Check file was created
            assert os.path.exists(output_path)

            # Check dataset was returned
            assert "alpaca" in result

    def test_list_available_formatters(self):
        """Test listing available formatters."""
        formatters = self.dataset.list_available_formatters()
        assert isinstance(formatters, list)
        assert len(formatters) > 0

    def test_formatter_error_handling(self):
        """Test error handling in formatter application."""
        formatter_configs = [
            {"name": "invalid", "template": "builtin://nonexistent.py", "config": {}}
        ]

        with pytest.raises(FormatterError):
            self.dataset.apply_formatters(formatter_configs)


class TestErrorHandling:
    """Test error handling across the formatter system."""

    def test_formatter_error_creation(self):
        """Test FormatterError creation and inheritance."""
        error = FormatterError("Test message")
        assert str(error) == "Test message"
        assert isinstance(error, Exception)

    def test_invalid_formatter_class(self):
        """Test loading file with invalid formatter class."""
        invalid_code = """
class NotAFormatter:
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(invalid_code)
            temp_path = f.name

        try:
            registry = FormatterRegistry()
            with pytest.raises(FormatterError, match="No BaseFormatter subclass found"):
                registry.load_formatter(f"file://{temp_path}")
        finally:
            os.unlink(temp_path)

    def test_formatter_instantiation_error(self):
        """Test error when formatter instantiation fails."""
        # Create a formatter that fails during __init__
        failing_code = """
from deepfabric.formatters.base import BaseFormatter

class FailingFormatter(BaseFormatter):
    def __init__(self, config=None):
        raise ValueError("Initialization failed")

    def format(self, dataset):
        return dataset
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(failing_code)
            temp_path = f.name

        try:
            registry = FormatterRegistry()
            with pytest.raises(FormatterError, match="Failed to instantiate formatter"):
                registry.load_formatter(f"file://{temp_path}")
        finally:
            os.unlink(temp_path)


class TestToolCallingFormatter:
    """Test the ToolCallingFormatter for embedded tool execution traces."""

    def setUp(self):
        """Set up test fixtures."""
        self.formatter = ToolCallingFormatter()

    def test_tool_calling_formatter_basic(self):
        """Test basic tool calling formatter functionality."""

        formatter = ToolCallingFormatter()

        # Unified Conversation schema sample with reasoning and tool context
        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's the weather in London?"),
                ChatMessage(
                    role="assistant",
                    content="The weather in London is currently 18°C with cloudy skies and light rain.",
                ),
            ],
            reasoning=ReasoningTrace(
                style="freetext",
                content="I need to check the current weather for London using the weather tool. Check weather for London. Weather tool is appropriate for this query. Use London, UK as location parameter. Tool returned current weather conditions.",
            ),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="get_weather",
                        arguments='{"location": "London, UK"}',
                        reasoning="Need current weather data for London",
                        result="18°C, cloudy with light rain",
                    )
                ],
            ),
            question="What's the weather in London?",
            final_answer="The weather in London is currently 18°C with cloudy skies and light rain.",
        )

        result = formatter.format_with_metadata([sample])
        assert len(result.samples) == 1

        formatted = result.samples[0]
        assert "messages" in formatted

        messages = formatted["messages"]
        assert (
            len(messages) == 4  # noqa: PLR2004
        )  # user, assistant (thinking+tool), tool, assistant (answer)  # noqa: PLR2004

        # Check message roles
        roles = [msg["role"] for msg in messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

        # Check user message
        assert messages[0]["content"] == sample.question

        # Check assistant message with thinking and tool call
        assistant_msg = messages[1]["content"]
        assert "<think>" in assistant_msg
        assert "</think>" in assistant_msg
        assert "<tool_call>" in assistant_msg
        assert "</tool_call>" in assistant_msg

        # Check tool response
        tool_msg = messages[2]["content"]
        assert "<tool_response>" in tool_msg
        assert "</tool_response>" in tool_msg
        assert "18°C, cloudy with light rain" in tool_msg

        # Check final answer
        assert "18°C with cloudy skies and light rain" in messages[3]["content"]

    def test_tool_calling_formatter_rich_reasoning(self):
        """Test tool calling formatter with rich CoT reasoning."""

        formatter = ToolCallingFormatter()

        # Unified Conversation with structured reasoning
        reasoning_content = "The user wants me to find the square root of 144, which is a mathematical calculation. I need to perform a mathematical operation (square root). The calculator tool is the appropriate choice for this task. I'll use the sqrt operation with value 144."

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Calculate the square root of 144"),
                ChatMessage(role="assistant", content="The square root of 144 is 12."),
            ],
            reasoning=ReasoningTrace(style="freetext", content=reasoning_content),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="calculator",
                        arguments='{"operation": "sqrt", "value": 144}',
                        reasoning="Calculate square root of 144",
                        result="12",
                    )
                ],
            ),
            question="Calculate the square root of 144",
            final_answer="The square root of 144 is 12.",
        )

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        messages = formatted["messages"]

        # Check that reasoning is included in thinking
        assistant_msg = messages[1]["content"]
        assert "<think>" in assistant_msg
        assert "mathematical calculation" in assistant_msg
        assert "square root" in assistant_msg

    def test_tool_calling_formatter_with_tools_in_system(self):
        """Test tool calling formatter with tools included in system message."""

        config = {
            "include_tools_in_system": True,
            "system_prompt": "You are a helpful AI assistant with access to tools.",
        }

        # Create tool registry with calculator tool
        calculator_tool = ToolDefinition(
            name="calculator",
            description="Perform mathematical calculations",
            parameters=[
                ToolParameter(
                    name="operation", type="str", description="Math operation", required=True
                ),
                ToolParameter(name="a", type="int", description="First number", required=True),
                ToolParameter(name="b", type="int", description="Second number", required=True),
            ],
            returns="The result of the mathematical operation",
        )
        tool_registry = ToolRegistry(tools=[calculator_tool])

        formatter = ToolCallingFormatter(config, tool_registry=tool_registry)

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's 5 + 3?"),
                ChatMessage(role="assistant", content="5 + 3 = 8"),
            ],
            reasoning=ReasoningTrace(
                style="freetext", content="I need to add two numbers. Add 5 and 3."
            ),
            tool_context=ToolContext(
                available_tools=[calculator_tool],
                executions=[
                    ToolExecution(
                        function_name="calculator",
                        arguments='{"operation": "add", "a": 5, "b": 3}',
                        reasoning="Add two numbers",
                        result="8",
                    )
                ],
            ),
            question="What's 5 + 3?",
            final_answer="5 + 3 = 8",
        )

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        messages = formatted["messages"]

        # Should have system message first
        assert len(messages) == 5  # system, user, assistant, tool, assistant  # noqa: PLR2004
        assert messages[0]["role"] == "system"

        # Check system message contains tools
        system_content = messages[0]["content"]
        assert "<tools>" in system_content
        assert "</tools>" in system_content
        assert "calculator" in system_content
        assert "mathematical calculations" in system_content

    def test_tool_calling_formatter_validation(self):
        """Test tool calling formatter validation."""

        formatter = ToolCallingFormatter()

        # Valid sample
        valid_sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Test question"),
                ChatMessage(role="assistant", content="Test answer"),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Test analysis"),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="test_tool",
                        arguments='{"param": "value"}',
                        reasoning="Test reasoning",
                        result="Test result",
                    )
                ],
            ),
            question="Test question",
            final_answer="Test answer",
        )
        assert formatter.validate(valid_sample.model_dump())

    def test_tool_calling_formatter_custom_formats(self):
        """Test tool calling formatter with custom formatting."""

        config = {
            "thinking_format": "<!-- thinking: {reasoning} -->",
            "tool_call_format": "```json\n{tool_call}\n```",
            "tool_response_format": "Result: {tool_output}",
        }
        formatter = ToolCallingFormatter(config)

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Test question"),
                ChatMessage(role="assistant", content="Test answer"),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Test reasoning"),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="test_tool",
                        arguments="{}",
                        reasoning="Test tool reasoning",
                        result="test output",
                    )
                ],
            ),
            question="Test question",
            final_answer="Test answer",
        )

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        messages = formatted["messages"]

        # Check custom thinking format markers are used
        assistant_msg = messages[1]["content"]
        assert "<!-- thinking:" in assistant_msg
        assert "-->" in assistant_msg
        assert "Test reasoning" in assistant_msg

        # Check custom tool call format
        assert "```json" in assistant_msg
        assert "```" in assistant_msg

        # Check custom tool response format
        tool_msg = messages[2]["content"]
        assert "Result: test output" in tool_msg

    def test_tool_calling_formatter_json_parsing(self):
        """Test tool calling formatter JSON parsing scenarios."""

        formatter = ToolCallingFormatter()

        # Test with single quotes in arguments string (will be parsed)
        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Test"),
                ChatMessage(role="assistant", content="answer"),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Test analysis"),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="test_tool",
                        arguments='{"key": "value"}',
                        reasoning="Test reasoning",
                        result="output",
                    )
                ],
            ),
            question="Test",
            final_answer="answer",
        )

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        messages = formatted["messages"]
        assistant_msg = messages[1]["content"]

        # Should contain properly formatted JSON
        assert '"key": "value"' in assistant_msg


class TestHarmonyFormatter:
    """Test the Harmony formatter."""

    def test_harmony_formatter_basic(self):
        """Test basic Harmony formatting with messages."""
        formatter = HarmonyFormatter()

        sample = {
            "messages": [
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 = 4"},
            ]
        }

        result = formatter.format_with_metadata([sample])
        assert len(result.samples) == 1

        formatted = result.samples[0]
        assert "text" in formatted

        text = formatted["text"]
        # Check for Harmony tokens
        assert "<|start|>" in text
        assert "<|end|>" in text
        assert "<|message|>" in text

        # Check roles
        assert "<|start|>system" in text
        assert "<|start|>user" in text
        assert "<|start|>assistant" in text

    def test_harmony_formatter_with_channels(self):
        """Test Harmony formatter with analysis and final channels."""
        config = {"default_channel": "final"}
        formatter = HarmonyFormatter(config)

        sample = {
            "question": "What is the capital of France?",
            "chain_of_thought": "I need to recall the capital city of France. France is a country in Europe, and its capital is Paris.",
            "answer": "The capital of France is Paris.",
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should have analysis channel for reasoning
        assert "<|channel|>analysis" in text
        # Should have final channel for answer
        assert "<|channel|>final" in text

    def test_harmony_formatter_with_developer_role(self):
        """Test Harmony formatter with developer role."""
        config = {
            "include_developer_role": True,
            "developer_instructions": "Always respond with detailed explanations",
        }
        formatter = HarmonyFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should include developer role
        assert "<|start|>developer" in text
        assert "Always respond with detailed explanations" in text

    def test_harmony_formatter_with_tools(self):
        """Test Harmony formatter with tool definitions."""
        config = {"include_developer_role": True, "tool_namespace": "functions"}
        formatter = HarmonyFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "I'll check the weather for you."},
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "parameters": {
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["location"],
                    },
                }
            ],
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should include tools in TypeScript format
        assert "namespace functions" in text
        assert "type get_weather" in text
        assert "location: string" in text
        assert 'unit?: "celsius" | "fahrenheit"' in text

    def test_harmony_formatter_with_metadata(self):
        """Test Harmony formatter with metadata in system message."""
        config = {
            "include_metadata": True,
            "knowledge_cutoff": "2024-01",
            "reasoning_level": "high",
            "current_date": "2024-03-15",  # Deterministic date for testing
        }
        formatter = HarmonyFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should include metadata in system message
        assert "Knowledge cutoff: 2024-01" in text
        assert "Current date: 2024-03-15" in text
        assert "Reasoning: high" in text
        assert "# Valid channels: analysis, commentary, final" in text

    def test_harmony_formatter_metadata_without_date(self):
        """Test Harmony formatter excludes current date when not configured."""
        config = {
            "include_metadata": True,
            "knowledge_cutoff": "2024-01",
            "reasoning_level": "high",
            # current_date is intentionally not provided
        }
        formatter = HarmonyFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ]
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should include metadata but NOT current date
        assert "Knowledge cutoff: 2024-01" in text
        assert "Reasoning: high" in text
        assert "Current date:" not in text  # Date should not be present

    def test_harmony_formatter_structured_output(self):
        """Test Harmony formatter with structured output format."""
        config = {"output_format": "structured"}
        formatter = HarmonyFormatter(config)

        sample = {"question": "What is AI?", "answer": "AI stands for Artificial Intelligence."}

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]

        # Should have messages list instead of text
        assert "messages" in formatted
        messages = formatted["messages"]

        # Check message structure
        assert all("role" in msg for msg in messages)
        assert all("content" in msg for msg in messages)

        # Check for channel in assistant messages
        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
        assert all("channel" in msg for msg in assistant_msgs)

    def test_harmony_formatter_tool_calls(self):
        """Test Harmony formatter with tool calls."""
        formatter = HarmonyFormatter()

        sample = {
            "messages": [
                {"role": "user", "content": "Calculate 5 * 7"},
                {
                    "role": "assistant",
                    "content": '{"operation": "multiply", "a": 5, "b": 7}',
                    "tool_calls": [
                        {
                            "function": {
                                "name": "calculator",
                                "arguments": '{"operation": "multiply", "a": 5, "b": 7}',
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "35"},
                {"role": "assistant", "content": "5 * 7 = 35"},
            ]
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Check for commentary channel for tool call
        assert "<|channel|>commentary" in text
        # Check for tool recipient
        assert "<|recipient|>functions.calculator" in text
        # Check for tool role
        assert "<|start|>tool" in text

    def test_harmony_formatter_validation(self):
        """Test Harmony formatter validation."""
        formatter = HarmonyFormatter()

        # Valid samples
        assert formatter.validate(
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                ]
            }
        )

        assert formatter.validate({"question": "Test?", "answer": "Response"})

        assert formatter.validate({"instruction": "Do something", "output": "Done"})

        # Invalid samples
        assert not formatter.validate({})
        assert not formatter.validate({"messages": []})
        assert not formatter.validate({"question": "Test?"})  # Missing answer
        assert not formatter.validate({"instruction": "Do"})  # Missing output

    def test_harmony_formatter_generic_format(self):
        """Test Harmony formatter with generic input format."""
        formatter = HarmonyFormatter()

        sample = {
            "prompt": "Explain quantum computing",
            "response": "Quantum computing uses quantum mechanics principles...",
            "context": "Educational context for advanced physics students",
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should format correctly
        assert "<|start|>user" in text
        assert "Explain quantum computing" in text
        assert "<|start|>assistant" in text
        assert "Quantum computing uses quantum mechanics" in text

    def test_harmony_formatter_load_from_registry(self):
        """Test loading Harmony formatter from registry."""
        registry = FormatterRegistry()
        formatter = registry.load_formatter("builtin://harmony.py")
        assert isinstance(formatter, HarmonyFormatter)

    def test_harmony_formatter_tools_without_names(self):
        """Test Harmony formatter skips tools without names."""
        config = {"include_developer_role": True, "tool_namespace": "functions"}
        formatter = HarmonyFormatter(config)

        sample = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "tools": [
                {"name": "valid_tool", "parameters": {"properties": {"arg": {"type": "string"}}}},
                {
                    # Missing name - should be skipped
                    "parameters": {"properties": {"arg": {"type": "string"}}}
                },
                {"name": "another_valid_tool", "parameters": {}},
            ],
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Should include only valid tools
        assert "type valid_tool" in text
        assert "type another_valid_tool" in text
        # Should not have "unknown" or undefined tools
        assert "type unknown" not in text

    def test_harmony_formatter_multiple_tool_calls(self):
        """Test Harmony formatter with multiple tool calls in a single message."""
        formatter = HarmonyFormatter()

        sample = {
            "messages": [
                {"role": "user", "content": "Calculate 5 * 7 and then 10 + 20"},
                {
                    "role": "assistant",
                    "content": "I'll calculate both for you.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "multiply",
                                "arguments": '{"a": 5, "b": 7}',
                            }
                        },
                        {
                            "function": {
                                "name": "add",
                                "arguments": '{"a": 10, "b": 20}',
                            }
                        },
                    ],
                },
                {"role": "tool", "content": "35"},
                {"role": "tool", "content": "30"},
                {"role": "assistant", "content": "5 * 7 = 35 and 10 + 20 = 30"},
            ]
        }

        result = formatter.format_with_metadata([sample])
        formatted = result.samples[0]
        text = formatted["text"]

        # Check that both tool calls are present with their recipients
        assert "<|recipient|>functions.multiply" in text
        assert "<|recipient|>functions.add" in text

        # Both should have commentary channel
        assert text.count("<|channel|>commentary") >= 2  # noqa: PLR2004


class TestOpenAISchemaFormatter:
    """Test the OpenAI Schema formatter."""

    def setup_method(self):
        """Set up test fixtures."""

        self.formatter = OpenAISchemaFormatter()

    def test_formatter_initialization(self):
        """Test formatter can be initialized."""

        formatter = OpenAISchemaFormatter()
        assert formatter is not None

    def test_formatter_with_config(self):
        """Test formatter initialization with config."""

        config = {
            "include_system_prompt": True,
            "validate_tool_schemas": True,
            "remove_available_tools_field": False,
        }
        formatter = OpenAISchemaFormatter(config)
        assert formatter is not None

    def test_validate_sample_with_messages(self):
        """Test validation of sample with messages."""
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }
        assert self.formatter.validate(sample)

    def test_validate_sample_without_messages(self):
        """Test validation fails without messages."""
        sample = {"text": "Hello"}
        assert not self.formatter.validate(sample)

    def test_validate_sample_empty_messages(self):
        """Test validation fails with empty messages."""
        sample = {"messages": []}
        assert not self.formatter.validate(sample)

    def test_format_sample_with_available_tools(self):
        """Test formatting sample with available_tools."""
        sample = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "Let me check..."},
            ],
            "available_tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": [
                        {
                            "name": "location",
                            "type": "str",
                            "description": "City name",
                            "required": True,
                        }
                    ],
                    "returns": "Weather information",
                    "category": "weather",
                }
            ],
        }

        result = self.formatter.format([sample])
        assert len(result) == 1

        # Convert FormattedOutput to dict
        formatted = result[0].model_dump()
        assert "messages" in formatted
        assert "tools" in formatted
        assert len(formatted["tools"]) == 1

        # Check tool format
        tool = formatted["tools"][0]
        assert tool["type"] == "function"
        assert "function" in tool
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get weather for a location"
        assert "parameters" in tool["function"]

    def test_format_sample_without_tools(self):
        """Test formatting sample without available_tools."""
        sample = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"},
            ]
        }

        result = self.formatter.format([sample])
        assert len(result) == 1

        # Convert FormattedOutput to dict
        formatted = result[0].model_dump()
        assert "messages" in formatted
        # No tools field should be added if not present in original
        assert "tools" not in formatted or formatted.get("tools") == []

    def test_tool_schema_conversion(self):
        """Test proper conversion of tools to OpenAI schema."""
        sample = {
            "messages": [{"role": "user", "content": "Test"}],
            "available_tools": [
                {
                    "name": "calculator",
                    "description": "Perform calculations",
                    "parameters": [
                        {
                            "name": "expression",
                            "type": "str",
                            "description": "Math expression",
                            "required": True,
                        },
                        {
                            "name": "precision",
                            "type": "int",
                            "description": "Decimal precision",
                            "required": False,
                            "default": "2",
                        },
                    ],
                    "returns": "Calculation result",
                    "category": "math",
                }
            ],
        }

        result = self.formatter.format([sample])
        formatted = result[0].model_dump()
        tool = formatted["tools"][0]

        # Verify OpenAI schema structure
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "calculator"
        assert tool["function"]["description"] == "Perform calculations"

        params = tool["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

        # Check parameter conversion
        assert "expression" in params["properties"]
        assert params["properties"]["expression"]["type"] == "string"
        assert "expression" in params["required"]

        assert "precision" in params["properties"]
        assert params["properties"]["precision"]["type"] == "integer"
        assert params["properties"]["precision"]["default"] == 2  # noqa: PLR2004
        assert "precision" not in params["required"]

    def test_multiple_tools_conversion(self):
        """Test conversion of multiple tools."""
        sample = {
            "messages": [{"role": "user", "content": "Test"}],
            "available_tools": [
                {
                    "name": "tool1",
                    "description": "First tool",
                    "parameters": [],
                    "returns": "Result",
                    "category": "general",
                },
                {
                    "name": "tool2",
                    "description": "Second tool",
                    "parameters": [],
                    "returns": "Result",
                    "category": "general",
                },
            ],
        }

        result = self.formatter.format([sample])
        formatted = result[0].model_dump()
        assert len(formatted["tools"]) == 2  # noqa: PLR2004
        assert formatted["tools"][0]["function"]["name"] == "tool1"
        assert formatted["tools"][1]["function"]["name"] == "tool2"

    def test_remove_available_tools_field(self):
        """Test removing available_tools field from output."""

        config = {"remove_available_tools_field": True}
        formatter = OpenAISchemaFormatter(config)

        sample = {
            "messages": [{"role": "user", "content": "Test"}],
            "available_tools": [
                {
                    "name": "test_tool",
                    "description": "Test",
                    "parameters": [],
                    "returns": "Result",
                    "category": "general",
                }
            ],
        }

        result = formatter.format([sample])
        formatted = result[0].model_dump()

        assert "tools" in formatted
        assert "available_tools" not in formatted

    def test_system_prompt_override(self):
        """Test overriding system prompt."""

        custom_prompt = "Custom system prompt for testing"
        config = {
            "include_system_prompt": True,
            "system_prompt_override": custom_prompt,
        }
        formatter = OpenAISchemaFormatter(config)

        sample = {
            "messages": [
                {"role": "system", "content": "Original prompt"},
                {"role": "user", "content": "Test"},
            ]
        }

        result = formatter.format([sample])
        formatted = result[0].model_dump()

        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][0]["content"] == custom_prompt

    def test_add_system_prompt_if_missing(self):
        """Test adding system prompt when missing."""

        custom_prompt = "New system prompt"
        config = {
            "include_system_prompt": True,
            "system_prompt_override": custom_prompt,
        }
        formatter = OpenAISchemaFormatter(config)

        sample = {"messages": [{"role": "user", "content": "Test"}]}

        result = formatter.format([sample])
        formatted = result[0].model_dump()

        assert formatted["messages"][0]["role"] == "system"
        assert formatted["messages"][0]["content"] == custom_prompt
        assert len(formatted["messages"]) == 2  # noqa: PLR2004

    def test_type_mapping(self):
        """Test correct type mapping from DeepFabric to JSON Schema."""
        sample = {
            "messages": [{"role": "user", "content": "Test"}],
            "available_tools": [
                {
                    "name": "test_types",
                    "description": "Test type mapping",
                    "parameters": [
                        {
                            "name": "str_param",
                            "type": "str",
                            "description": "String",
                            "required": True,
                        },
                        {
                            "name": "int_param",
                            "type": "int",
                            "description": "Integer",
                            "required": True,
                        },
                        {
                            "name": "float_param",
                            "type": "float",
                            "description": "Float",
                            "required": True,
                        },
                        {
                            "name": "bool_param",
                            "type": "bool",
                            "description": "Boolean",
                            "required": True,
                        },
                        {
                            "name": "list_param",
                            "type": "list",
                            "description": "List",
                            "required": True,
                        },
                        {
                            "name": "dict_param",
                            "type": "dict",
                            "description": "Dict",
                            "required": True,
                        },
                    ],
                    "returns": "Result",
                    "category": "test",
                }
            ],
        }

        result = self.formatter.format([sample])
        formatted = result[0].model_dump()
        props = formatted["tools"][0]["function"]["parameters"]["properties"]

        assert props["str_param"]["type"] == "string"
        assert props["int_param"]["type"] == "integer"
        assert props["float_param"]["type"] == "number"
        assert props["bool_param"]["type"] == "boolean"
        assert props["list_param"]["type"] == "array"
        assert props["dict_param"]["type"] == "object"

    def test_load_via_registry(self):
        """Test loading OpenAI Schema formatter via registry."""
        registry = FormatterRegistry()
        formatter = registry.load_formatter("builtin://openai")
        assert formatter is not None

        assert isinstance(formatter, OpenAISchemaFormatter)

    def test_example_config(self):
        """Test getting example configuration."""
        config = self.formatter.get_example_config()
        assert "include_system_prompt" in config
        assert "system_prompt_override" in config
        assert "validate_tool_schemas" in config
        assert "remove_available_tools_field" in config
        assert "parallel_tool_calls" in config

    def test_clean_openai_output(self):
        """Test that formatter outputs only OpenAI-compatible fields."""
        sample = {
            "messages": [
                {"role": "user", "content": "Test"},
                {"role": "assistant", "content": "Response"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "Test",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                }
            ],
            # DeepFabric-specific fields that should be removed
            "metadata": {"conversation_type": "chain_of_thought"},
            "reasoning": {"style": "structured", "content": []},
            "tool_context": {"available_tools": [], "executions": []},
            "agent_context": {"mode": "single_turn"},
            "question": "Test question",
            "final_answer": "Test answer",
        }

        result = self.formatter.format([sample])
        formatted = result[0].model_dump()

        # Should only have OpenAI fields
        assert "messages" in formatted
        assert "tools" in formatted

        # Should NOT have DeepFabric-specific fields
        assert "metadata" not in formatted
        assert "reasoning" not in formatted
        assert "tool_context" not in formatted
        assert "agent_context" not in formatted
        assert "question" not in formatted
        assert "final_answer" not in formatted

    def test_parallel_tool_calls_field(self):
        """Test parallel_tool_calls field configuration."""
        # Test without parallel_tool_calls (None = omit)
        formatter_none = OpenAISchemaFormatter({"parallel_tool_calls": None})
        sample = {"messages": [{"role": "user", "content": "Test"}]}

        result = formatter_none.format([sample])
        formatted = result[0].model_dump()
        assert "parallel_tool_calls" not in formatted

        # Test with parallel_tool_calls = False
        formatter_false = OpenAISchemaFormatter({"parallel_tool_calls": False})
        result = formatter_false.format([sample])
        formatted = result[0].model_dump()
        assert "parallel_tool_calls" in formatted
        assert formatted["parallel_tool_calls"] is False

        # Test with parallel_tool_calls = True
        formatter_true = OpenAISchemaFormatter({"parallel_tool_calls": True})
        result = formatter_true.format([sample])
        formatted = result[0].model_dump()
        assert "parallel_tool_calls" in formatted
        assert formatted["parallel_tool_calls"] is True
