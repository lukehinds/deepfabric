"""Tests for single tool call formatter."""

import json

from deepfabric.formatters.builtin.single_tool_call import (
    SingleToolCallConfig,
    SingleToolCallFormatter,
)
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


class TestSingleToolCallFormatter:
    """Test suite for SingleToolCallFormatter."""

    def test_basic_formatting(self):
        """Test basic single tool call formatting."""
        formatter = SingleToolCallFormatter()

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's the weather in Paris?"),
                ChatMessage(
                    role="assistant",
                    content="The weather in Paris is currently 15°C and partly cloudy.",
                ),
            ],
            reasoning=ReasoningTrace(
                style="freetext", content="Need to check the weather. Check weather for Paris."
            ),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="get_weather",
                        arguments='{"location": "Paris"}',
                        reasoning="Get weather for Paris",
                        result="15°C, partly cloudy",
                    )
                ],
            ),
            question="What's the weather in Paris?",
            final_answer="The weather in Paris is currently 15°C and partly cloudy.",
        )

        result = formatter._format_single_sample(sample)

        assert result is not None
        assert "messages" in result
        messages = result["messages"]

        # Check message structure
        min_expected_messages = 4  # system (optional), user, assistant, tool, assistant
        assert len(messages) >= min_expected_messages

        # Find message roles
        roles = [msg["role"] for msg in messages]
        assert "user" in roles
        assert "assistant" in roles
        assert "tool" in roles

        # Check user message
        user_msg = next(msg for msg in messages if msg["role"] == "user")
        assert user_msg["content"] == sample.question

        # Check first assistant message contains tool call
        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
        min_assistant_messages = 2
        assert len(assistant_msgs) >= min_assistant_messages
        assert "<tool_call>" in assistant_msgs[0]["content"]
        assert "get_weather" in assistant_msgs[0]["content"]

        # Check final answer
        assert assistant_msgs[-1]["content"] == sample.final_answer

    def test_with_custom_config(self):
        """Test formatter with custom configuration."""
        config = {
            "system_prompt": "Custom system prompt",
            "include_tools_in_system": False,
            "include_reasoning_prefix": False,
            "tool_call_format": "TOOL: {tool_call}",
            "tool_response_as_json": False,
        }

        formatter = SingleToolCallFormatter(config)

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Calculate 2+2"),
                ChatMessage(role="assistant", content="The result is 4."),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Need to calculate. Calculate 2+2."),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="calculator",
                        arguments='{"expression": "2+2"}',
                        reasoning="Calculate sum",
                        result="4",
                    )
                ],
            ),
            question="Calculate 2+2",
            final_answer="The result is 4.",
        )

        result = formatter._format_single_sample(sample)
        assert result is not None

        messages = result["messages"]

        # Check custom tool call format
        assistant_msg = next(msg for msg in messages if msg["role"] == "assistant")
        assert "TOOL:" in assistant_msg["content"]

        # Check tool response is not JSON
        tool_msg = next(msg for msg in messages if msg["role"] == "tool")
        assert tool_msg["content"] == "4"

    def test_with_available_tools(self):
        """Test formatter with tool registry."""

        tool_registry = ToolRegistry(
            tools=[
                ToolDefinition(
                    name="get_time",
                    description="Get current time in a timezone",
                    parameters=[
                        ToolParameter(
                            name="timezone",
                            type="str",
                            description="Timezone identifier",
                            required=True,
                        )
                    ],
                    returns="Current time in timezone",
                )
            ]
        )

        formatter = SingleToolCallFormatter(tool_registry=tool_registry)

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's the time in Tokyo?"),
                ChatMessage(role="assistant", content="The current time in Tokyo is 10:30 PM JST."),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Need time. Get time."),
            tool_context=ToolContext(
                available_tools=tool_registry.tools,
                executions=[
                    ToolExecution(
                        function_name="get_time",
                        arguments='{"timezone": "Asia/Tokyo"}',
                        reasoning="Get Tokyo time",
                        result='{"time": "22:30", "timezone": "JST"}',
                    )
                ],
            ),
            question="What's the time in Tokyo?",
            final_answer="The current time in Tokyo is 10:30 PM JST.",
        )

        result = formatter._format_single_sample(sample)
        assert result is not None

        messages = result["messages"]

        # Check system message includes tool definition
        system_msg = next(msg for msg in messages if msg["role"] == "system")
        assert "get_time" in system_msg["content"]
        assert "timezone" in system_msg["content"]

    def test_json_tool_response(self):
        """Test JSON formatting of tool responses."""
        formatter = SingleToolCallFormatter({"tool_response_as_json": True})

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Test"),
                ChatMessage(role="assistant", content="Done"),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Test"),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="test_tool",
                        arguments="{}",
                        reasoning="Test",
                        result='{"temperature": 20, "unit": "celsius"}',
                    )
                ],
            ),
            question="Test",
            final_answer="Done",
        )

        result = formatter._format_single_sample(sample)
        assert result is not None
        tool_msg = next(msg for msg in result["messages"] if msg["role"] == "tool")

        # Should be valid JSON
        parsed = json.loads(tool_msg["content"])
        assert "result" in parsed
        result_data = json.loads(parsed["result"])
        assert result_data["temperature"] == 20  # noqa: PLR2004
        assert result_data["unit"] == "celsius"

    def test_reasoning_prefix_generation(self):
        """Test reasoning prefix generation for different tools."""
        formatter = SingleToolCallFormatter(
            {
                "include_reasoning_prefix": True,
                "reasoning_prefix_template": "I'll {action} for you.",
            }
        )

        # Test weather tool with location
        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's the weather?"),
                ChatMessage(role="assistant", content="It's sunny."),
            ],
            reasoning=ReasoningTrace(style="freetext", content="Need weather. Get weather."),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="get_weather",
                        arguments='{"location": "Paris"}',
                        reasoning="Get weather",
                        result="Sunny",
                    )
                ],
            ),
            question="What's the weather?",
            final_answer="It's sunny.",
        )

        result = formatter._format_single_sample(sample)
        assert result is not None
        messages = result.get("messages", [])
        assistant_msg = next((msg for msg in messages if msg.get("role") == "assistant"), None)
        assert assistant_msg is not None
        assert "I'll check the weather in Paris for you." in assistant_msg["content"]

    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls."""
        formatter = SingleToolCallFormatter()

        sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's the weather and time?"),
                ChatMessage(
                    role="assistant",
                    content="The weather in Paris is 15°C and sunny. The current time is 14:30 CET.",
                ),
            ],
            reasoning=ReasoningTrace(
                style="freetext", content="Need weather and time. Get weather. Get time."
            ),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="get_weather",
                        arguments='{"location": "Paris"}',
                        reasoning="Get weather",
                        result="15°C, sunny",
                    ),
                    ToolExecution(
                        function_name="get_time",
                        arguments='{"timezone": "Europe/Paris"}',
                        reasoning="Get time",
                        result="14:30 CET",
                    ),
                ],
            ),
            question="What's the weather and time?",
            final_answer="The weather in Paris is 15°C and sunny. The current time is 14:30 CET.",
        )

        result = formatter._format_single_sample(sample)
        assert result is not None

        messages = result["messages"]

        # Count the number of assistant messages
        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
        # Should have: weather call, time call, final answer
        expected_min_assistant_msgs = 3
        assert len(assistant_msgs) >= expected_min_assistant_msgs

        # Check tool messages
        tool_msgs = [msg for msg in messages if msg["role"] == "tool"]
        expected_tool_msgs = 2
        assert len(tool_msgs) == expected_tool_msgs

        # Verify the final message is the answer
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == sample.final_answer

    def test_config_model(self):
        """Test configuration model."""
        config = SingleToolCallConfig()

        # Check defaults
        assert config.include_tools_in_system is True
        assert config.include_reasoning_prefix is True
        assert config.tool_response_as_json is True
        assert "<tool_call>" in config.tool_call_format

        # Test custom values
        custom_config = SingleToolCallConfig(
            system_prompt="Custom prompt",
            include_tools_in_system=False,
            tool_response_as_json=False,
        )

        assert custom_config.system_prompt == "Custom prompt"
        assert custom_config.include_tools_in_system is False
        assert custom_config.tool_response_as_json is False
