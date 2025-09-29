"""Tests for single tool call formatter."""

import json

from deepfabric.formatters.builtin.single_tool_call import (
    SingleToolCallConfig,
    SingleToolCallFormatter,
)


class TestSingleToolCallFormatter:
    """Test suite for SingleToolCallFormatter."""

    def test_basic_formatting(self):
        """Test basic single tool call formatting."""
        formatter = SingleToolCallFormatter()

        sample = {
            "question": "What's the weather in Paris?",
            "reasoning": "Need to check the weather",
            "tool_used": "get_weather",
            "tool_input": '{"location": "Paris"}',
            "tool_output": "15°C, partly cloudy",
            "answer": "The weather in Paris is currently 15°C and partly cloudy.",
        }

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
        assert user_msg["content"] == "What's the weather in Paris?"

        # Check first assistant message contains tool call
        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
        min_assistant_messages = 2
        assert len(assistant_msgs) >= min_assistant_messages
        assert "<tool_call>" in assistant_msgs[0]["content"]
        assert "get_weather" in assistant_msgs[0]["content"]

        # Check final answer
        assert assistant_msgs[-1]["content"] == sample["answer"]

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

        sample = {
            "question": "Calculate 2+2",
            "tool_used": "calculator",
            "tool_input": '{"expression": "2+2"}',
            "tool_output": "4",
            "answer": "The result is 4.",
        }

        result = formatter._format_single_sample(sample)
        assert result is not None

        messages = result["messages"]

        # Check no reasoning prefix
        assistant_msg = next(msg for msg in messages if msg["role"] == "assistant")
        assert not assistant_msg["content"].startswith("I'll")

        # Check custom tool call format
        assert "TOOL:" in assistant_msg["content"]

        # Check tool response is not JSON
        tool_msg = next(msg for msg in messages if msg["role"] == "tool")
        assert tool_msg["content"] == "4"

    def test_with_available_tools(self):
        """Test formatter with available tools list."""
        formatter = SingleToolCallFormatter()

        sample = {
            "question": "What's the time in Tokyo?",
            "tool_used": "get_time",
            "tool_input": '{"timezone": "Asia/Tokyo"}',
            "tool_output": '{"time": "22:30", "timezone": "JST"}',
            "answer": "The current time in Tokyo is 10:30 PM JST.",
            "available_tools": [
                {
                    "name": "get_time",
                    "description": "Get current time in a timezone",
                    "parameters": [
                        {
                            "name": "timezone",
                            "type": "string",
                            "description": "Timezone identifier",
                            "required": True,
                        }
                    ],
                }
            ],
        }

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

        sample = {
            "question": "Test",
            "tool_used": "test_tool",
            "tool_input": "{}",
            "tool_output": {"temperature": 20, "unit": "celsius"},
            "answer": "Done",
        }

        result = formatter._format_single_sample(sample)
        tool_msg = next(msg for msg in result["messages"] if msg["role"] == "tool")

        # Should be valid JSON
        parsed = json.loads(tool_msg["content"])
        expected_temperature = 20
        assert parsed["temperature"] == expected_temperature
        assert parsed["unit"] == "celsius"

    def test_reasoning_prefix_generation(self):
        """Test reasoning prefix generation for different tools."""
        formatter = SingleToolCallFormatter(
            {
                "include_reasoning_prefix": True,
                "reasoning_prefix_template": "I'll {action} for you.",
            }
        )

        # Test weather tool
        sample = {
            "question": "What's the weather?",
            "tool_used": "get_weather",
            "tool_input": '{"location": "Paris"}',
            "tool_output": "Sunny",
            "answer": "It's sunny.",
        }

        result = formatter._format_single_sample(sample)
        assistant_msg = next(msg for msg in result["messages"] if msg["role"] == "assistant")
        assert "I'll check the weather in Paris for you." in assistant_msg["content"]

        # Test generic tool
        sample["tool_used"] = "unknown_tool"
        result = formatter._format_single_sample(sample)
        assistant_msg = next(msg for msg in result["messages"] if msg["role"] == "assistant")
        assert "I'll use the unknown_tool tool for you." in assistant_msg["content"]

    def test_invalid_sample_handling(self):
        """Test handling of invalid samples."""
        formatter = SingleToolCallFormatter()

        # Missing required field
        invalid_sample = {
            "question": "Test",
            # Missing tool_used
            "answer": "Result",
        }

        result = formatter._format_single_sample(invalid_sample)
        assert result is None

        # Missing answer
        invalid_sample = {
            "question": "Test",
            "tool_used": "test_tool",
            # Missing answer or final_answer
        }

        result = formatter._format_single_sample(invalid_sample)
        assert result is None

    def test_multiple_tool_calls(self):
        """Test handling of multiple tool calls without duplicating final answer."""
        formatter = SingleToolCallFormatter()

        sample = {
            "question": "What's the weather and time?",
            "tool_used": "get_weather",
            "tool_input": '{"location": "Paris"}',
            "tool_output": "15°C, sunny",
            "additional_tools": [
                {
                    "name": "get_time",
                    "arguments": {"timezone": "Europe/Paris"},
                    "result": "14:30 CET",
                }
            ],
            "answer": "The weather in Paris is 15°C and sunny. The current time is 14:30 CET.",
        }

        result = formatter._format_single_sample(sample)
        assert result is not None

        messages = result["messages"]

        # Count the number of assistant messages with the final answer
        final_answer_count = sum(
            1 for msg in messages if msg["role"] == "assistant" and "14:30 CET" in msg["content"]
        )

        # Should only have one final answer message
        assert final_answer_count == 1

        # Verify message sequence
        roles = [msg["role"] for msg in messages]
        # Should have: system (optional), user, assistant (weather call), tool, assistant (time call), tool, assistant (final)
        expected_min_assistant_msgs = 3  # weather call, time call, final answer
        expected_tool_msgs = 2  # weather response, time response
        assert roles.count("assistant") >= expected_min_assistant_msgs
        assert roles.count("tool") == expected_tool_msgs

        # Check that the additional tool has proper reasoning prefix
        assistant_msgs = [msg for msg in messages if msg["role"] == "assistant"]
        # Find the message with the time tool call
        time_call_msg = next((msg for msg in assistant_msgs if "get_time" in msg["content"]), None)
        assert time_call_msg is not None
        # Should have specific action for timezone
        assert "check the time in Europe/Paris" in time_call_msg["content"]

        # Verify the final message is the answer
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == sample["answer"]

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
