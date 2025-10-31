"""
Integration tests for agent CoT rich schema and tool-calling system.

This module tests the complete integration between:
- RichAgentCoT schema
- Tool-calling formatter
- Generator with agent_cot_rich type
"""

from unittest.mock import MagicMock, patch  # noqa: F401

import pytest  # type: ignore # noqa: F401

from deepfabric.config import DeepFabricConfig
from deepfabric.formatters.builtin.tool_calling import ToolCallingFormatter
from deepfabric.generator import DataSetGenerator
from deepfabric.schemas import (
    ChatMessage,
    Conversation,
    ReasoningTrace,
    ToolContext,
    ToolExecution,
    get_conversation_schema,
)


class TestAgentCoTIntegration:
    """Test integration between agent CoT schema and the generator."""

    def test_chain_of_thought_schema_available(self):
        """Test that chain_of_thought schema (unified Conversation) is available."""

        schema = get_conversation_schema("chain_of_thought")
        assert schema is not None
        assert schema == Conversation

        # Test unified Conversation schema with tool_context and reasoning
        test_data = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ],
            "metadata": {},
            "question": "Test question",
            "final_answer": "Test answer",
            "reasoning": {
                "style": "structured",
                "content": [
                    {"step_number": 1, "thought": "Step 1", "action": None},
                    {"step_number": 2, "thought": "Step 2", "action": None},
                ],
            },
            "tool_context": {
                "available_tools": [],
                "executions": [
                    {
                        "function_name": "test_tool",
                        "arguments": '{"param": "value"}',
                        "reasoning": "Testing tool execution",
                        "result": "Test result",
                    }
                ],
            },
        }

        instance = schema(**test_data)
        assert isinstance(instance, Conversation)
        assert instance.question == "Test question"
        assert instance.reasoning is not None
        assert instance.reasoning.style == "structured"
        assert instance.tool_context is not None
        assert len(instance.tool_context.executions) == 1
        assert instance.tool_context.executions[0].function_name == "test_tool"

    @patch("deepfabric.generator.LLMClient")
    def test_generator_with_agent_mode(self, mock_llm_client):  # noqa: ARG002
        """Test DataSetGenerator with chain_of_thought + agent_mode configuration."""
        # Create generator with new modular configuration
        generator_config = {
            "generation_system_prompt": "Generate agent reasoning examples",
            "provider": "openai",
            "model_name": "gpt-4",
            "conversation_type": "chain_of_thought",
            "reasoning_style": "structured",
            "agent_mode": "single_turn",
            "available_tools": ["get_weather", "calculate"],
            "num_samples": 1,
            "batch_size": 1,
            "temperature": 0.7,
        }

        generator = DataSetGenerator(**generator_config)

        # Should have tool registry initialized when agent_mode is set
        assert generator.tool_registry is not None
        tool_names = generator.tool_registry.get_tool_names()
        assert "get_weather" in tool_names
        assert "calculate" in tool_names

    def test_tool_calling_formatter_with_rich_sample(self):
        """Test tool calling formatter with a rich agent CoT sample."""
        formatter = ToolCallingFormatter()

        rich_sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Calculate the area of a circle with radius 5"),
                ChatMessage(
                    role="assistant",
                    content="The area of a circle with radius 5 is approximately 78.54 square units.",
                ),
            ],
            reasoning=ReasoningTrace(
                style="freetext",
                content="This is a geometry problem requiring calculation of circle area using the formula π × r². Identify that this is a circle area calculation. Recall the formula: Area = π × radius². Substitute radius = 5 into the formula. Calculate π × 5² = π × 25 ≈ 78.54.",
            ),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="calculate",
                        arguments='{"operation": "circle_area", "radius": 5}',
                        reasoning="Need to calculate circle area with precise π value",
                        result="78.54",
                    )
                ],
            ),
            question="Calculate the area of a circle with radius 5",
            final_answer="The area of a circle with radius 5 is approximately 78.54 square units.",
        )

        result = formatter.format_with_metadata([rich_sample])
        assert len(result.samples) == 1

        formatted = result.samples[0]
        messages = formatted["messages"]

        # Should have proper message structure
        assert (
            len(messages) == 4  # noqa: PLR2004
        )  # user, assistant+thinking, tool, assistant+answer  # noqa: PLR2004

        # Check that all rich reasoning is preserved in thinking
        thinking_content = messages[1]["content"]
        # Verify reasoning content is included
        assert "geometry problem" in thinking_content
        assert "circle area" in thinking_content

    def test_end_to_end_rich_agent_workflow(self):
        """Test complete workflow from generation to formatting."""
        generated_sample = Conversation(
            messages=[
                ChatMessage(role="user", content="Find restaurants near Times Square"),
                ChatMessage(
                    role="assistant",
                    content="I found 3 restaurants near Times Square: Joe's Pizza, The View, and Olive Garden.",
                ),
            ],
            reasoning=ReasoningTrace(
                style="freetext",
                content="User wants restaurant recommendations for a specific location in NYC. Parse the location request for Times Square. Use restaurant search tool for that area. Filter results for quality and ratings. Restaurant search tool can find businesses by location. Use 'Times Square, NYC' as location with restaurant category filter. Tool returned list of restaurants with ratings and locations.",
            ),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="search_web",
                        arguments='{"location": "Times Square, NYC", "category": "restaurant"}',
                        reasoning="Search for restaurants in the specified location",
                        result='[{"name": "Joe\'s Pizza", "rating": 4.5}, ...]',
                    )
                ],
            ),
            question="Find restaurants near Times Square",
            final_answer="I found 3 restaurants near Times Square: Joe's Pizza, The View, and Olive Garden.",
        )

        formatter = ToolCallingFormatter(
            {
                "include_tools_in_system": True,
                "system_prompt": "You are a helpful assistant with restaurant search capabilities.",
            }
        )

        result = formatter.format_with_metadata([generated_sample])
        formatted_sample = result.samples[0]

        assert "messages" in formatted_sample
        messages = formatted_sample["messages"]

        # Should have embedded execution format
        assistant_thinking = messages[1]["content"]
        assert "<think>" in assistant_thinking and "</think>" in assistant_thinking
        assert "<tool_call>" in assistant_thinking and "</tool_call>" in assistant_thinking

        tool_response = messages[2]["content"]
        assert "<tool_response>" in tool_response and "</tool_response>" in tool_response

        # Verify rich reasoning is preserved
        assert "User wants restaurant recommendations" in assistant_thinking
        assert "Parse the location request" in assistant_thinking
        assert "Restaurant search tool can find" in assistant_thinking

    def test_multi_tool_execution_support(self):
        """Test that formatter handles multiple tool executions correctly."""
        formatter = ToolCallingFormatter()

        multi_tool_sample = Conversation(
            messages=[
                ChatMessage(role="user", content="What's the weather in NYC and London?"),
                ChatMessage(
                    role="assistant",
                    content="NYC is sunny at 72°F, while London is rainy at 15°C.",
                ),
            ],
            reasoning=ReasoningTrace(
                style="freetext",
                content="Need to check weather in two different cities. Identify two separate locations. Call weather API for NYC. Call weather API for London. Combine results into answer. Weather tool can retrieve current conditions for any location. Use city names as location parameters for each call. Both API calls succeeded, providing current conditions for each city.",
            ),
            tool_context=ToolContext(
                available_tools=[],
                executions=[
                    ToolExecution(
                        function_name="get_weather",
                        arguments='{"location": "NYC"}',
                        reasoning="Get current weather for New York City",
                        result="72°F, sunny",
                    ),
                    ToolExecution(
                        function_name="get_weather",
                        arguments='{"location": "London"}',
                        reasoning="Get current weather for London",
                        result="15°C, rainy",
                    ),
                ],
            ),
            question="What's the weather in NYC and London?",
            final_answer="NYC is sunny at 72°F, while London is rainy at 15°C.",
        )

        result = formatter.format_with_metadata([multi_tool_sample])
        formatted_sample = result.samples[0]
        messages = formatted_sample["messages"]

        # Should have: user, assistant+thinking+both_tool_calls, tool_response_1, tool_response_2, final_answer
        assert len(messages) == 5  # noqa: PLR2004

        # Both tool calls should be in the assistant message
        assistant_thinking = messages[1]["content"]
        assert assistant_thinking.count("<tool_call>") == 2  # noqa: PLR2004
        assert "get_weather" in assistant_thinking
        assert "NYC" in assistant_thinking
        assert "London" in assistant_thinking

        # Two separate tool response messages
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "tool"
        assert "72°F, sunny" in messages[2]["content"]
        assert "15°C, rainy" in messages[3]["content"]

    def test_config_integration_with_yaml_style(self):
        """Test that the config system properly handles modular agent configuration."""
        # Simulate YAML config structure with new modular config
        config_dict = {
            "dataset_system_prompt": "Generate agent reasoning examples",
            "topic_tree": {
                "topic_prompt": "Real-world scenarios requiring tools",
                "provider": "openai",
                "model": "gpt-4",
                "depth": 2,
                "degree": 3,
                "save_as": "topics.jsonl",
            },
            "data_engine": {
                "generation_system_prompt": "Create detailed agent reasoning with tools",
                "provider": "openai",
                "model": "gpt-4",
                "conversation_type": "chain_of_thought",
                "reasoning_style": "structured",
                "agent_mode": "single_turn",
                "available_tools": ["get_weather", "calculate"],
                "num_samples": 10,
                "save_as": "raw_data.jsonl",
            },
            "dataset": {
                "save_as": "final_dataset.jsonl",
                "creation": {"num_steps": 10, "batch_size": 2, "sys_msg": True},
                "formatters": [
                    {
                        "name": "tool_calling",
                        "template": "builtin://tool_calling",
                        "output": "formatted.jsonl",
                        "config": {"include_tools_in_system": True},
                    }
                ],
            },
        }

        # Should not raise validation errors
        config = DeepFabricConfig(**config_dict)

        # Verify modular configuration is accepted
        assert config.data_engine.conversation_type == "chain_of_thought"
        assert config.data_engine.reasoning_style == "structured"
        assert config.data_engine.agent_mode == "single_turn"
        assert config.data_engine.available_tools == ["get_weather", "calculate"]

        # Verify formatter configuration
        assert len(config.dataset.formatters) == 1
        formatter_config = config.dataset.formatters[0]
        assert formatter_config.template == "builtin://tool_calling"
        assert formatter_config.config["include_tools_in_system"] is True


class TestToolSystemIntegration:
    """Test integration between tool system and agent CoT generation."""

    def test_tool_registry_creation_for_agent_mode(self):
        """Test that tool registry is properly created when agent_mode is enabled."""
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(
                generation_system_prompt="Test prompt",
                conversation_type="chain_of_thought",
                reasoning_style="structured",
                agent_mode="single_turn",
                available_tools=["get_weather", "calculate"],
                provider="openai",
                model_name="gpt-4",
            )

        # Should have initialized tool registry when agent_mode is set
        assert generator.tool_registry is not None
        assert len(generator.tool_registry.tools) > 0

        # Should contain requested tools
        tool_names = generator.tool_registry.get_tool_names()
        assert "get_weather" in tool_names
        assert "calculate" in tool_names

    def test_tool_metadata_injection(self):
        """Test that available tools are properly injected into samples."""
        with patch("deepfabric.generator.LLMClient"):
            generator = DataSetGenerator(
                generation_system_prompt="Test prompt",
                conversation_type="chain_of_thought",
                reasoning_style="freetext",
                agent_mode="single_turn",
                available_tools=["get_weather"],
                provider="openai",
                model_name="gpt-4",
            )

            # Should have tool registry with correct tools
            assert generator.tool_registry is not None
            tool_names = generator.tool_registry.get_tool_names()
            assert "get_weather" in tool_names

            # Should have tool definitions available
            weather_tool = generator.tool_registry.get_tool("get_weather")
            assert weather_tool is not None
            assert weather_tool.name == "get_weather"
            assert len(weather_tool.parameters) > 0
