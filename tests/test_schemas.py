"""
Tests for DeepFabric schema system.

This module tests:
- Schema framework and registry
- Rich agent CoT schema validation
- Schema mixins and composition
- Mathematical validation
"""

import pytest

from deepfabric.schemas import CONVERSATION_SCHEMAS, get_conversation_schema


class TestSchemaFramework:
    """Test the schema framework and registry system."""

    def test_conversation_schemas_exist(self):
        """Test that expected schemas are available."""
        # Check that basic schemas are in mapping
        assert "basic" in CONVERSATION_SCHEMAS
        assert "agent_cot_tools" in CONVERSATION_SCHEMAS

    def test_get_conversation_schema(self):
        """Test conversation schema retrieval."""
        # Test basic schema
        basic_schema = get_conversation_schema("basic")
        assert basic_schema is not None

        # Test agent schemas
        agent_schema = get_conversation_schema("agent_cot_tools")
        assert agent_schema is not None


class TestAgentCoTSchemas:
    """Test agent CoT schema validation and functionality."""

    def test_agent_cot_tools_schema_available(self):
        """Test that agent_cot_tools schema is available."""
        schema = get_conversation_schema("agent_cot_tools")
        assert schema is not None

        # Should be able to create an instance
        sample_data = {
            "question": "Test question",
            "initial_analysis": "Test analysis",
            "reasoning_steps": ["Step 1", "Step 2"],
            "tool_selection_rationale": "Tool was chosen because...",
            "parameter_reasoning": "Parameters determined by...",
            "result_interpretation": "Result means...",
            "tool_used": "test_tool",
            "tool_input": '{"param": "value"}',
            "tool_output": "Test result",
            "answer": "Test answer",
        }

        instance = schema(**sample_data)
        assert instance.question == "Test question"  # type: ignore

    def test_agent_cot_hybrid_schema(self):
        """Test that agent_cot_hybrid schema works with hybrid data."""
        schema = get_conversation_schema("agent_cot_hybrid")
        assert schema is not None

        # Should be able to create an instance with hybrid data
        hybrid_data = {
            "question": "Test question",
            "chain_of_thought": "Natural language reasoning",
            "reasoning_trace": [
                {"step_number": 1, "thought": "Test thought", "action": "Test action"}
            ],
            "tool_selection_rationale": "Tool was chosen because...",
            "parameter_reasoning": "Parameters determined by...",
            "result_interpretation": "Result means...",
            "tool_used": "test_tool",
            "tool_input": '{"param": "value"}',
            "tool_output": "Test result",
            "final_answer": "Test answer",
        }

        instance = schema(**hybrid_data)
        assert instance.question == "Test question"  # type: ignore
        assert instance.chain_of_thought == "Natural language reasoning"  # type: ignore
        assert len(instance.reasoning_trace) == 1  # type: ignore
        assert instance.final_answer == "Test answer"  # type: ignore


class TestBasicSchemaFunctionality:
    """Test basic schema functionality that exists in the current system."""

    def test_tool_conversation_schema(self):
        """Test the tool conversation schema."""
        schema = get_conversation_schema("tool_calling")
        assert schema is not None

        # Test with basic tool conversation data
        tool_data = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "I'll check the weather for you."},
            ]
        }

        instance = schema(**tool_data)
        assert len(instance.messages) == 2  # type: ignore  # noqa: PLR2004

    def test_basic_conversation_schema(self):
        """Test the basic conversation schema."""
        schema = get_conversation_schema("basic")
        assert schema is not None

        # Test with basic conversation data
        basic_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        instance = schema(**basic_data)
        assert len(instance.messages) == 2  # type: ignore  # noqa: PLR2004


class TestSchemaIntegration:
    """Test schema integration with the broader system."""

    def test_conversation_schemas_mapping(self):
        """Test that conversation schemas mapping contains expected types."""

        # Check that key schemas exist
        expected_schemas = ["basic", "tool_calling", "agent_cot_tools", "agent_cot_hybrid"]
        for schema_type in expected_schemas:
            assert schema_type in CONVERSATION_SCHEMAS

    def test_schema_retrieval(self):
        """Test schema retrieval for different types."""
        # Test basic types work
        for schema_type in ["basic", "tool_calling", "agent_cot_tools", "agent_cot_hybrid"]:
            schema = get_conversation_schema(schema_type)
            assert schema is not None

    def test_unsupported_conversation_type(self):
        """Test error handling for unsupported conversation types."""
        with pytest.raises(ValueError) as exc_info:
            get_conversation_schema("nonexistent_type")

        error_message = str(exc_info.value)
        assert "Unsupported conversation type" in error_message
        assert "nonexistent_type" in error_message
