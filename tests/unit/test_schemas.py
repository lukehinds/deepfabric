"""
Tests for DeepFabric schema system.

This module tests:
- Schema framework and registry
- Rich agent CoT schema validation
- Schema mixins and composition
- Mathematical validation
"""

import pytest

from deepfabric.schemas import CONVERSATION_SCHEMAS, Conversation, get_conversation_schema


class TestSchemaFramework:
    """Test the schema framework and registry system."""

    def test_conversation_schemas_exist(self):
        """Test that expected schemas are available in new modular system."""
        # Check that new modular schemas are in mapping
        assert "basic" in CONVERSATION_SCHEMAS
        assert "chain_of_thought" in CONVERSATION_SCHEMAS

    def test_get_conversation_schema(self):
        """Test conversation schema retrieval with new modular types."""

        # All new conversation types return the unified Conversation schema
        basic_schema = get_conversation_schema("basic")
        assert basic_schema is not None
        assert basic_schema == Conversation

        chain_of_thought_schema = get_conversation_schema("chain_of_thought")
        assert chain_of_thought_schema is not None
        assert chain_of_thought_schema == Conversation


class TestUnifiedConversationSchema:
    """Test the new unified Conversation schema with capability fields."""

    def test_chain_of_thought_with_reasoning_capability(self):
        """Test Conversation schema with reasoning capability."""

        schema = get_conversation_schema("chain_of_thought")
        assert schema == Conversation

        # Create instance with reasoning capability (freetext style)
        sample_data = {
            "messages": [
                {"role": "user", "content": "Test question"},
                {"role": "assistant", "content": "Test answer"},
            ],
            "metadata": {},
            "question": "Test question",
            "final_answer": "Test answer",
            "reasoning": {
                "style": "freetext",
                "content": "This is my natural language reasoning...",
            },
        }

        instance = schema(**sample_data)
        assert instance.question == "Test question"
        assert instance.reasoning is not None
        assert instance.reasoning.style == "freetext"
        assert isinstance(instance.reasoning.content, str)

    def test_chain_of_thought_with_tool_context(self):
        """Test Conversation schema with tool_context capability."""

        schema = get_conversation_schema("chain_of_thought")
        assert schema == Conversation

        # Create instance with tool_context capability
        sample_data = {
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

        instance = schema(**sample_data)
        assert instance.question == "Test question"
        assert instance.reasoning is not None
        assert instance.reasoning.style == "structured"
        assert instance.tool_context is not None
        assert len(instance.tool_context.executions) == 1


class TestBasicSchemaFunctionality:
    """Test basic schema functionality with new unified Conversation schema."""

    def test_conversation_with_metadata(self):
        """Test conversation with metadata field."""

        schema = get_conversation_schema("basic")
        assert schema is not None
        assert schema == Conversation

        # Test with conversation data that has metadata
        data_with_metadata = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "I'll check the weather for you."},
            ],
            "metadata": {"topic": "weather"},
        }

        instance = schema(**data_with_metadata)
        assert len(instance.messages) == 2  # noqa: PLR2004

    def test_basic_conversation_schema(self):
        """Test the basic conversation schema."""

        schema = get_conversation_schema("basic")
        assert schema is not None
        assert schema == Conversation

        # Test with basic conversation data
        basic_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "metadata": {},
        }

        instance = schema(**basic_data)
        assert len(instance.messages) == 2  # noqa: PLR2004


class TestSchemaIntegration:
    """Test schema integration with the broader system."""

    def test_conversation_schemas_mapping(self):
        """Test that conversation schemas mapping contains new modular types."""
        # Check that new modular schemas exist
        expected_schemas = ["basic", "chain_of_thought"]
        for schema_type in expected_schemas:
            assert schema_type in CONVERSATION_SCHEMAS

    def test_schema_retrieval(self):
        """Test schema retrieval for modular types."""

        # Test all modular types return unified Conversation schema
        for schema_type in ["basic", "chain_of_thought"]:
            schema = get_conversation_schema(schema_type)
            assert schema is not None
            assert schema == Conversation

    def test_unsupported_conversation_type(self):
        """Test error handling for unsupported conversation types."""
        with pytest.raises(ValueError) as exc_info:
            get_conversation_schema("nonexistent_type")

        error_message = str(exc_info.value)
        assert "Unsupported conversation type" in error_message
        assert "nonexistent_type" in error_message
