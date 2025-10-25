"""
Tests for modular conversation configuration system.

This module tests the new modular configuration architecture where
conversation types, reasoning styles, agent modes, and output formats
are separate, orthogonal concerns that can be combined.
"""

import pytest

from deepfabric.config import DataEngineConfig


class TestModularConfigValidation:
    """Test validation rules for modular configuration."""

    def test_chain_of_thought_requires_reasoning_style(self):
        """Test that chain_of_thought requires reasoning_style to be set."""
        with pytest.raises(ValueError, match="reasoning_style must be specified"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="chain_of_thought",
                # Missing reasoning_style
            )

    def test_reasoning_style_only_with_chain_of_thought(self):
        """Test that reasoning_style can only be set with chain_of_thought."""
        with pytest.raises(ValueError, match="reasoning_style can only be set"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="basic",
                reasoning_style="freetext",  # Invalid for basic type
            )

    def test_agent_mode_requires_tools(self):
        """Test that agent_mode requires tools to be configured."""
        with pytest.raises(ValueError, match="agent_mode requires tools"):
            DataEngineConfig(
                generation_system_prompt="Test",
                provider="test",
                model="model",
                conversation_type="chain_of_thought",
                reasoning_style="hybrid",
                agent_mode="single_turn",
                # Missing tools configuration
            )


class TestModularConfigCombinations:
    """Test valid combinations of modular configuration options."""

    def test_basic_conversation(self):
        """Test basic conversation type (no reasoning, no agent)."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="basic",
        )

        assert config.conversation_type == "basic"
        assert config.reasoning_style is None
        assert config.agent_mode is None

    def test_chain_of_thought_freetext(self):
        """Test chain_of_thought with freetext reasoning."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="freetext",
        )

        assert config.conversation_type == "chain_of_thought"
        assert config.reasoning_style == "freetext"
        assert config.agent_mode is None

    def test_chain_of_thought_with_agent_single_turn(self):
        """Test chain_of_thought + agent_mode=single_turn."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="structured",
            agent_mode="single_turn",
            available_tools=["get_weather", "calculate"],
        )

        assert config.conversation_type == "chain_of_thought"
        assert config.reasoning_style == "structured"
        assert config.agent_mode == "single_turn"
        assert "get_weather" in config.available_tools
        assert "calculate" in config.available_tools

    def test_chain_of_thought_agent_multi_turn(self):
        """Test full combination: CoT + agent + multi_turn."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="hybrid",
            agent_mode="multi_turn",
            available_tools=["tool1", "tool2"],
        )

        assert config.conversation_type == "chain_of_thought"
        assert config.reasoning_style == "hybrid"
        assert config.agent_mode == "multi_turn"
        assert len(config.available_tools) == 2  # noqa: PLR2004

    def test_basic_conversation_explicit(self):
        """Test explicitly setting basic conversation type."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="basic",
        )

        assert config.conversation_type == "basic"
        assert config.reasoning_style is None
        assert config.agent_mode is None


class TestModularConfigDefaultValues:
    """Test default values for modular configuration fields."""

    def test_default_max_tools_per_query(self):
        """Test that max_tools_per_query has a default value."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="freetext",
            agent_mode="single_turn",
            available_tools=["tool1"],
        )

        assert config.max_tools_per_query == 3  # noqa: PLR2004

    def test_tools_default_to_empty_lists(self):
        """Test that tool-related fields default to empty lists."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="basic",
        )

        assert config.available_tools == []
        assert config.custom_tools == []


class TestReasoningStyleOptions:
    """Test all reasoning style options."""

    def test_freetext_reasoning(self):
        """Test freetext reasoning style."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="freetext",
        )

        assert config.reasoning_style == "freetext"

    def test_structured_reasoning(self):
        """Test structured reasoning style."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="structured",
        )

        assert config.reasoning_style == "structured"

    def test_hybrid_reasoning(self):
        """Test hybrid reasoning style."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="hybrid",
        )

        assert config.reasoning_style == "hybrid"


class TestAgentModeOptions:
    """Test agent mode options with tools."""

    def test_single_turn_agent(self):
        """Test single_turn agent mode."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="freetext",
            agent_mode="single_turn",
            available_tools=["tool1"],
        )

        assert config.agent_mode == "single_turn"

    def test_multi_turn_agent(self):
        """Test multi_turn agent mode."""
        config = DataEngineConfig(
            generation_system_prompt="Test",
            provider="test",
            model="model",
            conversation_type="chain_of_thought",
            reasoning_style="hybrid",
            agent_mode="multi_turn",
            available_tools=["tool1", "tool2"],
        )

        assert config.agent_mode == "multi_turn"
