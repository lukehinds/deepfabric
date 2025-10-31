import logging

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from .progress import ProgressReporter
from .schemas import ChatMessage, Conversation

if TYPE_CHECKING:
    from .generator import DataSetGeneratorConfig
    from .llm import LLMClient
    from .schemas import ToolRegistry

logger = logging.getLogger(__name__)


class ConversationBuilder(ABC):
    """Abstract base class for conversation builders.

    Each builder implements a specific strategy for generating conversations.
    Builders receive typed configuration and dependencies via constructor.

    Attributes:
        llm: LLM client for generation
        config: Typed configuration for the generator
        tool_registry: Optional tool registry for tool-calling conversations
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "DataSetGeneratorConfig",
        tool_registry: "ToolRegistry | None" = None,
        progress_reporter: ProgressReporter | None = None,
    ):
        """Initialize the conversation builder.

        Args:
            llm: LLM client for making generation requests
            config: Generator configuration (must be Pydantic model)
            tool_registry: Optional tool registry for tool-calling
            progress_reporter: Optional progress reporter for streaming feedback
        """
        self.llm = llm
        self.config = config
        self.tool_registry = tool_registry
        self.progress_reporter = progress_reporter

    @abstractmethod
    async def generate(self, topic_prompt: str) -> Conversation:
        """Generate a complete conversation.

        Args:
            topic_prompt: The topic/scenario prompt to generate conversation about

        Returns:
            Complete Conversation object (Pydantic model)

        Raises:
            ValueError: If generation fails validation
        """
        pass


class BuilderType(BaseModel):
    """Type discriminator for builder selection.

    This model ensures type-safe builder selection based on configuration.
    """

    name: str = Field(description="Builder type name")
    requires_tools: bool = Field(default=False, description="Whether this builder requires tools")

    class Config:
        frozen = True


# Builder type constants
SINGLE_SHOT_BUILDER = BuilderType(name="single_shot", requires_tools=False)
SINGLE_TURN_AGENT_BUILDER = BuilderType(name="single_turn_agent", requires_tools=True)
MULTI_TURN_AGENT_BUILDER = BuilderType(name="multi_turn_agent", requires_tools=True)


def determine_builder_type(config: "DataSetGeneratorConfig") -> BuilderType:
    """Determine the appropriate builder type from configuration.

    Args:
        config: Generator configuration (Pydantic model)

    Returns:
        BuilderType indicating which builder to use

    Raises:
        ValueError: If configuration is invalid or unsupported
    """
    # Agent mode with tools requires specialized builder
    if config.agent_mode:
        # Check that tools are configured in some way
        has_tools = config.tool_registry_path or config.available_tools or config.custom_tools
        if not has_tools:
            msg = (
                "agent_mode requires tools to be configured via tool_registry_path, "
                "available_tools, or custom_tools"
            )
            raise ValueError(msg)

        if config.agent_mode == "multi_turn":
            return MULTI_TURN_AGENT_BUILDER
        if config.agent_mode == "single_turn":
            return SINGLE_TURN_AGENT_BUILDER
        msg = f"Unknown agent_mode: {config.agent_mode}"
        raise ValueError(msg)

    # Non-agent conversations use single-shot generation
    if config.conversation_type in ("basic", "chain_of_thought"):
        return SINGLE_SHOT_BUILDER

    msg = f"Cannot determine builder type for conversation_type={config.conversation_type}"
    raise ValueError(msg)


class SingleShotBuilder(ConversationBuilder):
    """Builder for simple conversations using single-shot JSON generation.

    This builder generates the entire conversation in one LLM call using
    structured output with JSON schema validation. Suitable for:
    - Basic Q&A conversations
    - Chain-of-thought reasoning without tools
    - Any conversation that can be generated in one pass
    """

    async def generate(self, topic_prompt: str) -> Conversation:
        """Generate conversation using single LLM call with JSON schema.

        Args:
            topic_prompt: Topic or scenario to generate conversation about

        Returns:
            Complete Conversation object

        Raises:
            ValueError: If LLM fails to generate valid conversation
        """
        # Build the generation prompt
        generation_prompt = self._build_prompt(topic_prompt)

        # Use streaming if progress reporter is available
        conversation = None
        if self.progress_reporter:
            # Stream generation (no ProgressStep needed - generator emits sample-level steps)
            async for chunk, result in self.llm.generate_async_stream(
                prompt=generation_prompt,
                schema=Conversation,
                max_retries=self.config.max_retries,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk("conversation_gen", chunk)
                if result:
                    conversation = result
        else:
            # Fallback to non-streaming
            conversation = await self.llm.generate_async(
                prompt=generation_prompt,
                schema=Conversation,
                max_retries=self.config.max_retries,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        # Validate that generated conversation starts with user message
        # (system messages are added by builder, not generated by LLM)
        if conversation.messages and conversation.messages[0].role != "user":
            msg = (
                f"Generated conversation must start with 'user' message, got '{conversation.messages[0].role}'. "
                "System messages are added automatically by the builder."
            )
            raise ValueError(msg)

        # Insert system message if configured
        if self.config.sys_msg:
            conversation.messages.insert(
                0,
                ChatMessage(role="system", content=self.config.dataset_system_prompt or ""),
            )

        return conversation

    def _build_prompt(self, topic_prompt: str) -> str:
        """Build the generation prompt for single-shot generation.

        Args:
            topic_prompt: The topic to generate about

        Returns:
            Complete prompt string for the LLM
        """
        # Use the generation system prompt as the base
        prompt_parts = [self.config.generation_system_prompt]

        # Add topic/scenario
        prompt_parts.append(f"\nTopic/Scenario: {topic_prompt}")

        # Add any additional instructions
        if self.config.instructions:
            prompt_parts.append(f"\nAdditional Instructions: {self.config.instructions}")

        # Add reasoning-specific guidance based on style
        if self.config.conversation_type == "chain_of_thought":
            if self.config.reasoning_style == "freetext":
                prompt_parts.append(
                    "\nREASONING FORMAT: Generate natural, conversational reasoning content (string format). "
                    "Show your actual thinking process - explore ideas, consider alternatives, work through the problem. "
                    "Think like a human would: 'Hmm, let me think about this...', 'Wait, that doesn't work...', "
                    "'Actually, if I approach it this way...'. "
                    "DO NOT use numbered steps or structured outlines. "
                    "Use the 'content' field in reasoning as a plain string (not a list)."
                )
            elif self.config.reasoning_style == "structured":
                prompt_parts.append(
                    "\nREASONING FORMAT: Generate structured reasoning steps as a list of ReasoningStep objects. "
                    "Each step should have clear thought and action fields."
                )

        # Add explicit structure requirement
        prompt_parts.append(
            "\nIMPORTANT: Generate the conversation messages array starting with a 'user' message "
            "(the user's question or request), followed by an 'assistant' message (the response). "
            "Do NOT include any 'system' role messages - those are added separately."
        )

        return "\n".join(prompt_parts)


class ConversationBuilderFactory:
    """Factory for creating conversation builders.

    Provides type-safe builder instantiation based on configuration.
    """

    @staticmethod
    def create(
        config: "DataSetGeneratorConfig",
        llm: "LLMClient",
        tool_registry: "ToolRegistry | None" = None,
        progress_reporter: ProgressReporter | None = None,
    ) -> ConversationBuilder:
        """Create the appropriate conversation builder.

        Args:
            config: Generator configuration (Pydantic model)
            llm: LLM client for generation
            tool_registry: Optional tool registry (required for agent builders)
            progress_reporter: Optional progress reporter for streaming feedback

        Returns:
            Appropriate ConversationBuilder instance

        Raises:
            ValueError: If configuration is invalid or builder requirements not met
        """
        builder_type = determine_builder_type(config)

        # Validate tool registry requirement
        if builder_type.requires_tools and tool_registry is None:
            msg = (
                f"Builder type '{builder_type.name}' requires tool_registry but it was not provided"
            )
            raise ValueError(msg)

        # Instantiate appropriate builder
        if builder_type == SINGLE_SHOT_BUILDER:
            return SingleShotBuilder(llm, config, progress_reporter=progress_reporter)
        if builder_type == SINGLE_TURN_AGENT_BUILDER:
            from .builders_agent import SingleTurnAgentBuilder  # noqa: PLC0415

            return SingleTurnAgentBuilder(
                llm, config, cast("ToolRegistry", tool_registry), progress_reporter
            )
        if builder_type == MULTI_TURN_AGENT_BUILDER:
            from .builders_agent import MultiTurnAgentBuilder  # noqa: PLC0415

            return MultiTurnAgentBuilder(
                llm, config, cast("ToolRegistry", tool_registry), progress_reporter
            )
        msg = f"Unknown builder type: {builder_type.name}"
        raise ValueError(msg)
