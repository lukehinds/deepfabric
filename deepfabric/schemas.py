"""Pydantic schemas for structured conversation generation."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: str = Field(description="The content of the message")


class ChatTranscript(BaseModel):
    """A complete conversation transcript with messages."""

    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation", min_length=1
    )


class ReasoningStep(BaseModel):
    """A single step in a chain of reasoning."""

    step_number: int = Field(description="The step number in the reasoning chain")
    thought: str = Field(description="The reasoning or thought for this step")
    action: str | None = Field(
        default=None, description="Any action taken as part of this reasoning step"
    )


class StructuredConversation(BaseModel):
    """A conversation with optional structured reasoning and metadata."""

    messages: list[ChatMessage] = Field(
        description="List of messages in the conversation", min_length=1
    )
    reasoning_trace: list[ReasoningStep] | None = Field(
        default=None, description="Optional chain of reasoning steps"
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Optional metadata about the conversation"
    )


class FunctionCall(BaseModel):
    """A function call with arguments."""

    name: str = Field(description="The name of the function to call")
    arguments: dict[str, Any] = Field(description="Arguments to pass to the function")


class ToolMessage(BaseModel):
    """A message that includes tool/function calling."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="The role of the message sender"
    )
    content: str | None = Field(default=None, description="The text content of the message")
    function_call: FunctionCall | None = Field(
        default=None, description="Function call made by the assistant"
    )
    tool_calls: list[FunctionCall] | None = Field(
        default=None, description="Multiple tool calls made by the assistant"
    )


class ToolConversation(BaseModel):
    """A conversation that may include function/tool calls."""

    messages: list[ToolMessage] = Field(
        description="List of messages that may include tool calls", min_length=1
    )


# Topic generation schemas for tree and graph
class TopicList(BaseModel):
    """A list of subtopics for tree/graph generation."""

    subtopics: list[str] = Field(
        description="List of subtopic names",
        min_length=1,
    )


class TopicNode(BaseModel):
    """A topic node with subtopics for graph generation."""

    topic: str = Field(description="The topic name")
    subtopics: list[str] = Field(
        description="List of subtopic names",
        default_factory=list,
    )


class GraphSubtopic(BaseModel):
    """A subtopic with connections for graph generation."""

    topic: str = Field(description="The subtopic name")
    connections: list[int] = Field(
        description="List of existing node IDs to connect to, empty list if none"
    )


class GraphSubtopics(BaseModel):
    """List of subtopics with connections for graph generation."""

    subtopics: list[GraphSubtopic] = Field(
        description="List of subtopics with their connections",
        min_length=1,
    )


# Conversation type mapping for different generation modes
CONVERSATION_SCHEMAS = {
    "basic": ChatTranscript,
    "structured": StructuredConversation,
    "tool_calling": ToolConversation,
}


def get_conversation_schema(conversation_type: str = "basic") -> BaseModel:
    """Get the appropriate schema for a conversation type.

    Args:
        conversation_type: Type of conversation ("basic", "structured", "tool_calling")

    Returns:
        Pydantic model class for the conversation type

    Raises:
        ValueError: If conversation_type is not supported
    """
    if conversation_type not in CONVERSATION_SCHEMAS:
        raise ValueError(f"Unsupported conversation type: {conversation_type}")  # noqa: TRY003

    return CONVERSATION_SCHEMAS[conversation_type]
