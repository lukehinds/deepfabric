"""Pydantic schemas for structured conversation generation."""

import re

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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
    action: str = Field(description="Any action taken as part of this reasoning step")


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


# Chain of Thought schemas for reasoning-based dataset generation
class FreeTextCoT(BaseModel):
    """Chain of Thought dataset in free-text format (GSM8K style)."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    final_answer: str = Field(description="The definitive answer to the question")


class StructuredCoT(BaseModel):
    """Chain of Thought dataset with structured reasoning trace."""

    messages: list[ChatMessage] = Field(description="Conversation messages", min_length=1)
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")


class HybridCoT(BaseModel):
    """Chain of Thought dataset with both free-text and structured reasoning."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="The definitive answer to the question")


# Mathematical variants with numerical-only final answers
class MathematicalAnswerMixin:
    """Mixin class providing mathematical answer formatting and validation."""

    @classmethod
    def _format_mathematical_answer(cls, v: str) -> str:
        """Format mathematical answers with strict consistency rules."""
        v_stripped = v.strip()

        # Basic validation pattern
        pattern = r"^-?\d{1,3}(,\d{3})*(\.\d+)?([eE][+-]?\d+)?$|^-?\d+(\.\d+)?([eE][+-]?\d+)?$"
        if not re.match(pattern, v_stripped):
            msg = f"final_answer must be numerical, got: {v}"
            raise ValueError(msg)

        # Remove commas for processing
        v_clean = v_stripped.replace(",", "")

        # Apply formatting rules for consistency
        if cls._is_scientific_notation(v_clean):
            return v_clean  # Preserve scientific notation

        if "." in v_clean:
            decimal_parts = v_clean.split(".")
            if len(decimal_parts) == 2:  # noqa: PLR2004
                decimal_places = len(decimal_parts[1])
                # Round to 2 decimal places for precision artifacts
                if decimal_places >= 3:  # noqa: PLR2004
                    num = Decimal(v_clean)
                    rounded = num.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                    v_clean = str(rounded)
                # Ensure currency amounts always have 2 decimal places
                elif decimal_places == 1 and cls._looks_like_currency_value(v_clean):
                    v_clean = f"{v_clean}0"  # Add trailing zero
        # Handle integer values
        elif cls._is_clock_time_format(v_clean):
            v_clean = cls._format_clock_time_4digit(v_clean)
        elif cls._looks_like_large_currency(v_clean):
            v_clean = f"{v_clean}.00"  # Add .00 to large currency amounts

        return v_clean

    @staticmethod
    def _is_scientific_notation(value: str) -> bool:
        """Detect scientific notation."""
        return "e" in value.lower()

    @staticmethod
    def _looks_like_currency_value(value: str) -> bool:
        """Check if decimal value looks like currency (common decimal endings)."""
        try:
            num = float(value)
            # Currency typically >= $1 and ends in common decimal patterns
            return num >= 1.0 and value.split(".")[1] in ["0", "5", "25", "50", "75"]
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _is_clock_time_format(value: str) -> bool:
        """Check if integer value is likely a clock time (based on range and length)."""
        try:
            num = int(value)
        except ValueError:
            return False
        else:
            # Clock times are typically 3-4 digits in specific ranges
            # 600-2359 (6:00 AM to 11:59 PM) are most likely times
            # Exclude common counts/durations (1-180 minutes)
            return 600 <= num <= 2359 or num in [0, 100, 200, 300, 400, 500]  # noqa: PLR2004

    @staticmethod
    def _format_clock_time_4digit(value: str) -> str:
        """Format clock time as 4-digit HHMM (e.g., 600 -> 0600)."""
        try:
            num = int(value)
        except ValueError:
            return value
        else:
            return f"{num:04d}"  # Zero-pad to 4 digits

    @staticmethod
    def _looks_like_large_currency(value: str) -> bool:
        """Check if integer value is likely a large currency amount."""
        try:
            num = int(value)
        except ValueError:
            return False
        else:
            # Common currency amounts that should have .00
            return num >= 100 or num in [10, 15, 20, 25, 50, 75]  # noqa: PLR2004


class FreeTextCoTMathematical(BaseModel, MathematicalAnswerMixin):
    """Chain of Thought dataset in free-text format with numerical answer validation."""

    question: str = Field(description="The mathematical question or problem to solve")
    chain_of_thought: str = Field(description="Step-by-step mathematical reasoning")
    final_answer: str = Field(description="Numerical answer only (e.g., 42, 3.14, -17, 2.5e10)")

    @field_validator("final_answer")
    @classmethod
    def validate_numerical(cls, v: str) -> str:
        """Validate and format numerical answers with strict consistency rules.

        Formatting rules:
        - Currency amounts: always 2 decimal places (e.g., "104.00", "6.50")
        - Time durations: integers only (e.g., "90" for 90 minutes)
        - Clock times: 24-hour format as integers (e.g., "1615" for 4:15 PM)
        - Counts/quantities: integers (e.g., "42", "100")
        - Scientific values: preserve precision (e.g., "3.14", "2.5e10")
        """
        return cls._format_mathematical_answer(v)


class StructuredCoTMathematical(BaseModel, MathematicalAnswerMixin):
    """Chain of Thought dataset with structured reasoning and numerical answer validation."""

    messages: list[ChatMessage] = Field(description="Conversation messages", min_length=1)
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="Numerical answer only (e.g., 42, 3.14, -17)")

    @field_validator("final_answer")
    @classmethod
    def validate_numerical(cls, v: str) -> str:
        """Validate and format numerical answers with strict consistency rules."""
        return cls._format_mathematical_answer(v)


class HybridCoTMathematical(BaseModel, MathematicalAnswerMixin):
    """Chain of Thought dataset with hybrid reasoning and numerical answer validation."""

    question: str = Field(description="The mathematical question or problem to solve")
    chain_of_thought: str = Field(description="Natural language mathematical reasoning")
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    final_answer: str = Field(description="Numerical answer only (e.g., 42, 3.14, -17)")

    @field_validator("final_answer")
    @classmethod
    def validate_numerical(cls, v: str) -> str:
        """Validate and format numerical answers with strict consistency rules."""
        return cls._format_mathematical_answer(v)


# Conversation type mapping for different generation modes
CONVERSATION_SCHEMAS = {
    "basic": ChatTranscript,
    "structured": StructuredConversation,
    "tool_calling": ToolConversation,
    "cot_freetext": FreeTextCoT,
    "cot_structured": StructuredCoT,
    "cot_hybrid": HybridCoT,
}


def get_conversation_schema(
    conversation_type: str = "basic",
    reasoning_style: str = "general",
) -> type[BaseModel]:
    """Get the appropriate schema for a conversation type.

    Args:
        conversation_type: Type of conversation (basic, structured, tool_calling,
                          cot_freetext, cot_structured, cot_hybrid)
        reasoning_style: Style of reasoning (mathematical, logical, general)

    Returns:
        Pydantic model class for the conversation type

    Raises:
        ValueError: If conversation_type is not supported
    """
    if conversation_type not in CONVERSATION_SCHEMAS:
        valid_types = ", ".join(CONVERSATION_SCHEMAS.keys())
        msg = f"Unsupported conversation type: {conversation_type}. Valid types: {valid_types}"
        raise ValueError(msg)

    # Return mathematical variant for CoT types with mathematical reasoning
    if reasoning_style == "mathematical" and conversation_type.startswith("cot_"):
        mathematical_schemas = {
            "cot_freetext": FreeTextCoTMathematical,
            "cot_structured": StructuredCoTMathematical,
            "cot_hybrid": HybridCoTMathematical,
        }
        return mathematical_schemas.get(conversation_type, CONVERSATION_SCHEMAS[conversation_type])

    return CONVERSATION_SCHEMAS[conversation_type]
