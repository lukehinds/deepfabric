from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from ..schemas import Conversation


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant", "function", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: str | None = Field(default=None, description="The content of the message")
    tool_calls: list[dict] | None = Field(default=None, description="Tool calls made by assistant")

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v, info):
        # Allow empty content if tool_calls are present (for tool-calling messages)
        tool_calls = info.data.get("tool_calls")
        if tool_calls:
            return v

        # Allow empty content for assistant and tool roles (used in tool-calling flows)
        role = info.data.get("role")
        if role in ("assistant", "tool"):
            return v

        # For system and user roles, content must not be empty
        if v is None or not v.strip():
            raise ValueError("content cannot be empty or whitespace only")
        return v


class ConversationSample(BaseModel):
    """A dataset sample with conversation messages."""

    messages: list[Message] = Field(..., description="list of conversation messages")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class QASample(BaseModel):
    """A dataset sample with question-answer structure."""

    question: str = Field(..., min_length=1, description="The question or prompt")
    answer: str | None = Field(None, description="The answer or response")
    final_answer: str | None = Field(None, description="Alternative field for final answer")
    chain_of_thought: str | None = Field(None, description="Reasoning process")
    context: str | None = Field(None, description="Additional context")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")

    @field_validator("final_answer")
    @classmethod
    def must_have_answer(cls, v, info):
        answer = info.data.get("answer")
        if not v and not answer:
            raise ValueError("Must have either answer or final_answer")
        return v


class InstructionSample(BaseModel):
    """A dataset sample with instruction-following structure."""

    instruction: str = Field(..., min_length=1, description="The instruction or task")
    input: str | None = Field(None, description="Optional input context")
    output: str = Field(..., min_length=1, description="The expected output")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class GenericSample(BaseModel):
    """A generic dataset sample that can contain any fields."""

    data: dict[str, Any] = Field(..., description="The sample data")

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


class FormatterConfigModel(BaseModel):
    """Configuration for a formatter instance."""

    name: str = Field(..., min_length=1, description="Unique name for this formatter instance")
    template: str = Field(..., min_length=1, description="Template path (builtin:// or file://)")
    config: dict[str, Any] = Field(
        default_factory=dict, description="Formatter-specific configuration"
    )
    output: str | None = Field(None, description="Output file path for formatted dataset")

    @field_validator("template")
    @classmethod
    def validate_template_format(cls, v):
        if not (v.startswith("builtin://") or v.startswith("file://")):
            raise ValueError('Template must start with "builtin://" or "file://"')
        return v


class AlpacaOutput(BaseModel):
    """Pydantic model for Alpaca formatter output."""

    instruction: str = Field(..., min_length=1, description="The instruction or task")
    input: str | None = Field(None, description="Optional input context")
    output: str = Field(..., min_length=1, description="The expected response")


class GrpoOutput(BaseModel):
    """Pydantic model for GRPO formatter output."""

    messages: list[Message] = Field(..., description="Conversation messages with GRPO formatting")

    @field_validator("messages")
    @classmethod
    def messages_must_have_at_least_one(cls, v):
        if not v or len(v) < 1:
            raise ValueError("messages must contain at least one item")
        return v


class ChatmlStructuredOutput(BaseModel):
    """Pydantic model for ChatML structured output."""

    messages: list[Message] = Field(..., description="Conversation messages")

    @field_validator("messages")
    @classmethod
    def messages_must_have_at_least_one(cls, v):
        if not v or len(v) < 1:
            raise ValueError("messages must contain at least one item")
        return v


class ChatmlTextOutput(BaseModel):
    """Pydantic model for ChatML text output."""

    text: str = Field(..., min_length=1, description="ChatML formatted text")


class FormatterMetadata(BaseModel):
    """Metadata for formatted samples."""

    formatter_name: str = Field(..., description="Name of the formatter used")
    formatter_version: str = Field(..., description="Version of the formatter")
    original_format: str | None = Field(None, description="Original input format detected")
    processing_timestamp: str | None = Field(None, description="When the formatting was applied")
    validation_passed: bool = Field(True, description="Whether output validation passed")


class FormatterStats(BaseModel):
    """Statistics about formatter processing."""

    total_samples: int = Field(..., ge=0, description="Total number of input samples")
    processed_samples: int = Field(
        ..., ge=0, description="Number of successfully processed samples"
    )
    failed_samples: int = Field(..., ge=0, description="Number of failed samples")
    skipped_samples: int = Field(..., ge=0, description="Number of skipped samples")
    processing_time_seconds: float = Field(..., ge=0, description="Total processing time")

    @property
    def success_rate(self) -> float:
        """Calculate the success rate as a percentage."""
        if self.total_samples == 0:
            return 0.0
        return (self.processed_samples / self.total_samples) * 100


class ValidationResult(BaseModel):
    """Result of formatter validation."""

    is_valid: bool = Field(..., description="Whether the sample passed validation")
    errors: list[str] = Field(default_factory=list, description="list of validation errors")
    warnings: list[str] = Field(default_factory=list, description="list of validation warnings")


class FormatterResult(BaseModel):
    """Result of formatter processing."""

    samples: list[dict[str, Any]] = Field(..., description="Formatted samples")
    metadata: FormatterMetadata = Field(..., description="Formatter metadata")
    stats: FormatterStats = Field(..., description="Processing statistics")
    errors: list[str] = Field(default_factory=list, description="Processing errors")


# Configuration models for specific formatters


class GrpoConfig(BaseModel):
    """Configuration for GRPO formatter."""

    reasoning_start_tag: str = Field(
        default="<start_working_out>", description="Start tag for reasoning"
    )
    reasoning_end_tag: str = Field(default="<end_working_out>", description="End tag for reasoning")
    solution_start_tag: str = Field(default="<SOLUTION>", description="Start tag for solution")
    solution_end_tag: str = Field(default="</SOLUTION>", description="End tag for solution")
    system_prompt: str | None = Field(None, description="Custom system prompt")
    validate_numerical: bool = Field(
        default=True, description="Whether to validate numerical answers"
    )


class AlpacaConfig(BaseModel):
    """Configuration for Alpaca formatter."""

    instruction_field: str = Field(default="instruction", description="Field name for instructions")
    input_field: str = Field(default="input", description="Field name for input")
    output_field: str = Field(default="output", description="Field name for output")
    include_empty_input: bool = Field(
        default=True, description="Whether to include empty input fields"
    )
    instruction_template: str | None = Field(
        None, description="Template for instruction formatting"
    )


class ChatmlConfig(BaseModel):
    """Configuration for ChatML formatter."""

    start_token: str = Field(default="<|im_start|>", description="Start token for messages")
    end_token: str = Field(default="<|im_end|>", description="End token for messages")
    output_format: Literal["structured", "text"] = Field(
        default="structured", description="Output format"
    )
    default_system_message: str = Field(
        default="You are a helpful assistant.", description="Default system message"
    )
    require_system_message: bool = Field(
        default=False, description="Whether to require system message"
    )
    normalize_whitespace: bool = Field(
        default=True,
        description="Remove excessive blank lines and normalize whitespace in message content",
    )


# Union type for all possible sample formats
DatasetSample = (
    ConversationSample
    | QASample
    | InstructionSample
    | Conversation  # Unified schema with optional capability fields
    | GenericSample
)


class DatasetInput(BaseModel):
    """Input dataset containing a list of samples."""

    samples: list[DatasetSample] = Field(..., min_length=0, description="list of dataset samples")

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class FormattedOutput(BaseModel):
    """Base class for formatted output samples."""

    class Config:
        extra = "allow"  # Allow additional fields for flexibility


class DatasetOutput(BaseModel):
    """Output dataset containing formatted samples."""

    samples: list[FormattedOutput] = Field(
        ..., min_length=0, description="list of formatted samples"
    )

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


# Additional models for handling structured CoT and unified sample processing


class ReasoningStep(BaseModel):
    """A single step in a structured reasoning trace."""

    step_number: int | None = None
    thought: str = ""
    action: str = ""


class StructuredCoTSample(BaseModel):
    """Sample format for structured Chain of Thought reasoning."""

    messages: list[Message]
    reasoning_trace: list[ReasoningStep] | str
    final_answer: str

    def get_reasoning_text(self) -> str:
        """Extract reasoning as text from the trace."""
        if isinstance(self.reasoning_trace, str):
            return self.reasoning_trace

        parts = []
        for step in self.reasoning_trace:
            if step.thought:
                parts.append(step.thought)
            if step.action and step.action != step.thought:
                parts.append(step.action)
        return " ".join(parts) if parts else ""

    def has_assistant_message(self) -> bool:
        """Check if there's an assistant message."""
        return any(msg.role == "assistant" for msg in self.messages)

    def create_assistant_content(
        self, reasoning_start: str, reasoning_end: str, solution_start: str, solution_end: str
    ) -> str:
        """Create formatted assistant content with reasoning and solution tags."""
        reasoning = self.get_reasoning_text()
        if not reasoning:
            reasoning = "Let me solve this step by step."

        return f"{reasoning_start}{reasoning}{reasoning_end}{solution_start}{self.final_answer}{solution_end}"


class HarmonyConfig(BaseModel):
    """Configuration for Harmony format formatter."""

    start_token: str = Field(
        default="<|start|>", description="Token marking the start of a message"
    )
    end_token: str = Field(default="<|end|>", description="Token marking the end of a message")
    message_token: str = Field(
        default="<|message|>", description="Token separating role from content"
    )
    output_format: Literal["text", "structured"] = Field(
        default="text", description="Output format: text with tokens or structured messages"
    )
    default_channel: Literal["final", "analysis", "commentary"] = Field(
        default="final", description="Default channel for messages without explicit channel"
    )
    include_developer_role: bool = Field(
        default=False, description="Whether to include developer role in output"
    )
    developer_instructions: str | None = Field(
        None, description="Optional developer instructions to include"
    )
    system_message: str = Field(
        default="You are a large language model trained by OpenAI.",
        description="Default system message",
    )
    reasoning_level: Literal["none", "low", "medium", "high"] = Field(
        default="high", description="Reasoning effort level"
    )
    knowledge_cutoff: str | None = Field(
        default="2024-01", description="Knowledge cutoff date (YYYY-MM format)"
    )
    current_date: str | None = Field(
        default=None,
        description="Current date for system message (YYYY-MM-DD format). If not provided, no date is included.",
    )
    include_metadata: bool = Field(
        default=True, description="Whether to include metadata in system message"
    )
    tool_namespace: str = Field(
        default="functions", description="Namespace for tool/function definitions"
    )


class HarmonyMessage(BaseModel):
    """A single message in Harmony format."""

    role: Literal["system", "developer", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: str = Field(..., description="The content of the message")
    channel: Literal["final", "analysis", "commentary"] | None = Field(
        None, description="The channel for assistant messages"
    )
    recipient: str | None = Field(None, description="Tool recipient for tool calls")

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("content cannot be empty or whitespace only")
        return v


class HarmonyStructuredOutput(BaseModel):
    """Structured output for Harmony format."""

    messages: list[HarmonyMessage] = Field(..., description="List of Harmony messages")
    metadata: dict[str, Any] | None = Field(None, description="Optional metadata")


class HarmonyTextOutput(BaseModel):
    """Text output for Harmony format with special tokens."""

    text: str = Field(..., description="Formatted text with Harmony tokens")


class UnifiedSample(BaseModel):
    """Unified model that can handle multiple input formats with type-safe detection."""

    data: dict[str, Any]

    def detect_format(self) -> Literal["structured_cot", "messages", "qa", "generic", "unknown"]:
        """Detect the format of the sample with type safety."""
        # Check for structured CoT format
        if (
            "messages" in self.data
            and "reasoning_trace" in self.data
            and "final_answer" in self.data
        ):
            return "structured_cot"

        # Check for messages format
        if "messages" in self.data:
            messages = self.data["messages"]
            if isinstance(messages, list) and messages:
                return "messages"

        # Check for Q&A format
        if "question" in self.data and ("answer" in self.data or "final_answer" in self.data):
            return "qa"

        # Check for generic format with recognizable fields
        question_fields = ["question", "prompt", "problem", "input", "instruction"]
        answer_fields = ["answer", "output", "response", "solution", "final_answer"]

        has_question = any(field in self.data for field in question_fields)
        has_answer = any(field in self.data for field in answer_fields)

        if has_question or has_answer:
            return "generic"

        return "unknown"

    def as_structured_cot(self) -> StructuredCoTSample | None:
        """Try to parse as structured CoT sample."""
        try:
            return StructuredCoTSample(**self.data)
        except (ValidationError, TypeError) as e:
            print(f"Error parsing structured CoT sample: {e}")
            return None

    def as_conversation(self) -> ConversationSample | None:
        """Try to parse as conversation sample."""
        try:
            return ConversationSample(**self.data)
        except (ValidationError, TypeError) as e:
            print(f"Error parsing conversation sample: {e}")
            return None

    def as_qa(self) -> QASample | None:
        """Try to parse as Q&A sample."""
        try:
            return QASample(**self.data)
        except (ValidationError, TypeError) as e:
            print(f"Error parsing Q&A sample: {e}")
            return None

    def as_generic(self) -> GenericSample:
        """Parse as generic sample (always succeeds)."""
        return GenericSample(data=self.data)
