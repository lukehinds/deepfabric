import json
import logging
import re

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# Basic message schema
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


# Tool definition schemas for structured tool system
class ToolParameter(BaseModel):
    """A single parameter for a tool/function."""

    name: str = Field(description="Parameter name")
    type: Literal["str", "int", "float", "bool", "list", "dict"] = Field(
        description="Parameter type"
    )
    description: str = Field(description="What this parameter does")
    required: bool = Field(default=True, description="Whether this parameter is required")
    default: Any = Field(default=None, description="Default value if not provided")


class ToolDefinition(BaseModel):
    """Complete definition of a tool/function."""

    name: str = Field(description="Tool name (function name)")
    description: str = Field(description="What this tool does")
    parameters: list[ToolParameter] = Field(description="List of parameters this tool accepts")
    returns: str = Field(description="Description of what this tool returns")
    category: str = Field(default="general", description="Tool category for grouping")

    def to_signature(self) -> str:
        """Generate a function signature string."""
        params = []
        for p in self.parameters:
            if p.required:
                params.append(f"{p.name}: {p.type}")
            else:
                params.append(f"{p.name}: {p.type} = {p.default}")
        return f"{self.name}({', '.join(params)}) → {self.returns}"

    def to_openai_schema(self) -> dict[str, Any]:
        """
        Convert tool definition to OpenAI function calling schema format.

        This format is compatible with TRL's SFTTrainer and other HuggingFace
        training frameworks that support tool/function calling.

        Returns:
            Dictionary in OpenAI function calling schema format with:
            - type: Always "function"
            - function: Object containing name, description, and parameters schema
        """
        # Map DeepFabric types to JSON Schema types
        type_mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
        }

        properties = {}
        required = []

        for param in self.parameters:
            json_type = type_mapping.get(param.type, "string")
            properties[param.name] = {
                "type": json_type,
                "description": param.description,
            }

            # Add default value if present and not required
            if not param.required and param.default is not None:
                properties[param.name]["default"] = param.default

            # Track required parameters
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry(BaseModel):
    """Registry of available tools."""

    tools: list[ToolDefinition] = Field(description="List of available tool definitions")

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return next((t for t in self.tools if t.name == name), None)

    def get_tools_by_category(self, category: str) -> list[ToolDefinition]:
        """Get all tools in a category."""
        return [t for t in self.tools if t.category == category]

    def get_tool_names(self) -> list[str]:
        """Get list of all tool names."""
        return [t.name for t in self.tools]

    def to_trl_format(self) -> list[dict[str, Any]]:
        """
        Convert all tools to TRL/OpenAI function calling schema format.

        This method is specifically designed for use with HuggingFace TRL's
        SFTTrainer and other training frameworks that require tools to be
        provided in OpenAI function calling format.

        Returns:
            List of tool definitions in OpenAI function calling schema format.
            Each tool includes type="function" and a function object with
            name, description, and parameters.

        Example:
            >>> registry = ToolRegistry(tools=[...])
            >>> trl_tools = registry.to_trl_format()
            >>> # Use in dataset: {"messages": [...], "tools": trl_tools}
        """
        return [tool.to_openai_schema() for tool in self.tools]


# Agent tool-calling schemas
class ToolReasoningStep(BaseModel):
    """A reasoning step that leads to tool selection and parameter construction."""

    step_number: int = Field(description="The step number in the tool planning sequence")
    reasoning: str = Field(description="Why this tool is needed at this point")
    selected_tool: ToolDefinition = Field(description="The actual tool definition being selected")
    parameter_reasoning: dict[str, str] = Field(description="Reasoning for each parameter value")
    expected_result: str = Field(description="What the tool should return and how it helps")


class ToolExecution(BaseModel):
    """Represents actual execution of a tool with reasoning context."""

    function: str = Field(description="Name of the function/tool being called")
    arguments: dict[str, Any] = Field(description="Arguments passed to the function")
    reasoning: str = Field(description="Brief explanation of why executing now")
    result: str = Field(description="The result returned from the tool execution")


class SimpleAgentCoT(BaseModel):
    """Simplified Agent Chain of Thought that models can actually generate."""

    question: str = Field(description="The user's question or request")
    initial_analysis: str = Field(description="Initial understanding of what's needed")
    reasoning_steps: list[str] = Field(description="Step-by-step reasoning", min_length=1)
    tool_selection_rationale: str = Field(description="Why this tool was chosen")
    parameter_reasoning: str = Field(description="How tool parameters were determined")
    result_interpretation: str = Field(description="What the tool result means")
    tool_used: str = Field(description="Name of the tool that was used")
    tool_input: str = Field(description="JSON string of tool input parameters")
    tool_output: str = Field(description="The result from the tool execution")
    answer: str = Field(description="Final answer to the user's question")


class HybridAgentCoT(BaseModel):
    """Hybrid Agent CoT with structured reasoning trace and tool calling."""

    question: str = Field(description="The question or problem to solve")
    chain_of_thought: str = Field(description="Natural language reasoning explanation")
    reasoning_trace: list[ReasoningStep] = Field(
        description="Structured reasoning steps", min_length=1
    )
    tool_selection_rationale: str = Field(description="Why this tool was chosen")
    parameter_reasoning: str = Field(description="How tool parameters were determined")
    result_interpretation: str = Field(description="What the tool result means")
    tool_used: str = Field(description="Name of the tool that was used")
    tool_input: str = Field(description="JSON string of tool input parameters")
    tool_output: str = Field(description="The result from the tool execution")
    final_answer: str = Field(description="The definitive answer to the question")


class AgentCoTMultiTurn(BaseModel):
    """Multi-turn agent conversation with tool calling and reasoning."""

    messages: list[ChatMessage] = Field(
        description="Conversation messages between user and agent", min_length=1
    )
    tool_planning_trace: list[ToolReasoningStep] = Field(
        description="Complete planning trace for all tool usage", min_length=1
    )
    tool_execution_trace: list[ToolExecution] = Field(
        description="All tool executions across conversation turns", min_length=1
    )
    reasoning_summary: str = Field(
        description="Overall reasoning strategy across the multi-turn conversation"
    )


# XLAM 2.0 (APIGen-MT) Multi-Turn Agent Schemas
class XlamConversationTurn(BaseModel):
    """A single turn in an XLAM 2.0 multi-turn conversation."""

    from_: Literal["human", "gpt", "function_call", "observation"] = Field(
        alias="from", description="The speaker/actor for this turn"
    )
    value: str = Field(description="The content of this turn")

    class Config:
        populate_by_name = True
        extra = "forbid"


class XlamFunctionCall(BaseModel):
    """Structured function call for XLAM format."""

    name: str = Field(description="Function name to call")
    arguments: dict[str, Any] = Field(description="Function arguments as key-value pairs")

    class Config:
        extra = "forbid"


class XlamMultiTurnAgent(BaseModel):
    """
    XLAM 2.0 multi-turn agent conversation with function calling.

    This schema generates the core conversation structure that will be
    formatted into XLAM 2.0 (APIGen-MT) format by the XlamV2Formatter.

    Expected flow patterns:
    1. human → gpt → function_call → observation → gpt (tool use)
    2. human → gpt → human → gpt (clarification)
    3. human → function_call → observation → gpt (direct execution)
    """

    # Core conversation turns
    turns: list[XlamConversationTurn] = Field(
        min_length=3,
        max_length=15,
        description="Conversation turns in sequence (human, gpt, function_call, observation)",
    )

    # Metadata for context (not in final XLAM output, but useful for generation)
    scenario_description: str = Field(
        description="Brief description of the scenario/domain context"
    )
    # Domain-specific system prompt (becomes 'system' field in XLAM output)
    domain_policy: str = Field(
        description="Detailed domain-specific policy, rules, and guidelines for this scenario (e.g., airline booking policy, e-commerce return policy). This should be unique and detailed for each scenario.",
    )
    # Note: Made required for OpenAI structured output compatibility (OpenAI requires all fields in 'required' array)
    planning_notes: str = Field(
        description="Internal reasoning about conversation flow and tool usage strategy (can be empty string if not needed)",
    )

    class Config:
        extra = "forbid"

    def get_function_calls(self) -> list[dict[str, Any]]:
        """Extract all function calls from the conversation."""

        calls = []
        for turn in self.turns:
            if turn.from_ == "function_call":
                try:
                    call_data = json.loads(turn.value)
                    calls.append(call_data)
                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to decode JSON for function call. Value: %s, Error: %s",
                        turn.value[:100],  # Log first 100 chars to avoid huge logs
                        str(e),
                    )
                    continue
        return calls

    def validate_conversation_flow(self) -> bool:
        """Validate that conversation follows logical turn patterns."""
        # Must start with human
        if not self.turns or self.turns[0].from_ != "human":
            return False

        # function_call must be followed by observation
        for i, turn in enumerate(self.turns[:-1]):
            if turn.from_ == "function_call" and self.turns[i + 1].from_ != "observation":
                return False

        return True


# Tool calling schemas for conversations that include function calls
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


# Chain of Thought schemas
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

        # Handle cases where model returns multiple answers (e.g., "2, 3")
        # Take the first one if comma-separated list detected
        if ", " in v_stripped:
            v_stripped = v_stripped.split(", ")[0].strip()

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

        return v_clean

    @staticmethod
    def _is_scientific_notation(value: str) -> bool:
        """Detect scientific notation."""
        return "e" in value.lower()


class FreeTextCoTMathematical(BaseModel, MathematicalAnswerMixin):
    """Chain of Thought dataset in free-text format with numerical answer validation."""

    question: str = Field(description="The mathematical question or problem to solve")
    chain_of_thought: str = Field(description="Step-by-step mathematical reasoning")
    final_answer: str = Field(description="Numerical answer only (e.g., 42, 3.14, -17, 2.5e10)")

    @field_validator("final_answer")
    @classmethod
    def validate_numerical(cls, v: str) -> str:
        """Validate and format numerical answers with strict consistency rules."""
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
    "agent_cot_tools": SimpleAgentCoT,
    "agent_cot_hybrid": HybridAgentCoT,
    "agent_cot_multi_turn": AgentCoTMultiTurn,
    "xlam_multi_turn": XlamMultiTurnAgent,  # XLAM 2.0 (APIGen-MT) format
}


def get_conversation_schema(
    conversation_type: str = "basic",
    reasoning_style: str = "general",
) -> type[BaseModel]:
    """Get the appropriate schema for a conversation type.

    Args:
        conversation_type: Type of conversation (basic, structured, tool_calling,
                          cot_freetext, cot_structured, cot_hybrid,
                          agent_cot_tools, agent_cot_multi_turn)
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


# Topic generation schemas for tree and graph (needed by other modules)
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


# Update the conversation schemas to include agent schemas
CONVERSATION_SCHEMAS.update(
    {
        "agent_cot_tools": SimpleAgentCoT,
        "agent_cot_hybrid": HybridAgentCoT,
        "agent_cot_multi_turn": AgentCoTMultiTurn,
    }
)
