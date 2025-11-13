import logging
import random

from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from .builders import ConversationBuilder
from .exceptions import DataSetGeneratorError
from .progress import ProgressReporter
from .schemas import (
    AgentContext,
    ChatMessage,
    Conversation,
    ReasoningStep,
    ReasoningTrace,
    ToolContext,
    ToolExecution,
)

if TYPE_CHECKING:
    from .generator import DataSetGeneratorConfig
    from .llm import LLMClient
    from .schemas import ToolRegistry

logger = logging.getLogger(__name__)


class UserQuestion(BaseModel):
    """User's question or request."""

    content: str = Field(
        description="The user's question or request text - just the question itself, nothing else",
        min_length=10,
        max_length=1000,
    )


class Scenario(BaseModel):
    """Multi-turn scenario description."""

    description: str = Field(
        description="Brief scenario description requiring multiple turns",
        min_length=20,
        max_length=500,
    )


class AgentResponse(BaseModel):
    """Agent's response to user."""

    content: str = Field(
        description="The agent's response text - clear and concise",
        min_length=10,
        max_length=2000,  # Prevent truncation at max_tokens limit
    )


class ToolOutput(BaseModel):
    """Simulated tool execution output."""

    result: str = Field(description="The tool's output/result", min_length=1)


class ConclusionDecision(BaseModel):
    """Decision on whether to conclude conversation."""

    should_conclude: bool = Field(
        description="True if conversation task is complete, False if more turns needed"
    )


class AgentTurnData(BaseModel):
    """Typed data for a single turn in an agent conversation.

    This model ensures type safety when building multi-turn conversations.
    """

    user_message: ChatMessage = Field(description="User's message for this turn")
    reasoning_steps: list[ReasoningStep] = Field(
        description="Agent's reasoning steps for this turn"
    )
    tool_calls: list[ToolExecution] = Field(description="Tool executions for this turn")
    agent_response: ChatMessage = Field(description="Agent's final response for this turn")


class SingleTurnAgentBuilder(ConversationBuilder):
    """Builder for single-turn agent conversations with tool calling.

    Generates conversations using a multi-step process:
    1. Generate user question
    2. Generate agent reasoning + tool calls
    3. Simulate tool execution results
    4. Generate agent's final response

    This produces realistic tool-calling training data.
    """

    def __init__(
        self,
        llm: "LLMClient",
        config: "DataSetGeneratorConfig",
        tool_registry: "ToolRegistry",
        progress_reporter: ProgressReporter | None = None,
    ):
        """Initialize with required tool registry.

        Args:
            llm: LLM client for generation
            config: Generator configuration
            tool_registry: Tool registry (required for agent builders)
            progress_reporter: Optional progress reporter for streaming feedback
        """
        super().__init__(llm, config, tool_registry, progress_reporter)
        # Store as non-optional for type checker
        self.tool_registry: ToolRegistry = tool_registry

    async def generate(self, topic_prompt: str) -> Conversation:
        """Generate single-turn agent conversation with tools.

        Args:
            topic_prompt: Topic or scenario to generate conversation about

        Returns:
            Complete Conversation with tool calling

        Raises:
            ValueError: If generation fails at any step
        """
        # Step 1: Generate user question
        user_message = await self._generate_user_question(topic_prompt)

        # Step 2: Generate agent reasoning + tool calls
        reasoning, tool_calls = await self._generate_agent_thinking(user_message)

        # Step 3: Simulate tool executions
        tool_results = await self._simulate_tool_results(tool_calls)

        # Step 4: Generate agent's final response
        agent_response = await self._generate_agent_conclusion(
            user_message, reasoning, tool_results
        )

        # Assemble into Conversation
        return self._build_conversation(
            user_message, reasoning, tool_calls, tool_results, agent_response, topic_prompt
        )

    async def _generate_user_question(self, topic_prompt: str) -> ChatMessage:
        """Generate the user's question for this scenario.

        Args:
            topic_prompt: The scenario topic

        Returns:
            User message (typed ChatMessage)
        """
        prompt = f"""Generate a short, natural user question for this scenario:
{topic_prompt}

Requirements:
- Just the user's question - no reasoning, no explanations, no examples
- Should require using tools to answer
- 1-2 sentences maximum
- Natural, conversational tone

Example format: "Can you tell me the weather in Paris tomorrow and suggest what to wear?"

Generate only the user's question:"""

        response: UserQuestion
        if self.progress_reporter:
            temp_response = None
            async for chunk, result in self.llm.generate_async_stream(
                prompt=prompt,
                schema=UserQuestion,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk("user_question", chunk)
                if result:
                    temp_response = result
            if temp_response is None:
                raise DataSetGeneratorError("Failed to generate user question")
            response = cast(UserQuestion, temp_response)
        else:
            response = await self.llm.generate_async(
                prompt=prompt,
                schema=UserQuestion,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        return ChatMessage(role="user", content=response.content)

    async def _generate_agent_thinking(
        self, user_message: ChatMessage
    ) -> tuple[list[ReasoningStep], list[ToolExecution]]:
        """Generate agent's reasoning and tool calls.

        Args:
            user_message: The user's question

        Returns:
            Tuple of (reasoning_steps, tool_calls)
        """
        # Build prompt with available tools
        tools_info = self._format_tools_for_prompt()

        prompt = f"""{self.config.dataset_system_prompt}

User request: {user_message.content}

Available tools:
{tools_info}

Generate your reasoning and tool calls to handle this request.
Provide step-by-step reasoning (2-4 steps) and identify which tools to call with specific arguments."""

        class AgentThinking(BaseModel):
            reasoning_steps: list[ReasoningStep]
            tool_calls: list[ToolExecution]

        response: AgentThinking
        if self.progress_reporter:
            temp_response = None
            async for chunk, result in self.llm.generate_async_stream(
                prompt=prompt,
                schema=AgentThinking,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk("agent_reasoning", chunk)
                if result:
                    temp_response = result
            if temp_response is None:
                msg = "Failed to generate agent thinking"
                raise DataSetGeneratorError(msg)
            response = cast(AgentThinking, temp_response)
        else:
            response = await self.llm.generate_async(
                prompt=prompt,
                schema=AgentThinking,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        return response.reasoning_steps, response.tool_calls

    async def _simulate_tool_results(self, tool_calls: list[ToolExecution]) -> list[ToolExecution]:
        """Simulate tool execution results.

        The LLM generates realistic output strings for each tool call.

        Args:
            tool_calls: List of tool executions (without results yet)

        Returns:
            Same list but with 'result' field populated
        """
        completed_executions = []

        for tool_call in tool_calls:
            # Get tool definition from registry
            tool_def = self.tool_registry.get_tool(tool_call.function_name)
            if not tool_def:
                msg = f"Tool '{tool_call.function_name}' not found in registry"
                raise ValueError(msg)

            # Generate realistic output
            prompt = f"""Generate realistic output for this tool execution:

Tool: {tool_def.name}
Description: {tool_def.description}
Arguments called with: {tool_call.arguments}

Generate the tool's output/result. Make it realistic and appropriate for the tool and arguments."""

            result: ToolOutput
            if self.progress_reporter:
                temp_result = None
                async for chunk, res in self.llm.generate_async_stream(
                    prompt=prompt,
                    schema=ToolOutput,
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,
                ):
                    if chunk:
                        self.progress_reporter.emit_chunk(
                            f"tool_sim_{tool_call.function_name}", chunk
                        )
                    if res:
                        temp_result = res
                if temp_result is None:
                    msg = f"Failed to generate tool output for {tool_call.function_name}"
                    raise DataSetGeneratorError(msg)
                result = cast(ToolOutput, temp_result)
            else:
                result = await self.llm.generate_async(
                    prompt=prompt,
                    schema=ToolOutput,
                    max_tokens=self.config.max_tokens,
                    temperature=0.7,  # Slightly creative for variety
                )

            # Create new execution with result
            completed_executions.append(
                ToolExecution(
                    function_name=tool_call.function_name,
                    arguments=tool_call.arguments,
                    reasoning=tool_call.reasoning,
                    result=result.result,
                )
            )

        return completed_executions

    async def _generate_agent_conclusion(
        self,
        user_message: ChatMessage,
        _reasoning: list[ReasoningStep],
        tool_results: list[ToolExecution],
    ) -> ChatMessage:
        """Generate agent's final response interpreting tool results.

        Args:
            user_message: Original user question
            _reasoning: Agent's reasoning steps (unused, kept for interface consistency)
            tool_results: Tool execution results

        Returns:
            Agent's final response message
        """
        # Format tool results for prompt
        results_text = "\n".join(
            [f"Tool: {r.function_name}\nResult: {r.result}" for r in tool_results]
        )

        prompt = f"""{self.config.dataset_system_prompt}

User request: {user_message.content}

You executed these tools:
{results_text}

Based on these results, provide a clear, helpful response to the user."""

        response: AgentResponse
        if self.progress_reporter:
            temp_response = None
            async for chunk, result in self.llm.generate_async_stream(
                prompt=prompt,
                schema=AgentResponse,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk("agent_response", chunk)
                if result:
                    temp_response = result
            if temp_response is None:
                msg = "Failed to generate agent response"
                raise DataSetGeneratorError(msg)
            response = cast(AgentResponse, temp_response)
        else:
            response = await self.llm.generate_async(
                prompt=prompt,
                schema=AgentResponse,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        return ChatMessage(role="assistant", content=response.content)

    def _build_conversation(
        self,
        user_message: ChatMessage,
        reasoning: list[ReasoningStep],
        _tool_calls: list[ToolExecution],
        tool_results: list[ToolExecution],
        agent_response: ChatMessage,
        topic_prompt: str = "",
    ) -> Conversation:
        """Assemble all components into a Conversation.

        Args:
            user_message: User's question
            reasoning: Agent's reasoning steps
            _tool_calls: Tool calls made (unused, kept for interface consistency)
            tool_results: Tool execution results (contains completed tool calls with results)
            agent_response: Agent's final response
            topic_prompt: Topic used to generate this conversation (for metadata)

        Returns:
            Complete Conversation object
        """
        messages = []

        # Don't add system message for agent mode - it interferes with tool calling
        # The system prompt teaches models to explain tool usage instead of executing tools
        # For tool calling, the tool definitions themselves serve as instructions

        # Add user message
        messages.append(user_message)

        # Build tool_calls in OpenAI format
        tool_calls_openai = []
        for idx, result in enumerate(tool_results):
            tool_call_id = f"call_{idx}"
            tool_calls_openai.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": result.function_name, "arguments": result.arguments},
                }
            )

        # Add first assistant message with tool_calls
        # The ChatML formatter will add <think> tags and <tool_call> tags based on
        # reasoning and tool_context.executions
        messages.append(
            ChatMessage(
                role="assistant",
                content="",
                tool_calls=tool_calls_openai if tool_calls_openai else None,
            )
        )

        # Add tool response messages with tool_call_id
        for idx, result in enumerate(tool_results):
            tool_call_id = f"call_{idx}"
            messages.append(
                ChatMessage(role="tool", content=result.result, tool_call_id=tool_call_id)
            )

        # Add final assistant response with the answer
        messages.append(agent_response)

        # Build tool context
        tool_context = ToolContext(
            available_tools=self.tool_registry.tools,
            executions=tool_results,
        )

        # Build reasoning trace
        reasoning_trace = ReasoningTrace(
            style=self.config.reasoning_style or "structured", content=reasoning
        )

        # Build agent context
        agent_context = AgentContext(mode="single_turn")

        # Build metadata
        metadata = {
            "conversation_type": "chain_of_thought" if reasoning_trace else "basic",
            "topic": topic_prompt if topic_prompt else "general",
        }

        # Insert system message if configured
        self._insert_system_message_if_configured(messages)

        # Convert tools to OpenAI format
        tools_openai = [tool.to_openai() for tool in self.tool_registry.tools]

        return Conversation(
            messages=messages,
            reasoning=reasoning_trace,
            tool_context=tool_context,
            tools=tools_openai,
            agent_context=agent_context,
            question=user_message.content or "",  # Set question field for formatters
            final_answer=agent_response.content or "",  # Set final_answer field for formatters
            metadata=metadata,
        )

    def _format_tools_for_prompt(self) -> str:
        """Format available tools for inclusion in prompts.

        Returns:
            Formatted string describing available tools
        """
        tool_descriptions = []
        for tool in self.tool_registry.tools:
            params = ", ".join([f"{p.name}: {p.type}" for p in tool.parameters])
            tool_descriptions.append(f"- {tool.name}({params}): {tool.description}")

        return "\n".join(tool_descriptions)

    def _insert_system_message_if_configured(self, messages: list[ChatMessage]) -> None:
        """Insert system message at the beginning of messages if configured.

        Args:
            messages: List of messages to potentially prepend system message to
        """
        if self.config.sys_msg:
            messages.insert(
                0,
                ChatMessage(role="system", content=self.config.dataset_system_prompt or ""),
            )


class MultiTurnAgentBuilder(SingleTurnAgentBuilder):
    """Builder for multi-turn agent conversations.

    Extends SingleTurnAgentBuilder to generate conversations with multiple
    user-agent interaction turns. Each turn can involve different tools
    and builds on previous context.
    """

    async def generate(self, topic_prompt: str) -> Conversation:
        """Generate multi-turn agent conversation.

        Args:
            topic_prompt: Topic or scenario to generate conversation about

        Returns:
            Complete multi-turn Conversation

        Raises:
            ValueError: If generation fails or config is invalid
        """
        # Determine number of turns (from config range)
        num_turns = random.randint(self.config.min_turns, self.config.max_turns)  # noqa: S311 # nosec

        # Track conversation context
        turns: list[AgentTurnData] = []
        all_messages: list[ChatMessage] = []

        # Generate scenario overview
        scenario = await self._generate_scenario(topic_prompt, num_turns)

        for turn_idx in range(num_turns):
            # Generate this turn
            turn_data = await self._generate_turn(turn_idx, scenario, all_messages)
            turns.append(turn_data)

            # Accumulate messages for context
            all_messages.extend(
                [
                    turn_data.user_message,
                    turn_data.agent_response,
                ]
            )

            # Check if we should conclude early
            if turn_idx >= self.config.min_turns - 1 and await self._should_conclude_early(
                all_messages, scenario, turn_idx + 1
            ):
                break

        # Assemble into complete conversation
        return self._build_multi_turn_conversation(turns, scenario, topic_prompt)

    async def _generate_scenario(self, topic_prompt: str, num_turns: int) -> str:
        """Generate a multi-turn scenario description.

        Args:
            topic_prompt: Original topic
            num_turns: Number of turns to plan for

        Returns:
            Scenario description that requires multiple interactions
        """
        prompt = f"""Generate a realistic scenario for this topic that requires {num_turns} user-agent interaction turns:
{topic_prompt}

The scenario should:
- Require multiple steps to complete
- Each turn should build on previous turns
- Use tools progressively (different tools in different turns)

Keep it brief (2-3 sentences)."""

        response: Scenario
        if self.progress_reporter:
            temp_response = None
            async for chunk, result in self.llm.generate_async_stream(
                prompt=prompt,
                schema=Scenario,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk("scenario_gen", chunk)
                if result:
                    temp_response = result
            if temp_response is None:
                msg = "Failed to generate scenario"
                raise DataSetGeneratorError(msg)
            response = cast(Scenario, temp_response)
        else:
            response = await self.llm.generate_async(
                prompt=prompt,
                schema=Scenario,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        return response.description

    async def _generate_turn(
        self,
        turn_idx: int,
        scenario: str,
        previous_messages: list[ChatMessage],
    ) -> AgentTurnData:
        """Generate a single turn of the conversation.

        Args:
            turn_idx: Index of this turn (0-based)
            scenario: Overall scenario description
            previous_messages: Messages from previous turns

        Returns:
            Complete turn data
        """
        # Build context from previous messages
        context_text = self._format_message_context(previous_messages)

        # Generate user message for this turn
        user_message = await self._generate_turn_user_message(turn_idx, scenario, context_text)

        # Generate agent thinking (reasoning + tool calls)
        reasoning, tool_calls = await self._generate_agent_thinking_with_context(
            user_message, context_text
        )

        # Simulate tools
        tool_results = await self._simulate_tool_results(tool_calls)

        # Generate agent response
        agent_response = await self._generate_agent_conclusion(
            user_message, reasoning, tool_results
        )

        return AgentTurnData(
            user_message=user_message,
            reasoning_steps=reasoning,
            tool_calls=tool_results,
            agent_response=agent_response,
        )

    async def _generate_turn_user_message(
        self,
        turn_idx: int,
        scenario: str,
        context: str,
    ) -> ChatMessage:
        """Generate user message for a specific turn.

        Args:
            turn_idx: Turn index
            scenario: Overall scenario
            context: Previous conversation context

        Returns:
            User message for this turn
        """
        turn_guidance = {
            0: "Start with the initial request or question",
            1: "Request a follow-up action or ask for more information",
            2: "Request another related action or verify results",
            3: "Final request or verification",
        }

        guidance = turn_guidance.get(turn_idx, "Continue the conversation naturally")

        prompt = f"""Scenario: {scenario}

Previous conversation:
{context if context else "(No previous conversation)"}

Generate the user's message for turn {turn_idx + 1}.
Guidance: {guidance}

The message should reference or build upon previous conversation if applicable.
Keep it concise and natural."""

        response: UserQuestion
        if self.progress_reporter:
            temp_response = None
            async for chunk, result in self.llm.generate_async_stream(
                prompt=prompt,
                schema=UserQuestion,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk(
                        f"turn_{turn_idx}_user", chunk, turn=turn_idx + 1
                    )
                if result:
                    temp_response = result
            if temp_response is None:
                msg = f"Failed to generate user question for turn {turn_idx + 1}"
                raise DataSetGeneratorError(msg)
            response = cast(UserQuestion, temp_response)
        else:
            response = await self.llm.generate_async(
                prompt=prompt,
                schema=UserQuestion,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        return ChatMessage(role="user", content=response.content)

    async def _generate_agent_thinking_with_context(
        self, user_message: ChatMessage, context: str
    ) -> tuple[list[ReasoningStep], list[ToolExecution]]:
        """Generate agent thinking with conversation context.

        Args:
            user_message: Current user message
            context: Previous conversation context

        Returns:
            Tuple of (reasoning_steps, tool_calls)
        """
        tools_info = self._format_tools_for_prompt()

        prompt = f"""{self.config.dataset_system_prompt}

Previous conversation context:
{context if context else "(No previous context)"}

Current user request: {user_message.content}

Available tools:
{tools_info}

Generate your reasoning and tool calls for THIS specific request only.
Provide step-by-step reasoning (1-3 steps) and identify which tools are needed."""

        class AgentThinking(BaseModel):
            reasoning_steps: list[ReasoningStep]
            tool_calls: list[ToolExecution]

        response: AgentThinking
        if self.progress_reporter:
            temp_response = None
            async for chunk, result in self.llm.generate_async_stream(
                prompt=prompt,
                schema=AgentThinking,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            ):
                if chunk:
                    self.progress_reporter.emit_chunk("agent_thinking_mt", chunk)
                if result:
                    temp_response = result
            if temp_response is None:
                msg = "Failed to generate agent thinking for multi-turn"
                raise DataSetGeneratorError(msg)
            response = cast(AgentThinking, temp_response)
        else:
            response = await self.llm.generate_async(
                prompt=prompt,
                schema=AgentThinking,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

        return response.reasoning_steps, response.tool_calls

    async def _should_conclude_early(
        self, messages: list[ChatMessage], scenario: str, current_turn: int
    ) -> bool:
        """Determine if conversation should conclude before max_turns.

        Args:
            messages: All messages so far
            scenario: Original scenario
            current_turn: Current turn number

        Returns:
            True if conversation should end
        """
        # Format conversation so far
        conversation_text = self._format_message_context(messages)

        prompt = f"""Scenario: {scenario}

Conversation so far (after {current_turn} turns):
{conversation_text}

Is the user's original task/goal from the scenario fully completed?
- True: Task is complete, conversation can end naturally
- False: Task incomplete, more turns needed"""

        response = await self.llm.generate_async(
            prompt=prompt,
            schema=ConclusionDecision,
            max_tokens=10,
            temperature=0.3,
        )

        return response.should_conclude

    def _format_message_context(self, messages: list[ChatMessage]) -> str:
        """Format messages as readable context.

        Args:
            messages: List of chat messages

        Returns:
            Formatted string of messages
        """
        if not messages:
            return ""

        lines = []
        for msg in messages:
            lines.append(f"{msg.role}: {msg.content}")

        return "\n".join(lines)

    def _build_multi_turn_conversation(
        self, turns: list[AgentTurnData], scenario: str, topic_prompt: str = ""
    ) -> Conversation:
        """Assemble multi-turn conversation from turn data.

        Args:
            turns: List of turn data
            scenario: Scenario description
            topic_prompt: Topic used to generate this conversation (for metadata)

        Returns:
            Complete Conversation object
        """
        messages = []

        # Don't add system message for agent mode - it interferes with tool calling
        # The system prompt teaches models to explain tool usage instead of executing tools
        # For tool calling, the tool definitions themselves serve as instructions

        # Collect all reasoning steps and tool executions
        all_reasoning: list[ReasoningStep] = []
        all_executions: list[ToolExecution] = []

        # Track tool_call_id counter across all turns
        tool_call_counter = 0

        # Add messages from each turn in correct order:
        # user -> assistant (thinking/tool_calls) -> tool (responses) -> assistant (final answer)
        for turn in turns:
            # User message
            messages.append(turn.user_message)

            # Build tool_calls for this turn in OpenAI format
            tool_calls_openai = []
            turn_tool_call_ids = []
            for tool_exec in turn.tool_calls:
                tool_call_id = f"call_{tool_call_counter}"
                tool_call_counter += 1
                turn_tool_call_ids.append(tool_call_id)
                tool_calls_openai.append(
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_exec.function_name,
                            "arguments": tool_exec.arguments,
                        },
                    }
                )

            # First assistant message with tool_calls
            # This represents the assistant's "thinking" phase where it plans tool usage
            messages.append(
                ChatMessage(
                    role="assistant",
                    content="",
                    tool_calls=tool_calls_openai if tool_calls_openai else None,
                )
            )

            # Tool response messages with tool_call_id
            for idx, tool_exec in enumerate(turn.tool_calls):
                messages.append(
                    ChatMessage(
                        role="tool", content=tool_exec.result, tool_call_id=turn_tool_call_ids[idx]
                    )
                )

            # Final assistant response with the answer
            messages.append(turn.agent_response)

            # Accumulate reasoning and executions across all turns
            all_reasoning.extend(turn.reasoning_steps)
            all_executions.extend(turn.tool_calls)

        # Build tool context with all tools used across all turns
        tool_context = ToolContext(
            available_tools=self.tool_registry.tools,
            executions=all_executions,
        )

        # Build reasoning trace
        reasoning_trace = ReasoningTrace(
            style=self.config.reasoning_style or "structured",
            content=all_reasoning,
        )

        # Build agent context
        agent_context = AgentContext(
            mode="multi_turn",
            planning_trace=scenario,
            execution_summary=f"Completed {len(turns)}-turn conversation",
        )

        # Build metadata
        metadata = {
            "conversation_type": "chain_of_thought" if reasoning_trace else "basic",
            "topic": topic_prompt if topic_prompt else "general",
        }

        # Insert system message if configured
        self._insert_system_message_if_configured(messages)

        # Convert tools to OpenAI format
        tools_openai = [tool.to_openai() for tool in self.tool_registry.tools]

        return Conversation(
            messages=messages,
            reasoning=reasoning_trace,
            tool_context=tool_context,
            tools=tools_openai,
            agent_context=agent_context,
            metadata=metadata,
        )
