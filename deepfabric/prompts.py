class TreePromptBuilder:
    """Build dynamic prompts for topic tree expansion with domain-specific examples."""

    # Domain-specific expansion examples
    EXAMPLES = {
        "general": [
            {
                "path": ["Technology", "Artificial Intelligence"],
                "subtopics": [
                    "machine learning",
                    "neural networks",
                    "computer vision",
                    "natural language processing",
                    "robotics",
                ],
            },
            {
                "path": ["Entertainment", "Movies", "Actors"],
                "subtopics": [
                    "Tom Hanks",
                    "Meryl Streep",
                    "Leonardo DiCaprio",
                    "Jennifer Lawrence",
                    "Denzel Washington",
                ],
            },
        ],
        "conversational": [
            {
                "path": ["Small Talk Topics"],
                "subtopics": [
                    "weather",
                    "weekend plans",
                    "hobbies",
                    "family",
                    "books",
                    "food",
                    "music",
                ],
            },
            {
                "path": ["Small Talk Topics", "Family"],
                "subtopics": [
                    "parents",
                    "grandparents",
                    "siblings",
                    "family traditions",
                    "family vacations",
                ],
            },
            {
                "path": ["Small Talk Topics", "Hobbies", "Cooking"],
                "subtopics": [
                    "recipes",
                    "asian food",
                    "favourite dishes",
                    "cookbooks",
                    "kitchen gadgets",
                    "vegan cooking",
                ],
            },
        ],
        "technical": [
            {
                "path": ["Programming"],
                "subtopics": [
                    "algorithms",
                    "data structures",
                    "debugging",
                    "testing",
                    "version control",
                ],
            },
            {
                "path": ["Programming", "Python"],
                "subtopics": ["pandas", "flask", "pytest", "asyncio", "django"],
            },
        ],
        "educational": [
            {
                "path": ["Mathematics"],
                "subtopics": ["algebra", "geometry", "calculus", "statistics", "probability"],
            },
            {
                "path": ["Mathematics", "Algebra"],
                "subtopics": [
                    "linear equations",
                    "quadratic functions",
                    "polynomials",
                    "matrices",
                    "systems",
                ],
            },
        ],
    }

    @classmethod
    def build_expansion_prompt(
        cls,
        topic_path: list[str],
        num_subtopics: int,
        system_prompt: str = "",
        domain: str = "general",
    ) -> str:
        """Build a topic expansion prompt with relevant examples."""

        path_str = " -> ".join(f'"{topic}"' for topic in topic_path)
        examples = cls._format_examples(cls.EXAMPLES.get(domain, cls.EXAMPLES["general"]))

        return f"""Generate {num_subtopics} subtopics for training data organization.

Task: Create diverse but related subtopics that expand on the given topic path.

Examples:
{examples}

Context: {system_prompt}

Topic path: {path_str}
Generate {num_subtopics} subtopics as a Python list. Return only the list, nothing else."""

    @classmethod
    def _format_examples(cls, examples: list) -> str:
        """Format examples for inclusion in prompt."""
        formatted = []
        for ex in examples[:3]:  # Limit to 3 examples
            path_str = " -> ".join(f'"{topic}"' for topic in ex["path"])
            subtopics_str = str(ex["subtopics"])
            formatted.append(f"Path: {path_str}\nSubtopics: {subtopics_str}")
        return "\n\n".join(formatted)


# Structured Agent Tool-Calling Prompt Builder
class AgentPromptBuilder:
    """Build structured prompts for agent tool-calling training."""

    @staticmethod
    def build_tool_context_prompt(
        tool_registry, instructions: str = "", subtopics: str = ""
    ) -> str:
        """Build a minimal context prompt that relies on structured generation."""
        tool_signatures = []
        for tool in tool_registry.tools:
            tool_signatures.append(f"- {tool.to_signature()}")

        return f"""Generate a realistic agent training example with tool usage reasoning.

Available tools:
{chr(10).join(tool_signatures)}

Focus on WHY each tool is selected and HOW parameters are constructed.

{instructions}
{subtopics}

Generate a complete agent reasoning example using structured output."""

    @staticmethod
    def build_multi_turn_context_prompt(
        tool_registry, instructions: str = "", subtopics: str = ""
    ) -> str:
        """Build context for multi-turn conversations."""
        tool_signatures = []
        for tool in tool_registry.tools:
            tool_signatures.append(f"- {tool.to_signature()}")

        return f"""Generate a multi-turn agent conversation with evolving tool usage.

Available tools:
{chr(10).join(tool_signatures)}

Show tool dependencies and reasoning across conversation turns.

{instructions}
{subtopics}

Generate a complete multi-turn conversation using structured output."""


# Simplified prompts that delegate to structured generation
AGENT_COT_TOOLS_PROMPT = """Generate an agent tool-calling training example using the available tool definitions.

Focus on the reasoning process: WHY tools are selected, HOW parameters are constructed, and WHAT results are expected.

Create realistic scenarios that teach proper tool reasoning patterns.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

AGENT_COT_HYBRID_PROMPT = """Generate agent tool-calling examples with rich CoT reasoning traces and tool execution.

Combine natural language reasoning with structured step-by-step traces that include:
- Chain of thought analysis
- Structured reasoning steps with thoughts and actions
- Clear tool selection and parameter reasoning
- Tool execution with results

Focus on teaching both the reasoning process AND tool usage patterns.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

AGENT_COT_MULTI_TURN_PROMPT = """Generate a multi-turn agent conversation with tool usage across turns.

Show how reasoning evolves: tool dependencies, progressive refinement, and result synthesis.

Create realistic tool chaining patterns and decision-making processes.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

XLAM_MULTI_TURN_PROMPT = """Generate a realistic multi-turn conversation between a user (human) and an AI agent (gpt) that uses function calling.

SCENARIO: {{{{subtopics}}}}

AVAILABLE TOOLS (use these appropriately for the scenario):
{{{{tools}}}}

CRITICAL REQUIREMENTS:
1. **Generate a UNIQUE, DETAILED domain_policy for this specific scenario**
   - Examples:
     * Airline/Travel: "Airline Reservation Policy: Bookings can be modified up to 24 hours before departure. Cancellations within 24 hours incur 50% fee. Full refunds available if cancelled 7+ days in advance. All modifications require booking ID and verification."
     * E-commerce: "Return Policy: Items can be returned within 30 days of delivery. Refunds processed within 5-7 business days. Free return shipping for defective items. Restocking fee of 15% applies to non-defective returns."
     * Finance: "Transaction Policy: Daily transfer limit $10,000. International transfers require 2-factor authentication. Transfers processed within 1-3 business days. Fees: domestic $0, international 3%."
   - Make it comprehensive with specific rules, time limits, fees, and procedures
   - This becomes the 'system' field in the output

2. Create a natural 3-12 turn conversation showing realistic user-agent interaction
3. User may provide information incrementally across multiple turns
4. Agent must use function_call turns to invoke tools when needed (format: {{"name": "tool_name", "arguments": {{"param": "value"}}}} )
5. After each function_call, include an observation turn with realistic tool output
6. Agent must interpret observation results and respond helpfully to the user
7. Keep conversation focused on the scenario topic
8. Use tools from the available set that logically solve the user's needs
9. Show natural clarification exchanges when needed

TURN TYPES AND FLOW PATTERNS:
- human: User requests, clarifications, additional information
- gpt: Agent responses, questions for clarification, final answers
- function_call: Tool invocations (must be valid JSON with "name" and "arguments")
- observation: Tool execution results (can be empty string, JSON, or descriptive text)

COMMON PATTERNS:
1. human → gpt (ask clarifying question) → human (provide info) → function_call → observation → gpt (answer)
2. human → function_call → observation → gpt (direct tool use with immediate response)
3. human → gpt → function_call → observation → function_call → observation → gpt (multi-tool sequence)

INSTRUCTIONS:
{{{{instructions}}}}

Focus on creating realistic, natural conversations with a unique domain policy for this specific scenario."""

CONVERSATION_GENERATION_PROMPT = """Generate a training conversation for a language model with this system prompt:

<system_prompt>
{{{{system_prompt}}}}
</system_prompt>

Create a realistic conversation that demonstrates the system's capabilities. The conversation should:
- Start with a user question/request
- Have the assistant respond helpfully according to the system prompt
- Be natural and educational

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}

Generate one training sample as a conversation."""

# Legacy constant for backward compatibility
TOPIC_EXPANSION_PROMPT = """Generate subtopics for training data organization.

Create diverse but related subtopics that expand on the given topic path.

Topic path: {{{{subtopics_list}}}}
System prompt: {{{{system_prompt}}}}
Generate {{{{num_subtopics}}}} subtopics as a Python list. Return only the list."""

GRAPH_EXPANSION_PROMPT = """
You are an expert in knowledge graph generation. Your task is to expand a topic into a set of subtopics. For each subtopic, you should also identify if it connects to any other existing topics in the graph.

Here is the current state of the graph:
{{current_graph_summary}}

You are expanding the topic: "{{current_topic}}"

Generate a list of {{num_subtopics}} subtopics. For each subtopic, provide:
1. A "topic" string - the name of the new subtopic
2. A "connections" list of IDs of existing topics it should connect to for creating cross-links (use empty list if no connections)
"""

# Chain of Thought prompts for reasoning-based dataset generation
FREETEXT_COT_PROMPT = """Generate an educational reasoning problem that requires analytical thinking to solve.

Create problems involving mathematics, logic, science, or analytical reasoning that can be solved through clear thinking steps.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

STRUCTURED_COT_PROMPT = """Generate a training conversation that demonstrates systematic problem-solving.

Create realistic educational dialogues where complex problems are solved through methodical reasoning.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

HYBRID_COT_PROMPT = """Generate educational problems that require analytical and systematic thinking.

Create challenging reasoning problems suitable for training systematic problem-solving skills.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

# Mathematical variants with strict numerical output requirements
FREETEXT_COT_MATHEMATICAL_PROMPT = """Generate mathematical word problems with clear numerical solutions.

CRITICAL FORMATTING RULES for final_answer field:

**Always provide ONLY the numerical value (no units, symbols, or text).**

**Problem Type Guidelines:**
- Money/Currency problems → Always format as X.XX (2 decimal places): "104.00", "6.50"
- Time duration problems → Integer minutes: "90", "135"
- Clock time problems → 24-hour format (4 digits): "1615" (for 4:15 PM), "0830" (for 8:30 AM), "0600" (for 6:00 AM)
- Counting problems → Integers: "42", "100"
- Percentage problems → Number only: "25" (not "25%")
- Scientific/measurement → Preserve precision: "3.14", "2.5"

**Examples:**
✓ CORRECT: "104.00" (money), "90" (duration), "1615" (4:15 PM), "0830" (8:30 AM), "42" (count)
✗ INCORRECT: "$104", "104", "90 minutes", "4:15 PM", "830", "42 apples"

**Key Rule: Read the problem type carefully and format accordingly!**

Create problems that test mathematical reasoning skills with unambiguous numerical answers.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

STRUCTURED_COT_MATHEMATICAL_PROMPT = """Generate mathematical problems with structured reasoning steps and numerical answers.

CRITICAL FORMATTING RULES for final_answer field:

**Always provide ONLY the numerical value (no units, symbols, or text).**

**Problem Type Guidelines:**
- Money/Currency problems → Always format as X.XX (2 decimal places): "104.00", "6.50"
- Time duration problems → Integer minutes: "90", "135"
- Clock time problems → 24-hour format (4 digits): "1615" (for 4:15 PM), "0830" (for 8:30 AM), "0600" (for 6:00 AM)
- Counting problems → Integers: "42", "100"
- Percentage problems → Number only: "25" (not "25%")
- Scientific/measurement → Preserve precision: "3.14", "2.5"

**Examples:**
✓ CORRECT: "104.00" (money), "90" (duration), "1615" (4:15 PM), "0830" (8:30 AM), "42" (count)
✗ INCORRECT: "$104", "104", "90 minutes", "4:15 PM", "830", "42 apples"

**Key Rule: Read the problem type carefully and format accordingly!**

Create clear step-by-step reasoning traces that lead to precise numerical answers.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

HYBRID_COT_MATHEMATICAL_PROMPT = """Generate mathematical word problems with both natural and structured reasoning.

CRITICAL FORMATTING RULES for final_answer field:

**Always provide ONLY the numerical value (no units, symbols, or text).**

**Problem Type Guidelines:**
- Money/Currency problems → Always format as X.XX (2 decimal places): "104.00", "6.50"
- Time duration problems → Integer minutes: "90", "135"
- Clock time problems → 24-hour format (4 digits): "1615" (for 4:15 PM), "0830" (for 8:30 AM), "0600" (for 6:00 AM)
- Counting problems → Integers: "42", "100"
- Percentage problems → Number only: "25" (not "25%")
- Scientific/measurement → Preserve precision: "3.14", "2.5"

**Examples:**
✓ CORRECT: "104.00" (money), "90" (duration), "1615" (4:15 PM), "0830" (8:30 AM), "42" (count)
✗ INCORRECT: "$104", "104", "90 minutes", "4:15 PM", "830", "42 apples"

**Key Rule: Read the problem type carefully and format accordingly!**

Create problems that combine intuitive explanations with systematic step-by-step solutions.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""
