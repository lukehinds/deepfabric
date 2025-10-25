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
        tool_registry, instructions: str = "", subtopics: str = "", max_tools_per_query: int = 3
    ) -> str:
        """Build a minimal context prompt that relies on structured generation."""
        tool_signatures = []
        for tool in tool_registry.tools:
            tool_signatures.append(f"- {tool.to_signature()}")

        return f"""Generate a realistic agent training example with tool usage reasoning.

Available tools:
{chr(10).join(tool_signatures)}

You may use 1 to {max_tools_per_query} tools to complete the task.

Focus on WHY each tool is selected and HOW parameters are constructed.

{instructions}
{subtopics}

Generate a complete agent reasoning example using structured output with tool_executions list."""

    @staticmethod
    def build_multi_turn_context_prompt(
        tool_registry, instructions: str = "", subtopics: str = "", max_tools_per_query: int = 3
    ) -> str:
        """Build context for multi-turn conversations."""
        tool_signatures = []
        for tool in tool_registry.tools:
            tool_signatures.append(f"- {tool.to_signature()}")

        return f"""Generate a multi-turn agent conversation with evolving tool usage.

Available tools:
{chr(10).join(tool_signatures)}

You may use 1 to {max_tools_per_query} tools per query. Show tool dependencies and reasoning across conversation turns.

{instructions}
{subtopics}

Generate a complete multi-turn conversation using structured output with tool_executions list."""


# Simplified prompts that delegate to structured generation
AGENT_COT_TOOLS_PROMPT = """Generate an agent tool-calling training example using the available tool definitions.

You may use multiple tools (up to the specified limit) to complete the task.

Focus on the reasoning process: WHY each tool is selected, HOW parameters are constructed, and WHAT results are expected.

Create realistic scenarios that teach proper tool reasoning patterns and multi-tool orchestration.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

AGENT_COT_HYBRID_PROMPT = """Generate agent tool-calling examples with rich CoT reasoning traces and tool execution.

You may use multiple tools (up to the specified limit) to complete the task.

Combine natural language reasoning with structured step-by-step traces that include:
- Chain of thought analysis
- Structured reasoning steps with thoughts and actions
- Clear tool selection and parameter reasoning
- Multiple tool executions with results

Focus on teaching both the reasoning process AND multi-tool usage patterns.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

AGENT_COT_MULTI_TURN_PROMPT = """Generate a multi-turn agent conversation with tool usage across turns.

Show how reasoning evolves: tool dependencies, progressive refinement, and result synthesis.

Create realistic tool chaining patterns and decision-making processes.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

CONVERSATION_GENERATION_PROMPT = """Generate a training conversation for a language model with this system prompt:

<system_prompt>
{{{{system_prompt}}}}
</system_prompt>

Create a realistic single q&a that demonstrates the system's capabilities. The conversation should:
- Start with a user question/request
- Have the assistant respond helpfully according to the system prompt
- Be natural and educational

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}

Generate one training sample as question and answer."""

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
FREETEXT_COT_PROMPT = """Generate a reasoning problem that requires analytical thinking to solve.

Create problems involving mathematics, logic, science, or analytical reasoning that can be solved through clear thinking steps.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

STRUCTURED_COT_PROMPT = """Generate a training conversation that demonstrates systematic problem-solving.

Create realistic dialogues where complex problems are solved through methodical reasoning.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""

HYBRID_COT_PROMPT = """Generate problems that require analytical and systematic thinking.

Create challenging reasoning problems suitable for training systematic problem-solving skills.

{{{{instructions}}}}
{{{{examples}}}}
{{{{subtopics}}}}"""
