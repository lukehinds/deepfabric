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

TOPIC_EXPANSION_PROMPT = """Generate subtopics for training data organization. You'll receive a topic path and need to create relevant subtopics that expand on the given topic.

Your task: Given a topic path, generate specific subtopics that are related but diverse enough to create varied training content.

Examples:

Example 1:
node path: "News Topics" -> "Sports" -> "Football"
desired number of subtopics: 5
subtopics: ["college football", "football stadiums", "health consequences football", "Seattle Seahawks", "football sponsorships"]


Example 2:
node path: "News Topics" -> "Entertainment" -> "Movies" -> "Star Portraits"
desired number of subtopics: 8
subtopics: ["Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Jennifer Lawrence", "Denzel Washington", "Charlize Theron", "Robert Downey Jr.", "Emma Stone"]


Here are three new examples, this time for generating smalltalk topics for a friendly chat assistant:

Example 1:
node path: "Small Talk Topics"
desired number of subtopics: 7
subtopics: ["weather", "weekend plans", "hobbies", "family", "books", "food", "music"]

Example 2:
node path: "Small Talk Topics" -> "Family"
desired number of subtopics: 5
subtopics: ["parents", "grandparents", "siblings", "family traditions", "family vacations"]

Example 3:
node path: "Small Talk Topics" -> "Hobbies" -> "Cooking"
desired number of subtopics: 6
subtopics: ["recipes", "asian food", "favourite dishes", "cookbooks", "kitchen gadgets", "vegan cooking"]


Here is a description / the system prompt for the model we want to train:

<system_prompt>
{{{{system_prompt}}}}
</system_prompt>


Here is your topic input. When generating subtopics, remain somewhat vague. Things can only be tangentially related and they don't have to be interpreted in a single way. Importantly, make sure that the subtopics fit the system prompt, if one was supplied:
node path: {{{{subtopics_list}}}}
desired number of subtopics: {{{{num_subtopics}}}}

Now return the subtopics as a python list, and return it in just one line, not multiple ones. Don't return anything else."""

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
