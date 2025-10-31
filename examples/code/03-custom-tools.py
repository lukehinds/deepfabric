"""
Custom Tools Example

This example demonstrates how to use custom tools with DeepFabric.
Tools are best configured via YAML - this file shows the structure.

See examples/configs/xlam-example.yaml for a complete working example.
"""


def show_custom_tools_structure():
    """Demonstrate how to define custom tools in YAML."""

    print("=" * 80)
    print("Custom Tools Configuration")
    print("=" * 80)

    yaml_example = """
# Custom tools are defined in YAML config under data_engine

data_engine:
  # ... other config ...

  # Define custom tools
  custom_tools:
    - name: "query_database"
      description: "Execute a SQL query on a database and return results"
      parameters:
        properties:
          query:
            type: "string"
            description: "The SQL query to execute"
          database:
            type: "string"
            description: "Database name to query"
          limit:
            type: "integer"
            description: "Maximum number of rows to return"
        required: ["query", "database"]

    - name: "send_email"
      description: "Send an email message to recipients"
      parameters:
        properties:
          to:
            type: "string"
            description: "Recipient email address"
          subject:
            type: "string"
            description: "Email subject line"
          body:
            type: "string"
            description: "Email body content"
        required: ["to", "subject", "body"]

# To use these tools, you must also set agent_mode:
data_engine:
  agent_mode: "single_turn"  # or "multi_turn"
  custom_tools:
    # ... tool definitions ...
"""

    print(yaml_example)


def show_complete_tool_example():
    """Show a complete YAML configuration with custom tools."""

    print("\n" + "=" * 80)
    print("Complete Example with Custom Tools")
    print("=" * 80)

    complete_example = """
# Complete configuration for tool-using agents

dataset_system_prompt: "You are an AI assistant with access to tools."

topic_tree:
  topic_prompt: "Tasks requiring tool usage (database queries, emails, etc.)"
  provider: "openai"
  model: "gpt-4o"
  depth: 2
  degree: 3

data_engine:
  generation_system_prompt: "Generate examples showing proper tool usage"
  provider: "openai"
  model: "gpt-4o"
  temperature: 0.8

  # Enable agent mode for tool calling
  agent_mode: "single_turn"

  # Define custom tools
  custom_tools:
    - name: "query_database"
      description: "Execute SQL queries on databases"
      parameters:
        properties:
          query:
            type: "string"
            description: "The SQL query to execute"
          database:
            type: "string"
            description: "Database name"
        required: ["query", "database"]

    - name: "send_email"
      description: "Send email messages"
      parameters:
        properties:
          to:
            type: "string"
            description: "Recipient email"
          subject:
            type: "string"
            description: "Email subject"
          body:
            type: "string"
            description: "Email content"
        required: ["to", "subject", "body"]

dataset:
  save_as: "tool_dataset.jsonl"
  creation:
    num_steps: 20
    batch_size: 2
    sys_msg: true

  # Format for tool calling (e.g., XLAM, tool_calling)
  formatters:
    - name: "tool_calling"
      template: "builtin://tool_calling"
      output: "tool_dataset_formatted.jsonl"
      config:
        include_tools_in_system: true
"""

    print(complete_example)


def show_usage_instructions():
    """Show how to use custom tools in practice."""

    print("\n" + "=" * 80)
    print("Usage Instructions")
    print("=" * 80)

    instructions = """
1. CREATE A YAML CONFIG with custom tools (see above example)

2. RUN GENERATION:
   deepfabric generate my_tool_config.yaml

3. OUTPUT FORMATS:
   - Use "tool_calling" formatter for standard tool use format
   - Use "xlam_v2" formatter for XLAM agent training format
   - Use "single_tool_call" for constrained single-tool scenarios

4. AGENT MODES:
   - single_turn: One-shot tool usage (tool call → result → response)
   - multi_turn: Extended conversations with multiple tool calls

5. EXAMPLE CONFIG FILES:
   - examples/configs/xlam-example.yaml (complete multi-turn agent)
   - examples/configs/03-multi-turn-agent.yaml (agent with tools)

KEY POINTS:
- Custom tools MUST be used with agent_mode set
- Tools are defined in YAML under data_engine.custom_tools
- Each tool needs: name, description, parameters (with properties + required fields)
- Use appropriate formatters for tool-calling output

For more details, see the documentation or example configs.
"""

    print(instructions)


def main():
    """Main demonstration."""

    show_custom_tools_structure()
    show_complete_tool_example()
    show_usage_instructions()

    print("\n" + "=" * 80)
    print("See examples/configs/xlam-example.yaml for a working example!")
    print("=" * 80)


if __name__ == "__main__":
    main()
