# YAML Configuration for Agent Tool-Calling

Agent tool-calling datasets require specialized configuration to handle tool definitions, reasoning parameters, and output formatting. This guide covers all agent-specific YAML configuration options.

## Basic Agent Configuration

### Minimal Configuration
```yaml
dataset_system_prompt: "You are an AI assistant with access to various tools."

data_engine:
  generation_system_prompt: "You excel at reasoning about tool selection and usage."
  provider: "openai"
  model: "gpt-4o-mini"
  conversation_type: "agent_cot_tools"  # or agent_cot_multi_turn

dataset:
  creation:
    num_steps: 10
    batch_size: 2
  save_as: "agent_dataset.jsonl"
```

### Complete Configuration Example
```yaml
# Agent Tool-Calling Dataset Configuration
dataset_system_prompt: |
  You are an intelligent AI assistant with access to various tools and functions.
  When presented with a task, analyze what's needed, select appropriate tools,
  execute them with proper parameters, and provide clear answers.

topic_tree:
  topic_prompt: "Real-world scenarios requiring tool usage for task completion"
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
  depth: 3
  degree: 3
  save_as: "agent_topics.jsonl"

data_engine:
  # Core generation parameters
  generation_system_prompt: |
    Generate realistic scenarios demonstrating systematic tool reasoning.
    Focus on WHY tools are selected, HOW parameters are constructed,
    and WHAT results mean in context.

  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.8
  max_retries: 3

  # Agent-specific parameters
  conversation_type: "agent_cot_tools"
  reasoning_style: "general"

  # Tool configuration
  available_tools:
    - "get_weather"
    - "search_web"
    - "calculator"
    - "book_restaurant"
    - "get_workout_plan"

  max_tools_per_query: 3

  # Custom tool definitions (optional)
  tool_registry_path: "custom_tools.yaml"

dataset:
  creation:
    num_steps: 25
    batch_size: 5
    sys_msg: false  # Agent formats typically don't use system messages

  save_as: "agent_tool_dataset.jsonl"

  # Apply formatters for different output formats
  formatters:
    - name: "tool_calling_embedded"
      template: "builtin://tool_calling"
      output: "agent_formatted.jsonl"
      config:
        system_prompt: "You are a function calling AI model."
        include_tools_in_system: true
        thinking_format: "<think>{reasoning}</think>"
        tool_call_format: "<tool_call>\n{tool_call}\n</tool_call>"
        tool_response_format: "<tool_response>\n{tool_output}\n</tool_response>"
```

## Agent-Specific Parameters

### `conversation_type`
Specifies the agent format to use:

```yaml
data_engine:
  conversation_type: "agent_cot_tools"      # Single-turn agent reasoning
  # OR
  conversation_type: "agent_cot_multi_turn" # Multi-turn conversational agent
```

### `available_tools`
List of tools the agent can use. Mix default and custom tools:

```yaml
data_engine:
  available_tools:
    # Default DeepFabric tools
    - "get_weather"
    - "search_web"
    - "calculator"
    - "get_time"

    # Custom tools (defined in tool_registry_path)
    - "book_restaurant"
    - "analyze_stock"
    - "get_workout_plan"
```

### `max_tools_per_query`
Maximum number of tools the agent can use in a single interaction:

```yaml
data_engine:
  max_tools_per_query: 3  # Agent can use up to 3 tools per query
```

### `tool_registry_path`
Path to custom tool definitions file:

```yaml
data_engine:
  tool_registry_path: "custom_tools.yaml"  # or "tools/domain_tools.json"
```

## Tool Definition Configuration

### Custom Tools in YAML
Reference external tool definitions:

```yaml
# custom_tools.yaml
tools:
  - name: "book_restaurant"
    description: "Book a restaurant reservation"
    parameters:
      - name: "restaurant"
        type: "str"
        description: "Restaurant name"
        required: true
      - name: "party_size"
        type: "int"
        description: "Number of people"
        required: true
      - name: "date"
        type: "str"
        description: "Reservation date"
        required: true
    returns: "Reservation confirmation details"
    category: "booking"
```

### Inline Custom Tools
Define tools directly in configuration:

```yaml
data_engine:
  custom_tools:
    - name: "send_email"
      description: "Send an email message"
      parameters:
        - name: "recipient"
          type: "str"
          description: "Email recipient address"
          required: true
        - name: "subject"
          type: "str"
          description: "Email subject line"
          required: true
        - name: "body"
          type: "str"
          description: "Email message body"
          required: true
      returns: "Email delivery confirmation"
      category: "communication"
```

## Provider-Specific Configurations

### OpenAI Configuration
```yaml
data_engine:
  provider: "openai"
  model: "gpt-4o"          # or gpt-4o-mini, gpt-4-turbo
  temperature: 0.8
  max_retries: 3
```

### Anthropic Configuration
```yaml
data_engine:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  temperature: 0.7
  max_retries: 2
```

### Local/Ollama Configuration
```yaml
data_engine:
  provider: "ollama"
  model: "llama3.1:8b"     # or mistral:latest, qwen:7b
  temperature: 0.6
  max_retries: 5
```

## Output Format Configuration

### Basic Dataset Output
```yaml
dataset:
  save_as: "agent_dataset.jsonl"
  creation:
    num_steps: 20
    batch_size: 4
    sys_msg: false  # Agent formats typically don't need system messages
```

### With Formatters
```yaml
dataset:
  save_as: "agent_raw.jsonl"
  formatters:
    # Tool-calling format for training
    - name: "tool_calling"
      template: "builtin://tool_calling"
      output: "agent_tool_format.jsonl"
      config:
        system_prompt: "You are a function calling AI model."
        include_tools_in_system: true

    # Conversation format
    - name: "conversation"
      template: "builtin://conversation"
      output: "agent_conversation.jsonl"
      config:
        system_prompt: "You are a helpful assistant with tool access."
```

## Multi-Turn Specific Configuration

### Multi-Turn Agent Settings
```yaml
data_engine:
  conversation_type: "agent_cot_multi_turn"

  # Multi-turn specific parameters
  max_conversation_turns: 6     # Maximum conversation length
  context_retention: true       # Maintain context across turns

dataset:
  creation:
    sys_msg: true  # Multi-turn conversations can use system messages
```

## Quality Control Parameters

### Generation Quality
```yaml
data_engine:
  temperature: 0.8              # Balance creativity and consistency
  max_retries: 3                # Retry failed generations
  reasoning_style: "general"    # or "mathematical", "logical"

  # Validation parameters
  min_tool_usage: 1             # Minimum tools per example
  max_tool_usage: 5             # Maximum tools per example
  require_reasoning: true       # Must include reasoning traces
```

### Topic Quality
```yaml
topic_tree:
  temperature: 0.7              # Moderate creativity for topic generation
  depth: 3                      # Sufficient depth for variety
  degree: 3                     # Good branching for coverage
```

## Environment and Secrets

### API Key Configuration
```yaml
# Set via environment variables:
# OPENAI_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here

data_engine:
  provider: "openai"
  # API key loaded automatically from environment
```

### Advanced Provider Settings
```yaml
data_engine:
  provider: "openai"
  model: "gpt-4o"

  # Advanced settings
  timeout: 60                   # Request timeout in seconds
  rate_limit: 100               # Requests per minute
  batch_size: 5                 # Concurrent requests
```

## Complete Configuration Template

Here's a production-ready template:

```yaml
# Agent Tool-Calling Production Configuration
dataset_system_prompt: "You are an AI assistant with systematic tool reasoning capabilities."

topic_tree:
  topic_prompt: "Professional scenarios requiring intelligent tool usage"
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.7
  depth: 3
  degree: 4

data_engine:
  generation_system_prompt: |
    Create realistic agent scenarios with detailed tool reasoning.
    Focus on systematic thinking and clear parameter construction.

  provider: "openai"
  model: "gpt-4o"
  temperature: 0.8
  max_retries: 3

  conversation_type: "agent_cot_tools"
  reasoning_style: "general"

  available_tools:
    - "get_weather"
    - "search_web"
    - "calculator"
    - "book_restaurant"
    - "analyze_stock"

  max_tools_per_query: 3
  tool_registry_path: "custom_tools.yaml"

dataset:
  creation:
    num_steps: 50
    batch_size: 10
    sys_msg: false

  save_as: "production_agent_dataset.jsonl"

  formatters:
    - name: "tool_calling"
      template: "builtin://tool_calling"
      output: "agent_tool_calling.jsonl"
      config:
        system_prompt: "You are a function calling AI model."
        include_tools_in_system: true

# Optional: Upload to Hugging Face
huggingface:
  repository: "your-username/agent-tool-dataset"
  tags: ["agent", "tool-calling", "reasoning", "cot"]
```

This configuration creates a comprehensive agent tool-calling dataset with proper tool reasoning, multiple output formats, and production-ready settings.