# YAML Tool Configuration

DeepFabric's structured tool system supports YAML configuration, allowing you to define custom tools separately from your code.

## Benefits

YAML tool configuration provides several advantages:

- Human-readable tool definitions in structured format
- Version control for tool definitions alongside datasets
- Team collaboration through shared tool configurations
- Separation of tool definitions from generation logic
- Automatic validation via Pydantic schemas

## File Structure

```
your_project/
├── tools/
│   ├── custom_tools.yaml
│   └── domain_tools.json
├── config.yaml
└── generate_dataset.py
```

## Defining Tools in YAML

### Basic Tool Definition

```yaml
# custom_tools.yaml
tools:
  - name: "get_weather"
    description: "Get current weather for a location"
    parameters:
      - name: "location"
        type: "str"
        description: "City name"
        required: true
      - name: "units"
        type: "str"
        description: "Temperature units (celsius/fahrenheit)"
        required: false
        default: "celsius"
    returns: "Weather data with temperature and conditions"
    category: "information"
```

### Advanced Tool with Multiple Parameters

```yaml
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
        description: "Reservation date (YYYY-MM-DD)"
        required: true
      - name: "time"
        type: "str"
        description: "Preferred time (HH:MM)"
        required: true
      - name: "special_requests"
        type: "str"
        description: "Special dietary requirements or requests"
        required: false
        default: ""
    returns: "Reservation confirmation with details"
    category: "booking"
```

## Configuration Usage

### Complete YAML Configuration

```yaml
# config.yaml
dataset_system_prompt: "You are an AI assistant with specialized tools."

data_engine:
  generation_system_prompt: "You excel at tool reasoning and selection."
  provider: "openai"
  model: "gpt-4o"
  conversation_type: "agent_cot_tools"  # For single-turn agent reasoning
  # OR
  # conversation_type: "agent_cot_multi_turn"  # For multi-turn conversations
  tool_registry_path: "tools/custom_tools.yaml"
  available_tools:
    - "get_weather"
    - "book_restaurant"
    - "analyze_stock"

dataset:
  creation:
    num_steps: 5
    batch_size: 1
  save_as: "agent_dataset.jsonl"
```

### Programmatic Usage

```python
from deepfabric import DataSetGenerator
from deepfabric.tools.loader import load_tools_from_file

# Load tools from YAML
custom_tools = load_tools_from_file("tools/custom_tools.yaml")

# Create generator with YAML tools
engine = DataSetGenerator(
    generation_system_prompt="You are an AI assistant...",
    provider="openai",
    model_name="gpt-4o",
    conversation_type="agent_cot_tools",
    tool_registry_path="tools/custom_tools.yaml",
    available_tools=["get_weather", "book_restaurant"],
)
```

## Parameter Types

The following parameter types are supported:

| Type | Description | Example |
|------|-------------|---------|
| `str` | String values | `"Paris"`, `"2024-01-15"` |
| `int` | Integer numbers | `42`, `100` |
| `float` | Decimal numbers | `3.14`, `98.6` |
| `bool` | Boolean values | `true`, `false` |
| `list` | Arrays/lists | `["item1", "item2"]` |
| `dict` | Objects/dictionaries | `{"key": "value"}` |

## Tool Categories

Tools can be organized by category:

- `information` - Data retrieval tools
- `communication` - Messaging and notifications
- `booking` - Reservation and scheduling
- `analysis` - Data processing and analysis
- `computation` - Mathematical operations
- `navigation` - Location and travel
- `productivity` - Task management

## Mixing Default and Custom Tools

```python
engine = DataSetGenerator(
    # ... config ...
    available_tools=[
        "get_weather",      # Default DeepFabric tool
        "search_web",       # Default DeepFabric tool
        "book_restaurant",  # Custom tool from YAML
        "analyze_stock",    # Custom tool from YAML
    ],
    tool_registry_path="tools/custom_tools.yaml",
)
```

## Tool Validation

DeepFabric automatically validates YAML tool definitions. Invalid configurations will produce helpful error messages:

```bash
ConfigurationError: Invalid tool definition: Field 'name' is required
ConfigurationError: Parameter type 'string' not supported. Use 'str'
```

## Best Practices

1. Use descriptive tool names: `book_restaurant` rather than `book_res`
2. Write clear descriptions explaining what each tool does and returns
3. Use consistent categories to group related tools
4. Mark parameters as required or optional appropriately
5. Provide sensible defaults for optional parameters
6. Track tool definitions in version control

## JSON Alternative

Tools can also be defined in JSON format:

```json
{
  "tools": [
    {
      "name": "send_notification",
      "description": "Send a notification to a user",
      "parameters": [
        {
          "name": "user_id",
          "type": "str",
          "description": "User identifier",
          "required": true
        }
      ],
      "returns": "Notification delivery confirmation",
      "category": "communication"
    }
  ]
}
```

## Examples

Complete examples are available in the `examples/` directory:

- `examples/custom_tools.yaml` - YAML tool definitions
- `examples/agent_tool_calling.yaml` - Complete agent configuration
- `examples/agent_tool_calling.py` - Programmatic usage with both approaches

This structured approach transforms tool calling from hardcoded text into a maintainable system that fully leverages Outlines' capabilities while providing complete control over tool definitions.
