# MCP Tool Dataset Examples

This directory contains examples for generating synthetic datasets from MCP (Model Context Protocol) server Tool calling. These configs allow you to create training data that shows how an AI assistant would use various tools provided by an MCP server.

## Firecrawl MCP Server Example

The Firecrawl example demonstrates how to generate training data for web scraping and data extraction tasks.

### Files

- **`firecrawl_tools.yaml`**: Tool definitions for all Firecrawl MCP server tools
- **`firecrawl_config.yaml`**: Complete DeepFabric configuration for generating a dataset
- **`README.md`**: This file

### Quick Start

1. **Generate the dataset:**

```bash
deepfabric start examples/mcp/firecrawl_config.yaml
```

2. **Customize the configuration:**

Edit `firecrawl_config.yaml` to adjust:
- Number of samples (`num_steps` × `batch_size`)
- Agent mode (`single_turn` vs `multi_turn`)
- Model provider and name
- Topic tree for diverse scenarios
- Custom instructions for specific use cases

3. **View the output:**

The generated dataset will be saved to `firecrawl_dataset.jsonl` with examples like:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant with access to web scraping tools..."
    },
    {
      "role": "user",
      "content": "I need to extract product prices from these e-commerce pages"
    },
    {
      "role": "assistant",
      "content": "I'll use the firecrawl_extract tool to get that information",
      "tool_calls": [{
        "name": "firecrawl_extract",
        "arguments": {
          "urls": ["https://example.com/product1", "https://example.com/product2"],
          "prompt": "Extract product name, price, and availability",
          "schema": {
            "type": "object",
            "properties": {
              "name": {"type": "string"},
              "price": {"type": "number"},
              "in_stock": {"type": "boolean"}
            }
          }
        }
      }]
    }
  ],
  "available_tools": [...],
  "tool_context": {...}
}
```

### Configuration Options

#### Agent Modes

- **`single_turn`**: One user request, one assistant response with tool calls
- **`multi_turn`**: Extended conversations with multiple tool calls and results

#### Reasoning Styles

- **`freetext`**: Natural language reasoning
- **`structured`**: Step-by-step reasoning traces
- **`hybrid`**: Both structured and natural reasoning (recommended)

#### Tool Selection

Customize which tools to include:

```yaml
data_engine:
  args:
    available_tools: ["firecrawl_scrape", "firecrawl_extract"]  # Only these tools
    # OR
    available_tools: []  # All tools from registry
```

### Adding Custom Tools

You can add your own tool definitions to `firecrawl_tools.yaml` or create a new YAML file:

```yaml
tools:
  - name: my_custom_tool
    description: "What this tool does"
    category: "tool_category"
    parameters:
      - name: param_name
        type: str
        description: "Parameter description"
        required: true
    returns: "What the tool returns"
```

### Example Use Cases

The default configuration generates diverse scenarios including:

- E-commerce product extraction
- News article scraping
- Website mapping and discovery
- Multi-page crawling
- Structured data extraction with schemas
- Web search integration

### Advanced Usage

#### Upload to Hugging Face

Uncomment the `huggingface` section in `firecrawl_config.yaml`:

```yaml
huggingface:
  repo_id: "your-username/firecrawl-tools-dataset"
  private: false
```

Then set your HuggingFace token:

```bash
export HF_TOKEN="your_token_here"
deepfabric start examples/mcp/firecrawl_config.yaml
```

#### Override Settings via CLI

```bash
deepfabric start examples/mcp/firecrawl_config.yaml \
  --model gpt-4 \
  --temperature 0.9 \
  --batch-size 10 \
  --num-steps 20
```

## Creating Datasets for Other MCP Servers

The same approach works for any MCP server:

1. Create a `<server_name>_tools.yaml` with tool definitions
2. Copy and modify `firecrawl_config.yaml`
3. Update the `dataset_system_prompt` to reflect the tool's purpose
4. Adjust the topic tree root prompt for relevant scenarios

### Other MCP Server Examples

You can use this pattern for:

- **Filesystem MCP**: File operations and directory management
- **GitHub MCP**: Repository interactions and code management
- **Slack MCP**: Messaging and workspace operations
- **Memory MCP**: Persistent storage and retrieval
- **Any custom MCP server**: Define the tools and generate training data

## Tips for Quality Datasets

1. **Realistic scenarios**: Use the `instructions` field to guide realistic usage
2. **Diverse topics**: Use a good topic tree to cover different use cases
3. **Appropriate tools**: Let the model choose tools naturally, don't force usage
4. **Good schemas**: For extraction tools, include varied but realistic schemas
5. **Multi-turn for complexity**: Use `multi_turn` mode for tasks requiring multiple steps

## Notes

- **No actual API calls**: Tools are never executed - only schemas for training data generation
- **Mock responses**: The LLM generates realistic tool call patterns based on descriptions
- **Flexible schemas**: Tool definitions follow the OpenAI function calling format
- **Training data only**: This is for creating datasets to train other models, not for actual tool execution
