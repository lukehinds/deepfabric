# Mock Tools Component

A generic mock tool execution component that loads tool definitions at runtime using the [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) schema format.

## Features

- Load tool schemas from JSON/YAML files or directly from MCP servers
- Execute tools with mock responses (with `{{placeholder}}` interpolation)
- Update mock responses for individual tools after loading
- **Fixtures**: Define different responses based on input arguments (e.g., different content for different file paths)
- Persistent storage via Spin KV store
- Compatible with MCP `tools/list` response format

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mock/load-schema` | POST | Load tool definitions from JSON/YAML body |
| `/mock/pull` | POST | Pull tools from an MCP server via HTTP |
| `/mock/execute` | POST | Execute a loaded tool |
| `/mock/update-response` | POST | Update mock response for a specific tool |
| `/mock/add-fixture` | POST | Add argument-specific mock response |
| `/mock/list-tools` | GET | List all loaded tools |
| `/mock/clear` | POST | Clear all loaded tools |

## Quick Start

### 1. Load tools from an MCP server (stdio)

```bash
# Load tools from GitHub MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | docker run -i --rm \
      -e GITHUB_PERSONAL_ACCESS_TOKEN=your_token \
      ghcr.io/github/github-mcp-server 2>/dev/null \
  | head -1 \
  | jq '.result' \
  | curl -X POST http://localhost:3000/mock/load-schema \
      -H "Content-Type: application/json" --data-binary @-
```

### 2. Update mock responses for tools

```bash
# Set a custom mock response for get_file_contents
curl -X POST http://localhost:3000/mock/update-response \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_file_contents",
    "mockResponse": {
      "path": "{{path}}",
      "content": "# {{path}}\n\nThis is mock content for {{owner}}/{{repo}}",
      "encoding": "utf-8",
      "sha": "abc123def456"
    }
  }'

# Set mock response for list_issues
curl -X POST http://localhost:3000/mock/update-response \
  -H "Content-Type: application/json" \
  -d '{
    "name": "list_issues",
    "mockResponse": {
      "total_count": 2,
      "items": [
        {"number": 1, "title": "First issue", "state": "open"},
        {"number": 2, "title": "Second issue", "state": "closed"}
      ]
    }
  }'

# Set mock response for create_issue
curl -X POST http://localhost:3000/mock/update-response \
  -H "Content-Type: application/json" \
  -d '{
    "name": "create_issue",
    "mockResponse": {
      "id": 123456,
      "number": 42,
      "title": "{{title}}",
      "state": "open",
      "html_url": "https://github.com/{{owner}}/{{repo}}/issues/42"
    }
  }'
```

### 3. Execute tools

```bash
curl -X POST http://localhost:3000/mock/execute \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_file_contents",
    "arguments": {"owner": "octocat", "repo": "hello-world", "path": "README.md"}
  }'
```

Response:
```json
{
  "result": {
    "path": "README.md",
    "content": "# README.md\n\nThis is mock content for octocat/hello-world",
    "encoding": "utf-8",
    "sha": "abc123def456"
  }
}
```

## How It Works

### What the MCP Server Provides

When you load tools from an MCP server (like GitHub's), you get **tool definitions only** - the tool names, descriptions, and input schemas (what arguments they accept). **No mock responses are included.**

Example of what's loaded from the MCP server:
```json
{
  "tools": [
    {
      "name": "get_file_contents",
      "description": "Get file contents from a repository",
      "inputSchema": {
        "type": "object",
        "properties": {
          "owner": { "type": "string" },
          "repo": { "type": "string" },
          "path": { "type": "string" }
        },
        "required": ["owner", "repo", "path"]
      }
    }
  ]
}
```

### What You Define

After loading tools, **you** define what mock responses to return using:

1. **`/mock/update-response`** - Set a default response template for a tool
2. **`/mock/add-fixture`** - Set argument-specific responses (e.g., different content for different files)

The `response` field in fixtures and `mockResponse` in update-response are **entirely user-defined** - you decide what JSON structure to return. This lets you mimic real API responses or return simplified test data.

## Schema Format

Tools use the [MCP Tool schema](https://modelcontextprotocol.io/specification/2025-06-18/schema#tool). The `mockResponse` field is a custom extension (not part of MCP spec) that you can include when loading from a JSON/YAML file:

```json
{
  "tools": [
    {
      "name": "get_file_contents",
      "description": "Get file contents from a repository",
      "inputSchema": {
        "type": "object",
        "properties": {
          "owner": { "type": "string" },
          "repo": { "type": "string" },
          "path": { "type": "string" }
        },
        "required": ["owner", "repo", "path"]
      },
      "mockResponse": {
        "path": "{{path}}",
        "content": "Mock content for {{path}} in {{owner}}/{{repo}}"
      }
    }
  ]
}
```

Use `{{placeholder}}` syntax to interpolate input arguments into responses.

## Placeholder Interpolation

Mock responses support `{{placeholder}}` syntax to insert argument values:

```json
{
  "mockResponse": {
    "message": "Created issue '{{title}}' in {{owner}}/{{repo}}",
    "url": "https://github.com/{{owner}}/{{repo}}/issues/{{number}}"
  }
}
```

When called with `{"title": "Bug fix", "owner": "acme", "repo": "app", "number": 42}`:

```json
{
  "message": "Created issue 'Bug fix' in acme/app",
  "url": "https://github.com/acme/app/issues/42"
}
```

## Fixtures (Argument-Specific Responses)

Fixtures allow different mock responses based on input argument values. This is useful when you want different content returned for different file paths, repositories, etc.

### Adding Fixtures

```bash
# Add fixture for README.md
curl -X POST http://localhost:3000/mock/add-fixture \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_file_contents",
    "match": {"path": "README.md"},
    "response": {"content": "# README\n\nThis is the readme file.", "path": "README.md"}
  }'

# Add fixture for test.txt
curl -X POST http://localhost:3000/mock/add-fixture \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_file_contents",
    "match": {"path": "test.txt"},
    "response": {"content": "Test file content here.", "path": "test.txt"}
  }'

# Add fixture with multiple match criteria (more specific)
curl -X POST http://localhost:3000/mock/add-fixture \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_file_contents",
    "match": {"owner": "acme", "repo": "myapp", "path": "config.json"},
    "response": {"content": "{\"version\": \"1.0\"}", "path": "config.json"}
  }'
```

### How Fixtures Work

1. When executing a tool, fixtures are checked first
2. If multiple fixtures match, the most specific one wins (most match criteria)
3. If no fixture matches, falls back to `mockResponse` template
4. If no `mockResponse`, returns default response

### Example Usage

```bash
# Request README.md - returns fixture response
curl -X POST http://localhost:3000/mock/execute \
  -H "Content-Type: application/json" \
  -d '{"name": "get_file_contents", "arguments": {"owner": "test", "repo": "repo", "path": "README.md"}}'
# Returns: {"result":{"content":"# README\n\nThis is the readme file.","path":"README.md"}}

# Request test.txt - returns different fixture response
curl -X POST http://localhost:3000/mock/execute \
  -H "Content-Type: application/json" \
  -d '{"name": "get_file_contents", "arguments": {"owner": "test", "repo": "repo", "path": "test.txt"}}'
# Returns: {"result":{"content":"Test file content here.","path":"test.txt"}}

# Request unknown file - falls back to mockResponse template
curl -X POST http://localhost:3000/mock/execute \
  -H "Content-Type: application/json" \
  -d '{"name": "get_file_contents", "arguments": {"owner": "test", "repo": "repo", "path": "other.txt"}}'
```

## Default Mock Response

If no fixture matches and no `mockResponse` is defined for a tool, it returns:

```json
{
  "tool": "tool_name",
  "description": "Tool description from schema",
  "input_received": { ... },
  "mock_result": "Successfully executed tool_name",
  "status": "success"
}
```

## Loading from MCP Servers

### HTTP-based MCP servers

If your MCP server supports HTTP transport:

```bash
curl -X POST http://localhost:3000/mock/pull \
  -H "Content-Type: application/json" \
  -d '{"url": "http://localhost:8080/mcp"}'
```

### Stdio-based MCP servers

Most MCP servers use stdio transport. Use this pattern:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | <mcp-server-command> 2>/dev/null \
  | head -1 \
  | jq '.result' \
  | curl -X POST http://localhost:3000/mock/load-schema \
      -H "Content-Type: application/json" --data-binary @-
```

Examples:

```bash
# GitHub MCP server (Docker)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=xxx ghcr.io/github/github-mcp-server 2>/dev/null \
  | head -1 | jq '.result' \
  | curl -X POST http://localhost:3000/mock/load-schema -H "Content-Type: application/json" --data-binary @-

# GitHub MCP server (npx)
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | npx -y @modelcontextprotocol/server-github 2>/dev/null \
  | head -1 | jq '.result' \
  | curl -X POST http://localhost:3000/mock/load-schema -H "Content-Type: application/json" --data-binary @-
```

## Complete Workflow Example

```bash
# 1. Start the spin server
cd tools-sdk && spin up

# 2. Load tools from GitHub MCP server
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | docker run -i --rm -e GITHUB_PERSONAL_ACCESS_TOKEN=xxx ghcr.io/github/github-mcp-server 2>/dev/null \
  | head -1 | jq '.result' \
  | curl -X POST http://localhost:3000/mock/load-schema -H "Content-Type: application/json" --data-binary @-

# 3. Check loaded tools
curl http://localhost:3000/mock/list-tools | jq '.tools | length'

# 4. Set custom mock responses
curl -X POST http://localhost:3000/mock/update-response \
  -H "Content-Type: application/json" \
  -d '{
    "name": "get_file_contents",
    "mockResponse": {
      "path": "{{path}}",
      "content": "# Mock README\nContent for {{owner}}/{{repo}}",
      "sha": "mock123"
    }
  }'

# 5. Execute mock tools
curl -X POST http://localhost:3000/mock/execute \
  -H "Content-Type: application/json" \
  -d '{"name": "get_file_contents", "arguments": {"owner": "test", "repo": "repo", "path": "README.md"}}'

# 6. Clear when done
curl -X POST http://localhost:3000/mock/clear
```

## Use Cases

- **Synthetic dataset generation**: Generate training data for LLM tool-calling without hitting real APIs
- **Testing**: Test tool integrations without external dependencies
- **Development**: Prototype tool workflows before implementing real backends
- **Cost savings**: Avoid API costs during development/testing phases
