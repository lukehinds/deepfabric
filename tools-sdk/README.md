# DeepFabric Tools SDK

WebAssembly-based tool execution service for DeepFabric dataset generation.

## Quick Start

### Prerequisites

- [Spin CLI](https://developer.fermyon.com/spin/v2/install) (v2.0+)
- Rust with `wasm32-wasi` target: `rustup target add wasm32-wasi`
- Python 3.11+ with `componentize-py` for Python components

### Build and Run

```bash
cd tools-sdk

# Build all components
spin build

# Start the service (VFS only)
spin up

# Start with GitHub token for GitHub tools (allows all repos)
SPIN_VARIABLE_GITHUB_TOKEN=ghp_xxx spin up

# Start with repo restrictions (recommended for dataset generation)
SPIN_VARIABLE_GITHUB_TOKEN=ghp_xxx \
SPIN_VARIABLE_ALLOWED_REPOS="myorg/repo1,myorg/repo2" \
spin up

# Enable write operations (disabled by default)
SPIN_VARIABLE_GITHUB_TOKEN=ghp_xxx \
SPIN_VARIABLE_ALLOWED_REPOS="myorg/repo1" \
SPIN_VARIABLE_ALLOW_WRITES=true \
spin up
```

The service will start on `http://localhost:3000`.

### Test the Service

```bash
# VFS Health check
curl http://localhost:3000/vfs/health

# GitHub Health check
curl http://localhost:3000/github/health

# List VFS components
curl http://localhost:3000/vfs/components

# List GitHub tools
curl http://localhost:3000/github/components
```

### VFS Examples

```bash
# Write a file
curl -X POST http://localhost:3000/vfs/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "tool": "write_file",
    "args": {"file_path": "hello.py", "content": "print(\"Hello, World!\")"}
  }'

# Read the file back
curl -X POST http://localhost:3000/vfs/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "tool": "read_file",
    "args": {"file_path": "hello.py"}
  }'

# List files in session
curl -X POST http://localhost:3000/vfs/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session",
    "tool": "list_files",
    "args": {}
  }'

# Cleanup session
curl -X DELETE http://localhost:3000/vfs/session/test-session
```

### GitHub Examples

```bash
# Search repositories
curl -X POST http://localhost:3000/github/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "tool": "gh_search_repositories",
    "args": {"query": "deepfabric language:python", "perPage": 5}
  }'

# Get file contents
curl -X POST http://localhost:3000/github/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "tool": "gh_get_file_contents",
    "args": {"owner": "anthropics", "repo": "courses", "path": "README.md"}
  }'

# List issues
curl -X POST http://localhost:3000/github/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "tool": "gh_list_issues",
    "args": {"owner": "fermyon", "repo": "spin", "state": "open", "perPage": 5}
  }'

# Get commit details
curl -X POST http://localhost:3000/github/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "tool": "gh_get_commit",
    "args": {"owner": "fermyon", "repo": "spin", "sha": "main"}
  }'
```

## HTTP API

Each component has its own route prefix (`/vfs/...`, `/github/...`).

### POST /{component}/execute

Execute a tool.

**Request:**
```json
{
  "session_id": "uuid-here",
  "tool": "read_file",
  "args": {"file_path": "main.py"}
}
```

**Response (success):**
```json
{
  "success": true,
  "result": "file content here"
}
```

**Response (error):**
```json
{
  "success": false,
  "result": "File not found: main.py",
  "error_type": "FileNotFound"
}
```

### DELETE /{component}/session/{session_id}

Clean up all files for a session (VFS only).

### GET /{component}/health

Health check endpoint.

### GET /{component}/components

List available tools for the component.

## Available Components

### VFS (Virtual Filesystem) - `/vfs/...`

In-memory file storage for agent tool tracing. Written in Rust.

**Tools:**
- `read_file` - Read file content
  - Args: `file_path` (string, required)
- `write_file` - Write content to file
  - Args: `file_path` (string, required), `content` (string, required)
- `list_files` - List all files in session
  - Args: none
- `delete_file` - Delete a file
  - Args: `file_path` (string, required)

### GitHub - `/github/...`

Real GitHub API access for repository analysis. Written in Python.

**Configuration:**
- `SPIN_VARIABLE_GITHUB_TOKEN` - GitHub personal access token (required)
- `SPIN_VARIABLE_ALLOWED_REPOS` - Comma-separated list of allowed repos, e.g., `"org/repo1,org/repo2"` (optional, allows all if not set)
- `SPIN_VARIABLE_ALLOW_WRITES` - Set to `"true"` to enable write operations (default: false)

**Safety Features:**
- **Repo Allowlisting**: When `ALLOWED_REPOS` is set, requests to other repos return helpful error messages
- **Write Protection**: Write operations (like `gh_add_issue_comment`) are disabled by default
- **Error Feedback**: Non-allowed repo access returns: `"Repository 'owner/repo' is not in the allowed list. Allowed repositories: repo1, repo2"`

**Tools:**
- `gh_get_file_contents` - Get file or directory contents from a repo
  - Args: `owner` (required), `repo` (required), `path` (optional), `ref` (optional)
- `gh_search_code` - Search code across GitHub
  - Args: `query` (required), `perPage` (optional), `page` (optional)
- `gh_search_repositories` - Search for repositories
  - Args: `query` (required), `perPage` (optional), `page` (optional), `sort` (optional), `order` (optional)
- `gh_list_issues` - List issues in a repository
  - Args: `owner` (required), `repo` (required), `state` (optional), `perPage` (optional)
- `gh_get_issue` - Get issue details
  - Args: `owner` (required), `repo` (required), `issue_number` (required)
- `gh_list_pull_requests` - List PRs in a repository
  - Args: `owner` (required), `repo` (required), `state` (optional), `perPage` (optional)
- `gh_get_pull_request` - Get PR details
  - Args: `owner` (required), `repo` (required), `pullNumber` (required)
- `gh_list_commits` - List commits in a repository
  - Args: `owner` (required), `repo` (required), `sha` (optional), `perPage` (optional)
- `gh_get_commit` - Get commit details
  - Args: `owner` (required), `repo` (required), `sha` (required)
- `gh_list_branches` - List branches in a repository
  - Args: `owner` (required), `repo` (required), `perPage` (optional)
- `gh_add_issue_comment` - Add comment to an issue (requires auth)
  - Args: `owner` (required), `repo` (required), `issue_number` (required), `body` (required)

## Adding Custom Components

### Rust Components

See `components/vfs/src/lib.rs` as a reference.

1. Create a new component directory under `components/`
2. Create `Cargo.toml` with `tools-sdk` dependency
3. Implement the HTTP handler using `#[http_component]`
4. Add the component to `spin.toml`

### Python Components

See `components/github/app.py` as a reference.

1. Create a new component directory under `components/`
2. Create `requirements.txt` with `tools-sdk>=3.0.0`
3. Create `app.py` with `IncomingHandler` class:
   ```python
   from spin_sdk import http

   class IncomingHandler(http.IncomingHandler):
       def handle_request(self, request: http.Request) -> http.Response:
           return http.Response(200, {"content-type": "text/plain"}, b"Hello!")
   ```
4. Add the component to `spin.toml`:
   ```toml
   [[trigger.http]]
   route = "/mycomponent/..."
   component = "mycomponent"

   [component.mycomponent]
   source = "components/mycomponent/app.py"
   allowed_outbound_hosts = ["https://api.example.com"]

   [component.mycomponent.build]
   command = "componentize-py -w spin-http componentize app -o mycomponent.wasm"
   workdir = "components/mycomponent"
   ```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SPIN_VARIABLE_GITHUB_TOKEN` | GitHub personal access token (required for GitHub component) |
| `SPIN_VARIABLE_ALLOWED_REPOS` | Comma-separated allowed repos, e.g., `"org/repo1,org/repo2"` |
| `SPIN_VARIABLE_ALLOW_WRITES` | Set to `"true"` to enable write operations (default: false) |
