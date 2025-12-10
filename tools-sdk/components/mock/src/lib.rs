use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use spin_sdk::{
    http::{IntoResponse, Method, Request, Response},
    http_component,
    key_value::Store,
};
use std::collections::HashMap;

const TOOLS_KEY: &str = "mock_tools";

/// Schema definition - supports MCP format
/// Can load either:
/// - Array of tools: [{ "name": "...", "inputSchema": {...} }, ...]
/// - Object with tools array: { "tools": [...] }
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(untagged)]
enum Schema {
    /// MCP tools/list response format: { "tools": [...] }
    WithWrapper { tools: Vec<ToolDefinition> },
    /// Direct array of tools: [...]
    Direct(Vec<ToolDefinition>),
}

impl Schema {
    fn into_tools(self) -> Vec<ToolDefinition> {
        match self {
            Schema::WithWrapper { tools } => tools,
            Schema::Direct(tools) => tools,
        }
    }
}

/// MCP Tool Definition
/// https://modelcontextprotocol.io/specification/2025-06-18/schema#tool
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolDefinition {
    /// Tool name
    name: String,
    /// Human-readable description
    #[serde(default)]
    description: Option<String>,
    /// JSON Schema for tool input (MCP uses camelCase)
    #[serde(default)]
    input_schema: Option<Value>,
    /// Optional annotations (MCP spec)
    #[serde(default)]
    annotations: Option<ToolAnnotations>,
    /// Custom: mock response template (not part of MCP spec)
    #[serde(default)]
    mock_response: Option<Value>,
    /// Custom: fixtures for argument-specific responses
    /// Key is a match pattern (e.g., "path=README.md" or "owner=test,repo=myrepo")
    #[serde(default)]
    mock_fixtures: HashMap<String, Value>,
}

/// MCP Tool Annotations
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
struct ToolAnnotations {
    /// Human-readable title
    #[serde(default)]
    title: Option<String>,
    /// If true, tool doesn't modify state
    #[serde(default)]
    read_only_hint: Option<bool>,
    /// If true, tool may be destructive
    #[serde(default)]
    destructive_hint: Option<bool>,
    /// If true, may take a long time
    #[serde(default)]
    idempotent_hint: Option<bool>,
    /// If true, interacts with external world
    #[serde(default)]
    open_world_hint: Option<bool>,
}

/// Request to execute a tool
#[derive(Debug, Deserialize)]
struct ExecuteRequest {
    name: String,
    arguments: Option<Value>,
}

/// Response from tool execution
#[derive(Debug, Serialize)]
struct ExecuteResponse {
    result: Value,
}

/// Response from schema loading
#[derive(Debug, Serialize)]
struct LoadResponse {
    loaded: usize,
    tools: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Request to update a tool's mock response
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct UpdateResponseRequest {
    /// Tool name to update
    name: String,
    /// Mock response template to set
    mock_response: Value,
}

/// Response from updating a tool
#[derive(Debug, Serialize)]
struct UpdateResponse {
    updated: String,
}

/// Request to add a fixture for a tool
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct AddFixtureRequest {
    /// Tool name
    name: String,
    /// Match criteria - map of argument name to expected value
    /// e.g., {"path": "README.md"} or {"owner": "test", "repo": "myrepo"}
    #[serde(rename = "match")]
    match_args: HashMap<String, Value>,
    /// Response to return when match criteria are met
    response: Value,
}

/// Response from adding a fixture
#[derive(Debug, Serialize)]
struct AddFixtureResponse {
    tool: String,
    fixture_key: String,
    total_fixtures: usize,
}

/// Request to pull tools from an MCP server via HTTP
#[derive(Debug, Deserialize)]
struct PullRequest {
    /// URL of the MCP server (HTTP endpoint)
    url: String,
}

/// MCP JSON-RPC response wrapper
#[derive(Debug, Deserialize)]
struct McpResponse {
    result: Option<McpToolsResult>,
    error: Option<Value>,
}

#[derive(Debug, Deserialize)]
struct McpToolsResult {
    tools: Vec<ToolDefinition>,
}

#[http_component]
async fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    let path = req.path();

    let response = if path.ends_with("/load-schema") {
        handle_load_schema(&req)
    } else if path.ends_with("/pull") {
        handle_pull(&req).await
    } else if path.ends_with("/execute") {
        handle_execute(&req)
    } else if path.ends_with("/update-response") {
        handle_update_response(&req)
    } else if path.ends_with("/add-fixture") {
        handle_add_fixture(&req)
    } else if path.ends_with("/list-tools") {
        handle_list_tools()
    } else if path.ends_with("/clear") {
        handle_clear()
    } else {
        Ok(Response::builder()
            .status(404)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&ErrorResponse {
                error: format!("Not found: {}", path),
            })?)
            .build())
    };

    response
}

fn handle_load_schema(req: &Request) -> Result<Response> {
    let body = req.body();

    // Try parsing as JSON first, then YAML
    let schema: Schema = serde_json::from_slice(body)
        .or_else(|_| serde_yaml::from_slice(body))
        .map_err(|e| anyhow::anyhow!("Failed to parse schema: {}", e))?;

    // Convert to tools vec (handles both wrapped and direct formats)
    let tools = schema.into_tools();

    // Open KV store and load existing tools
    let store = Store::open_default()?;
    let mut tools_map: HashMap<String, ToolDefinition> = match store.get(TOOLS_KEY)? {
        Some(data) => serde_json::from_slice(&data).unwrap_or_default(),
        None => HashMap::new(),
    };

    // Add/update tools from schema
    let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
    for tool in tools {
        tools_map.insert(tool.name.clone(), tool);
    }

    // Persist to KV store
    let serialized = serde_json::to_vec(&tools_map)?;
    store.set(TOOLS_KEY, &serialized)?;

    let response = LoadResponse {
        loaded: tool_names.len(),
        tools: tool_names,
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

/// Pull tools from an MCP server via HTTP
/// Sends JSON-RPC tools/list request and loads the response
async fn handle_pull(req: &Request) -> Result<Response> {
    let body = req.body();
    let pull_req: PullRequest = serde_json::from_slice(body)
        .map_err(|e| anyhow::anyhow!("Failed to parse pull request: {}", e))?;

    // Build JSON-RPC request for tools/list
    let jsonrpc_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list"
    });

    // Make outbound HTTP request to MCP server
    let outbound_req = Request::builder()
        .method(Method::Post)
        .uri(&pull_req.url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_vec(&jsonrpc_request)?)
        .build();

    let outbound_resp: Response = spin_sdk::http::send(outbound_req).await?;

    // Check response status
    if *outbound_resp.status() != 200 {
        return Ok(Response::builder()
            .status(502)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&ErrorResponse {
                error: format!(
                    "MCP server returned status {}: {}",
                    outbound_resp.status(),
                    String::from_utf8_lossy(outbound_resp.body())
                ),
            })?)
            .build());
    }

    // Parse MCP JSON-RPC response
    let mcp_resp: McpResponse = serde_json::from_slice(outbound_resp.body())
        .map_err(|e| anyhow::anyhow!("Failed to parse MCP response: {}", e))?;

    // Check for JSON-RPC error
    if let Some(err) = mcp_resp.error {
        return Ok(Response::builder()
            .status(502)
            .header("content-type", "application/json")
            .body(serde_json::to_vec(&ErrorResponse {
                error: format!("MCP server error: {}", err),
            })?)
            .build());
    }

    // Extract tools from result
    let tools = match mcp_resp.result {
        Some(result) => result.tools,
        None => {
            return Ok(Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: "MCP response missing result".to_string(),
                })?)
                .build());
        }
    };

    // Load tools into KV store
    let store = Store::open_default()?;
    let mut tools_map: HashMap<String, ToolDefinition> = match store.get(TOOLS_KEY)? {
        Some(data) => serde_json::from_slice(&data).unwrap_or_default(),
        None => HashMap::new(),
    };

    let tool_names: Vec<String> = tools.iter().map(|t| t.name.clone()).collect();
    for tool in tools {
        tools_map.insert(tool.name.clone(), tool);
    }

    let serialized = serde_json::to_vec(&tools_map)?;
    store.set(TOOLS_KEY, &serialized)?;

    let response = LoadResponse {
        loaded: tool_names.len(),
        tools: tool_names,
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

fn handle_execute(req: &Request) -> Result<Response> {
    let body = req.body();
    let exec_req: ExecuteRequest = serde_json::from_slice(body)
        .map_err(|e| anyhow::anyhow!("Failed to parse request: {}", e))?;

    // Load tools from KV store
    let store = Store::open_default()?;
    let tools_map: HashMap<String, ToolDefinition> = match store.get(TOOLS_KEY)? {
        Some(data) => serde_json::from_slice(&data)?,
        None => {
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: "No tools loaded. POST schema to /load-schema first".to_string(),
                })?)
                .build());
        }
    };

    // Find the tool
    let tool = match tools_map.get(&exec_req.name) {
        Some(t) => t,
        None => {
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: format!("Unknown tool: {}", exec_req.name),
                })?)
                .build());
        }
    };

    // Generate mock response
    let result = generate_mock_response(tool, &exec_req.arguments);

    let response = ExecuteResponse { result };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

/// Update a tool's mock response template
fn handle_update_response(req: &Request) -> Result<Response> {
    let body = req.body();
    let update_req: UpdateResponseRequest = serde_json::from_slice(body)
        .map_err(|e| anyhow::anyhow!("Failed to parse update request: {}", e))?;

    // Load tools from KV store
    let store = Store::open_default()?;
    let mut tools_map: HashMap<String, ToolDefinition> = match store.get(TOOLS_KEY)? {
        Some(data) => serde_json::from_slice(&data)?,
        None => {
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: "No tools loaded. POST schema to /load-schema first".to_string(),
                })?)
                .build());
        }
    };

    // Find and update the tool
    match tools_map.get_mut(&update_req.name) {
        Some(tool) => {
            tool.mock_response = Some(update_req.mock_response);
        }
        None => {
            return Ok(Response::builder()
                .status(404)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: format!("Tool not found: {}", update_req.name),
                })?)
                .build());
        }
    };

    // Persist to KV store
    let serialized = serde_json::to_vec(&tools_map)?;
    store.set(TOOLS_KEY, &serialized)?;

    let response = UpdateResponse {
        updated: update_req.name,
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

/// Add a fixture (argument-specific mock response) for a tool
fn handle_add_fixture(req: &Request) -> Result<Response> {
    let body = req.body();
    let fixture_req: AddFixtureRequest = serde_json::from_slice(body)
        .map_err(|e| anyhow::anyhow!("Failed to parse add-fixture request: {}", e))?;

    // Load tools from KV store
    let store = Store::open_default()?;
    let mut tools_map: HashMap<String, ToolDefinition> = match store.get(TOOLS_KEY)? {
        Some(data) => serde_json::from_slice(&data)?,
        None => {
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: "No tools loaded. POST schema to /load-schema first".to_string(),
                })?)
                .build());
        }
    };

    // Build fixture key from match args (sorted for consistency)
    let mut match_parts: Vec<String> = fixture_req
        .match_args
        .iter()
        .map(|(k, v)| {
            let v_str = match v {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            format!("{}={}", k, v_str)
        })
        .collect();
    match_parts.sort();
    let fixture_key = match_parts.join(",");

    // Find and update the tool
    let total_fixtures = match tools_map.get_mut(&fixture_req.name) {
        Some(tool) => {
            tool.mock_fixtures.insert(fixture_key.clone(), fixture_req.response);
            tool.mock_fixtures.len()
        }
        None => {
            return Ok(Response::builder()
                .status(404)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&ErrorResponse {
                    error: format!("Tool not found: {}", fixture_req.name),
                })?)
                .build());
        }
    };

    // Persist to KV store
    let serialized = serde_json::to_vec(&tools_map)?;
    store.set(TOOLS_KEY, &serialized)?;

    let response = AddFixtureResponse {
        tool: fixture_req.name,
        fixture_key,
        total_fixtures,
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

fn handle_list_tools() -> Result<Response> {
    let store = Store::open_default()?;
    let tools_map: HashMap<String, ToolDefinition> = match store.get(TOOLS_KEY)? {
        Some(data) => serde_json::from_slice(&data)?,
        None => HashMap::new(),
    };

    // Return full tool definitions (name, description, inputSchema) for LLM consumption
    let tools: Vec<Value> = tools_map
        .values()
        .map(|t| {
            json!({
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            })
        })
        .collect();

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&json!({ "tools": tools }))?)
        .build())
}

fn handle_clear() -> Result<Response> {
    let store = Store::open_default()?;
    store.delete(TOOLS_KEY)?;

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&json!({ "cleared": true }))?)
        .build())
}

fn generate_mock_response(tool: &ToolDefinition, args: &Option<Value>) -> Value {
    // First, check if any fixtures match the input arguments
    if !tool.mock_fixtures.is_empty() {
        if let Some(fixture_resp) = find_matching_fixture(tool, args) {
            return interpolate_response(fixture_resp, args);
        }
    }

    // If tool has a custom mock_response defined, use it with interpolation
    if let Some(mock_resp) = &tool.mock_response {
        return interpolate_response(mock_resp.clone(), args);
    }

    // Default mock response based on tool schema
    json!({
        "tool": tool.name,
        "description": tool.description,
        "input_received": args,
        "mock_result": format!("Successfully executed {}", tool.name),
        "status": "success"
    })
}

/// Find a fixture that matches the given arguments
/// Fixtures are matched by checking if all fixture match criteria are present in args
fn find_matching_fixture(tool: &ToolDefinition, args: &Option<Value>) -> Option<Value> {
    let args_map = match args {
        Some(Value::Object(map)) => map,
        _ => return None,
    };

    // Try each fixture and find the best match (most specific = most match criteria)
    let mut best_match: Option<(usize, &Value)> = None;

    for (fixture_key, fixture_response) in &tool.mock_fixtures {
        // Parse the fixture key back into match criteria
        let mut all_match = true;
        let mut match_count = 0;

        for part in fixture_key.split(',') {
            if let Some((key, expected_value)) = part.split_once('=') {
                match_count += 1;
                match args_map.get(key) {
                    Some(actual_value) => {
                        let actual_str = match actual_value {
                            Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        if actual_str != expected_value {
                            all_match = false;
                            break;
                        }
                    }
                    None => {
                        all_match = false;
                        break;
                    }
                }
            }
        }

        if all_match {
            // Keep the most specific match (most criteria)
            match &best_match {
                Some((count, _)) if *count >= match_count => {}
                _ => best_match = Some((match_count, fixture_response)),
            }
        }
    }

    best_match.map(|(_, resp)| resp.clone())
}

fn interpolate_response(template: Value, args: &Option<Value>) -> Value {
    match template {
        Value::String(s) => {
            let mut result = s.clone();
            if let Some(Value::Object(args_map)) = args {
                for (key, value) in args_map {
                    let placeholder = format!("{{{{{}}}}}", key);
                    let replacement = match value {
                        Value::String(v) => v.clone(),
                        other => other.to_string(),
                    };
                    result = result.replace(&placeholder, &replacement);
                }
            }
            Value::String(result)
        }
        Value::Object(map) => {
            let new_map: serde_json::Map<String, Value> = map
                .into_iter()
                .map(|(k, v)| (k, interpolate_response(v, args)))
                .collect();
            Value::Object(new_map)
        }
        Value::Array(arr) => {
            Value::Array(arr.into_iter().map(|v| interpolate_response(v, args)).collect())
        }
        other => other,
    }
}
