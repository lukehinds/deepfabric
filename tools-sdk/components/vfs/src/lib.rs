use anyhow::Result;
use serde::{Deserialize, Serialize};
use spin_sdk::{
    http::{Request, Response},
    http_component,
    key_value::Store,
};

/// Request payload for tool execution
#[derive(Deserialize)]
struct ExecuteRequest {
    session_id: String,
    tool: String,
    args: serde_json::Value,
}

/// Response payload for tool execution
#[derive(Serialize)]
struct ExecuteResponse {
    success: bool,
    result: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_type: Option<String>,
}

/// Response for health check
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    components: Vec<String>,
}

/// Response for component listing
#[derive(Serialize)]
struct ComponentsResponse {
    components: Vec<String>,
}

#[http_component]
fn handle_request(req: Request) -> Result<Response> {
    let path = req.path();
    let method = req.method().to_string();

    // Handle paths with or without /vfs prefix (Spin wildcard routes include the prefix)
    let normalized_path = path.strip_prefix("/vfs").unwrap_or(path);

    match (method.as_str(), normalized_path) {
        ("POST", "/execute") => handle_execute(req),
        ("DELETE", p) if p.starts_with("/session/") => handle_session_cleanup(req),
        ("GET", "/health") => handle_health(),
        ("GET", "/components") => handle_components(),
        _ => Ok(Response::builder()
            .status(404)
            .header("content-type", "application/json")
            .body(r#"{"error": "Not found"}"#)
            .build()),
    }
}

fn handle_execute(req: Request) -> Result<Response> {
    let body = req.body();
    let request: ExecuteRequest = match serde_json::from_slice(body) {
        Ok(r) => r,
        Err(e) => {
            let response = ExecuteResponse {
                success: false,
                result: format!("Invalid request: {}", e),
                error_type: Some("InvalidRequest".to_string()),
            };
            return Ok(Response::builder()
                .status(400)
                .header("content-type", "application/json")
                .body(serde_json::to_vec(&response)?)
                .build());
        }
    };

    let store = Store::open_default()?;

    let response = match request.tool.as_str() {
        "read_file" => handle_read_file(&store, &request),
        "write_file" => handle_write_file(&store, &request),
        "list_files" => handle_list_files(&store, &request),
        "delete_file" => handle_delete_file(&store, &request),
        _ => ExecuteResponse {
            success: false,
            result: format!("Unknown tool: {}", request.tool),
            error_type: Some("UnknownTool".to_string()),
        },
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

fn handle_read_file(store: &Store, req: &ExecuteRequest) -> ExecuteResponse {
    let file_path = match req.args.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            return ExecuteResponse {
                success: false,
                result: "Missing required argument: file_path".to_string(),
                error_type: Some("InvalidArguments".to_string()),
            }
        }
    };

    let key = format!("{}:{}", req.session_id, file_path);

    match store.get(&key) {
        Ok(Some(content)) => ExecuteResponse {
            success: true,
            result: String::from_utf8_lossy(&content).to_string(),
            error_type: None,
        },
        Ok(None) => ExecuteResponse {
            success: false,
            result: format!("File not found: {}", file_path),
            error_type: Some("FileNotFound".to_string()),
        },
        Err(e) => ExecuteResponse {
            success: false,
            result: format!("IO error: {}", e),
            error_type: Some("IOError".to_string()),
        },
    }
}

fn handle_write_file(store: &Store, req: &ExecuteRequest) -> ExecuteResponse {
    let file_path = match req.args.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            return ExecuteResponse {
                success: false,
                result: "Missing required argument: file_path".to_string(),
                error_type: Some("InvalidArguments".to_string()),
            }
        }
    };

    let content = match req.args.get("content").and_then(|v| v.as_str()) {
        Some(c) => c,
        None => {
            return ExecuteResponse {
                success: false,
                result: "Missing required argument: content".to_string(),
                error_type: Some("InvalidArguments".to_string()),
            }
        }
    };

    let key = format!("{}:{}", req.session_id, file_path);

    // Also track this file in the session's file list for list_files
    let files_key = format!("{}:__files__", req.session_id);
    let mut files: Vec<String> = store
        .get(&files_key)
        .ok()
        .flatten()
        .and_then(|v| serde_json::from_slice(&v).ok())
        .unwrap_or_default();

    if !files.contains(&file_path.to_string()) {
        files.push(file_path.to_string());
        let _ = store.set(&files_key, &serde_json::to_vec(&files).unwrap_or_default());
    }

    match store.set(&key, content.as_bytes()) {
        Ok(()) => ExecuteResponse {
            success: true,
            result: format!("Successfully wrote {} bytes to {}", content.len(), file_path),
            error_type: None,
        },
        Err(e) => ExecuteResponse {
            success: false,
            result: format!("IO error: {}", e),
            error_type: Some("IOError".to_string()),
        },
    }
}

fn handle_list_files(store: &Store, req: &ExecuteRequest) -> ExecuteResponse {
    let files_key = format!("{}:__files__", req.session_id);

    match store.get(&files_key) {
        Ok(Some(data)) => {
            let files: Vec<String> = serde_json::from_slice(&data).unwrap_or_default();
            ExecuteResponse {
                success: true,
                result: serde_json::to_string(&files).unwrap_or_else(|_| "[]".to_string()),
                error_type: None,
            }
        }
        Ok(None) => ExecuteResponse {
            success: true,
            result: "[]".to_string(),
            error_type: None,
        },
        Err(e) => ExecuteResponse {
            success: false,
            result: format!("IO error: {}", e),
            error_type: Some("IOError".to_string()),
        },
    }
}

fn handle_delete_file(store: &Store, req: &ExecuteRequest) -> ExecuteResponse {
    let file_path = match req.args.get("file_path").and_then(|v| v.as_str()) {
        Some(p) => p,
        None => {
            return ExecuteResponse {
                success: false,
                result: "Missing required argument: file_path".to_string(),
                error_type: Some("InvalidArguments".to_string()),
            }
        }
    };

    let key = format!("{}:{}", req.session_id, file_path);

    // Check if file exists first
    match store.get(&key) {
        Ok(Some(_)) => {
            // Delete the file
            if let Err(e) = store.delete(&key) {
                return ExecuteResponse {
                    success: false,
                    result: format!("IO error: {}", e),
                    error_type: Some("IOError".to_string()),
                };
            }

            // Remove from file list
            let files_key = format!("{}:__files__", req.session_id);
            if let Ok(Some(data)) = store.get(&files_key) {
                let mut files: Vec<String> = serde_json::from_slice(&data).unwrap_or_default();
                files.retain(|f| f != file_path);
                let _ = store.set(&files_key, &serde_json::to_vec(&files).unwrap_or_default());
            }

            ExecuteResponse {
                success: true,
                result: format!("Successfully deleted {}", file_path),
                error_type: None,
            }
        }
        Ok(None) => ExecuteResponse {
            success: false,
            result: format!("File not found: {}", file_path),
            error_type: Some("FileNotFound".to_string()),
        },
        Err(e) => ExecuteResponse {
            success: false,
            result: format!("IO error: {}", e),
            error_type: Some("IOError".to_string()),
        },
    }
}

fn handle_session_cleanup(req: Request) -> Result<Response> {
    let path = req.path();
    let session_id = path.strip_prefix("/session/").unwrap_or("");

    if session_id.is_empty() {
        return Ok(Response::builder()
            .status(400)
            .header("content-type", "application/json")
            .body(r#"{"error": "Missing session_id"}"#)
            .build());
    }

    let store = Store::open_default()?;

    // Get the file list for this session
    let files_key = format!("{}:__files__", session_id);
    let files: Vec<String> = store
        .get(&files_key)
        .ok()
        .flatten()
        .and_then(|v| serde_json::from_slice(&v).ok())
        .unwrap_or_default();

    // Delete all files
    for file in &files {
        let key = format!("{}:{}", session_id, file);
        let _ = store.delete(&key);
    }

    // Delete the file list
    let _ = store.delete(&files_key);

    let response = serde_json::json!({
        "success": true,
        "deleted_files": files.len()
    });

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

fn handle_health() -> Result<Response> {
    let response = HealthResponse {
        status: "healthy".to_string(),
        components: vec!["vfs".to_string()],
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}

fn handle_components() -> Result<Response> {
    let response = ComponentsResponse {
        components: vec!["vfs".to_string()],
    };

    Ok(Response::builder()
        .status(200)
        .header("content-type", "application/json")
        .body(serde_json::to_vec(&response)?)
        .build())
}
