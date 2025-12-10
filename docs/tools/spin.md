# Spin Framework

[Spin](https://www.fermyon.com/spin) is a WebAssembly framework for building serverless applications. DeepFabric uses Spin to run tool execution in isolated sandboxes.

## Installation

### macOS

```bash
brew install fermyon/tap/spin
```

### Linux

```bash
curl -fsSL https://developer.fermyon.com/downloads/install.sh | bash
sudo mv spin /usr/local/bin/
```

### Verify

```bash
spin --version
```

## Building Components

Navigate to the tools-sdk directory and build:

```bash
cd tools-sdk
spin build
```

This compiles all Rust components to WebAssembly. Build output:
- `components/vfs/target/wasm32-wasip1/release/vfs.wasm`
- `components/mock/target/wasm32-wasip1/release/mock.wasm`

## Running the Service

```bash
spin up
```

The service starts at `http://localhost:3000`.

### Custom Port

```bash
spin up --listen 0.0.0.0:8080
```

### Background Mode

```bash
spin up --background
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/vfs/execute` | POST | Execute VFS tool |
| `/vfs/health` | GET | Health check |
| `/vfs/components` | GET | List VFS tools |
| `/vfs/session/{id}` | DELETE | Clean up session |
| `/mock/load-schema` | POST | Load tool definitions |
| `/mock/execute` | POST | Execute mock tool |
| `/mock/list-tools` | GET | List loaded tools |

## Testing

Verify the service is running:

```bash
curl http://localhost:3000/vfs/health
# {"status":"healthy","components":["vfs"]}
```

Execute a tool:

```bash
curl -X POST http://localhost:3000/vfs/execute \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-123",
    "tool": "write_file",
    "args": {"file_path": "hello.txt", "content": "Hello, World!"}
  }'
# {"success":true,"result":"Successfully wrote 13 bytes to hello.txt"}
```

## Configuration

The `spin.toml` file defines components and routes:

```toml
spin_manifest_version = 2

[application]
name = "deepfabric-tools"
version = "0.1.0"

[[trigger.http]]
route = "/vfs/..."
component = "vfs"

[component.vfs]
source = "components/vfs/target/wasm32-wasip1/release/vfs.wasm"
key_value_stores = ["default"]
```

## Troubleshooting

**Port already in use**
```bash
# Find process
lsof -i :3000
# Kill it
kill -9 <PID>
```

**Build failures**
```bash
# Ensure Rust WASM target is installed
rustup target add wasm32-wasip1
```

**Permission errors**
```bash
# Check spin binary location
which spin
# Ensure it's executable
chmod +x $(which spin)
```
