# Error Codes

DeepFabric uses standardized error codes to provide consistent, actionable feedback during dataset generation. Error codes appear in the TUI Events panel and help diagnose issues without enabling debug mode.

## Overview

Error codes follow the format `DF-XNN`:

- **DF**: DeepFabric prefix
- **X**: Category letter (R=Rate limit, A=Auth/API, N=Network, P=Parse, T=Tool, X=Unknown)
- **NN**: Number within category

Errors are classified by severity:

- **Sample-level**: Generation continues; the failed sample is skipped
- **Fatal**: CLI exits immediately; requires configuration fix

## TUI Display

During generation, errors appear in the Events panel with colored indicators:

```
X [42] DF-R01 Rate limit (RPM) - retry 3s
✓ Generated +2 samples
X [45] DF-P01 JSON parse error
✓ Generated +3 samples
```

- Red **X** indicates an error with the error code and brief description
- Green **✓** indicates successful sample generation
- The number in brackets `[42]` is the sample index

## Error Code Reference

### Rate Limit Errors (DF-R0x)

Rate limit errors occur when the LLM provider throttles requests. These are typically retryable.

| Code | Message | Severity | Description |
|------|---------|----------|-------------|
| `DF-R01` | Rate limit (RPM) | Sample | Requests per minute limit exceeded. Provider is throttling requests. |
| `DF-R02` | Rate limit (daily) | Sample | Daily quota exhausted. Resets at midnight (provider timezone). |
| `DF-R03` | Rate limit (tokens) | Sample | Token per minute limit exceeded. |
| `DF-R04` | Rate limit | Sample | Generic rate limit error from provider. |

**Common causes:**

- Too many concurrent requests (reduce `batch_size`)
- High request volume (add `rate_limit` configuration)
- Free tier limits (upgrade API plan or switch providers)

**Solutions:**

```yaml
data_engine:
  # Reduce concurrency
  batch_size: 1

  # Add rate limiting
  rate_limit:
    base_delay: 3.0
    max_delay: 120.0
    max_retries: 5
```

### Auth/API Errors (DF-A0x)

Authentication and API configuration errors.

| Code | Message | Severity | Description |
|------|---------|----------|-------------|
| `DF-A01` | Auth failed | Fatal | Authentication failed. Check your API key environment variable. |
| `DF-A02` | Model not found | Fatal | The specified model does not exist or is not accessible. |
| `DF-A03` | API error | Sample | Generic API error from the provider. |

**Common causes:**

- Missing or invalid API key
- Incorrect model name
- API key lacks required permissions

**Solutions:**

```bash
# Ensure API key is set
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export ANTHROPIC_API_KEY=...

# Verify model name matches provider's naming
# OpenAI: gpt-4o, gpt-4o-mini
# Gemini: gemini-2.0-flash, gemini-1.5-pro
# Anthropic: claude-3-5-sonnet-20241022
```

### Network Errors (DF-N0x)

Network and connectivity issues.

| Code | Message | Severity | Description |
|------|---------|----------|-------------|
| `DF-N01` | Network error | Sample | Connection failed. Check your internet connection. |
| `DF-N02` | Timeout | Sample | Request timed out waiting for provider response. |
| `DF-N03` | Service unavailable | Sample | Provider service temporarily unavailable (503/502). |

**Common causes:**

- Internet connectivity issues
- Provider service outage
- Firewall blocking requests
- Request timeout too short

**Solutions:**

```yaml
data_engine:
  # Increase request timeout
  request_timeout: 60  # seconds

  # Configure retries for transient failures
  rate_limit:
    max_retries: 5
    backoff_strategy: "exponential_jitter"
```

### Parse Errors (DF-P0x)

Errors parsing or validating LLM responses.

| Code | Message | Severity | Description |
|------|---------|----------|-------------|
| `DF-P01` | JSON parse error | Sample | Failed to parse JSON from LLM response. |
| `DF-P02` | Schema validation | Sample | Response does not match expected schema structure. |
| `DF-P03` | Empty response | Sample | LLM returned an empty or whitespace-only response. |
| `DF-P04` | Malformed response | Sample | Response structure is malformed or incomplete. |

**Common causes:**

- LLM producing invalid JSON
- Schema too strict for model capabilities
- Temperature too high causing inconsistent outputs
- Model context length exceeded

**Solutions:**

```yaml
data_engine:
  # Lower temperature for more consistent outputs
  temperature: 0.5

  # Ensure sufficient token budget
  max_tokens: 2000
```

### Tool Errors (DF-T0x)

Errors related to tool/function calling in agent mode.

| Code | Message | Severity | Description |
|------|---------|----------|-------------|
| `DF-T01` | Tool validation | Sample | Tool call format is invalid or missing required fields. |
| `DF-T02` | Tool limit exceeded | Sample | Sample exceeded maximum tool calls per query. |
| `DF-T03` | No tool execution | Sample | Agent mode requires at least one tool execution. |

**Common causes:**

- `max_tools_per_query` set too low
- Tool schema incompatible with model
- Model not capable of structured tool calling

**Solutions:**

```yaml
data_engine:
  # Increase tool limit if needed
  max_tools_per_query: 5

  # Set to false to keep samples that exceed limit (truncated)
  max_tools_strict: false

  # Use a model with good tool calling support
  provider: "openai"
  model: "gpt-4o"
```

### Unknown Errors (DF-X0x)

Uncategorized errors that don't match known patterns.

| Code | Message | Severity | Description |
|------|---------|----------|-------------|
| `DF-X01` | Unknown error | Sample | An unexpected error occurred. |

**Debugging unknown errors:**

```bash
# Run with debug flag for full error details
deepfabric start config.yaml --debug
```

## Provider-Specific Notes

### Gemini

Gemini has stricter rate limits and schema requirements:

- Uses RPM (requests per minute) and RPD (requests per day) limits
- Daily quota resets at midnight Pacific time
- Requires `minItems`/`maxItems` for arrays in schemas
- Does not support `additionalProperties` in schemas

**Recommended configuration:**

```yaml
data_engine:
  provider: "gemini"
  model: "gemini-2.0-flash"

  rate_limit:
    base_delay: 3.0
    max_delay: 120.0
```

### OpenAI

OpenAI provides detailed rate limit headers and retry-after information:

- Supports strict mode for schema validation
- Good tool calling support across models
- Provides `x-ratelimit-*` headers for capacity tracking

### Anthropic

Anthropic uses a token bucket algorithm:

- Separate limits for requests, input tokens, and output tokens
- Provides `anthropic-ratelimit-*` headers
- Good at following complex prompts

### Ollama

Ollama runs locally with minimal rate limiting:

- Rate limits unlikely unless resource-constrained
- Connection errors usually mean server not running
- Use `ollama serve` to start the server

## Programmatic Access

Error codes can be accessed programmatically:

```python
from deepfabric.error_codes import (
    classify_error,
    ALL_ERROR_CODES,
    DF_R01,
    ErrorCategory,
    ErrorSeverity,
)

# Classify an error
classified = classify_error(some_exception, provider="gemini")
print(classified.error_code.code)  # "DF-R01"
print(classified.error_code.short_message)  # "Rate limit (RPM)"
print(classified.to_event())  # "DF-R01 Rate limit (RPM) - retry 3s"

# Check error properties
if classified.error_code.severity == ErrorSeverity.FATAL:
    print("This error requires immediate attention")

# List all error codes
for code, error in ALL_ERROR_CODES.items():
    print(f"{code}: {error.description}")
```

## Troubleshooting Workflow

1. **Check the Events panel** for error codes during generation
2. **Look up the error code** in this reference
3. **Apply recommended solutions** from the relevant section
4. **Run with `--debug`** if you need more details:
   ```bash
   deepfabric start config.yaml --debug
   ```
5. **Check provider status** if seeing persistent `DF-N03` errors
6. **Review rate limit configuration** if seeing `DF-R0x` errors frequently
