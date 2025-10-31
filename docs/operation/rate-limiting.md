# Rate Limiting & Intelligent Retry

DeepFabric includes a rate limiting and retry system that handles API rate limits, quota exhaustion, and transient errors across different LLM providers.

## Overview

The rate limiting system provides:

- **Provider-Aware**: Automatically detects and handles rate limits for OpenAI, Anthropic Claude, Google Gemini, and Ollama
- **Backoff**: Exponential backoff with jitter to prevent thundering herd problems
- **Retry-After Headers**: Respects server-specified wait times when available
- **Fail-Fast Detection**: Identifies non-retryable errors (e.g., daily quota exhaustion) to avoid wasting time
- **Configurable**: Fine-grained control via YAML or Python API
- **Type-Safe**: Full Pydantic validation and type hints

## Quick Start

### Using Provider Defaults

The simplest approach is to let DeepFabric use provider-specific defaults:

```yaml
data_engine:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  generation_system_prompt: "You are a helpful AI assistant."
  # Rate limiting uses defaults - no configuration needed!
```

Each provider has optimized defaults:

- **OpenAI**: `max_retries=5`, `base_delay=1.0s`, `max_delay=60s`
- **Anthropic**: `max_retries=5`, `base_delay=1.0s`, `max_delay=60s`
- **Gemini**: `max_retries=5`, `base_delay=2.0s`, `max_delay=120s` (more conservative)
- **Ollama**: `max_retries=2`, `base_delay=0.5s`, `max_delay=5s` (local, minimal retry)

### Custom Rate Limiting

For fine-grained control, add a `rate_limit` section:

```yaml
data_engine:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  generation_system_prompt: "You are a helpful AI assistant."

  rate_limit:
    max_retries: 7
    base_delay: 3.0
    max_delay: 180.0
    backoff_strategy: "exponential_jitter"
    exponential_base: 2.0
    jitter: true
    respect_retry_after: true
```

## Configuration Options

### `max_retries`

- **Type**: Integer (0-20)
- **Default**: 5
- **Description**: Maximum number of retry attempts before giving up

```yaml
rate_limit:
  max_retries: 5  # Try up to 5 times total
```

### `base_delay`

- **Type**: Float (0.1-60.0 seconds)
- **Default**: 1.0 (OpenAI/Anthropic), 2.0 (Gemini)
- **Description**: Base delay in seconds before the first retry

```yaml
rate_limit:
  base_delay: 2.0  # Wait 2 seconds before first retry
```

### `max_delay`

- **Type**: Float (1.0-300.0 seconds)
- **Default**: 60.0 (OpenAI/Anthropic), 120.0 (Gemini)
- **Description**: Maximum delay between retries (prevents excessive wait times)

```yaml
rate_limit:
  max_delay: 120.0  # Never wait more than 2 minutes
```

### `backoff_strategy`

- **Type**: String (enum)
- **Default**: `"exponential_jitter"`
- **Options**:
  - `"exponential"`: `delay = base_delay * (exponential_base ^ attempt)`
  - `"exponential_jitter"`: Exponential with ±25% randomization (recommended)
  - `"linear"`: `delay = base_delay * attempt`
  - `"constant"`: Always use `base_delay`

```yaml
rate_limit:
  backoff_strategy: "exponential_jitter"  # Recommended for production
```

**Why Exponential with Jitter?**

Jitter adds randomization (±25%) to prevent the "thundering herd" problem where multiple clients retry simultaneously, creating spikes that trigger more rate limits.

### `exponential_base`

- **Type**: Float (1.1-10.0)
- **Default**: 2.0
- **Description**: Base multiplier for exponential backoff

```yaml
rate_limit:
  exponential_base: 2.0  # Delays: 2s, 4s, 8s, 16s, 32s, ...
```

Example delays with different bases:
- `base=1.5`: 1.5s, 2.25s, 3.375s, 5.06s, 7.59s
- `base=2.0`: 2s, 4s, 8s, 16s, 32s
- `base=3.0`: 3s, 9s, 27s, 81s, ...

### `jitter`

- **Type**: Boolean
- **Default**: `true`
- **Description**: Add ±25% randomization to delays

```yaml
rate_limit:
  jitter: true  # Prevents synchronized retries
```

### `respect_retry_after`

- **Type**: Boolean
- **Default**: `true`
- **Description**: Honor retry-after headers from provider responses

```yaml
rate_limit:
  respect_retry_after: true  # Use server-specified wait times
```

When `true`, the system prioritizes server-specified wait times over calculated backoff.

## Provider-Specific Behavior

### OpenAI

- **Headers Monitored**: `x-ratelimit-remaining-requests`, `x-ratelimit-limit-requests`, `retry-after`
- **Rate Limit Types**: RPM (requests per minute), TPM (tokens per minute)
- **Quota Errors**: Distinguishes between rate limits and quota exhaustion
- **Retry Strategy**: Respects `retry-after` header

```yaml
data_engine:
  provider: "openai"
  model: "gpt-4"
  rate_limit:
    max_retries: 5
    respect_retry_after: true  # Always honor OpenAI's retry-after
```

### Anthropic Claude

- **Headers Monitored**: `anthropic-ratelimit-requests-remaining`, `anthropic-ratelimit-tokens-remaining`, `retry-after`
- **Algorithm**: Token bucket with continuous replenishment
- **Rate Limit Types**: RPM, ITPM (input tokens/min), OTPM (output tokens/min)
- **Tiers**: 4 automatic tiers based on credit purchases

```yaml
data_engine:
  provider: "anthropic"
  model: "claude-3-5-sonnet-20241022"
  rate_limit:
    max_retries: 5
    base_delay: 1.0
    gradual_rampup: true  # Anthropic recommends gradual traffic increases
```

### Google Gemini

- **Rate Limit Types**: RPM, TPM, RPD (requests per day)
- **Daily Quota**: Resets at midnight Pacific time
- **Error Format**: `429 RESOURCE_EXHAUSTED` with `QuotaFailure` details
- **No Retry-After Header**: Uses conservative backoff strategy
- **Fail-Fast**: Detects daily quota exhaustion and stops retrying

```yaml
data_engine:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  rate_limit:
    max_retries: 5
    base_delay: 2.0      # Higher default for Gemini
    max_delay: 120.0     # Longer max for daily quota
    daily_quota_aware: true  # Detect RPD exhaustion
```

**Gemini Daily Quota Exhaustion**:
When Gemini's daily quota is exhausted (RPD limit), the system detects this and fails fast rather than retrying, since the quota won't reset until midnight Pacific time.

### Ollama (Local)

- **Local Deployment**: Minimal rate limiting needed
- **Retry Logic**: Primarily for connection issues
- **Conservative Settings**: Lower retries and delays

```yaml
data_engine:
  provider: "ollama"
  model: "mistral:latest"
  rate_limit:
    max_retries: 2       # Minimal retries for local
    base_delay: 0.5      # Short delays
    max_delay: 5.0
```

## Python API

### Programmatic Configuration

```python
from deepfabric import DataSetGenerator

generator = DataSetGenerator(
    generation_system_prompt="You are a helpful AI assistant.",
    provider="gemini",
    model_name="gemini-2.0-flash-exp",
    temperature=0.5,

    # Rate limiting configuration
    rate_limit={
        "max_retries": 7,
        "base_delay": 3.0,
        "max_delay": 180.0,
        "backoff_strategy": "exponential_jitter",
        "exponential_base": 2.0,
        "jitter": True,
        "respect_retry_after": True,
    }
)
```

### Using Provider Defaults

```python
# Omit rate_limit to use intelligent defaults
generator = DataSetGenerator(
    generation_system_prompt="You are a helpful AI assistant.",
    provider="gemini",
    model_name="gemini-2.0-flash-exp",
    # rate_limit automatically uses Gemini defaults
)
```

### Advanced: LLMClient Direct Usage

```python
from deepfabric.llm import LLMClient
from deepfabric.llm.rate_limit_config import GeminiRateLimitConfig

# Create custom config
config = GeminiRateLimitConfig(
    max_retries=10,
    base_delay=2.0,
    max_delay=300.0,
    backoff_strategy="exponential_jitter",
    parse_quota_details=True,
    daily_quota_aware=True,
)

client = LLMClient(
    provider="gemini",
    model_name="gemini-2.0-flash-exp",
    rate_limit_config=config,
)
```

## Intelligent Features

### 1. Fail-Fast Detection

The system detects errors that shouldn't be retried:

- **Daily Quota Exhaustion**: Gemini RPD (requests per day) won't reset for hours
- **Zero Quota Limit**: Indicates account setup issue, not transient

When detected, the system fails immediately rather than wasting time retrying.

```python
# Example Gemini daily quota error:
# "429 RESOURCE_EXHAUSTED. Quota exceeded for metric:
#  generate_requests_per_model_per_day, limit: 0"
#
# System detects "per_day" and "limit: 0", fails fast
```

### 2. Provider-Specific Error Parsing

Each provider has unique error formats:

**OpenAI**:
```json
{
  "error": {
    "message": "Rate limit exceeded",
    "type": "rate_limit_error"
  }
}
```
Extracts: remaining capacity, retry-after

**Anthropic**:
```json
{
  "error": {
    "type": "rate_limit_error",
    "message": "This request would exceed your organization's rate limit"
  }
}
```
Extracts: retry-after, token bucket status

**Gemini**:
```json
{
  "error": {
    "code": 429,
    "status": "RESOURCE_EXHAUSTED",
    "details": [{
      "@type": "type.googleapis.com/google.rpc.QuotaFailure",
      "violations": [{
        "quotaMetric": "generativelanguage.googleapis.com/generate_requests_per_model_per_day"
      }]
    }]
  }
}
```
Extracts: quota metric type, daily vs minute limits

### 3. Exponential Backoff with Jitter

Prevents thundering herd when multiple requests retry:

```python
# Without jitter:
# Request 1: retry at 2s, 4s, 8s, 16s...
# Request 2: retry at 2s, 4s, 8s, 16s...
# Request 3: retry at 2s, 4s, 8s, 16s...
# All retry simultaneously → triggers more rate limits!

# With jitter (±25%):
# Request 1: retry at 2.1s, 3.8s, 8.5s, 14.2s...
# Request 2: retry at 1.7s, 4.3s, 7.1s, 17.8s...
# Request 3: retry at 2.4s, 3.5s, 8.9s, 15.1s...
# Distributed retries → smooth load
```

### 4. Retry-After Header Priority

When providers specify wait time, the system uses it:

```python
# OpenAI response headers:
# retry-after: 15

# System uses 15 seconds (capped at max_delay)
# Ignores calculated exponential delay
```

### 5. Retryable vs Non-Retryable Errors

**Retries on**:
- `429` (rate limit)
- `500`, `502`, `503`, `504` (server errors)
- Timeout, connection, network errors

**Does NOT retry**:
- `4xx` errors (except 429) - client errors
- Authentication failures
- Invalid API keys
- Daily quota exhaustion (Gemini)
- Zero quota limit

## Monitoring and Logging

The system logs retry attempts with detailed information:

```
WARNING - Rate limit/transient error for gemini on attempt 1, backing off 2.34s (quota_type: requests_per_minute): 429 RESOURCE_EXHAUSTED
WARNING - Rate limit/transient error for gemini on attempt 2, backing off 4.87s (quota_type: requests_per_minute): 429 RESOURCE_EXHAUSTED
ERROR - Giving up after 5 attempts for gemini: 429 RESOURCE_EXHAUSTED
```

### Log Levels

- **WARNING**: Retry attempts with backoff duration
- **ERROR**: Giving up after max retries
- **DEBUG**: Header parsing, quota info extraction

## Best Practices

### 1. Start with Defaults

Begin with provider defaults and adjust based on observed behavior:

```yaml
data_engine:
  provider: "gemini"
  # Let DeepFabric use intelligent defaults first
```

### 2. Monitor Failure Rates

DeepFabric tracks failures by category:

```python
generator.print_failure_summary()

# Output:
# === Failure Analysis Summary ===
# Total Failed Samples: 5
#
# Failure Types Breakdown:
# API Errors: 5
#   1. Rate limit exceeded for gemini/gemini-2.0-flash-exp...
```

### 3. Adjust for Your Use Case

**High Volume, Paid Tier**:
```yaml
rate_limit:
  max_retries: 3          # Fail faster
  base_delay: 0.5         # Quick retries
  max_delay: 10.0         # Short max wait
```

**Free Tier, Aggressive Limits**:
```yaml
rate_limit:
  max_retries: 10         # More persistent
  base_delay: 5.0         # Longer initial wait
  max_delay: 300.0        # Patient max wait
  exponential_base: 2.0   # Exponential growth
```

**Local Ollama**:
```yaml
rate_limit:
  max_retries: 2
  base_delay: 0.5
  max_delay: 5.0
```

### 4. Use Batch Sizes Wisely

Combine batch size with rate limiting:

```yaml
dataset:
  creation:
    batch_size: 3         # Smaller batches for aggressive limits
    num_steps: 10
```

### 5. Enable Jitter in Production

Always use jitter for production workloads:

```yaml
rate_limit:
  backoff_strategy: "exponential_jitter"
  jitter: true
```

## Troubleshooting

### Issue: Still Hitting Rate Limits

**Solution 1: Reduce Batch Size**
```yaml
dataset:
  creation:
    batch_size: 1  # Process one at a time
```

**Solution 2: Increase Base Delay**
```yaml
rate_limit:
  base_delay: 5.0  # Wait longer between retries
```

**Solution 3: Check Provider Tier**
- OpenAI: Verify tier and limits
- Anthropic: Check organization tier
- Gemini: Confirm usage tier (Free/1/2/3)

### Issue: Daily Quota Exhausted (Gemini)

The system detects this and fails fast:

```
ERROR - Failing fast for gemini: 429 RESOURCE_EXHAUSTED (quota_info: QuotaInfo(is_rate_limit=True, quota_type=requests_per_day, daily_quota_exhausted=True))
```

**Solutions**:
- Wait until midnight Pacific time
- Upgrade Gemini tier
- Switch to different provider temporarily

### Issue: Too Many Retries Wasting Time

**Solution: Reduce max_retries**
```yaml
rate_limit:
  max_retries: 3  # Give up sooner
```

### Issue: Requests Timing Out

**Solution: Increase Request Timeout**
```yaml
data_engine:
  request_timeout: 60  # Increase from default 30s
```

## Migration from Legacy `max_retries`

The old `max_retries` parameter is deprecated in favor of `rate_limit`:

**Before**:
```yaml
data_engine:
  max_retries: 5
```

**After**:
```yaml
data_engine:
  rate_limit:
    max_retries: 5
    # Plus many more options...
```

The old parameter still works but is ignored if `rate_limit` is present.

## Example: Complete Rate Limiting Setup

```yaml
dataset_system_prompt: "You are a helpful AI assistant."

topic_tree:
  topic_prompt: "Python programming"
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  temperature: 0.7
  degree: 3
  depth: 2

data_engine:
  provider: "gemini"
  model: "gemini-2.0-flash-exp"
  temperature: 0.5
  generation_system_prompt: "You are a Python instructor."

  # Comprehensive rate limiting configuration
  rate_limit:
    max_retries: 7
    base_delay: 3.0
    max_delay: 180.0
    backoff_strategy: "exponential_jitter"
    exponential_base: 2.0
    jitter: true
    respect_retry_after: true

dataset:
  creation:
    num_steps: 20
    batch_size: 2  # Conservative batch size
    sys_msg: true
  save_as: "python_dataset.jsonl"
```

## References

- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Anthropic Claude Rate Limits](https://docs.claude.com/en/api/rate-limits)
- [Google Gemini Rate Limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Exponential Backoff and Jitter](https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/)
