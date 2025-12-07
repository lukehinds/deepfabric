# DeepFabric Data Format Specification

## 1\. Introduction

DeepFabric has a standard JSON schema that is used for curating datasets involving Agentic Tool Use (Function Calling).

It is designed to solve the "Interoperability Crisis" that exists in dataset formats, where a lot of the industry has taken their own route without all conforming to a set standard. HuggingFace have done lots to address this with chat template tools, but its still a fragmented landscape. 

We elected to settle on a format capable of training a wide variety of model architectures (from Llama, Qwen to Mistral) and compatible with major fine-tuning platforms (OpenAI, Azure) with minimum structural changes.

-----

## 2\. Core Specifications

The DeepFabric format relies on three immutable rules. Strict adherence to these rules is required for the dataset to pass validation.

### Rule 1: Arguments are Objects

Input arguments for tools must be stored as **Native JSON Objects (Dictionaries)**. This allows training libraries to dynamically format prompts without needing to parse double-escaped strings.

  * **Do:** `"arguments": {"city": "Tokyo", "limit": 5}`
  * **Don't:** `"arguments": "{\"city\": \"Tokyo\", \"limit\": 5}"`

### Rule 2: Strict ID Syntax

Every tool call must generate a unique identifier. To ensure compatibility with open-weight models (specifically the Mistral family), IDs must adhere to a strict Regex pattern.

  * **Length:** Exactly 9 characters.
  * **Charset:** Alphanumeric (a-z, A-Z, 0-9).
  * **Regex:** `^[a-zA-Z0-9]{9}$`

### Rule 3: Explicit Linking

Every `tool` response message must explicitly reference the `id` of the `assistant` message that triggered it. This preserves causality even if messages are processed asynchronously or reordered.

-----

## 3\. Schema Reference

### 3.1 The Assistant Message (Request)

This message represents the model deciding to use a tool.

**Fields:**

  * **`role`**: Must be `"assistant"`.
  * **`tool_calls`**: A list of call objects (supports parallel calling).
  * **`id`**: The 9-char alphanumeric ID (e.g., `call8B2x9`).
  * **`type`**: Always set to `"function"`.
  * **`function.arguments`**: The parameter dictionary.

**Example:**

```json
{
  "role": "assistant",
  "tool_calls": [
    {
      "id": "call8B2x9",
      "type": "function",
      "function": {
        "name": "search_database",
        "arguments": {"query": "Q3 Revenue", "year": 2024}
      }
    }
  ]
}
```

### 3.2 The Tool Message (Response)

This message represents the output returned by the API/Function.

**Fields:**

  * **`role`**: Must be `"tool"`.
  * **`tool_call_id`**: Must match the Assistant's `id` exactly.
  * **`content`**: The output of the tool. This **must be a string**.
      * *Best Practice:* Return a serialized JSON string to teach the model to parse structured outputs (e.g., `stdout` vs `stderr`).

**Example:**

```json
{
  "role": "tool",
  "tool_call_id": "call8B2x9",
  "name": "search_database",
  "content": "{\"results\": [{\"id\": 1, \"value\": \"1.2M\"}], \"status\": \"success\"}"
}
```

-----

## 4\. Full Dataset Example

Below is a valid DeepFabric entry demonstrating a **Parallel Heterogeneous Tool Call** (calling two different tools at once).

```json
{
    "messages": [
        {
            "content": "You are an AI assistant with access to various tools and functions.\nWhen given a task:\n1. Analyze what tools are needed\n2. Execute tools with proper parameters\n3. Interpret results and provide a clear answer\n",
            "role": "system"
        },
        {
            "content": "I need to find all instances of 'if-else' statements in the main Python file and then update the comments within those blocks to mention the specific condition being checked.",
            "role": "user"
        },
        {
            "content": "",
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "arguments": {
                            "file_path": "main.py"
                        },
                        "name": "read_file"
                    },
                    "id": "fPubFet0k",
                    "type": "function"
                }
            ]
        },
        {
            "content": "print(\"Hello, World!\")\n\ndef greet(name):\n    return f\"Hello, {name}!\"\n\nif __name__ == \"__main__\":\n    print(greet(\"Alice\"))\n",
            "role": "tool",
            "tool_call_id": "fPubFet0k"
        },
        {
            "content": "I can help with that. However, the provided file content does not contain any 'if-else' statements. If you'd like me to proceed with searching for 'if-else' statements in a different file or if you can provide the correct file content, please let me know.",
            "role": "assistant"
        }
    ]
}
```