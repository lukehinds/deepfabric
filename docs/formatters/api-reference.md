# Formatter API Reference

This document provides complete API reference for the DeepFabric formatter system.

## Core Classes

### BaseFormatter

Abstract base class for all formatters.

```python
from deepfabric.formatters.base import BaseFormatter, FormatterError
```

#### Constructor

```python
def __init__(self, config: Dict[str, Any] = None)
```

Initialize the formatter with configuration.

**Parameters:**
- `config` (dict, optional): Configuration dictionary specific to this formatter

#### Abstract Methods

##### format()

```python
@abstractmethod
def format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]
```

Transform the dataset to the target format.

**Parameters:**
- `dataset` (List[Dict]): List of samples in DeepFabric's internal format

**Returns:**
- `List[Dict]`: List of samples in the formatter's target format

**Raises:**
- `FormatterError`: If formatting fails

#### Virtual Methods

##### validate()

```python
def validate(self, entry: Dict[str, Any]) -> bool
```

Validate that an entry meets the formatter's requirements.

**Parameters:**
- `entry` (Dict): A single dataset entry to validate

**Returns:**
- `bool`: True if the entry is valid, False otherwise

**Default Implementation:**
```python
return isinstance(entry, dict)
```

##### get_description()

```python
def get_description(self) -> str
```

Get a human-readable description of this formatter.

**Returns:**
- `str`: String description of what this formatter does

##### get_supported_formats()

```python
def get_supported_formats(self) -> List[str]
```

Get list of input formats this formatter can handle.

**Returns:**
- `List[str]`: List of supported input format names

**Default Implementation:**
```python
return ["messages"]
```

### FormatterError

Exception raised when formatting operations fail.

```python
from deepfabric.formatters.base import FormatterError

raise FormatterError("Detailed error message")
```

Inherits from Python's built-in `Exception` class.

---

## Registry System

### FormatterRegistry

Registry for managing formatter loading and instantiation.

```python
from deepfabric.formatters import FormatterRegistry

registry = FormatterRegistry()
```

#### Methods

##### load_formatter()

```python
def load_formatter(self, template: str, config: Dict[str, Any] = None) -> BaseFormatter
```

Load and instantiate a formatter from a template string.

**Parameters:**
- `template` (str): Template string like "builtin://grpo.py" or "file://./my_formatter.py"
- `config` (dict, optional): Configuration dictionary to pass to the formatter

**Returns:**
- `BaseFormatter`: Instantiated formatter instance

**Raises:**
- `FormatterError`: If the formatter cannot be loaded or instantiated

**Example:**
```python
# Load built-in formatter
grpo = registry.load_formatter("builtin://grpo.py", {
    "reasoning_start_tag": "<think>",
    "reasoning_end_tag": "</think>"
})

# Load custom formatter
custom = registry.load_formatter("file://./my_formatter.py", {
    "custom_option": "value"
})
```

##### list_builtin_formatters()

```python
def list_builtin_formatters(self) -> List[str]
```

List all available built-in formatters.

**Returns:**
- `List[str]`: List of built-in formatter names

**Example:**
```python
formatters = registry.list_builtin_formatters()
# Returns: ["grpo", "alpaca", "chatml"]
```

##### clear_cache()

```python
def clear_cache(self) -> None
```

Clear the formatter cache. Useful for development when formatters are being modified.

---

## Dataset Integration

### Dataset.apply_formatters()

Apply formatters to a dataset and return formatted datasets.

```python
def apply_formatters(self, formatter_configs: List[Dict[str, Any]]) -> Dict[str, "Dataset"]
```

**Parameters:**
- `formatter_configs` (List[Dict]): List of formatter configuration dictionaries

**Returns:**
- `Dict[str, Dataset]`: Dictionary mapping formatter names to formatted Dataset instances

**Raises:**
- `FormatterError`: If any formatter fails to process the dataset

**Configuration Format:**
```python
formatter_config = {
    "name": "grpo_math",
    "template": "builtin://grpo.py",
    "config": {
        "reasoning_start_tag": "<think>",
        "reasoning_end_tag": "</think>"
    },
    "output": "grpo_formatted.jsonl"  # Optional
}
```

**Example:**
```python
from deepfabric.dataset import Dataset

dataset = Dataset.from_jsonl("input.jsonl")

formatter_configs = [
    {
        "name": "grpo",
        "template": "builtin://grpo.py",
        "config": {"validate_numerical": True},
        "output": "grpo_output.jsonl"
    },
    {
        "name": "alpaca",
        "template": "builtin://alpaca.py",
        "config": {"include_empty_input": False},
        "output": "alpaca_output.jsonl"
    }
]

formatted_datasets = dataset.apply_formatters(formatter_configs)

# Access formatted datasets
grpo_dataset = formatted_datasets["grpo"]
alpaca_dataset = formatted_datasets["alpaca"]
```

### Dataset.list_available_formatters()

List all available built-in formatters.

```python
def list_available_formatters(self) -> List[str]
```

**Returns:**
- `List[str]`: List of built-in formatter names

---

## Configuration System

### FormatterConfig

Pydantic model for formatter configuration.

```python
from deepfabric.config import FormatterConfig

formatter_config = FormatterConfig(
    name="my_formatter",
    template="builtin://grpo.py",
    config={"option": "value"},
    output="output.jsonl"
)
```

#### Fields

- `name` (str, required): Unique identifier for this formatter instance
- `template` (str, required): Template path (builtin:// or file://)
- `config` (Dict[str, Any], optional): Formatter-specific configuration options
- `output` (str, optional): Output file path for this formatter

### DeepFabricConfig.get_formatter_configs()

Get list of formatter configurations from the main configuration.

```python
def get_formatter_configs(self) -> List[Dict[str, Any]]
```

**Returns:**
- `List[Dict]`: List of formatter configuration dictionaries

**Example:**
```python
from deepfabric.config import DeepFabricConfig

config = DeepFabricConfig.from_yaml("config.yaml")
formatter_configs = config.get_formatter_configs()

# Apply to dataset
dataset.apply_formatters(formatter_configs)
```

---

## Built-in Formatters

### GrpoFormatter

**Template**: `builtin://grpo.py`

Mathematical reasoning formatter with configurable reasoning and solution tags.

#### Configuration Options

```python
config = {
    "reasoning_start_tag": str,      # Default: "<start_working_out>"
    "reasoning_end_tag": str,        # Default: "<end_working_out>"
    "solution_start_tag": str,       # Default: "<SOLUTION>"
    "solution_end_tag": str,         # Default: "</SOLUTION>"
    "system_prompt": str,            # Default: Auto-generated
    "validate_numerical": bool       # Default: True
}
```

#### Supported Input Formats

- `messages`: Chat format with system/user/assistant roles
- `question_answer`: Q&A format with optional reasoning
- `chain_of_thought`: Questions with reasoning traces
- `generic`: Any format with question/answer patterns

### AlpacaFormatter

**Template**: `builtin://alpaca.py`

Instruction-following format for supervised fine-tuning.

#### Configuration Options

```python
config = {
    "instruction_field": str,        # Default: "instruction"
    "input_field": str,              # Default: "input"
    "output_field": str,             # Default: "output"
    "include_empty_input": bool,     # Default: True
    "instruction_template": str      # Default: None
}
```

#### Supported Input Formats

- `messages`: Chat format
- `instruction_output`: Direct instruction/output format
- `question_answer`: Q&A format
- `generic`: Any instruction-like patterns

### ChatmlFormatter

**Template**: `builtin://chatml.py`

Conversation format with ChatML markup.

#### Configuration Options

```python
config = {
    "start_token": str,              # Default: "<|im_start|>"
    "end_token": str,                # Default: "<|im_end|>"
    "output_format": str,            # Default: "structured" ("structured" or "text")
    "default_system_message": str,   # Default: "You are a helpful assistant."
    "require_system_message": bool   # Default: False
}
```

#### Supported Input Formats

- `messages`: Direct chat format
- `question_answer`: Q&A pairs
- `instruction_response`: Instruction-following patterns
- `generic`: Any conversational patterns

---

## Error Handling

### FormatterError Exception

`FormatterError` is the primary exception class used for all formatter-related errors. It can include optional details for debugging.

```python
class FormatterError(Exception):
    """Exception raised when formatting operations fail."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}
```

This single exception type is raised for various failure scenarios:
- **Loading errors**: When a formatter cannot be loaded from a template
- **Configuration errors**: When formatter configuration is invalid
- **Processing errors**: When formatting operations fail
- **Validation errors**: When sample validation fails

### Common Error Scenarios

#### Template Loading Errors

```python
try:
    formatter = registry.load_formatter("builtin://nonexistent.py")
except FormatterError as e:
    print(f"Failed to load formatter: {e}")
    # Error: Built-in formatter 'nonexistent' not found
```

#### Configuration Errors

```python
try:
    formatter = registry.load_formatter("builtin://grpo.py", {
        "invalid_option": "value"
    })
except FormatterError as e:
    print(f"Configuration error: {e}")
```

#### Processing Errors

```python
try:
    formatted_data = formatter.format(invalid_dataset)
except FormatterError as e:
    print(f"Processing failed: {e}")
    # Error: Failed to format sample 5: Missing required field 'messages'
```

---

## Performance Considerations

### Caching

- Formatter classes are cached after first load
- Use `registry.clear_cache()` during development
- Consider memory usage with large formatter caches

### Memory Usage

- Formatters process entire datasets in memory
- For large datasets, consider batch processing
- Custom formatters can implement streaming

### Validation Overhead

- Input validation adds processing time
- Output validation can be disabled for performance
- Custom validators should be efficient

---

## Type Definitions

### Common Types

```python
from typing import Dict, List, Any, Optional

# Dataset sample
Sample = Dict[str, Any]

# Dataset
Dataset = List[Sample]

# Formatter configuration
FormatterConfig = Dict[str, Any]

# Template string
Template = str  # "builtin://name.py" or "file://path.py"
```

### Configuration Schema

```python
# Complete formatter configuration
{
    "name": str,                    # Required: formatter instance name
    "template": str,                # Required: formatter template path
    "config": Dict[str, Any],       # Optional: formatter-specific config
    "output": Optional[str]         # Optional: output file path
}
```

---

## Migration Guide

### From Previous Versions

If you have existing formatter code, update it to use the new API:

```python
# Old style (deprecated)
class OldFormatter:
    def transform(self, data):
        return data

# New style (recommended)
from deepfabric.formatters.base import BaseFormatter

class NewFormatter(BaseFormatter):
    def format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return dataset
```

### Configuration Updates

Update YAML configuration to use the new formatter section:

```yaml
# Old style (deprecated)
formatters:
  - name: "grpo"
    path: "./grpo_formatter.py"

# New style (recommended)
dataset:
  formatters:
    - name: "grpo"
      template: "builtin://grpo.py"
      config:
        reasoning_start_tag: "<think>"
      output: "grpo_output.jsonl"
```