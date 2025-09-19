# Custom Formatter Development Guide

This guide walks you through creating custom formatters for DeepFabric. Custom formatters allow you to transform datasets into any format required by your specific training pipeline or research needs.

## Quick Start

### 1. Create a Formatter File

Create a Python file for your custom formatter:

```python
# my_custom_formatter.py
from deepfabric.formatters.base import BaseFormatter, FormatterError
from typing import Dict, List, Any


class MyCustomFormatter(BaseFormatter):
    """Custom formatter for my specific training framework."""

    def format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Transform dataset to my custom format."""
        formatted_samples = []

        for sample in dataset:
            if not self.validate(sample):
                continue

            # Your custom transformation logic here
            formatted_sample = self._transform_sample(sample)
            formatted_samples.append(formatted_sample)

        return formatted_samples

    def _transform_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample."""
        # Implement your transformation logic
        return {
            "custom_field": sample.get("messages", []),
            "metadata": {"source": "deepfabric"}
        }
```

### 2. Configure the Formatter

Add it to your YAML configuration:

```yaml
dataset:
  formatters:
    - name: "my_custom"
      template: "file://./my_custom_formatter.py"
      config:
        custom_option: "value"
      output: "custom_formatted.jsonl"
```

### 3. Run DeepFabric

Your custom formatter will be applied during dataset generation.

## BaseFormatter Interface

All formatters must inherit from `BaseFormatter` and implement the required methods:

### Required Methods

#### `format(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]`

The main transformation method that processes the entire dataset.

```python
def format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform the dataset to the target format.

    Args:
        dataset: List of samples in DeepFabric's internal format

    Returns:
        List of samples in your target format

    Raises:
        FormatterError: If formatting fails
    """
    formatted_samples = []

    for i, sample in enumerate(dataset):
        try:
            if self.validate(sample):
                formatted_sample = self._format_sample(sample)
                formatted_samples.append(formatted_sample)
        except Exception as e:
            raise FormatterError(f"Failed to format sample {i}: {str(e)}") from e

    return formatted_samples
```

### Optional Methods to Override

#### `validate(entry: Dict[str, Any]) -> bool`

Validate input samples before processing:

```python
def validate(self, entry: Dict[str, Any]) -> bool:
    """Check if entry can be formatted."""
    if not super().validate(entry):
        return False

    # Your custom validation logic
    required_fields = ["messages", "metadata"]
    return all(field in entry for field in required_fields)
```

#### `get_description() -> str`

Provide a human-readable description:

```python
def get_description(self) -> str:
    """Get formatter description."""
    return """
    Custom formatter for XYZ training framework.

    Transforms DeepFabric datasets to include:
    - Custom metadata fields
    - Framework-specific structure
    - Validation for required fields
    """
```

#### `get_supported_formats() -> List[str]`

List supported input formats:

```python
def get_supported_formats(self) -> List[str]:
    """Get supported input formats."""
    return ["messages", "custom_format", "question_answer"]
```

## Configuration System

### Accessing Configuration

Your formatter receives configuration through the constructor:

```python
def __init__(self, config: Dict[str, Any] = None):
    super().__init__(config)

    # Access configuration options
    self.custom_option = self.config.get("custom_option", "default_value")
    self.enable_feature = self.config.get("enable_feature", True)
    self.template_string = self.config.get("template", "Default: {content}")
```

### Configuration Validation

Validate configuration in the constructor:

```python
def __init__(self, config: Dict[str, Any] = None):
    super().__init__(config)

    # Validate required configuration
    required_configs = ["api_endpoint", "format_version"]
    for key in required_configs:
        if key not in self.config:
            raise FormatterError(f"Missing required configuration: {key}")

    # Validate configuration values
    if self.config.get("max_length", 0) <= 0:
        raise FormatterError("max_length must be positive")
```

## Common Patterns

### Handling Different Input Formats

```python
def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Format based on input structure."""
    if "messages" in sample:
        return self._format_messages(sample["messages"])
    elif "question" in sample and "answer" in sample:
        return self._format_qa(sample)
    elif "instruction" in sample:
        return self._format_instruction(sample)
    else:
        return self._format_generic(sample)
```

### Template-Based Formatting

```python
def __init__(self, config: Dict[str, Any] = None):
    super().__init__(config)
    self.template = self.config.get("template", "Q: {question}\nA: {answer}")

def _format_qa(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Format Q&A using template."""
    formatted_text = self.template.format(
        question=sample["question"],
        answer=sample["answer"]
    )
    return {"text": formatted_text}
```

### Chain of Thought Integration

```python
def _format_with_reasoning(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Include reasoning in output."""
    output = sample["answer"]

    if "chain_of_thought" in sample:
        reasoning = sample["chain_of_thought"]
        output = f"Reasoning: {reasoning}\n\nAnswer: {output}"

    return {"formatted_response": output}
```

### Metadata Preservation

```python
def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve metadata during transformation."""
    formatted = self._core_transformation(sample)

    # Preserve original metadata
    if "metadata" in sample:
        formatted["metadata"] = sample["metadata"]

    # Add formatter metadata
    formatted["metadata"] = formatted.get("metadata", {})
    formatted["metadata"]["formatter"] = "my_custom_formatter"
    formatted["metadata"]["version"] = "1.0"

    return formatted
```

## Advanced Features

### Multi-Output Formatters

Create formatters that generate multiple outputs:

```python
def format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Create multiple format variants."""
    outputs = []

    for sample in dataset:
        # Create base format
        base_format = self._format_base(sample)
        outputs.append(base_format)

        # Create variant if configured
        if self.config.get("create_variants", False):
            variant = self._format_variant(sample)
            outputs.append(variant)

    return outputs
```

### Streaming Support

For large datasets, implement streaming:

```python
def format_streaming(self, dataset_generator):
    """Stream format large datasets."""
    for sample in dataset_generator:
        if self.validate(sample):
            yield self._format_sample(sample)
```

### Custom Validation

Implement domain-specific validation:

```python
def validate_output(self, entry: Dict[str, Any]) -> bool:
    """Validate formatted output."""
    if not isinstance(entry, dict):
        return False

    # Check required output fields
    required_fields = ["formatted_text", "metadata"]
    if not all(field in entry for field in required_fields):
        return False

    # Domain-specific validation
    text = entry["formatted_text"]
    if len(text) < self.config.get("min_length", 10):
        return False

    return True
```

## Testing Your Formatter

### Unit Testing

```python
import unittest
from my_custom_formatter import MyCustomFormatter

class TestMyCustomFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = MyCustomFormatter({
            "custom_option": "test_value"
        })

    def test_format_messages(self):
        """Test formatting messages."""
        input_data = [{
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]
        }]

        result = self.formatter.format(input_data)
        self.assertEqual(len(result), 1)
        self.assertIn("custom_field", result[0])

    def test_validation(self):
        """Test input validation."""
        valid_sample = {"messages": []}
        invalid_sample = {"invalid": "data"}

        self.assertTrue(self.formatter.validate(valid_sample))
        self.assertFalse(self.formatter.validate(invalid_sample))
```

### Integration Testing

Test with real DeepFabric data:

```python
def test_with_deepfabric():
    """Test formatter with DeepFabric-generated data."""
    from deepfabric.dataset import Dataset

    # Load test dataset
    dataset = Dataset.from_jsonl("test_data.jsonl")

    # Apply formatter
    formatter_config = {
        "name": "test",
        "template": "file://./my_custom_formatter.py",
        "config": {"custom_option": "test"}
    }

    formatted_datasets = dataset.apply_formatters([formatter_config])
    formatted_data = formatted_datasets["test"]

    # Validate results
    assert len(formatted_data) > 0
    assert all("custom_field" in sample for sample in formatted_data.samples)
```

## Error Handling Best Practices

### Graceful Error Handling

```python
def _format_sample(self, sample: Dict[str, Any]) -> Dict[str, Any] | None:
    """Format with graceful error handling."""
    try:
        return self._core_format(sample)
    except KeyError as e:
        # Log missing field but continue
        print(f"Warning: Missing field {e} in sample, skipping")
        return None
    except Exception as e:
        # Re-raise with context
        raise FormatterError(f"Formatting failed: {str(e)}") from e
```

### Informative Error Messages

```python
def validate(self, entry: Dict[str, Any]) -> bool:
    """Validate with helpful error messages."""
    if not isinstance(entry, dict):
        raise FormatterError("Entry must be a dictionary")

    if "messages" not in entry:
        available_fields = list(entry.keys())
        raise FormatterError(
            f"Required field 'messages' not found. "
            f"Available fields: {available_fields}"
        )

    return True
```

## Performance Optimization

### Efficient Processing

```python
def format(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Optimized batch processing."""
    # Pre-compile regex patterns
    if not hasattr(self, '_compiled_patterns'):
        self._compile_patterns()

    # Process in batches for memory efficiency
    batch_size = self.config.get("batch_size", 1000)
    results = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        batch_results = self._process_batch(batch)
        results.extend(batch_results)

    return results
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def _expensive_transformation(self, text: str) -> str:
    """Cache expensive transformations."""
    # Expensive processing here
    return processed_text
```

## Next Steps

- Review [Built-in Formatter Reference](built-in-reference.md) for inspiration
- See [API Reference](api-reference.md) for detailed API documentation
- Join the community to share your formatters with others