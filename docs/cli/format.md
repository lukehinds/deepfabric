# Format Command

The `format` command allows you to apply formatters to existing datasets without needing to regenerate them. This is useful when you want to transform already generated data into different training formats.

## Usage

```bash
# From a local JSONL file
deepfabric format INPUT_FILE [OPTIONS]

# Or directly from a Hugging Face dataset repo
deepfabric format --repo ORG/DATASET [OPTIONS]
```

## Arguments

- `INPUT_FILE` - Path to the input JSONL dataset file to format
  - Alternatively, use `--repo ORG/DATASET` to load from the Hugging Face Hub

## Options

- `-c, --config-file PATH` - YAML config file containing formatter settings
- `-f, --formatter [conversations|alpaca|chatml|grpo|harmony|trl|xlam_v2]` - Quick formatter selection with default settings
- `-o, --output TEXT` - Output file path
  - Local file input: defaults to `input_file_formatter.jsonl`
  - `--repo` input: defaults to `formatted.jsonl`
- `--repo TEXT` - Hugging Face dataset repo id (e.g., `org/dataset-name`)
- `--split TEXT` - Dataset split to load when using `--repo` (default: `train`)
- `--help` - Show help message

## Examples

### Using a specific formatter

Apply the `chatml` formatter with default settings:

```bash
deepfabric format dataset.jsonl -f chatml
```

This creates `dataset_chatml.jsonl` with the formatted output.

### Using a custom output path

```bash
deepfabric format dataset.jsonl -f alpaca -o training_data.jsonl
```

### Using a configuration file

For more control over formatter settings, use a YAML configuration file:

```bash
deepfabric format dataset.jsonl -c formatter_config.yaml
```

Example `formatter_config.yaml`:

```yaml
dataset:
  formatters:
    - name: "chatml_training"
      template: "builtin://chatml.py"
      output: "formatted_output.jsonl"
      config:
        output_format: "text"
        start_token: "<|im_start|>"
        end_token: "<|im_end|>"
```

## Supported Formatters

### alpaca

Formats data for Alpaca-style instruction tuning.

**Default configuration:**

```yaml
instruction_template: "### Instruction:\n{instruction}\n\n### Response:"
include_empty_input: false
```

### chatml

Formats data in ChatML format (structured or text).

**Default configuration:**

```yaml
output_format: "text"
start_token: "<|im_start|>"
end_token: "<|im_end|>"
include_system: false
```

### grpo

Formats data for GRPO (Guided Reasoning Process Optimization) training.

**Default configuration:**

```yaml
reasoning_start_tag: "<start_working_out>"
reasoning_end_tag: "<end_working_out>"
solution_start_tag: "<SOLUTION>"
solution_end_tag: "</SOLUTION>"
```

## Input Format

The command expects a JSONL file where each line is a JSON object. Supported formats include:

1. **Question-Answer format:**

```json
{
  "question": "What is recursion?",
  "answer": "Recursion is a programming technique..."
}
```

2. **Messages format:**

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
  ]
}
```

3. **Instruction format:**

```json
{
  "instruction": "Write a function to calculate factorial",
  "input": "n = 5",
  "output": "def factorial(n):..."
}
```

### Working with Hugging Face Datasets

You can now format datasets directly from the Hugging Face Hub using `--repo`, or continue using local JSONL files. Many HF datasets come in compatible formats:

**Format directly from a Hub repo:**

```bash
# Pull from Hub, format to Harmony, write to formatted.jsonl
deepfabric format --repo "org/dataset-name" --format harmony

# Load with datasets library
python - <<'PY'
from datasets import load_dataset
ds = load_dataset("json", data_files="formatted.jsonl")
print(ds)
PY
```

**For datasets with `messages` field (e.g., chat datasets):**

```bash
# Download dataset using datasets library or git
huggingface-cli download microsoft/DialoGPT-medium --repo-type dataset

# Convert to JSONL if needed
python -c "
from datasets import load_dataset
ds = load_dataset('microsoft/orca-math-word-problems-200k')
ds['train'].to_json('orca_math.jsonl')
"

# Apply formatter
deepfabric format orca_math.jsonl -f chatml
```

**For datasets with instruction format (e.g., Alpaca-style):**

```bash
# Many HF datasets use instruction/input/output format
deepfabric format alpaca_dataset.jsonl -f chatml
```

**Common HuggingFace dataset formats supported:**

- OpenAI ChatML format (`messages` field)
- Alpaca format (`instruction`, `input`, `output`)
- ShareGPT format (`conversations`)
- Q&A format (`question`, `answer` or `response`)

**Example conversion workflow (local):**

```bash
# 1. Download from HuggingFace
huggingface-cli download tatsu-lab/alpaca --repo-type dataset

# 2. Convert to JSONL (if not already)
python convert_hf_to_jsonl.py

# 3. Apply multiple formatters
deepfabric format alpaca.jsonl -f chatml -o alpaca_chatml.jsonl
deepfabric format alpaca.jsonl -f grpo -o alpaca_grpo.jsonl
```

## Workflow Example

1. Generate a dataset:

```bash
deepfabric generate config.yaml
```

2. Apply different formatters to the same dataset:

```bash
# For ChatML training
deepfabric format dataset_raw.jsonl -f chatml -o dataset_chatml.jsonl

# For Alpaca training
deepfabric format dataset_raw.jsonl -f alpaca -o dataset_alpaca.jsonl

# For GRPO training
deepfabric format dataset_raw.jsonl -f grpo -o dataset_grpo.jsonl
```

This allows you to prepare the same dataset for different training frameworks without regenerating the data.

### TRL SFT Tools

Use `-f trl` to convert agent/tool datasets to the Hugging Face TRL SFT tool-calling format. This maps to the built-in `trl_sft_tools` formatter.

```bash
# Format from a local file
deepfabric format dataset.jsonl -f trl -o trl_sft_tools.jsonl

# Or format directly from a Hub repo
deepfabric format --repo org/dataset -f trl -o trl_sft_tools.jsonl
```
