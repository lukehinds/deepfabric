# Chain of Thought Troubleshooting Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues when working with Chain of Thought datasets in DeepFabric. Issues are organized by category with step-by-step solutions.

## Quick Diagnostic Checklist

Before diving into specific issues, run through this quick checklist:

```bash
# 1. Verify installation
deepfabric --version

# 2. Check API keys
echo $OPENAI_API_KEY | head -c 10
echo $ANTHROPIC_API_KEY | head -c 10

# 3. Test basic functionality
deepfabric validate examples/quickstart.yaml

# 4. Check Python imports
python -c "from deepfabric import DataSetGenerator; print('✅ Imports working')"
```

## Configuration Issues

### Error: "Invalid conversation_type"

**Symptoms**:
```
ValueError: Input should be 'basic', 'structured', 'tool_calling', 'cot_freetext', 'cot_structured', or 'cot_hybrid'
```

**Causes & Solutions**:

#### 1. Typo in conversation_type
```yaml
# ❌ Incorrect
conversation_type: "cot-freetext"    # Hyphen instead of underscore
conversation_type: "cot_free_text"   # Extra underscore
conversation_type: "freetext_cot"    # Wrong order

# ✅ Correct
conversation_type: "cot_freetext"
conversation_type: "cot_structured"
conversation_type: "cot_hybrid"
```

#### 2. Missing conversation_type for CoT
```yaml
# ❌ Missing - will default to "basic"
data_engine:
  reasoning_style: "mathematical"    # This has no effect without conversation_type

# ✅ Correct
data_engine:
  conversation_type: "cot_freetext"
  reasoning_style: "mathematical"
```

#### 3. Python API casing
```python
# ❌ Incorrect
generator = DataSetGenerator(
    conversation_type="COT_FREETEXT"  # All caps
)

# ✅ Correct
generator = DataSetGenerator(
    conversation_type="cot_freetext"  # Lowercase with underscores
)
```

### Error: "Schema validation failed"

**Symptoms**:
```
OpenAI does not support your schema: Invalid schema for response_format 'default'
```

**Solutions**:

#### 1. Check provider compatibility
```yaml
# Some providers have different schema requirements
data_engine:
  provider: "openai"
  model: "gpt-4o-mini"     # ✅ Supports structured output
  # model: "gpt-4-turbo"  # ⚠️  Limited schema support

  conversation_type: "cot_hybrid"  # Complex schema
```

#### 2. Use simpler format for problematic providers
```yaml
# If having schema issues, try simpler format
data_engine:
  conversation_type: "cot_freetext"  # Simpler schema
  # instead of "cot_hybrid"
```

#### 3. Local model alternatives
```python
# For local models that don't support complex schemas
generator = DataSetGenerator(
    provider="ollama",
    model_name="mistral:latest",
    conversation_type="cot_freetext",  # Simpler format works better
    temperature=0.3
)
```

## API and Authentication Issues

### Error: "API key not found"

**Symptoms**:
```
DataSetGeneratorError: OPENAI_API_KEY environment variable not set
```

**Solutions**:

#### 1. Set environment variable
```bash
# Temporary (current session only)
export OPENAI_API_KEY="sk-your-key-here"

# Permanent (add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="sk-your-key-here"' >> ~/.bashrc
source ~/.bashrc
```

#### 2. Verify key format
```bash
# OpenAI keys start with "sk-"
echo $OPENAI_API_KEY | grep "^sk-"

# Anthropic keys start with "sk-ant-"
echo $ANTHROPIC_API_KEY | grep "^sk-ant-"
```

#### 3. Check key permissions
```python
# Test API key with simple request
import openai
openai.api_key = "your-key-here"

try:
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=10
    )
    print("✅ API key working")
except Exception as e:
    print(f"❌ API error: {e}")
```

### Error: "Rate limit exceeded"

**Symptoms**:
```
RateLimitError: Rate limit reached for requests
```

**Solutions**:

#### 1. Reduce generation speed
```yaml
# Smaller batch sizes
dataset:
  creation:
    batch_size: 1          # Instead of larger batches
    num_steps: 10          # Start smaller

# Add delays between requests
data_engine:
  max_retries: 2           # Fewer retries
```

#### 2. Upgrade API plan
- Check your OpenAI/Anthropic usage limits
- Consider upgrading to higher tier plans
- Monitor usage in provider dashboards

#### 3. Implement retry with backoff
```python
import time
import random

def generate_with_backoff(generator, **kwargs):
    """Generate with exponential backoff on rate limits."""
    for attempt in range(5):
        try:
            return generator.create_data(**kwargs)
        except RateLimitError:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"Rate limited, waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"Other error: {e}")
            break
    return None
```

## Generation Quality Issues

### Issue: Generated samples are too short/brief

**Symptoms**: Chain of thought lacks detail, reasoning is superficial

**Diagnostic questions**:
- Are reasoning sections under 100 characters?
- Do samples skip important steps?
- Is mathematical work not shown?

**Solutions**:

#### 1. Adjust temperature and prompts
```yaml
data_engine:
  temperature: 0.2           # Lower for more consistent, detailed reasoning

  generation_system_prompt: "You are a detailed educator who shows EVERY step of your reasoning process. Never skip intermediate calculations or logical steps."

  instructions: "Create problems that require detailed, multi-step reasoning. Each step should be clearly explained with mathematical work or logical justification shown."
```

#### 2. Emphasize step-by-step in prompts
```python
generator = DataSetGenerator(
    generation_system_prompt="""You are a thorough teacher who:
    - Shows every calculation step explicitly
    - Explains the reasoning behind each decision
    - Uses phrases like 'first', 'then', 'next', 'finally'
    - Always verifies answers when possible""",
    conversation_type="cot_freetext",
    reasoning_style="mathematical"
)
```

#### 3. Use higher-capability models
```yaml
data_engine:
  provider: "openai"
  model: "gpt-4o"            # Instead of gpt-4o-mini for more detailed reasoning
```

### Issue: Mathematical errors in reasoning

**Symptoms**: Incorrect calculations, wrong formulas, arithmetic mistakes

**Solutions**:

#### 1. Emphasize verification in prompts
```yaml
data_engine:
  generation_system_prompt: "You are a careful mathematics teacher who ALWAYS double-checks calculations and shows verification steps. If you make a calculation, verify it with a different method."

  reasoning_style: "mathematical"
  temperature: 0.1           # Very low for calculation accuracy
```

#### 2. Add calculation checking instructions
```python
generator = DataSetGenerator(
    instructions="""Create math problems where you:
    1. Show every arithmetic step clearly
    2. Use proper mathematical notation
    3. Verify your final answer by substituting back
    4. Double-check calculations using estimation""",
    reasoning_style="mathematical"
)
```

#### 3. Filter out incorrect solutions
```python
def validate_math_accuracy(sample):
    """Basic validation for mathematical accuracy."""
    reasoning = sample.get('chain_of_thought', '')

    # Look for verification indicators
    has_verification = any(indicator in reasoning.lower()
                          for indicator in ['verify', 'check', '✓', 'correct'])

    # Look for calculation steps
    has_calculations = any(char in reasoning for char in '=+-×÷')

    # Basic quality score
    if has_verification and has_calculations:
        return True

    return len(reasoning) > 100  # At least substantial reasoning

# Filter dataset
filtered_samples = [s for s in dataset.samples if validate_math_accuracy(s)]
```

### Issue: Repetitive or similar problems

**Symptoms**: Generated problems are too similar, lack diversity

**Solutions**:

#### 1. Increase topic diversity
```yaml
topic_tree:
  degree: 4                  # More subtopics per node
  depth: 3                   # Deeper tree structure
  temperature: 0.8           # Higher creativity for topics
```

#### 2. Use more diverse topic prompts
```yaml
topic_tree:
  topic_prompt: "Diverse mathematical scenarios including real-world applications in business, science, everyday life, sports, cooking, travel, construction, and technology. Include problems with different contexts, number ranges, and solution approaches."
```

#### 3. Generate in smaller targeted batches
```python
# Instead of one large batch, generate multiple smaller themed batches
themes = [
    "money and shopping problems",
    "time and scheduling problems",
    "measurement and conversion problems",
    "geometry and spatial problems",
    "ratio and proportion problems"
]

for theme in themes:
    tree = Tree(topic_prompt=f"Mathematics problems about {theme}")
    # Generate 10 samples per theme
```

## Format-Specific Issues

### Free-text CoT Issues

#### Issue: Reasoning doesn't flow naturally

**Solutions**:
```yaml
data_engine:
  generation_system_prompt: "You are a tutor explaining to a student. Use natural language that flows conversationally while showing clear logical progression."

  reasoning_style: "general"    # More flexible than "mathematical"
  temperature: 0.4              # Balance between consistency and naturalness
```

#### Issue: Missing final answer or unclear conclusion

**Solutions**:
```python
# Emphasize clear conclusions
generator = DataSetGenerator(
    generation_system_prompt="Always end your reasoning with a clear, definitive final answer. Use phrases like 'Therefore,' or 'The answer is' to signal your conclusion.",
    conversation_type="cot_freetext"
)
```

### Structured CoT Issues

#### Issue: Conversation feels unnatural or scripted

**Solutions**:
```yaml
data_engine:
  generation_system_prompt: "Create realistic educational conversations with natural student responses, including confusion, clarification questions, and gradual understanding."

  temperature: 0.5              # Higher for more natural dialogue
```

#### Issue: Reasoning trace doesn't match conversation

**Solutions**:
```python
# Add validation for alignment
def validate_structured_cot_alignment(sample):
    """Check if reasoning trace aligns with conversation."""
    messages = sample.get('messages', [])
    reasoning_trace = sample.get('reasoning_trace', [])

    assistant_messages = [m for m in messages if m['role'] == 'assistant']

    # Each assistant message should have corresponding reasoning steps
    return len(reasoning_trace) >= len(assistant_messages)

# Filter misaligned samples
valid_samples = [s for s in dataset.samples if validate_structured_cot_alignment(s)]
```

### Hybrid CoT Issues

#### Issue: Chain of thought and reasoning trace are redundant

**Solutions**:
```yaml
data_engine:
  generation_system_prompt: "In hybrid format, use chain_of_thought for intuitive explanation and reasoning_trace for systematic step-by-step breakdown. They should complement, not duplicate each other."
```

#### Issue: Inconsistent information between sections

**Solutions**:
```python
# Emphasize consistency in prompts
generator = DataSetGenerator(
    generation_system_prompt="""For hybrid CoT:
    1. Chain of thought: Natural, intuitive explanation
    2. Reasoning trace: Systematic, structured steps
    3. Ensure both sections are consistent and support the same conclusion
    4. Use reasoning trace to break down the logic from chain of thought""",
    conversation_type="cot_hybrid"
)
```

## Performance and Scale Issues

### Issue: Generation is too slow

**Symptoms**: Taking too long to generate datasets

**Solutions**:

#### 1. Optimize model selection
```yaml
# Use faster models for development/testing
data_engine:
  model: "gpt-4o-mini"       # Faster than gpt-4o
  # model: "gpt-4-turbo"   # Even faster for simple tasks
```

#### 2. Reduce complexity
```yaml
dataset:
  creation:
    num_steps: 5             # Start small for testing
    batch_size: 1            # Don't increase for CoT formats

# Use simpler format during development
data_engine:
  conversation_type: "cot_freetext"  # Fastest CoT format
```

#### 3. Parallel generation for multiple datasets
```python
import concurrent.futures

def generate_parallel_datasets(topics, max_workers=3):
    """Generate multiple topic-specific datasets in parallel."""

    def generate_single(topic):
        tree = Tree(topic_prompt=f"Problems about {topic}")
        # Build tree and generate dataset
        return generate_dataset_for_topic(topic, tree)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(generate_single, topics))

    return results

# Usage
topics = ["arithmetic", "geometry", "algebra"]
datasets = generate_parallel_datasets(topics)
```

### Issue: High API costs

**Symptoms**: Unexpected high bills from API providers

**Solutions**:

#### 1. Monitor token usage
```python
import tiktoken

def estimate_tokens(text: str, model: str = "gpt-4") -> int:
    """Estimate token count for text."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(sample_count: int, avg_tokens_per_sample: int = 1000):
    """Estimate generation cost."""
    total_tokens = sample_count * avg_tokens_per_sample
    # GPT-4o-mini pricing (example)
    cost_per_1k_tokens = 0.00015  # Input tokens
    estimated_cost = (total_tokens / 1000) * cost_per_1k_tokens
    return estimated_cost

print(f"Estimated cost for 50 samples: ${estimate_cost(50):.2f}")
```

#### 2. Use cost-effective configurations
```yaml
# Budget-friendly setup
data_engine:
  provider: "openai"
  model: "gpt-4o-mini"           # Most cost-effective
  conversation_type: "cot_freetext"  # Simplest format
  temperature: 0.3               # Reduce retries
  max_retries: 2                 # Limit retries

topic_tree:
  degree: 2                      # Smaller topic tree
  depth: 2
```

#### 3. Implement cost monitoring
```python
class CostMonitor:
    def __init__(self, budget_limit: float = 10.0):
        self.budget_limit = budget_limit
        self.current_cost = 0.0

    def estimate_sample_cost(self, tokens: int, model: str = "gpt-4o-mini") -> float:
        # Rough cost estimation
        cost_per_1k = {"gpt-4o-mini": 0.00015, "gpt-4o": 0.005}
        return (tokens / 1000) * cost_per_1k.get(model, 0.001)

    def check_budget(self, estimated_tokens: int, model: str) -> bool:
        estimated_cost = self.estimate_sample_cost(estimated_tokens, model)
        return (self.current_cost + estimated_cost) <= self.budget_limit

# Usage
monitor = CostMonitor(budget_limit=5.0)  # $5 limit
if monitor.check_budget(1000, "gpt-4o-mini"):
    # Proceed with generation
    pass
```

## Validation and Data Quality Issues

### Issue: Samples failing validation

**Symptoms**: High rate of failed samples during dataset creation

**Diagnostic steps**:

#### 1. Check validation errors
```python
from deepfabric.dataset import Dataset

# Load and analyze failures
dataset = Dataset.from_jsonl("output.jsonl")
print(f"Successful samples: {len(dataset.samples)}")
print(f"Failed samples: {len(dataset.failed_samples)}")

# Examine failed samples
for i, failed in enumerate(dataset.failed_samples[:3]):
    print(f"\nFailed sample {i+1}:")
    print(f"Keys: {list(failed.keys()) if isinstance(failed, dict) else type(failed)}")
```

#### 2. Test validation manually
```python
# Test individual sample validation
test_sample = {
    "question": "Test question",
    "chain_of_thought": "Test reasoning",
    "final_answer": "Test answer"
}

is_valid = Dataset.validate_sample(test_sample)
print(f"Sample valid: {is_valid}")

# Check required fields for each format
formats = {
    "freetext": ["question", "chain_of_thought", "final_answer"],
    "structured": ["messages", "reasoning_trace", "final_answer"],
    "hybrid": ["question", "chain_of_thought", "reasoning_trace", "final_answer"]
}

for format_name, required_fields in formats.items():
    print(f"\n{format_name} format requires: {required_fields}")
```

### Issue: Inconsistent sample quality

**Solutions**:

#### 1. Implement quality scoring
```python
def quality_score(sample: dict) -> float:
    """Calculate comprehensive quality score."""
    score = 0.0

    # Length checks
    question_len = len(sample.get('question', ''))
    reasoning_len = len(sample.get('chain_of_thought', ''))
    answer_len = len(sample.get('final_answer', ''))

    if 20 <= question_len <= 200:
        score += 0.25
    if 100 <= reasoning_len <= 1000:
        score += 0.25
    if 5 <= answer_len <= 100:
        score += 0.25

    # Content quality
    reasoning = sample.get('chain_of_thought', '').lower()
    if any(word in reasoning for word in ['step', 'first', 'then', 'calculate']):
        score += 0.25

    return score

# Filter by quality
high_quality = [s for s in dataset.samples if quality_score(s) >= 0.75]
```

#### 2. Post-generation filtering
```python
def comprehensive_filter(samples: list) -> list:
    """Apply multiple quality filters."""

    def has_math_content(sample):
        reasoning = sample.get('chain_of_thought', '')
        return any(char in reasoning for char in '=+-×÷') or \
               any(word in reasoning.lower() for word in ['calculate', 'multiply', 'add'])

    def has_step_indicators(sample):
        reasoning = sample.get('chain_of_thought', '').lower()
        return any(word in reasoning for word in ['step', 'first', 'then', 'next'])

    def reasonable_length(sample):
        reasoning = sample.get('chain_of_thought', '')
        return 50 <= len(reasoning) <= 800

    filters = [has_math_content, has_step_indicators, reasonable_length]

    filtered = []
    for sample in samples:
        if all(f(sample) for f in filters):
            filtered.append(sample)

    print(f"Filtered {len(samples)} → {len(filtered)} samples")
    return filtered

# Apply filtering
clean_samples = comprehensive_filter(dataset.samples)
```

## Environment and Setup Issues

### Issue: Import errors

**Symptoms**:
```python
ImportError: No module named 'deepfabric'
```

**Solutions**:

#### 1. Verify installation
```bash
pip list | grep deepfabric
# or
pip show deepfabric
```

#### 2. Reinstall if needed
```bash
pip uninstall deepfabric
pip install deepfabric
# or
pip install --upgrade deepfabric
```

#### 3. Check Python environment
```bash
which python
python --version
# Ensure you're in the right virtual environment
```

### Issue: Dependency conflicts

**Solutions**:

#### 1. Create clean environment
```bash
# Create new virtual environment
python -m venv deepfabric_env
source deepfabric_env/bin/activate  # Linux/Mac
# deepfabric_env\Scripts\activate     # Windows

# Install in clean environment
pip install deepfabric
```

#### 2. Check dependency versions
```bash
pip list | grep -E "(pydantic|openai|anthropic)"
```

## Getting Help

### Debugging Checklist

When you encounter issues, gather this information:

```python
# System information
import sys
print(f"Python version: {sys.version}")
print(f"DeepFabric version: {deepfabric.__version__}")

# Configuration details
print(f"Conversation type: {your_config.conversation_type}")
print(f"Reasoning style: {your_config.reasoning_style}")
print(f"Provider: {your_config.provider}")
print(f"Model: {your_config.model_name}")

# Sample details (if generation issue)
print(f"Generated samples: {len(dataset.samples)}")
print(f"Failed samples: {len(dataset.failed_samples)}")
```

### Common Support Resources

1. **Check GitHub issues**: Search for similar problems
2. **Review documentation**: Ensure you're following current patterns
3. **Test with minimal examples**: Isolate the issue
4. **Check provider status**: API outages affect generation

### Emergency Workarounds

If you need to continue working while debugging:

```python
# Fallback to simpler format
generator = DataSetGenerator(
    conversation_type="cot_freetext",  # Simplest that usually works
    reasoning_style="general",         # Most flexible
    temperature=0.3,                   # Conservative
    max_retries=2,                     # Don't waste on retries
    provider="openai",
    model_name="gpt-4o-mini"           # Most reliable
)

# Generate smaller test datasets
dataset = generator.create_data(
    num_steps=3,      # Very small for testing
    batch_size=1,
    sys_msg=False
)
```

This comprehensive troubleshooting guide should help you resolve most issues you encounter when working with Chain of Thought datasets. Remember to start with simple configurations and gradually increase complexity as you verify each component is working correctly.