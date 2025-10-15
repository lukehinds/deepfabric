# Advanced Workflows

Advanced DeepFabric workflows demonstrate sophisticated patterns for complex dataset generation scenarios, including multi-stage processing, quality control pipelines, and large-scale production deployments. These examples showcase techniques that go beyond basic configuration to leverage the full capabilities of the system.

## Multi-Provider Pipeline

This workflow uses different model providers optimized for different stages of the generation process:

```yaml
# multi-provider-pipeline.yaml
dataset_system_prompt: "You are creating comprehensive educational content for software engineering professionals."

# Fast, economical topic generation
topic_tree:
  topic_prompt: "Advanced software engineering practices"
  topic_system_prompt: "You are creating comprehensive educational content for software engineering professionals."
  degree: 5
  depth: 3
  temperature: 0.7
  provider: "openai"
  model: "gpt-3.5-turbo"
  save_as: "engineering_topics.jsonl"

# High-quality content generation
data_engine:
  instructions: "Create detailed, practical explanations with real-world examples and code samples suitable for senior developers."
  generation_system_prompt: "You are creating comprehensive educational content for software engineering professionals."
  provider: "anthropic"
  model: "claude-3-opus"
  temperature: 0.8
  max_retries: 5

# Balanced final generation
dataset:
  creation:
    num_steps: 500
    batch_size: 8
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "engineering_dataset.jsonl"
```

This approach optimizes cost and quality by using GPT-3.5-turbo for broad topic exploration, Claude-3-Opus for detailed content generation, and GPT-4 for final dataset creation.

## Topic Graph with Visualization

Advanced topic graph generation with comprehensive analysis and visualization:

```yaml
# research-graph-analysis.yaml
dataset_system_prompt: "You are mapping the interconnected landscape of machine learning research areas with focus on practical applications and theoretical foundations."

topic_graph:
  topic_prompt: "Machine learning research and applications in industry"
  topic_system_prompt: "You are mapping the interconnected landscape of machine learning research areas with focus on practical applications and theoretical foundations."
  degree: 4
  depth: 4
  temperature: 0.8
  provider: "anthropic"
  model: "claude-3-opus"
  save_as: "ml_research_graph.json"

data_engine:
  instructions: "Create comprehensive research summaries with current trends, practical applications, and technical depth appropriate for graduate-level study."
  generation_system_prompt: "You are mapping the interconnected landscape of machine learning research areas with focus on practical applications and theoretical foundations."
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_retries: 3

dataset:
  creation:
    num_steps: 200
    batch_size: 6
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "ml_research_dataset.jsonl"

huggingface:
  repository: "research-org/ml-research-synthesis"
  tags:
    - "machine-learning"
    - "research"
    - "graduate-level"
    - "industry-applications"
```

Generate and analyze the complete workflow:

```bash
# Generate with graph visualization
deepfabric generate research-graph-analysis.yaml

# Create visualization for analysis
deepfabric visualize ml_research_graph.json --output research_structure

# Validate before publishing
deepfabric validate research-graph-analysis.yaml

# Upload to Hugging Face with metadata
deepfabric upload ml_research_dataset.jsonl --repo research-org/ml-research-synthesis
```

## Quality Control Pipeline

Sophisticated quality control through validation, filtering, and iterative refinement:

```yaml
# quality-controlled-generation.yaml
dataset_system_prompt: "You are creating high-quality technical documentation with emphasis on accuracy, clarity, and practical utility."

topic_tree:
  topic_prompt: "Modern web development frameworks and best practices"
  topic_system_prompt: "You are creating high-quality technical documentation with emphasis on accuracy, clarity, and practical utility."
  degree: 4
  depth: 3
  temperature: 0.6  # Lower temperature for consistency
  provider: "openai"
  model: "gpt-4"
  save_as: "webdev_topics.jsonl"

data_engine:
  instructions: "Create technically accurate documentation with working code examples, best practices, and common pitfalls. Include version-specific information and real-world usage patterns."
  generation_system_prompt: "You are creating high-quality technical documentation with emphasis on accuracy, clarity, and practical utility."
  provider: "anthropic"
  model: "claude-3-opus"
  temperature: 0.7
  max_retries: 5
  request_timeout: 60  # Extended timeout for quality

dataset:
  creation:
    num_steps: 300
    batch_size: 4  # Smaller batches for quality control
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "webdev_documentation.jsonl"
```

Implement additional quality control through scripted validation:

```bash
#!/bin/bash
# quality-control-workflow.sh

# Step 1: Validate configuration
echo "Validating configuration..."
deepfabric validate quality-controlled-generation.yaml
if [ $? -ne 0 ]; then
    echo "Configuration validation failed"
    exit 1
fi

# Step 2: Generate with monitoring
echo "Starting generation with quality monitoring..."
deepfabric generate quality-controlled-generation.yaml

# Step 3: Post-generation analysis
echo "Analyzing generated dataset..."
python analyze_dataset.py webdev_documentation.jsonl

# Step 4: Quality metrics evaluation
echo "Evaluating quality metrics..."
python quality_metrics.py webdev_documentation.jsonl

# Step 5: Conditional upload based on quality scores
if [ $? -eq 0 ]; then
    echo "Quality thresholds met, uploading to Hugging Face..."
    deepfabric upload webdev_documentation.jsonl --repo tech-docs/webdev-guide
else
    echo "Quality thresholds not met, review and regenerate"
    exit 1
fi
```

## Large-Scale Production Dataset

Configuration for generating large datasets with resource management and checkpointing:

```yaml
# production-scale-dataset.yaml
dataset_system_prompt: "You are creating comprehensive training data for customer service AI systems, focusing on natural conversation patterns and helpful problem-solving approaches."

topic_tree:
  topic_prompt: "Customer service scenarios across different industries and interaction types"
  topic_system_prompt: "You are creating comprehensive training data for customer service AI systems, focusing on natural conversation patterns and helpful problem-solving approaches."
  degree: 6  # Broad coverage
  depth: 4   # Deep exploration
  temperature: 0.8
  provider: "openai"
  model: "gpt-4"
  save_as: "customer_service_topics.jsonl"

data_engine:
  instructions: "Create realistic customer service conversations showing empathetic, helpful responses to various customer needs, complaints, and inquiries. Include diverse customer personalities and complex problem-solving scenarios."
  generation_system_prompt: "You are creating comprehensive training data for customer service AI systems, focusing on natural conversation patterns and helpful problem-solving approaches."
  provider: "openai"
  model: "gpt-4"
  temperature: 0.8
  max_retries: 5
  request_timeout: 45

dataset:
  creation:
    num_steps: 5000  # Large-scale generation
    batch_size: 10   # Optimized for throughput
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "customer_service_dataset.jsonl"

huggingface:
  repository: "enterprise-ai/customer-service-training"
  tags:
    - "customer-service"
    - "conversation"
    - "enterprise"
    - "training-data"
```

Production deployment script with monitoring and resource management:

```python
# production_deployment.py
import asyncio
import time
import logging
from deepfabric import DeepFabricConfig, DataSetGenerator, Tree

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def deploy_large_scale_generation(config_path, checkpoint_interval=500):
    """Deploy large-scale generation with checkpointing and monitoring."""
    
    config = DeepFabricConfig.from_yaml(config_path)
    
    # Load or create topic tree
    tree = Tree(**config.get_tree_args())

    async def _build_tree() -> None:
        async for _ in tree.build_async():
            pass

    asyncio.run(_build_tree())
    tree.save("production_topics.jsonl")

    # Create generator with production settings
    generator = DataSetGenerator(**config.get_engine_args())
    
    # Large-scale generation with checkpointing
    dataset_config = config.get_dataset_config()
    total_steps = dataset_config["creation"]["num_steps"]
    batch_size = dataset_config["creation"]["batch_size"]
    
    completed = 0
    start_time = time.time()
    
    while completed < total_steps:
        remaining = min(checkpoint_interval, total_steps - completed)
        
        logger.info(f"Generating batch {completed}-{completed + remaining}")
        
        batch_dataset = generator.create_data(
            num_steps=remaining,
            batch_size=batch_size,
            topic_model=tree
        )
        
        # Save checkpoint
        checkpoint_file = f"checkpoint_{completed}_{completed + remaining}.jsonl"
        batch_dataset.save(checkpoint_file)
        
        completed += remaining
        elapsed = time.time() - start_time
        rate = completed / elapsed
        
        logger.info(f"Progress: {completed}/{total_steps} ({completed/total_steps:.1%})")
        logger.info(f"Rate: {rate:.1f} examples/second")
        logger.info(f"ETA: {(total_steps - completed) / rate / 60:.1f} minutes")

if __name__ == "__main__":
    deploy_large_scale_generation("production-scale-dataset.yaml")
```

## Domain-Specific Validation

Custom validation pipeline for specialized domains:

```python
# domain_validator.py
import json
import re
from typing import List, Dict, Tuple

def validate_code_examples(dataset_path: str) -> Dict[str, int]:
    """Validate code examples in generated dataset."""
    
    validation_results = {
        "total_examples": 0,
        "valid_code_blocks": 0,
        "syntax_errors": 0,
        "missing_explanations": 0,
        "quality_score": 0
    }
    
    with open(dataset_path, 'r') as f:
        for line in f:
            example = json.loads(line)
            validation_results["total_examples"] += 1
            
            # Extract code blocks
            code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', 
                                   example["messages"][-1]["content"], 
                                   re.DOTALL)
            
            if code_blocks:
                validation_results["valid_code_blocks"] += 1
                
                # Basic syntax validation (simplified)
                for code in code_blocks:
                    try:
                        compile(code, '<string>', 'exec')
                    except SyntaxError:
                        validation_results["syntax_errors"] += 1
            
            # Check for explanations
            content = example["messages"][-1]["content"]
            if len(content) > 200 and any(word in content.lower() 
                                        for word in ["because", "this", "when", "why"]):
                validation_results["quality_score"] += 1
    
    # Calculate quality metrics
    if validation_results["total_examples"] > 0:
        quality_rate = validation_results["quality_score"] / validation_results["total_examples"]
        validation_results["overall_quality"] = quality_rate
    
    return validation_results

def main():
    results = validate_code_examples("webdev_documentation.jsonl")
    print(f"Dataset Quality Report:")
    print(f"Total Examples: {results['total_examples']}")
    print(f"Code Block Coverage: {results['valid_code_blocks']}/{results['total_examples']}")
    print(f"Syntax Error Rate: {results['syntax_errors']}/{results['valid_code_blocks']}")
    print(f"Overall Quality Score: {results['overall_quality']:.2%}")

if __name__ == "__main__":
    main()
```

## Dataset Transformation Pipeline

Download existing datasets from Hugging Face Hub, transform them with multiple formatters, validate, and republish. This workflow is ideal for dataset curation and format standardization:

```bash
#!/bin/bash
# dataset-transformation-pipeline.sh

set -e  # Exit on error

SOURCE_REPO="community/agent-reasoning-dataset"
TARGET_REPO="your-org/curated-reasoning-dataset"
TEMP_DIR="./pipeline_temp"

echo "=== Dataset Transformation Pipeline ==="
echo "Source: $SOURCE_REPO"
echo "Target: $TARGET_REPO"

# Create temporary working directory
mkdir -p $TEMP_DIR
cd $TEMP_DIR

# Stage 1: Download and format from Hub
echo ""
echo "Stage 1: Downloading and formatting from Hub..."
deepfabric format --repo $SOURCE_REPO --formatter trl -o stage1_trl.jsonl

# Stage 2: Apply secondary formatting for different training frameworks
echo ""
echo "Stage 2: Creating multiple format variants..."
deepfabric format stage1_trl.jsonl -f harmony -o stage2_harmony.jsonl
deepfabric format stage1_trl.jsonl -f unsloth -o stage2_unsloth.jsonl
deepfabric format stage1_trl.jsonl -f chatml -o stage2_chatml.jsonl

# Stage 3: Validate all outputs
echo ""
echo "Stage 3: Validating transformed datasets..."
python ../validate_formats.py stage1_trl.jsonl stage2_harmony.jsonl stage2_unsloth.jsonl stage2_chatml.jsonl

# Stage 4: Quality assessment
echo ""
echo "Stage 4: Running quality assessment..."
python ../assess_quality.py stage2_*.jsonl

# Stage 5: Upload curated versions
echo ""
echo "Stage 5: Uploading curated datasets..."

deepfabric upload stage1_trl.jsonl \
  --repo ${TARGET_REPO}-trl \
  --tags curated trl agent-tools training

deepfabric upload stage2_harmony.jsonl \
  --repo ${TARGET_REPO}-harmony \
  --tags curated harmony gpt-oss training

deepfabric upload stage2_unsloth.jsonl \
  --repo ${TARGET_REPO}-unsloth \
  --tags curated unsloth training

deepfabric upload stage2_chatml.jsonl \
  --repo ${TARGET_REPO}-chatml \
  --tags curated chatml training

echo ""
echo "=== Pipeline Complete ==="
echo "Curated datasets available at:"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-trl"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-harmony"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-unsloth"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-chatml"

# Cleanup
cd ..
rm -rf $TEMP_DIR
```

Validation script for the pipeline:

```python
# validate_formats.py
import sys
import json
from typing import List, Dict

def validate_trl_format(data: Dict) -> bool:
    """Validate TRL SFT Tools format."""
    required_fields = ["messages", "tools"]
    return all(field in data for field in required_fields)

def validate_harmony_format(data: Dict) -> bool:
    """Validate Harmony format."""
    if "messages" not in data:
        return False
    # Check for role hierarchy
    valid_roles = {"system", "developer", "user", "assistant", "tool"}
    return all(msg.get("role") in valid_roles for msg in data["messages"])

def validate_unsloth_format(data: Dict) -> bool:
    """Validate Unsloth format."""
    return "conversations" in data

def validate_chatml_format(data: Dict) -> bool:
    """Validate ChatML format."""
    return "messages" in data or "text" in data

VALIDATORS = {
    "trl": validate_trl_format,
    "harmony": validate_harmony_format,
    "unsloth": validate_unsloth_format,
    "chatml": validate_chatml_format,
}

def validate_file(filepath: str) -> Dict[str, any]:
    """Validate a formatted dataset file."""

    # Detect format from filename
    format_type = None
    for fmt in VALIDATORS:
        if fmt in filepath:
            format_type = fmt
            break

    if not format_type:
        return {
            "file": filepath,
            "valid": False,
            "error": "Could not detect format from filename"
        }

    validator = VALIDATORS[format_type]
    total = 0
    valid = 0
    errors = []

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total += 1
            try:
                data = json.loads(line)
                if validator(data):
                    valid += 1
                else:
                    errors.append(f"Line {line_num}: Invalid format")
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: JSON decode error - {e}")

            # Limit error collection
            if len(errors) >= 5:
                errors.append("... (additional errors truncated)")
                break

    return {
        "file": filepath,
        "format": format_type,
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "success_rate": valid / total if total > 0 else 0,
        "errors": errors[:5]  # First 5 errors
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_formats.py <file1> [file2] ...")
        sys.exit(1)

    print("=== Format Validation Report ===\n")

    all_valid = True
    results = []

    for filepath in sys.argv[1:]:
        result = validate_file(filepath)
        results.append(result)

        print(f"File: {result['file']}")
        print(f"Format: {result.get('format', 'unknown')}")
        print(f"Total entries: {result.get('total', 0)}")
        print(f"Valid entries: {result.get('valid', 0)}")
        print(f"Success rate: {result.get('success_rate', 0):.1%}")

        if result.get('errors'):
            print("Errors:")
            for error in result['errors']:
                print(f"  - {error}")

        print()

        if result.get('success_rate', 0) < 0.95:  # 95% threshold
            all_valid = False

    print("=== Summary ===")
    if all_valid:
        print("✓ All files passed validation")
        sys.exit(0)
    else:
        print("✗ Some files failed validation")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

Quality assessment script:

```python
# assess_quality.py
import sys
import json
from collections import Counter
from typing import Dict, List

def assess_dataset_quality(filepath: str) -> Dict:
    """Assess quality metrics for a dataset."""

    metrics = {
        "total_examples": 0,
        "avg_message_length": 0,
        "role_distribution": Counter(),
        "has_tools": 0,
        "avg_tools_per_example": 0,
        "quality_score": 0
    }

    total_message_length = 0
    total_tools = 0

    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            metrics["total_examples"] += 1

            # Analyze messages
            if "messages" in data:
                for msg in data["messages"]:
                    metrics["role_distribution"][msg.get("role", "unknown")] += 1
                    total_message_length += len(msg.get("content", ""))
            elif "conversations" in data:
                for conv in data["conversations"]:
                    metrics["role_distribution"][conv.get("role", "unknown")] += 1
                    total_message_length += len(conv.get("content", ""))

            # Analyze tools
            if "tools" in data:
                metrics["has_tools"] += 1
                total_tools += len(data["tools"])

    # Calculate averages
    if metrics["total_examples"] > 0:
        metrics["avg_message_length"] = total_message_length / metrics["total_examples"]
        metrics["avg_tools_per_example"] = total_tools / metrics["total_examples"]

        # Simple quality score (adjust as needed)
        score = 0
        if metrics["avg_message_length"] > 50:
            score += 0.3
        if metrics["has_tools"] > 0:
            score += 0.4
        if len(metrics["role_distribution"]) >= 2:
            score += 0.3
        metrics["quality_score"] = score

    return metrics

def main():
    if len(sys.argv) < 2:
        print("Usage: python assess_quality.py <file1> [file2] ...")
        sys.exit(1)

    print("=== Quality Assessment Report ===\n")

    for filepath in sys.argv[1:]:
        metrics = assess_dataset_quality(filepath)

        print(f"File: {filepath}")
        print(f"Total examples: {metrics['total_examples']}")
        print(f"Avg message length: {metrics['avg_message_length']:.0f} chars")
        print(f"Role distribution: {dict(metrics['role_distribution'])}")
        print(f"Examples with tools: {metrics['has_tools']} ({metrics['has_tools']/metrics['total_examples']*100:.1f}%)")
        print(f"Avg tools per example: {metrics['avg_tools_per_example']:.1f}")
        print(f"Quality score: {metrics['quality_score']:.1%}")
        print()

    print("✓ Quality assessment complete")

if __name__ == "__main__":
    main()
```

This pipeline demonstrates a complete workflow for downloading, transforming, validating, and republishing datasets in multiple formats suitable for different training frameworks.

---

These advanced workflows demonstrate production-ready patterns for sophisticated dataset generation scenarios, including resource optimization, quality control, and comprehensive validation pipelines.
