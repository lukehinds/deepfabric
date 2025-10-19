# Hugging Face Integration

DeepFabric's Hugging Face Hub integration streamlines dataset publishing with automatic metadata generation, dataset cards, and community sharing features. This integration transforms synthetic datasets into discoverable, well-documented resources for the machine learning community.

## Basic Hub Integration

Simple dataset upload with automatic documentation:

```yaml
# basic-hf-upload.yaml
dataset_system_prompt: "You are creating educational programming content for computer science students."

topic_tree:
  topic_prompt: "Python programming fundamentals for beginners"
  topic_system_prompt: "You are creating educational programming content for computer science students."
  degree: 4
  depth: 2
  temperature: 0.7
  provider: "openai"
  model: "gpt-4-turbo"
  save_as: "python_basics_topics.jsonl"

data_engine:
  instructions: "Create clear, beginner-friendly programming examples with step-by-step explanations and practical exercises."
  generation_system_prompt: "You are creating educational programming content for computer science students."
  provider: "openai"
  model: "gpt-4"
  temperature: 0.8
  max_retries: 3

dataset:
  creation:
    num_steps: 100
    batch_size: 5
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "python_beginners_dataset.jsonl"

# Hugging Face Hub configuration
huggingface:
  repository: "education/python-programming-basics"
  tags:
    - "programming"
    - "python"
    - "education"
    - "beginner-friendly"
    - "code-examples"
```

Generate and upload with single command:

```bash
# Set authentication
export HF_TOKEN="your-huggingface-token"

# Generate and auto-upload
deepfabric generate basic-hf-upload.yaml
```

## Multi-Dataset Repository

Organize related datasets in a single repository with different components:

```yaml
# comprehensive-ml-course.yaml
dataset_system_prompt: "You are creating a comprehensive machine learning curriculum with theoretical foundations and practical applications."

topic_tree:
  topic_prompt: "Machine learning concepts from basics to advanced applications"
  topic_system_prompt: "You are creating a comprehensive machine learning curriculum with theoretical foundations and practical applications."
  degree: 5
  depth: 3
  temperature: 0.7
  provider: "anthropic"
  model: "claude-3-sonnet"
  save_as: "ml_course_topics.jsonl"

data_engine:
  instructions: "Create detailed explanations with mathematical foundations, practical examples, and real-world applications suitable for undergraduate and graduate students."
  generation_system_prompt: "You are creating a comprehensive machine learning curriculum with theoretical foundations and practical applications."
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  temperature: 0.8
  max_retries: 3

dataset:
  creation:
    num_steps: 300
    batch_size: 6
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "ml_course_dataset.jsonl"

huggingface:
  repository: "university/comprehensive-ml-curriculum"
  tags:
    - "machine-learning"
    - "education"
    - "curriculum"
    - "undergraduate"
    - "graduate"
    - "mathematics"
    - "practical-applications"
```

Upload multiple related datasets:

```bash
# Generate different course components
deepfabric generate comprehensive-ml-course.yaml

# Generate specialized components with parameter overrides
deepfabric generate comprehensive-ml-course.yaml \
  --dataset-save-as "ml_fundamentals.jsonl" \
  --num-steps 150 \
  --depth 2

deepfabric generate comprehensive-ml-course.yaml \
  --dataset-save-as "ml_advanced_topics.jsonl" \
  --num-steps 200 \
  --temperature 0.9

# Upload each component with specific tags
deepfabric upload ml_course_dataset.jsonl \
  --repo university/comprehensive-ml-curriculum \
  --tags fundamentals theory

deepfabric upload ml_fundamentals.jsonl \
  --repo university/comprehensive-ml-curriculum \
  --tags basics introduction

deepfabric upload ml_advanced_topics.jsonl \
  --repo university/comprehensive-ml-curriculum \
  --tags advanced research-topics
```

## Enterprise Dataset Publishing

Professional dataset publishing with comprehensive documentation:

```yaml
# enterprise-customer-support.yaml
dataset_system_prompt: "You are creating professional customer support training data that demonstrates excellence in customer service across various industries and scenarios."

topic_tree:
  topic_prompt: "Customer support excellence across industries: retail, technology, healthcare, finance, and services"
  topic_system_prompt: "You are creating professional customer support training data that demonstrates excellence in customer service across various industries and scenarios."
  degree: 5
  depth: 4
  temperature: 0.8
  provider: "openai"
  model: "gpt-4"
  save_as: "customer_support_topics.jsonl"

data_engine:
  instructions: "Create realistic, professional customer service interactions demonstrating empathy, problem-solving skills, and industry-specific knowledge. Include complex scenarios, difficult customers, and exemplary resolution techniques."
  generation_system_prompt: "You are creating professional customer support training data that demonstrates excellence in customer service across various industries and scenarios."
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  temperature: 0.8
  max_retries: 5
  request_timeout: 60

dataset:
  creation:
    num_steps: 1000
    batch_size: 8
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "enterprise_customer_support.jsonl"

huggingface:
  repository: "enterprise-ai/customer-support-excellence"
  tags:
    - "customer-service"
    - "professional-training"
    - "multi-industry"
    - "conversation"
    - "enterprise"
    - "support-excellence"
    - "training-data"
```

Professional deployment with quality assurance:

```python
# enterprise_deployment.py
import os
import json
import logging
from pathlib import Path
from typing import Dict, List
from deepfabric import DeepFabricConfig

def validate_enterprise_dataset(dataset_path: str) -> Dict[str, any]:
    """Validate enterprise dataset for quality and compliance."""
    
    validation_metrics = {
        "total_conversations": 0,
        "average_length": 0,
        "professional_language_score": 0,
        "industry_coverage": set(),
        "quality_indicators": {
            "empathy_markers": 0,
            "solution_oriented": 0,
            "professional_tone": 0
        }
    }
    
    professional_markers = ["understand", "apologize", "help", "resolve", "appreciate"]
    solution_markers = ["solution", "fix", "resolve", "address", "handle"]
    
    with open(dataset_path, 'r') as f:
        conversations = []
        for line in f:
            conversation = json.loads(line)
            conversations.append(conversation)
            validation_metrics["total_conversations"] += 1
            
            # Analyze conversation content
            content = conversation["messages"][-1]["content"].lower()
            
            # Check for professional markers
            empathy_count = sum(1 for marker in professional_markers if marker in content)
            solution_count = sum(1 for marker in solution_markers if marker in content)
            
            validation_metrics["quality_indicators"]["empathy_markers"] += empathy_count
            validation_metrics["quality_indicators"]["solution_oriented"] += solution_count
            
            # Estimate professional tone (simplified)
            if empathy_count > 0 and solution_count > 0:
                validation_metrics["quality_indicators"]["professional_tone"] += 1
    
    # Calculate averages
    if validation_metrics["total_conversations"] > 0:
        total = validation_metrics["total_conversations"]
        validation_metrics["professional_language_score"] = (
            validation_metrics["quality_indicators"]["professional_tone"] / total
        )
    
    return validation_metrics

def deploy_enterprise_dataset(config_path: str):
    """Deploy enterprise dataset with full validation pipeline."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load and validate configuration
    logger.info("Loading configuration...")
    config = DeepFabricConfig.from_yaml(config_path)
    
    # Validate configuration
    logger.info("Validating configuration...")
    validation = config.validate()
    if not validation.is_valid:
        logger.error("Configuration validation failed")
        for error in validation.errors:
            logger.error(f"  - {error}")
        return False
    
    # Generate dataset (this would typically use the CLI)
    logger.info("Dataset generation would occur here...")
    
    # Post-generation validation
    dataset_path = config.get_dataset_config()["save_as"]
    logger.info(f"Validating generated dataset: {dataset_path}")
    
    metrics = validate_enterprise_dataset(dataset_path)
    
    # Quality gates
    min_professional_score = 0.8
    min_conversations = 500
    
    if metrics["professional_language_score"] < min_professional_score:
        logger.error(f"Professional language score {metrics['professional_language_score']:.2f} below threshold {min_professional_score}")
        return False
        
    if metrics["total_conversations"] < min_conversations:
        logger.error(f"Total conversations {metrics['total_conversations']} below minimum {min_conversations}")
        return False
    
    logger.info("All quality gates passed")
    logger.info(f"Professional Language Score: {metrics['professional_language_score']:.2%}")
    logger.info(f"Total Conversations: {metrics['total_conversations']}")
    
    # Upload to Hugging Face
    hf_config = config.get_huggingface_config()
    repo = hf_config.get("repository")
    
    if repo:
        logger.info(f"Uploading to Hugging Face Hub: {repo}")
        # Upload command would go here
        # subprocess.run(["deepfabric", "upload", dataset_path, "--repo", repo])
        
    return True

if __name__ == "__main__":
    deploy_enterprise_dataset("enterprise-customer-support.yaml")
```

## Research Dataset with Comprehensive Metadata

Academic dataset publication with detailed provenance and methodology documentation:

```yaml
# research-nlp-dataset.yaml
dataset_system_prompt: "You are creating research-quality natural language processing datasets with focus on linguistic diversity, theoretical soundness, and reproducibility."

# Auto-detects graph mode since topic_graph section is present
topic_graph:
  topic_prompt: "Natural language processing research areas: syntax, semantics, pragmatics, computational linguistics, and applications"
  topic_system_prompt: "You are creating research-quality natural language processing datasets with focus on linguistic diversity, theoretical soundness, and reproducibility."
  degree: 4
  depth: 3
  temperature: 0.8
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  save_as: "nlp_research_graph.json"

data_engine:
  instructions: "Create academically rigorous natural language processing examples with theoretical grounding, citing relevant literature where appropriate, and demonstrating complex linguistic phenomena suitable for graduate-level research."
  generation_system_prompt: "You are creating research-quality natural language processing datasets with focus on linguistic diversity, theoretical soundness, and reproducibility."
  provider: "openai"
  model: "gpt-4"
  temperature: 0.7
  max_retries: 5

dataset:
  creation:
    num_steps: 400
    batch_size: 4
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "nlp_research_dataset.jsonl"

huggingface:
  repository: "research-lab/nlp-theoretical-foundations"
  tags:
    - "natural-language-processing"
    - "computational-linguistics"
    - "research"
    - "theoretical"
    - "graduate-level"
    - "linguistics"
    - "syntax"
    - "semantics"
    - "pragmatics"
```

Complete research workflow with visualization and documentation:

```bash
#!/bin/bash
# research-publication-workflow.sh

echo "=== NLP Research Dataset Publication Workflow ==="

# Step 1: Configuration validation
echo "Step 1: Validating research configuration..."
deepfabric validate research-nlp-dataset.yaml
if [ $? -ne 0 ]; then
    echo "Configuration validation failed - aborting"
    exit 1
fi

# Step 2: Generate dataset with graph structure
echo "Step 2: Generating research dataset..."
deepfabric generate research-nlp-dataset.yaml

# Step 3: Create research visualizations
echo "Step 3: Creating topic graph visualization..."
deepfabric visualize nlp_research_graph.json --output research_topology

# Step 4: Generate research documentation
echo "Step 4: Generating research documentation..."
python generate_research_metadata.py nlp_research_dataset.jsonl nlp_research_graph.json

# Step 5: Quality assessment
echo "Step 5: Conducting quality assessment..."
python research_quality_assessment.py nlp_research_dataset.jsonl

# Step 6: Upload with comprehensive metadata
echo "Step 6: Publishing to Hugging Face Hub..."
deepfabric upload nlp_research_dataset.jsonl \
  --repo research-lab/nlp-theoretical-foundations \
  --tags nlp research theoretical graduate-level

echo "=== Publication workflow complete ==="
echo "Dataset available at: https://huggingface.co/datasets/research-lab/nlp-theoretical-foundations"
echo "Visualization available at: research_topology.svg"
```

## Community Dataset with Collaborative Features

Open-source community dataset with broad accessibility:

```yaml
# community-programming-help.yaml
dataset_system_prompt: "You are creating community-driven programming help content that demonstrates collaborative problem-solving, mentoring approaches, and inclusive technical communication."

topic_tree:
  topic_prompt: "Programming help and mentorship across languages, frameworks, and skill levels"
  topic_system_prompt: "You are creating community-driven programming help content that demonstrates collaborative problem-solving, mentoring approaches, and inclusive technical communication."
  degree: 6
  depth: 3
  temperature: 0.8
  provider: "openai"
  model: "gpt-4"
  save_as: "programming_help_topics.jsonl"

data_engine:
  instructions: "Create supportive, educational programming discussions that demonstrate effective mentoring, inclusive language, and collaborative problem-solving approaches suitable for diverse technical communities."
  generation_system_prompt: "You are creating community-driven programming help content that demonstrates collaborative problem-solving, mentoring approaches, and inclusive technical communication."
  provider: "openai"
  model: "gpt-4"
  temperature: 0.8
  max_retries: 3

dataset:
  creation:
    num_steps: 750
    batch_size: 10
    provider: "openai"
    model: "gpt-4"
    sys_msg: true
  save_as: "community_programming_help.jsonl"

huggingface:
  repository: "community/programming-mentorship"
  tags:
    - "programming"
    - "mentorship"
    - "community"
    - "collaborative"
    - "inclusive"
    - "help"
    - "education"
    - "open-source"
```

The Hugging Face integration provides a complete pathway from synthetic data generation to community sharing, enabling researchers and practitioners to contribute high-quality synthetic datasets to the broader machine learning ecosystem.

## Downloading and Reformatting Hub Datasets

DeepFabric can download datasets directly from Hugging Face Hub and transform them into different training formats without requiring local files. This bidirectional workflow enables dataset curation, format conversion, and preparation for specific training frameworks.

### Basic Download and Format

Download a dataset from the Hub and apply a formatter:

```bash
# Download and format to TRL SFT Tools format
deepfabric format --repo lukehinds/smol-test-sample --formatter trl

# Download and format to ChatML
deepfabric format --repo username/conversation-dataset --formatter chatml

# Download and format to GRPO for reasoning training
deepfabric format --repo org/math-problems --formatter grpo -o grpo_math.jsonl
```

### Multi-Format Conversion Workflow

Convert a single Hub dataset to multiple training formats:

```bash
#!/bin/bash
# multi-format-conversion.sh

REPO="community/agent-tool-dataset"
BASE_NAME="agent_training"

echo "Downloading and converting dataset: $REPO"

# Format for TRL SFTTrainer
deepfabric format --repo $REPO --formatter trl -o "${BASE_NAME}_trl.jsonl"

# Format for conversations training (Unsloth, Axolotl, etc.)
deepfabric format --repo $REPO --formatter conversations -o "${BASE_NAME}_conversations.jsonl"

# Format for Harmony (gpt-oss)
deepfabric format --repo $REPO --formatter harmony -o "${BASE_NAME}_harmony.jsonl"

# Format for single tool call training
deepfabric format --repo $REPO --formatter chatml -o "${BASE_NAME}_chatml.jsonl"

echo "Conversion complete. Created 4 formatted versions."
```

### Dataset Curation Pipeline

Download, format, validate, and re-upload a curated version:

```yaml
# curation-config.yaml
# Configuration for post-format processing if needed
dataset:
  formatters:
    - name: "trl_curated"
      template: "builtin://trl_sft_tools"
      output: "curated_trl.jsonl"
      config:
        include_system_prompt: true
        system_prompt_override: |
          You are a function calling AI model. You are provided with function
          signatures within <tools></tools> XML tags. You may call one or more
          functions to assist with the user query.
        validate_tool_schemas: true
        remove_available_tools_field: true
```

Complete curation workflow:

```bash
#!/bin/bash
# dataset-curation.sh

SOURCE_REPO="community/raw-agent-dataset"
TARGET_REPO="your-org/curated-agent-dataset"

echo "=== Dataset Curation Pipeline ==="

# Step 1: Download and format from Hub
echo "Step 1: Downloading and formatting dataset..."
deepfabric format --repo $SOURCE_REPO --formatter trl -o stage1_formatted.jsonl

# Step 2: Apply custom formatting with config (if needed for advanced options)
echo "Step 2: Applying advanced formatting options..."
deepfabric format stage1_formatted.jsonl -c curation-config.yaml

# Step 3: Validate the curated dataset
echo "Step 3: Validating curated dataset..."
python validate_curated.py curated_trl.jsonl

# Step 4: Upload curated version to your organization
echo "Step 4: Uploading curated dataset..."
deepfabric upload curated_trl.jsonl \
  --repo $TARGET_REPO \
  --tags curated agent-tools trl-format

echo "=== Curation complete ==="
echo "Source: https://huggingface.co/datasets/$SOURCE_REPO"
echo "Curated: https://huggingface.co/datasets/$TARGET_REPO"
```

### Split-Specific Processing

Process different dataset splits with different formatters:

```bash
# Process training split for TRL
deepfabric format --repo org/dataset --split train --formatter trl -o train_trl.jsonl

# Process validation split for evaluation (different format)
deepfabric format --repo org/dataset --split validation --formatter chatml -o val_chatml.jsonl

# Process test split
deepfabric format --repo org/dataset --split test --formatter chatml -o test_chatml.jsonl
```

### Real-World Example: Reformatting for Fine-Tuning

Convert a public agent dataset for TRL SFTTrainer fine-tuning:

```bash
#!/bin/bash
# prepare-for-finetuning.sh

echo "Preparing dataset for fine-tuning with TRL SFTTrainer"

# Download and format from Hub
deepfabric format \
  --repo lukehinds/smol-test-sample \
  --formatter trl \
  --split train \
  -o training_data.jsonl

# Verify the format
python - <<'PY'
from datasets import load_dataset
import json

# Load and inspect
with open("training_data.jsonl", "r") as f:
    first_example = json.loads(f.readline())

print("Example structure:")
print(json.dumps(first_example, indent=2))

# Verify required fields
assert "messages" in first_example, "Missing 'messages' field"
assert "tools" in first_example, "Missing 'tools' field"

print("\n✓ Format validated for TRL SFTTrainer")
print(f"✓ Sample has {len(first_example['tools'])} tools available")
PY

echo "Dataset ready for training!"
echo "Next: Use with TRL SFTTrainer"
```

This bidirectional integration enables a complete ecosystem workflow: generate datasets with DeepFabric → upload to Hub → share with community → download and reformat for specific use cases → iterate and improve.