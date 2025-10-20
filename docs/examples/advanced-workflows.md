# Advanced Workflows

Advanced DeepFabric workflows demonstrate patterns for complex dataset generation scenarios, including multi-stage processing,
quality control pipelines, and large-scale production deployments. These examples showcase techniques that go beyond basic
configuration to leverage the full capabilities of the system.

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
  model: "gpt-4-turbo"
  save_as: "engineering_topics.jsonl"

# High-quality content generation
data_engine:
  instructions: "Create detailed, practical explanations with real-world examples and code samples suitable for senior developers."
  generation_system_prompt: "You are creating comprehensive educational content for software engineering professionals."
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  temperature: 0.8
  max_retries: 5

# Balanced final generation
dataset:
  creation:
    num_steps: 500
    batch_size: 8
    provider: "openai"
    model: "gpt-5"
    sys_msg: true
  save_as: "engineering_dataset.jsonl"
```

This approach optimizes cost and quality by using GPT-3.5-turbo for broad topic exploration, claude-sonnet-4-5 for detailed content generation, and GPT-5 for final dataset creation.

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
  model: "claude-sonnet-4-5"
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
  model: "claude-sonnet-4-5"
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
deepfabric format stage1_trl.jsonl -f conversations -o stage2_conversations.jsonl
deepfabric format stage1_trl.jsonl -f chatml -o stage2_chatml.jsonl

# Stage 3: Validate all outputs
echo ""
echo "Stage 3: Validating transformed datasets..."
python ../validate_formats.py stage1_trl.jsonl stage2_harmony.jsonl stage2_conversations.jsonl stage2_chatml.jsonl

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

deepfabric upload stage2_conversations.jsonl \
  --repo ${TARGET_REPO}-conversations \
  --tags curated conversations training

deepfabric upload stage2_chatml.jsonl \
  --repo ${TARGET_REPO}-chatml \
  --tags curated chatml training

echo ""
echo "=== Pipeline Complete ==="
echo "Curated datasets available at:"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-trl"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-harmony"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-conversations"
echo "  - https://huggingface.co/datasets/${TARGET_REPO}-chatml"

# Cleanup
cd ..
rm -rf $TEMP_DIR
```
