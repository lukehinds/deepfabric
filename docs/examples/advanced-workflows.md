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
    tree.build()
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

These advanced workflows demonstrate production-ready patterns for sophisticated dataset generation scenarios, including resource optimization, quality control, and comprehensive validation pipelines.