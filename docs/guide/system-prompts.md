# System Prompt Strategy Guide

System prompts are crucial for controlling the behavior, style, and quality of your generated datasets. This guide explains how to craft effective system prompts for different stages of the DeepFabric pipeline and provides strategies for different use cases.

## Understanding the Pipeline Stages

DeepFabric has multiple stages where system prompts can be specified, each serving different purposes:

### 1. Dataset System Prompt

```yaml
dataset_system_prompt: "Your main prompt here..."
```

This prompt serves as the system message in your final dataset (the first message in each conversation). It defines the AI assistant's role and behavior for end users. This is what will be seen when your dataset is used for training or inference.

### 2. Topic Generation System Prompt

```yaml
topic_tree:
  topic_system_prompt: "Prompt for topic exploration..."
  # or for graphs:
topic_graph:
  topic_system_prompt: "Prompt for topic exploration..."
```

Controls how topics are generated and organized in your tree or graph structure.

### 3. Data Engine System Prompt

```yaml
data_engine:
  generation_system_prompt: "Prompt for content generation..."
```

Controls how the actual training data (question-answer pairs) are generated.

## System Prompt Strategy by Use Case

### Educational Content Creation

**Topic Generation Prompt:**

```yaml
topic_tree:
  topic_system_prompt: "You are a curriculum designer creating comprehensive learning paths. Focus on logical progression, prerequisite relationships, and ensuring complete coverage of essential concepts. Think broadly about what learners need to know and organize topics in a pedagogically sound sequence."
```

**Data Generation Prompt:**

```yaml
data_engine:
  generation_system_prompt: "You are an experienced educator teaching [SUBJECT] to [LEVEL] students. Provide clear, step-by-step explanations with concrete examples. Avoid jargon unless you explain it. Include common misconceptions and how to avoid them. Always provide practical examples that students can relate to."
```

### Technical Documentation

**Topic Generation Prompt:**

```yaml
topic_tree:
  topic_system_prompt: "You are a technical documentation architect organizing complex software concepts. Focus on logical dependencies, real-world implementation patterns, and comprehensive coverage of both theoretical foundations and practical applications."
```

**Data Generation Prompt:**

```yaml
data_engine:
  generation_system_prompt: "You are a senior software engineer writing comprehensive technical documentation. Provide detailed explanations with working code examples, discuss trade-offs between different approaches, include performance considerations, and demonstrate patterns with real-world scenarios. Always include complete, runnable code snippets that follow best practices."
```

### Customer Service Training

**Topic Generation Prompt:**

```yaml
topic_tree:
  topic_system_prompt: "You are a customer service training specialist identifying comprehensive scenarios across industries and interaction types. Focus on common customer issues, escalation patterns, cultural considerations, and industry-specific challenges."
```

**Data Generation Prompt:**

```yaml
data_engine:
  generation_system_prompt: "You are an expert customer service trainer creating realistic interaction scenarios. Demonstrate empathetic communication, active listening, problem-solving techniques, and professional responses to difficult situations. Include diverse customer personalities and complex problem-solving scenarios while maintaining a helpful, patient tone."
```

### Research and Academic Content

**Topic Generation Prompt:**

```yaml
topic_graph:
  topic_system_prompt: "You are a research committee chair organizing academic knowledge domains. Focus on interdisciplinary connections, current research trends, methodological approaches, and theoretical frameworks. Ensure comprehensive coverage of both foundational concepts and cutting-edge developments."
```

**Data Generation Prompt:**

```yaml
data_engine:
  generation_system_prompt: "You are a distinguished researcher and academic writing comprehensive research content. Provide rigorous analysis with proper citations, discuss methodological approaches, present balanced perspectives on controversial topics, and maintain academic writing standards. Include relevant examples from current literature and practical applications of research findings."
```

## Advanced Prompt Engineering Techniques

### 1. Persona Definition

Be specific about the expertise level and background:

```yaml
system_prompt: "You are a senior machine learning engineer with 10+ years in production ML systems at tech companies like Google and Netflix..."
```

### 2. Output Format Specification

Define the expected structure and style:

```yaml
dataset_system_prompt: "...Your responses should follow this structure: 1) Brief concept explanation, 2) Practical code example, 3) Common pitfalls, 4) Best practices, 5) Real-world applications."
```

### 3. Audience Targeting

Specify the target audience clearly:

```yaml
dataset_system_prompt: "...Tailor your explanations for software engineers with 2-5 years of experience who understand basic programming concepts but are new to machine learning."
```

### 4. Quality Standards

Set specific quality expectations:

```yaml
dataset_system_prompt: "...All code examples must be runnable, follow PEP 8 standards, include proper error handling, and demonstrate production-ready patterns."
```

## Common Patterns and Best Practices

### When to Use the Same Prompt

Use identical prompts across stages when:

- You want consistent voice and style throughout
- The expertise level and approach should be uniform
- You're creating content for a single, well-defined audience

Example:

```yaml
dataset_system_prompt: "You are a Python instructor teaching intermediate developers."

topic_tree:
  topic_system_prompt: "You are a Python instructor teaching intermediate developers."

data_engine:
  generation_system_prompt: "You are a Python instructor teaching intermediate developers."
```

### When to Use Different Prompts

Use specialized prompts when:

- Different stages require different thinking modes (broad vs. deep)
- You want different levels of detail (overview vs. implementation)
- Different expertise is needed (architect vs. implementer)

Example:

```yaml
dataset_system_prompt: "Python education pipeline"

topic_tree:
  topic_system_prompt: "You are a curriculum designer organizing Python concepts for logical learning progression."

data_engine:
  generation_system_prompt: "You are a hands-on Python instructor providing detailed coding examples and debugging help."
```

## Prompt Testing and Iteration

### Start Simple, Then Specialize

1. Begin with basic, clear prompts
2. Generate small samples to test behavior
3. Identify issues or desired improvements
4. Add specific instructions to address them
5. Test again and refine

### A/B Testing Approach

Create multiple prompt variants and compare outputs:

```yaml
# Version A: General
dataset_system_prompt: "You are a helpful programming assistant."

# Version B: Specific
dataset_system_prompt: "You are a senior Python developer specializing in web frameworks, with expertise in Django, Flask, and FastAPI. You provide production-ready code examples with security best practices."
```

### Quality Metrics to Consider

- **Accuracy**: Technical correctness of information
- **Consistency**: Uniform style and approach across examples
- **Completeness**: Coverage of essential details
- **Practicality**: Real-world applicability of examples
- **Clarity**: Accessibility to target audience

## Common Pitfalls to Avoid

### 1. Overly Generic Prompts

❌ **Avoid:** "You are a helpful assistant."
✅ **Better:** "You are a machine learning engineer specializing in recommendation systems..."

### 2. Conflicting Instructions

❌ **Avoid:** Different stages with contradictory personas or styles
✅ **Better:** Consistent expertise levels and complementary approaches

### 3. Missing Context

❌ **Avoid:** Prompts without audience, format, or quality specifications
✅ **Better:** Clear definitions of who, what, how, and why

### 4. Too Many Instructions

❌ **Avoid:** Extremely long prompts with dozens of requirements
✅ **Better:** Focused prompts with 3-5 clear, specific instructions

## Example Configurations

### Complete Configuration Examples

See our example configurations for different domains:

- [`example-specialized-prompts.yaml`](https://github.com/lukehinds/deepfabric/blob/main/examples/example-specialized-prompts.yaml) - Different prompts per stage
- [`example-domain-specific-prompts.yaml`](https://github.com/lukehinds/deepfabric/blob/main/examples/example-domain-specific-prompts.yaml) - Medical domain specialization
- [`example-mixed-prompts.yaml`](https://github.com/lukehinds/deepfabric/blob/main/examples/example-mixed-prompts.yaml) - Combining consistent and custom prompts

### Quick Reference Templates

**For Code Generation:**

```yaml
dataset_system_prompt: "You are a [LANGUAGE] expert with [X] years of experience in [DOMAIN]. Provide working code examples with proper error handling, follow [STANDARDS], and explain complex concepts clearly."
```

**For Educational Content:**

```yaml
dataset_system_prompt: "You are an experienced [SUBJECT] instructor teaching [LEVEL] students. Use clear explanations, practical examples, and address common misconceptions."
```

**For Business/Professional:**

```yaml
dataset_system_prompt: "You are a [ROLE] professional with expertise in [DOMAIN]. Provide practical, actionable guidance based on industry best practices and real-world experience."
```
