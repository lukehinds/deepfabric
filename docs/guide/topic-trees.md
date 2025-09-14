# Topic Trees

Topic trees provide hierarchical exploration of domains through structured branching from root concepts to specific subtopics. This approach works particularly well for domains with clear categorical relationships, educational content, and systematic knowledge organization.

The tree generation process transforms a single root prompt into a comprehensive structure that captures the breadth and depth of your domain. Each node in the tree represents a focused topic area, with children nodes exploring increasingly specific aspects of their parent concepts.

## Tree Structure and Parameters

Topic trees are controlled by three primary parameters that determine their shape and coverage characteristics:

**Tree Degree** controls the number of subtopics generated at each level. Higher degree values create broader exploration with more parallel topics, while lower values focus on fewer, more detailed paths.

**Tree Depth** determines how many levels the tree extends from the root. Greater depth enables detailed exploration of specific areas, while shallower trees maintain broader coverage.

**Temperature** affects the creativity and diversity of topic generation. Lower values produce more conventional, expected topics, while higher values encourage creative connections and unexpected directions.

## Configuration Patterns

A typical topic tree configuration demonstrates the key parameters:

```yaml
topic_tree:
  topic_prompt: "Modern software development practices"
  topic_system_prompt: "You are an expert knowledge organizer creating comprehensive topic structures for educational and professional content. You provide well-structured, logically connected topic hierarchies with clear relationships and thorough coverage."
  degree: 4      # Four subtopics per node
  depth: 3       # Three levels deep
  temperature: 0.7    # Balanced creativity
  provider: "openai"
  model: "gpt-4"
  save_as: "software_topics.jsonl"
```

This configuration creates a balanced exploration with moderate breadth and depth, suitable for comprehensive coverage without excessive generation time.

## Generation Process

Tree building occurs incrementally, starting from the root prompt and expanding level by level. Each node generates its children independently, allowing for diverse exploration while maintaining thematic coherence within each branch.

The process displays real-time progress showing node counts and generation status, enabling you to monitor coverage and identify any issues during construction.

## Quality Control

Several techniques ensure high-quality topic generation:

**Consistent System Prompts** provide context about the domain and desired topic characteristics across all generation calls.

**Temperature Management** balances creativity with relevance, preventing both overly conservative and completely random topic selection.

**Hierarchical Context** ensures that each subtopic maintains appropriate relationship to its parent while exploring new aspects.

??? tip "Optimizing Tree Parameters"
    Start with moderate values like degree=3, depth=2 for initial experiments. Increase degree for broader coverage or depth for more detailed exploration based on your specific needs. Very high values exponentially increase generation time and costs.

## Output Format

Generated trees are saved in JSONL format with each line representing a path from root to leaf:

```json
{"path": ["Modern software development practices", "Testing Strategies", "Unit Testing"]}
{"path": ["Modern software development practices", "Testing Strategies", "Integration Testing"]}
{"path": ["Modern software development practices", "DevOps Practices", "Continuous Integration"]}
```

This format enables easy analysis of topic coverage, depth distribution, and branch characteristics.

## Use Cases

Topic trees excel in several scenarios:

**Educational Content**: Creating structured learning paths with logical progression from basic to advanced concepts.

**Documentation Systems**: Organizing technical information with clear hierarchical relationships between concepts.

**Product Catalogs**: Structuring items with natural category and subcategory relationships.

**Research Organization**: Mapping academic fields with traditional subdiscipline boundaries.

## Loading Existing Trees

Previously generated trees can be loaded to skip the topic generation phase:

```bash
deepfabric generate config.yaml --load-tree existing_topics.jsonl
```

This approach enables experimentation with dataset generation parameters while maintaining consistent topic structure, accelerating iterative development processes.