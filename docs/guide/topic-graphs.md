# Topic Graphs

Topic graphs extend beyond hierarchical topic trees by enabling cross-connections between related concepts, creating more realistic representations of complex domains where ideas naturally interconnect. This experimental feature allows multiple parent relationships and creates richer topic structures for sophisticated dataset generation.

The graph approach recognizes that many domains contain concepts that span traditional categorical boundaries. Research areas, technical systems, and social phenomena often exhibit network-like relationships that hierarchical trees cannot adequately capture.

## Graph Structure and Connectivity

Topic graphs are built as directed acyclic graphs (DAGs) where nodes represent topics and edges represent conceptual relationships. Unlike trees, nodes can have multiple parents, enabling representation of concepts that belong to multiple categories or influence multiple areas.

The graph generation process creates both hierarchical relationships similar to trees and lateral connections between related concepts across different branches. This dual approach maintains organizational structure while capturing the interconnected nature of complex domains.

## Configuration for Graphs

Topic graph configuration uses similar parameters to trees but with additional connectivity options:

```yaml
# Graph mode is auto-detected since topic_graph section is present
topic_graph:
  topic_prompt: "Artificial intelligence research areas"
  topic_system_prompt: "You are an AI research expert creating comprehensive topic structures for academic and professional research purposes. You provide detailed, interconnected topic hierarchies with precise relationships and thorough coverage of complex domains."
  degree: 4     # Connections per node
  depth: 3      # Maximum distance from root
  temperature: 0.8    # Higher creativity for connections
  provider: "anthropic"
  model: "claude-sonnet-4-5"
  save_as: "ai_research_graph.json"
```

Graph mode is automatically detected when a `topic_graph` section is present in the configuration, distinguishing it from the default tree behavior.

## Graph Parameters

Graph-specific parameters control both structure and connectivity:

**Graph Degree** determines the number of direct connections each node can form, including both hierarchical children and cross-connections.

**Graph Depth** sets the maximum distance from the root node, similar to tree depth but measured through the shortest path in the graph.

**Cross-Connection Probability** influences how frequently the system creates lateral connections between concepts in different hierarchical branches.

## Visualization Capabilities

Topic graphs can be visualized as SVG diagrams showing both hierarchical relationships and cross-connections:

```bash
deepfabric visualize ai_research_graph.json --output research_structure.svg
```

The visualization uses different visual elements to distinguish hierarchical parent-child relationships from cross-topic connections, making the graph structure comprehensible even for complex domains.

## Generation Process

Graph building proceeds in multiple phases to ensure both hierarchical coherence and meaningful cross-connections:

**Hierarchical Phase** creates the basic tree structure similar to standard topic trees, establishing primary categorical relationships.

**Connection Phase** analyzes existing nodes to identify potential cross-connections based on semantic similarity and conceptual overlap.

**Validation Phase** ensures that all connections maintain logical consistency and that the resulting structure remains acyclic.

## Output Format

Graph structures are saved in JSON format capturing both nodes and relationships:

```json
{
  "nodes": {
    "root": {
      "prompt": "Artificial intelligence research areas",
      "children": ["machine_learning", "natural_language_processing"],
      "connections": []
    },
    "machine_learning": {
      "prompt": "Machine learning techniques and applications",
      "children": ["supervised_learning", "unsupervised_learning"],
      "connections": ["natural_language_processing"]
    }
  },
  "edges": [
    {"from": "root", "to": "machine_learning", "type": "hierarchical"},
    {"from": "machine_learning", "to": "natural_language_processing", "type": "cross_connection"}
  ]
}
```

This format preserves complete structural information enabling both dataset generation and further analysis.

## Advanced Use Cases

Topic graphs excel in domains with complex interconnections:

**Research Fields**: Capturing interdisciplinary relationships where concepts span multiple traditional boundaries.

**Technical Systems**: Representing software architectures where components interact across hierarchical boundaries.

**Social Networks**: Modeling relationships and influences that don't follow strict hierarchical patterns.

**Knowledge Management**: Creating comprehensive knowledge bases where information naturally cross-references.

??? tip "When to Choose Graphs Over Trees"
    Use graphs when your domain has significant conceptual overlap between categories, when you need to represent influence patterns, or when hierarchical structures feel artificially constraining. Graphs require more generation time but produce richer topic relationships.

## Performance Considerations

Graph generation requires more computational resources than tree generation due to the additional connection analysis phase. Consider starting with smaller degree and depth values while exploring graph capabilities before scaling to production sizes.

The cross-connection analysis phase scales quadratically with node count, making parameter selection important for large-scale graph generation projects.