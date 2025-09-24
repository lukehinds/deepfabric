# Configuration Guide

The DeepFabric configuration guide provides comprehensive coverage of YAML-based configuration for CLI usage, from basic setup to advanced topic modeling techniques. This section builds systematically from foundational concepts to sophisticated workflows that leverage the full power of synthetic data generation through configuration files.

Understanding DeepFabric requires grasping its core philosophy: topic-driven generation creates more consistent and comprehensive datasets than prompt-based approaches alone. The system transforms a single root concept into a structured exploration of the domain, ensuring broad coverage while maintaining thematic coherence.

## Generation Pipeline

DeepFabric operates through a three-stage pipeline that transforms abstract concepts into concrete training data. The topic modeling stage expands your root prompt into a comprehensive structure representing your domain. The dataset generation stage creates training examples based on these topics, supporting multiple instruction formats from simple conversations to complex agent tool-calling scenarios. The output formatting stage packages everything into standard formats ready for immediate use.

Each stage operates independently, allowing you to experiment with different parameters without regenerating previous stages. This modularity accelerates iterative development and enables sophisticated workflows where different components use different model providers or configurations.

**Instruction Format Support**: DeepFabric supports multiple instruction formats including basic conversations, Chain of Thought reasoning datasets, and advanced agent tool-calling formats that capture systematic reasoning about external tool usage. This enables creation of training data for everything from conversational AI to sophisticated function-calling agents.

## Configuration Philosophy

The YAML configuration system provides comprehensive control over every aspect of generation while maintaining readability and version control compatibility. Rather than requiring complex command-line arguments, configurations capture your complete experimental setup in a single, shareable file.

The configuration structure mirrors the generation pipeline, with distinct sections for topic modeling, dataset generation, and output formatting. This separation allows precise control over each stage while providing sensible defaults for rapid experimentation.

For programmatic usage and SDK integration, see the **[SDK Guide](generator-pattern.md)** which covers using DeepFabric directly in Python code.

## Topic Modeling Approaches

DeepFabric supports two distinct approaches to topic modeling, each suited to different domain characteristics and use cases. Topic trees provide hierarchical structures ideal for domains with clear categorical relationships. Topic graphs enable cross-connections between concepts, better representing complex domains where ideas naturally interconnect.

The choice between trees and graphs depends on your domain's inherent structure. Educational content, product catalogs, and organizational hierarchies often benefit from tree structures. Research areas, technical concepts, and social phenomena frequently require the additional connectivity that graphs provide.

## Content Sections

Each section of this guide builds upon previous concepts while introducing new capabilities:

[**Configuration**](configuration.md) explores the YAML format in depth, covering system prompts, placeholder substitution, and provider integration patterns.

[**Topic Trees**](topic-trees.md) details hierarchical topic generation, including depth and degree parameters, branching strategies, and quality control techniques.

[**Topic Graphs**](topic-graphs.md) introduces experimental graph-based modeling with cross-connections, visualization capabilities, and advanced relationship modeling.

[**Dataset Generation**](dataset-generation.md) covers the conversion from topics to training examples, including template systems, batch processing, and quality assurance.

[**Provider Integration**](provider-integration.md) explains provider integration, authentication patterns, and optimization strategies for different model providers.

## Advanced Workflows

Beyond basic generation, DeepFabric supports sophisticated workflows involving configuration validation, intermediate result analysis, and iterative refinement. The modular CLI design enables complex pipelines where validation, generation, visualization, and publishing occur as separate, composable operations.

These advanced patterns become particularly valuable when working with large datasets, multiple model providers, or complex domain structures that benefit from iterative development and refinement processes.