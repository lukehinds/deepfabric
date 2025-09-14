from .config import DeepFabricConfig
from .graph import Graph
from .topic_model import TopicModel
from .tree import Tree


def create_topic_generator(
    config: DeepFabricConfig,
    tree_overrides: dict | None = None,
    graph_overrides: dict | None = None,
) -> TopicModel:
    """Factory function to create a topic generator based on the configuration."""
    # Auto-detect based on which sections are present
    has_tree = config.topic_tree is not None
    has_graph = config.topic_graph is not None

    if has_tree and has_graph:
        # Both sections present
        msg = "Both 'topic_tree' and 'topic_graph' sections present in config - please specify only one"
        raise ValueError(msg)

    if has_graph:
        # Only graph section present - use graph
        graph_params = config.get_topic_graph_params(**(graph_overrides or {}))
        return Graph(**graph_params)

    if has_tree:
        # Only tree section present - use tree
        tree_params = config.get_topic_tree_params(**(tree_overrides or {}))
        return Tree(**tree_params)

    # Neither section present - error
    msg = "Configuration must contain either 'topic_tree' or 'topic_graph' section"
    raise ValueError(msg)
