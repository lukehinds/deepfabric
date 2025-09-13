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
    if config.topic_generator == "graph":
        graph_params = config.get_topic_graph_params(**(graph_overrides or {}))
        return Graph(**graph_params)

    tree_params = config.get_topic_tree_params(**(tree_overrides or {}))
    return Tree(**tree_params)
