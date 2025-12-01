from .config import DeepFabricConfig
from .graph import Graph
from .topic_model import TopicModel
from .tree import Tree


def create_topic_generator(
    config: DeepFabricConfig,
    topics_overrides: dict | None = None,
) -> TopicModel:
    """Factory function to create a topic generator based on the configuration.

    Args:
        config: DeepFabricConfig object with topics configuration
        topics_overrides: Override parameters for topic generation

    Returns:
        TopicModel (Tree or Graph) based on topics.mode
    """
    topics_params = config.get_topics_params(**(topics_overrides or {}))

    if config.topics.mode == "graph":
        return Graph(**topics_params)

    # Default to tree mode
    return Tree(**topics_params)
