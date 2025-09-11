from .config import PromptWrightConfig
from .topic_graph import TopicGraph
from .topic_model import TopicModel
from .topic_tree import TopicTree


def create_topic_generator(config: PromptWrightConfig) -> TopicModel:
    """Factory function to create a topic generator based on the configuration."""
    if config.topic_generator == "graph":
        graph_args = config.get_topic_graph_args()
        return TopicGraph(graph_args)

    tree_args = config.get_topic_tree_args()
    return TopicTree(tree_args)
