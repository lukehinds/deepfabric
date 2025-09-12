"""
This module defines the abstract base class for topic models.
"""

from abc import ABC, abstractmethod


class TopicModel(ABC):
    """Abstract base class for topic models like Tree and Graph."""

    @abstractmethod
    def build(self, *args, **kwargs) -> None:
        """Builds the topic model."""
        pass

    @abstractmethod
    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        pass
