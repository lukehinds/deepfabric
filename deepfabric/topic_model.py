from abc import ABC, abstractmethod


class TopicModel(ABC):
    """Abstract base class for topic models like Tree and Graph."""

    @abstractmethod
    async def build_async(self) -> None:
        """Asynchronously build the topic model."""
        raise NotImplementedError

    def build(self) -> None:  # pragma: no cover - legacy compatibility
        """Deprecated synchronous entry point kept for legacy compatibility."""
        msg = "TopicModel.build() is no longer supported. Use build_async() instead."
        raise RuntimeError(msg)

    @abstractmethod
    def get_all_paths(self) -> list[list[str]]:
        """Returns all the paths in the topic model."""
        raise NotImplementedError
