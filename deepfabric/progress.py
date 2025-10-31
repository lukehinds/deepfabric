"""Progress reporting system for dataset generation.

This module provides a modular event-based progress reporting system that
allows components to emit progress events (streaming text, step markers, etc.)
without coupling to specific display implementations.

The system uses the Observer pattern to enable multiple observers (TUI, logging,
metrics, etc.) to react to progress events.
"""

from typing import Any, Protocol


class StreamObserver(Protocol):
    """Protocol for observers that react to progress events.

    Implementations can choose which events to handle based on their needs.
    This protocol supports both dataset generation and tree/graph building.
    """

    def on_stream_chunk(self, source: str, chunk: str, metadata: dict[str, Any]) -> None:
        """Called when a chunk of streaming text is received from an LLM.

        Args:
            source: Identifier for the generation source
                - Dataset: "user_question", "agent_reasoning", "tool_sim_weather"
                - Tree/Graph: "topic_generation", "subtopic_expansion"
            chunk: The text chunk received from the LLM
            metadata: Additional context (sample_idx, node_path, depth, etc.)
        """
        ...

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Called when a generation step begins.

        Args:
            step_name: Human-readable name of the step
                - Dataset: "Generating user question", "Simulating tool: get_weather"
                - Tree/Graph: "Expanding node: AI/ML", "Generating subtopics (depth 2)"
            metadata: Additional context (sample_idx, turn_idx, depth, node_path, etc.)
        """
        ...

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Called when a generation step completes.

        Args:
            step_name: Human-readable name of the step
            metadata: Additional context including results (tokens_used, duration, success, etc.)
        """
        ...


class ProgressReporter:
    """Central progress reporter that notifies observers of generation events.

    This class acts as the subject in the Observer pattern, managing a list of
    observers and broadcasting events to them.

    Example:
        >>> reporter = ProgressReporter()
        >>> reporter.attach(my_tui_observer)
        >>> reporter.emit_step_start("Generating question", sample_idx=1)
        >>> reporter.emit_chunk("user_question", "What is the weather", sample_idx=1)
        >>> reporter.emit_step_complete("Generating question", sample_idx=1)
    """

    def __init__(self):
        """Initialize an empty progress reporter."""
        self._observers: list[StreamObserver] = []

    def attach(self, observer: StreamObserver) -> None:
        """Attach an observer to receive progress events.

        Args:
            observer: Observer implementing StreamObserver protocol
        """
        if observer not in self._observers:
            self._observers.append(observer)

    def detach(self, observer: StreamObserver) -> None:
        """Detach an observer from receiving progress events.

        Args:
            observer: Observer to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)

    def emit_chunk(self, source: str, chunk: str, **metadata) -> None:
        """Emit a streaming text chunk to all observers.

        Args:
            source: Identifier for the generation source
            chunk: Text chunk from LLM
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            observer.on_stream_chunk(source, chunk, metadata)

    def emit_step_start(self, step_name: str, **metadata) -> None:
        """Emit a step start event to all observers.

        Args:
            step_name: Human-readable step name
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            observer.on_step_start(step_name, metadata)

    def emit_step_complete(self, step_name: str, **metadata) -> None:
        """Emit a step complete event to all observers.

        Args:
            step_name: Human-readable step name
            **metadata: Additional context as keyword arguments
        """
        for observer in self._observers:
            observer.on_step_complete(step_name, metadata)


# Convenience context manager for tracking steps
class ProgressStep:
    """Context manager for automatic step start/complete reporting.

    Example:
        >>> with ProgressStep(reporter, "Generating question", sample_idx=1):
        ...     # Do work
        ...     reporter.emit_chunk("question", "What is...")
    """

    def __init__(self, reporter: ProgressReporter | None, step_name: str, **metadata: Any):
        """Initialize progress step tracker.

        Args:
            reporter: Progress reporter (None = no-op)
            step_name: Human-readable step name
            **metadata: Additional context
        """
        self.reporter = reporter
        self.step_name = step_name
        self.metadata = metadata

    def __enter__(self):
        """Enter context: emit step start."""
        if self.reporter:
            self.reporter.emit_step_start(self.step_name, **self.metadata)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: emit step complete."""
        if self.reporter:
            self.reporter.emit_step_complete(self.step_name, **self.metadata)
        return False  # Don't suppress exceptions
