import re

from collections import deque
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table
from rich.text import Text

from .progress import StreamObserver

# Constants
STREAM_BUFFER_DISPLAY_THRESHOLD = 100  # Show ellipsis if accumulated text exceeds this
STREAM_TEXT_MAX_LENGTH = 80  # Max characters to display in streaming text


class DeepFabricTUI:
    """Main TUI controller for DeepFabric operations."""

    def __init__(self, console: Console | None = None):
        """Initialize the TUI with rich console."""
        self.console = console or Console()

    def create_header(self, title: str, subtitle: str = "") -> Panel:
        """Create a styled header panel."""
        content = Text(title, style="bold cyan")
        if subtitle:
            content.append(f"\n{subtitle}", style="dim")

        return Panel(
            content,
            border_style="bright_blue",
            padding=(1, 2),
        )

    def create_stats_table(self, stats: dict[str, Any]) -> Table:
        """Create a statistics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")

        for key, value in stats.items():
            table.add_row(f"{key}:", str(value))

        return table

    def success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f" {message}", style="green")

    def warning(self, message: str) -> None:
        """Display a warning message."""
        self.console.print(f"⚠️  {message}", style="yellow")

    def error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"❌ {message}", style="red")

    def info(self, message: str) -> None:
        """Display an info message."""
        self.console.print(f" {message}", style="blue")


class TreeBuildingTUI(StreamObserver):
    """TUI for tree building operations with simplified progress and streaming."""

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.progress = None
        self.overall_task = None
        self.generated_paths = 0
        self.failed_attempts = 0
        self.current_depth = 0
        self.max_depth = 0
        self.stream_buffer = deque(maxlen=2000)
        self.live_display = None
        self.stream_progress = None  # Separate progress for streaming display
        self.stream_task = None

    def start_building(self, model_name: str, depth: int, degree: int) -> None:
        """Start the tree building process."""
        self.max_depth = depth

        # Show header
        header = self.tui.create_header(
            "DeepFabric Tree Generation", f"Building hierarchical topic structure with {model_name}"
        )
        self.console.print(header)
        self.console.print(f"Configuration: depth={depth}, degree={degree}")
        self.console.print()

        # Create simple progress display with indeterminate progress
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", table_column=Column(width=50)),
            BarColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

        # Create separate progress for streaming with animated spinner
        self.stream_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="dim"),
            console=self.console,
        )
        self.stream_task = self.stream_progress.add_task(
            "  Processing: (waiting for LLM output...)"
        )

        # Start Live display with both progress bars
        self.live_display = Live(
            Group(self.progress, self.stream_progress),
            console=self.console,
            refresh_per_second=10,
        )
        self.live_display.start()
        self.overall_task = self.progress.add_task(f"Building topic tree (depth 1/{depth})")

    def start_depth_level(self, depth: int) -> None:
        """Update progress for new depth level."""
        self.current_depth = depth
        if self.progress and self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                description=f"Building topic tree (depth {depth}/{self.max_depth})",
            )

    def start_subtree_generation(self, node_path: list[str], num_subtopics: int) -> None:
        """Log subtree generation without updating progress to avoid flicker."""
        pass

    def complete_subtree_generation(self, success: bool, generated_count: int) -> None:
        """Track completion without updating progress bar."""
        if success:
            self.generated_paths += generated_count
        else:
            self.failed_attempts += 1

    def add_failure(self) -> None:
        """Record a generation failure."""
        self.failed_attempts += 1

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle streaming text from tree generation."""
        self.stream_buffer.append(chunk)

        if self.live_display and self.stream_progress and self.stream_task is not None:
            accumulated_text = "".join(self.stream_buffer)
            if len(accumulated_text) > STREAM_BUFFER_DISPLAY_THRESHOLD:
                display_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]
            else:
                display_text = accumulated_text

            display_text = display_text.replace("\n", " ").replace("\r", "")
            display_text = re.sub(r"\s+", " ", display_text)

            # Update the streaming task description
            self.stream_progress.update(
                self.stream_task, description=f"  Processing: {display_text}"
            )

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Handle step start - tree building doesn't need specific handling."""
        pass

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Handle step complete - tree building doesn't need specific handling."""
        pass

    def finish_building(self, total_paths: int, failed_generations: int) -> None:
        """Finish the tree building process."""
        if self.live_display:
            self.live_display.stop()

        # Final summary
        self.console.print()
        if failed_generations > 0:
            self.tui.warning(f"Tree building complete with {failed_generations} failures")
        else:
            self.tui.success("Tree building completed successfully")

        self.tui.info(f"Generated {total_paths} total paths")


class GraphBuildingTUI(StreamObserver):
    """TUI for graph building operations with simplified progress and streaming."""

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.progress = None
        self.overall_task = None
        self.nodes_count = 1  # Start with root
        self.edges_count = 0
        self.failed_attempts = 0
        self.stream_buffer = deque(maxlen=2000)
        self.live_display = None
        self.stream_progress = None
        self.stream_task = None

    def start_building(self, model_name: str, depth: int, degree: int) -> None:
        """Start the graph building process."""
        # Show header
        header = self.tui.create_header(
            "DeepFabric Graph Generation",
            f"Building interconnected topic structure with {model_name}",
        )
        self.console.print(header)
        self.console.print(f"Configuration: depth={depth}, degree={degree}")
        self.console.print()

        # Create simple progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", table_column=Column(width=50)),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

        # Create separate progress for streaming with animated spinner
        self.stream_progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}", style="dim"),
            console=self.console,
        )
        self.stream_task = self.stream_progress.add_task(
            "  Processing: (waiting for LLM output...)"
        )

        # Start Live display with both progress bars
        self.live_display = Live(
            Group(self.progress, self.stream_progress),
            console=self.console,
            refresh_per_second=10,
        )
        self.live_display.start()
        self.overall_task = self.progress.add_task("  Building topic graph", total=depth)

    def start_depth_level(self, depth: int, leaf_count: int) -> None:
        """Update for new depth level."""
        if self.progress and self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                description=f"  Building graph - depth {depth} ({leaf_count} nodes to expand)",
            )

    def complete_node_expansion(
        self, node_topic: str, subtopics_added: int, connections_added: int
    ) -> None:
        """Track node expansion."""
        _ = node_topic  # Mark as intentionally unused
        self.nodes_count += subtopics_added
        self.edges_count += subtopics_added + connections_added

    def complete_depth_level(self, depth: int) -> None:
        """Complete a depth level."""
        _ = depth  # Mark as intentionally unused
        if self.progress and self.overall_task is not None:
            self.progress.advance(self.overall_task, 1)

    def add_failure(self, node_topic: str) -> None:
        """Record a generation failure."""
        _ = node_topic  # Mark as intentionally unused
        self.failed_attempts += 1

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle streaming text from graph generation."""
        self.stream_buffer.append(chunk)

        if self.live_display and self.stream_progress and self.stream_task is not None:
            accumulated_text = "".join(self.stream_buffer)
            if len(accumulated_text) > STREAM_BUFFER_DISPLAY_THRESHOLD:
                display_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]
            else:
                display_text = accumulated_text

            display_text = display_text.replace("\n", " ").replace("\r", "")
            display_text = re.sub(r"\s+", " ", display_text)

            # Update the streaming task description
            self.stream_progress.update(
                self.stream_task, description=f"  Processing: {display_text}"
            )

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Handle step start - graph building doesn't need specific handling."""
        pass

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Handle step complete - graph building doesn't need specific handling."""
        pass

    def finish_building(self, failed_generations: int) -> None:
        """Finish the graph building process."""
        if self.live_display:
            self.live_display.stop()

        # Show final stats
        self.console.print()
        stats_table = self.tui.create_stats_table(
            {
                "Total Nodes": self.nodes_count,
                "Total Edges": self.edges_count,
                "Failed Attempts": self.failed_attempts,
            }
        )
        self.console.print(Panel(stats_table, title="Final Statistics", border_style="dim"))

        # Final summary
        if failed_generations > 0:
            self.tui.warning(f"Graph building complete with {failed_generations} failures")
        else:
            self.tui.success("Graph building completed successfully")


class DatasetGenerationTUI(StreamObserver):
    """Enhanced TUI for dataset generation with rich integration and streaming display."""

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.stream_buffer = deque(maxlen=2000)  # Last ~2000 chars of streaming text
        self.current_step = ""
        self.current_sample_type = ""  # Track the type of sample being generated
        self.live_display = None  # Will be set by dataset_manager
        self.progress = None
        self.stream_progress = None  # Separate progress for streaming display
        self.stream_task = None
        self.stream_text = Text()  # Rich Text object for streaming content

    def create_rich_progress(self) -> Progress:
        """Create a rich progress bar for dataset generation (without TimeRemainingColumn)."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}", table_column=Column(width=50)),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        return self.progress

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle incoming streaming text chunks from LLM.

        Args:
            source: Source identifier (e.g., "user_question", "tool_sim_weather")
            chunk: Text chunk from LLM
            metadata: Additional context
        """
        # Append chunk to buffer (deque auto-trims to maxlen)
        self.stream_buffer.append(chunk)

        # Update the live display if it's running
        if self.live_display and self.stream_progress and self.stream_task is not None:
            # Get accumulated text and keep only last 80 chars for single-line display
            accumulated_text = "".join(self.stream_buffer)
            # Truncate to last 80 chars to fit on one line in most terminals
            if len(accumulated_text) > STREAM_BUFFER_DISPLAY_THRESHOLD:
                display_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]
            else:
                display_text = accumulated_text

            # Remove newlines and excessive whitespace to keep it on a single line
            display_text = display_text.replace("\n", " ").replace("\r", "")
            # Collapse multiple spaces to single space
            display_text = re.sub(r"\s+", " ", display_text)

            # Build description with sample type if available
            if self.current_sample_type:
                description = f"Processing ({self.current_sample_type}): {display_text}"
            else:
                description = f"  Processing: {display_text}"

            # Update the streaming task description
            self.stream_progress.update(self.stream_task, description=description)

    def on_step_start(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Update current step display.

        Args:
            step_name: Human-readable step name
            metadata: Additional context (sample_idx, conversation_type, etc.)
        """
        # Update current step
        self.current_step = step_name

        # Extract and update sample type from metadata if available
        if "conversation_type" in metadata:
            conv_type = metadata["conversation_type"]
            # Map conversation types to friendly names
            type_map = {
                "basic": "Basic Q&A",
                "chain_of_thought": "Chain of Thought",
                "single_turn_agent": "Single-Turn Agent (Tool Calling)",
                "multi_turn_agent": "Multi-Turn Agent (Tool Calling)",
            }
            self.current_sample_type = type_map.get(conv_type, conv_type)
        elif "agent_mode" in metadata:
            agent_mode = metadata["agent_mode"]
            if agent_mode == "single_turn":
                self.current_sample_type = "Single-Turn Agent (Tool Calling)"
            elif agent_mode == "multi_turn":
                self.current_sample_type = "Multi-Turn Agent (Tool Calling)"
            else:
                self.current_sample_type = f"Agent ({agent_mode})"

        # Don't print anything - the progress bar already shows progress
        # Just silently update internal state

    def on_step_complete(self, step_name: str, metadata: dict[str, Any]) -> None:
        """Handle step completion.

        Args:
            step_name: Human-readable step name
            metadata: Additional context
        """
        # Could add completion markers or timing info here if desired
        pass

    def get_stream_display(self) -> str:
        """Build the streaming text display from buffer.

        Returns:
            Formatted string of recent LLM output
        """
        if not self.stream_buffer:
            return "[dim italic]Waiting for generation...[/dim italic]"

        # Get recent text from buffer
        recent_text = "".join(self.stream_buffer)

        # Truncate if too long and add ellipsis
        max_display_length = 500
        if len(recent_text) > max_display_length:
            recent_text = "..." + recent_text[-max_display_length:]

        return f"[dim]{recent_text}[/dim]"

    def clear_stream_buffer(self) -> None:
        """Clear the streaming text buffer (e.g., between samples)."""
        self.stream_buffer.clear()

    def show_generation_header(self, model_name: str, num_steps: int, batch_size: int) -> None:
        """Display the dataset generation header."""
        header = self.tui.create_header(
            "DeepFabric Dataset Generation", f"Creating synthetic training data with {model_name}"
        )

        stats = {
            "Model": model_name,
            "Steps": num_steps,
            "Batch Size": batch_size,
            "Total Samples": num_steps * batch_size,
        }

        stats_table = self.tui.create_stats_table(stats)

        self.console.print(header)
        self.console.print(Panel(stats_table, title="Generation Parameters", border_style="dim"))
        self.console.print()

    def success(self, message: str) -> None:
        """Display a success message."""
        self.tui.success(message)

    def warning(self, message: str) -> None:
        """Display a warning message."""
        self.tui.warning(message)

    def error(self, message: str) -> None:
        """Display an error message."""
        self.tui.error(message)


# Global TUI instances
_tui_instance = None
_dataset_tui_instance = None


def get_tui() -> DeepFabricTUI:
    """Get the global TUI instance."""
    global _tui_instance  # noqa: PLW0603
    if _tui_instance is None:
        _tui_instance = DeepFabricTUI()
    return _tui_instance


def get_tree_tui() -> TreeBuildingTUI:
    """Get a tree building TUI instance."""
    return TreeBuildingTUI(get_tui())


def get_graph_tui() -> GraphBuildingTUI:
    """Get a graph building TUI instance."""
    return GraphBuildingTUI(get_tui())


def get_dataset_tui() -> DatasetGenerationTUI:
    """Get the global dataset generation TUI instance (singleton)."""
    global _dataset_tui_instance  # noqa: PLW0603
    if _dataset_tui_instance is None:
        _dataset_tui_instance = DatasetGenerationTUI(get_tui())
    return _dataset_tui_instance
