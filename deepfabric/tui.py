from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


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


class TreeBuildingTUI:
    """TUI for tree building operations with simplified progress."""

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.progress = None
        self.overall_task = None
        self.generated_paths = 0
        self.failed_attempts = 0
        self.current_depth = 0
        self.max_depth = 0

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
            TextColumn("[bold blue]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
        )

        self.progress.start()
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

    def finish_building(self, total_paths: int, failed_generations: int) -> None:
        """Finish the tree building process."""
        if self.progress:
            self.progress.stop()

        # Final summary
        self.console.print()
        if failed_generations > 0:
            self.tui.warning(f"Tree building complete with {failed_generations} failures")
        else:
            self.tui.success("Tree building completed successfully")

        self.tui.info(f"Generated {total_paths} total paths")


class GraphBuildingTUI:
    """TUI for graph building operations with simplified progress."""

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.progress = None
        self.overall_task = None
        self.nodes_count = 1  # Start with root
        self.edges_count = 0
        self.failed_attempts = 0

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
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )

        self.progress.start()
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

    def finish_building(self, failed_generations: int) -> None:
        """Finish the graph building process."""
        if self.progress:
            self.progress.stop()

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


class DatasetGenerationTUI:
    """Enhanced TUI for dataset generation with rich integration."""

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console

    def create_rich_progress(self) -> Progress:
        """Create a rich progress bar for dataset generation."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("• [bold green]{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )

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


# Global TUI instance
_tui_instance = None


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
    """Get a dataset generation TUI instance."""
    return DatasetGenerationTUI(get_tui())
