import contextlib
import os
import re

from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Any

from rich.console import Console, RenderableType
from rich.layout import Layout
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
STREAM_BUFFER_DISPLAY_THRESHOLD = 1000  # Show ellipsis if accumulated text exceeds this
STREAM_TEXT_MAX_LENGTH = 8000  # Max characters to display in streaming text
STREAM_PANEL_WIDTH = 64  # Minimum width for the right-hand streaming pane
EVENT_LOG_MAX_LINES = 8
STREAM_RENDER_THROTTLE_S = 0.06
STREAM_FIXED_LINES = 16  # Fixed visible lines for streaming preview (used by all previews)
MIN_PREVIEW_LINES = 4  # Minimum preview lines to enforce


@dataclass
class TUISettings:
    mode: str = "rich"  # 'rich' or 'simple'
    syntax: bool = True  # enable syntax highlighting in preview


_tui_settings = TUISettings()


def configure_tui(mode: str) -> None:
    mode = (mode or "rich").lower().strip()
    if mode not in {"rich", "simple"}:
        mode = "rich"
    _tui_settings.mode = mode
    _tui_settings.syntax = mode == "rich"


def get_tui_settings() -> TUISettings:
    return _tui_settings


def get_preview_lines() -> int:
    """Get preview height in lines, overridable by DF_TUI_PREVIEW_LINES env var."""
    try:
        v = int(os.getenv("DF_TUI_PREVIEW_LINES", str(STREAM_FIXED_LINES)))
    except Exception:  # noqa: BLE001
        return STREAM_FIXED_LINES
    else:
        return v if v > MIN_PREVIEW_LINES else STREAM_FIXED_LINES


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
            padding=(0, 1),
        )

    def build_stream_panel(
        self, content: RenderableType | str, title: str = "Streaming Preview"
    ) -> Panel:
        """Create a compact right-hand panel showing recent streaming output.

        Accepts any Rich renderable (Text, Syntax, Group, etc.) or a plain string.
        """
        renderable: RenderableType
        renderable = Text(content, style="dim") if isinstance(content, str) else content
        return Panel(renderable, title=title, border_style="dim", padding=(0, 1), expand=True)

    def build_events_panel(self, events: list[str], title: str = "Events") -> Panel:
        """Create a compact events panel for the left column."""
        if not events:
            text = Text("Waiting...", style="dim")
        else:
            # Keep events short; show newest at bottom
            text = Text("\n".join(events[-EVENT_LOG_MAX_LINES:]))
        return Panel(text, title=title, border_style="dim", padding=(0, 1))

    def create_footer(self, layout: Layout, title: str = "Run Status") -> Progress:
        """Attach a footer progress panel to the provided root layout.

        Expects the root layout to have children: 'main' and 'footer'.
        Returns a Progress instance so callers can create tasks and update.
        """
        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        layout["footer"].update(Panel(progress, title=title, border_style="dim", padding=(0, 1)))
        return progress

    def create_stats_table(self, stats: dict[str, Any]) -> Table:
        """Create a statistics table."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")

        for key, value in stats.items():
            table.add_row(f"{key}:", str(value))

        return table

    def build_context_panel(
        self,
        *,
        root_topic: str | None,
        topic_model_type: str | None,
        path: list[str] | None,
    ) -> Panel:
        """Create a small context panel for current topic path info."""
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")

        label_current = "Current Leaf" if topic_model_type == "tree" else "Current Node"

        if root_topic:
            table.add_row("Root Topic:", root_topic)

        if path:
            parent = path[-2] if len(path) > 1 else "-"
            leaf = path[-1]
            table.add_row(f"{label_current}:", leaf)
            table.add_row("Parent:", str(parent))
            table.add_row("Path:", " → ".join(path))
        else:
            table.add_row("Status:", "Waiting for first sample...")

        return Panel(table, title="Context", border_style="dim", padding=(0, 1))

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
        self.live_layout: Layout | None = None
        self.events_log = deque(maxlen=EVENT_LOG_MAX_LINES)
        self.simple_mode = False
        self.current_topic_path: list[str] | None = None
        self.root_topic: str | None = None

    def start_building(self, model_name: str, depth: int, degree: int, root_topic: str) -> None:
        """Start the tree building process."""
        self.max_depth = depth
        self.root_topic = root_topic

        # If simple/headless mode, print static header and return without Live
        if get_tui_settings().mode == "simple":
            header_panel = self.tui.create_header(
                "DeepFabric Tree Generation",
                f"Building hierarchical topic structure with {model_name}",
            )
            self.console.print(header_panel)
            self.console.print(f"Configuration: depth={depth}, degree={degree}")
            self.console.print()
            self.simple_mode = True
            return

        # Create simple progress display with indeterminate progress
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            TimeElapsedColumn(),
            console=self.console,
        )
        # Two-pane layout: left header + progress + events; right status + preview
        layout = Layout(name="root")
        layout.split(Layout(name="main"), Layout(name="footer", size=3))
        left = Layout(name="left", ratio=3)
        right = Layout(name="right", ratio=2)
        right.minimum_size = STREAM_PANEL_WIDTH

        header_panel = self.tui.create_header(
            "DeepFabric Tree Generation",
            f"Building hierarchical topic structure with {model_name}",
        )
        stats = {"Model": model_name, "Depth": f"{depth}", "Degree": f"{degree}"}
        stats_table = self.tui.create_stats_table(stats)
        params_panel = Panel(stats_table, title="Generation Parameters", border_style="dim")

        left.split(
            Layout(name="header", size=4),
            Layout(name="params", size=5),
            Layout(name="context", size=5),
            Layout(name="events"),
        )
        left["header"].update(header_panel)
        left["params"].update(params_panel)
        left["context"].update(self._context_panel())
        left["events"].update(self.tui.build_events_panel(list(self.events_log)))
        # Right column: status + fixed-height preview
        right.split(
            Layout(name="status", size=6),
            Layout(name="preview", size=get_preview_lines()),
        )
        layout["main"].split_row(left, right)
        right["status"].update(self._status_panel())
        right["preview"].update(self.tui.build_stream_panel("Waiting for generation..."))

        # Start Live display with layout
        self.live_layout = layout
        # Footer progress
        self.footer_progress = self.tui.create_footer(layout, title="Run Status")
        self.footer_task = self.footer_progress.add_task("Building topic tree", total=depth)

        self.live_display = Live(layout, console=self.console, refresh_per_second=15, screen=True)
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
        self.events_log.append(f"→ Depth {depth}/{self.max_depth} started")
        self._refresh_left()
        # Advance footer on each depth start (only after first)
        self.update_status_panel()

    def start_subtree_generation(self, node_path: list[str], _num_subtopics: int) -> None:
        """Log subtree generation without updating progress to avoid flicker."""
        self.current_topic_path = node_path
        self._refresh_context()
        pass

    def complete_subtree_generation(self, success: bool, generated_count: int) -> None:
        """Track completion without updating progress bar."""
        if success:
            self.generated_paths += generated_count
        else:
            self.failed_attempts += 1
        # Log succinct outcome
        status = "ok" if success else "fail"
        self.events_log.append(f"✓ Subtree {status} (+{generated_count} paths)")
        self._refresh_left()
        self.update_status_panel()
        # Advance footer on completed depth
        with contextlib.suppress(Exception):
            self.footer_progress.update(self.footer_task, advance=1)

    def add_failure(self) -> None:
        """Record a generation failure."""
        self.failed_attempts += 1
        self.events_log.append("✗ Generation failed")
        self._refresh_left()
        self.update_status_panel()

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle streaming text from tree generation."""
        self.stream_buffer.append(chunk)
        if self.live_display and self.live_layout is not None:
            accumulated_text = "".join(self.stream_buffer)
            # Trim to last N chars for performance
            if len(accumulated_text) > STREAM_TEXT_MAX_LENGTH:
                accumulated_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]
            # Clean CRs, normalize spaces but preserve newlines
            display_text = accumulated_text.replace("\r", "")
            display_text = re.sub(r"[^\S\n]+", " ", display_text)

            # Compute fixed-height preview tail
            target_lines = get_preview_lines()
            lines = display_text.splitlines()
            visible = "\n".join(lines[-target_lines:])

            # Update right-hand panel (nested preview)
            try:
                container = self.live_layout["main"]["right"]["preview"]
            except Exception:
                container = self.live_layout["main"]["right"]
            container.update(self.tui.build_stream_panel(visible))

    def _context_panel(self) -> Panel:
        return self.tui.build_context_panel(
            root_topic=self.root_topic,
            topic_model_type="tree",
            path=self.current_topic_path,
        )

    def _refresh_context(self) -> None:
        if self.live_layout is not None:
            try:
                self.live_layout["main"]["left"]["context"].update(self._context_panel())
            except Exception:
                return

    def _refresh_left(self) -> None:
        if self.live_layout is not None:
            try:
                # Update events panel in left column
                self.live_layout["main"]["left"]["events"].update(
                    self.tui.build_events_panel(list(self.events_log))
                )
            except Exception:
                return

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
        self.events_log.append("✓ Tree building completed")
        self.update_status_panel()

    # ---- Status panel for Tree ----
    def _status_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Depth:", f"{self.current_depth}/{self.max_depth}")
        table.add_row("Nodes:", str(self.generated_paths))
        if self.failed_attempts:
            table.add_row("Failed:", str(self.failed_attempts))
        return Panel(table, title="Status", border_style="dim", padding=(0, 1))

    def update_status_panel(self) -> None:
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["right"]["status"].update(self._status_panel())
        except Exception:
            return


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
        self.live_layout: Layout | None = None
        self.events_log = deque(maxlen=EVENT_LOG_MAX_LINES)
        self.simple_mode = False
        self.current_topic_path: list[str] | None = None
        self.root_topic: str | None = None

    def start_building(self, model_name: str, depth: int, degree: int, root_topic: str) -> None:
        """Start the graph building process."""
        self.max_depth = depth
        self.current_depth = 0
        self.root_topic = root_topic
        # If simple/headless mode, print static header and return
        if get_tui_settings().mode == "simple":
            header = self.tui.create_header(
                "DeepFabric Graph Generation",
                f"Building interconnected topic structure with {model_name}",
            )
            self.console.print(header)
            self.console.print(f"Configuration: depth={depth}, degree={degree}")
            self.console.print()
            self.simple_mode = True
            return

        # Create simple progress display
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        # Two-pane layout: left header + events; right status + preview with footer at bottom
        layout = Layout(name="root")
        layout.split(Layout(name="main"), Layout(name="footer", size=3))
        left = Layout(name="left", ratio=3)
        right = Layout(name="right", ratio=2)
        right.minimum_size = STREAM_PANEL_WIDTH

        header_panel = self.tui.create_header(
            "DeepFabric Graph Generation",
            f"Building interconnected topic structure with {model_name}",
        )
        stats = {"Model": model_name, "Depth": f"{depth}", "Degree": f"{degree}"}
        stats_table = self.tui.create_stats_table(stats)
        params_panel = Panel(stats_table, title="Generation Parameters", border_style="dim")

        left.split(
            Layout(name="header", size=4),
            Layout(name="params", size=5),
            Layout(name="context", size=5),
            Layout(name="events"),
        )
        left["header"].update(header_panel)
        left["params"].update(params_panel)
        left["context"].update(self._context_panel())
        left["events"].update(self.tui.build_events_panel(list(self.events_log)))
        right.split(
            Layout(name="status", size=6),
            Layout(name="preview", size=get_preview_lines()),
        )
        layout["main"].split_row(left, right)
        right["status"].update(self._status_panel())
        right["preview"].update(self.tui.build_stream_panel("Waiting for generation..."))

        # Footer progress
        self.footer_progress = self.tui.create_footer(layout, title="Run Status")
        self.footer_task = self.footer_progress.add_task("Building topic graph", total=depth)

        self.live_layout = layout
        self.live_display = Live(layout, console=self.console, refresh_per_second=15, screen=True)
        self.live_display.start()
        self.overall_task = self.progress.add_task("  Building topic graph", total=depth)

    def start_depth_level(self, depth: int, leaf_count: int) -> None:
        """Update for new depth level."""
        if self.progress and self.overall_task is not None:
            self.progress.update(
                self.overall_task,
                description=f"  Building graph - depth {depth} ({leaf_count} nodes to expand)",
            )
        self.events_log.append(f"→ Depth {depth} start ({leaf_count} nodes)")
        self._refresh_left()
        self.current_depth = depth
        self.update_status_panel()

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
        self.events_log.append(f"✓ Depth {depth} complete")
        self._refresh_left()
        self.update_status_panel()
        # Advance footer on depth complete
        with contextlib.suppress(Exception):
            self.footer_progress.update(self.footer_task, advance=1)

    def add_failure(self, node_topic: str) -> None:
        """Record a generation failure."""
        _ = node_topic  # Mark as intentionally unused
        self.failed_attempts += 1
        self.events_log.append("✗ Node expansion failed")
        self._refresh_left()

    def on_stream_chunk(self, _source: str, chunk: str, _metadata: dict[str, Any]) -> None:
        """Handle streaming text from graph generation."""
        self.stream_buffer.append(chunk)
        if self.live_display and self.live_layout is not None:
            accumulated_text = "".join(self.stream_buffer)
            if len(accumulated_text) > STREAM_TEXT_MAX_LENGTH:
                accumulated_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]
            display_text = accumulated_text.replace("\r", "")
            display_text = re.sub(r"[^\S\n]+", " ", display_text)

            target_lines = get_preview_lines()
            lines = display_text.splitlines()
            visible = "\n".join(lines[-target_lines:])

            # Update the streaming panel
            try:
                container = self.live_layout["main"]["right"]["preview"]
            except Exception:
                container = self.live_layout["main"]["right"]
            container.update(self.tui.build_stream_panel(visible))

    def _context_panel(self) -> Panel:
        return self.tui.build_context_panel(
            root_topic=self.root_topic,
            topic_model_type="graph",
            path=self.current_topic_path,
        )

    def _refresh_context(self) -> None:
        if self.live_layout is not None:
            try:
                self.live_layout["main"]["left"]["context"].update(self._context_panel())
            except Exception:
                return

    def _refresh_left(self) -> None:
        if self.live_layout is not None:
            try:
                self.live_layout["main"]["left"]["events"].update(
                    self.tui.build_events_panel(list(self.events_log))
                )
            except Exception:
                return

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
        self.events_log.append("✓ Graph building completed")
        self.update_status_panel()

    # ---- Status panel for Graph ----
    def _status_panel(self) -> Panel:
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Depth:", f"{self.current_depth}/{getattr(self, 'max_depth', 0)}")
        table.add_row("Nodes:", str(self.nodes_count))
        table.add_row("Edges:", str(self.edges_count))
        if self.failed_attempts:
            table.add_row("Failed:", str(self.failed_attempts))
        return Panel(table, title="Status", border_style="dim", padding=(0, 1))

    def update_status_panel(self) -> None:
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["right"]["status"].update(self._status_panel())
        except Exception:
            return


class DatasetGenerationTUI(StreamObserver):
    """Enhanced TUI for dataset generation with rich integration and streaming display."""

    live_display: Live | None
    live_layout: Layout | None

    def __init__(self, tui: DeepFabricTUI):
        self.tui = tui
        self.console = tui.console
        self.stream_buffer = deque(maxlen=2000)  # Last ~2000 chars of streaming text
        self.current_step = ""
        self.current_sample_type = ""  # Track the type of sample being generated
        self.live_display = None  # Will be set by dataset_manager
        self.live_layout = None  # Provided by dataset_manager
        self.progress = None
        self.stream_text = Text("Waiting for generation...", style="dim")  # Streaming content
        self.events_log = deque(maxlen=EVENT_LOG_MAX_LINES)
        # Context tracking
        self.root_topic_prompt: str | None = None
        self.topic_model_type: str | None = None  # 'tree' or 'graph'
        self.current_topic_path: list[str] | None = None
        self._last_render_t = 0.0
        self._last_visible_key = ""
        # Status tracking
        self.status_total_steps = 0
        self.status_current_step = 0
        self.status_total_samples = 0
        self.status_samples_done = 0
        self.status_failed_total = 0
        self.status_step_started_at = 0.0

    def create_rich_progress(self) -> Progress:
        """Create a rich progress bar for dataset generation (without TimeRemainingColumn)."""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(
                "[bold blue]{task.description}",
                table_column=Column(ratio=1, overflow="ellipsis"),
            ),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
        )
        return self.progress

    def build_generation_panels(
        self, model_name: str, num_steps: int, batch_size: int
    ) -> tuple[Panel, Panel]:
        """Return header and parameters panels for layout use (no direct printing)."""
        header = self.tui.create_header(
            "DeepFabric Dataset Generation",
            f"Creating synthetic traces with {model_name}",
        )
        stats = {
            "Model": model_name,
            "Steps": num_steps,
            "Batch Size": batch_size,
            "Total Samples": num_steps * batch_size,
        }
        stats_table = self.tui.create_stats_table(stats)
        params_panel = Panel(stats_table, title="Generation Parameters", border_style="dim")

        # Seed events log
        self.events_log.append(
            f"Start • steps={num_steps} batch={batch_size} total={num_steps * batch_size}"
        )
        return header, params_panel

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
        if self.live_display and self.live_layout is not None:
            self.update_stream_panel()

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

        # Update current topic path if provided
        topic_path = metadata.get("topic_path") if isinstance(metadata, dict) else None
        if topic_path:
            # Ensure list[str]
            try:
                self.current_topic_path = list(topic_path)
            except Exception:  # noqa: BLE001
                self.current_topic_path = None
            self.update_context_panel()

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
        max_display_length = 300
        if len(recent_text) > max_display_length:
            recent_text = "..." + recent_text[-max_display_length:]

        return f"[dim]{recent_text}[/dim]"

    def clear_stream_buffer(self) -> None:
        """Clear the streaming text buffer (e.g., between samples)."""
        self.stream_buffer.clear()

    # Deprecated printer retained for backward compatibility
    def show_generation_header(self, model_name: str, num_steps: int, batch_size: int) -> None:
        header, params_panel = self.build_generation_panels(model_name, num_steps, batch_size)
        self.console.print(header)
        self.console.print(params_panel)
        self.console.print()

    def _context_panel(self) -> Panel:
        return self.tui.build_context_panel(
            root_topic=self.root_topic_prompt,
            topic_model_type=self.topic_model_type,
            path=self.current_topic_path,
        )

    def update_context_panel(self) -> None:
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["left"]["context"].update(self._context_panel())
        except Exception:
            return

    # --- Status Panel helpers ---
    def init_status(self, total_steps: int, total_samples: int) -> None:
        self.status_total_steps = total_steps
        self.status_total_samples = total_samples
        self.status_current_step = 0
        self.status_samples_done = 0
        self.status_failed_total = 0
        self.status_step_started_at = 0.0

    def status_step_start(self, step: int, total_steps: int | None = None) -> None:
        self.status_current_step = step
        if total_steps is not None:
            self.status_total_steps = total_steps
        self.status_step_started_at = monotonic()
        self.update_status_panel()

    def status_step_complete(self, samples_generated: int, failed_in_step: int = 0) -> None:
        self.status_samples_done += max(0, int(samples_generated))
        self.status_failed_total += max(0, int(failed_in_step))
        self.update_status_panel()

    def _status_panel(self) -> Panel:
        elapsed = 0.0
        if self.status_step_started_at:
            elapsed = max(0.0, monotonic() - self.status_step_started_at)
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(style="white")
        table.add_row("Step:", f"{self.status_current_step}/{self.status_total_steps}")
        table.add_row("Step Elapsed:", f"{elapsed:0.1f}s")
        table.add_row("Generated:", f"{self.status_samples_done}/{self.status_total_samples}")
        if self.status_failed_total:
            table.add_row("Failed:", str(self.status_failed_total))
        return Panel(table, title="Status", border_style="dim", padding=(0, 1))

    def update_status_panel(self) -> None:
        if self.live_layout is None:
            return
        try:
            self.live_layout["main"]["right"]["status"].update(self._status_panel())
        except Exception:
            return

    def success(self, message: str) -> None:
        """Display a success message."""
        self.tui.success(message)

    def warning(self, message: str) -> None:
        """Display a warning message."""
        self.tui.warning(message)

    def error(self, message: str) -> None:
        """Display an error message."""
        self.tui.error(message)

    def info(self, message: str) -> None:
        """Display an info message."""
        self.tui.info(message)

    # --- Compact two-pane helpers ---
    def update_stream_panel(self) -> None:
        """Refresh the right-hand streaming panel with current buffer text."""
        if self.live_layout is None:
            return

        # Throttle: avoid re-rendering too frequently
        now = monotonic()
        if now - getattr(self, "_last_render_t", 0.0) < STREAM_RENDER_THROTTLE_S:  # noqa: PLR2004
            return

        # Build multi-line text; show the last N lines based on terminal height
        accumulated_text = "".join(self.stream_buffer)
        if len(accumulated_text) > STREAM_TEXT_MAX_LENGTH:
            accumulated_text = "..." + accumulated_text[-STREAM_TEXT_MAX_LENGTH:]

        normalized = accumulated_text.replace("\r", "")
        normalized = re.sub(r"[^\S\n]+", " ", normalized)

        # Use fixed number of lines to keep motion stable
        target_lines = STREAM_FIXED_LINES
        lines = normalized.splitlines()
        if len(lines) >= int(target_lines / 2):
            # Plenty of newlines: take the last N lines
            visible_lines = lines[-target_lines:]
        else:
            # Low-newline content: take a character tail and then split
            approx_right_cols = max(40, int(self.console.size.width * 0.42))
            char_tail = max(800, approx_right_cols * max(8, target_lines - 2))
            tail = normalized[-char_tail:]
            visible_lines = tail.splitlines()[-target_lines:]

        # Fixed-height layout handles stability; render just the last N lines
        visible = "\n".join(visible_lines)

        # Skip update if content suffix hasn't changed
        key = visible[-200:]
        if key == getattr(self, "_last_visible_key", ""):
            return
        self._last_visible_key = key

        # Build simple dim text renderable (syntax highlighting removed)
        renderable = Text(visible, style="dim")

        title = (
            f"Streaming Preview • {self.current_sample_type}"
            if self.current_sample_type
            else "Streaming Preview"
        )
        # Support both old layout (right only) and new split layout (right.preview)
        try:
            container = self.live_layout["main"]["right"]["preview"]
        except Exception:
            container = self.live_layout["main"]["right"]
        container.update(self.tui.build_stream_panel(renderable, title=title))
        self._last_render_t = now

    def log_event(self, message: str) -> None:
        """Append an event to the left-side event log and refresh."""
        self.events_log.append(message)
        if self.live_layout is not None:
            self.live_layout["main"]["left"]["events"].update(
                self.tui.build_events_panel(list(self.events_log))
            )


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
