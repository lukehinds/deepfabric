import json
import textwrap
import time

from typing import Any

import litellm

from pydantic import BaseModel, Field

from .constants import (
    DEFAULT_MAX_TOKENS,
    MAX_RETRY_ATTEMPTS,
    RETRY_BASE_DELAY,
    TOPIC_TREE_DEFAULT_MODEL,
    TOPIC_TREE_DEFAULT_TEMPERATURE,
)
from .topic_model import TopicModel
from .tui import get_graph_tui, get_tui

# The prompt to be used for generating the graph
GRAPH_GENERATION_PROMPT = """
You are an expert in knowledge graph generation. Your task is to expand a topic into a set of subtopics. For each subtopic, you should also identify if it connects to any other existing topics in the graph.

Here is the current state of the graph:
{{current_graph_summary}}

You are expanding the topic: "{{current_topic}}"

Generate a list of {{num_subtopics}} subtopics. For each subtopic, provide:
1. A "topic" string.
2. A "connections" list of IDs of existing topics it should connect to. This is for creating cross-links. If there are no connections, provide an empty list.

Your response MUST be a JSON array of objects, like this:
{
    "subtopics": [
        {
            "topic": "New Subtopic 1",
            "connections": [1, 2]
        },
        {
            "topic": "New Subtopic 2",
            "connections": []
        }
    ]
}
"""


def validate_graph_response(response_text: str) -> dict[str, Any] | None:
    """Clean and validate the JSON response for the graph from the LLM."""
    try:
        return json.loads(response_text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse the input string as JSON.\n{e}")
        return None


class GraphArguments(BaseModel):
    """Arguments for constructing a topic graph."""

    root_prompt: str = Field(
        ..., min_length=1, description="The initial prompt to start the topic graph"
    )
    model_name: str = Field(
        TOPIC_TREE_DEFAULT_MODEL, min_length=1, description="The name of the model to be used"
    )
    temperature: float = Field(
        TOPIC_TREE_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperature for model generation",
    )
    graph_degree: int = Field(3, ge=1, le=10, description="The branching factor of the graph")
    graph_depth: int = Field(2, ge=1, le=5, description="The depth of the graph")


# Pydantic Models for strict data representation


class NodeModel(BaseModel):
    """Pydantic model for a node in the graph."""

    id: int
    topic: str
    children: list[int] = Field(default_factory=list)
    parents: list[int] = Field(default_factory=list)


class GraphModel(BaseModel):
    """Pydantic model for the entire topic graph."""

    nodes: dict[int, NodeModel]
    root_id: int


# Core graph implementation


class Node:
    """Represents a node in the Graph for runtime manipulation."""

    def __init__(self, topic: str, node_id: int):
        self.topic: str = topic
        self.id: int = node_id
        self.children: list[Node] = []
        self.parents: list[Node] = []

    def to_pydantic(self) -> NodeModel:
        """Converts the runtime Node to its Pydantic model representation."""
        return NodeModel(
            id=self.id,
            topic=self.topic,
            children=[child.id for child in self.children],
            parents=[parent.id for parent in self.parents],
        )


class Graph(TopicModel):
    """Represents the topic graph and manages its structure."""

    def __init__(self, args: GraphArguments):
        self.args = args
        self.root: Node = Node(args.root_prompt, 0)
        self.nodes: dict[int, Node] = {0: self.root}
        self._next_node_id: int = 1
        self.failed_generations: list[dict[str, Any]] = []

    def _wrap_text(self, text: str, width: int = 30) -> str:
        """Wrap text to a specified width."""
        return "\n".join(textwrap.wrap(text, width=width))

    def add_node(self, topic: str) -> Node:
        """Adds a new node to the graph."""
        node = Node(topic, self._next_node_id)
        self.nodes[node.id] = node
        self._next_node_id += 1
        return node

    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Adds a directed edge from a parent to a child node, avoiding duplicates."""
        parent_node = self.nodes.get(parent_id)
        child_node = self.nodes.get(child_id)
        if parent_node and child_node:
            if child_node not in parent_node.children:
                parent_node.children.append(child_node)
            if parent_node not in child_node.parents:
                child_node.parents.append(parent_node)

    def to_pydantic(self) -> GraphModel:
        """Converts the runtime graph to its Pydantic model representation."""
        return GraphModel(
            nodes={node_id: node.to_pydantic() for node_id, node in self.nodes.items()},
            root_id=self.root.id,
        )

    def to_json(self) -> str:
        """Returns a JSON representation of the graph."""
        pydantic_model = self.to_pydantic()
        return pydantic_model.model_dump_json(indent=2)

    def save(self, save_path: str) -> None:
        """Save the topic graph to a file."""
        try:
            with open(save_path, "w") as f:
                f.write(self.to_json())

            tui = get_tui()
            tui.success(f"Topic graph saved to {save_path}")

        except Exception as e:
            tui = get_tui()
            tui.error(f"Error saving topic graph: {str(e)}")
            raise

    @classmethod
    def from_json(cls, json_path: str, args: GraphArguments) -> "Graph":
        """Load a topic graph from a JSON file."""
        with open(json_path) as f:
            data = json.load(f)

        graph_model = GraphModel(**data)
        graph = cls(args)
        graph.nodes = {}

        # Create nodes
        for node_model in graph_model.nodes.values():
            node = Node(node_model.topic, node_model.id)
            graph.nodes[node.id] = node
            if node.id == graph_model.root_id:
                graph.root = node

        # Create edges
        for node_model in graph_model.nodes.values():
            for child_id in node_model.children:
                graph.add_edge(node_model.id, child_id)

        graph._next_node_id = max(graph.nodes.keys()) + 1
        return graph

    def visualize(self, save_path: str) -> None:
        """Visualize the graph and save it to a file."""
        try:
            from mermaid import Mermaid  # noqa: PLC0415
        except ImportError:
            print("Please install mermaid-py to visualize the graph: uv add mermaid-py")
            return

        graph_definition = "graph TD\n"
        for node in self.nodes.values():
            graph_definition += f'    {node.id}["{self._wrap_text(node.topic)}"]\n'

        for node in self.nodes.values():
            for child in node.children:
                graph_definition += f"    {node.id} --> {child.id}\n"

        mermaid = Mermaid(graph_definition)
        mermaid.to_svg(f"{save_path}.svg")

    def build(self) -> None:
        """Builds the graph by iteratively calling the LLM to get subtopics and connections."""
        # Initialize TUI
        tui = get_graph_tui()
        tui.start_building(self.args.model_name, self.args.graph_depth, self.args.graph_degree)

        try:
            for depth in range(self.args.graph_depth):
                leaf_nodes = [node for node in self.nodes.values() if not node.children]
                tui.start_depth_level(depth + 1, len(leaf_nodes))

                for node in leaf_nodes:
                    subtopics_added, connections_added = self.get_subtopics_and_connections(
                        node, self.args.graph_degree, tui
                    )
                    tui.complete_node_expansion(node.topic, subtopics_added, connections_added)

                tui.complete_depth_level(depth + 1)

            tui.finish_building(len(self.failed_generations))

        except Exception as e:
            if tui.progress:
                tui.progress.stop()
            tui.tui.error(f"Error building graph: {str(e)}")
            raise

    def get_subtopics_and_connections(  # noqa: PLR0912
        self, parent_node: Node, num_subtopics: int, tui=None
    ) -> tuple[int, int]:
        """Generate subtopics and connections for a given node. Returns (subtopics_added, connections_added)."""
        subtopics_added = 0
        connections_added = 0
        graph_summary = self.to_json()  # A simple summary for now
        prompt = GRAPH_GENERATION_PROMPT.replace("{{current_graph_summary}}", graph_summary)
        prompt = prompt.replace("{{current_topic}}", parent_node.topic)
        prompt = prompt.replace("{{num_subtopics}}", str(num_subtopics))

        max_retries = MAX_RETRY_ATTEMPTS
        retries = 0
        last_error = "No error recorded"

        while retries < max_retries:
            try:
                completion_args = {
                    "model": self.args.model_name,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "temperature": self.args.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "response_format": {"type": "json_object"},
                    "stream": False,
                }

                response = litellm.completion(**completion_args)

                # Extract content from the response - use getattr to safely access attributes
                response_content = None

                # Try standard OpenAI format first
                choices = getattr(response, "choices", None)
                if choices and len(choices) > 0:
                    choice = choices[0]
                    message = getattr(choice, "message", None)
                    if message:
                        response_content = getattr(message, "content", None)
                    if not response_content:
                        response_content = getattr(choice, "text", None)

                # Try direct content access if no content found yet
                if not response_content:
                    response_content = getattr(response, "content", None)

                if not response_content:
                    response_content = getattr(response, "text", None)

                # Try getting content from response as dict if still no content
                if not response_content and isinstance(response, dict):
                    try:
                        if "choices" in response and response["choices"]:
                            choice = response["choices"][0]
                            if isinstance(choice, dict):
                                if "message" in choice and "content" in choice["message"]:
                                    response_content = choice["message"]["content"]
                                elif "text" in choice:
                                    response_content = choice["text"]
                        elif "content" in response:
                            response_content = response["content"]
                    except (KeyError, IndexError, TypeError):
                        pass

                if not response_content:
                    raise ValueError("No content in response")  # noqa: TRY003, TRY301

                subtopics_data = validate_graph_response(response_content)

                if (
                    subtopics_data
                    and "subtopics" in subtopics_data
                    and isinstance(subtopics_data["subtopics"], list)
                ):
                    for subtopic_data in subtopics_data["subtopics"]:
                        new_node = self.add_node(subtopic_data["topic"])
                        self.add_edge(parent_node.id, new_node.id)
                        subtopics_added += 1
                        for connection_id in subtopic_data.get("connections", []):
                            if connection_id in self.nodes:
                                self.add_edge(connection_id, new_node.id)
                                connections_added += 1
                    return subtopics_added, connections_added

                last_error = "Insufficient valid subtopics generated"
                if not tui:  # Only print if no TUI
                    print(f"Attempt {retries + 1}: {last_error}. Retrying...")

            except Exception as e:
                last_error = str(e)
                if not tui:  # Only print if no TUI
                    print(
                        f"Error generating subtopics (attempt {retries + 1}/{max_retries}): {last_error}"
                    )

            retries += 1
            if retries < max_retries:
                time.sleep(RETRY_BASE_DELAY * retries)  # Linear backoff instead of exponential

        self.failed_generations.append(
            {"node_id": parent_node.id, "attempts": retries, "last_error": last_error}
        )
        if tui:
            tui.add_failure(parent_node.topic)
        else:
            print(
                f"Failed to generate valid subtopics for node {parent_node.id} after {max_retries} attempts."
            )
        return 0, 0  # No subtopics or connections added on failure

    def get_all_paths(self) -> list[list[str]]:
        """Returns all paths from the root to leaf nodes."""
        paths = []
        self._dfs_paths(self.root, [self.root.topic], paths)
        return paths

    def _dfs_paths(self, node: Node, current_path: list[str], paths: list[list[str]]) -> None:
        """Helper function for DFS traversal to find all paths."""
        if not node.children:
            paths.append(current_path)
            return

        for child in node.children:
            self._dfs_paths(child, current_path + [child.topic], paths)

    def _has_cycle_util(self, node: Node, visited: set[int], recursion_stack: set[int]) -> bool:
        """Utility function for cycle detection."""
        visited.add(node.id)
        recursion_stack.add(node.id)

        for child in node.children:
            if child.id not in visited:
                if self._has_cycle_util(child, visited, recursion_stack):
                    return True
            elif child.id in recursion_stack:
                return True

        recursion_stack.remove(node.id)
        return False

    def has_cycle(self) -> bool:
        """Checks if the graph contains a cycle."""
        visited: set[int] = set()
        recursion_stack: set[int] = set()
        for node_id in self.nodes:
            if node_id not in visited and self._has_cycle_util(
                self.nodes[node_id], visited, recursion_stack
            ):
                return True
        return False
