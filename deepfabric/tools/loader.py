"""Tool loading and management utilities."""

import json

from pathlib import Path
from typing import Any

import yaml

from ..exceptions import ConfigurationError
from ..schemas import ToolDefinition, ToolRegistry
from .defaults import DEFAULT_TOOL_REGISTRY


def load_tools_from_file(file_path: str) -> ToolRegistry:
    """Load tool definitions from a JSON or YAML file.

    Args:
        file_path: Path to the tool definitions file

    Returns:
        ToolRegistry with loaded tools

    Raises:
        ConfigurationError: If file cannot be loaded or is invalid
    """
    path = Path(file_path)

    if not path.exists():
        raise ConfigurationError(f"Tool definition file not found: {file_path}")

    try:
        with open(path, encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            elif path.suffix.lower() == ".json":
                data = json.load(f)
            else:
                raise ConfigurationError(  # noqa: TRY301
                    f"Unsupported file format: {path.suffix}. Use .json, .yaml, or .yml"
                )

    except Exception as e:
        raise ConfigurationError(f"Failed to load tool file {file_path}: {str(e)}") from e

    # Validate and convert to ToolRegistry
    try:
        if isinstance(data, dict) and "tools" in data:
            # File contains a tool registry
            return ToolRegistry.model_validate(data)
        if isinstance(data, list):
            # File contains a list of tools
            return ToolRegistry(tools=[ToolDefinition.model_validate(tool) for tool in data])
        raise ConfigurationError("File must contain either a 'tools' key or a list of tools")  # noqa: TRY301

    except Exception as e:
        raise ConfigurationError(f"Invalid tool definitions in {file_path}: {str(e)}") from e


def load_tools_from_dict(tool_dicts: list[dict[str, Any]]) -> ToolRegistry:
    """Load tool definitions from a list of dictionaries.

    Args:
        tool_dicts: List of tool definition dictionaries

    Returns:
        ToolRegistry with loaded tools

    Raises:
        ConfigurationError: If tool definitions are invalid
    """
    try:
        tools = [ToolDefinition.model_validate(tool_dict) for tool_dict in tool_dicts]
        return ToolRegistry(tools=tools)
    except Exception as e:
        raise ConfigurationError(f"Invalid tool definitions: {str(e)}") from e


def merge_tool_registries(*registries: ToolRegistry) -> ToolRegistry:
    """Merge multiple tool registries into one.

    Args:
        *registries: Tool registries to merge

    Returns:
        Combined tool registry

    Note:
        If tools have the same name, later registries override earlier ones.
    """
    tool_map: dict[str, ToolDefinition] = {}

    for registry in registries:
        for tool in registry.tools:
            tool_map[tool.name] = tool

    return ToolRegistry(tools=list(tool_map.values()))


def get_available_tools(
    available_tool_names: list[str] | None = None,
    custom_registry: ToolRegistry | None = None,
) -> ToolRegistry:
    """Get available tools based on configuration.

    Args:
        available_tool_names: List of tool names to include (None means all)
        custom_registry: Custom tool registry to merge with defaults

    Returns:
        ToolRegistry with available tools
    """
    # Start with defaults
    if custom_registry is not None:
        # Merge custom tools with defaults (custom tools override defaults)
        registry = merge_tool_registries(DEFAULT_TOOL_REGISTRY, custom_registry)
    else:
        registry = DEFAULT_TOOL_REGISTRY

    # Filter by available tool names if specified
    if available_tool_names:
        available_tools = []
        for name in available_tool_names:
            tool = registry.get_tool(name)
            if tool is not None:
                available_tools.append(tool)
        registry = ToolRegistry(tools=available_tools)

    return registry


def validate_tool_definition(tool_dict: dict[str, Any]) -> ToolDefinition:
    """Validate a single tool definition dictionary.

    Args:
        tool_dict: Dictionary containing tool definition

    Returns:
        Validated ToolDefinition

    Raises:
        ConfigurationError: If tool definition is invalid
    """
    try:
        return ToolDefinition.model_validate(tool_dict)
    except Exception as e:
        raise ConfigurationError(f"Invalid tool definition: {str(e)}") from e
