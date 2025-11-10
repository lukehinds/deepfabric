"""Tool loading and management utilities."""

import json
import logging

from pathlib import Path
from typing import Any

import yaml

from pydantic import ValidationError

from ..exceptions import ConfigurationError
from ..schemas import ToolDefinition, ToolRegistry
from .defaults import DEFAULT_TOOL_REGISTRY

logger = logging.getLogger(__name__)


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
        custom_registry: Custom tool registry (if provided, REPLACES defaults entirely)

    Returns:
        ToolRegistry with available tools
    """
    # Use custom tools if provided (replaces defaults), otherwise use defaults
    registry = custom_registry if custom_registry is not None else DEFAULT_TOOL_REGISTRY

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


def tools_to_trl_format(tools: list[ToolDefinition] | ToolRegistry) -> list[dict[str, Any]]:
    """Convert tool definitions to TRL/OpenAI function calling schema format.

    This is a convenience function for converting either a list of ToolDefinitions
    or a ToolRegistry to the format required by HuggingFace TRL's SFTTrainer.

    Args:
        tools: Either a list of ToolDefinition objects or a ToolRegistry

    Returns:
        List of tool schemas in OpenAI function calling format

    Example:
        >>> from deepfabric.tools.defaults import DEFAULT_TOOL_REGISTRY
        >>> trl_tools = tools_to_trl_format(DEFAULT_TOOL_REGISTRY)
        >>> # Use in dataset: {"messages": [...], "tools": trl_tools}
    """
    if isinstance(tools, ToolRegistry):
        return tools.to_trl_format()
    return [tool.to_openai() for tool in tools]


def convert_available_tools_to_trl(sample: dict) -> dict:
    """Convert a sample's available_tools to TRL format with a 'tools' field.

    This function takes a DeepFabric dataset sample and adds a 'tools' field
    in TRL/OpenAI format while preserving the original sample structure.

    Args:
        sample: Dataset sample containing 'available_tools' field

    Returns:
        Updated sample with 'tools' field added in TRL format

    Example:
        >>> sample = {"messages": [...], "available_tools": [...]}
        >>> trl_sample = convert_available_tools_to_trl(sample)
        >>> # trl_sample now has: {"messages": [...], "tools": [...]}
    """
    if "available_tools" not in sample:
        return sample

    # Convert available_tools (list of dicts) to ToolDefinition objects
    try:
        tool_defs = [ToolDefinition.model_validate(tool) for tool in sample["available_tools"]]
        sample["tools"] = [tool.to_openai() for tool in tool_defs]
    except (ValidationError, TypeError, KeyError) as e:
        # If conversion fails, log and return the original sample
        logger.warning(
            "Failed to convert 'available_tools' to TRL format for a sample. Error: %s",
            e,
            exc_info=True,
        )
        return sample
    return sample
