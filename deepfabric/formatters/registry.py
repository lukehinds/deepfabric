import importlib
import importlib.util
import os

from pathlib import Path
from typing import Any

from .base import BaseFormatter, FormatterError


class FormatterRegistry:
    """
    Registry for managing formatter loading and instantiation.

    Supports two loading mechanisms:
    - builtin://formatter_name.py - loads from deepfabric.formatters.builtin
    - file://path/to/formatter.py - loads from user-specified file path
    """

    def __init__(self):
        self._cache: dict[str, type[BaseFormatter]] = {}

    def load_formatter(
        self, template: str, config: "dict[str, Any] | None" = None, tool_registry=None
    ) -> BaseFormatter:
        """
        Load and instantiate a formatter from a template string.

        Args:
            template: Template string like "builtin://grpo.py" or "file://./my_formatter.py"
            config: Configuration dictionary to pass to the formatter
            tool_registry: Optional tool registry for agent tool-calling formatters

        Returns:
            Instantiated formatter instance

        Raises:
            FormatterError: If the formatter cannot be loaded or instantiated
        """
        if template in self._cache:
            formatter_class = self._cache[template]
        else:
            formatter_class = self._load_formatter_class(template)
            self._cache[template] = formatter_class

        try:
            return formatter_class(config, tool_registry=tool_registry)
        except Exception as e:
            raise FormatterError(f"Failed to instantiate formatter {template}: {str(e)}") from e

    def _load_formatter_class(self, template: str) -> type[BaseFormatter]:
        """
        Load a formatter class from the template string.

        Args:
            template: Template string specifying the formatter location

        Returns:
            The formatter class (not instantiated)

        Raises:
            FormatterError: If the formatter cannot be loaded
        """
        if template.startswith("builtin://"):
            return self._load_builtin_formatter(template)
        if template.startswith("file://"):
            return self._load_file_formatter(template)
        raise FormatterError(f"Invalid template format: {template}. Use 'builtin://' or 'file://'")

    def _load_builtin_formatter(self, template: str) -> type[BaseFormatter]:
        """
        Load a built-in formatter from deepfabric.formatters.builtin.

        Args:
            template: Template like "builtin://grpo.py"

        Returns:
            The formatter class

        Raises:
            FormatterError: If the built-in formatter cannot be found
        """
        # Extract formatter name from template (remove .py extension)
        formatter_name = template[len("builtin://") :]
        if formatter_name.endswith(".py"):
            formatter_name = formatter_name[:-3]

        try:
            module = importlib.import_module(f"deepfabric.formatters.builtin.{formatter_name}")
            return self._extract_formatter_class(module, formatter_name)
        except ImportError as e:
            raise FormatterError(
                f"Built-in formatter '{formatter_name}' not found: {str(e)}"
            ) from e

    def _load_file_formatter(self, template: str) -> type[BaseFormatter]:
        """
        Load a custom formatter from a file path.

        Args:
            template: Template like "file://./my_formatter.py"

        Returns:
            The formatter class

        Raises:
            FormatterError: If the file cannot be loaded or doesn't contain a valid formatter
        """
        # Extract file path from template
        file_path = template[len("file://") :]

        # Resolve relative paths
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        if not os.path.exists(file_path):
            raise FormatterError(f"Formatter file not found: {file_path}")

        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location("custom_formatter", file_path)
            if spec is None or spec.loader is None:
                raise FormatterError(f"Cannot load module from {file_path}")  # noqa: TRY301

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Extract formatter name for class detection
            formatter_name = Path(file_path).stem
            return self._extract_formatter_class(module, formatter_name)

        except Exception as e:
            raise FormatterError(f"Failed to load formatter from {file_path}: {str(e)}") from e

    def _extract_formatter_class(self, module, formatter_name: str) -> type[BaseFormatter]:
        """
        Extract a BaseFormatter subclass from a module.

        Args:
            module: The loaded module
            formatter_name: Name hint for finding the class

        Returns:
            The formatter class

        Raises:
            FormatterError: If no valid formatter class is found
        """
        # Look for common naming patterns
        potential_names = [
            f"{formatter_name.title()}Formatter",  # e.g., GrpoFormatter
            f"{formatter_name.upper()}Formatter",  # e.g., GRPOFormatter
            "Formatter",  # Generic name
        ]

        # First, try the common naming patterns
        for name in potential_names:
            if hasattr(module, name):
                cls = getattr(module, name)
                if isinstance(cls, type) and issubclass(cls, BaseFormatter):
                    return cls

        # If no common names found, search for any BaseFormatter subclass
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, BaseFormatter)
                and attr is not BaseFormatter
            ):
                return attr

        raise FormatterError(f"No BaseFormatter subclass found in module {formatter_name}")

    def clear_cache(self):
        """Clear the formatter cache."""
        self._cache.clear()

    def list_builtin_formatters(self) -> list[str]:
        """
        List all available built-in formatters.

        Returns:
            List of built-in formatter names
        """
        try:
            builtin_path = Path(__file__).parent / "builtin"
            if not builtin_path.exists():
                return []

            formatters = []
            for file_path in builtin_path.glob("*.py"):
                if file_path.name != "__init__.py":
                    formatters.append(file_path.stem)
        except Exception:
            return []
        else:
            return formatters
