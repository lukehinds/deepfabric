"""Registry for managing evaluators."""

from .base import BaseEvaluator


class EvaluatorRegistry:
    """Registry for managing evaluators (similar to FormatterRegistry).

    Provides a central place to register and retrieve evaluators.
    Supports both built-in and custom evaluators.
    """

    def __init__(self):
        """Initialize registry with built-in evaluators."""
        self._evaluators: dict[str, type[BaseEvaluator]] = {}
        self._register_builtin_evaluators()

    def register(self, evaluator_class: type[BaseEvaluator]) -> None:
        """Register an evaluator class.

        Args:
            evaluator_class: Evaluator class to register
        """
        # Create temporary instance to get name
        temp_instance = evaluator_class()
        name = temp_instance.get_name()
        self._evaluators[name] = evaluator_class

    def get(self, name: str, config: dict | None = None) -> BaseEvaluator:
        """Get evaluator instance by name.

        Args:
            name: Evaluator name
            config: Optional configuration for the evaluator

        Returns:
            Evaluator instance

        Raises:
            KeyError: If evaluator not found
        """
        if name not in self._evaluators:
            available = ", ".join(self._evaluators.keys())
            msg = f"Evaluator '{name}' not found. Available: {available}"
            raise KeyError(msg)

        evaluator_class = self._evaluators[name]
        return evaluator_class(config=config)

    def list_evaluators(self) -> list[str]:
        """List all registered evaluator names.

        Returns:
            List of evaluator names
        """
        return list(self._evaluators.keys())

    def _register_builtin_evaluators(self) -> None:
        """Register built-in evaluators."""
        from .builtin.tool_calling import ToolCallingEvaluator  # noqa: PLC0415

        self.register(ToolCallingEvaluator)
        # More built-in evaluators can be registered here in the future
        # Future: self.register(AnswerQualityEvaluator)
        # Future: self.register(SafetyEvaluator)
        # Future: self.register(GuardrailsEvaluator)
