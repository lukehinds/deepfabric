"""
DeepFabric Formatter System

This module provides a pluggable post-processing system for dataset formatting.
Formatters can be built-in (provided by DeepFabric) or custom (user-defined).
"""

from .base import BaseFormatter
from .registry import FormatterRegistry

__all__ = ["BaseFormatter", "FormatterRegistry"]
