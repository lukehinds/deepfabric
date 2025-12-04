from .cli import cli
from .config import DeepFabricConfig
from .exceptions import (
    APIError,
    ConfigurationError,
    DatasetError,
    DataSetGeneratorError,
    DeepFabricError,
    HubUploadError,
    JSONParsingError,
    ModelError,
    RetryExhaustedError,
    TreeError,
    ValidationError,
)
from .generator import DataSetGenerator, DataSetGeneratorConfig
from .graph import Graph, GraphConfig
from .hf_hub import HFUploader
from .training import (
    DeepFabricCallback,
    MetricsSender,
    inject_callback,
)
from .tree import Tree, TreeConfig

__version__ = "0.1.0"


# Trigger auto-injection when deepfabric is imported
# This handles development mode where .pth files aren't executed
def _trigger_auto_inject():
    """Setup auto-injection for development mode."""
    import os  # noqa: PLC0415

    if os.getenv("DEEPFABRIC_DISABLE_AUTO_LOGGING") == "1":
        return
    if os.getenv("DEEPFABRIC_TESTING") == "True":
        return

    import contextlib  # noqa: PLC0415

    with contextlib.suppress(ImportError):
        from .training import _auto_inject  # noqa: PLC0415, F401


_trigger_auto_inject()

__all__ = [
    "Tree",
    "TreeConfig",
    "Graph",
    "GraphConfig",
    "DataSetGenerator",
    "DataSetGeneratorConfig",
    "DeepFabricConfig",
    "HFUploader",
    "cli",
    # Training metrics logging
    "DeepFabricCallback",
    "MetricsSender",
    "inject_callback",
    # Exceptions
    "DeepFabricError",
    "ConfigurationError",
    "ValidationError",
    "ModelError",
    "TreeError",
    "DataSetGeneratorError",
    "DatasetError",
    "HubUploadError",
    "JSONParsingError",
    "APIError",
    "RetryExhaustedError",
]
