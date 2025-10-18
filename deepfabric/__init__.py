from .cli import cli
from .config import DeepFabricConfig
from .dataset import Dataset
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
from .pipeline import DeepFabricPipeline
from .tree import Tree, TreeConfig

# Training module (optional dependencies)
try:
    from .training import DeepFabricSFTTrainer, SFTTrainingConfig
    from .training.sft_config import LoRAConfig, QuantizationConfig

    _has_training = True
except ImportError:
    _has_training = False
    DeepFabricSFTTrainer = None
    SFTTrainingConfig = None
    LoRAConfig = None
    QuantizationConfig = None

# Transformers provider (optional dependencies)
try:
    from .llm.transformers_provider import TransformersConfig, TransformersProvider

    _has_transformers = True
except ImportError:
    _has_transformers = False
    TransformersConfig = None
    TransformersProvider = None

__version__ = "0.1.0"

__all__ = [
    "Tree",
    "TreeConfig",
    "Graph",
    "GraphConfig",
    "DataSetGenerator",
    "DataSetGeneratorConfig",
    "Dataset",
    "DeepFabricConfig",
    "HFUploader",
    "DeepFabricPipeline",
    "cli",
    # Training (optional)
    "DeepFabricSFTTrainer",
    "SFTTrainingConfig",
    "LoRAConfig",
    "QuantizationConfig",
    # Transformers provider (optional)
    "TransformersConfig",
    "TransformersProvider",
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
