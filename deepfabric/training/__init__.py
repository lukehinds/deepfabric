"""Training module for DeepFabric.

This module provides integration with HuggingFace TRL's SFTTrainer for
supervised fine-tuning of language models using DeepFabric-generated datasets.
"""

from .sft_config import LoRAConfig, QuantizationConfig, SFTTrainingConfig
from .sft_trainer import DeepFabricSFTTrainer

__all__ = [
    "SFTTrainingConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "DeepFabricSFTTrainer",
]
