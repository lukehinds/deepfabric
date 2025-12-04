"""DeepFabric training metrics logging.

This module provides integration with HuggingFace Trainer and TRL trainers
to log training metrics to the DeepFabric SaaS backend.

Features:
- Non-blocking async metrics sending
- Notebook-friendly API key prompts (like wandb)
- Graceful handling of failures without impacting training

Usage:
    from deepfabric.training import DeepFabricCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    trainer.add_callback(DeepFabricCallback(trainer))
    trainer.train()

Environment Variables:
    DEEPFABRIC_API_KEY: API key for authentication
    DEEPFABRIC_API_URL: SaaS backend URL (default: https://api.deepfabric.ai)
"""

from __future__ import annotations

from .callback import DeepFabricCallback
from .metrics_sender import MetricsSender

__all__ = [
    "DeepFabricCallback",
    "MetricsSender",
]
