"""DeepFabric automatic training metrics logging.

This module provides automatic integration with HuggingFace Trainer and TRL
trainers to log training metrics to the DeepFabric SaaS backend.

Features:
- Automatic callback injection via .pth file (installed with pip install)
- Non-blocking async metrics sending
- Notebook-friendly API key prompts (like wandb)
- Zero code changes required from users

Usage:
    # Option 1: Automatic (default - just install deepfabric)
    pip install deepfabric
    export DEEPFABRIC_API_KEY=your-key
    python train.py  # Metrics logged automatically!

    # Option 2: Explicit callback (for more control)
    from deepfabric.training import DeepFabricCallback

    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=[DeepFabricCallback(api_key="...")],
    )

    # Option 3: Manual injection
    from deepfabric.training import inject_callback
    inject_callback()  # Patches Trainer classes

Environment Variables:
    DEEPFABRIC_API_KEY: API key for authentication
    DEEPFABRIC_API_URL: SaaS backend URL (default: https://api.deepfabric.ai)
    DEEPFABRIC_DISABLE_AUTO_LOGGING: Set to "1" to disable auto-injection
"""

from __future__ import annotations

from .callback import DeepFabricCallback
from .injection import inject_callback
from .metrics_sender import MetricsSender

__all__ = [
    "DeepFabricCallback",
    "MetricsSender",
    "inject_callback",
]
