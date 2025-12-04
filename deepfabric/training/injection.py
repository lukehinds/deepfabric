"""Callback injection for automatic Trainer integration."""

from __future__ import annotations

import functools
import logging

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-level state
_injection_done = False
_api_key: str | None = None
_api_key_resolved = False


def _get_api_key() -> str | None:
    """Get API key, prompting if needed (only once per session)."""
    global _api_key, _api_key_resolved  # noqa: PLW0603

    if _api_key_resolved:
        return _api_key

    from .api_key_prompt import get_api_key  # noqa: PLC0415

    _api_key = get_api_key()
    _api_key_resolved = True

    return _api_key


def _create_callback() -> Any | None:
    """Create DeepFabric callback instance if API key is available."""
    api_key = _get_api_key()

    if api_key is None:
        logger.debug("No API key available, DeepFabric logging disabled")
        return None

    from .callback import DeepFabricCallback  # noqa: PLC0415

    return DeepFabricCallback(api_key=api_key)


def _wrap_trainer_init(original_init: Callable[..., None]) -> Callable[..., None]:
    """Wrap Trainer.__init__ to inject DeepFabric callback.

    Args:
        original_init: Original __init__ method

    Returns:
        Wrapped __init__ method
    """

    @functools.wraps(original_init)
    def wrapped_init(self: Any, *args: Any, **kwargs: Any) -> None:
        # Call original __init__ first
        original_init(self, *args, **kwargs)

        # Try to inject callback
        try:
            _inject_callback_into_trainer(self)
        except Exception as e:
            logger.debug(f"Failed to inject DeepFabric callback: {e}")

    return wrapped_init


def _inject_callback_into_trainer(trainer: Any) -> None:
    """Inject DeepFabric callback into a trainer instance.

    Args:
        trainer: Trainer instance
    """
    # Check if callback already present (avoid duplicates)
    if hasattr(trainer, "callback_handler"):
        for cb in trainer.callback_handler.callbacks:
            if type(cb).__name__ == "DeepFabricCallback":
                logger.debug("DeepFabric callback already present")
                return

    # Create callback
    callback = _create_callback()
    if callback is None:
        return

    # Add callback to trainer
    if hasattr(trainer, "add_callback"):
        trainer.add_callback(callback)
        logger.debug("DeepFabric callback injected into Trainer")
    else:
        logger.debug("Trainer does not have add_callback method")


def inject_callback() -> None:
    """Inject DeepFabric callback into all Trainer classes.

    This function monkey-patches transformers.Trainer and TRL trainer classes
    (SFTTrainer, DPOTrainer, etc.) to automatically include the DeepFabric
    callback.

    This should be called once, typically via sitecustomize.py or manually
    before any training starts.

    Example:
        from deepfabric.training import inject_callback
        inject_callback()

        # Now any Trainer instance will automatically have DeepFabric logging
        trainer = Trainer(model=model, args=args)
        trainer.train()
    """
    global _injection_done  # noqa: PLW0603

    if _injection_done:
        logger.debug("Callback injection already done")
        return

    trainers_patched = []

    # Patch transformers.Trainer
    try:
        from transformers import Trainer  # noqa: PLC0415

        if not getattr(Trainer.__init__, "_deepfabric_wrapped", False):
            Trainer.__init__ = _wrap_trainer_init(Trainer.__init__)
            Trainer.__init__._deepfabric_wrapped = True  # type: ignore[attr-defined]
            trainers_patched.append("transformers.Trainer")
    except ImportError:
        logger.debug("transformers not available")

    # Patch TRL trainers
    trl_trainers = [
        ("trl", "SFTTrainer"),
        ("trl", "DPOTrainer"),
        ("trl", "PPOTrainer"),
        ("trl", "RewardTrainer"),
        ("trl", "ORPOTrainer"),
        ("trl", "KTOTrainer"),
    ]

    for module_name, class_name in trl_trainers:
        try:
            module = __import__(module_name, fromlist=[class_name])
            trainer_class = getattr(module, class_name, None)

            if trainer_class is None:
                continue

            if not getattr(trainer_class.__init__, "_deepfabric_wrapped", False):
                trainer_class.__init__ = _wrap_trainer_init(trainer_class.__init__)
                trainer_class.__init__._deepfabric_wrapped = True  # type: ignore[attr-defined]
                trainers_patched.append(f"{module_name}.{class_name}")

        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"Failed to patch {module_name}.{class_name}: {e}")

    _injection_done = True

    if trainers_patched:
        logger.debug(f"DeepFabric callback injection complete: {trainers_patched}")
    else:
        logger.debug("No trainers found to patch")


def is_injection_enabled() -> bool:
    """Check if callback injection has been performed.

    Returns:
        True if inject_callback() has been called
    """
    return _injection_done


def reset_injection() -> None:
    """Reset injection state (for testing only)."""
    global _injection_done, _api_key, _api_key_resolved  # noqa: PLW0603
    _injection_done = False
    _api_key = None
    _api_key_resolved = False
