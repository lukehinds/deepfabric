"""Auto-injection module for DeepFabric training metrics.

This module sets up automatic injection of the DeepFabric callback into
HuggingFace Trainer instances. It can be loaded either:
1. Via .pth file on Python startup (when installed from wheel)
2. When deepfabric package is imported (for development/editable installs)

The injection patches Trainer.__init__ so that all future Trainer instances
automatically get the DeepFabric callback.
"""

from __future__ import annotations

import os
import sys

_setup_done = False


def _setup_auto_inject():
    """Setup injection of DeepFabric callback into Trainer classes."""
    global _setup_done  # noqa: PLW0603

    if _setup_done:
        return

    _setup_done = True

    # Skip if explicitly disabled
    if os.getenv("DEEPFABRIC_DISABLE_AUTO_LOGGING") == "1":
        return

    # Skip if running tests
    if os.getenv("DEEPFABRIC_TESTING") == "True":
        return

    # If transformers is already imported, inject immediately
    if "transformers.trainer" in sys.modules or "transformers" in sys.modules:
        _do_inject()
        return

    # Otherwise, set up import hook to inject when transformers is imported
    class _TransformersImportHook:
        """Import hook that triggers injection when transformers is imported."""

        def find_module(self, fullname, path=None):  # noqa: ARG002
            """Called for each import."""
            # Trigger on transformers or transformers.trainer import
            if fullname in ("transformers", "transformers.trainer"):
                # Remove ourselves first to avoid recursion
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
                # Schedule injection after import completes
                _schedule_post_import_inject()
            return None

    sys.meta_path.insert(0, _TransformersImportHook())


def _schedule_post_import_inject():
    """Schedule injection to run after current import completes."""
    import atexit  # noqa: PLC0415

    # Use atexit-like mechanism but for import completion
    # We'll use a second import hook that fires on the next import
    class _PostImportHook:
        """Fires injection on the next import after transformers."""

        _done = False

        def find_module(self, fullname, path=None):  # noqa: ARG002
            if self._done:
                return None
            self._done = True
            # Remove ourselves
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            # Now inject - transformers should be fully loaded
            _do_inject()
            return None

    sys.meta_path.append(_PostImportHook())


def _do_inject():
    """Perform the actual callback injection."""
    try:
        from deepfabric.training.injection import inject_callback  # noqa: PLC0415

        inject_callback()
    except ImportError:
        pass  # deepfabric.training not fully available
    except Exception:
        pass  # Silently fail on any error


# Run setup on module import
_setup_auto_inject()
