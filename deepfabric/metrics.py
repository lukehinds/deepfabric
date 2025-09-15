"""
Simple analytics for DeepFabric using PostHog.

Provides a single trace() function for anonymous usage analytics.
All analytics can be disabled by setting ANONYMIZED_TELEMETRY=False.
"""

import contextlib
import os

try:
    import importlib.metadata

    VERSION = importlib.metadata.version("deepfabric")
except (ImportError, importlib.metadata.PackageNotFoundError):
    VERSION = "development"

try:
    from posthog import Posthog

    POSTHOG_AVAILABLE = True
except ImportError:
    POSTHOG_AVAILABLE = False

# Initialize PostHog client
if POSTHOG_AVAILABLE:
    posthog = Posthog(
        project_api_key="phc_Kn8hKQIXHm5OHp5OTxvMvFDUmT7HyOUNlJvWkduB9qO",
        host="https://us.i.posthog.com",
    )
else:
    posthog = None


def trace(event_name, event_properties=None):
    """
    Send an analytics event if telemetry is enabled.

    Args:
        event_name: Name of the event to track
        event_properties: Optional dictionary of event properties
    """
    # Quick exit if telemetry is disabled
    if os.environ.get("ANONYMIZED_TELEMETRY") == "False":
        return

    # Quick exit if during testing
    if os.environ.get("DEEPFABRIC_TESTING") == "True":
        return

    # Quick exit if PostHog not available
    if not POSTHOG_AVAILABLE or not posthog:
        return

    with contextlib.suppress(Exception):
        # Add version to all events
        properties = event_properties or {}
        properties["version"] = VERSION

        posthog.capture(distinct_id="deepfabric", event=event_name, properties=properties)


def is_enabled():
    """Check if analytics is currently enabled."""
    return (
        os.environ.get("ANONYMIZED_TELEMETRY") != "False"
        and os.environ.get("DEEPFABRIC_TESTING") != "True"
        and POSTHOG_AVAILABLE
    )
