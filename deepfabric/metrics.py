"""
Simple analytics for DeepFabric using PostHog.

Provides a single trace() function for anonymous usage analytics.
All analytics can be disabled by setting ANONYMIZED_TELEMETRY=False.

Privacy-respecting identity:
- Generates a stable, anonymous user ID based on machine characteristics
- Uses DEEPFABRIC_DEVELOPER=True to mark developer sessions for filtering
- Never collects PII (names, emails, IP addresses, etc.)
"""

import contextlib
import hashlib
import os
import platform
import uuid

try:
    import importlib.metadata

    VERSION = importlib.metadata.version("deepfabric")
except (ImportError, importlib.metadata.PackageNotFoundError):
    VERSION = "development"

try:
    from posthog import Posthog, identify_context, new_context

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

# Cache for the generated user ID using a dict to avoid global statement
_user_id_cache: dict[str, str | None] = {"id": None}


def _get_user_id() -> str:
    """
    Generate a stable, anonymous user ID based on machine characteristics.

    Creates a UUID from a hash of platform.node() (hostname) and uuid.getnode()
    (MAC address). This provides a persistent identifier across sessions without
    collecting PII.

    Returns:
        str: A stable UUID string unique to this machine
    """
    if _user_id_cache["id"] is not None:
        return _user_id_cache["id"]

    # Combine hostname and MAC address for machine-specific identifier
    machine_info = f"{platform.node()}-{uuid.getnode()}"

    # Create SHA256 hash and convert to UUID format
    hash_digest = hashlib.sha256(machine_info.encode()).hexdigest()

    # Use first 32 hex chars to create a valid UUID
    user_uuid = str(uuid.UUID(hash_digest[:32]))

    _user_id_cache["id"] = user_uuid
    return user_uuid


def _is_developer() -> bool:
    """
    Check if this session is marked as a developer session.

    Returns:
        bool: True if DEEPFABRIC_DEVELOPER environment variable is set to 'True'
    """
    return os.environ.get("DEEPFABRIC_DEVELOPER") == "True"


def trace(event_name, event_properties=None):
    """
    Send an analytics event if telemetry is enabled.

    Uses privacy-respecting identity tracking with a stable, anonymous user ID
    generated from machine characteristics. Developer sessions are marked with
    the is_developer flag when DEEPFABRIC_DEVELOPER=True.

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
        # Generate stable user ID
        user_id = _get_user_id()

        # Add version and developer flag to all events
        properties = event_properties or {}
        properties["version"] = VERSION
        properties["is_developer"] = _is_developer()

        # Use identity context to associate events with the user
        with new_context():
            identify_context(user_id)
            posthog.capture(event=event_name, properties=properties)


def is_enabled():
    """Check if analytics is currently enabled."""
    return (
        os.environ.get("ANONYMIZED_TELEMETRY") != "False"
        and os.environ.get("DEEPFABRIC_TESTING") != "True"
        and POSTHOG_AVAILABLE
    )
