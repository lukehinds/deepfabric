import importlib.metadata
import json
import logging
import os
import urllib.error
import urllib.request

from typing import TypedDict

from packaging.version import Version, parse

from .metrics import trace
from .tui import get_tui

logger = logging.getLogger(__name__)


class PyPIPackageInfo(TypedDict, total=False):
    """PyPI package info section."""

    version: str


class PyPIResponse(TypedDict, total=False):
    """PyPI JSON API response structure."""

    info: PyPIPackageInfo


# PyPI API endpoint for deepfabric package
PYPI_API_URL = "https://pypi.org/pypi/deepfabric/json"

# Timeout for PyPI API request (2 seconds)
REQUEST_TIMEOUT = 2.0


def _get_current_version() -> str | None:
    """
    Get the current installed version of deepfabric.

    Returns:
        str | None: Version string or None if unable to determine
    """
    try:
        return importlib.metadata.version("deepfabric")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        logger.debug("Unable to determine current version")
        return None


def _is_update_check_disabled() -> bool:
    """
    Check if update checking is disabled via environment variable.

    Returns:
        bool: True if DEEPFABRIC_NO_UPDATE_CHECK is set to any truthy value
    """
    env_value = os.environ.get("DEEPFABRIC_NO_UPDATE_CHECK", "").lower()
    return env_value in ("1", "true", "yes", "on")


def _fetch_latest_version_from_pypi() -> str | None:
    """
    Fetch the latest version from PyPI API.

    Returns:
        str | None: Latest version string or None if fetch fails
    """
    try:
        with urllib.request.urlopen(  # noqa: S310 # nosec
            PYPI_API_URL, timeout=REQUEST_TIMEOUT
        ) as response:
            data: PyPIResponse = json.loads(response.read().decode("utf-8"))
            latest_version = data.get("info", {}).get("version")
            if latest_version:
                logger.debug("Fetched latest version from PyPI: %s", latest_version)
                return latest_version
            logger.debug("No version found in PyPI response")
            return None
    except TimeoutError:
        logger.debug("PyPI request timed out after %s seconds", REQUEST_TIMEOUT)
        return None
    except urllib.error.URLError as e:
        logger.debug("Failed to fetch from PyPI: %s", e)
        return None
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        logger.debug("Failed to parse PyPI response: %s", e)
        return None


def _compare_versions(current: str, latest: str) -> bool:
    """
    Compare version strings to determine if an update is available.

    Args:
        current: Current version string
        latest: Latest version string

    Returns:
        bool: True if latest > current, False otherwise
    """
    try:
        current_version: Version = parse(current)
        latest_version: Version = parse(latest)
    except Exception as e:
        logger.debug("Failed to compare versions: %s", e)
        return False
    else:
        return latest_version > current_version


def check_for_updates() -> None:
    """
    Check for available updates and notify user if a newer version exists.

    This function:
    1. Checks if update checking is disabled via environment variable
    2. Gets the current installed version
    3. Fetches the latest version from PyPI
    4. Compares versions and displays a warning if update is available
    5. Tracks metrics about the update check

    The function is designed to fail silently and never block CLI execution.
    All errors are logged at DEBUG level and do not interrupt the user.
    """
    # Check if update checking is disabled
    if _is_update_check_disabled():
        logger.debug("Update check disabled via DEEPFABRIC_NO_UPDATE_CHECK")
        return

    # Get current version
    current_version = _get_current_version()
    if not current_version or current_version == "development":
        logger.debug("Skipping update check for development version")
        return

    # Fetch latest version from PyPI
    latest_version = _fetch_latest_version_from_pypi()
    if not latest_version:
        logger.debug("Could not fetch latest version from PyPI")
        return

    # Track metrics about the check
    try:
        trace(
            "update_check_performed",
            {
                "current_version": current_version,
                "latest_version": latest_version,
                "update_available": _compare_versions(current_version, latest_version),
            },
        )
    except Exception as e:
        logger.debug("Failed to track update check metrics: %s", e)

    # Compare versions and notify user if update is available
    if _compare_versions(current_version, latest_version):
        try:
            tui = get_tui()
            tui.warning(
                f"Update available: deepfabric {latest_version} "
                f"(you have {current_version})\n"
                f"   Run: pip install --upgrade deepfabric"
            )
        except Exception as e:
            logger.debug("Failed to display update notification: %s", e)
