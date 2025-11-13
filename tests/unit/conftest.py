"""
Pytest configuration for DeepFabric tests.

This file is automatically loaded by pytest and configures the test environment.
"""

import os

# Disable analytics/telemetry during tests to improve performance
# This must be set before importing any deepfabric modules
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Optionally set testing flag that components can check
os.environ["DEEPFABRIC_TESTING"] = "True"


def pytest_configure(config):  # noqa: ARG001
    """Configure pytest settings."""
    # Ensure telemetry is disabled
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
