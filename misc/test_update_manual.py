#!/usr/bin/env python3
"""Manual test script for update checker."""

import json

from unittest.mock import Mock, patch


# Mock the PyPI response with a newer version
def test_with_newer_version():
    print("Testing update checker with simulated newer version...")

    mock_response = Mock()
    mock_response.read.return_value = json.dumps({
        "info": {"version": "99.99.99"}  # Simulate a much newer version
    }).encode("utf-8")
    mock_response.__enter__ = Mock(return_value=mock_response)
    mock_response.__exit__ = Mock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_response):
        from deepfabric.update_checker import check_for_updates  # noqa: PLC0415
        print("Running check_for_updates()...")
        check_for_updates()
        print("Check complete! You should see a warning above if TUI is working.")

if __name__ == "__main__":
    test_with_newer_version()
