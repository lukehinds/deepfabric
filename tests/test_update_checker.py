"""Tests for the update checker module."""

import importlib.metadata
import json
import urllib.error

from unittest.mock import MagicMock, Mock, patch

import pytest

from deepfabric.update_checker import (
    _compare_versions,
    _fetch_latest_version_from_pypi,
    _get_current_version,
    _is_update_check_disabled,
    check_for_updates,
)


class TestGetCurrentVersion:
    """Tests for _get_current_version function."""

    def test_get_current_version_success(self):
        """Test successful version retrieval."""
        with patch("importlib.metadata.version", return_value="2.12.0"):
            version = _get_current_version()
            assert version == "2.12.0"

    def test_get_current_version_not_found(self):
        """Test version retrieval when package not found."""
        with patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError,
        ):
            version = _get_current_version()
            assert version is None

    def test_get_current_version_import_error(self):
        """Test version retrieval when importlib.metadata is not available."""
        with patch("importlib.metadata.version", side_effect=ImportError):
            version = _get_current_version()
            assert version is None


class TestIsUpdateCheckDisabled:
    """Tests for _is_update_check_disabled function."""

    def test_update_check_enabled_by_default(self, monkeypatch):
        """Test that update check is enabled by default."""
        monkeypatch.delenv("DEEPFABRIC_NO_UPDATE_CHECK", raising=False)
        assert _is_update_check_disabled() is False

    @pytest.mark.parametrize("env_value", ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"])
    def test_update_check_disabled_truthy_values(self, monkeypatch, env_value):
        """Test that update check is disabled for truthy environment values."""
        monkeypatch.setenv("DEEPFABRIC_NO_UPDATE_CHECK", env_value)
        assert _is_update_check_disabled() is True

    @pytest.mark.parametrize("env_value", ["0", "false", "False", "no", "off", ""])
    def test_update_check_enabled_falsy_values(self, monkeypatch, env_value):
        """Test that update check is enabled for falsy environment values."""
        monkeypatch.setenv("DEEPFABRIC_NO_UPDATE_CHECK", env_value)
        assert _is_update_check_disabled() is False


class TestFetchLatestVersionFromPyPI:
    """Tests for _fetch_latest_version_from_pypi function."""

    def test_fetch_version_success(self):
        """Test successful version fetch from PyPI."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"info": {"version": "2.13.0"}}).encode(
            "utf-8"
        )
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            version = _fetch_latest_version_from_pypi()
            assert version == "2.13.0"

    def test_fetch_version_timeout(self):
        """Test version fetch with timeout."""
        with patch("urllib.request.urlopen", side_effect=TimeoutError):
            version = _fetch_latest_version_from_pypi()
            assert version is None

    def test_fetch_version_url_error(self):
        """Test version fetch with URL error."""
        with patch(
            "urllib.request.urlopen", side_effect=urllib.error.URLError("Connection failed")
        ):
            version = _fetch_latest_version_from_pypi()
            assert version is None

    def test_fetch_version_invalid_json(self):
        """Test version fetch with invalid JSON response."""
        mock_response = Mock()
        mock_response.read.return_value = b"invalid json"
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            version = _fetch_latest_version_from_pypi()
            assert version is None

    def test_fetch_version_missing_info(self):
        """Test version fetch with missing info field."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"data": {}}).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            version = _fetch_latest_version_from_pypi()
            assert version is None

    def test_fetch_version_missing_version_field(self):
        """Test version fetch with missing version field."""
        mock_response = Mock()
        mock_response.read.return_value = json.dumps({"info": {}}).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            version = _fetch_latest_version_from_pypi()
            assert version is None


class TestCompareVersions:
    """Tests for _compare_versions function."""

    @pytest.mark.parametrize(
        "current,latest,expected",
        [
            ("2.12.0", "2.13.0", True),
            ("2.12.0", "3.0.0", True),
            ("2.12.0", "2.12.1", True),
            ("2.12.0", "2.12.0", False),
            ("2.13.0", "2.12.0", False),
            ("3.0.0", "2.12.0", False),
            ("2.12.1", "2.12.0", False),
        ],
    )
    def test_version_comparison(self, current, latest, expected):
        """Test version comparison with various version strings."""
        assert _compare_versions(current, latest) == expected

    def test_version_comparison_with_prerelease(self):
        """Test version comparison with pre-release versions."""
        assert _compare_versions("2.12.0", "2.13.0rc1") is True
        assert _compare_versions("2.13.0rc1", "2.13.0") is True

    def test_version_comparison_invalid_version(self):
        """Test version comparison with invalid version strings."""
        # Should return False on error
        assert _compare_versions("invalid", "2.13.0") is False
        assert _compare_versions("2.12.0", "invalid") is False


class TestCheckForUpdates:
    """Tests for check_for_updates function."""

    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_check_disabled(self, mock_disabled):
        """Test that check is skipped when disabled."""
        mock_disabled.return_value = True
        # Should not raise any exceptions
        check_for_updates()
        mock_disabled.assert_called_once()

    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_check_skipped_for_development_version(self, mock_disabled, mock_version):
        """Test that check is skipped for development versions."""
        mock_disabled.return_value = False
        mock_version.return_value = "development"
        # Should not raise any exceptions
        check_for_updates()

    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_check_skipped_when_version_unknown(self, mock_disabled, mock_version):
        """Test that check is skipped when current version is unknown."""
        mock_disabled.return_value = False
        mock_version.return_value = None
        # Should not raise any exceptions
        check_for_updates()

    @patch("deepfabric.update_checker.get_tui")
    @patch("deepfabric.update_checker.trace")
    @patch("deepfabric.update_checker._compare_versions")
    @patch("deepfabric.update_checker._fetch_latest_version_from_pypi")
    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_update_available_shows_warning(
        self,
        mock_disabled,
        mock_current,
        mock_fetch,
        mock_compare,
        mock_trace,
        mock_tui,
    ):
        """Test that warning is shown when update is available."""
        mock_disabled.return_value = False
        mock_current.return_value = "2.12.0"
        mock_fetch.return_value = "2.13.0"
        mock_compare.return_value = True

        mock_tui_instance = MagicMock()
        mock_tui.return_value = mock_tui_instance

        check_for_updates()

        # Verify warning was shown
        mock_tui_instance.warning.assert_called_once()
        warning_msg = mock_tui_instance.warning.call_args[0][0]
        assert "2.13.0" in warning_msg
        assert "2.12.0" in warning_msg
        assert "pip install --upgrade deepfabric" in warning_msg

        # Verify metrics were tracked
        mock_trace.assert_called_once_with(
            "update_check_performed",
            {
                "current_version": "2.12.0",
                "latest_version": "2.13.0",
                "update_available": True,
            },
        )

    @patch("deepfabric.update_checker.get_tui")
    @patch("deepfabric.update_checker.trace")
    @patch("deepfabric.update_checker._compare_versions")
    @patch("deepfabric.update_checker._fetch_latest_version_from_pypi")
    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_no_update_available_no_warning(
        self,
        mock_disabled,
        mock_current,
        mock_fetch,
        mock_compare,
        mock_trace,
        mock_tui,
    ):
        """Test that no warning is shown when already on latest version."""
        mock_disabled.return_value = False
        mock_current.return_value = "2.13.0"
        mock_fetch.return_value = "2.13.0"
        mock_compare.return_value = False

        mock_tui_instance = MagicMock()
        mock_tui.return_value = mock_tui_instance

        check_for_updates()

        # Verify no warning was shown
        mock_tui_instance.warning.assert_not_called()

        # Verify metrics were still tracked
        mock_trace.assert_called_once_with(
            "update_check_performed",
            {
                "current_version": "2.13.0",
                "latest_version": "2.13.0",
                "update_available": False,
            },
        )

    @patch("deepfabric.update_checker._fetch_latest_version_from_pypi")
    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_check_handles_fetch_failure(self, mock_disabled, mock_current, mock_fetch):
        """Test that check handles PyPI fetch failure gracefully."""
        mock_disabled.return_value = False
        mock_current.return_value = "2.12.0"
        mock_fetch.return_value = None  # Simulates fetch failure

        # Should not raise any exceptions
        check_for_updates()

    @patch("deepfabric.update_checker.get_tui")
    @patch("deepfabric.update_checker._compare_versions")
    @patch("deepfabric.update_checker._fetch_latest_version_from_pypi")
    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_check_handles_tui_error(
        self, mock_disabled, mock_current, mock_fetch, mock_compare, mock_tui
    ):
        """Test that check handles TUI error gracefully."""
        mock_disabled.return_value = False
        mock_current.return_value = "2.12.0"
        mock_fetch.return_value = "2.13.0"
        mock_compare.return_value = True
        mock_tui.side_effect = Exception("TUI error")

        # Should not raise any exceptions
        check_for_updates()

    @patch("deepfabric.update_checker.trace")
    @patch("deepfabric.update_checker._compare_versions")
    @patch("deepfabric.update_checker._fetch_latest_version_from_pypi")
    @patch("deepfabric.update_checker._get_current_version")
    @patch("deepfabric.update_checker._is_update_check_disabled")
    def test_check_handles_metrics_error(
        self, mock_disabled, mock_current, mock_fetch, mock_compare, mock_trace
    ):
        """Test that check handles metrics error gracefully."""
        mock_disabled.return_value = False
        mock_current.return_value = "2.12.0"
        mock_fetch.return_value = "2.13.0"
        mock_compare.return_value = True
        mock_trace.side_effect = Exception("Metrics error")

        # Should not raise any exceptions
        check_for_updates()
