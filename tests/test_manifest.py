"""
Offline unit tests for gcs_io.validate_manifest (Plan 02-01 Task 1).

All tests use pure in-memory data — no network, no GCS, no filesystem I/O.

Test coverage:
  - test_validate_manifest_hard_fails: sys.exit(1) on each of the four
    error modes (regex mismatch, unlisted file, phantom manifest entry,
    template mismatch).
  - test_validate_manifest_returns_id_to_template: clean inputs return the
    expected {sanitized_id: template} mapping.
"""

from __future__ import annotations

import importlib
import sys
import unittest.mock as mock

import pytest


# ---------------------------------------------------------------------------
# Import guard — LOUD fail (never skip) so the Nyquist gate fires offline
# ---------------------------------------------------------------------------

_import_error: Exception | None = None
try:
    import gcs_io  # type: ignore[import]
    from gcs_io import validate_manifest
except ImportError as _e:
    _import_error = _e


def _require_gcs_io() -> None:
    if _import_error is not None:
        pytest.fail(
            f"gcs_io not yet implemented: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Fixtures — clean manifest
# ---------------------------------------------------------------------------

_CLEAN_FILENAMES = ["europe_01.geojson", "europe_02.geojson", "americas_01.geojson"]

_CLEAN_MANIFEST = {
    "version": "1",
    "entries": {
        "europe_01.geojson": {"template": "europe"},
        "europe_02.geojson": {"template": "europe"},
        "americas_01.geojson": {"template": "americas"},
    },
}


# ---------------------------------------------------------------------------
# test_validate_manifest_hard_fails
# ---------------------------------------------------------------------------

class TestValidateManifestHardFails:
    """validate_manifest must sys.exit(1) on each of the four error modes."""

    def test_regex_mismatch_hard_fails(self):
        """A filename that fails the naming regex triggers sys.exit(1)."""
        _require_gcs_io()
        # "Europe_01.geojson" fails the regex (uppercase E)
        bad_filenames = ["Europe_01.geojson", "europe_02.geojson", "americas_01.geojson"]
        bad_manifest = {
            "version": "1",
            "entries": {
                "Europe_01.geojson": {"template": "europe"},
                "europe_02.geojson": {"template": "europe"},
                "americas_01.geojson": {"template": "americas"},
            },
        }
        with pytest.raises(SystemExit) as exc_info:
            validate_manifest(bad_filenames, bad_manifest)
        assert exc_info.value.code == 1

    def test_unlisted_file_hard_fails(self):
        """A filename not in manifest['entries'] triggers sys.exit(1)."""
        _require_gcs_io()
        filenames_with_extra = _CLEAN_FILENAMES + ["east_asia_01.geojson"]
        with pytest.raises(SystemExit) as exc_info:
            validate_manifest(filenames_with_extra, _CLEAN_MANIFEST)
        assert exc_info.value.code == 1

    def test_phantom_manifest_entry_hard_fails(self):
        """A manifest entry not present in gcs_filenames triggers sys.exit(1)."""
        _require_gcs_io()
        # Provide only 2 of the 3 filenames listed in the manifest
        filenames_missing_one = ["europe_01.geojson", "europe_02.geojson"]
        with pytest.raises(SystemExit) as exc_info:
            validate_manifest(filenames_missing_one, _CLEAN_MANIFEST)
        assert exc_info.value.code == 1

    def test_template_mismatch_hard_fails(self):
        """An entry whose template != fn.rsplit('_',1)[0] triggers sys.exit(1)."""
        _require_gcs_io()
        bad_template_manifest = {
            "version": "1",
            "entries": {
                "europe_01.geojson": {"template": "americas"},   # wrong template
                "europe_02.geojson": {"template": "europe"},
                "americas_01.geojson": {"template": "americas"},
            },
        }
        with pytest.raises(SystemExit) as exc_info:
            validate_manifest(_CLEAN_FILENAMES, bad_template_manifest)
        assert exc_info.value.code == 1

    def test_hard_fail_exits_not_raises_value_error(self):
        """validate_manifest must call sys.exit, not raise ValueError."""
        _require_gcs_io()
        bad_filenames = ["INVALID.geojson"]
        bad_manifest = {
            "version": "1",
            "entries": {"INVALID.geojson": {"template": "invalid"}},
        }
        # Should raise SystemExit, not ValueError
        with pytest.raises(SystemExit):
            validate_manifest(bad_filenames, bad_manifest)


# ---------------------------------------------------------------------------
# test_validate_manifest_returns_id_to_template
# ---------------------------------------------------------------------------

class TestValidateManifestCleanPass:
    """On clean inputs, validate_manifest returns {sanitized_id: template}."""

    def test_returns_id_to_template_mapping(self):
        """Clean inputs return a dict mapping sanitized stems to templates."""
        _require_gcs_io()
        result = validate_manifest(_CLEAN_FILENAMES, _CLEAN_MANIFEST)
        assert isinstance(result, dict)
        # europe_01.geojson -> stem = "europe_01" -> sanitized = "europe_01" -> template "europe"
        assert result.get("europe_01") == "europe"
        assert result.get("europe_02") == "europe"
        assert result.get("americas_01") == "americas"

    def test_returns_all_filenames_as_keys(self):
        """Every input filename maps to an entry in the result dict."""
        _require_gcs_io()
        result = validate_manifest(_CLEAN_FILENAMES, _CLEAN_MANIFEST)
        assert len(result) == len(_CLEAN_FILENAMES)

    def test_east_asia_sanitization(self):
        """Filenames with underscores in stem are sanitized correctly."""
        _require_gcs_io()
        filenames = ["east_asia_01.geojson"]
        manifest = {
            "version": "1",
            "entries": {"east_asia_01.geojson": {"template": "east_asia"}},
        }
        result = validate_manifest(filenames, manifest)
        assert "east_asia_01" in result
        assert result["east_asia_01"] == "east_asia"
