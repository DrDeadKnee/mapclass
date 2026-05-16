"""
Offline GCS-checkpoint test scaffold for Phase 4 (plan 04-01, filled in plan 04-02).

Guard imports from ``seg.gcs_checkpoint`` which ships in plan 04-02.
This file fails LOUDLY (pytest.fail, not pytest.skip) if seg.gcs_checkpoint is not
importable, honouring the Phase-4 Nyquist gate.

Body filled in plan 04-02; only the scaffold and loud-fail guard are set up here.
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard — LOUD fail (never skip) so the Nyquist gate fires offline
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from seg.gcs_checkpoint import (  # noqa: F401
        gcs_save_checkpoint,
        gcs_latest_checkpoint,
        validate_config_name,
    )
except ImportError as _e:
    _import_error = _e


def _require_gcs():
    if _import_error is not None:
        pytest.fail(
            f"seg.gcs_checkpoint not yet implemented: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# TestGCSCheckpointRoundTrip — mock GCS round-trip (filled in plan 04-02)
# ---------------------------------------------------------------------------

class TestGCSCheckpointRoundTrip:
    """Checkpoint write + resume round-trip using mocked gcsfs (plan 04-02)."""

    def test_save_and_resume_placeholder(self, tmp_path):
        """Placeholder: calls _require_gcs then skips (body filled in 04-02)."""
        _require_gcs()
        pytest.skip("filled in 04-02")
