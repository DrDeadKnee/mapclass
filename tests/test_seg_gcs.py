"""
Offline GCS-checkpoint test suite for Phase 4 (plan 04-02).

All GCS I/O is replaced by ``_MockGCSFileSystem`` — no network, no gcsfs
installation required.  Tests are fully deterministic and finish in < 30s.

Classes:
    TestGCSCheckpointRoundTrip — write + resume round-trips
    TestConfigNaming           — D-07 config-naming contract + traversal rejection
    TestCorruptCheckpointSkip  — T-04-05 corrupt-blob DoS guard

Guard: ``_require_gcs()`` fires ``pytest.fail`` (never ``pytest.skip``) if
``seg.gcs_checkpoint`` is not importable — honouring the Phase-4 Nyquist gate.
"""

from __future__ import annotations

import io
import unittest.mock as mock

import pytest
import torch

# ---------------------------------------------------------------------------
# Import guard — LOUD fail (never skip) so the Nyquist gate fires offline
# ---------------------------------------------------------------------------

_import_error: Exception | None = None
try:
    from seg.gcs_checkpoint import (  # noqa: F401
        config_prefix,
        gcs_latest_checkpoint,
        gcs_save_checkpoint,
        validate_config_name,
    )
except ImportError as _e:
    _import_error = _e


def _require_gcs() -> None:
    if _import_error is not None:
        pytest.fail(
            f"seg.gcs_checkpoint not yet implemented: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# _MockGCSFileSystem — in-memory stand-in for gcsfs.GCSFileSystem
# ---------------------------------------------------------------------------


class _MockContextManager:
    """BytesIO-backed context manager returned by _MockGCSFileSystem.open()."""

    def __init__(self, store: dict[str, bytes], path: str, mode: str) -> None:
        # Strip gs:// prefix for the internal store key
        self._key = path.lstrip("gs://")
        self._mode = mode
        self._store = store
        self._buf: io.BytesIO

    def __enter__(self) -> "_MockContextManager":
        if "r" in self._mode:
            data = self._store.get(self._key, b"")
            self._buf = io.BytesIO(data)
        else:
            self._buf = io.BytesIO()
        return self._buf

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if "w" in self._mode and exc_type is None:
            self._store[self._key] = self._buf.getvalue()


class _MockGCSFileSystem:
    """Offline in-memory stand-in for ``gcsfs.GCSFileSystem``.

    Stores blobs as ``dict[str, bytes]`` keyed by their bare path (no gs://).
    All instances within a test share a single class-level store so that
    separate save + load calls see each other's data.
    """

    # Shared in-memory blob store (reset between tests via _reset classmethod)
    _store: dict[str, bytes] = {}

    @classmethod
    def _reset(cls) -> None:
        cls._store.clear()

    def __init__(self, project: str | None = None) -> None:
        # project arg accepted but ignored (we're offline)
        pass

    def open(self, path: str, mode: str = "rb") -> _MockContextManager:
        return _MockContextManager(self._store, path, mode)

    def ls(self, prefix: str) -> list[str]:
        """Return stored keys whose bare path starts with *prefix* (no gs://)."""
        bare = prefix.lstrip("gs://")
        return [k for k in self._store if k.startswith(bare)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _MockGCSModule:
    """Minimal stub for the ``gcsfs`` module, used when gcsfs is not installed."""
    GCSFileSystem = _MockGCSFileSystem


def _patch_gcsfs():
    """Return a mock.patch context patching the gcsfs module reference in seg.gcs_checkpoint.

    When gcsfs is not installed the module attribute is ``None``; we replace the
    entire module reference with ``_MockGCSModule`` which exposes
    ``GCSFileSystem = _MockGCSFileSystem``.
    """
    return mock.patch("seg.gcs_checkpoint.gcsfs", _MockGCSModule)


def _make_state(step: int) -> dict:
    return {
        "step": step,
        "model_state_dict": {"w": torch.tensor([1.0, 2.0])},
        "optimizer_state_dict": {"lr": 1e-4},
        "config": "siglip-b",
    }


# ---------------------------------------------------------------------------
# TestGCSCheckpointRoundTrip
# ---------------------------------------------------------------------------


class TestGCSCheckpointRoundTrip:
    """Checkpoint write + resume round-trips using mocked gcsfs (D-07/D-09)."""

    def setup_method(self):
        _MockGCSFileSystem._reset()

    def test_save_and_resume(self):
        """Write a single checkpoint and confirm round-trip fidelity."""
        _require_gcs()
        with _patch_gcsfs():
            state = _make_state(42)
            gcs_save_checkpoint("siglip-b", step=42, state=state)
            step, loaded = gcs_latest_checkpoint("siglip-b")
        assert step == 42
        assert loaded is not None
        assert loaded["step"] == 42
        assert torch.allclose(loaded["model_state_dict"]["w"], state["model_state_dict"]["w"])

    def test_resume_returns_zero_on_empty_prefix(self):
        """If no checkpoint exists under a config prefix, return (0, None)."""
        _require_gcs()
        with _patch_gcsfs():
            step, state = gcs_latest_checkpoint("nonexistent-config")
        assert step == 0
        assert state is None

    def test_resume_picks_highest_step(self):
        """When multiple checkpoints exist, resume returns the highest step."""
        _require_gcs()
        with _patch_gcsfs():
            for s in (10, 200, 30):
                gcs_save_checkpoint("siglip-b", step=s, state=_make_state(s))
            step, loaded = gcs_latest_checkpoint("siglip-b")
        assert step == 200
        assert loaded is not None
        assert loaded["step"] == 200


# ---------------------------------------------------------------------------
# TestCorruptCheckpointSkip  (T-04-05)
# ---------------------------------------------------------------------------


class TestCorruptCheckpointSkip:
    """Corrupt-blob DoS guard: a truncated blob is skipped (T-04-05)."""

    def setup_method(self):
        _MockGCSFileSystem._reset()

    def test_corrupt_latest_skipped(self):
        """A garbage-bytes blob at a higher step is skipped; valid lower step returned."""
        _require_gcs()
        # Write a valid checkpoint at step 10
        with _patch_gcsfs():
            gcs_save_checkpoint("siglip-b", step=10, state=_make_state(10))
            # Manually inject corrupt bytes at step 99 into the shared store
            corrupt_key = "mapclass-training-northeast1/models/siglip-b/step_0000099.pt"
            _MockGCSFileSystem._store[corrupt_key] = b"not a checkpoint"
            # Resume should skip 99 and return 10
            step, loaded = gcs_latest_checkpoint("siglip-b")
        assert step == 10
        assert loaded is not None
        assert loaded["step"] == 10


# ---------------------------------------------------------------------------
# TestConfigNaming  (D-07 / T-04-04)
# ---------------------------------------------------------------------------


class TestConfigNaming:
    """GCS checkpoint prefix uses lowercased ``<backbone>-<variant>`` naming (D-07)."""

    @pytest.mark.parametrize("backbone, variant, expected", [
        ("siglip",  "B", "siglip-b"),
        ("dinov2",  "A", "dinov2-a"),
        ("swin",    "B", "swin-b"),
    ])
    def test_config_name_format(self, backbone: str, variant: str, expected: str):
        """config_prefix lowercases both args and returns the validated string."""
        _require_gcs()
        result = config_prefix(backbone, variant)
        assert result == expected

    def test_rejects_traversal(self):
        """Path-traversal strings are rejected before any GCS URI is built (T-04-04)."""
        _require_gcs()
        with pytest.raises(ValueError):
            validate_config_name("../../x")
        with pytest.raises(ValueError):
            validate_config_name("a/b")

    def test_rejects_uppercase(self):
        """Uppercase config names are rejected (would break the GCS-safe regex)."""
        _require_gcs()
        with pytest.raises(ValueError):
            validate_config_name("Siglip-B")

    def test_rejects_leading_hyphen(self):
        """A leading hyphen is rejected."""
        _require_gcs()
        with pytest.raises(ValueError):
            validate_config_name("-siglip")
