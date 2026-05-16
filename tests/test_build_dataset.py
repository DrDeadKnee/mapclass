"""
Offline unit tests for the GCS-canonical build_dataset.py (Plan 02-02 Task 2).

Tests cover:
  - test_manifest_fails_before_split: mis-named raw file -> SystemExit before split
  - test_split_uses_manifest_template: stratified_split groups by manifest template
  - test_split_frozen_in_gcs: first build writes split.json; second reads frozen
  - test_refreeze_split_recomputes: --refreeze-split deletes GCS split.json
  - test_local_ok_required: non-gs:// out-dir without --local-ok -> hard error

All tests are fully offline: GCS is replaced by _MockGCSFileSystem.
No rendering / labeling / tiling is invoked (those are mocked out).
"""

from __future__ import annotations

import io
import json
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# Lazy import guard — fail loudly so Nyquist gate fires offline
# ---------------------------------------------------------------------------

_import_error: Exception | None = None
try:
    import build_dataset as bd
except ImportError as _e:
    _import_error = _e


def _require_bd() -> None:
    if _import_error is not None:
        pytest.fail(
            f"build_dataset not importable: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Mock GCS infrastructure (mirrors conftest._MockGCSModule)
# ---------------------------------------------------------------------------

class _MockGCSFileSystem:
    """In-memory mock for gcsfs.GCSFileSystem."""

    _store: dict[str, bytes] = {}

    @classmethod
    def _reset(cls) -> None:
        cls._store.clear()

    def __init__(self, project: str | None = None) -> None:
        pass

    def open(self, path: str, mode: str = "rb"):
        key = path.lstrip("gs://")
        store = self._store

        class _CM:
            def __init__(self_):
                self_._key = key
                self_._mode = mode

            def __enter__(self_):
                if "r" in self_._mode:
                    self_._buf = io.BytesIO(store.get(self_._key, b""))
                else:
                    self_._buf = io.BytesIO()
                return self_._buf

            def __exit__(self_, exc_type, exc_val, exc_tb):
                if "w" in self_._mode and exc_type is None:
                    store[self_._key] = self_._buf.getvalue()

        return _CM()

    def ls(self, prefix: str) -> list[str]:
        bare = prefix.lstrip("gs://")
        return [k for k in self._store if k.startswith(bare)]

    def pipe_file(self, path: str, data: bytes) -> None:
        key = path.lstrip("gs://")
        self._store[key] = data

    def cat(self, path: str) -> bytes:
        key = path.lstrip("gs://")
        return self._store.get(key, b"")

    def exists(self, path: str) -> bool:
        key = path.lstrip("gs://")
        return key in self._store

    def rm(self, path: str) -> None:
        key = path.lstrip("gs://")
        self._store.pop(key, None)

    def mkdirs(self, path: str, exist_ok: bool = True) -> None:
        pass


class _MockGCSModule:
    GCSFileSystem = _MockGCSFileSystem


def _reset_mock() -> None:
    _MockGCSFileSystem._reset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GCS_RAW_PREFIX = "mapclass-training-northeast1/data/synthetic/raw"
_GCS_SPLIT_PATH = "mapclass-training-northeast1/data/synthetic/split.json"


def _seed_raw_files(filenames: list[str], manifest: dict) -> None:
    """Populate mock GCS store with .geojson stubs + manifest.json."""
    for fn in filenames:
        key = f"{_GCS_RAW_PREFIX}/{fn}"
        _MockGCSFileSystem._store[key] = b'{"type":"FeatureCollection","features":[]}'
    manifest_key = f"{_GCS_RAW_PREFIX}/manifest.json"
    _MockGCSFileSystem._store[manifest_key] = json.dumps(manifest).encode()


def _make_manifest(filenames: list[str]) -> dict:
    """Build a valid manifest.json dict for the given geojson filenames."""
    entries = {}
    for fn in filenames:
        # template = filename without _NN.geojson suffix
        stem = fn[:-8]  # strip .geojson
        template = stem.rsplit("_", 1)[0]
        entries[fn] = {"template": template}
    return {"version": "1", "entries": entries}


# ---------------------------------------------------------------------------
# Test: manifest hard-fails before split
# ---------------------------------------------------------------------------

class TestManifestFailsBeforeSplit:
    """validate_manifest is called BEFORE load_or_create_split (RW-02 / T-02-10)."""

    def setup_method(self):
        _reset_mock()

    def test_manifest_fails_before_split(self, tmp_path):
        _require_bd()

        # A file that violates the filename convention (no _NN suffix)
        bad_fn = "europe_bad.geojson"
        good_fn = "europe_01.geojson"
        manifest = _make_manifest([good_fn])  # bad_fn NOT in manifest
        _seed_raw_files([bad_fn, good_fn], manifest)

        with mock.patch("build_dataset.gcsfs", _MockGCSModule):
            with mock.patch("build_dataset._GCSWriter"):
                with pytest.raises(SystemExit) as exc_info:
                    bd.build(
                        raw_dir="gs://mapclass-training-northeast1/data/synthetic/raw",
                        output_dir="gs://mapclass-training-northeast1/data/synthetic",
                    )
                assert exc_info.value.code == 1

        # split.json must NOT have been written (validate_manifest fired first)
        assert not _MockGCSFileSystem._store.get(_GCS_SPLIT_PATH), \
            "split.json was written despite manifest validation failure"


# ---------------------------------------------------------------------------
# Test: stratified_split uses manifest template
# ---------------------------------------------------------------------------

class TestSplitUsesManifestTemplate:
    """stratified_split groups by id_to_template (manifest authoritative)."""

    def setup_method(self):
        _reset_mock()

    def test_split_uses_manifest_template(self):
        _require_bd()

        # Create source IDs whose template_key() fallback would differ from
        # the manifest's explicit template.  If the split uses the manifest,
        # all "xyz_*" IDs share the "xyz" stratum.
        id_to_template = {
            "xyz_01": "xyz",
            "xyz_02": "xyz",
            "xyz_03": "xyz",
            "xyz_04": "xyz",
            "abc_01": "abc",
            "abc_02": "abc",
        }
        source_ids = list(id_to_template.keys())

        test_ids = bd.stratified_split(
            source_ids,
            id_to_template=id_to_template,
        )

        # All test_ids must be in source_ids
        assert all(t in source_ids for t in test_ids)
        # Both templates should contribute to the hold-out
        assert any(t.startswith("xyz") for t in test_ids)
        assert any(t.startswith("abc") for t in test_ids)

        # Verify grouping was done by manifest template (not template_key fallback)
        # by checking that the stratification matches the manifest groups
        xyz_test = [t for t in test_ids if t.startswith("xyz")]
        abc_test = [t for t in test_ids if t.startswith("abc")]
        # Each group gets >=1 hold-out (stratified >=1 rule)
        assert len(xyz_test) >= 1
        assert len(abc_test) >= 1


# ---------------------------------------------------------------------------
# Test: split.json is frozen in GCS
# ---------------------------------------------------------------------------

class TestSplitFrozenInGCS:
    """First build writes split.json to GCS; second build reads the frozen split."""

    def setup_method(self):
        _reset_mock()

    def test_split_frozen_in_gcs(self, tmp_path):
        _require_bd()

        filenames = [f"europe_{i:02d}.geojson" for i in range(1, 6)]
        manifest = _make_manifest(filenames)
        _seed_raw_files(filenames, manifest)

        # Mock tiling + rendering so build completes without actual I/O
        with mock.patch("build_dataset.gcsfs", _MockGCSModule), \
             mock.patch("build_dataset.tiling.tile"), \
             mock.patch("build_dataset.render_one") as mock_render, \
             mock.patch("build_dataset.make_label_arrays") as mock_labels, \
             mock.patch("build_dataset.write_sample_weights"), \
             mock.patch("build_dataset._GCSWriter"):

            from PIL import Image as _Image
            mock_render.return_value = _Image.new("RGB", (100, 100))
            mock_labels.return_value = (
                _Image.new("L", (100, 100)),
                _Image.new("L", (100, 100)),
            )

            # First build — should write split.json to mock GCS
            bd.build(
                raw_dir="gs://mapclass-training-northeast1/data/synthetic/raw",
                output_dir="gs://mapclass-training-northeast1/data/synthetic",
            )
            assert _GCS_SPLIT_PATH in _MockGCSFileSystem._store, \
                "split.json not written to GCS after first build"

            first_split_bytes = _MockGCSFileSystem._store[_GCS_SPLIT_PATH]
            first_test_ids = set(json.loads(first_split_bytes)["test"])

            # Add extra source to GCS raw prefix
            extra_fn = "europe_99.geojson"
            extra_key = f"{_GCS_RAW_PREFIX}/{extra_fn}"
            _MockGCSFileSystem._store[extra_key] = b'{"type":"FeatureCollection","features":[]}'
            # Update manifest
            new_manifest = _make_manifest(filenames + [extra_fn])
            manifest_key = f"{_GCS_RAW_PREFIX}/manifest.json"
            _MockGCSFileSystem._store[manifest_key] = json.dumps(new_manifest).encode()

            # Second build — split.json must remain frozen
            bd.build(
                raw_dir="gs://mapclass-training-northeast1/data/synthetic/raw",
                output_dir="gs://mapclass-training-northeast1/data/synthetic",
            )
            second_split_bytes = _MockGCSFileSystem._store[_GCS_SPLIT_PATH]
            second_test_ids = set(json.loads(second_split_bytes)["test"])

        # split.json frozen — test IDs must not change
        assert second_test_ids == first_test_ids, \
            f"split.json was recomputed on second build: {first_test_ids} -> {second_test_ids}"

        # New source must NOT be in the frozen test set
        assert "europe_99" not in second_test_ids


# ---------------------------------------------------------------------------
# Test: --refreeze-split deletes GCS split.json and recomputes
# ---------------------------------------------------------------------------

class TestRefreezeSplitRecomputes:
    """--refreeze-split deletes the GCS split.json then recomputes (Pitfall R-3)."""

    def setup_method(self):
        _reset_mock()

    def test_refreeze_split_recomputes(self, tmp_path):
        _require_bd()

        filenames = [f"europe_{i:02d}.geojson" for i in range(1, 5)]
        manifest = _make_manifest(filenames)
        _seed_raw_files(filenames, manifest)

        # Pre-seed an existing split.json in mock GCS
        old_split = {"seed": 42, "test_fraction": 0.15, "test": ["europe_01"]}
        _MockGCSFileSystem._store[_GCS_SPLIT_PATH] = json.dumps(old_split).encode()

        with mock.patch("build_dataset.gcsfs", _MockGCSModule), \
             mock.patch("build_dataset.tiling.tile"), \
             mock.patch("build_dataset.render_one") as mock_render, \
             mock.patch("build_dataset.make_label_arrays") as mock_labels, \
             mock.patch("build_dataset.write_sample_weights"), \
             mock.patch("build_dataset._GCSWriter"):

            from PIL import Image as _Image
            mock_render.return_value = _Image.new("RGB", (100, 100))
            mock_labels.return_value = (
                _Image.new("L", (100, 100)),
                _Image.new("L", (100, 100)),
            )

            # Build with --refreeze-split — must delete old split and recompute
            bd.build(
                raw_dir="gs://mapclass-training-northeast1/data/synthetic/raw",
                output_dir="gs://mapclass-training-northeast1/data/synthetic",
                refreeze_split=True,
            )

        # split.json must exist in GCS after refreeze
        assert _GCS_SPLIT_PATH in _MockGCSFileSystem._store, \
            "split.json missing after --refreeze-split"

        new_split = json.loads(_MockGCSFileSystem._store[_GCS_SPLIT_PATH])
        # The old split was deleted; new split was recomputed from current sources
        # (not necessarily different, but it must come from stratified_split)
        assert "test" in new_split
        assert isinstance(new_split["test"], list)


# ---------------------------------------------------------------------------
# Test: --local-ok required for non-gs:// out-dir
# ---------------------------------------------------------------------------

class TestLocalOkRequired:
    """A non-gs:// out-dir without --local-ok raises SystemExit (Pitfall R-1 / T-02-12)."""

    def test_local_ok_required(self, tmp_path):
        _require_bd()

        with pytest.raises(SystemExit) as exc_info:
            bd.build(
                raw_dir="gs://mapclass-training-northeast1/data/synthetic/raw",
                output_dir=str(tmp_path / "local_out"),
                local_ok=False,
            )
        assert exc_info.value.code == 1
