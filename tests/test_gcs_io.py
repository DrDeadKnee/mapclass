"""
Offline unit tests for gcs_io._GCSWriter (Plan 02-01 Task 1 + Task 2).

All GCS I/O is replaced by a local fsspec filesystem or mock — no network,
no gcsfs installation required. Tests are fully deterministic.

Coverage (Task 1):
  - test_gcs_writer_writes_bytes
  - test_gcs_writer_write_text
  - test_gcs_writer_truediv
  - test_gcs_writer_pillow_compat
  - test_gcs_io_importable_without_gcsfs

Coverage (Task 2):
  - test_pull_idempotent
  - test_verify_raises_on_missing_map
  - test_verify_raises_on_truncated_pyramid
  - test_verify_passes_on_complete
  - test_concurrent_tile_writes
"""

from __future__ import annotations

import io
import json
import threading
import unittest.mock as mock
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import fsspec
import pytest
from PIL import Image


# ---------------------------------------------------------------------------
# Import guard — LOUD fail (never skip) so the Nyquist gate fires offline
# ---------------------------------------------------------------------------

_import_error: Exception | None = None
try:
    import gcs_io
    from gcs_io import _GCSWriter, validate_manifest
except ImportError as _e:
    _import_error = _e


def _require_gcs_io() -> None:
    if _import_error is not None:
        pytest.fail(
            f"gcs_io not yet implemented: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _local_fs():
    """Return a local fsspec filesystem for offline testing."""
    return fsspec.filesystem("file")


# ---------------------------------------------------------------------------
# Task 1 Tests — _GCSWriter
# ---------------------------------------------------------------------------

class TestGCSWriterWritesBytes:
    """_GCSWriter.write_bytes writes bytes that can be read back."""

    def test_gcs_writer_writes_bytes(self, tmp_path):
        _require_gcs_io()
        fs = _local_fs()
        writer = _GCSWriter(fs, str(tmp_path / "a"))
        writer.write_bytes(b"hello")
        assert Path(tmp_path / "a").read_bytes() == b"hello"

    def test_gcs_writer_write_text(self, tmp_path):
        _require_gcs_io()
        fs = _local_fs()
        writer = _GCSWriter(fs, str(tmp_path / "b"))
        writer.write_text("hello world")
        assert Path(tmp_path / "b").read_bytes() == b"hello world"

    def test_gcs_writer_write_text_utf8_encoding(self, tmp_path):
        _require_gcs_io()
        fs = _local_fs()
        writer = _GCSWriter(fs, str(tmp_path / "c"))
        writer.write_text("café")  # é is multi-byte in utf-8
        assert Path(tmp_path / "c").read_bytes() == "café".encode("utf-8")


class TestGCSWriterTruediv:
    """_GCSWriter.__truediv__ produces correct child writer."""

    def test_gcs_writer_truediv(self, tmp_path):
        _require_gcs_io()
        fs = _local_fs()
        parent = _GCSWriter(fs, "a")
        child = parent / "b"
        assert child._prefix == "a/b"

    def test_gcs_writer_truediv_name_property(self):
        _require_gcs_io()
        fs = _local_fs()
        writer = _GCSWriter(fs, "a/b/c")
        assert writer.name == "c"

    def test_gcs_writer_nested_truediv(self):
        _require_gcs_io()
        fs = _local_fs()
        root = _GCSWriter(fs, "root")
        deep = root / "level1" / "level2"
        assert deep._prefix == "root/level1/level2"

    def test_gcs_writer_truediv_name_matches(self):
        _require_gcs_io()
        fs = _local_fs()
        parent = _GCSWriter(fs, "a")
        child = parent / "myname"
        assert child.name == "myname"


class TestGCSWriterPillowCompat:
    """_GCSWriter.open('wb') is usable as a PIL Image save target."""

    def test_gcs_writer_pillow_compat(self, tmp_path):
        _require_gcs_io()
        fs = _local_fs()
        out_path = tmp_path / "test.png"
        writer = _GCSWriter(fs, str(out_path))
        img = Image.new("RGB", (16, 16), color=(255, 0, 0))
        with writer.open("wb") as fh:
            img.save(fh, format="PNG")
        # Verify round-trip: read back and check it's a valid PNG
        loaded = Image.open(out_path)
        assert loaded.size == (16, 16)
        assert loaded.mode == "RGB"

    def test_gcs_writer_mkdir_no_error(self, tmp_path):
        """mkdir() should not raise on local fs."""
        _require_gcs_io()
        fs = _local_fs()
        writer = _GCSWriter(fs, str(tmp_path / "newdir"))
        writer.mkdir(parents=True, exist_ok=True)  # should not raise


class TestGCSIoImportableWithoutGcsfs:
    """gcs_io imports successfully even when gcsfs is None/absent."""

    def test_gcs_io_importable_without_gcsfs(self):
        _require_gcs_io()
        # Patch gcs_io.gcsfs to None (simulates absent gcsfs)
        with mock.patch.object(gcs_io, "gcsfs", None):
            # Module-level constants must still be accessible
            assert gcs_io.GCS_PROJECT == "narrative-campaign"
            assert gcs_io.BUCKET == "mapclass-training-northeast1"
            assert gcs_io.DATA_PREFIX == "mapclass-training-northeast1/data"
            # _GCSWriter must still be instantiable (it doesn't call gcsfs directly)
            fs = fsspec.filesystem("file")
            writer = _GCSWriter(fs, "some/prefix")
            assert writer._prefix == "some/prefix"


# ---------------------------------------------------------------------------
# Task 2 Tests — pull_dataset_from_gcs + verify_pull
# ---------------------------------------------------------------------------

# Lazily import task-2 functions so Task-1-only test runs are still possible
_pull_fn = None
_verify_fn = None
try:
    from gcs_io import pull_dataset_from_gcs, verify_pull
    _pull_fn = pull_dataset_from_gcs
    _verify_fn = verify_pull
except ImportError:
    pass


def _require_task2() -> None:
    if _pull_fn is None or _verify_fn is None:
        pytest.fail(
            "pull_dataset_from_gcs / verify_pull not yet implemented in gcs_io",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# Mock GCS module for pull tests
# ---------------------------------------------------------------------------

class _MockGCSFileSystem:
    """In-memory mock for gcsfs.GCSFileSystem, extended for pull tests."""

    _store: dict[str, bytes] = {}

    @classmethod
    def _reset(cls):
        cls._store.clear()

    def __init__(self, project=None):
        pass

    def open(self, path, mode="rb"):
        key = path.lstrip("gs://")

        class _CM:
            def __init__(self_, path_key, mode_):
                self_._key = path_key
                self_._mode = mode_

            def __enter__(self_):
                if "r" in self_._mode:
                    self_._buf = io.BytesIO(_MockGCSFileSystem._store.get(self_._key, b""))
                else:
                    self_._buf = io.BytesIO()
                return self_._buf

            def __exit__(self_, exc_type, exc_val, exc_tb):
                if "w" in self_._mode and exc_type is None:
                    _MockGCSFileSystem._store[self_._key] = self_._buf.getvalue()

        return _CM(key, mode)

    def ls(self, prefix):
        bare = prefix.lstrip("gs://")
        return [k for k in self._store if k.startswith(bare)]

    def pipe_file(self, path, data):
        key = path.lstrip("gs://")
        _MockGCSFileSystem._store[key] = data

    def cat(self, path):
        key = path.lstrip("gs://")
        return _MockGCSFileSystem._store.get(key, b"")

    def exists(self, path):
        key = path.lstrip("gs://")
        return key in _MockGCSFileSystem._store

    def get(self, remote, local, recursive=False):
        """Materialise remote keys under local directory."""
        bare_remote = remote.lstrip("gs://").rstrip("/")
        local_root = Path(local)
        local_root.mkdir(parents=True, exist_ok=True)
        for key, data in list(_MockGCSFileSystem._store.items()):
            if key.startswith(bare_remote + "/"):
                rel = key[len(bare_remote) + 1:]
                dest = local_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)

    def mkdirs(self, path, exist_ok=True):
        pass


class _MockGCSModule:
    GCSFileSystem = _MockGCSFileSystem


def _make_split_json(train_ids=("map_a",), test_ids=("map_b",)):
    return {"seed": 42, "test_fraction": 0.15, "train": list(train_ids), "test": list(test_ids)}


class TestPullIdempotent:
    """pull_dataset_from_gcs is idempotent — second call does not fail or corrupt."""

    def setup_method(self):
        _MockGCSFileSystem._reset()

    def test_pull_idempotent(self, tmp_path):
        _require_task2()
        # Seed a map in the mock GCS store
        _MockGCSFileSystem._store[
            "mapclass-training-northeast1/data/synthetic/train/map_a/image.png"
        ] = b"fakepng"

        with mock.patch("gcs_io.gcsfs", _MockGCSModule):
            result1 = pull_dataset_from_gcs("train", tmp_path)
            # Second call — idempotent, should not raise
            result2 = pull_dataset_from_gcs("train", tmp_path)

        assert result1 == result2
        assert (tmp_path / "train" / "map_a" / "image.png").read_bytes() == b"fakepng"


class TestVerifyPull:
    """verify_pull raises RuntimeError on missing dir or truncated pyramid."""

    def _make_complete_pyramid(self, base: Path, map_id: str, n_pngs: int = 63):
        """Create a local directory structure that verify_pull considers complete."""
        map_dir = base / map_id
        pyr_dir = map_dir / "pyramids" / "py_r0"
        pyr_dir.mkdir(parents=True, exist_ok=True)
        (pyr_dir / "pyramid.json").write_text(json.dumps({"tiles": []}))
        for i in range(n_pngs):
            (pyr_dir / f"tile_{i:03d}.png").write_bytes(b"fakepng")
        return map_dir

    def test_verify_raises_on_missing_map(self, tmp_path):
        _require_task2()
        local_root = tmp_path / "train"
        local_root.mkdir()
        split_json = _make_split_json(train_ids=["map_missing"])
        with pytest.raises(RuntimeError):
            verify_pull(local_root, split_json, "train")

    def test_verify_raises_on_truncated_pyramid(self, tmp_path):
        _require_task2()
        local_root = tmp_path / "train"
        # Create map dir but with only 5 PNGs (< 60 threshold)
        self._make_complete_pyramid(local_root, "map_truncated", n_pngs=5)
        split_json = _make_split_json(train_ids=["map_truncated"])
        with pytest.raises(RuntimeError):
            verify_pull(local_root, split_json, "train")

    def test_verify_passes_on_complete(self, tmp_path):
        _require_task2()
        local_root = tmp_path / "train"
        self._make_complete_pyramid(local_root, "map_ok", n_pngs=63)
        split_json = _make_split_json(train_ids=["map_ok"])
        # Should not raise
        verify_pull(local_root, split_json, "train")


# ---------------------------------------------------------------------------
# Task 5 Tests — _BUILD_COMPLETE sentinel (OQ1) + family_subset_prefix (OQ2)
# ---------------------------------------------------------------------------

# Lazily import new Task-5 symbols so earlier tasks still pass without them.
_mark_complete_fn = None
_is_complete_fn = None
_family_prefix_fn = None
try:
    from gcs_io import mark_build_complete, is_build_complete, family_subset_prefix
    _mark_complete_fn = mark_build_complete
    _is_complete_fn = is_build_complete
    _family_prefix_fn = family_subset_prefix
except ImportError:
    pass


def _require_task5() -> None:
    if any(fn is None for fn in [_mark_complete_fn, _is_complete_fn, _family_prefix_fn]):
        pytest.fail(
            "mark_build_complete / is_build_complete / family_subset_prefix "
            "not yet implemented in gcs_io",
            pytrace=False,
        )


class TestBuildCompleteSentinel:
    """_BUILD_COMPLETE sentinel: written last, gates skip correctly (OQ1)."""

    def setup_method(self):
        _MockGCSFileSystem._reset()

    def test_build_complete_written_last(self, tmp_path):
        """After mark_build_complete, sentinel object exists in the mock store."""
        _require_task5()
        fs = _MockGCSFileSystem()
        map_dir_prefix = "mapclass-training-northeast1/data/synthetic/train/europe_01__flat"
        # Simulate writing some pyramid objects first (normal build)
        fs.pipe_file(f"{map_dir_prefix}/pyramids/py_r0/pyramid.json", b"{}")
        # Now write the sentinel last
        mark_build_complete(fs, map_dir_prefix)
        sentinel_key = f"{map_dir_prefix}/_BUILD_COMPLETE"
        assert sentinel_key in _MockGCSFileSystem._store, (
            f"Expected sentinel at {sentinel_key!r}; store keys: "
            f"{list(_MockGCSFileSystem._store.keys())}"
        )

    def test_skip_only_if_sentinel_present(self, tmp_path):
        """A prefix that exists but lacks sentinel is NOT skipped (partial/aborted)."""
        _require_task5()
        fs = _MockGCSFileSystem()
        map_dir_prefix = "mapclass-training-northeast1/data/synthetic/train/europe_02__flat"

        # Write some pyramids but NO sentinel → should NOT be considered complete
        fs.pipe_file(f"{map_dir_prefix}/pyramids/py_r0/pyramid.json", b"{}")
        assert not is_build_complete(fs, map_dir_prefix), (
            "is_build_complete must return False when sentinel is absent (partial build)"
        )

        # Write sentinel → now it IS complete
        mark_build_complete(fs, map_dir_prefix)
        assert is_build_complete(fs, map_dir_prefix), (
            "is_build_complete must return True after mark_build_complete"
        )

    def test_is_build_complete_false_for_empty_prefix(self):
        """is_build_complete returns False for a prefix with no objects at all."""
        _require_task5()
        _MockGCSFileSystem._reset()
        fs = _MockGCSFileSystem()
        assert not is_build_complete(fs, "mapclass-training-northeast1/data/synthetic/train/nonexistent")


class TestLayoutConstants:
    """family_subset_prefix resolves (family, subset) → expected GCS path (OQ2)."""

    def test_layout_constants(self):
        """family_subset_prefix returns the family-rooted GCS path string."""
        _require_task5()
        result = family_subset_prefix("synthetic", "train")
        assert result == "mapclass-training-northeast1/data/synthetic/train", (
            f"Expected 'mapclass-training-northeast1/data/synthetic/train', got {result!r}"
        )

    def test_layout_constants_historical(self):
        """family_subset_prefix works for historical family too."""
        _require_task5()
        result = family_subset_prefix("historical", "dataset")
        assert result == "mapclass-training-northeast1/data/historical/dataset"

    def test_layout_constants_satellite(self):
        """family_subset_prefix works for satellite family."""
        _require_task5()
        result = family_subset_prefix("satellite", "train")
        assert result == "mapclass-training-northeast1/data/satellite/train"


class TestConcurrentTileWrites:
    """32-thread concurrent writes via _GCSWriter to local fs produce no key loss."""

    def test_concurrent_tile_writes(self, tmp_path):
        _require_gcs_io()
        fs = _local_fs()
        root = _GCSWriter(fs, str(tmp_path / "pyramid"))
        # Create parent dir first — mirrors tiling.py pdir.mkdir() call before writes
        root.mkdir()

        n_tiles = 64
        errors = []

        def _write_tile(i: int):
            try:
                tile_writer = root / f"tile_{i:03d}.png"
                buf = io.BytesIO()
                img = Image.new("L", (8, 8), i % 256)
                img.save(buf, format="PNG")
                tile_writer.write_bytes(buf.getvalue())
            except Exception as exc:
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(_write_tile, range(n_tiles)))

        assert not errors, f"Concurrent writes raised exceptions: {errors}"
        # Verify all tiles were written (no key loss)
        written = list((tmp_path / "pyramid").glob("tile_*.png"))
        assert len(written) == n_tiles, (
            f"Expected {n_tiles} tiles, found {len(written)}"
        )
