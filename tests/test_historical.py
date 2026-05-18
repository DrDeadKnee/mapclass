"""
Offline GCS-write tests for build_historical_dataset.py (Plan 02-03, Task 1).

Three behaviour tests (RW-04 / T-02-20..T-02-22):

  test_manifest_written_to_gcs
      search path emits unregistered_manifest.json under the GCS key
      mapclass-training-northeast1/data/historical/raw/unregistered_manifest.json
      in the mock GCS store, NOT to a local path.

  test_historical_dataset_written_to_gcs
      build path routes tiling.tile through _GCSWriter so mock GCS store
      receives pyramid objects under data/historical/dataset/<sample>/pyramids.

  test_raw_geotiff_stays_local
      the raw Rumsey GeoTIFF download target remains a local scratch Path;
      no GCS write of raw .tif files (RW-04 / T-02-23).

All tests are fully offline: GCS is replaced by _MockGCSFileSystem.
No network calls; tiling / labelling are mocked.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Ensure scripts/ is on sys.path
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# ---------------------------------------------------------------------------
# Lazy import guard — fail loudly so Nyquist gate fires offline
# ---------------------------------------------------------------------------

_import_error: Exception | None = None
try:
    import build_historical_dataset as bhd
except ImportError as _e:
    _import_error = _e


def _require_bhd() -> None:
    if _import_error is not None:
        pytest.fail(
            f"build_historical_dataset not importable: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# _MockGCSFileSystem — in-memory stand-in for gcsfs.GCSFileSystem
# Mirror of _MockGCSFileSystem from tests/test_build_dataset.py
# ---------------------------------------------------------------------------

class _MockGCSFileSystem:
    """In-memory mock for gcsfs.GCSFileSystem.

    Stores blobs as dict[str, bytes] keyed by bare path (no gs://).
    Instances within a test share a single class-level store.
    """

    _store: dict[str, bytes] = {}

    @classmethod
    def _reset(cls) -> None:
        cls._store.clear()

    def __init__(self, project: str | None = None) -> None:
        pass

    def pipe_file(self, path: str, data: bytes) -> None:
        key = path.lstrip("gs://")
        self._store[key] = data

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

    def mkdirs(self, path: str, exist_ok: bool = True) -> None:
        pass

    def ls(self, prefix: str) -> list[str]:
        bare = prefix.lstrip("gs://")
        return [k for k in self._store if k.startswith(bare)]

    def exists(self, path: str) -> bool:
        key = path.lstrip("gs://")
        return key in self._store


class _MockGCSModule:
    GCSFileSystem = _MockGCSFileSystem


_GCS_MANIFEST_KEY = (
    "mapclass-training-northeast1/data/historical/raw/unregistered_manifest.json"
)
_GCS_DATASET_PREFIX = "mapclass-training-northeast1/data/historical/dataset"


def _reset_mock() -> None:
    _MockGCSFileSystem._reset()


# ---------------------------------------------------------------------------
# Test: manifest written to GCS (not local path)
# ---------------------------------------------------------------------------

class TestManifestWrittenToGCS:
    """search writes unregistered_manifest.json to GCS (RW-04 / T-02-20)."""

    def setup_method(self):
        _reset_mock()

    def test_manifest_written_to_gcs(self, tmp_path):
        """search emits manifest JSON to GCS key, not to a local file."""
        _require_bhd()

        # Two fake LUNA items: one not_in_allmaps, one gcps_insufficient
        items = [
            {"id": "rumsey::001", "title": "Map A"},
            {"id": "rumsey::002", "title": "Map B"},
        ]
        unregistered = [(items[0], "not_in_allmaps"), (items[1], "gcps_insufficient")]

        with mock.patch("build_historical_dataset.gcsfs", _MockGCSModule):
            with mock.patch(
                "build_historical_dataset.rumsey.search_maps",
                return_value=items,
            ):
                # download_georeferenced returns (path, status)
                # First item not_in_allmaps, second gcps_insufficient
                statuses = ["not_in_allmaps", "gcps_insufficient"]
                call_count = [0]

                def fake_download(item, geo_dir):
                    idx = call_count[0]
                    call_count[0] += 1
                    return (geo_dir / f"fake_{idx}.tif"), statuses[idx]

                with mock.patch(
                    "build_historical_dataset.rumsey.download_georeferenced",
                    side_effect=fake_download,
                ):
                    with mock.patch(
                        "build_historical_dataset.rumsey.emit_manifest",
                        wraps=lambda items_list, path: path.write_text(
                            json.dumps([{"id": it[0].get("id"), "status": it[1]}
                                        for it in items_list], indent=2)
                        ),
                    ):
                        bhd.cmd_search(tmp_path / "raw", max_maps=2)

        # The GCS store must contain the manifest at the canonical key
        assert _GCS_MANIFEST_KEY in _MockGCSFileSystem._store, (
            f"manifest NOT found in mock GCS store under {_GCS_MANIFEST_KEY!r}; "
            f"store keys: {list(_MockGCSFileSystem._store.keys())}"
        )

        # Verify the manifest is parseable JSON with expected content
        raw = _MockGCSFileSystem._store[_GCS_MANIFEST_KEY]
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        ids = [entry.get("id") for entry in parsed]
        assert "rumsey::001" in ids
        assert "rumsey::002" in ids

        # Verify NO raw .tif was written to GCS
        gcs_tif_keys = [
            k for k in _MockGCSFileSystem._store if k.endswith(".tif")
        ]
        assert not gcs_tif_keys, (
            f"Raw GeoTIFF uploaded to GCS (should stay local): {gcs_tif_keys}"
        )


# ---------------------------------------------------------------------------
# Test: historical dataset written to GCS via _GCSWriter
# ---------------------------------------------------------------------------

class TestHistoricalDatasetWrittenToGCS:
    """build path routes tiling.tile through _GCSWriter (RW-04 / T-02-21)."""

    def setup_method(self):
        _reset_mock()

    def test_historical_dataset_written_to_gcs(self, tmp_path):
        """_process_one passes out_root=_GCSWriter(...) to tiling.tile."""
        _require_bhd()

        # Create a fake georeferenced GeoTIFF in tmp_path/raw/georeferenced/
        raw_dir = tmp_path / "raw"
        geo_dir = raw_dir / "georeferenced"
        geo_dir.mkdir(parents=True)
        fake_tif = geo_dir / "test_map.tif"
        fake_tif.write_bytes(b"\x00" * 16)  # dummy content

        tile_calls: list[dict] = []

        def mock_make_labels(tif_path, sample_dir):
            """Create the required files so the tiling guard passes."""
            sample_dir.mkdir(parents=True, exist_ok=True)
            (sample_dir / "image.png").write_bytes(b"PNG")
            (sample_dir / "land_cover.png").write_bytes(b"PNG")
            (sample_dir / "topography.png").write_bytes(b"PNG")
            (sample_dir / "sample_weights.json").write_text(
                json.dumps({"land_cover_weights": {}, "topography_weight": 1.0})
            )

        def mock_tile(map_dir, out_root=None, max_workers=32):
            """Capture the out_root argument — must be a _GCSWriter."""
            tile_calls.append({"map_dir": map_dir, "out_root": out_root})

        with mock.patch("build_historical_dataset.gcsfs", _MockGCSModule):
            with mock.patch(
                "build_historical_dataset.hist_label.make_labels",
                side_effect=mock_make_labels,
            ):
                with mock.patch(
                    "build_historical_dataset.tiling.tile",
                    side_effect=mock_tile,
                ):
                    out_dir = "gs://mapclass-training-northeast1/data/historical/dataset"
                    bhd.cmd_build(raw_dir, out_dir=out_dir, workers=1)

        assert tile_calls, "tiling.tile was never called"

        from gcs_io import _GCSWriter
        for call in tile_calls:
            out_root = call["out_root"]
            assert isinstance(out_root, _GCSWriter), (
                f"tiling.tile out_root must be _GCSWriter, got {type(out_root)!r}"
            )
            # Check the prefix targets the historical dataset path
            assert "historical/dataset" in out_root._prefix, (
                f"_GCSWriter prefix does not target historical/dataset: "
                f"{out_root._prefix!r}"
            )
            assert "pyramids" in out_root._prefix, (
                f"_GCSWriter prefix does not include /pyramids: {out_root._prefix!r}"
            )


# ---------------------------------------------------------------------------
# Test: raw GeoTIFF stays local (no GCS upload of raw .tif)
# ---------------------------------------------------------------------------

class TestRawGeoTIFFStaysLocal:
    """Raw Rumsey GeoTIFFs remain ephemeral local scratch (RW-04 / T-02-23)."""

    def setup_method(self):
        _reset_mock()

    def test_raw_geotiff_stays_local(self, tmp_path):
        """search downloads raw GeoTIFFs to a local Path; no fs.pipe_file for .tif."""
        _require_bhd()

        items = [{"id": "rumsey::003", "title": "Map C"}]

        tif_pipe_calls: list[str] = []
        original_pipe_file = _MockGCSFileSystem.pipe_file

        def tracking_pipe_file(self_, path: str, data: bytes) -> None:
            if path.endswith(".tif"):
                tif_pipe_calls.append(path)
            original_pipe_file(self_, path, data)

        with mock.patch("build_historical_dataset.gcsfs", _MockGCSModule):
            with mock.patch(
                "build_historical_dataset.rumsey.search_maps",
                return_value=items,
            ):
                def fake_download(item, geo_dir):
                    local_tif = geo_dir / "map_c.tif"
                    local_tif.parent.mkdir(parents=True, exist_ok=True)
                    local_tif.write_bytes(b"\x00" * 16)
                    return local_tif, "ok"

                with mock.patch(
                    "build_historical_dataset.rumsey.download_georeferenced",
                    side_effect=fake_download,
                ):
                    with mock.patch(
                        "build_historical_dataset.rumsey.emit_manifest",
                        wraps=lambda items_list, path: path.write_text(
                            json.dumps([], indent=2)
                        ),
                    ):
                        with mock.patch.object(
                            _MockGCSFileSystem,
                            "pipe_file",
                            side_effect=tracking_pipe_file,
                            autospec=True,
                        ):
                            bhd.cmd_search(tmp_path / "raw", max_maps=1)

        assert not tif_pipe_calls, (
            f"Raw GeoTIFF was uploaded to GCS (should stay local): {tif_pipe_calls}"
        )
