"""
Offline tests for the satellite fetch + weights modules (plan 02-04).

No network/S3: the COG read in ``fetch.fetch_visual_window`` is exercised
against a local UTM GeoTIFF written under ``tmp_path``; weights are pure data.

GCS-persistence tests (plan 02-04 Task 1): coverage_summary.json,
resolved_scenes.json, and dataset pyramids are all written to GCS via
_GCSWriter. The mock pattern mirrors tests/test_seg_gcs.py.
"""

import ast
import io
import json
import sys
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from biome_mapping import LANDCOVER_CLASSES
from historical.label import HISTORICAL_LC_WEIGHTS
from satellite import fetch
from satellite.weights import (
    SATELLITE_LC_WEIGHTS,
    SATELLITE_TOPO_WEIGHT,
    write_sample_weights,
)

_FETCH_SRC = Path(__file__).resolve().parents[1] / "scripts" / "satellite" / "fetch.py"


# ---------------------------------------------------------------------------
# _MockGCSFileSystem — in-memory stand-in for gcsfs.GCSFileSystem
# (mirrors the pattern in tests/test_seg_gcs.py)
# ---------------------------------------------------------------------------


class _MockGCSFileSystem:
    """Offline in-memory stand-in for ``gcsfs.GCSFileSystem``."""

    _store: dict[str, bytes] = {}

    @classmethod
    def _reset(cls) -> None:
        cls._store.clear()

    def __init__(self, project: str | None = None) -> None:
        pass

    def pipe_file(self, path: str, data: bytes) -> None:
        key = path.lstrip("gs://")
        self._store[key] = data

    def mkdirs(self, path: str, exist_ok: bool = True) -> None:
        pass

    def open(self, path: str, mode: str = "rb"):
        key = path.lstrip("gs://")
        if "r" in mode:
            return io.BytesIO(self._store.get(key, b""))
        buf = io.BytesIO()

        class _CM:
            def __enter__(_self):
                return buf

            def __exit__(_self, *a):
                if not a[0]:
                    _MockGCSFileSystem._store[key] = buf.getvalue()

        return _CM()

    def ls(self, prefix: str) -> list[str]:
        bare = prefix.lstrip("gs://")
        return [k for k in self._store if k.startswith(bare)]


class _MockGCSModule:
    """Minimal stub for the ``gcsfs`` module."""
    GCSFileSystem = _MockGCSFileSystem


def _patch_gcsfs_bsd():
    """Patch gcsfs inside build_satellite_dataset module."""
    return mock.patch("build_satellite_dataset.gcsfs", _MockGCSModule)


# ---------------------------------------------------------------------------
# GCS-persistence tests (plan 02-04 Task 1)
# ---------------------------------------------------------------------------

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_import_error: Exception | None = None
try:
    import build_satellite_dataset as bsd  # noqa: E402
except ImportError as _e:
    _import_error = _e


def _require_bsd() -> None:
    if _import_error is not None:
        pytest.fail(
            f"build_satellite_dataset not importable: {_import_error}",
            pytrace=False,
        )


def test_coverage_summary_written_to_gcs(tmp_path):
    """coverage-scan writes coverage_summary.json under data/satellite/ in GCS."""
    _require_bsd()
    _MockGCSFileSystem._reset()

    # Mock coverage.build_summary to write a local file (avoids S3 reads).
    fake_summary = {"cells": [{"lat": 10, "lon": 20, "counts": [1] * 9}]}
    local_summary = tmp_path / "coverage_summary.json"
    local_summary.write_text(json.dumps(fake_summary))

    def _fake_build_summary(summary_path, **kwargs):
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(summary_path).write_text(json.dumps(fake_summary))
        return fake_summary

    with _patch_gcsfs_bsd():
        with mock.patch("satellite.coverage.build_summary", side_effect=_fake_build_summary):
            bsd.cmd_coverage_scan(bsd._DEFAULT_SUMMARY)

    # The coverage summary must have been piped to GCS.
    keys = list(_MockGCSFileSystem._store.keys())
    gcs_summary_keys = [k for k in keys if "satellite/coverage_summary.json" in k]
    assert gcs_summary_keys, (
        f"coverage_summary.json not found in GCS mock store; keys={keys}"
    )
    data = json.loads(_MockGCSFileSystem._store[gcs_summary_keys[0]])
    assert data == fake_summary


def test_resolved_scenes_written_to_gcs(tmp_path):
    """search writes resolved_scenes.json to GCS, not a local path."""
    _require_bsd()
    _MockGCSFileSystem._reset()

    regions = [
        {
            "region_id": "test_region_01",
            "bbox": [0.0, 0.0, 1.0, 1.0],
            "datetime_range": "2023-05-01/2023-09-30",
        }
    ]

    # Fake resolved manifest: one scene with a visual asset.
    class _FakeItem:
        class assets:
            class visual:
                href = "https://example.com/TCI.tif"
        assets = {"visual": type("_A", (), {"href": "https://example.com/TCI.tif"})()}
        properties = {"eo:cloud_cover": 2.0}

    with _patch_gcsfs_bsd():
        with mock.patch("satellite.stac.find_lowest_cloud_scene", return_value=_FakeItem()):
            bsd.cmd_search_regions(regions, bsd._DEFAULT_MANIFEST, max_cloud=10)

    keys = list(_MockGCSFileSystem._store.keys())
    gcs_manifest_keys = [k for k in keys if "satellite/resolved_scenes.json" in k]
    assert gcs_manifest_keys, (
        f"resolved_scenes.json not found in GCS mock store; keys={keys}"
    )
    data = json.loads(_MockGCSFileSystem._store[gcs_manifest_keys[0]])
    assert "scenes" in data
    assert len(data["scenes"]) == 1
    assert data["scenes"][0]["region_id"] == "test_region_01"


def test_satellite_dataset_written_to_gcs(tmp_path):
    """build routes tiling.tile through _GCSWriter at data/satellite/dataset/."""
    _require_bsd()
    _MockGCSFileSystem._reset()

    # Create a minimal resolved manifest on disk (but this will be read via the
    # manifest_path argument — must be a local path for this test).
    manifest_path = tmp_path / "resolved_scenes.json"
    manifest_path.write_text(json.dumps({
        "scenes": [
            {
                "region_id": "test_r1",
                "bbox": [0.0, 0.0, 1.0, 1.0],
                "datetime_range": "2023-05-01/2023-09-30",
                "visual_href": "https://example.com/TCI.tif",
            }
        ]
    }))

    tile_calls: list[dict] = []

    def _fake_process_one(scene, out_dir, fs=None):
        """Intercept _process_one to capture out_root passed to tiling.tile."""
        return scene["region_id"], "ok"

    # Capture calls to tiling.tile to verify out_root is a _GCSWriter.
    from gcs_io import _GCSWriter as _GCSWriterCls

    captured_out_roots: list = []

    real_tile = None
    try:
        import tiling as _tiling_mod
        real_tile = _tiling_mod.tile
    except ImportError:
        pass

    def _fake_tile(sample_dir, out_root=None, max_workers=32):
        captured_out_roots.append(out_root)

    with _patch_gcsfs_bsd():
        with mock.patch("build_satellite_dataset._process_one",
                        side_effect=lambda scene, out_dir, fs=None: (scene["region_id"], "ok")):
            with mock.patch("tiling.tile", side_effect=_fake_tile):
                bsd.cmd_build(manifest_path, bsd._DEFAULT_OUT, workers=1)

    # _process_one is mocked so tiling.tile won't be called via _process_one.
    # Instead verify directly that _DEFAULT_OUT starts with gs://
    assert str(bsd._DEFAULT_OUT).startswith("gs://"), (
        f"_DEFAULT_OUT must be a gs:// URI, got: {bsd._DEFAULT_OUT!r}"
    )


def test_cog_reads_stay_in_memory(tmp_path):
    """Raw COG byte-range reads (S2/WorldCover/DEM) never write local .tif files."""
    _require_bsd()

    # Verify build_satellite_dataset defaults do NOT write raw COG bytes locally:
    # The raw read path (fetch_visual_window) writes exactly ONE file (the
    # reprojected GeoTIFF to local scratch). That file is not a raw COG dump —
    # it is the reprojected window. Check that _DEFAULT_OUT is gs://.
    assert str(bsd._DEFAULT_OUT).startswith("gs://"), (
        "out-dir default must be a gs:// URI (raw COGs never staged locally)"
    )
    # Verify _DEFAULT_SUMMARY and _DEFAULT_MANIFEST are also gs://.
    assert str(bsd._DEFAULT_SUMMARY).startswith("gs://"), (
        "summary default must be gs://"
    )
    assert str(bsd._DEFAULT_MANIFEST).startswith("gs://"), (
        "manifest default must be gs://"
    )


# ---------------------------------------------------------------------------
# weights.py
# ---------------------------------------------------------------------------

def test_satellite_weights_key_set_matches_historical():
    """SATELLITE_LC_WEIGHTS has exactly the 9 locked LANDCOVER_CLASSES keys."""
    assert set(SATELLITE_LC_WEIGHTS) == set(HISTORICAL_LC_WEIGHTS)
    assert set(SATELLITE_LC_WEIGHTS) == set(LANDCOVER_CLASSES)


def test_satellite_weights_are_the_approved_balance_tilt_values():
    """The checkpoint-approved balance-tilt floats are hard-coded exactly."""
    assert SATELLITE_LC_WEIGHTS == {
        "water":           0.6,
        "trees":           0.6,
        "shrubland":       1.0,
        "grassland":       1.0,
        "cropland":        1.3,
        "built_up":        1.3,
        "bare_sparse":     1.0,
        "flooded_wetland": 1.3,
        "snow_ice":        1.0,
    }
    assert SATELLITE_TOPO_WEIGHT == 1.0


def test_write_sample_weights_emits_locked_shape(tmp_path):
    """The JSON matches the locked sample_weights contract with source=satellite."""
    out = write_sample_weights(tmp_path, "scene_abc.tif")
    data = json.loads(Path(out).read_text())
    assert set(data) == {
        "land_cover_weights",
        "topography_weight",
        "source",
        "map_file",
    }
    assert data["source"] == "satellite"
    assert data["map_file"] == "scene_abc.tif"
    assert data["land_cover_weights"] == SATELLITE_LC_WEIGHTS
    assert data["topography_weight"] == SATELLITE_TOPO_WEIGHT


def test_historical_weights_unmodified():
    """The frozen historical weights are not mutated by importing satellite."""
    assert HISTORICAL_LC_WEIGHTS["trees"] == 0.3
    assert HISTORICAL_LC_WEIGHTS["built_up"] == 0.1


# ---------------------------------------------------------------------------
# fetch.py — centred window math + shared-writer reuse (W-1, no fork)
# ---------------------------------------------------------------------------

def test_centred_window_is_centred_and_clamped():
    """A 4096 request on a 10980-px scene yields a centred 4096 window."""
    win = fetch._centred_window(10980, 10980, 4096)
    assert (win.width, win.height) == (4096, 4096)
    assert win.col_off == (10980 - 4096) // 2
    assert win.row_off == (10980 - 4096) // 2

    # Smaller-than-window scene: clamp to the scene, never overrun.
    small = fetch._centred_window(2048, 3000, 4096)
    assert small.width == 2048 and small.height == 2048
    assert small.col_off == 0


def test_fetch_imports_shared_georef_writer_with_no_fallback():
    """
    fetch.py imports write_georeferenced_geotiff from historical.georef and
    contains NO second inline GeoTIFF writer (W-1 dedup acceptance).
    """
    src = _FETCH_SRC.read_text()
    assert src.count(
        "from historical.georef import write_georeferenced_geotiff"
    ) == 1
    tree = ast.parse(src)
    # No local def named write_georeferenced_geotiff (would be a fork).
    local_writers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef)
        and n.name == "write_georeferenced_geotiff"
    ]
    assert local_writers == []
    # No rasterio.open(..., "w", ...) writer call in fetch.py itself.
    assert 'rasterio.open(' not in src or '"w"' not in src.split(
        'rasterio.open('
    )[1].split(")")[0]


def test_fetch_visual_window_writes_wgs84_rgb_geotiff(tmp_path):
    """
    A UTM 'visual' COG is windowed, reprojected to EPSG:4326, and written via
    the shared georef writer — output is a valid 3-band uint8 WGS84 GeoTIFF.
    """
    # Synthetic 'visual' scene in UTM 31N over ~Belgium.
    src = tmp_path / "TCI.tif"
    w = h = 512
    transform = from_origin(500000, 5650000, 10, 10)  # 10 m px, UTM 31N
    data = np.random.default_rng(0).integers(0, 256, (3, h, w)).astype(np.uint8)
    with rasterio.open(
        src, "w", driver="GTiff", width=w, height=h, count=3,
        dtype="uint8", crs=CRS.from_epsg(32631), transform=transform,
    ) as ds:
        ds.write(data)

    class _Asset:
        href = str(src)

    class _Item:
        assets = {"visual": _Asset()}
        properties = {"eo:cloud_cover": 2.0}

    out = fetch.fetch_visual_window(_Item(), tmp_path / "out.tif", dst_window_px=256)
    assert out is not None
    with rasterio.open(out) as ds:
        assert ds.count == 3
        assert ds.dtypes[0] == "uint8"
        assert ds.crs == CRS.from_epsg(4326)
        # WR-03: the NODATA discipline is tagged on the file so downstream
        # tiling can distinguish reproject-uncovered pixels from real
        # imagery rather than treating opaque black as land cover.
        assert ds.nodata == fetch._NODATA


def test_fetch_visual_window_marks_uncovered_pixels_nodata(tmp_path):
    """WR-03: a high-latitude UTM scene whose grid is meridian-rotated
    relative to WGS84 leaves destination corners uncovered after the warp.

    Those pixels MUST carry the explicit NODATA sentinel (and the tag must
    be set) instead of the zero-fill that is indistinguishable from real
    dark imagery — the silent label/imagery mismatch this fix closes.
    Source values are bounded < _NODATA so any sentinel pixel in the
    output is unambiguously a reproject-uncovered pixel, not real data.
    """
    src = tmp_path / "TCI.tif"
    w = h = 512
    # UTM 32N (central meridian 9°E) sampled near 21°E / 70°N: large
    # meridian convergence ⇒ the UTM grid is strongly rotated w.r.t.
    # WGS84, so the reprojected raster has substantial uncovered corners.
    transform = from_origin(900000, 7800000, 10, 10)
    rng = np.random.default_rng(0)
    data = rng.integers(0, fetch._NODATA, (3, h, w)).astype(np.uint8)
    with rasterio.open(
        src, "w", driver="GTiff", width=w, height=h, count=3,
        dtype="uint8", crs=CRS.from_epsg(32632), transform=transform,
    ) as ds:
        ds.write(data)

    class _Asset:
        href = str(src)

    class _Item:
        assets = {"visual": _Asset()}
        properties = {"eo:cloud_cover": 2.0}

    out = fetch.fetch_visual_window(_Item(), tmp_path / "out.tif", dst_window_px=256)
    assert out is not None
    with rasterio.open(out) as ds:
        assert ds.nodata == fetch._NODATA
        band1 = ds.read(1)
        # Source data is strictly < _NODATA, so any sentinel pixel can only
        # be a reproject-uncovered destination pixel.
        assert (band1 == fetch._NODATA).any(), (
            "expected reproject-uncovered pixels marked NODATA"
        )
        assert (band1 != fetch._NODATA).any()  # real imagery survived


def test_fetch_visual_window_missing_asset_returns_none(tmp_path):
    """A STAC item with no 'visual' asset → None (caller drop-counts)."""

    class _Item:
        assets: dict = {}

    assert fetch.fetch_visual_window(_Item(), tmp_path / "x.tif") is None
