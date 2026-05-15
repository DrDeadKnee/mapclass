"""
Offline tests for the satellite fetch + weights modules (plan 02-04).

No network/S3: the COG read in ``fetch.fetch_visual_window`` is exercised
against a local UTM GeoTIFF written under ``tmp_path``; weights are pure data.
"""

import ast
import json
from pathlib import Path

import numpy as np
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
