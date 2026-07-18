"""
Offline regression tests for two 2026-07 field bugs found on the first
multi-tile hexprior build (iberia came back 1% valid):

1. fetch_worldcover mosaicking: rasterio.warp.reproject with dst_nodata
   re-initialises the WHOLE destination on every call, so each 3° tile wiped
   the previously fetched ones and only the last tile survived. Fixed with
   init_dest_nodata=False (same fix in dem.fetch_topo's tile loop).

2. coverage.build_summary latitude orientation: raster row 0 is the NORTH
   edge of a tile, so the 1° row blocks were latitude-mirrored within every
   3° tile (a Gulf-of-Mexico cell inherited Houston's class counts).

No network: synthetic tiles are served from rasterio MemoryFiles.
"""

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from historical import worldcover
from historical.worldcover import NODATA, fetch_worldcover
from hexprior.aggregate import open_grid_dataset
from satellite import coverage


def _synthetic_tile(bounds, arr):
    """Return an OPEN in-memory uint8 dataset over WGS84 ``bounds``."""
    h, w = arr.shape
    memfile = MemoryFile()
    ds = memfile.open(
        driver="GTiff", width=w, height=h, count=1, dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=from_bounds(*bounds, w, h),
    )
    ds.write(arr, 1)
    ds.close()
    return memfile.open()


def _serve_tiles(monkeypatch, module, tiles_by_name):
    """Patch ``module``'s rasterio.open to serve synthetic tiles by tile id."""
    real_open = rasterio.open

    def fake_open(url, *args, **kwargs):
        for name, ds in tiles_by_name.items():
            if isinstance(url, str) and name in url:
                return ds
        return real_open(url, *args, **kwargs)

    monkeypatch.setattr(module.rasterio, "open", fake_open)


class TestWorldcoverMosaic:
    def test_multi_tile_fetch_keeps_all_tiles(self, monkeypatch):
        """A bbox spanning two 3° tiles must retain data from BOTH."""
        trees = np.full((30, 30), 10, dtype=np.uint8)   # WC 10 → canonical 1
        water = np.full((30, 30), 80, dtype=np.uint8)   # WC 80 → canonical 0
        _serve_tiles(monkeypatch, worldcover, {
            "N00E000": _synthetic_tile((0, 0, 3, 3), trees),
            "N00E003": _synthetic_tile((3, 0, 6, 3), water),
        })

        bbox = (0.0, 0.0, 6.0, 3.0)
        with open_grid_dataset(bbox, grid_res_deg=0.1) as grid_ds:
            lc = fetch_worldcover(grid_ds, bbox)

        west_half = lc[:, : lc.shape[1] // 2]
        east_half = lc[:, lc.shape[1] // 2:]
        assert (west_half == 1).all(), "western tile was clobbered"
        assert (east_half == 0).all(), "eastern tile was clobbered"
        assert not (lc == NODATA).any()


class TestSummaryLatitudeOrientation:
    def test_north_strip_gets_northern_latitude(self, monkeypatch, tmp_path):
        """Row block 0 of a 3° tile is the strip at lat_sw + 2, not lat_sw."""
        arr = np.full((30, 30), 80, dtype=np.uint8)  # water everywhere ...
        arr[:10, :] = 10                             # ... except trees up north
        _serve_tiles(monkeypatch, coverage, {"N00E000": _synthetic_tile((0, 0, 3, 3), arr)})

        summary = coverage.build_summary(
            summary_path=tmp_path / "s.json", bbox=(0.0, 0.0, 3.0, 3.0), force=True,
        )
        by_lat = {c["lat"]: c["counts"] for c in summary["cells"] if c["lon"] == 0}
        trees_idx, water_idx = 1, 0
        assert by_lat[2][trees_idx] > 0 and by_lat[2][water_idx] == 0
        assert by_lat[0][water_idx] > 0 and by_lat[0][trees_idx] == 0
