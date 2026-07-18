"""
Offline tests for scripts/hexprior/: composite label space, grid geometry,
H3 cell assignment, and hex count aggregation. No network, no S3.
"""

import numpy as np
import pytest

from hexprior.labels import (
    COMPOSITE_NAMES,
    INVALID,
    N_COMPOSITE,
    NODATA,
    composite_from_rasters,
    composite_index,
)
from hexprior.aggregate import (
    aggregate_counts,
    cells_for_grid,
    grid_shape,
    open_grid_dataset,
    pixel_centers,
)


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

class TestCompositeIndex:
    def test_water_ignores_topo(self):
        # DEM tags water as NODATA topo; must still be class 0
        assert composite_index(0, NODATA) == 0
        assert composite_index(0, 1) == 0

    def test_land_classes_bijective(self):
        seen = set()
        for lc in range(1, 9):
            for topo in range(3):
                idx = composite_index(lc, topo)
                assert 1 <= idx < N_COMPOSITE
                seen.add(idx)
        assert len(seen) == N_COMPOSITE - 1  # all 24 land composites distinct

    def test_nodata_invalid(self):
        assert composite_index(NODATA, 0) == INVALID
        assert composite_index(3, NODATA) == INVALID  # DEM gap over land

    def test_names_aligned(self):
        assert len(COMPOSITE_NAMES) == N_COMPOSITE
        assert COMPOSITE_NAMES[0] == "water"
        assert COMPOSITE_NAMES[composite_index(1, 2)] == "trees/mountainous"

    def test_vectorised_matches_scalar(self):
        lc = np.array([[0, 1, 8], [NODATA, 4, 3]], dtype=np.uint8)
        topo = np.array([[NODATA, 2, 0], [0, NODATA, 1]], dtype=np.uint8)
        out = composite_from_rasters(lc, topo)
        expected = np.array([
            [composite_index(0, NODATA), composite_index(1, 2), composite_index(8, 0)],
            [INVALID, INVALID, composite_index(3, 1)],
        ], dtype=np.int16)
        np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# grid geometry
# ---------------------------------------------------------------------------

BBOX = (4.0, 50.0, 5.0, 51.0)  # 1° box over land (matches tiny_geotiff extent)


class TestGrid:
    def test_grid_shape(self):
        h, w = grid_shape(BBOX, grid_res_deg=0.01)
        assert (h, w) == (100, 100)

    def test_pixel_centers_orientation_and_bounds(self):
        lats, lons = pixel_centers(BBOX, grid_res_deg=0.01)
        assert lats[0] > lats[-1]  # row 0 = north edge (raster convention)
        assert lats[0] == pytest.approx(51.0 - 0.005)
        assert lats[-1] == pytest.approx(50.0 + 0.005)
        assert lons[0] == pytest.approx(4.0 + 0.005)
        assert lons[-1] == pytest.approx(5.0 - 0.005)

    def test_open_grid_dataset_matches_geometry(self):
        with open_grid_dataset(BBOX, grid_res_deg=0.01) as ds:
            assert (ds.height, ds.width) == (100, 100)
            assert ds.crs.to_epsg() == 4326
            w, s, e, n = ds.bounds
            assert (w, s, e, n) == pytest.approx(BBOX)


# ---------------------------------------------------------------------------
# H3 cell assignment (h3 is a pure local dependency — no network)
# ---------------------------------------------------------------------------

class TestCells:
    def test_cells_valid_and_contiguous(self):
        cells = cells_for_grid(BBOX, h3_res=5, grid_res_deg=0.02)
        assert cells.dtype == np.uint64
        assert cells.shape == (50, 50)
        # A 1° box at res 5 (~250 km² hexes) spans multiple but not many cells
        n_unique = np.unique(cells).size
        assert 10 <= n_unique <= 200
        # Adjacent pixels mostly share a cell (spatial coherence); at 0.02°
        # sampling a res-5 hex is ~8 px across, so ~1/8 of steps cross an edge
        same_as_right = (cells[:, :-1] == cells[:, 1:]).mean()
        assert same_as_right > 0.8

    def test_resolution_monotonic(self):
        coarse = np.unique(cells_for_grid(BBOX, h3_res=4, grid_res_deg=0.05)).size
        fine = np.unique(cells_for_grid(BBOX, h3_res=6, grid_res_deg=0.05)).size
        assert fine > coarse


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

class TestAggregate:
    def test_counts_and_coverage(self):
        cells = np.array([7, 7, 7, 9, 9], dtype=np.uint64)
        comp = np.array([0, 0, 4, INVALID, 1], dtype=np.int16)
        unique_cells, counts, n_pixels = aggregate_counts(cells, comp)

        np.testing.assert_array_equal(unique_cells, [7, 9])
        assert counts.shape == (2, N_COMPOSITE)
        assert counts[0, 0] == 2 and counts[0, 4] == 1 and counts[0].sum() == 3
        # invalid pixel excluded from counts but included in n_pixels
        assert counts[1, 1] == 1 and counts[1].sum() == 1
        np.testing.assert_array_equal(n_pixels, [3, 2])

    def test_soft_label_normalisation(self):
        cells = np.full(15, 42, dtype=np.uint64)
        comp = np.array([0] * 5 + [10] * 10, dtype=np.int16)  # 1/3 water, 2/3 class 10
        _, counts, _ = aggregate_counts(cells, comp)
        dist = counts[0] / counts[0].sum()
        assert dist[0] == pytest.approx(1 / 3)
        assert dist[10] == pytest.approx(2 / 3)

    def test_all_invalid_cell(self):
        cells = np.array([1, 1], dtype=np.uint64)
        comp = np.array([INVALID, INVALID], dtype=np.int16)
        unique_cells, counts, n_pixels = aggregate_counts(cells, comp)
        assert counts.sum() == 0
        np.testing.assert_array_equal(n_pixels, [2])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            aggregate_counts(np.zeros(3, dtype=np.uint64),
                             np.zeros(4, dtype=np.int16))

    def test_2d_inputs(self):
        cells = np.array([[5, 5], [5, 6]], dtype=np.uint64)
        comp = np.array([[0, 0], [1, 0]], dtype=np.int16)
        unique_cells, counts, n_pixels = aggregate_counts(cells, comp)
        np.testing.assert_array_equal(unique_cells, [5, 6])
        assert counts[0, 0] == 2 and counts[0, 1] == 1
        assert counts[1, 0] == 1


# ---------------------------------------------------------------------------
# end-to-end (offline): synthetic rasters through build/save path
# ---------------------------------------------------------------------------

class TestSaveRegion:
    def test_save_and_reload_local(self, tmp_path):
        from hexprior.build_hex_dataset import save_region

        unique_cells = np.array([11, 22], dtype=np.uint64)
        counts = np.zeros((2, N_COMPOSITE), dtype=np.uint32)
        counts[0, 0] = 10
        counts[1, 3] = 7
        n_pixels = np.array([10, 8], dtype=np.uint32)

        dest = save_region(str(tmp_path), "testregion", BBOX,
                           unique_cells, counts, n_pixels,
                           h3_res=6, grid_res_deg=0.003)

        loaded = np.load(dest)
        np.testing.assert_array_equal(loaded["cells"], unique_cells)
        np.testing.assert_array_equal(loaded["counts"], counts)
        np.testing.assert_array_equal(loaded["n_pixels"], n_pixels)

        import json
        meta = json.loads((tmp_path / "testregion_res6.meta.json").read_text())
        assert meta["n_cells"] == 2
        assert meta["composite_classes"] == COMPOSITE_NAMES
