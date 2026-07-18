"""
Aggregate per-pixel composite terrain classes into H3 hex soft labels.

The sampling grid is a plain WGS84 lat/lon raster over a region bbox at
~300 m spacing (grid_res_deg = 0.003): fine enough for hundreds of samples
per res-6 hex (~36 km²), coarse enough that a 3°×3° region is ~1M pixels.

Counts (not normalised distributions) are stored per hex so regions can be
merged and confidence-weighted later; normalisation happens at training
time. Cells are stored as H3 uint64 ints.
"""

from contextlib import contextmanager

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

try:  # h3-py v4: int-based API avoids 1M-pixel str churn
    from h3.api import basic_int as h3
except ImportError:  # pragma: no cover
    import h3  # type: ignore[no-redef]

from hexprior.labels import INVALID, N_COMPOSITE

DEFAULT_H3_RES = 6        # ~36 km² per hex, ~7 km across — the "10 km tile"
DEFAULT_GRID_RES_DEG = 0.003  # ~330 m at the equator

_WGS84 = CRS.from_epsg(4326)


def grid_shape(bbox: tuple[float, float, float, float],
               grid_res_deg: float = DEFAULT_GRID_RES_DEG) -> tuple[int, int]:
    """Return (height, width) of the sampling grid covering bbox."""
    west, south, east, north = bbox
    width = max(1, round((east - west) / grid_res_deg))
    height = max(1, round((north - south) / grid_res_deg))
    return height, width


@contextmanager
def open_grid_dataset(bbox: tuple[float, float, float, float],
                      grid_res_deg: float = DEFAULT_GRID_RES_DEG):
    """
    Yield an open in-memory rasterio dataset defining the WGS84 sampling
    grid over bbox — the ``map_ds`` target that fetch_worldcover / fetch_topo
    reproject into.
    """
    west, south, east, north = bbox
    height, width = grid_shape(bbox, grid_res_deg)
    transform = from_bounds(west, south, east, north, width, height)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="uint8",
            crs=_WGS84,
            transform=transform,
        ) as ds:
            yield ds


def pixel_centers(bbox: tuple[float, float, float, float],
                  grid_res_deg: float = DEFAULT_GRID_RES_DEG
                  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (lats, lons) 1-D arrays of the grid's row / column pixel centres.
    lats descend (row 0 is the northern edge), matching raster convention.
    """
    west, south, east, north = bbox
    height, width = grid_shape(bbox, grid_res_deg)
    lat_step = (north - south) / height
    lon_step = (east - west) / width
    lats = north - lat_step * (np.arange(height) + 0.5)
    lons = west + lon_step * (np.arange(width) + 0.5)
    return lats, lons


def cells_for_grid(bbox: tuple[float, float, float, float],
                   h3_res: int = DEFAULT_H3_RES,
                   grid_res_deg: float = DEFAULT_GRID_RES_DEG) -> np.ndarray:
    """
    Assign every grid pixel to its H3 cell. Returns a uint64 array of shape
    (height, width).

    h3-py has no vectorised latlng_to_cell, so this is a Python loop over
    rows — ~1M scalar calls for a 3°×3° region, a few seconds.
    """
    lats, lons = pixel_centers(bbox, grid_res_deg)
    out = np.empty((lats.size, lons.size), dtype=np.uint64)
    lls = lons.tolist()
    to_cell = h3.latlng_to_cell
    for i, lat in enumerate(lats.tolist()):
        out[i, :] = [to_cell(lat, lon, h3_res) for lon in lls]
    return out


def aggregate_counts(cells: np.ndarray, composite: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Aggregate per-pixel composite classes into per-hex class counts.

    Parameters
    ----------
    cells : uint64 array, H3 cell id per pixel (any shape)
    composite : int16 array, same shape — composite class or INVALID

    Returns
    -------
    (unique_cells, counts, n_pixels)
      unique_cells : (n,) uint64, sorted
      counts       : (n, N_COMPOSITE) uint32 — valid-pixel class counts
      n_pixels     : (n,) uint32 — total pixels per cell incl. invalid,
                     for coverage/edge filtering downstream
    """
    if cells.shape != composite.shape:
        raise ValueError(f"shape mismatch: cells {cells.shape} vs composite {composite.shape}")
    cells = cells.ravel()
    composite = composite.ravel()

    unique_cells, inverse = np.unique(cells, return_inverse=True)
    n = unique_cells.size

    n_pixels = np.bincount(inverse, minlength=n).astype(np.uint32)

    valid = composite != INVALID
    counts = np.zeros((n, N_COMPOSITE), dtype=np.uint32)
    if valid.any():
        # flat index = cell_index * N_COMPOSITE + class
        flat = inverse[valid] * N_COMPOSITE + composite[valid].astype(np.int64)
        binned = np.bincount(flat, minlength=n * N_COMPOSITE)
        counts += binned.reshape(n, N_COMPOSITE).astype(np.uint32)

    return unique_cells, counts, n_pixels
