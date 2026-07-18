"""
Composite terrain class space for hex soft labels.

The prior model predicts a distribution over COMPOSITE classes — the joint
(land cover, topography) space, collapsed where the joint is degenerate:

  index 0        : water              (land cover 0; topography undefined)
  index 1..24    : (lc, topo) for lc in 1..8, topo in 0..2
                   index = 1 + (lc - 1) * 3 + topo

25 classes total. Pixels with NODATA land cover (255), or land cover with
NODATA topography (DEM gap over land), are invalid and excluded from hex
counts. Land-cover / topography indices match scripts/biome_mapping.py and
the WorldCover / DEM fetchers in scripts/historical/.
"""

import numpy as np

from biome_mapping import LANDCOVER_CLASSES, TOPO_CLASSES

NODATA = 255           # matches historical.worldcover.NODATA / dem.WATER_TOPO
N_COMPOSITE = 1 + (len(LANDCOVER_CLASSES) - 1) * len(TOPO_CLASSES)  # 25
INVALID = -1

# Human-readable names, index-aligned with the composite space.
COMPOSITE_NAMES: list[str] = ["water"] + [
    f"{lc}/{topo}"
    for lc in LANDCOVER_CLASSES[1:]
    for topo in TOPO_CLASSES
]


def composite_index(lc: int, topo: int) -> int:
    """
    Map a (land_cover, topography) pair to its composite class index, or
    INVALID for NODATA / inconsistent pairs.
    """
    if lc == 0:
        return 0  # water — topography ignored (DEM tags water as NODATA)
    if lc == NODATA or topo == NODATA or not (1 <= lc <= 8) or not (0 <= topo <= 2):
        return INVALID
    return 1 + (lc - 1) * 3 + topo


def composite_from_rasters(lc: np.ndarray, topo: np.ndarray) -> np.ndarray:
    """
    Vectorised composite_index over uint8 land-cover / topography rasters
    (as returned by fetch_worldcover / fetch_topo). Returns int16 array of
    composite indices with INVALID where either input is NODATA.
    """
    lc = lc.astype(np.int16)
    topo = topo.astype(np.int16)
    out = np.full(lc.shape, INVALID, dtype=np.int16)

    water = lc == 0
    out[water] = 0

    land = (lc >= 1) & (lc <= 8) & (topo >= 0) & (topo <= 2)
    out[land] = 1 + (lc[land] - 1) * 3 + topo[land]
    return out
