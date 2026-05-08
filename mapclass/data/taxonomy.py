"""
Canonical taxonomy re-export shim for the mapclass package.

Re-exports LANDCOVER_CLASSES, LANDCOVER_IDX, TOPO_CLASSES, TOPO_IDX from
scripts.biome_mapping (the validated taxonomy source) and the sentinel
constants WATER_TOPO=255, NODATA=255 from scripts.label. Adds taxonomy_hash()
for embedding in safetensors checkpoint metadata (PITFALL 4 prevention #3).

DO NOT redefine the taxonomy here — this is the single source of truth in
mapclass-land, and it points at the brownfield source. PITFALL 3 prevention #2.
"""

import hashlib
import json

from scripts.biome_mapping import (
    LANDCOVER_CLASSES,
    LANDCOVER_IDX,
    TOPO_CLASSES,
    TOPO_IDX,
    h_to_landcover,
    h_to_topo,
)
from scripts.label import WATER_TOPO, NODATA


def taxonomy_hash() -> str:
    """sha256 over the canonical taxonomy. Embedded in checkpoint metadata (TS-8)."""
    payload = json.dumps(
        {"land_cover": LANDCOVER_CLASSES, "topography": TOPO_CLASSES},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
