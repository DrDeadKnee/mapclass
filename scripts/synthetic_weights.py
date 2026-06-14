"""
Per-source loss weights for the SYNTHETIC (Azgaar + Pillow) dataset stream.

The dict *shape* and the ``topography_weight`` are LOCKED to mirror the
historical writer (``scripts/historical/label.py``):

    {"land_cover_weights": {<class>: float, ...},
     "topography_weight": float,
     "source": "synthetic",
     "map_file": "<name>"}

Synthetic labels are exact ground truth by construction — they are rasterised
directly from the Azgaar source, so no temporal-drift discount applies (unlike
the historical stream, where ``HISTORICAL_LC_WEIGHTS`` down-weights classes
subject to centuries of land-use change). Every per-class weight is therefore
``1.0``. Class-imbalance correction is deferred to the Phase 3/4
DataLoader/sampler, not folded into this per-source weight layer.

Keys are the exact 9 canonical LANDCOVER_CLASSES names from
``scripts/biome_mapping.py`` (set-equal to ``HISTORICAL_LC_WEIGHTS`` keys).
"""

import json
from pathlib import Path

# Uniform 1.0 — synthetic labels are exact by construction (user-approved
# 2026-05-15; plan 02-03 checkpoint:decision option "uniform").
SYNTHETIC_LC_WEIGHTS: dict[str, float] = {
    "water":           1.0,
    "trees":           1.0,
    "shrubland":       1.0,
    "grassland":       1.0,
    "cropland":        1.0,
    "built_up":        1.0,
    "bare_sparse":     1.0,
    "flooded_wetland": 1.0,
    "snow_ice":        1.0,
}
SYNTHETIC_TOPO_WEIGHT: float = 1.0


def write_sample_weights(output_dir: str | Path, map_file: str) -> Path:
    """Write the locked-shape ``sample_weights.json`` into ``output_dir``.

    Returns the written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "land_cover_weights": SYNTHETIC_LC_WEIGHTS,
        "topography_weight": SYNTHETIC_TOPO_WEIGHT,
        "source": "synthetic",
        "map_file": map_file,
    }
    out_path = output_dir / "sample_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)
    return out_path
