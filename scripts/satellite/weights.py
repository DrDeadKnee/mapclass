"""
Per-source loss weights for the satellite (Sentinel-2 + ESA WorldCover) family.

This module mirrors the locked ``sample_weights.json`` contract emitted by
``historical.label.make_labels`` (see ``historical/label.py:138-147``). The
historical pipeline freezes its own ``HISTORICAL_LC_WEIGHTS``; the satellite
stream supplies its OWN dict of *identical shape* (same 9 ``LANDCOVER_CLASSES``
keys, a single ``topography_weight`` float, ``source`` / ``map_file``).

Approved decision: **balance-tilt** (checkpoint:decision, resolved 2026-05-15).

Rationale (locked — do NOT re-derive or re-tune these constants):
  * WorldCover labels are *contemporaneous* with the Sentinel-2 imagery, so the
    historical temporal-drift discount does NOT apply here — values stay near
    1.0 unless there is a class-balance reason to move them.
  * ``trees`` and ``water`` are globally over-represented (most of the Earth's
    land surface is forest or water), so they are discounted to 0.6 to keep the
    loss from being dominated by the easy majority classes.
  * ``cropland``, ``built_up`` and ``flooded_wetland`` are synthetic-absent
    (the synthetic Azgaar source cannot produce them) and PROJECT.md explicitly
    up-weights them; satellite is their PRIMARY source, so they are boosted to
    1.3.
  * All remaining classes (``shrubland``, ``grassland``, ``bare_sparse``,
    ``snow_ice``) and the topography channel stay at the neutral 1.0.
"""

import json
from pathlib import Path

# Per-class land cover loss weights for the satellite source.
# Keys MUST match LANDCOVER_CLASSES from scripts/biome_mapping.py exactly
# (same 9 keys as historical.label.HISTORICAL_LC_WEIGHTS).
SATELLITE_LC_WEIGHTS: dict[str, float] = {
    "water":           0.6,   # globally over-represented — discount
    "trees":           0.6,   # globally over-represented — discount
    "shrubland":       1.0,
    "grassland":       1.0,
    "cropland":        1.3,   # synthetic-absent, PROJECT.md up-weighted
    "built_up":        1.3,   # synthetic-absent, PROJECT.md up-weighted
    "bare_sparse":     1.0,
    "flooded_wetland": 1.3,   # synthetic-absent, PROJECT.md up-weighted
    "snow_ice":        1.0,
}
SATELLITE_TOPO_WEIGHT: float = 1.0


def write_sample_weights(output_dir: Path, map_file: str) -> Path:
    """
    Write ``sample_weights.json`` for a satellite per-map directory using the
    locked dict shape (matching ``historical.label`` line 138-147) with
    ``"source": "satellite"``.

    Parameters
    ----------
    output_dir : the per-map directory (created if it does not exist)
    map_file   : the source GeoTIFF file name recorded in the manifest

    Returns
    -------
    Path to the written ``sample_weights.json``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = {
        "land_cover_weights": SATELLITE_LC_WEIGHTS,
        "topography_weight": SATELLITE_TOPO_WEIGHT,
        "source": "satellite",
        "map_file": map_file,
    }
    out_path = output_dir / "sample_weights.json"
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)
    return out_path
