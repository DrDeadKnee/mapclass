"""
On-disk sample contract validator (PITFALL 3 / PITFALL 5 prevention setup).

assert_sample_valid(path) is the brownfield-to-mapclass boundary: it consumes
samples emitted by scripts/build_dataset.py and scripts/build_historical_dataset.py
and raises SampleContractError loudly on any schema drift.

Called once per training run at startup (NOT per-batch). Failure is loud — no
print-and-skip, no warnings (deviates from the historical pipeline's
public-boundary downgrade pattern; this is intentional per CONTEXT.md
Claude's Discretion).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from mapclass.data.taxonomy import LANDCOVER_CLASSES, TOPO_CLASSES, WATER_TOPO


class SampleContractError(ValueError):
    """Raised when a sample directory does not match the on-disk contract.

    Subclasses ValueError to match the brownfield convention
    (scripts/historical/label.py:57; scripts/augment.py:120 also raise ValueError).
    """


# Required per-sample files. Image filename varies by source: synthetic uses
# {flat,illustrated,satellite}.png; historical uses image.png.
_REQUIRED_FILES = ("land_cover.png", "topography.png", "sample_weights.json", "manifest.json")
_IMAGE_CANDIDATES = ("image.png", "flat.png", "illustrated.png", "satellite.png")
_MANIFEST_REQUIRED_KEYS = ("source", "source_subtype")


def assert_sample_valid(path: str | Path) -> None:
    """Validate a sample directory against the on-disk contract.

    Parameters
    ----------
    path : sample directory containing image*.png, land_cover.png, topography.png,
           sample_weights.json, manifest.json

    Raises
    ------
    SampleContractError
        On any of: missing files, image dimension mismatch, label values
        outside [0, num_classes) (modulo WATER_TOPO=255 and NODATA=255),
        manifest.json missing required keys, sample_weights.json malformed.

    Notes
    -----
    Honors WATER_TOPO=255 and NODATA=255 sentinels from scripts/label.py.
    Values 255 in label rasters are valid and ignored by the validator.
    """
    path = Path(path)
    if not path.is_dir():
        raise SampleContractError(f"sample dir does not exist: {path}")

    # 1. required files
    for fname in _REQUIRED_FILES:
        if not (path / fname).is_file():
            raise SampleContractError(f"sample {path.name}: missing {fname}")

    image_paths = [path / f for f in _IMAGE_CANDIDATES if (path / f).is_file()]
    if not image_paths:
        raise SampleContractError(
            f"sample {path.name}: missing image — expected one of {_IMAGE_CANDIDATES}"
        )

    # 2. image dim consistency
    label_lc = np.array(Image.open(path / "land_cover.png"))
    label_topo = np.array(Image.open(path / "topography.png"))
    img = np.array(Image.open(image_paths[0]).convert("RGB"))
    if img.shape[:2] != label_lc.shape[:2]:
        raise SampleContractError(
            f"sample {path.name}: image {img.shape[:2]} != land_cover {label_lc.shape[:2]}"
        )
    if img.shape[:2] != label_topo.shape[:2]:
        raise SampleContractError(
            f"sample {path.name}: image {img.shape[:2]} != topography {label_topo.shape[:2]}"
        )

    # 3. label range. Allowed: [0, len(LANDCOVER_CLASSES)) ∪ {255}; same for topo.
    n_lc = len(LANDCOVER_CLASSES)
    n_topo = len(TOPO_CLASSES)
    lc_max = int(label_lc.max())
    topo_max = int(label_topo.max())
    invalid_lc = (label_lc >= n_lc) & (label_lc != 255)
    invalid_topo = (label_topo >= n_topo) & (label_topo != WATER_TOPO)
    if invalid_lc.any():
        raise SampleContractError(
            f"sample {path.name}: land_cover has values outside [0, {n_lc}) ∪ {{255}}; max={lc_max}"
        )
    if invalid_topo.any():
        raise SampleContractError(
            f"sample {path.name}: topography has values outside [0, {n_topo}) ∪ {{{WATER_TOPO}}}; max={topo_max}"
        )

    # 4. manifest.json schema
    with open(path / "manifest.json") as f:
        manifest = json.load(f)
    for key in _MANIFEST_REQUIRED_KEYS:
        if key not in manifest:
            raise SampleContractError(f"sample {path.name}: manifest.json missing key '{key}'")

    # 5. sample_weights.json must parse + contain land_cover_weights and topography_weight
    with open(path / "sample_weights.json") as f:
        weights = json.load(f)
    for key in ("land_cover_weights", "topography_weight"):
        if key not in weights:
            raise SampleContractError(
                f"sample {path.name}: sample_weights.json missing key '{key}'"
            )
