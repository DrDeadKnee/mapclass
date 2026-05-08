# Phase 1: End-to-End Skeleton — Pattern Map

**Mapped:** 2026-05-08
**Files analyzed:** 22 new + 1 modified
**Analogs found:** 18 / 23 (5 files have NO close analog — pure greenfield, see "No Analog Found")

This phase introduces the new `mapclass/` package alongside the existing `scripts/` tree. The brownfield codebase is pure-functional, no classes, no logging framework, no config files, no formatter/linter, no tests, and uses bare `sys.path` mutation for cross-script imports. The new `mapclass/` package is a real Python package (with a real `__init__.py`) and CAN use modern Python idioms, but every file in this phase MUST inherit the existing conventions where they exist (snake_case, ALL_CAPS constants with `_underscore_prefix` for module-private, PEP-604/585 type hints on public surfaces, leading-`_` for private helpers, prose docstrings with NumPy-section blocks where contracts are non-obvious, `print()` not `logging`, sentinel constants like `WATER_TOPO=255` / `NODATA=255`).

The single sub-package that already exists as a real package is `scripts/historical/`; its `__init__.py` is intentionally empty (package marker only). `mapclass/__init__.py` will follow that same precedent.

---

## File Classification

| New / Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `mapclass/__init__.py` | package-marker | none | `scripts/historical/__init__.py` | exact |
| `mapclass/data/__init__.py` | package-marker | none | `scripts/historical/__init__.py` | exact |
| `mapclass/model/__init__.py` | package-marker | none | `scripts/historical/__init__.py` | exact |
| `mapclass/data/taxonomy.py` | re-export module | none | `scripts/biome_mapping.py` | exact |
| `mapclass/data/contract.py` | pure-data-validator | request-response (path → raise/return) | `scripts/historical/label.py:_to_wgs84_bbox` | partial (raise-on-contract-violation pattern) |
| `mapclass/data/loss_weights.py` | config-loader + validator | file-I/O (YAML → dataclass) | `scripts/historical/label.py` (HISTORICAL_LC_WEIGHTS dict shape) + `scripts/biome_mapping.py` (taxonomy linkage) | role-match |
| `mapclass/data/splits.py` | deterministic-split-builder | file-I/O (sample-id list → splits.json) | NONE — no splits / hashing in tree today | no analog |
| `mapclass/data/dataset.py` | torch.utils.data.Dataset adapter | file-I/O streaming (PNG → tensor) | `scripts/historical/label.py:_read_rgb` (per-sample image read) + `scripts/historical/dem.py:fetch_topo` (multi-source loop) | partial |
| `mapclass/model/backbone.py` | ABC interface | request-response (Tensor → dict[Tensor]) | NONE — no ABCs anywhere in tree | no analog (use ARCHITECTURE.md Pattern 3) |
| `mapclass/model/mock_backbone.py` | nn.Module concrete | request-response (Tensor → dict[Tensor]) | NONE — no torch.nn modules in tree yet | no analog (greenfield) |
| `mapclass/model/seg_heads.py` | nn.Module concrete (×2 heads) | request-response (dict[Tensor] → Tensor) | NONE — same as above | no analog (greenfield) |
| `mapclass/train.py` | CLI entrypoint + training loop | event-driven (DataLoader → optimizer step) | `scripts/build_historical_dataset.py` (argparse + per-item loop + error accumulation) | partial (CLI shape only; loop body is greenfield) |
| `mapclass/infer.py` | public API + CLI entrypoint | request-response (image → probabilities) | `scripts/historical/label.py:make_labels` (single-image processor + saves outputs) | partial (single-image shape only; model load is greenfield) |
| `mapclass/eval.py` | CLI entrypoint + report writer | batch (DataLoader → JSON report) | `scripts/build_historical_dataset.py:cmd_build` (loop + summary) + `scripts/historical/label.py` (JSON dump) | partial |
| `mapclass/seeding.py` | utility | request-response (int → side-effect on RNG) | NONE — no seed plumbing exists today (CONCERNS.md 8a) | no analog (greenfield, but see PITFALL 15 for spec) |
| `mapclass/configs/v0.yaml` | config (declarative) | static | NONE — no YAML in tree today | no analog (greenfield) |
| `mapclass/configs/EVAL-03_protocol.md` | documentation | static | `.planning/codebase/CONVENTIONS.md` (prose markdown contract) | partial (markdown shape only) |
| `mapclass/configs/loss_weights.yaml` | config (declarative) | static | `scripts/historical/label.py:HISTORICAL_LC_WEIGHTS` (the Python dict that this YAML generalises) | role-match |
| `mapclass/configs/splits.json` | data manifest | static | `data/historical/raw/unregistered_manifest.json` (existing JSON manifest convention; emitted by `scripts/historical/rumsey.py:emit_manifest`) | partial |
| `pyproject.toml` | build config | static | NONE — currently absent | no analog (greenfield, follow STACK.md template) |
| `requirements.lock.txt` | pinned deps | static | `requirements.txt` (existing unpinned source) | partial |
| `.python-version` | tool config | static | NONE | no analog (one-line file) |
| `README.md` (MODIFY) | project documentation | static | existing `README.md` | exact (in-place extension) |

---

## Pattern Assignments

### `mapclass/__init__.py`, `mapclass/data/__init__.py`, `mapclass/model/__init__.py` — package markers

**Analog:** `/home/drdreadknee/mapclass/scripts/historical/__init__.py` (empty file)

**Pattern:** Empty file. The historical sub-package marks itself with an empty `__init__.py` and does NOT re-export sibling modules. CONVENTIONS.md item "Barrel Files" notes: *"`scripts/historical/__init__.py` is empty. Sibling modules under `historical/` are imported individually as `from historical import rumsey` or `from historical.dem import fetch_topo` rather than re-exported through the package."*

**Action for Phase 1:** All three `__init__.py` files in `mapclass/` MUST be empty. Do NOT add `__all__`. Do NOT re-export. Consumers write `from mapclass.data.taxonomy import LANDCOVER_CLASSES`, not `from mapclass import LANDCOVER_CLASSES`. This matches the existing pattern exactly and minimises the chance of circular-import surprises during the Phase-2 SmolVLM swap.

---

### `mapclass/data/taxonomy.py` (re-export module)

**Analog:** `/home/drdreadknee/mapclass/scripts/biome_mapping.py` (canonical taxonomy source)

**Imports pattern** (`scripts/biome_mapping.py:1-8` — module docstring style):

```python
"""
Azgaar Fantasy Map Generator → canonical taxonomy mappings.

Azgaar biome IDs (0-21) and height values (h) are exported in GeoJSON
as properties 'biome' and 'height' respectively.
Height range: 0-19 = water, 20-100 = land.
"""
```

**Constants pattern** (`scripts/biome_mapping.py:35-52`):

```python
# Canonical 9-class land cover taxonomy (class index → name).
# Classes 4 (cropland), 5 (built_up), 7 (flooded_wetland) do not appear in
# synthetic data; they come from satellite-derived data only.
LANDCOVER_CLASSES = [
    "water",           # 0
    "trees",           # 1
    "shrubland",       # 2
    "grassland",       # 3
    "cropland",        # 4  — satellite only
    "built_up",        # 5  — satellite only
    "bare_sparse",     # 6
    "flooded_wetland", # 7  — satellite only
    "snow_ice",        # 8
]
LANDCOVER_IDX = {name: i for i, name in enumerate(LANDCOVER_CLASSES)}

TOPO_CLASSES = ["flat", "hilly", "mountainous"]
TOPO_IDX = {name: i for i, name in enumerate(TOPO_CLASSES)}
```

**Action for Phase 1:** `mapclass/data/taxonomy.py` is a *thin re-export shim*. It MUST NOT redefine these constants. CONTEXT.md `## Integration Points` is explicit: *"`mapclass.data.taxonomy` re-exports `scripts.biome_mapping.LANDCOVER_CLASSES` etc. — do NOT copy-paste; do NOT redefine."*

The shim looks like this (no analog for the import path itself — `scripts/` is not a real package, so `from scripts.biome_mapping import ...` won't work without the existing `sys.path` mutation hack from `scripts/build_historical_dataset.py:33-35`). Phase 1 must SOLVE this: either (a) add a `scripts/__init__.py` to make `scripts` an importable package, or (b) have `mapclass/data/taxonomy.py` include its own `sys.path` insertion and import by bare name.

**Recommended approach:** Approach (a) — add an empty `scripts/__init__.py`. This is a 1-byte change, follows the same pattern as `scripts/historical/__init__.py`, and unblocks `from scripts.biome_mapping import LANDCOVER_CLASSES, LANDCOVER_IDX, TOPO_CLASSES, TOPO_IDX, h_to_landcover, h_to_topo` cleanly. CONCERNS.md does not mark `scripts/` lacking `__init__.py` as a deliberate decision; CONVENTIONS.md item "Path Aliases" describes it as fragile.

**Taxonomy hash helper** (per CONTEXT.md `## Claude's Discretion` — "Taxonomy hash algorithm"): also lives here as a function:

```python
import hashlib
import json

def taxonomy_hash() -> str:
    """sha256 over the canonical taxonomy. Embedded in checkpoint metadata (TS-8)."""
    payload = json.dumps(
        {"land_cover": LANDCOVER_CLASSES, "topography": TOPO_CLASSES},
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()
```

**Why here:** PITFALL 3 prevention #2 ("Single canonical taxonomy module exports a frozen dict") + PITFALL 4 prevention #3 (taxonomy hash in metadata) both anchor on this file.

---

### `mapclass/data/contract.py` (pure-data-validator)

**Analog:** `/home/drdreadknee/mapclass/scripts/historical/label.py:_to_wgs84_bbox` (lines 54-73) — the only file in the tree that raises `ValueError` on a contract violation.

**Raise-on-contract-violation pattern** (`scripts/historical/label.py:54-58`):

```python
def _to_wgs84_bbox(ds: rasterio.DatasetReader) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) in WGS84 for a rasterio dataset."""
    if ds.crs is None:
        raise ValueError("Dataset has no CRS — was it properly georeferenced?")
```

**Public-boundary downgrade pattern** (`scripts/historical/label.py:108-113`):

```python
print(f"Processing: {map_geotiff.name}")

with rasterio.open(map_geotiff) as ds:
    if ds.crs is None:
        print(f"  ERROR: {map_geotiff.name} has no CRS — skipping")
        return
```

**JSON-output pattern for dumping per-sample manifests** (`scripts/historical/label.py:138-146`):

```python
weights = {
    "land_cover_weights": HISTORICAL_LC_WEIGHTS,
    "topography_weight": HISTORICAL_TOPO_WEIGHT,
    "source": "historical",
    "map_file": map_geotiff.name,
}
with open(output_dir / "sample_weights.json", "w") as f:
    json.dump(weights, f, indent=2)
```

**Sample-contract on-disk schema** (CONTEXT.md `## Claude's Discretion` — `assert_sample_valid()` scope):

The validator must check that the directory pointed to by `path` contains:
- `image.png` (or one of `flat.png|illustrated.png|satellite.png` for synthetic), `land_cover.png`, `topography.png` — uint8 PNGs at matching dimensions.
- `sample_weights.json` — dict with `land_cover_weights` and `topography_weight` (existing schema, see `scripts/historical/label.py:138-146`).
- `manifest.json` — NEW per ARCHITECTURE.md Pattern 1 (`source`, `source_subtype`, `source_id`, `registered_via`, `image_files`, `label_files`, `weights_file`, `bbox_wgs84`, `created`).
- Class-index validation: every land-cover pixel value ∈ `[0, 9) ∪ {255}`, every topography pixel value ∈ `[0, 3) ∪ {255}` (where `255` = `WATER_TOPO`/`NODATA`).
- `sample_weights.json` must have a row for the sample's `source_subtype` per `loss_weights.yaml` (cross-file validation).

**Action for Phase 1:** Phase 1 deviates from the brownfield "downgrade-to-print-and-return at public boundary" pattern. CONTEXT.md `## Claude's Discretion` is explicit: *"Failure raises a custom `SampleContractError` exception (loud — NOT a warning, NOT silent)."* This is intentional — `assert_sample_valid()` is the brownfield-to-mapclass boundary, and the loud failure is the schema-drift detector PITFALL 3 / PITFALL 5 prevention requires.

Define the exception in this file:

```python
class SampleContractError(ValueError):
    """Raised when a sample directory does not match the on-disk contract.

    Used at training startup (once per run, not per-batch) to catch schema drift
    between scripts/build_*.py output and mapclass.data.dataset's expectations.
    """
```

`SampleContractError` extends `ValueError` to match the brownfield convention (`scripts/historical/label.py:57` raises `ValueError`, `scripts/augment.py:120` raises `ValueError`); CONVENTIONS.md notes the codebase has no custom exception hierarchy. Subclassing `ValueError` is the lightest-weight extension.

**Validator function signature pattern** (matches `scripts/historical/label.py:95` `make_labels(map_geotiff: Path, output_dir: Path) -> None` — single-public-function-per-module):

```python
def assert_sample_valid(path: str | Path) -> None:
    """
    Validate a sample directory against the on-disk contract.

    Raises SampleContractError loudly if any check fails. No print-and-skip.
    Called once at training startup per sample, not per batch.

    Parameters
    ----------
    path : sample directory containing image*.png, land_cover.png, topography.png,
           sample_weights.json, manifest.json

    Notes
    -----
    Honors WATER_TOPO=255 and NODATA=255 sentinels from scripts/label.py and
    scripts/historical/dem.py — values 255 in label rasters are valid.
    """
```

---

### `mapclass/data/loss_weights.py` (config-loader + validator)

**Analog:** `/home/drdreadknee/mapclass/scripts/historical/label.py` (the existing `HISTORICAL_LC_WEIGHTS` table that this YAML generalises) + ARCHITECTURE.md Pattern 2 example.

**Existing dict-shape pattern** (`scripts/historical/label.py:38-51`):

```python
# Per-class land cover loss weights for the historical source.
# Keys match LANDCOVER_CLASSES from biome_mapping.py.
HISTORICAL_LC_WEIGHTS: dict[str, float] = {
    "water":           1.0,
    "trees":           0.3,   # significant deforestation / reforestation
    "shrubland":       0.7,
    "grassland":       0.7,
    "cropland":        0.15,  # agricultural change is dramatic
    "built_up":        0.1,   # medieval cities vs. modern footprints
    "bare_sparse":     1.0,
    "flooded_wetland": 0.5,
    "snow_ice":        1.0,
}
HISTORICAL_TOPO_WEIGHT: float = 1.0
```

This dict becomes the `rumsey_registered` row in the new `loss_weights.yaml` verbatim. The hand-aligned columnar comment style is preserved.

**Strict-validator pattern** (PITFALL 3 prevention strategy #1; no in-tree analog — this is the kind of guard PITFALL 3 says is missing today):

```python
from dataclasses import dataclass
from pathlib import Path
import hashlib

from mapclass.data.taxonomy import LANDCOVER_CLASSES, TOPO_CLASSES


@dataclass(frozen=True)
class LossWeights:
    """Per-source × per-class weights, loaded once from YAML at training startup.

    Schema is validated strictly on load (PITFALL 3 prevention) — every land-cover
    class in LANDCOVER_CLASSES must appear in every source row, with explicit
    0.0 for "absent" classes (not implicit default).

    See mapclass/configs/loss_weights.yaml for the canonical schema.
    """
    land_cover: dict[str, dict[str, float]]   # source_subtype -> class_name -> weight
    topography: dict[str, float]              # source_subtype -> weight
    source_class_weights_hash: str            # sha256 of the loaded YAML, embedded in checkpoint metadata (TS-8)

    @classmethod
    def load(cls, path: str | Path) -> "LossWeights":
        ...

    def lookup(self, source_subtype: str, lc_class_name: str) -> tuple[float, float]:
        """Returns (lc_weight, topo_weight)."""
        ...
```

**Validation rules** (PITFALL 3 prevention strategy #1):
1. Every `class_name` key in any source row MUST be a member of `LANDCOVER_CLASSES`. Unknown name → `LossWeightsSchemaError("unknown class 'foo' in source 'bar'; valid: {LANDCOVER_CLASSES}")`.
2. Every source row MUST list ALL 9 classes. Missing class → `LossWeightsSchemaError("source 'bar' missing class 'cropland' — explicit 0.0 required, no implicit default")`.
3. Every `inherit:` directive (e.g. `rumsey_bootstrapped` inheriting from `rumsey_registered`) MUST resolve to another source in the same file.
4. `topography` value per source MUST be a single float ∈ `[0.0, 1.0]` (matching `HISTORICAL_TOPO_WEIGHT: float = 1.0` shape).

**Source-class-weights hash** (TS-8 / Phase-1 success criterion 1): computed as `sha256(yaml_text.encode())`, embedded in safetensors metadata header at `mapclass.train` checkpoint write time.

**Module-private helper pattern** (`scripts/biome_mapping.py:90-94`, `scripts/label.py:28`): use leading-underscore for any internal validation helpers; the public surface is `LossWeights`, `LossWeights.load`, `LossWeights.lookup`, `LossWeightsSchemaError`.

**Exception name convention** (per `SampleContractError` in `mapclass/data/contract.py`): `LossWeightsSchemaError(ValueError)` — same parent class as `SampleContractError` for a uniform "validator failures are `ValueError` subclasses" convention.

---

### `mapclass/data/splits.py` (deterministic-split-builder)

**Analog:** NONE in tree (no splits, no hashing today). Closest reference: PITFALL 5 prevention strategy #1 ("Split by sample-ID hash, not glob order").

**Splits.json structure** (CONTEXT.md `## Claude's Discretion`):

```json
{
  "version": 1,
  "by_split": {
    "train": ["luna_4567", "luna_4568", "synth_0001", ...],
    "val":   ["luna_4571", ...],
    "test":  ["luna_4583", ...]
  },
  "by_id_hash_bucket": {
    "luna_4567": 3,
    "synth_0001": 7,
    ...
  }
}
```

**Determinism pattern** (PITFALL 5 prevention #1):

```python
import hashlib

def _bucket_for_sample_id(sample_id: str) -> int:
    """sha256 hash → bucket 0..9. Pure function of sample ID, immune to glob order."""
    h = hashlib.sha256(sample_id.encode()).hexdigest()
    return int(h[:8], 16) % 10  # first 8 hex chars → 32-bit int → bucket 0..9


def build_splits(sample_ids: list[str]) -> dict:
    """Buckets 0-7 → train, 8 → val, 9 → test."""
    ...
```

**Eval-time overlap assertion** (PITFALL 5 prevention #3): Phase 1's `mapclass.eval` MUST assert that the `test` split sample-IDs are disjoint from the in-memory train-source manifest at startup. Failure raises a `SplitsContaminationError(ValueError)` (matching the validator-error naming convention).

**JSON write/read** (matching `scripts/historical/label.py:145-146` style with `indent=2`):

```python
with open(splits_path, "w") as f:
    json.dump(splits, f, indent=2)
```

**Action for Phase 1:** `mapclass/configs/splits.json` is committed once at v0 build time (PITFALL 5 prevention #2 — "Held-out IDs locked at v0 build time"). v1 (Phase 4) and EVAL-03 (Phase 5) read the same file. The seeding utility's `training_seed` is NOT a parameter to `_bucket_for_sample_id` — the hash is content-only (sample_id), so the split is replicable from sample IDs alone, not from a seed.

---

### `mapclass/data/dataset.py` (torch.utils.data.Dataset adapter)

**Analog:** `/home/drdreadknee/mapclass/scripts/historical/label.py` (the per-sample image read pattern) + `scripts/historical/dem.py:fetch_topo` (the iterate-and-accumulate pattern).

**Single-sample image read pattern** (`scripts/historical/label.py:76-92`):

```python
def _read_rgb(ds: rasterio.DatasetReader) -> np.ndarray:
    """Read the first three bands as (H, W, 3) uint8."""
    n = min(ds.count, 3)
    bands = ds.read(list(range(1, n + 1)))  # (C, H, W)
    if bands.dtype != np.uint8:
        # Normalise to uint8 range
        lo, hi = bands.min(), bands.max()
        if hi > lo:
            bands = ((bands - lo) / (hi - lo) * 255).astype(np.uint8)
        else:
            bands = bands.astype(np.uint8)
    arr = np.moveaxis(bands, 0, -1)  # (H, W, C)
    ...
```

**Action for Phase 1:** `mapclass/data/dataset.py` is greenfield torch code. Copy the path-handling style from `scripts/historical/label.py:51-54` and the docstring NumPy-section format from `make_labels` (lines 96-103). Use `PIL.Image.open(...).convert("RGB")` and `np.array(...)` rather than `rasterio` (rasterio is the historical pipeline's load tool, not a training-time dep).

```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mapclass.data.contract import assert_sample_valid


class MapClassDataset(Dataset):
    """PyTorch Dataset over samples emitted by scripts/build_*.py.

    Parameters
    ----------
    sample_ids : list of sample IDs (matches splits.json "by_split" entries)
    root       : root directory containing per-sample subdirs (e.g. data/historical/dataset)
    validate   : if True, run assert_sample_valid() once per sample at __init__
    """
    def __init__(self, sample_ids: list[str], root: str | Path, validate: bool = True):
        self.root = Path(root)
        self.sample_ids = sample_ids
        if validate:
            for sid in sample_ids:
                assert_sample_valid(self.root / sid)
```

**Source-subtype handling** (Anti-Pattern 1 from ARCHITECTURE.md — "Mixing source identity into the model"): the `__getitem__` method returns a dict that includes `source_subtype` (read from `manifest.json`) so the loss-weight lookup happens at the trainer level, NOT in the model forward pass.

---

### `mapclass/model/backbone.py` (`Backbone` ABC)

**Analog:** NONE in tree (no ABCs anywhere in the codebase; CONVENTIONS.md notes "No custom classes are defined anywhere in `scripts/`"). Anchor: ARCHITECTURE.md Pattern 3 verbatim.

**Pattern from ARCHITECTURE.md Pattern 3 + PITFALL 4 prevention #1 ("Backbone owns the processor"):**

```python
"""
Backbone interface — swap surface for the small VL backbone.

The Backbone ABC is the only place that knows about the specific VL model.
Concrete subclasses (MockBackbone in Phase 1; SmolVLMBackbone in Phase 2;
PaliGemma2Backbone in Phase 5) wrap one specific model and expose a uniform
extract_features + preprocess interface.

PITFALL 4 (CRITICAL): the Backbone owns preprocess() — never re-implement
torchvision.transforms.Compose at the training-loop or inference-path level.
The processor identity hash is embedded in checkpoint metadata; load_model
refuses on mismatch.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from PIL.Image import Image


class Backbone(ABC):
    """Swap surface for the small VL backbone.

    Subclasses MUST set image_size and feature_channels as class attributes,
    and MUST implement extract_features() and preprocess().
    """

    image_size: int                        # input resolution (e.g. 384 for SmolVLM, 224 for PaliGemma)
    feature_channels: dict[str, int]       # stage_name -> channel count, e.g. {"features": 768}
    processor_identity: str                # short identifier embedded in checkpoint metadata (TS-8)

    @abstractmethod
    def preprocess(self, image: "Image | torch.Tensor") -> "torch.Tensor":
        """Convert a PIL.Image or HxWx3 uint8 tensor to the backbone's expected input.

        Owns resize, normalize, padding, dtype. The training loop and inference path
        both call this — never re-implement preprocessing externally.
        """
        ...

    @abstractmethod
    def extract_features(self, image: "torch.Tensor") -> "dict[str, torch.Tensor]":
        """Returns a dict of feature maps. Caller is the seg-head / OCR-head.

        Phase 1 (MockBackbone): single key "features", single tensor.
        Phase 2+ (SmolVLMBackbone): same shape; multi-stage punted to Phase 2 plan.
        """
        ...
```

**Phase-2 lock-in:** D-02 in CONTEXT.md confirms Phase 1 returns a single key. Phase 2's plan must explicitly include the multi-stage refactor — flagged loudly in PATTERNS.md so the planner picks it up.

---

### `mapclass/model/mock_backbone.py` (`MockBackbone` learnable conv stub)

**Analog:** NONE in tree (no torch.nn modules exist yet). Anchor: D-01 / D-03 in CONTEXT.md.

**Pattern (greenfield; D-01 spec):**

```python
import torch
from torch import nn

from mapclass.model.backbone import Backbone


class MockBackbone(Backbone, nn.Module):
    """A tiny learnable conv stub. Phase 1 only.

    ~10–100k parameters (1–2 conv layers RGB → features). Training actually
    backprops through it; gradient flow exercises the full optimizer + scheduler
    + mixed-precision path. Eval NLL may improve modestly over the
    class-frequency prior — that is a positive diagnostic, NOT a ship metric.

    Phase 2 swaps this for SmolVLMBackbone via the same Backbone interface.

    See CONTEXT.md D-01..D-04 for the design decisions:
      - D-01: learnable, not random (real gradient path tested)
      - D-02: single-key feature dict (multi-stage deferred to Phase 2)
      - D-03: init seeded from training_seed (PITFALL 15 prevention)
      - D-04: backbone='mock' tag in safetensors metadata (discipline-only)
    """

    image_size: int = 384  # match SmolVLM's input so Phase-2 swap is trivial
    feature_channels: dict[str, int] = {"features": 64}
    processor_identity: str = "mock-v1"

    def __init__(self, training_seed: int):
        nn.Module.__init__(self)
        # D-03: seed before parameter init so two runs with the same seed produce
        # byte-identical weights.
        from mapclass.seeding import set_global_seed
        set_global_seed(training_seed)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

    def preprocess(self, image) -> torch.Tensor:
        # Match SmolVLM's normalize stats roughly so Phase-2 doesn't change input space.
        ...

    def extract_features(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = torch.relu(self.conv1(image))
        x = torch.relu(self.conv2(x))
        return {"features": x}
```

**Type-hint convention** (CONVENTIONS.md "modern type hints"): public surface fully annotated; helpers can be relaxed.

---

### `mapclass/model/seg_heads.py` (two dense heads)

**Analog:** NONE in tree. Anchor: ARCHITECTURE.md Pattern 3 example (`scripts/biome_mapping.py:LANDCOVER_CLASSES` count = 9 for `LandCoverHead`; `TOPO_CLASSES` count = 3 for `TopographyHead`).

**Pattern (greenfield; consume Backbone's feature dict):**

```python
import torch
from torch import nn

from mapclass.data.taxonomy import LANDCOVER_CLASSES, TOPO_CLASSES


class LandCoverHead(nn.Module):
    """Dense seg head emitting logits over the 9-class land cover taxonomy.

    Forward: dict[str, Tensor] (from Backbone.extract_features) → Tensor [B, 9, H, W].

    Phase 1 consumes the single-key feature dict (D-02). Phase 2 must refactor to
    consume multi-stage skip connections — flagged loudly in Phase 2 plan.
    """

    def __init__(self, in_channels: int, num_classes: int = len(LANDCOVER_CLASSES)):
        super().__init__()
        ...

    def forward(self, feats: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feats["features"]  # Phase 1: single key per D-02
        ...


class TopographyHead(nn.Module):
    """Dense seg head emitting logits over the 3-class topography taxonomy.

    Forward: dict[str, Tensor] → Tensor [B, 3, H, W].

    The WATER_TOPO=255 sentinel from scripts/label.py is NOT predicted — it is
    overlaid at inference time from the land-cover head's argmax==water position.
    See mapclass/infer.py for the overlay logic.
    """
    def __init__(self, in_channels: int, num_classes: int = len(TOPO_CLASSES)):
        super().__init__()
        ...
```

**Critical sentinel** (`scripts/label.py:25`):

```python
WATER_TOPO = 255  # sentinel: water cells have no topography class
```

The TopographyHead does NOT learn a 4th class for water; it learns 3 classes only. At inference, `mapclass/infer.py` overlays `WATER_TOPO=255` at every pixel where the land-cover head's argmax == `LANDCOVER_IDX["water"]`. This matches the existing convention in `scripts/historical/dem.py:_classify` (lines 86-92):

```python
def _classify(slope: np.ndarray, water_mask: np.ndarray) -> np.ndarray:
    topo = np.zeros(slope.shape, dtype=np.uint8)
    topo[slope >= FLAT_MAX_DEG] = 1
    topo[slope >= HILLY_MAX_DEG] = 2
    topo[np.isnan(slope)] = WATER_TOPO
    topo[water_mask] = WATER_TOPO   # ← the overlay pattern
    return topo
```

---

### `mapclass/train.py` (CLI entrypoint + training loop)

**Analog:** `/home/drdreadknee/mapclass/scripts/build_historical_dataset.py` (the only multi-subcommand CLI in tree, lines 124-164) — for the `argparse` shape and the per-item-loop-with-error-accumulation pattern.

**CLI pattern** (`scripts/build_historical_dataset.py:124-156`):

```python
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble historical map training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    ...
    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args.raw_dir, args.max_maps)
```

**Action for Phase 1:** `mapclass/train.py` does NOT need subcommands (it has a single mode: train), so a flat `argparse` is fine — no `add_subparsers`. But the `RawDescriptionHelpFormatter` + module-docstring-as-epilog pattern is preserved:

```python
"""
Train a GeoViLM model end-to-end.

Usage:
    python -m mapclass.train --config mapclass/configs/v0.yaml [--seed 42]

Reads samples emitted by scripts/build_*.py, validates them via
mapclass.data.contract.assert_sample_valid, loads loss weights from
mapclass.data.loss_weights.LossWeights.load, and writes a safetensors checkpoint
with embedded metadata (model_version, dataset_manifest_sha, taxonomy_hash,
training_seed, source_class_weights_hash) per Phase 1 success criterion 1.
"""

import argparse
from pathlib import Path

# Stdlib (block 1)
# Third-party (block 2)
# First-party (block 3)
from mapclass.data.taxonomy import taxonomy_hash
from mapclass.data.loss_weights import LossWeights
from mapclass.data.contract import assert_sample_valid
from mapclass.data.dataset import MapClassDataset
from mapclass.model.mock_backbone import MockBackbone
from mapclass.model.seg_heads import LandCoverHead, TopographyHead
from mapclass.seeding import set_global_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a GeoViLM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to YAML training config (e.g. mapclass/configs/v0.yaml)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Training seed (single value, no --no-seed override; PITFALL 15)")
    args = parser.parse_args()

    set_global_seed(args.seed)
    ...


if __name__ == "__main__":
    main()
```

**Phase-progress logging pattern** (matching `scripts/build_historical_dataset.py:49,92`):

```python
print("=== Train: Phase 1 (Mock backbone) ===")
print(f"  Config: {args.config}")
print(f"  Seed:   {args.seed}")
```

**Per-batch logging is two-space indented** (matches `scripts/historical/label.py:108-149`):

```python
print(f"  Epoch {ep}/{n_epochs} step {step}: loss={loss.item():.4f}")
```

**Safetensors checkpoint write** (Phase 1 success criterion 1, TS-2 + TS-8):

```python
from safetensors.torch import save_file

metadata = {
    "model_version": "0.1.0",
    "dataset_manifest_sha": dataset_manifest_sha,
    "taxonomy_hash": taxonomy_hash(),
    "training_seed": str(args.seed),                         # safetensors metadata is str-only
    "source_class_weights_hash": loss_weights.source_class_weights_hash,
    "backbone": "mock",                                       # D-04: discipline-only tag
    "processor_identity": backbone.processor_identity,        # PITFALL 4 prevention #3
}
save_file(state_dict, ckpt_path, metadata=metadata)
```

**Critical: safetensors metadata is `dict[str, str]`** — every value must be a string. Encode ints/dicts as JSON strings before saving (PITFALL "Integration Gotchas" table from PITFALLS.md: *"safetensors metadata: String-only values; complex objects need JSON-encoding"*).

---

### `mapclass/infer.py` (public API + CLI entrypoint)

**Analog:** `/home/drdreadknee/mapclass/scripts/historical/label.py:make_labels` (single-image-in / artifacts-out shape).

**Single-image processor pattern** (`scripts/historical/label.py:95-149`):

```python
def make_labels(map_geotiff: Path, output_dir: Path) -> None:
    """
    Produce image.png, land_cover.png, topography.png, and sample_weights.json
    for a single georeferenced historical map GeoTIFF.

    Parameters
    ----------
    map_geotiff : path to a georeferenced GeoTIFF (WMS download or QGIS export)
    output_dir  : directory to write outputs (created if it does not exist)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    map_geotiff = Path(map_geotiff)

    print(f"Processing: {map_geotiff.name}")
    ...
```

**Action for Phase 1 / TS-3 contract** (FEATURES.md inference module API surface):

```python
"""
Inference module — the stable contract surface to the downstream hex-grid app.

Usage:
    python -m mapclass.infer <checkpoint> <image>

    or as a Python API (Phase 1 minimal; Phase 6 finalises):

        from mapclass.infer import load_model
        predictor = load_model("models/geovilm_v0.pt")
        result = predictor.predict(image)

Returns a dict:
    {
      "land_cover": float16 [9, H, W] — softmax probabilities, sums to 1.0 ± fp16 epsilon
      "topography": float16 [3, H, W] — softmax probabilities; pixels where
                                        land_cover argmax == water carry
                                        WATER_TOPO=255 sentinel via overlay.
      "model_version": str,
      "taxonomy": {"land_cover": [9 names], "topography": [3 names]},
    }

CRITICAL (Anti-Pattern 4): mapclass.infer MUST NOT import anything from
mapclass.data. The inference module's deps are model + numpy + PIL + safetensors only
(no rasterio, no torch.utils.data). Phase 6's startup-import test enforces this.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from mapclass.data.taxonomy import LANDCOVER_CLASSES, LANDCOVER_IDX, TOPO_CLASSES, taxonomy_hash
from mapclass.model.mock_backbone import MockBackbone
from mapclass.model.seg_heads import LandCoverHead, TopographyHead


def load_model(checkpoint: str | Path, device: str = "auto", dtype: str = "float16"):
    """Load a trained GeoViLM checkpoint.

    Reads metadata (model_version, dataset_manifest_sha, taxonomy_hash, ...) from
    the safetensors header. Refuses to load on taxonomy_hash mismatch (PITFALL 4
    prevention #3 + Phase 1 success criterion 4 corollary).
    """
    ...


def predict(model, image) -> dict:
    """See module docstring for the contract."""
    ...
```

**WATER_TOPO overlay pattern** (anchored on `scripts/historical/dem.py:_classify` and `scripts/label.py:WATER_TOPO=255`): after softmax on both heads, take `argmax(land_cover_probs, axis=0) == LANDCOVER_IDX["water"]` as the water mask, and mark those pixels in the topography output as `WATER_TOPO=255` in the integer label form (the float-prob form keeps its 3-class softmax).

**Module-private constants** (matching `scripts/historical/label.py:36`):

```python
_WATER_CLASS_IDX = LANDCOVER_IDX["water"]   # 0
```

---

### `mapclass/eval.py` (CLI entrypoint + report writer)

**Analog:** `/home/drdreadknee/mapclass/scripts/build_historical_dataset.py:cmd_build` (lines 82-117) — loop + error accumulation + summary; `scripts/historical/label.py:138-146` for JSON writing.

**Loop + summary pattern** (`scripts/build_historical_dataset.py:92-117`):

```python
print(f"=== Building labels for {len(tifs)} map(s) ===")
out_dir.mkdir(parents=True, exist_ok=True)

errors: list[tuple[Path, Exception]] = []

if workers == 1:
    for tif in tifs:
        _, exc = _process_one(tif, out_dir)
        if exc:
            print(f"  ERROR {tif.name}: {exc}")
            errors.append((tif, exc))
else:
    ...

print(f"\nBuild complete.")
print(f"  Success: {len(tifs) - len(errors)}/{len(tifs)}")
```

**Action for Phase 1:** `mapclass/eval.py` mirrors this shape but instead of accumulating errors per sample, it accumulates per-source per-class NLL into a dict and writes `eval_report.json` at the end. The eval report schema (CONTEXT.md `## Claude's Discretion`):

```python
report = {
    "model_version": ...,                    # from checkpoint metadata
    "checkpoint": str(ckpt_path),
    "split": "test",
    "splits_json_sha": ...,                  # sha256 of mapclass/configs/splits.json
    "by_source": {
        "synthetic_azgaar": {
            "by_class": {"water": 0.83, "trees": 0.91, ...},  # per-class NLL
            "mean": 1.04,
            "n_samples": 80,
        },
        # ... other sources ...
    },
    "overall_mean_nll": 1.07,
    "calibration": None,                     # Phase-2+ extension (TS-6)
    "notes": "Phase 1 / Mock backbone — numbers near log(num_classes) expected",
}
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
```

**CLI shape** (mirror `mapclass/train.py`):

```python
parser.add_argument("--split", choices=["train", "val", "test", "heldout"], default="heldout",
                    help="Which sample-id-hash split to evaluate on (heldout = test)")
parser.add_argument("--checkpoint", type=Path, required=True)
```

**Eval-time overlap assertion** (PITFALL 5 prevention #3) — runs at startup, before any forward pass:

```python
def _assert_no_split_leakage(splits_path: Path, train_manifest_paths: list[Path]) -> None:
    """Fail loudly if a test sample-id appears in any training source's manifest."""
    ...
```

---

### `mapclass/seeding.py` (utility)

**Analog:** NONE (CONCERNS.md 8a — no seeds anywhere). Anchor: PITFALL 15 prevention #1.

**Greenfield pattern (~15 lines per PITFALL 15):**

```python
"""
Project-wide seeding utility (PITFALL 15 prevention).

Single seed value, no --no-seed override flag. Called at every entry point
(mapclass.train, mapclass.eval, mapclass.infer in test mode). Pins random,
numpy, torch, torch.cuda, cudnn deterministic + benchmark off.

Phase 1 lands this; Phase 2+ inherits.
"""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed every RNG that affects training/inference reproducibility.

    Note: full bit-determinism costs ~10% throughput (cudnn.deterministic=True).
    Documented in research/PITFALLS.md PITFALL 15 prevention #1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
```

**No CLI override** (CONTEXT.md D-03 + PITFALL 15): `mapclass/train.py --seed N` defaults to `42`; there is no `--no-seed` flag. Seed value is mandatory and non-overridable in this sense.

---

### `mapclass/configs/v0.yaml` (training config)

**Analog:** NONE in tree (no YAML files in the codebase). Anchor: STACK.md "OmegaConf + dataclass-typed configs".

**Greenfield pattern (per OmegaConf + Phase 1 success criterion 1):**

```yaml
# mapclass/configs/v0.yaml — Phase 1 (Mock backbone) training config.
#
# Phase 1 is the synthetic-only smoke run. Phase 2 reuses this filename pattern
# but introduces v0_thin.yaml for SmolVLM, and v0.yaml gets re-written at Phase 3
# for the full v0 (3-source) run.

run_name: phase1_mock_smoke
seed: 42                                    # consumed by mapclass.seeding (PITFALL 15)
data:
  root: data/renders                         # synthetic-only for Phase 1
  splits_path: mapclass/configs/splits.json  # PITFALL 5: committed deterministic splits
  loss_weights_path: mapclass/configs/loss_weights.yaml
  num_samples: 100                           # CONTEXT.md "Skeleton-run dataset size"
  batch_size: 8
model:
  backbone: mock                             # D-04: tagged in checkpoint metadata
training:
  n_epochs: 2                                # smoke run, NOT a ship metric
  lr: 1e-3
  optimizer: adam
checkpoint:
  out_path: models/geovilm_phase1_mock.pt
```

**Constants commented at end-of-line** (matching `scripts/biome_mapping.py:38-48` style — hand-aligned columnar comments).

---

### `mapclass/configs/EVAL-03_protocol.md` (one-page bellwether protocol)

**Analog:** `.planning/codebase/CONVENTIONS.md` and similar prose-markdown docs.

**Required content (CONTEXT.md D-05, D-06; ROADMAP.md Phase 1 success criterion 6):**

Markdown sections, one page, human-readable:
1. Held-out split hash (sha256 of `mapclass/configs/splits.json`).
2. Loss-weights file hash (sha256 of `mapclass/configs/loss_weights.yaml`).
3. Augmentation policy (the rotation/jitter/color/parchment policy strings; not a code import).
4. Fine-tune budget shape (epochs × batch-size × LR-schedule descriptor — not the literal numbers, the *shape*).
5. Evaluation seed (the literal int).
6. Kill-switch criterion: D-06 + D-08 — "the small backbone ships if NLL gap to PaliGemma is within `DECIDE_AT_PHASE_5`% relative on the held-out splits, otherwise reconsider MODEL-01."

**Critical literal — must commit verbatim** (D-08 / PITFALL 2 prevention):

```
The decision threshold is `DECIDE_AT_PHASE_5`% relative NLL gap.

This placeholder MUST be replaced with a numeric threshold in a SEPARATE commit
before eval_03_compare.py runs. Phase 5's eval_03_compare.py reads this file at
startup and refuses to run (raises RuntimeError) if the literal string
`DECIDE_AT_PHASE_5` is still present.
```

Phase 5's `eval_03_compare.py` (NOT a Phase 1 deliverable) will grep for this exact string. The grep must succeed against this file content as committed in Phase 1.

---

### `mapclass/configs/loss_weights.yaml` (canonical per-source × per-class table)

**Analog:** ARCHITECTURE.md Pattern 2 example + `scripts/historical/label.py:HISTORICAL_LC_WEIGHTS` (the existing dict that becomes one row).

**Pattern (Phase 1 has only `synthetic_azgaar` and `rumsey_registered` rows; Phase 2 adds `rumsey_bootstrapped` placeholder; Phase 3 adds `road_osm`):**

```yaml
# mapclass/configs/loss_weights.yaml
#
# Per-source × per-class loss weights. Schema validated strictly at training
# startup by mapclass.data.loss_weights.LossWeights.load (PITFALL 3 prevention).
# Every source row MUST list ALL 9 land-cover classes with explicit floats —
# no missing keys, no implicit defaults.
#
# Initial values for rumsey_registered are copied verbatim from
# scripts/historical/label.py:HISTORICAL_LC_WEIGHTS (lines 40-50).

sources:
  synthetic_azgaar:
    land_cover:
      water:           1.0
      trees:           1.0
      shrubland:       1.0
      grassland:       1.0
      cropland:        0.0   # absent from synthetic; explicit 0.0 (no implicit default)
      built_up:        0.0   # absent from synthetic
      bare_sparse:     1.0
      flooded_wetland: 0.0   # absent from synthetic
      snow_ice:        1.0
    topography: 1.0

  rumsey_registered:
    # Initial values from scripts/historical/label.py:HISTORICAL_LC_WEIGHTS
    land_cover:
      water:           1.0
      trees:           0.3   # significant deforestation / reforestation
      shrubland:       0.7
      grassland:       0.7
      cropland:        0.15  # agricultural change is dramatic
      built_up:        0.1   # medieval cities vs. modern footprints
      bare_sparse:     1.0
      flooded_wetland: 0.5
      snow_ice:        1.0
    topography: 1.0
```

**Hand-aligned columns** preserved from `scripts/historical/label.py` style.

**No Phase 1 row for `rumsey_bootstrapped` or `road_osm`** — those land in Phases 4 and 3 respectively. The strict validator (`LossWeights.load`) does NOT enforce that all 4 sources are present; only that for every source listed, all 9 classes are listed.

---

### `mapclass/configs/splits.json` (committed deterministic split manifest)

**Analog:** `data/historical/raw/unregistered_manifest.json` (existing manifest convention; emitted by `scripts/historical/rumsey.py:emit_manifest` — though that file is gitignored, the JSON format style is the precedent).

**JSON write style** (matching `scripts/historical/label.py:145-146`):

```python
with open(splits_path, "w") as f:
    json.dump(splits, f, indent=2)
```

**Phase 1 content:** generated from a one-shot script (`scripts/build_splits.py` is OUT OF SCOPE for Phase 1 — splits.json is generated *inline* by `mapclass.data.splits.build_splits()` at Phase 1 commit time, then committed to the repo). The actual content depends on which sample IDs exist when the file is first generated; the structure is the locked deliverable, not the IDs themselves.

---

### `pyproject.toml` (NEW build config)

**Analog:** NONE in tree. Anchor: STACK.md "Installation" section + ARCHITECTURE.md "Cloud-VM SSH workflow contract" (must support `pip install -e .`).

**Greenfield template (setuptools-backed, minimal — STACK.md "smallest viable choice"):**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mapclass"
version = "0.1.0"
description = "GeoViLM training, evaluation, and inference for stylized cartography"
requires-python = ">=3.10"
license = {file = "LICENSE"}
readme = "README.md"
dependencies = [
  # Core ML (per research/STACK.md minimums)
  "torch>=2.4,<3.0",
  "transformers>=4.47,<5.0",
  "accelerate>=0.34",
  "peft>=0.11",
  "bitsandbytes>=0.43",
  "safetensors>=0.4.5",
  # Existing data pipeline (already in requirements.txt)
  "Pillow",
  "numpy",
  "rasterio",
  "pyproj",
  "requests",
  # New for Phase 1 (config / logging / metrics)
  "omegaconf>=2.3",
  "tensorboard>=2.18",
  "pyyaml>=6.0.2",
  "tqdm>=4.66",
  # Phase-3+ deps (declared early to avoid dep-resolver thrash later):
  # python-doctr[torch]>=1.0.1, torchmetrics>=1.4, scikit-image>=0.24,
  # opencv-python-headless>=4.10  — uncomment when Phase 3 lands
]

[tool.setuptools.packages.find]
include = ["mapclass*"]
```

**Action for Phase 1:** Phase 1 picks the *minimum* dep set needed for the Mock-backbone smoke run. STACK.md's full `requirements.in` is for Phase 3+. Phase 1 declares only what `mapclass.train|infer|eval` (Mock) actually imports; PITFALL 19 prevention #4 ("Pin a Python version") covered by `requires-python = ">=3.10"`.

**`requirements.txt` (existing) supersession:** CONTEXT.md `## Existing Code Insights` says: *"`requirements.txt` can stay as a developer convenience or be deleted — Phase 1 planner's call."* Recommended action: DELETE `requirements.txt` to avoid drift with `pyproject.toml`. The Phase 1 planner can also keep it as a one-line `-e .` redirect if SSH-VM workflow expects it.

---

### `requirements.lock.txt` (frozen lockfile)

**Analog:** existing `/home/drdreadknee/mapclass/requirements.txt` (the unpinned source).

**Generation pattern** (per STACK.md "Installation"): `uv pip compile pyproject.toml -o requirements.lock.txt`

**Format:** standard pip-compile output (one line per package with `==` exact pins, plus `--hash=` lines if compiled with `--generate-hashes`). Phase 1 doesn't require hashes — just exact pins are enough to satisfy PITFALL 19.

---

### `.python-version`

**Analog:** NONE. One-line file: `3.10` (matches `pyproject.toml requires-python`).

---

### `README.md` (MODIFY)

**Analog:** existing `/home/drdreadknee/mapclass/README.md`.

**Action for Phase 1:** ADD a new section with the new entrypoint commands. CONTEXT.md `## Canonical References` is explicit: *"Phase 1's new entrypoints (`python -m mapclass.train|infer|eval`) get added to README at Phase 1's commit time."* Do NOT delete or restructure existing README content — only append a "Training the model" or similar new section near the existing scripts section (`README.md` lines mentioning `scripts/build_dataset.py`).

Add cross-link to the VM-provisioning repo per PITFALL 19 prevention #2 ("Cross-link the VM-provisioning repo from this repo's README"); CONCERNS.md 8c also recommends this.

**Add OSM attribution placeholder** (PITFALL 16 prevention #2 — though OSM source lands in Phase 3, the README's "Attribution" section is added in Phase 1 with a placeholder for OSM and existing entries for Rumsey, Copernicus DEM, ESA WorldCover, Azgaar).

---

## Shared Patterns

### Pattern A — Module docstring with usage block (apply to ALL new .py files)

**Source:** `/home/drdreadknee/mapclass/scripts/historical/label.py:1-21`, `/home/drdreadknee/mapclass/scripts/build_historical_dataset.py:1-24`

**Apply to:** Every `.py` file in `mapclass/` except the empty `__init__.py` files.

```python
"""
<one-line purpose>.

<one or two paragraphs of context, including REQ-IDs from REQUIREMENTS.md
and pitfall numbers from research/PITFALLS.md when the file mitigates one>.

Usage:
    <runnable command if applicable, otherwise import example>

<additional context: outputs, file format, sentinels, contracts>
"""
```

---

### Pattern B — ASCII section dividers in long files

**Source:** `/home/drdreadknee/mapclass/scripts/build_historical_dataset.py:41-43, 69-71, 120-122` (multi-section CLI orchestrator); `scripts/historical/rumsey.py:59-61, 188-190, 279-281, 359-361`.

```python
# ---------------------------------------------------------------------------
# <Section name>
# ---------------------------------------------------------------------------
```

**Apply to:** Files > 150 lines (CONVENTIONS.md guideline). Likely candidates: `mapclass/train.py`, `mapclass/eval.py`, `mapclass/infer.py`, `mapclass/data/dataset.py`, `mapclass/data/loss_weights.py`.

---

### Pattern C — Sentinel constants

**Source:**
- `scripts/label.py:24-25` — `NODATA = 255  # fill value for pixels not covered by any cell` and `WATER_TOPO = 255  # sentinel: water cells have no topography class`
- `scripts/historical/dem.py:43` — `WATER_TOPO = 255`
- `scripts/historical/worldcover.py:56` — `NODATA = 255`

**Apply to:** `mapclass/data/contract.py` (must accept `255` in label rasters as valid), `mapclass/data/dataset.py` (must mask `255` from loss computation), `mapclass/model/seg_heads.py` (TopographyHead does NOT predict a 4th class for water — it learns 3 only; `255` is overlaid at inference time), `mapclass/infer.py` (overlay logic at predict time).

**Re-export rule:** Define these in `mapclass.data.taxonomy` once (alongside the re-exported `LANDCOVER_CLASSES`) so all four files import from the same place. NEVER redefine `WATER_TOPO = 255` in multiple `mapclass/` files — that's a PITFALL-3-style schema-drift hazard.

---

### Pattern D — Validator-Error exception hierarchy

**Source:** `scripts/historical/label.py:57` (`raise ValueError(...)`), `scripts/augment.py:120` (`raise ValueError("images list is empty")`).

**Apply to:** All Phase 1 validators. Define each as a subclass of `ValueError`:

```python
class SampleContractError(ValueError):
    """Raised by mapclass.data.contract.assert_sample_valid on schema drift."""

class LossWeightsSchemaError(ValueError):
    """Raised by mapclass.data.loss_weights.LossWeights.load on schema drift."""

class SplitsContaminationError(ValueError):
    """Raised by mapclass.eval at startup if test split overlaps train manifest."""

class TaxonomyHashMismatchError(ValueError):
    """Raised by mapclass.infer.load_model on taxonomy_hash mismatch (PITFALL 4)."""
```

Single, uniform parent class (`ValueError`) — minimal departure from the brownfield convention.

---

### Pattern E — Print-based progress logging

**Source:** `scripts/build_historical_dataset.py:49,92,112` (`=== <phase> ===` banners + tail summary), `scripts/historical/label.py:108-149` (per-item indented progress with two-space prefix, `→` arrows for input/output).

**Apply to:** `mapclass/train.py` (per-epoch banner + per-step indented log), `mapclass/eval.py` (per-source banner + per-class indented log), `mapclass/infer.py` (single banner + result line).

CONVENTIONS.md is firm: NO `logging` module, NO `warnings.warn`, NO custom logger. Just `print()` + Unicode `→` and `…` glyphs. TS-7 (TensorBoard logging for loss curves) is the *machine-readable* layer; print-statements are the *human-skim* layer.

---

### Pattern F — Three-block import organisation

**Source:** Every Python file in `scripts/` (CONVENTIONS.md "Import Organization" section).

```python
# Block 1: stdlib
import argparse
import json
from pathlib import Path

# Block 2: third-party
import numpy as np
import torch
from PIL import Image

# Block 3: first-party (mapclass.* now)
from mapclass.data.taxonomy import LANDCOVER_CLASSES
from mapclass.data.contract import assert_sample_valid
```

Blank line between blocks. `from X import a, b` style preferred over qualified `X.a` when used multiple times.

---

### Pattern G — Type-hint discipline

**Source:** CONVENTIONS.md "Type Hints" section + every `make_labels`/`fetch_*` signature.

```python
def some_public_function(path: str | Path, *, validate: bool = True) -> SomeReturn:
    """..."""
    path = Path(path)
    ...

def _internal_helper(x, y) -> float:   # private, may skip annotations on primitives
    return x + y
```

PEP 604 (`X | None`) and PEP 585 (`list[...]`, `dict[...]`) — no `Optional`, no `List`. Forward references for third-party types as string literals (`"torch.Tensor"`, `"PIL.Image.Image"`) where the import would be heavy.

---

### Pattern H — JSON file format (indent=2)

**Source:** `scripts/historical/label.py:145-146`:

```python
with open(output_dir / "sample_weights.json", "w") as f:
    json.dump(weights, f, indent=2)
```

**Apply to:** `mapclass/configs/splits.json`, `mapclass/eval.py`'s `eval_report.json`, any committed JSON.

---

## No Analog Found

Files with no close match in the existing codebase. Planner should use research patterns (ARCHITECTURE.md, STACK.md, PITFALLS.md, FEATURES.md) rather than tree analogs:

| File | Role | Data Flow | Reason | Anchor Reference |
|------|------|-----------|--------|------------------|
| `mapclass/data/splits.py` | deterministic-split-builder | file-I/O | No split logic, no hashing, no `splits.json` exists today | PITFALL 5 prevention #1; CONTEXT.md `## Claude's Discretion` "Splits.json structure" |
| `mapclass/model/backbone.py` | ABC interface | request-response | No ABCs anywhere in tree (`scripts/` is purely function-oriented) | ARCHITECTURE.md Pattern 3; PITFALL 4 prevention #1 |
| `mapclass/model/mock_backbone.py` | nn.Module concrete | request-response | No `torch.nn` modules in tree yet | CONTEXT.md D-01..D-04 |
| `mapclass/model/seg_heads.py` | nn.Module concrete | request-response | Same as above | ARCHITECTURE.md "Key Abstractions" + Pattern 3 example |
| `mapclass/seeding.py` | utility | side-effect | CONCERNS.md 8a — no seeds anywhere in repo | PITFALL 15 prevention #1 (15-line spec) |
| `mapclass/configs/v0.yaml` | config | static | No YAML files in tree | STACK.md "OmegaConf + dataclass-typed configs" |
| `mapclass/configs/EVAL-03_protocol.md` | documentation | static | No multi-phase protocol docs in tree | CONTEXT.md D-05..D-08 |
| `pyproject.toml` | build config | static | Currently absent; CONCERNS.md 8c flagged | STACK.md "Installation" template |
| `.python-version` | tool config | static | Currently absent | STACK.md `Development Tools` |

For these files, the planner's plan-task descriptions should reference the research-doc anchor (e.g. "implement `set_global_seed` per PITFALL 15 prevention strategy #1, ~15 lines") rather than pointing at a code analog.

---

## Cross-cutting Gotchas to Pin in Plans

These are not patterns but are load-bearing facts the planner must surface in plan-action sections so executors don't trip over them:

1. **`scripts/__init__.py` does NOT exist.** `mapclass/data/taxonomy.py` cannot do `from scripts.biome_mapping import ...` until Phase 1 adds an empty `scripts/__init__.py` (1-byte change; matches the pattern of `scripts/historical/__init__.py`). Without it, the only working path is `sys.path.insert(...)` like `scripts/build_historical_dataset.py:33-35`. Recommended: add the `__init__.py`. Alternative: keep `mapclass.data.taxonomy` self-contained by re-exporting via `sys.path` mutation — but this perpetuates a fragility CONVENTIONS.md flagged.

2. **Safetensors metadata is `dict[str, str]`, not `dict[str, Any]`.** Every value (training_seed int, model_version, taxonomy_hash, source_class_weights_hash, dataset_manifest_sha, processor_identity, backbone tag) MUST be JSON-serialised to a string before saving and JSON-parsed on load. Reference: PITFALLS.md "Integration Gotchas" table — "String-only values; complex objects need JSON-encoding".

3. **`mapclass.infer` MUST NOT import `mapclass.data`.** ARCHITECTURE.md Anti-Pattern 4. Phase 6 will add a startup-import test (`python -c "import mapclass.infer; assert 'mapclass.data' not in sys.modules"`); Phase 1 plan tasks must not introduce this coupling preemptively.

4. **The `WATER_TOPO=255` sentinel is overlaid at inference time, not predicted.** TopographyHead emits 3 classes only. `mapclass/infer.py` reads `argmax(land_cover) == LANDCOVER_IDX["water"]` and writes `WATER_TOPO=255` into the integer-label output (the float-prob output keeps 3-class softmax). Reference: `scripts/historical/dem.py:_classify` lines 86-92.

5. **Mock backbone must produce a visibly decreasing loss curve in the smoke run.** CONTEXT.md `## Specific Ideas` says: *"a Mock-trained run should produce a loss curve that visibly decreases (even if the absolute NLL is poor), confirming end-to-end gradient flow is intact. Phase 1 planner should add a Mock-NLL-decreases assertion to the smoke run."* This is a Phase 1 success criterion in disguise — bake it into a plan task.

6. **`DECIDE_AT_PHASE_5` is committed verbatim** in `mapclass/configs/EVAL-03_protocol.md`. Phase 5's `eval_03_compare.py` (out of Phase 1 scope) greps for this exact string. The Phase 1 file must contain this literal — do NOT replace it with `<TBD>`, `XXX`, `???`, or any other placeholder.

7. **The two `__init__.py` files in `mapclass/data/` and `mapclass/model/` are intentionally empty.** Do NOT add `__all__`, do NOT re-export. CONVENTIONS.md "Module Design / Barrel Files" + `scripts/historical/__init__.py:1` (single empty file).

8. **Phase 1 Mock backbone single-key feature dict (D-02) creates a Phase-2 refactor debt.** Seg heads in Phase 1 do `feats["features"]`. Phase 2 will need `feats["stage_2"]`, `feats["stage_3"]`, etc. Surface this in the Phase 2 plan scope explicitly — add a "TODO(phase-2): refactor seg heads for multi-stage feature dict per ARCHITECTURE.md Pattern 3" comment in `mapclass/model/seg_heads.py` so it's not forgotten.

9. **CONCERNS.md 8a/8b/8c are mitigated by Phase 1 deliverables** — `mapclass/seeding.py` (8a no seeds), checkpoint metadata + `splits.json` + `manifest.json` per sample (8b no dataset versioning), `pyproject.toml` + `requirements.lock.txt` + `.python-version` (8c no lockfile). Plan tasks should explicitly reference these concern numbers in their action sections so the brownfield-fix lineage is auditable.

---

## Metadata

**Analog search scope:**
- `/home/drdreadknee/mapclass/scripts/` (8 modules)
- `/home/drdreadknee/mapclass/scripts/historical/` (5 modules + `__init__.py`)
- `/home/drdreadknee/mapclass/requirements.txt`, `/home/drdreadknee/mapclass/README.md`
- `/home/drdreadknee/mapclass/.planning/research/{ARCHITECTURE,STACK,FEATURES,PITFALLS}.md`
- `/home/drdreadknee/mapclass/.planning/codebase/{STRUCTURE,CONVENTIONS,CONCERNS}.md`

**Files scanned:** 13 source modules, 1 requirements file, 7 research/codebase docs, 1 CONTEXT.md, 1 ROADMAP.md, 1 REQUIREMENTS.md.

**Pattern extraction date:** 2026-05-08
