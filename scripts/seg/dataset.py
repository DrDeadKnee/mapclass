"""
PyramidDataset: Phase-2 nested-pyramid tile reader for dense segmentation (D-06).

Strategy:
  - Accepts one or more pyramid directories (each containing ``pyramid.json``),
    i.e. the OUTPUTS of ``scripts/tiling.tile()``.
  - Indexes tiles by their stable IDs from the manifest — never re-derives paths
    from coordinates (D-04: manifest-driven geometry).
  - Returns dict[str, Tensor | dict] per tile: image (3,H,W float32), land_cover
    (H,W long), topography (H,W long), sample_weights (raw dict, byte-identical
    to the pyramid's sample_weights.json — no re-normalisation; Phase-4 owns
    weighting).
  - EVAL-01 split-safety: constructor asserts and enforces that no indexed path
    contains a ``test/`` component (mirrors test_tiling.py:213-215 leakage guard).
    Constructor raises ``ValueError`` if the index is empty after filtering.
  - NEVER calls ``glob("**/pyramid.json")`` across a synthetic root (Pitfall 6).
  - NO training loop, optimizer, or backward() call (D-06 construction-only).

Requires:
  pip install torch pillow

Usage:
  from seg.dataset import PyramidDataset
  ds = PyramidDataset(pyramid_dir)           # single pyramid directory
  ds = PyramidDataset([pyr_dir_a, pyr_dir_b])  # multiple pyramid directories

Covered decisions: D-04, D-06, D-claude-discretion (batching discretion), EVAL-01.
"""

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class PyramidDataset(Dataset):
    """
    torch.utils.data.Dataset over one or more Phase-2 nested-pyramid directories.

    Each sample in the dataset corresponds to one tile (image/land_cover/topography
    triplet) inside a pyramid.  The sample dict returned by ``__getitem__`` is:

      {
        "image":          (3, H, W) float32 Tensor, normalised to [0, 1]
        "land_cover":     (H, W)    long    Tensor  (0-indexed class label)
        "topography":     (H, W)    long    Tensor  (0-indexed class label)
        "sample_weights": dict  — byte-identical to the pyramid's sample_weights.json
      }

    EVAL-01 guarantee: no indexed path contains a ``test/`` component.  This is
    enforced at construction time; it is also the caller's responsibility to pass
    only ``train/``-subtree roots (D-17).

    Raises ``ValueError`` if the resolved index is empty (ToonDataset idiom,
    finetune_paligemma.py lines 93-94) or if any indexed path contains a ``test/``
    component.
    """

    def __init__(self, pyramid_root) -> None:
        """
        Parameters
        ----------
        pyramid_root : Path or str, or list/iterable of Path/str
            One or more pyramid directories each containing ``pyramid.json``.
            Typically the output of ``tiling.tile(map_dir)`` — a directory of
            named pyramid sub-directories — or a single such sub-directory.

            Two accepted shapes:
              (a) A single pyramid directory (has ``pyramid.json`` directly)
              (b) An iterable of pyramid directories (each has ``pyramid.json``)

            The caller MUST point only at ``train/``-subtree directories to
            honour EVAL-01 zero-leakage; this constructor additionally asserts
            at runtime that no indexed path contains a ``test/`` component.
        """
        # Normalise to list[Path]
        if isinstance(pyramid_root, (str, Path)):
            pyramid_root = Path(pyramid_root)
            # If it contains pyramid.json directly — treat as single pyramid dir
            if (pyramid_root / "pyramid.json").exists():
                roots = [pyramid_root]
            else:
                # Treat as parent dir whose subdirs are pyramid dirs
                roots = sorted(p for p in pyramid_root.iterdir() if p.is_dir()
                               and (p / "pyramid.json").exists())
        else:
            # Iterable of paths
            roots = [Path(p) for p in pyramid_root]

        self.samples: list[dict] = []  # {pdir, tile_entry, weights}

        for pdir in roots:
            manifest_path = pdir / "pyramid.json"
            if not manifest_path.exists():
                continue

            # Manifest-driven geometry (D-04): read verbatim, never re-derive.
            manifest = json.loads(manifest_path.read_text())
            tiles = manifest["tiles"]

            # Load per-pyramid sample_weights byte-identically (D-claude-discretion).
            weights_path = pdir / "sample_weights.json"
            if not weights_path.exists():
                continue
            weights = json.loads(weights_path.read_text())

            for tile in tiles:
                entry = {
                    "pdir": pdir,
                    "tile": tile,
                    "weights": weights,
                }
                self.samples.append(entry)

        # ToonDataset lines 93-94 idiom: raise on empty index.
        if not self.samples:
            raise ValueError(
                f"PyramidDataset: no tiles found under the given root(s). "
                "Ensure pyramid.json and sample_weights.json are present. "
                "If pointing at a test/ subtree, switch to the train/ root (EVAL-01)."
            )

        # EVAL-01 / T-03-07: hard guard — no indexed path may contain 'test/'
        # (mirrors test_tiling.py:213-215 leakage assertion).
        for entry in self.samples:
            parts = Path(entry["pdir"]).parts
            if "test" in parts:
                raise ValueError(
                    f"EVAL-01 violation: dataset path contains a 'test/' component: "
                    f"{entry['pdir']}\n"
                    "PyramidDataset must never enumerate paths under test/ (Pitfall 6)."
                )

    def __len__(self) -> int:
        return len(self.samples)

    def tile_path(self, idx: int) -> Path:
        """Return the pyramid directory for sample at ``idx``.

        Used by split-safety tests to assert no path contains a ``test/``
        component (test_seg_dataset.py:TestSplitSafety).
        """
        return Path(self.samples[idx]["pdir"])

    def __getitem__(self, idx: int) -> dict:
        """
        Load one tile and return a dict of tensors + sample_weights.

        Returns
        -------
        dict with keys:
          "image"          : (3, H, W) torch.float32 in [0, 1]
          "land_cover"     : (H, W)    torch.int64  (class label)
          "topography"     : (H, W)    torch.int64  (class label)
          "sample_weights" : dict      (byte-identical to sample_weights.json)
        """
        entry = self.samples[idx]
        pdir: Path = Path(entry["pdir"])
        tile: dict = entry["tile"]

        # Load image: (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
        img = Image.open(pdir / tile["image"]).convert("RGB")
        img_tensor = torch.from_numpy(
            # H x W x 3
            _pil_to_numpy(img)
        ).permute(2, 0, 1).float().div(255.0)  # (3, H, W) float32

        # Load land_cover: L-mode (H, W) uint8 → (H, W) int64
        lc = Image.open(pdir / tile["land_cover"])
        lc_tensor = torch.from_numpy(_pil_to_numpy(lc)).long()  # (H, W)

        # Load topography: L-mode (H, W) uint8 → (H, W) int64
        topo = Image.open(pdir / tile["topography"])
        topo_tensor = torch.from_numpy(_pil_to_numpy(topo)).long()  # (H, W)

        return {
            "image": img_tensor,
            "land_cover": lc_tensor,
            "topography": topo_tensor,
            "sample_weights": entry["weights"],  # raw dict, unchanged
        }


def _pil_to_numpy(img: Image.Image):
    """Convert a PIL Image to a writable numpy ndarray.

    ``np.asarray`` returns a read-only view for some PIL modes; ``np.array``
    (with ``copy=True`` semantics as of NumPy 2.x) always returns a writable
    array, which avoids the ``torch.from_numpy`` non-writable tensor warning.
    """
    import numpy as np
    return np.array(img)
