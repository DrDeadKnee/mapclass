"""
torch.utils.data.Dataset adapter over samples emitted by scripts/build_*.py.

Calls assert_sample_valid() once per sample at __init__ (training startup), NOT
per __getitem__ (per-batch overhead would be unacceptable). The brownfield-to-
mapclass schema check happens before the first forward pass.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from mapclass.data.contract import assert_sample_valid


_IMAGE_CANDIDATES = ("image.png", "flat.png", "illustrated.png", "satellite.png")


class MapClassDataset(Dataset):
    """Per-sample dataset reading {image*, land_cover, topography, manifest, weights}.

    Parameters
    ----------
    sample_ids : list of sample IDs (matching subdir names under root)
    root       : directory containing per-sample subdirs
    validate   : if True, run assert_sample_valid() once per sample at __init__
    """

    def __init__(self, sample_ids: list[str], root: str | Path, validate: bool = True):
        self.root = Path(root)
        self.sample_ids = sample_ids
        if validate:
            for sid in sample_ids:
                assert_sample_valid(self.root / sid)

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.sample_ids[idx]
        sdir = self.root / sid
        # First image candidate that exists
        img_path = next(sdir / f for f in _IMAGE_CANDIDATES if (sdir / f).is_file())
        img = np.array(Image.open(img_path).convert("RGB"))           # (H, W, 3) uint8
        lc = np.array(Image.open(sdir / "land_cover.png"))            # (H, W) uint8
        topo = np.array(Image.open(sdir / "topography.png"))          # (H, W) uint8
        with open(sdir / "manifest.json") as f:
            manifest = json.load(f)

        return {
            "image": torch.from_numpy(np.moveaxis(img, -1, 0)).float() / 255.0,  # (3, H, W)
            "land_cover": torch.from_numpy(lc.astype(np.int64)),                 # (H, W)
            "topography": torch.from_numpy(topo.astype(np.int64)),               # (H, W)
            "source_subtype": manifest["source_subtype"],                        # for loss-weight lookup at trainer
            "sample_id": sid,
        }
