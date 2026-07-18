"""
Torch dataset producing masked hex windows from region npz files (Phase B).

Each item: a BFS window of up to ``window_size`` hexes from one region,
50–95% hidden as contiguous blobs, padded to fixed length. Regions are
sampled proportionally to their hex count; item ``i`` is fully determined
by ``(seed, i)`` so epochs are reproducible.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from hexprior.windows import (
    blob_mask,
    build_adjacency,
    load_region,
    local_xy,
    sample_window,
)

DEFAULT_WINDOW_SIZE = 128
DEFAULT_MASK_RANGE = (0.5, 0.95)


class HexWindowDataset(Dataset):
    def __init__(
        self,
        region_paths: list[str],
        window_size: int = DEFAULT_WINDOW_SIZE,
        mask_range: tuple[float, float] = DEFAULT_MASK_RANGE,
        epoch_len: int = 10_000,
        seed: int = 0,
    ):
        if not region_paths:
            raise ValueError("no region paths given")
        self.window_size = window_size
        self.mask_range = mask_range
        self.epoch_len = epoch_len
        self.seed = seed

        self.regions = []
        for path in region_paths:
            reg = load_region(path)
            if reg["cells"].size < 8:
                print(f"  skipping tiny region {path} ({reg['cells'].size} hexes)")
                continue
            adj = build_adjacency(reg["cells"])
            soft_by_cell = {
                int(c): reg["soft"][k] for k, c in enumerate(reg["cells"])
            }
            self.regions.append({
                "path": path,
                "adjacency": adj,
                "soft_by_cell": soft_by_cell,
                "n": reg["cells"].size,
            })
        if not self.regions:
            raise ValueError("all regions empty/tiny")
        weights = np.array([r["n"] for r in self.regions], dtype=np.float64)
        self.region_p = weights / weights.sum()

    def __len__(self) -> int:
        return self.epoch_len

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng((self.seed, i))
        region = self.regions[int(rng.choice(len(self.regions), p=self.region_p))]

        window = sample_window(region["adjacency"], self.window_size, rng)
        xy = local_xy(window)
        ratio = rng.uniform(*self.mask_range)
        hidden = blob_mask(window, region["adjacency"], ratio, rng)
        soft = np.stack([region["soft_by_cell"][c] for c in window])

        k, ksz = len(window), self.window_size
        n_classes = soft.shape[1]
        out_soft = np.zeros((ksz, n_classes), dtype=np.float32)
        out_xy = np.zeros((ksz, 2), dtype=np.float32)
        out_mask = np.zeros(ksz, dtype=bool)
        out_pad = np.ones(ksz, dtype=bool)
        out_soft[:k] = soft
        out_xy[:k] = xy
        out_mask[:k] = hidden
        out_pad[:k] = False

        return {
            "soft": torch.from_numpy(out_soft),
            "xy": torch.from_numpy(out_xy),
            "mask": torch.from_numpy(out_mask),
            "pad": torch.from_numpy(out_pad),
        }
