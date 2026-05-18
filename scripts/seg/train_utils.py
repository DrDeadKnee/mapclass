"""
Training building blocks for Phase 4 segmentation fine-tuning.

EVAL-01: This module never touches test/ paths.  All dataset construction
operates exclusively on train/-subtree directories.  PyramidDataset's
constructor enforces EVAL-01 at runtime (T-04-02).

Public API (consumed by downstream plans and scripts/finetune_seg.py):
    build_lc_weight_tensor(sw_dict, device) -> Tensor              # (9,) float32
    weighted_joint_loss(lc_logits, topo_logits, lc_targets,
                        topo_targets, sample_weights_batch, device) -> Tensor  # scalar
    train_step_variant_a(model, pyramid, optimizer, device) -> float
    train_step_variant_b(model, pyramid, optimizer, device) -> float
    make_optimizer(model, lr=1e-4, weight_decay=1e-2) -> AdamW
    carve_train_val(train_root_dirs, val_frac=0.2, seed=42) -> (train_ds, val_ds)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split

# --- sys.path bootstrap -------------------------------------------------
# This file lives in scripts/seg/; scripts/ must be on sys.path for the
# sibling biome_mapping import to resolve (mirrors other seg modules).
_HERE = Path(__file__).parent          # scripts/seg/
_SCRIPTS = _HERE.parent                # scripts/
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from biome_mapping import LANDCOVER_CLASSES  # 9 canonical class names, ordered
from seg.dataset import PyramidDataset
from seg.recursive import cold_start_prior, crop_prior_to_child


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------

def build_lc_weight_tensor(sw_dict: dict, device) -> Tensor:
    """
    Build a (9,) float32 weight tensor ordered by LANDCOVER_CLASSES.

    Parameters
    ----------
    sw_dict : dict
        sample_weights.json payload — must contain:
          ``land_cover_weights``: dict mapping all 9 LANDCOVER_CLASSES to float
          ``topography_weight``  : float
    device : str or torch.device

    Returns
    -------
    Tensor
        (9,) float32 class-weight tensor ordered exactly by LANDCOVER_CLASSES.

    Raises
    ------
    ValueError
        If any of the 9 LANDCOVER_CLASSES keys is missing from
        ``sw_dict["land_cover_weights"]`` (T-04-01: descriptive error, not KeyError).
    """
    if "land_cover_weights" not in sw_dict:
        raise ValueError(
            "build_lc_weight_tensor: sw_dict is missing the 'land_cover_weights' key. "
            f"Got keys: {list(sw_dict.keys())}"
        )
    lc_w = sw_dict["land_cover_weights"]
    missing = [c for c in LANDCOVER_CLASSES if c not in lc_w]
    if missing:
        raise ValueError(
            f"build_lc_weight_tensor: sample_weights.json is missing "
            f"{len(missing)} LANDCOVER_CLASSES key(s): {missing}. "
            f"All 9 classes must be present: {LANDCOVER_CLASSES}"
        )
    return torch.tensor(
        [lc_w[c] for c in LANDCOVER_CLASSES], dtype=torch.float32, device=device
    )


def weighted_joint_loss(
    lc_logits: Tensor,
    topo_logits: Tensor,
    lc_targets: Tensor,
    topo_targets: Tensor,
    sample_weights_batch: list,
    device,
) -> Tensor:
    """
    Per-sample weighted cross-entropy loss for joint LC + topo heads.

    For sample i:
        lc_loss   = CE(lc_logits[i], lc_targets[i], weight=lc_class_w)
        topo_loss = CE(topo_logits[i], topo_targets[i]) * topography_weight
        sample_loss = lc_loss + topo_loss

    Returns mean over batch B.

    Parameters
    ----------
    lc_logits : Tensor  (B, 9, H, W) raw logits
    topo_logits : Tensor  (B, 3, H, W) raw logits
    lc_targets : Tensor  (B, H, W) int64 class labels
    topo_targets : Tensor  (B, H, W) int64 class labels
    sample_weights_batch : list[dict]  one sample_weights.json payload per sample
    device : str or torch.device

    Returns
    -------
    Tensor  scalar (0-dim)
    """
    B = lc_logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)

    for i, sw in enumerate(sample_weights_batch):
        lc_class_w = build_lc_weight_tensor(sw, device)
        topo_w = torch.tensor(
            float(sw["topography_weight"]), dtype=torch.float32, device=device
        )

        # ignore_index=255: label.py already emits 255 (LC = NODATA/void,
        # topo = water — no valid slope class). Without this, those pixels
        # are out-of-range class indices for 9-/3-way CE and trigger a
        # device-side index assert. Mirrors evaluate_seg.py's
        # `valid = (lc_gt < 9) & (topo_gt < 3)` eval mask.
        lc_loss = F.cross_entropy(
            lc_logits[i:i + 1],
            lc_targets[i:i + 1],
            weight=lc_class_w,
            ignore_index=255,
            reduction="mean",
        )
        topo_loss = F.cross_entropy(
            topo_logits[i:i + 1],
            topo_targets[i:i + 1],
            ignore_index=255,
            reduction="mean",
        ) * topo_w

        total_loss = total_loss + (lc_loss + topo_loss)

    return total_loss / B


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def make_optimizer(
    model: torch.nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 1e-2,
) -> torch.optim.AdamW:
    """
    Build an AdamW optimizer with two param groups:

    - Decay group   : all requires_grad=True params NOT in {bias, norm, LayerNorm}
    - No-decay group: all requires_grad=True params in  {bias, norm, LayerNorm}

    Frozen backbone params (requires_grad=False) are excluded from all groups.

    Parameters
    ----------
    model : torch.nn.Module
    lr : float
    weight_decay : float

    Returns
    -------
    torch.optim.AdamW
    """
    no_decay_keywords = {"bias", "norm", "LayerNorm"}

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen backbone params (EVAL-01 / T-04-02)
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(param_groups, lr=lr)


# ---------------------------------------------------------------------------
# Val carve
# ---------------------------------------------------------------------------

def carve_train_val(
    train_root_dirs,
    val_frac: float = 0.2,
    seed: int = 42,
):
    """
    Split a PyramidDataset constructed from train/ roots into train/val subsets.

    EVAL-01: this function constructs PyramidDataset from ``train_root_dirs``
    only.  Any path containing a 'test/' component will cause PyramidDataset
    to raise ValueError at construction time (T-04-02).

    Parameters
    ----------
    train_root_dirs : path-like or iterable of path-like
        One or more pyramid root directories under the train/ subtree.
    val_frac : float   fraction of samples for validation (default 0.2)
    seed : int         RNG seed for reproducible splits (default 42)

    Returns
    -------
    tuple[Subset, Subset]  (train_ds, val_ds)
    """
    ds = PyramidDataset(train_root_dirs)
    n_val = int(val_frac * len(ds))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val], generator=gen)


# ---------------------------------------------------------------------------
# Internal: image reader (mirrors recursive.py _default_read_rgb)
# ---------------------------------------------------------------------------

def _read_rgb_tensor(pyramid_dir: Path, tile_entry: dict) -> Tensor:
    """Load tile image as (1, 3, S, S) float32 in [0, 1]."""
    from PIL import Image
    import numpy as np

    path = pyramid_dir / tile_entry["image"]
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return t


def _read_label_tensor(pyramid_dir: Path, tile_entry: dict, key: str) -> Tensor:
    """Load land_cover or topography label as (1, H, W) int64."""
    from PIL import Image
    import numpy as np

    path = pyramid_dir / tile_entry[key]
    img = Image.open(path)
    arr = np.array(img, dtype=np.int64)
    return torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)


# ---------------------------------------------------------------------------
# Teacher-forced coarse-to-fine train step — Variant A
# ---------------------------------------------------------------------------

def train_step_variant_a(
    model: torch.nn.Module,
    pyramid: "Union[Path, str]",
    optimizer: torch.optim.Optimizer,
    device,
) -> float:
    """
    One teacher-forced coarse-to-fine training step over a full pyramid (Variant A).

    Walk order: 896 root → 448 children → 224 grandchildren (depth-first, mirrors
    seg/recursive.py _predict).

    Key contract (T-04-03 / RESEARCH Pitfall 2):
        After each tile's forward, the LC + topo logits are softmax'd and DETACHED
        before being cached as the child prior.  This prevents backprop-through-time
        across both c2f passes, which would cause an OOM explosion on CPU.

    Parameters
    ----------
    model : SegModelVariantA  (forward(rgb, prior) → (lc_logits, topo_logits))
    pyramid : path to the pyramid directory (containing pyramid.json)
    optimizer : AdamW (or compatible)
    device : str or torch.device

    Returns
    -------
    float  total loss across all 21 tiles, divided by B=1
    """
    pyramid = Path(pyramid)
    man = json.loads((pyramid / "pyramid.json").read_text())
    sw = json.loads((pyramid / "sample_weights.json").read_text())

    by_id: dict[str, dict] = {t["id"]: t for t in man["tiles"]}
    root = next(t for t in man["tiles"] if t["size"] == 896)

    model.train()
    optimizer.zero_grad()

    prob_cache: dict[str, Tensor] = {}
    total_loss = 0.0

    def _step(tile: dict, parent: "dict | None") -> None:
        nonlocal total_loss

        rgb = _read_rgb_tensor(pyramid, tile).to(device)

        if parent is None:
            prior = cold_start_prior(1, tile["size"]).to(device)
        else:
            prior = crop_prior_to_child(
                prob_cache[parent["id"]],  # already detached (T-04-03)
                parent,
                tile,
            ).to(device)

        lc_logits, topo_logits = model(rgb, prior)  # grads flow for this tile

        # Cache DETACHED softmax probabilities — no grad through the cached probs
        lc_prob_det = torch.softmax(lc_logits.detach(), dim=1)   # (1, 9, H, W)
        topo_prob_det = torch.softmax(topo_logits.detach(), dim=1)  # (1, 3, H, W)
        prob_cache[tile["id"]] = torch.cat([lc_prob_det, topo_prob_det], dim=1)

        # Per-tile loss (non-detached logits, grads flow)
        lc_targets = _read_label_tensor(pyramid, tile, "land_cover").to(device)
        topo_targets = _read_label_tensor(pyramid, tile, "topography").to(device)
        tile_loss = weighted_joint_loss(
            lc_logits, topo_logits, lc_targets, topo_targets, [sw], device
        )
        # Backward per tile so only ONE tile's autograd graph is ever live.
        # The T-04-03 detach already isolates each tile's graph, so summing
        # the losses then a single backward is mathematically identical to
        # accumulating each tile_loss.backward() into .grad — but the deferred
        # form kept all 21 graphs resident at once (CPU OOM). zero_grad() was
        # called once above; grads accumulate across tiles as intended.
        tile_loss.backward()
        total_loss += float(tile_loss.detach())

        for cid in tile.get("children", []):
            _step(by_id[cid], tile)

    _step(root, None)

    optimizer.step()

    return total_loss


# ---------------------------------------------------------------------------
# Teacher-forced coarse-to-fine train step — Variant B
# ---------------------------------------------------------------------------

def train_step_variant_b(
    model: torch.nn.Module,
    pyramid: "Union[Path, str]",
    optimizer: torch.optim.Optimizer,
    device,
) -> float:
    """
    One teacher-forced coarse-to-fine training step over a full pyramid (Variant B).

    Variant B concatenates [rgb, prior12] into 15-ch input before calling model(x15).
    Otherwise identical walk and detach protocol to Variant A.

    Parameters
    ----------
    model : SegModelVariantB  (forward(x15) → (lc_logits, topo_logits))
    pyramid : path to the pyramid directory
    optimizer : AdamW (or compatible)
    device : str or torch.device

    Returns
    -------
    float  total loss across all 21 tiles
    """
    pyramid = Path(pyramid)
    man = json.loads((pyramid / "pyramid.json").read_text())
    sw = json.loads((pyramid / "sample_weights.json").read_text())

    by_id: dict[str, dict] = {t["id"]: t for t in man["tiles"]}
    root = next(t for t in man["tiles"] if t["size"] == 896)

    model.train()
    optimizer.zero_grad()

    prob_cache: dict[str, Tensor] = {}
    total_loss = 0.0

    def _step(tile: dict, parent: "dict | None") -> None:
        nonlocal total_loss

        rgb = _read_rgb_tensor(pyramid, tile).to(device)

        if parent is None:
            prior = cold_start_prior(1, tile["size"]).to(device)
        else:
            prior = crop_prior_to_child(
                prob_cache[parent["id"]],  # already detached
                parent,
                tile,
            ).to(device)

        # Resize prior to match rgb spatial dims if needed (Variant B input-level concat)
        s = tile["size"]
        if prior.shape[-2] != s or prior.shape[-1] != s:
            prior = F.interpolate(prior, size=(s, s), mode="bilinear", align_corners=False)

        x15 = torch.cat([rgb, prior], dim=1)  # (1, 15, H, W)
        lc_logits, topo_logits = model(x15)   # grads flow

        # Cache detached probs
        lc_prob_det = torch.softmax(lc_logits.detach(), dim=1)
        topo_prob_det = torch.softmax(topo_logits.detach(), dim=1)
        prob_cache[tile["id"]] = torch.cat([lc_prob_det, topo_prob_det], dim=1)

        lc_targets = _read_label_tensor(pyramid, tile, "land_cover").to(device)
        topo_targets = _read_label_tensor(pyramid, tile, "topography").to(device)
        tile_loss = weighted_joint_loss(
            lc_logits, topo_logits, lc_targets, topo_targets, [sw], device
        )
        # Backward per tile (see Variant A note): bounds peak memory to one
        # tile's graph instead of all 21. Grads accumulate into .grad exactly
        # as the prior single sum-of-losses backward did.
        tile_loss.backward()
        total_loss += float(tile_loss.detach())

        for cid in tile.get("children", []):
            _step(by_id[cid], tile)

    _step(root, None)

    optimizer.step()

    return total_loss
