"""
Recursive coarse-to-fine (c2f) inference orchestrator (D-03 / D-04).

Strategy:
  - Walks ``pyramid.json`` parent→child literally using the stored
    ``tiles[].{id, x, y, size, children}`` boxes (D-04: never recompute
    pyramid geometry; read the manifest and trust it).
  - Pass 1 (896 cold-start): prior = zeros(1, 12, S, S) — no parent exists.
  - Pass 2+ (448 → 224): crop the parent probability map at the exact quadrant
    box ``parent_prob[..., cy-oy:cy-oy+s, cx-ox:cx-ox+s]``, then bilinearly
    resize to the child tile grid.  Recurse depth-first.
  - Variant-agnostic: the orchestrator only calls ``model(rgb, prior)`` (Variant A)
    or forwards the responsibility of building a 15-ch input to callers (Variant B
    callers concatenate prior into the input before calling this orchestrator, or
    use ``recursive_predict_variant_b``).
  - Forward-only: everything runs under ``torch.no_grad`` (D-06; no optimizer,
    no .backward(), no overfit run).

Exported symbols
----------------
- ``crop_prior_to_child(parent_prob, parent_tile, child_tile)``
    Crop the parent probability map at the exact pyramid.json quadrant box, then
    bilinearly resize to the child tile grid.  The load-bearing D-04 / Pitfall-5
    alignment primitive.
- ``cold_start_prior(batch, tile_size)``
    Return all-zeros (B, 12, tile_size, tile_size) for the 896 cold-start.
- ``PriorEncoder``
    Re-exported from ``seg.model``; a small trainable module mapping the 12-ch
    prior to a target spatial resolution for Variant A decoder injection.
- ``recursive_predict(model, pyramid_dir, read_tile_fn)``
    Walk the full 896→448→224 tree; return ``{tile_id: prob_tensor}`` dict.
- ``recursive_predict_variant_b(model, pyramid_dir, read_tile_fn)``
    Same walk but for Variant B (prior concatenated at input level).

Trust model:
  T-03-12: Crop coordinates come from pyramid.json boxes only (cy-oy:cy-oy+s,
           cx-ox:cx-ox+s) — never recomputed geometry (D-04 mitigate).
           One-hot blob in a parent quadrant must land in exactly the correct
           child (Pitfall 5 alignment test is a blocking gate).

Covered decisions: D-03, D-03a, D-04, D-06.

Requires:
  seg.model — SegModelVariantA / SegModelVariantB (plan 03-05)
  seg.decoder — SegDecoder (plan 03-04)
  torch, torch.nn.functional
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import torch
import torch.nn.functional as F
from torch import Tensor

# Re-export PriorEncoder so ``from seg.recursive import PriorEncoder`` works
# (test_seg_recursive.py imports it from this module).
from seg.model import PriorEncoder  # noqa: F401

# ---------------------------------------------------------------------------
# Prior-channel constants
# ---------------------------------------------------------------------------

_PRIOR_CHANNELS: int = 12   # 9 LC + 3 topo softmax-probability channels


# ---------------------------------------------------------------------------
# cold_start_prior
# ---------------------------------------------------------------------------

def cold_start_prior(batch: int, tile_size: int) -> Tensor:
    """
    Return the all-zero 896 cold-start prior (D-03).

    At the coarsest 896 level there is no parent, so the prior is zero.

    Parameters
    ----------
    batch:
        Batch dimension B.
    tile_size:
        Spatial size S of the tile (typically 896 at cold-start, but the
        function is usable for any scale).

    Returns
    -------
    Tensor
        (B, 12, S, S) zeros (float32).
    """
    return torch.zeros(batch, _PRIOR_CHANNELS, tile_size, tile_size)


# ---------------------------------------------------------------------------
# crop_prior_to_child
# ---------------------------------------------------------------------------

def crop_prior_to_child(
    parent_prob: Tensor,
    parent_tile: dict,
    child_tile: dict,
) -> Tensor:
    """
    Crop the parent probability map to the child's quadrant, then resize
    to the child tile grid (D-04 / RESEARCH Pattern 4 / Pitfall-5 guard).

    The crop coordinates come SOLELY from the ``pyramid.json`` manifest boxes
    (T-03-12 mitigate — never recomputed):
        ``ox, oy = parent_tile["x"], parent_tile["y"]``
        ``cx, cy, s = child_tile["x"], child_tile["y"], child_tile["size"]``
        crop = ``parent_prob[..., cy-oy : cy-oy+s, cx-ox : cx-ox+s]``

    Parameters
    ----------
    parent_prob:
        (B, 12, Sp, Sp) softmax-probability map from the parent tile's forward.
    parent_tile:
        Parent manifest entry dict: ``{id, x, y, size, children, ...}``.
    child_tile:
        Child manifest entry dict: ``{id, x, y, size, children, ...}``.

    Returns
    -------
    Tensor
        (B, 12, s, s) probability map at the child tile's resolution,
        where ``s = child_tile["size"]``.
    """
    ox, oy = parent_tile["x"], parent_tile["y"]
    cx, cy, s = child_tile["x"], child_tile["y"], child_tile["size"]

    # Quadrant crop using manifest boxes (D-04 literal; never recomputed)
    crop = parent_prob[..., cy - oy : cy - oy + s, cx - ox : cx - ox + s]

    # Bilinear resize to the child tile grid
    child_size = (s, s)
    if crop.shape[-2] != child_size[0] or crop.shape[-1] != child_size[1]:
        crop = F.interpolate(
            crop,
            size=child_size,
            mode="bilinear",
            align_corners=False,
        )
    return crop


# ---------------------------------------------------------------------------
# recursive_predict (Variant A / orchestrator-agnostic)
# ---------------------------------------------------------------------------

def recursive_predict(
    model: "torch.nn.Module",
    pyramid_dir: "str | Path",
    read_tile_fn: "Callable[[Path, dict], Tensor] | None" = None,
) -> Dict[str, Tensor]:
    """
    Walk the pyramid.json tree depth-first and run ``model(rgb, prior)`` on
    every tile (D-03 / D-04 literal walk; Variant A orchestration).

    Pass 1 — 896 root:
        ``prior = cold_start_prior(1, 896)`` (all-zeros, D-03).
        Run model; softmax LC+topo logits → 12-ch probability map; cache.

    Pass 2 — 448 children; Pass 3 — 224 grandchildren:
        ``prior = crop_prior_to_child(parent_prob, parent_tile, child_tile)``
        (exact manifest-box crop + bilinear resize, D-04).
        Run model; softmax; cache.

    Parameters
    ----------
    model:
        A ``SegModelVariantA`` (or any module with ``forward(rgb, prior)`` →
        ``(lc_logits, topo_logits)``).
    pyramid_dir:
        Directory containing ``pyramid.json`` and per-tile image PNGs.
    read_tile_fn:
        ``read_tile_fn(pyramid_dir, tile_entry) → (1, 3, S, S) float32 [0,1]``.
        If ``None`` a default PIL-based reader is used.

    Returns
    -------
    Dict[str, Tensor]
        ``{tile_id: (1, 12, S, S) softmax probs}`` for every tile in the pyramid.
        The 224-level entries are the finest-scale outputs.
    """
    pyramid_dir = Path(pyramid_dir)
    man = json.loads((pyramid_dir / "pyramid.json").read_text())
    by_id: dict[str, dict] = {t["id"]: t for t in man["tiles"]}
    root = next(t for t in man["tiles"] if t["size"] == 896)

    if read_tile_fn is None:
        read_tile_fn = _default_read_rgb

    prob_cache: Dict[str, Tensor] = {}

    def _predict(tile: dict, parent: "dict | None") -> None:
        rgb = read_tile_fn(pyramid_dir, tile)   # (1, 3, S, S)

        if parent is None:
            # 896 cold-start: prior = zeros (D-03)
            prior = cold_start_prior(rgb.shape[0], tile["size"])
        else:
            # Crop parent prob map at exact manifest quadrant (D-04 / Pitfall 5)
            prior = crop_prior_to_child(prob_cache[parent["id"]], parent, tile)

        with torch.no_grad():
            lc_logits, topo_logits = model(rgb, prior)

        # Softmax to raw probabilities (12-ch: 9 LC + 3 topo)
        lc_prob = torch.softmax(lc_logits, dim=1)      # (1, 9, S, S)
        topo_prob = torch.softmax(topo_logits, dim=1)  # (1, 3, S, S)
        prob_cache[tile["id"]] = torch.cat([lc_prob, topo_prob], dim=1)  # (1, 12, S, S)

        # Recurse over children (D-04: follow stored children list)
        for cid in tile.get("children", []):
            _predict(by_id[cid], tile)

    _predict(root, None)
    return prob_cache


# ---------------------------------------------------------------------------
# recursive_predict_variant_b
# ---------------------------------------------------------------------------

def recursive_predict_variant_b(
    model: "torch.nn.Module",
    pyramid_dir: "str | Path",
    read_tile_fn: "Callable[[Path, dict], Tensor] | None" = None,
) -> Dict[str, Tensor]:
    """
    Walk the pyramid.json tree for Variant B models.

    Variant B takes a 15-ch input (RGB + 12 prior channels concatenated).
    At the 896 cold-start, the 12 prior channels are all-zero.

    Parameters
    ----------
    model:
        A ``SegModelVariantB`` (``forward(x15) → (lc_logits, topo_logits)``).
    pyramid_dir:
        Directory containing ``pyramid.json`` and per-tile image PNGs.
    read_tile_fn:
        ``read_tile_fn(pyramid_dir, tile_entry) → (1, 3, S, S) float32 [0,1]``.
        If ``None`` a default PIL-based reader is used.

    Returns
    -------
    Dict[str, Tensor]
        ``{tile_id: (1, 12, S, S) softmax probs}``.
    """
    pyramid_dir = Path(pyramid_dir)
    man = json.loads((pyramid_dir / "pyramid.json").read_text())
    by_id: dict[str, dict] = {t["id"]: t for t in man["tiles"]}
    root = next(t for t in man["tiles"] if t["size"] == 896)

    if read_tile_fn is None:
        read_tile_fn = _default_read_rgb

    prob_cache: Dict[str, Tensor] = {}

    def _predict(tile: dict, parent: "dict | None") -> None:
        rgb = read_tile_fn(pyramid_dir, tile)   # (1, 3, S, S)

        if parent is None:
            prior = cold_start_prior(rgb.shape[0], tile["size"])
        else:
            prior = crop_prior_to_child(prob_cache[parent["id"]], parent, tile)

        # Resize prior to input H×W if needed
        s = tile["size"]
        if prior.shape[-2] != s or prior.shape[-1] != s:
            prior = F.interpolate(prior, size=(s, s), mode="bilinear", align_corners=False)

        # Concatenate to form 15-ch input
        x15 = torch.cat([rgb, prior], dim=1)   # (1, 15, S, S)

        with torch.no_grad():
            lc_logits, topo_logits = model(x15)

        lc_prob = torch.softmax(lc_logits, dim=1)
        topo_prob = torch.softmax(topo_logits, dim=1)
        prob_cache[tile["id"]] = torch.cat([lc_prob, topo_prob], dim=1)

        for cid in tile.get("children", []):
            _predict(by_id[cid], tile)

    _predict(root, None)
    return prob_cache


# ---------------------------------------------------------------------------
# Default tile reader (PIL-based, no torch dependency for the test fixture)
# ---------------------------------------------------------------------------

def _default_read_rgb(pyramid_dir: Path, tile_entry: dict) -> Tensor:
    """
    Load ``tile_entry["image"]`` as a (1, 3, S, S) float32 tensor in [0, 1].

    Used when no custom ``read_tile_fn`` is supplied to ``recursive_predict``.
    """
    from PIL import Image
    import numpy as np

    path = pyramid_dir / tile_entry["image"]
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)
