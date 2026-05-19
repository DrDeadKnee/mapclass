"""Generator for notebooks/01_single_slice.ipynb (Phase 1 finish line).

Run with the pinned venv:
    .venv/bin/python notebooks/_build_01_single_slice.py

Authoring the notebook from a script keeps the cell sources reviewable as
plain Python and the ipynb byte-stable. The notebook itself is the committed
artifact; this generator is a build helper kept in-tree for reproducibility.
"""

import json
import os

CELLS = []


def md(text):
    CELLS.append(
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(
    """
# Phase 1 Single-Slice Attribution (01_single_slice)

The Phase 1 **finish line**: SigLIP-2 forward + dynamic LRP against the
contrastive similarity scalar `logits_per_image[0,0]` on the locked slice
(`manifest[-1]`, D-05) + control query `"a river"` (D-06), reconstructed to a
27x27 overlay aligned to the source map, plus the **three mandatory sanity
controls** (query-swap vs `"a xylophone"`, vision-tower randomization, top-k
vs random occlusion) rendered side-by-side for the visual eyeball gate
(D-03/D-04 — judged by your eyes, **no quantitative thresholds**).
"""
)

# --- Cell 1: path wiring, imports, seed, CUDA assert ----------------------
code(
    """
# Cell 1 - path wiring, imports, deterministic seed, CUDA fail-fast
import sys, os, random
from pathlib import Path

_REPO_ROOT = Path.cwd()
if (_REPO_ROOT / "notebooks").exists() is False and (_REPO_ROOT.name == "notebooks"):
    _REPO_ROOT = _REPO_ROOT.parent
for _p in (_REPO_ROOT / "third_party" / "dynamicLRP" / "src", _REPO_ROOT / "src"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

import numpy as np
import torch
import transformers
import matplotlib.pyplot as plt

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

assert torch.cuda.is_available(), (
    "CUDA required: SigLIP-2 + dynamic-LRP on so400m is impractical on CPU "
    "(CLAUDE.md). Run this notebook on the GPU VM."
)
print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("device      :", torch.cuda.get_device_name(0))
print("seed        :", SEED, "(torch/numpy/cuda set + recorded)")
"""
)

# --- Cell 2: locked slice ------------------------------------------------
code(
    """
# Cell 2 - resolve the LOCKED slice = manifest[-1] (D-05, highest LIST index)
from mapclass.manifest import load_manifest, entry_by_index

m = load_manifest()
entry = entry_by_index(m, -1)
print("manifest length :", len(m))
print("locked slice id :", entry["id"])
assert entry["id"] == "RUMSEY~8~1~344476~90112460", entry["id"]
print("D-05 satisfied  : highest list index, no richness sort, no user input")
"""
)

# --- Cell 3: model + coverage probe --------------------------------------
code(
    """
# Cell 3 - load the GCS-mirrored singleton SigLIP-2 + processor; coverage probe
from mapclass.model_loader import get_model_and_processor
from mapclass.data_loader import load_slice
from mapclass.attribution import attribute, coverage_probe

model, processor = get_model_and_processor()
print("model class :", type(model).__name__)
print("vision tower:", type(model.vision_model).__name__)

# Fallback-Ladder step 1: static op-coverage probe BEFORE the first relevance
# pass (a coverage artifact, not a gate).
_it, _ii, _am, _pil = load_slice(entry, "a river")
_ops = coverage_probe(model, _it, _ii, _am)
try:
    _names, _count, _graph = _ops
except (TypeError, ValueError):
    _count = len(_ops) if hasattr(_ops, "__len__") else _ops
print("get_model_operations op count:", _count)
del _it, _ii, _am, _pil
torch.cuda.empty_cache()
"""
)

# --- Cell 4: control run + overlay + grid demo + peak VRAM ----------------
code(
    """
# Cell 4 - CONTROL RUN: locked slice + "a river" -> overlay aligned to the map
from mapclass.overlay import to_patch_grid, composite, draw_patch_grid

img_tensor, input_ids, attention_mask, pil = load_slice(entry, "a river")
res_river = attribute(model, img_tensor, input_ids, attention_mask)

grid_signed, grid_mag = to_patch_grid(res_river.relevance)

# Peak VRAM for the single so400m + LRP attribution (STATE.md blocker -
# gates Phase 2 sweep sizing). Recorded into the figure AND printed.
PEAK_VRAM_BYTES = res_river.peak_vram_bytes
PEAK_VRAM_GB = PEAK_VRAM_BYTES / (1024 ** 3)
print(f"target form used        : {res_river.target_form}")
print(f"PEAK VRAM (single attr) : {PEAK_VRAM_GB:.3f} GB "
      f"({PEAK_VRAM_BYTES:,} bytes)")
print(f"relevance shape         : {tuple(res_river.relevance.shape)}")
print(f"signed grid min/max     : {grid_signed.min():.4g} / {grid_signed.max():.4g}")
print(f"has negative relevance  : {bool((grid_signed < 0).any())}")

fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs[0].imshow(pil); axs[0].set_title(f"source map\\n{entry['id']}"); axs[0].set_axis_off()
composite(axs[1], pil, grid_signed, grid_mag,
          title='"a river" signed overlay (bwr, 0-centered) [nearest]')
composite(axs[2], pil, grid_signed, grid_mag,
          title='"a river" signed overlay [interpolated]',
          interpolation="bilinear")
fig.suptitle(
    f'CONTROL: locked slice + "a river"  |  peak VRAM '
    f'{PEAK_VRAM_GB:.2f} GB  |  LRP target {res_river.target_form}'
)
fig.tight_layout(); plt.show()

# D-07: explicit grid-placement + 6-px edge-exclusion demonstration.
fig2, ax = plt.subplots(1, 1, figsize=(9, 9))
draw_patch_grid(ax, pil)
fig2.tight_layout(); plt.show()
"""
)

# --- Cell 5: CONTROL A query-swap ----------------------------------------
code(
    """
# Cell 5 - CONTROL A (query-swap): "a river" vs "a xylophone" (D-06), ADJACENT.
# PASS by eye: substantially DIFFERENT regions. FAIL: near-identical
# (image-saliency, not query-driven).
it_x, ii_x, am_x, pil_x = load_slice(entry, "a xylophone")
res_xylo = attribute(model, it_x, ii_x, am_x)
gx_signed, gx_mag = to_patch_grid(res_xylo.relevance)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
composite(axs[0], pil, grid_signed, grid_mag, title='"a river" (control query)')
composite(axs[1], pil_x, gx_signed, gx_mag, title='"a xylophone" (query-swap)')
fig.suptitle("CONTROL A - query-swap (eyeball: regions must differ substantially)")
fig.tight_layout(); plt.show()
del it_x, ii_x, am_x
torch.cuda.empty_cache()
"""
)

# --- Cell 6: CONTROL B model-randomization -------------------------------
code(
    """
# Cell 6 - CONTROL B (model-randomization): re-init the VISION tower only.
# Reload a fresh model copy so the trained singleton is NOT clobbered (the
# singleton stays trained for any later use). Keep the text tower trained.
# PASS by eye: randomized overlay collapses to STRUCTURELESS NOISE next to the
# structured trained map. FAIL: still structured.
from transformers import AutoModel
from mapclass import config as _cfg

rand_model = (
    AutoModel.from_pretrained(_cfg.LOCAL_MODEL_CACHE_DIR,
                              attn_implementation="eager")
    .to(_cfg.DEVICE).eval()
)

@torch.no_grad()
def _reinit(module):
    for _n, p in module.named_parameters(recurse=False):
        if p.dim() >= 2:
            torch.nn.init.xavier_uniform_(p)
        else:
            torch.nn.init.zeros_(p)

rand_model.vision_model.apply(_reinit)  # VISION tower only; text tower intact

it_r, ii_r, am_r, _pil_r = load_slice(entry, "a river")
res_rand = attribute(rand_model, it_r, ii_r, am_r)
gr_signed, gr_mag = to_patch_grid(res_rand.relevance)

fig, axs = plt.subplots(1, 2, figsize=(16, 8))
composite(axs[0], pil, grid_signed, grid_mag,
          title='trained vision tower - "a river"')
composite(axs[1], pil, gr_signed, gr_mag,
          title='RANDOMIZED vision tower - "a river"')
fig.suptitle("CONTROL B - model-randomization "
             "(eyeball: randomized must collapse to noise)")
fig.tight_layout(); plt.show()

del rand_model, it_r, ii_r, am_r
torch.cuda.empty_cache()
"""
)

# --- Cell 7: CONTROL C occlusion -----------------------------------------
code(
    """
# Cell 7 - CONTROL C (occlusion): mean-fill the top-k highest-relevance 14x14
# patches vs k RANDOM patches; recompute logits_per_image[0,0] for each.
# PASS by eye: occluding TOP-relevance patches drops similarity MORE than
# random. The similarity numbers are RENDERED INTO the figure as captions
# (a visual annotation, NOT a printed scalar / assert - consistent with D-03).
from mapclass.overlay import GRID, PATCH_SIZE, VALID_DIM

K = 20  # number of 14x14 patches to occlude

@torch.no_grad()
def _similarity(px):
    out = model(pixel_values=px, input_ids=input_ids,
                attention_mask=attention_mask)
    return float(out.logits_per_image[0, 0].item())

# Per-patch relevance ordering from the control "a river" run (magnitude).
flat_order = np.argsort(grid_mag.ravel())[::-1]  # high -> low
topk_idx = flat_order[:K]
rng = np.random.default_rng(SEED)
rand_idx = rng.choice(GRID * GRID, size=K, replace=False)

base_px = img_tensor.detach().clone()
fill = base_px.mean().item()

def _occlude(idxs):
    px = base_px.clone()
    for fi in idxs:
        r, c = divmod(int(fi), GRID)
        y0, x0 = r * PATCH_SIZE, c * PATCH_SIZE
        px[:, :, y0:y0 + PATCH_SIZE, x0:x0 + PATCH_SIZE] = fill
    return px

px_top = _occlude(topk_idx)
px_rand = _occlude(rand_idx)
sim_orig = _similarity(base_px)
sim_top = _similarity(px_top)
sim_rand = _similarity(px_rand)

def _to_disp(px):
    a = px.detach()[0].cpu().float()
    a = (a - a.min()) / (a.max() - a.min() + 1e-8)
    return a.permute(1, 2, 0).numpy()

fig, axs = plt.subplots(1, 3, figsize=(21, 7))
axs[0].imshow(_to_disp(base_px))
axs[0].set_title(f"original\\nsimilarity = {sim_orig:.4f}")
axs[0].set_axis_off()
axs[1].imshow(_to_disp(px_top))
axs[1].set_title(f"top-{K} relevance occluded\\nsimilarity = {sim_top:.4f}\\n"
                 f"drop = {sim_orig - sim_top:+.4f}")
axs[1].set_axis_off()
axs[2].imshow(_to_disp(px_rand))
axs[2].set_title(f"{K} RANDOM patches occluded\\nsimilarity = {sim_rand:.4f}\\n"
                 f"drop = {sim_orig - sim_rand:+.4f}")
axs[2].set_axis_off()
fig.suptitle("CONTROL C - occlusion "
             "(eyeball: top-k drop should exceed random drop)")
fig.tight_layout(); plt.show()
torch.cuda.empty_cache()
"""
)

# --- Cell 8: markdown summary --------------------------------------------
md(
    """
## Phase 1 Visual Eyeball Gate (D-02 / D-03 / D-04)

This is a **pure visual judgment** — no quantitative threshold or `assert` is
applied to the three controls (D-04 explicit prohibition). Judge each by eye
from the figures above:

| Control | PASS (by eye) | FAIL (by eye) |
|---|---|---|
| **A. Query-swap** (`"a river"` vs `"a xylophone"`, adjacent) | Heatmaps highlight **substantially different** regions | Heatmaps look essentially the same → image-saliency, not query-driven |
| **B. Model-randomization** (randomized vision tower next to trained) | Randomized overlay collapses to **structureless noise** | Randomized overlay still structured → not reading the model |
| **C. Occlusion** (original / top-k-occluded / random-occluded triptych) | Occluding **top-relevance** patches drops similarity **more** than random | top-k drop ≈ random drop → MAP-pool relevance not faithful |

**Peak VRAM** for the single so400m + LRP attribution is printed in Cell 4
(gates Phase 2 sweep sizing — STATE.md blocker).

**Phase 1 is DONE** only when BOTH: (1) all three controls visually PASS here
(D-03), AND (2) the full 1,544-image mirror + outcome manifest from Plan 01-02
is confirmed complete (D-02 — verified APPROVED in 01-02-SUMMARY.md). On any
control FAIL, the documented **Fallback Ladder** in `attribution.py` is the
response path.
"""
)

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {
            "display_name": "mapclass (.venv)",
            "language": "python",
            "name": "mapclass",
        },
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = os.path.join(os.path.dirname(__file__), "01_single_slice.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("wrote", out, "with", len(CELLS), "cells")
