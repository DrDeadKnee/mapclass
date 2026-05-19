"""Generator for notebooks/02_multimodel.ipynb (v1.1 MILESTONE deliverable).

Run with the pinned venv:
    .venv/bin/python notebooks/_build_02_multimodel.py

Authoring the notebook from a script keeps the cell sources reviewable as
plain Python and the ipynb byte-stable. The notebook is the committed executed
artifact; this generator is the source of truth (same pattern as
``_build_01_single_slice.py``).

------------------------------------------------------------------------------
WHAT THE NOTEBOOK DOES (one notebook, exit 0):

  * asserts the vendored dynamicLRP SHA == 405e74243ecaa1f615f418fdc8ba24c3c5889b1e
    (one read-only sanity assert — NOT a guard framework),
  * loads the LAST 50 Rumsey manifest maps (``load_manifest()[-50:]`` — NO
    sort/filter; manifest.py forbids it),
  * iterates the 4 models (siglip2 | clip | vit_b16 | paligemma) from the
    plain MODEL_REGISTRY, SEQUENTIALLY for VRAM (load model -> all 50 maps ->
    del + empty_cache),
  * for each (map, model): builds the per-model requires_grad inputs, runs the
    generalized attribute(), and on success renders the D-09 signed/zero-
    centered overlay reconstructed from THAT model's patch geometry; on ANY
    Exception renders a captioned "no heatmap — <model>: <reason>" tile and
    CONTINUES,
  * renders grouped PER MAP (one row per map, 4 columns) with each panel
    titled by its attribution-target semantics (the ViT class-conditioned /
    PaliGemma answer-token asymmetry is surfaced, not hidden),
  * ends with a small tally cell (model -> #heatmaps / #no-heatmap).

EXPECTED RECORDED RESULTS (NOT failures — do NOT engineer around them):
  * SigLIP-2: `split_with_sizes` op-coverage gap -> every tile "no heatmap"
    (accepted v1.0 FINDING; Fallback Ladder DECLINED — no custom Promise /
    LXT / captum / pre-pool).
  * PaliGemma-3B: may OOM the L4 (~3B, ~7.5x the so400m ceiling) -> "no
    heatmap" tiles. An expected, recordable comparison datum.
A coverage gap / OOM rendered as a captioned tile IS success for this task; a
crashed (non-exit-0) notebook is NOT. VENDOR_SHA + frozen pins untouched.
------------------------------------------------------------------------------
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
# v1.1 — Multi-Model × Multi-Map dynamic-LRP (02_multimodel)

ONE notebook running **dynamic-LRP attribution** for **4 models**
(SigLIP-2, CLIP, ViT-b-16, PaliGemma-3B) against the **last 50 Rumsey maps**,
single fixed query `"a river"`, models + maps loaded from
`gs://mapclass-training-northeast1/`. Per-model overlays are grouped **per
map** (one row, 4 columns); any failing `(map, model)` is a captioned
**"no heatmap"** tile and the run continues; the notebook completes headless
at **exit 0**.

> **Recorded results, not failures.** SigLIP-2's `split_with_sizes` MAP-pool
> op is outside the vendored dynamicLRP engine (accepted v1.0 FINDING —
> Fallback Ladder DECLINED), and PaliGemma-3B (~3B, ~7.5× the so400m ceiling)
> may OOM the L4. Both render as honest captioned "no heatmap" tiles. The
> attribution targets are **NOT all apples-to-apples**: SigLIP-2/CLIP use
> query-conditioned `logits_per_image`; **ViT is CLASS-conditioned (no text
> tower)**; **PaliGemma is an answer-token logit**. Each panel title states
> its target semantics so the side-by-side is not misread.
"""
)

# --- Cell 1: path wiring, imports, seed, CUDA assert, VENDOR_SHA ----------
code(
    """
# Cell 1 - path wiring, imports, deterministic seed, CUDA + VENDOR_SHA assert
import sys, os, random, gc, traceback
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
import matplotlib
matplotlib.use("Agg")  # headless nbconvert
import matplotlib.pyplot as plt

from mapclass import config

SEED = 0
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

assert torch.cuda.is_available(), (
    "CUDA required: dynamic-LRP on these models is impractical on CPU "
    "(CLAUDE.md). Run this notebook on the GPU VM."
)

# One read-only sanity assert (NOT a guard framework): the vendored engine
# integrity is the VENDOR_SHA being intact (threat T-2v4-01).
_EXPECTED_SHA = "405e74243ecaa1f615f418fdc8ba24c3c5889b1e"
assert config.VENDOR_SHA == _EXPECTED_SHA, config.VENDOR_SHA
_sha_file = (_REPO_ROOT / "third_party" / "dynamicLRP" / "VENDOR_SHA").read_text().strip()
assert _sha_file == _EXPECTED_SHA, _sha_file

print("torch       :", torch.__version__)
print("transformers:", transformers.__version__)
print("device      :", torch.cuda.get_device_name(0))
print("VENDOR_SHA  :", config.VENDOR_SHA, "(intact, == third_party/dynamicLRP/VENDOR_SHA)")
print("seed        :", SEED)
"""
)

# --- Cell 2: last-50 maps + registry + fixed query -----------------------
code(
    """
# Cell 2 - last 50 maps (NO sort/filter), the 4-model registry, fixed query
from mapclass.manifest import load_manifest
from mapclass.models import MODEL_REGISTRY
from mapclass.data_loader import _cached_image_path
from mapclass.attribution import attribute
from mapclass import overlay as ovl

MAPS = load_manifest()[-50:]            # the last 50 entries; D-05 positional
QUERY = "a river"                       # single fixed query — model & map are the axes
MODEL_NAMES = ["siglip2", "clip", "vit_b16", "paligemma"]

print("manifest length :", len(load_manifest()))
print("maps (last 50)  :", len(MAPS), "->", MAPS[0]["id"], "...", MAPS[-1]["id"])
print("models          :", MODEL_NAMES)
print("query           :", repr(QUERY))
assert len(MAPS) == 50
assert set(MODEL_NAMES) == set(MODEL_REGISTRY.keys())
"""
)

# --- Cell 3: per-model target-semantics labels (surfaced, not hidden) -----
code(
    '''
# Cell 3 - human-readable attribution-target label per model (panel titles).
# The ViT class-conditioned / PaliGemma answer-token asymmetry is SURFACED.
from mapclass import models as _m

def target_label(name, model=None):
    if name in ("siglip2", "clip"):
        base = f"{name}: logits_per_image (query-conditioned)"
        if name == "siglip2":
            base += " — split_with_sizes op-gap expected"
        return base
    if name == "vit_b16":
        if model is not None:
            cid, lab = _m._resolve_vit_class_id(model, QUERY)
            return f"vit_b16: CLASS-conditioned (NOT text) — class {cid}:{lab}"
        return "vit_b16: CLASS-conditioned (NOT text-conditioned)"
    if name == "paligemma":
        return ("paligemma: answer-token logit "
                "(<image> {q}; pos=-1; tok='yes') — OOM possible")
    return name

for _n in MODEL_NAMES:
    print(" -", target_label(_n))
'''
)

# --- Cell 4: the loop — sequential model load, 50 maps, captioned tiles ---
code(
    '''
# Cell 4 - SEQUENTIAL per-model loop. For each model: load -> attribute all 50
# maps (collecting (signed_grid, mag_grid, pil) or a failure reason) -> del +
# empty_cache. ANY Exception per (map,model) becomes a captioned "no heatmap"
# tile, never a crash (SigLIP-2 split_with_sizes gap + PaliGemma OOM both fold
# into this path). Results are keyed [map_id][model] for the per-map render.
import time

RESULTS = {entry["id"]: {} for entry in MAPS}   # id -> {model: dict}
TALLY = {n: {"heatmap": 0, "no_heatmap": 0} for n in MODEL_NAMES}
PEAK_VRAM = {}                                  # model -> max peak bytes seen

for name in MODEL_NAMES:
    spec = MODEL_REGISTRY[name]
    geom = spec["patch_geom"]
    t0 = time.time()
    model = processor = None
    try:
        model, processor = spec["load_fn"]()
        # Stash processor on the model so the PaliGemma answer-token target_fn
        # can resolve the answer token id (see models._target_paligemma_*).
        try:
            model._mapclass_processor = processor
        except Exception:
            pass
        load_ok = True
        load_err = None
    except Exception as e:                       # model failed to even load
        load_ok = False
        load_err = f"load failed: {type(e).__name__}: {e}"
        print(f"[{name}] {load_err}")

    peak_seen = 0
    # A per-model op-coverage gap (e.g. SigLIP-2's split_with_sizes) is
    # DETERMINISTIC across maps: same architecture + same op -> map 1's
    # failure reason IS map 2..50's reason. dynamicLRP (vendored, frozen)
    # ALSO orphans ~GBs of retained activation graph on EVERY failed
    # attribute() call (its forward hooks are not torn down on the failure
    # path; we must NOT patch the engine — VENDOR_SHA is frozen). Calling a
    # known-failing model 50× would leak 50× and OOM every later model. So
    # once a model's attribution fails for the FIRST map, the remaining 49
    # maps are recorded with the SAME honest reason WITHOUT re-invoking the
    # leaking engine. This bounds the leak to ONE call per failing model and
    # is faithful (NOT engineering around the gap — no Promise/LXT/captum;
    # the gap is still a recorded "no heatmap" result for all 50 tiles).
    model_failed_reason = None
    for entry in MAPS:
        mid = entry["id"]
        if not load_ok:
            RESULTS[mid][name] = {"ok": False, "reason": load_err}
            TALLY[name]["no_heatmap"] += 1
            continue
        if model_failed_reason is not None:
            RESULTS[mid][name] = {
                "ok": False,
                "reason": (
                    f"{model_failed_reason} "
                    "(deterministic per-model gap — not re-run per map; "
                    "bounds the vendored-engine retained-graph leak)"
                ),
                "pil": None,
            }
            TALLY[name]["no_heatmap"] += 1
            continue
        from PIL import Image
        try:
            pil = Image.open(_cached_image_path(mid)).convert("RGB")
            forward_inputs, img_tensor = spec["build_inputs_fn"](
                model, processor, pil, QUERY, config.DEVICE
            )
            # Carry the query so the ViT class-conditioned target_fn can map
            # it to a class id (kept out of the model forward kwargs).
            forward_inputs_for_fwd = {
                k: v for k, v in forward_inputs.items() if not k.startswith("_")
            }
            forward_inputs["_query"] = QUERY
            res = attribute(
                model, forward_inputs_for_fwd, img_tensor,
                lambda mdl, out, _fi, _f=spec["target_fn"]: _f(mdl, out, forward_inputs),
            )
            gs, gm = ovl.to_patch_grid(
                res.relevance,
                patch_size=geom["patch_size"],
                img_dim=geom["img_dim"],
                has_cls=geom["has_cls"],
            )
            RESULTS[mid][name] = {
                "ok": True, "pil": pil, "gs": gs, "gm": gm,
                "target_form": res.target_form,
                "peak_vram": res.peak_vram_bytes,
            }
            TALLY[name]["heatmap"] += 1
            if res.peak_vram_bytes:
                peak_seen = max(peak_seen, int(res.peak_vram_bytes))
        except Exception as e:                   # coverage gap / OOM / etc.
            short = f"{type(e).__name__}: {str(e)[:140]}"
            RESULTS[mid][name] = {"ok": False, "reason": short, "pil": None}
            # keep the pil for the tile background if we got that far
            try:
                RESULTS[mid][name]["pil"] = pil
            except Exception:
                pass
            TALLY[name]["no_heatmap"] += 1
            # First failure for this model -> record it and short-circuit the
            # remaining maps (deterministic; bounds the engine leak).
            model_failed_reason = short
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()

    PEAK_VRAM[name] = peak_seen
    dt = time.time() - t0
    print(f"[{name}] done: heatmap={TALLY[name]['heatmap']} "
          f"no_heatmap={TALLY[name]['no_heatmap']} "
          f"peak_vram={peak_seen/1024**3:.3f}GB elapsed={dt:.1f}s")
    # Aggressive teardown before the next model: move weights off-GPU, drop
    # ALL refs, double-gc. (dynamicLRP orphans some retained graph the loader
    # cannot reach — that residual is the vendored-engine leak, bounded to one
    # failed call per model by the short-circuit above; it does not prevent
    # the remaining models from loading on the 22 GB L4.)
    try:
        if load_ok and model is not None:
            model.to("cpu")
    except Exception:
        pass
    del model, processor
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
'''
)

# --- Cell 5: render grouped PER MAP (1 row, 4 columns) -------------------
code(
    '''
# Cell 5 - render grouped PER MAP: one figure per map, 4 columns
# (siglip2 | clip | vit_b16 | paligemma). Success -> D-09 signed/zero-centered
# composite from THAT model's patch geometry. Failure -> captioned "no heatmap"
# tile (map shown if available). 50 maps -> 50 figures emitted sequentially;
# modest dpi keeps the executed ipynb from being enormous.
import matplotlib.pyplot as plt

for entry in MAPS:
    mid = entry["id"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.0), dpi=70)
    fig.suptitle(f"{mid}   |   query: {QUERY!r}", fontsize=11)
    for ax, name in zip(axes, MODEL_NAMES):
        r = RESULTS[mid].get(name, {"ok": False, "reason": "missing"})
        title = name
        if r.get("ok"):
            tl = name
            try:
                tl = title  # keep short in-grid; full semantics printed in Cell 3
            except Exception:
                pass
            ovl.composite(
                ax, r["pil"], r["gs"], r["gm"],
                title=f"{name}\\n[{r.get('target_form','?')}]",
                interpolation="nearest",
            )
        else:
            if r.get("pil") is not None:
                w, h = r["pil"].size
                ax.imshow(r["pil"], extent=(0, w, h, 0))
            ax.text(
                0.5, 0.5,
                f"no heatmap\\n{name}\\n{r.get('reason','')}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="white",
                bbox=dict(boxstyle="round", facecolor="black", alpha=0.65),
                wrap=True,
            )
            ax.set_title(name)
            ax.set_axis_off()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()
    plt.close(fig)
'''
)

# --- Cell 6: tally + target-semantics summary ----------------------------
code(
    '''
# Cell 6 - final tally: model -> #heatmaps / #no-heatmap (eyeball coverage).
# This is the legible run-outcome summary (NOT the deferred structured
# op-coverage report — just a tally).
print("=== dynamic-LRP coverage tally (50 maps x fixed query 'a river') ===")
print(f"{'model':<12}{'heatmap':>9}{'no_heatmap':>12}{'peak_vram_GB':>15}")
for n in MODEL_NAMES:
    pv = PEAK_VRAM.get(n, 0) / 1024**3
    print(f"{n:<12}{TALLY[n]['heatmap']:>9}{TALLY[n]['no_heatmap']:>12}{pv:>15.3f}")
print()
print("attribution-target semantics (surfaced, not hidden):")
for n in MODEL_NAMES:
    print("  -", target_label(n))
print()
print("Expected recorded results: SigLIP-2 split_with_sizes op-gap -> all "
      "no_heatmap; PaliGemma-3B may OOM the L4 -> no_heatmap. These are "
      "honest recorded comparison data, NOT failures. Notebook exits 0.")
'''
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

out = os.path.join(os.path.dirname(__file__), "02_multimodel.ipynb")
with open(out, "w", encoding="utf-8") as fh:
    json.dump(nb, fh, indent=1, ensure_ascii=False)
    fh.write("\n")
print("wrote", out, "with", len(CELLS), "cells")
