"""Generator for notebooks/01_single_slice.ipynb (Phase 1 SMOKE TEST).

Run with the pinned venv:
    .venv/bin/python notebooks/_build_01_single_slice.py

Authoring the notebook from a script keeps the cell sources reviewable as
plain Python and the ipynb byte-stable. The notebook itself is the committed
artifact; this generator is a build helper kept in-tree for reproducibility.

------------------------------------------------------------------------------
HONEST SCOPE (user decision at the 01-03 human-verify checkpoint, 2026-05-19):

dynamicLRP's op-coverage FAILS on SigLIP-2-so400m: `split_with_sizes` in the
MAP-pool / attention head is outside the current engine's covered ops, so
`attribute()` raises a `RuntimeError` and **no LRP relevance is produced for
SigLIP-2**. The user reviewed this and explicitly DECLINED the entire Fallback
Ladder (NO custom Promise, NO pre-pool / `use_attn_lrp` engineering, NO LXT,
NO captum IG) — quote: "Smoke-test was good, it didn't crash. Let's leave well
enough alone and move on." The project is reframed as a multi-model comparison
of dynamic-LRP; SigLIP-2 is one model and this is an ACCEPTED per-model FINDING.
The Phase 1 D-02/D-03 visual-eyeball gate is CONSCIOUSLY WAIVED for SigLIP-2.

Therefore this notebook is an honest END-TO-END SMOKE TEST, not the original
overlay + three-controls finish line:
  * loads SigLIP-2 from the GCS weights mirror,
  * builds the requires_grad (1,3,384,384) pixel tensor for the locked slice
    (manifest[-1], query "a river") via the existing loaders,
  * runs the forward and measures peak forward VRAM (closes the STATE.md
    peak-VRAM blocker),
  * runs the dynamicLRP coverage probe and prints the op count,
  * calls attribute() inside a try/except that CATCHES the RuntimeError and
    renders a clear, labeled FINDING cell (NOT an uncaught traceback),
  * still shows the source map inline so the slice is visually identified,
  * finishes with exit code 0 — the finding is a recorded result, not a crash.
The three D-02/D-03 sanity controls are NOT rendered because no relevance
exists to overlay; this is the user-WAIVED gate, documented honestly here and
in 01-03-SUMMARY.md (not a silent pass). The Fallback Ladder was DECLINED by
the user and must NOT be re-attempted later.
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
# Phase 1 Single-Slice — dynamicLRP × SigLIP-2 SMOKE TEST (01_single_slice)

> **Honest scope.** This notebook is an **end-to-end smoke test**, not the
> original overlay + three-sanity-controls finish line. dynamicLRP's
> op-coverage **does not cover SigLIP-2-so400m**: the `split_with_sizes` op in
> SigLIP-2's MAP-pool / attention head is outside the current engine, so
> `attribute()` raises a `RuntimeError` and **no LRP relevance / heatmap is
> produced for SigLIP-2**. This was reviewed at the 01-03 human-verify
> checkpoint and the user **explicitly declined the entire Fallback Ladder**
> (no custom Promise, no pre-pool/`use_attn_lrp`, no LXT, no captum IG):
> *"Smoke-test was good, it didn't crash. Let's leave well enough alone and
> move on."* The project is reframed as a **multi-model comparison of
> dynamic-LRP** — SigLIP-2 is one model and this is an **accepted per-model
> FINDING**. The Phase 1 D-02/D-03 visual-eyeball gate is **consciously
> WAIVED for SigLIP-2** (documented in `01-03-SUMMARY.md`, not a silent pass).

What this notebook DOES, end to end, exiting 0:
1. loads SigLIP-2 from the GCS weights mirror,
2. builds the `requires_grad` `(1,3,384,384)` pixel tensor for the **locked
   slice** (`manifest[-1]`, D-05) + locked control query `"a river"` (D-06),
3. runs the forward and **measures peak forward VRAM** (closes the STATE.md
   peak-VRAM blocker),
4. runs the dynamicLRP **coverage probe** and prints the op count,
5. calls `attribute()` in a `try/except` that **catches** the `RuntimeError`
   and prints the FINDING (not a traceback),
6. shows the **source map** inline so the slice is visually identified.
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

# dynamicLRP static op-coverage probe BEFORE the relevance pass. This is the
# coverage artifact for the multi-model comparison (an artifact, not a gate).
_it, _ii, _am, _pil = load_slice(entry, "a river")
_ops = coverage_probe(model, _it, _ii, _am)
try:
    _names, _count, _graph = _ops
except (TypeError, ValueError):
    _count = len(_ops) if hasattr(_ops, "__len__") else _ops
print("dynamicLRP get_model_operations op count:", _count)
print(
    "NOTE: SigLIP-2's MAP-pool/attention head uses `split_with_sizes`, which "
    "is OUTSIDE the current dynamicLRP engine coverage (see the FINDING cell)."
)
del _it, _ii, _am, _pil
torch.cuda.empty_cache()
"""
)

# --- Cell 4: locked-slice forward + peak forward VRAM + source map --------
code(
    """
# Cell 4 - locked slice + "a river": forward, PEAK FORWARD VRAM, source map.
# The forward succeeds; only the dynamicLRP relevance pass fails (Cell 5).
# This cell CLOSES the STATE.md peak-VRAM blocker by measuring the peak VRAM
# of the SigLIP-2 forward on the locked (1,3,384,384) slice tensor.
img_tensor, input_ids, attention_mask, pil = load_slice(entry, "a river")
print("pixel tensor shape :", tuple(img_tensor.shape),
      "| requires_grad =", bool(img_tensor.requires_grad))

torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    _fwd = model(pixel_values=img_tensor, input_ids=input_ids,
                 attention_mask=attention_mask)
    _sim = float(_fwd.logits_per_image[0, 0].item())
PEAK_FWD_VRAM_BYTES = int(torch.cuda.max_memory_allocated())
PEAK_FWD_VRAM_GB = PEAK_FWD_VRAM_BYTES / (1024 ** 3)
del _fwd
torch.cuda.empty_cache()

print(f'similarity logits_per_image[0,0] ("a river") : {_sim:.4f}')
print(f"PEAK FORWARD VRAM (single so400m forward)    : "
      f"{PEAK_FWD_VRAM_GB:.3f} GB ({PEAK_FWD_VRAM_BYTES:,} bytes)")
print("STATE.md peak-VRAM blocker: CLOSED (forward peak measured + recorded "
      "above; the dynamicLRP relevance pass does not run for SigLIP-2 — see "
      "the FINDING cell).")

# Show the source map inline so the locked slice is visually identified.
fig, ax = plt.subplots(1, 1, figsize=(9, 9))
ax.imshow(pil)
ax.set_title(f"locked slice (manifest[-1], D-05)\\n{entry['id']}")
ax.set_axis_off()
fig.tight_layout(); plt.show()
"""
)

# --- Cell 5: attribute() FINDING (caught RuntimeError) --------------------
code(
    """
# Cell 5 - dynamicLRP attribution attempt: CAUGHT op-coverage FINDING.
# We call attribute() honestly and CATCH the RuntimeError it raises (the
# `split_with_sizes` coverage gap). This is a recorded per-model RESULT for
# the multi-model dynamic-LRP comparison, NOT a cell error / traceback.
import traceback

LRP_RELEVANCE = None
LRP_FINDING = None
try:
    _res = attribute(model, img_tensor, input_ids, attention_mask)
    LRP_RELEVANCE = _res.relevance
    print("UNEXPECTED: attribute() returned relevance of shape",
          tuple(LRP_RELEVANCE.shape),
          "- the documented SigLIP-2 op-coverage gap did NOT occur. "
          "Investigate before treating this as a finding.")
except RuntimeError as exc:
    LRP_FINDING = repr(exc)
    print("dynamicLRP attribute() raised (EXPECTED, CAUGHT):")
    print(" ", LRP_FINDING)
finally:
    torch.cuda.empty_cache()
"""
)

# --- Cell 6: finding markdown --------------------------------------------
md(
    """
## FINDING — dynamicLRP op-coverage on SigLIP-2 (user-accepted; Fallback Ladder DECLINED)

**dynamicLRP op-coverage FINDING.** SigLIP-2-so400m's MAP-pool /
attention-pooling head uses the `split_with_sizes` op
(`SplitWithSizesBackward0` in the autograd graph). The current dynamicLRP
engine registers `SplitWithSizesBackward → SplitBackwardProp` but its Promise
consumer chokes on SigLIP-2's split topology (`'DummyPromise' object is not
iterable` / `No valid curnode candidate was found`), and a 0-dim target form
raises `IndexError`. **`split_with_sizes` is outside the current engine's
covered ops for SigLIP-2 as a contrastive MAP-pool encoder, so no LRP
relevance — and therefore no attribution heatmap — is produced for this
model.** This is exactly the MEDIUM-LOW research risk flagged in `CLAUDE.md`
(*"Whether dynamicLRP covers 100% of SigLIP-2's specific ops out of the box —
MEDIUM-LOW"*).

**This is a recorded per-model RESULT for the multi-model dynamic-LRP
comparison**, the reframed purpose of the project — not a project failure.

**Fallback Ladder: DECLINED by the user** at the 01-03 human-verify checkpoint
(2026-05-19). The user reviewed the smoke-test and the finding and explicitly
declined **every** rung — **no** custom dynamicLRP Promise, **no**
pre-pool / `use_attn_lrp` engineering, **no** vendored LXT, **no** captum
Integrated Gradients baseline. Quote: *"Smoke-test was good, it didn't crash.
Let's leave well enough alone and move on."* This must **not** be
re-attempted later (also recorded in the `attribution.py` module docstring and
`01-03-SUMMARY.md`).

**Phase 1 D-02/D-03 visual-eyeball gate: WAIVED for SigLIP-2.** The three
sanity controls (query-swap, model-randomization, occlusion) are **NOT**
rendered here because **no relevance exists to overlay** — there is nothing to
eyeball. This is a *consciously waived* gate for SigLIP-2 under the reframed
multi-model-comparison purpose, recorded honestly in `01-03-SUMMARY.md`. It is
**not** a silent pass and **not** "all controls passed".

**Smoke-test status — PASS (honest):**

| Item | Result |
|---|---|
| Pinned env + GCS-mirrored SigLIP-2 loads | OK |
| Locked slice (`manifest[-1]`, `"a river"`) `requires_grad` `(1,3,384,384)` tensor built via loaders | OK |
| SigLIP-2 forward + `logits_per_image[0,0]` similarity | OK |
| **Peak forward VRAM measured + recorded** (Cell 4 — STATE.md blocker CLOSED) | OK |
| dynamicLRP coverage probe op count printed (Cell 3) | OK |
| dynamicLRP `attribute()` — `split_with_sizes` coverage gap, **caught** as a recorded finding | FINDING (no heatmap) |
| Three D-02/D-03 sanity controls | **WAIVED for SigLIP-2** (no relevance to render) |
| Notebook runs end-to-end, exit code 0 | OK |
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
