---
phase: 01-pinned-environment-gcs-mirrors-and-a-verified-single-slice-a
plan: 01
subsystem: toolchain
status: paused-at-checkpoint
tags: [environment, pinning, dynamicLRP, vendoring, pytest, vit-repro]
requires: []
provides:
  - pinned requirements.txt (dynamicLRP pins verbatim + project additions)
  - vendored dynamicLRP src/lrp_engine @ SHA 405e74243ecaa1f615f418fdc8ba24c3c5889b1e
  - src/mapclass package + config.py (GCS/model/device/VENDOR_SHA constants)
  - pytest scaffold (pytest.ini, tests/conftest.py sys.path wiring)
  - notebooks/00_vit_repro.ipynb (ViT dynamic-LRP repro on pinned stack)
affects:
  - all downstream plans (01-02 mirrors, 01-03 attribution) consume this env + config
tech-stack:
  added:
    - torch==2.7.1, transformers==4.52.3, torchvision==0.22.1 (pinned)
    - dynamicLRP (vendored in-tree, not pip/submodule)
    - pytest>=8,<9, jupyterlab>=4.4,<5, google-cloud-storage>=3.0,<4
  patterns:
    - vendored-third-party-with-recorded-SHA (D-08)
    - venv+pip primary env; conda used ONLY to source a 3.10 interpreter
    - uninstalled source imports via sys.path (conftest + notebook bootstrap)
key-files:
  created:
    - requirements.txt
    - src/mapclass/__init__.py
    - src/mapclass/config.py
    - third_party/dynamicLRP/VENDOR_SHA
    - third_party/dynamicLRP/src/lrp_engine/ (full engine package, 18 files)
    - pytest.ini
    - tests/__init__.py
    - tests/conftest.py
    - notebooks/00_vit_repro.ipynb
  modified:
    - .gitignore (added /.cache/, /.py310base/)
decisions:
  - "Used conda-forge to materialize a Python 3.10 interpreter (3.10 absent from PATH; only 3.12/3.13 present). venv+pip remains the primary env manager per CLAUDE.md — conda is only the interpreter source, not the resolver. Tracked as Rule 3 deviation."
  - "Committed notebook un-executed; executed copy lives at /tmp/00_vit_repro_executed.ipynb per the plan's verify step."
  - "Reference ViT.ipynb uses HF ViTForImageClassification (not timm) — reproduced exactly what upstream uses; added the documented 224/16=14 patch-grid reshape as the plan-required shape artifact on top of the reference .sum(dim=0) visualization."
metrics:
  tasks_completed: 2
  tasks_total: 3
  status: paused at Task 3 (checkpoint:human-verify)
  completed_date: 2026-05-18
---

# Phase 1 Plan 01: Pinned Environment + ViT Repro Summary

Stood up the reproducible pinned toolchain (torch==2.7.1 / transformers==4.52.3),
vendored `keeinlev/dynamicLRP`'s `src/lrp_engine/` in-tree at the fixed SHA
`405e74243ecaa1f615f418fdc8ba24c3c5889b1e` (D-08), scaffolded the
`src/mapclass` package + pytest, and reproduced the reference ViT dynamic-LRP
attribution end-to-end on the pinned stack. **Paused at Task 3 (human-verify
checkpoint)** — the human must visually judge the reproduced heatmap.

## What Was Built

### Task 1 — Pin environment, vendor dynamicLRP, scaffold (commit `36a68fe`)
- `requirements.txt`: the 9 dynamicLRP `requirements.txt` pins **verbatim**
  (`einops==0.8.1`, `scikit_learn==1.7.0`, `torch==2.7.1`, `tqdm==4.66.4`,
  `transformers==4.52.3`, `omegaconf==2.3.0`, `matplotlib==3.8.0`,
  `seaborn==0.13.2`, `timm==1.0.20`) + conservatively-pinned previously-unpinned
  LRP deps (`datasets>=3,<4`, `captum>=0.7,<0.9`) + project additions
  (`torchvision==0.22.1`, `Pillow>=10.3,<12`, `google-cloud-storage>=3.0,<4`,
  `requests`, `ipykernel`, `jupyterlab>=4.4,<5`, `pytest>=8,<9`).
- `.python-version` confirmed `3.10` (already correct, left as-is).
- Vendored `src/lrp_engine/` (18 files incl. `promises/`, `model_specific/`)
  via a temp clone at the exact SHA, then `VENDOR_SHA` recorded as a single
  tracked line. No `git+https`, no submodule, no clone-at-setup.
- `src/mapclass/config.py`: `GCS_BUCKET`, `GCS_DATA_PREFIX`, `GCS_MODELS_PREFIX`,
  `MODEL_REPO_ID`, `MODEL_GCS_DIR`, `DEVICE` (CUDA-resolved), `VENDOR_SHA`
  (mirrors the tracked file), `LOCAL_IMAGE_CACHE_DIR`, `MANIFEST_PATH`.
- pytest scaffold: `pytest.ini` (`testpaths=tests`), `tests/conftest.py`
  prepends vendored `lrp_engine` src + project `src` to `sys.path`.
- All automated verify + acceptance checks passed.

### Task 2 — Build pinned venv + reproduce ViT.ipynb (commit `2d7a8fd`)
- `.venv` built from a Python 3.10.20 interpreter; `pip install -r
  requirements.txt` resolved with **no conflict**. Confirmed `torch
  2.7.1+cu126`, `transformers 4.52.3`, `torchvision 0.22.1`, all verbatim
  pins exact, `numpy 1.26.4` (torch-constrained). `mapclass (.venv)` kernel
  registered.
- `notebooks/00_vit_repro.ipynb` mirrors the reference `ViT.ipynb` mechanism:
  `ViTForImageClassification` patch16/224, one CIFAR10 image with the reference
  transform, `LRPEngine(use_gamma=True, no_recompile=True)`,
  `params_to_interpret=[img_tensor]`, `run(output.logits)`,
  `get_model_operations` coverage probe, 224/16=14 patch-grid reshape, and an
  alpha-blended relevance overlay.
- `nbconvert --execute` (1200s timeout): exit 0, **0 cell errors**, rendered
  heatmap PNG present, `get_model_operations` op count = **16**, relevance
  shape `(1,3,224,224)` == input shape, patch grid `(14,14)`, vendored engine
  imported **offline**. Executed copy: `/tmp/00_vit_repro_executed.ipynb`.

### Task 3 — Human-verify checkpoint (NOT executed — gate)
The pinned environment and executed notebook are ready for the human's visual
judgment of the reproduced ViT relevance heatmap. This is the toolchain-proof
gate (ROADMAP Success Criterion 1).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Python 3.10 absent from the environment**
- **Found during:** Task 2 (venv build)
- **Issue:** The plan/`.python-version` require Python 3.10 for `.venv`, but the
  worktree host has only `python3.12` and `python3` (3.13) on PATH; 3.10 is not
  installed and not available via apt. Building the venv is impossible without
  a 3.10 interpreter.
- **Fix:** Used the available `conda` (conda-forge channel, `--override-channels`
  to avoid the Anaconda ToS gate) to materialize a standalone Python 3.10.20
  interpreter at `./.py310base`, then built the plain `.venv` from it with
  `venv` + `pip install -r requirements.txt`. This respects CLAUDE.md's "venv+pip
  primary, no conda/poetry/uv as the resolver" rule — conda only sources the
  interpreter binary; the dependency resolve is plain pip against the pinned
  requirements (which resolved with zero conflict, satisfying the D-08 hard-pin
  invariant).
- **Files modified:** `.gitignore` (ignore `/.py310base/`)
- **Commit:** `2d7a8fd`

### Environment notes (normal flow, not deviations)
- No GPU in the worktree (`torch.cuda.is_available() == False`). The ViT repro
  is one forward + one LRP pass on a base ViT for a single image — feasible on
  CPU within the nbconvert timeout. SigLIP-2/GPU work is Plan 03 (a different
  environment/runtime concern; flagged in research as a GPU assumption).
- `relevance.std` is small in absolute magnitude but non-zero with a real
  min/max spread on the patch grid (max ≈ 8.5× min). Visual structure judgment
  is explicitly the human's at the Task 3 gate, not an automated assert.

## Known Stubs

None. All created files are functional; no placeholder/empty-value stubs.

## Self-Check: PASSED

- `requirements.txt` — FOUND
- `src/mapclass/config.py` — FOUND
- `third_party/dynamicLRP/VENDOR_SHA` — FOUND (== pinned SHA)
- `third_party/dynamicLRP/src/lrp_engine/lrp.py` — FOUND (`class LRPEngine`)
- `pytest.ini`, `tests/conftest.py` — FOUND
- `notebooks/00_vit_repro.ipynb` — FOUND
- commit `36a68fe` — FOUND
- commit `2d7a8fd` — FOUND

## Next Step

Task 3 is a blocking `checkpoint:human-verify`. The human opens
`notebooks/00_vit_repro.ipynb` (or `/tmp/00_vit_repro_executed.ipynb`) on the
`mapclass (.venv)` kernel, Restart Kernel & Run All, and judges whether the ViT
relevance heatmap is structured/non-uniform (matching the reference dynamicLRP
qualitative look). Resume signal: `"approved"` or a description of what is wrong.
