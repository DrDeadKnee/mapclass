---
phase: 01-pinned-environment-gcs-mirrors-and-a-verified-single-slice-a
plan: 02
subsystem: persistence-and-data-access
tags: [gcs-mirror, manifest, siglip2, data-loader, idempotent-ingest]
status: paused-at-checkpoint
requires:
  - "src/mapclass/config.py (Plan 01-01 — GCS_BUCKET / prefixes / MODEL_REPO_ID / DEVICE / MANIFEST_PATH)"
  - "pinned venv (Plan 01-01 — torch==2.7.1, transformers==4.52.3, google-cloud-storage, huggingface_hub, PIL, pytest)"
provides:
  - "src/mapclass/manifest.py — ordered-index manifest reader (locked slice = manifest[-1], D-05)"
  - "src/mapclass/ingest_images.py — idempotent validated Rumsey image mirror + per-id outcome manifest"
  - "src/mapclass/mirror_model.py — idempotent HF snapshot → GCS model mirror"
  - "src/mapclass/model_loader.py — GCS-only singleton SigLIP-2 + processor (eager attn, fixed-res)"
  - "src/mapclass/data_loader.py — id → (requires_grad pixel tensor, PIL) with local byte cache"
  - "notebooks/scripts/run_ingest.py — full 1,544 ingest runner + status histogram"
affects:
  - "Plan 01-03 attribution (consumes model_loader + the SAME requires_grad tensor from data_loader)"
tech-stack:
  added: []
  patterns:
    - "module-level singleton model loader (load multi-GB weights once per kernel)"
    - "skip-if-exists-and-size>0 idempotent GCS mirror with per-id outcome manifest"
    - "tensor-identity invariant: data_loader returns inputs[pixel_values].requires_grad_() with no clone/detach/re-.to()"
key-files:
  created:
    - src/mapclass/__init__.py
    - src/mapclass/config.py
    - src/mapclass/manifest.py
    - src/mapclass/ingest_images.py
    - src/mapclass/mirror_model.py
    - src/mapclass/model_loader.py
    - src/mapclass/data_loader.py
    - notebooks/scripts/run_ingest.py
    - tests/test_manifest.py
    - tests/test_ingest_idempotency.py
    - tests/test_loaders.py
  modified: []
decisions:
  - "config.py reproduced in this worktree to Plan 01-01's documented contract (parallel wave-1 dependency) so the worktrees converge byte-identically on merge"
  - "tests are self-contained (each adds src/ to sys.path) — tests/conftest.py / __init__.py / pytest.ini are Plan 01-01-owned; not created here to avoid a merge collision"
metrics:
  duration: "~1 task-session (Tasks 1-2); Task 3 is a blocking human-verify checkpoint"
  completed-date: 2026-05-18
---

# Phase 01 Plan 02: Persistence and Data-Access Layer Summary

Idempotent GCS mirror + manifest reader + GCS-only singleton SigLIP-2 loader and a
data loader that hands back the exact `requires_grad_()` pixel tensor — Tasks 1-2
complete and committed; Task 3 (full 1,544 mirror) is a blocking human-verify gate.

## What Was Built

**Task 1 (commit `f981080`):**
- `manifest.py` — `load_manifest()` returns the raw JSON list (verified `len==1544`);
  `entry_by_index(m,-1)` resolves the locked slice `RUMSEY~8~1~344476~90112460`
  (D-05); `count_down_from` yields descending `(i, entry)` for Phase 2. Module
  docstring explicitly documents `richness_score` is NOT monotonic with index
  (Pitfall A); no sort/filter on `richness_score` anywhere in the module.
- `ingest_images.py` — run-once idempotent resumable Rumsey mirror:
  ADC fail-fast (`verify_adc`, T-01-07); `mirror_one` skip-if-exists-and-size>0
  (Pitfall F); per-download validation (HTTP 200 → else `dead-url`; Content-Type
  contains `image` → else `bad-content-type`; Content-Length + streamed byte cap
  and pixel-dimension cap for decompression-bomb defense → `too-large`;
  `PIL.verify()` → else `decode-fail`) before any GCS write (Security V5,
  T-01-04/05); bounded `ThreadPoolExecutor` concurrency with exponential backoff
  on transient failures; per-id outcome manifest written to GCS + a local copy.
- `mirror_model.py` — `huggingface_hub.snapshot_download` then recursive upload,
  skipping blobs already present with a matching size (idempotent convergence).
- `notebooks/scripts/run_ingest.py` — drives the full 1,544 ingest, prints the
  status histogram and id-coverage count.

**Task 2 (commit `fbc973f`):**
- `model_loader.py` — module-level `(model, processor)` singleton; cache-first
  GCS download of the model dir, then
  `AutoModel.from_pretrained(local_dir, attn_implementation="eager").to(DEVICE).eval()`
  + `AutoProcessor.from_pretrained(local_dir)` (fixed-res `model_type: siglip`
  path; NaFlex symbols deliberately never referenced); GPU fail-fast; never
  re-loads, never touches HF at runtime (Pitfall G).
- `data_loader.py` — `load_slice(entry, query, ...)` does a local-disk-cache-first
  GCS fetch (GCS only on cache miss, DATA-04), opens `PIL.Image ...convert("RGB")`,
  runs the shared processor with `padding="max_length", max_length=64`
  (Assumption A2, asserted at runtime), and returns
  `(inputs["pixel_values"].requires_grad_(), input_ids, attention_mask, pil)` —
  the SAME tensor object, with no `.clone()` / `.detach()` / re-`.to()` after
  (Pattern 2 tensor-identity invariant for Plan 03's attribution).

## Verification

All automatable acceptance criteria for Tasks 1-2 were exercised and pass:

- `tests/test_manifest.py` — list length 1544; `entry_by_index(m,-1)['id'] ==
  "RUMSEY~8~1~344476~90112460"`; `count_down_from(m,1543,3)` → `[1543,1542,1541]`;
  docstring documents non-monotonicity; no sort/filter on `richness_score`.
- `tests/test_ingest_idempotency.py` — existing-blob → `skipped` + 0 uploads;
  non-200 → `dead-url`; undecodable → `decode-fail`; bad content-type →
  `bad-content-type`; ADC-absent → clear `RuntimeError`; ok path uploads once.
- `tests/test_loaders.py` — `attn_implementation="eager"` present; no NaFlex
  symbols in either loader; no clone/detach on the returned-tensor path;
  singleton returns the same object identity (loaded once); 2nd `load_slice`
  is a pure local-cache hit (0 GCS calls); full `(1,3,384,384)` `[-1,1]`
  `requires_grad` tensor contract (self-skips when the pinned stack is absent).

**Verification mechanism note:** the pinned `.venv` (torch/transformers/
google-cloud-storage/PIL/pytest) is built by parallel wave-1 Plan 01-01 and is
absent in this isolated worktree. Tests were run via a stdlib harness (all
network/GCS/PIL/torch surfaces mocked or self-skipped). The plan's
`.venv/bin/python -m pytest` commands and the full-tensor test are run by the
orchestrator's post-merge pinned suite; the live full-mirror run is Task 3's
human-verify gate. This is the expected parallel-wave verification split, not a
defect.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 — Blocking issue] Created `src/mapclass/config.py` + `__init__.py`**
- **Found during:** Task 1 (every module imports `from mapclass import config`)
- **Issue:** `config.py` is owned by parallel wave-1 Plan 01-01 and is absent in
  this isolated worktree, blocking all imports and tests.
- **Fix:** Reproduced `config.py` exactly to Plan 01-01's documented contract
  (`GCS_BUCKET`, `GCS_DATA_PREFIX`, `GCS_MODELS_PREFIX`, `MODEL_REPO_ID`,
  `MODEL_GCS_DIR`, `MANIFEST_PATH`, `LOCAL_IMAGE_CACHE_DIR`, `DEVICE`,
  `VENDOR_SHA`) so the two worktrees converge byte-compatibly on merge; added
  `__init__.py`. `DEVICE` resolution defers the torch import so pure-stdlib
  consumers/tests work without the pinned venv.
- **Files:** `src/mapclass/config.py`, `src/mapclass/__init__.py`
- **Commit:** `f981080`

**2. [Rule 1 — Bug] Fixed test fakes leaking out of patch scope**
- **Found during:** Task 1 verification (ingest tests `ModuleNotFoundError: PIL`)
- **Issue:** lazily-imported `requests`/`PIL` in `ingest_images` resolve at call
  time; the first test harness popped the fakes before the call.
- **Fix:** introduced a `faked_env` context manager keeping fakes installed for
  the whole test body.
- **Files:** `tests/test_ingest_idempotency.py`
- **Commit:** `f981080`

**3. [Rule 1 — Bug] Removed literal NaFlex symbol from model_loader docstring**
- **Found during:** Task 2 verification (no-NaFlex grep AC failed on the
  docstring's "do NOT use `Siglip2ImageProcessor`" warning).
- **Issue:** the AC requires NO reference to `Siglip2ImageProcessor` /
  `pixel_attention_mask` / `spatial_shapes` *anywhere* in the file; the
  docstring mention tripped the literal grep.
- **Fix:** reworded the docstring to convey the same guidance without writing
  the forbidden tokens.
- **Files:** `src/mapclass/model_loader.py`
- **Commit:** `fbc973f`

### Scope decision (no permission needed — structural, not architectural)

- Did NOT create `tests/conftest.py`, `tests/__init__.py`, or `pytest.ini`
  (Plan 01-01-owned). Each test file is self-contained (adds `src/` to
  `sys.path`) to avoid a merge collision while still being runnable.

## Authentication Gates

None encountered. `ingest_images.verify_adc()` is an explicit fail-fast ADC
check that runs at Task 3 (the human-verify checkpoint), not during Tasks 1-2.

## Checkpoint Status

Task 3 is `type="checkpoint:human-verify" gate="blocking"` — the full 1,544
Rumsey image mirror + SigLIP-2 weights mirror + outcome-manifest completeness +
re-run idempotent convergence (D-01/D-02). This is a long-running live run that
requires GCS ADC, network, the pinned venv, and human completeness/idempotency
judgment. Per `auto_advance: false` and the blocking-checkpoint protocol,
execution STOPS here and returns structured checkpoint state to the
orchestrator. Tasks 1-2 deliver the code that Task 3 exercises.

## Known Stubs

None. All modules are fully wired; no placeholder data paths. (The pinned-venv
absence in this worktree is an environment split, not a code stub — the
orchestrator's post-merge suite and Task 3 exercise the live paths.)

## Self-Check: PASSED

All 12 created files exist on disk; both task commits (f981080, fbc973f) are present in git history.
