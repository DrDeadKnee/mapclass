---
phase: 03
plan: "01"
subsystem: seg-test-infra
tags:
  - test-infrastructure
  - dependencies
  - tdd
  - nyquist
dependency_graph:
  requires:
    - scripts/tiling.py   # mini_pyramid fixture calls real tiling.tile()
    - tests/conftest.py   # existing Phase-2 fixture idioms mirrored
  provides:
    - scripts/seg/__init__.py    # importable seg package
    - requirements.txt timm pin  # DINOv2/Swin backbone dependency
    - tests/conftest.py::mini_pyramid
    - tests/conftest.py::stub_vision_config
    - tests/test_seg_backbones.py
    - tests/test_seg_decoder.py
    - tests/test_seg_recursive.py
    - tests/test_seg_dataset.py
    - tests/test_seg_smoke.py
    - tests/integration/test_seg_online.py
  affects:
    - All Phase-3 plans (03-02..03-05) depend on these test scaffolds as Nyquist gates
tech_stack:
  added:
    - timm>=1.0.27   # DINOv2 + Swin benchmark backbones (EVAL-03)
  patterns:
    - TDD RED: test scaffolds fail with ImportError until seg.* modules land
    - mini_pyramid fixture uses real tiling.tile() producer (D-04 compliance)
    - offline-first: all unit fixtures use Pillow + tiling only, no torch/transformers
    - import-guard pattern: try/except at module level + pytest.fail in test body for clean collection
key_files:
  created:
    - requirements.txt          # timm>=1.0.27 added; transformers tightened to >=5.8
    - scripts/seg/__init__.py   # package marker, __all__ = []
    - tests/test_seg_backbones.py  # 15 tests: Backbone protocol, strides, Gemma-absence, Variant B
    - tests/test_seg_decoder.py    # 9 tests: decoder+head output shapes, Pitfall 4 guard
    - tests/test_seg_recursive.py  # 8 tests: prior crop alignment, cold-start, PriorEncoder
    - tests/test_seg_dataset.py    # 9 tests: split safety, weight surfacing, batch shapes
    - tests/test_seg_smoke.py      # 7 tests: both variants end-to-end forward
    - tests/integration/test_seg_online.py  # 3 tests: gated real adapter + real tile
  modified:
    - tests/conftest.py   # mini_pyramid + stub_vision_config fixtures appended
decisions:
  - "Tightened transformers pin to >=5.8 (was >=4.41) with inline comment documenting that 5.x puts PaliGemma submodules at .model.vision_tower; resolves RESEARCH Pitfall 3 deterministically"
  - "Import-guard pattern chosen over pytest.importorskip to satisfy dual constraint: collect cleanly (--co exit 0) AND fail assertively at runtime with clear ImportError message (not silent skip)"
  - "mini_pyramid fixture returns single pyramid directory (not the map dir or pyramid root) to match the tiny_geotiff return-path idiom"
  - "stub_vision_config uses SigLIP-So400m/14 numbers (hidden_size=1152, patch_size=14, 27 layers) for offline shape contracts without loading 3B weights"
metrics:
  started: "2026-05-15"
  completed: "2026-05-15"
  tasks_completed: 3
  tasks_total: 3
  files_created: 8
  files_modified: 2
---

# Phase 3 Plan 01: Seg Test Infra + Dependency Resolution Summary

Wave 0 complete: timm declared, transformers 5.x path resolved, seg package importable, 50 Nyquist-gate tests collecting cleanly as TDD RED failures awaiting 03-02..03-05 implementation.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Resolve and pin dependencies + seg package | fbc9c1e | requirements.txt, scripts/seg/__init__.py |
| 2 | Shared offline mini-pyramid + stub-config fixtures | a352327 | tests/conftest.py |
| 3 | Seg test scaffolds (5 unit + 1 gated integration) TDD RED | cbd1130 | 6 new test files |

## What Was Built

**Dependency resolution (Task 1):**
- `timm>=1.0.27` added to `requirements.txt` — the only hard-missing dependency
  for DINOv2/Swin backbone construction (EVAL-03 blocker resolved)
- `transformers>=4.41` tightened to `transformers>=5.8` with an inline comment:
  `# 5.x layout: PaliGemma submodules at .model.vision_tower (not .vision_tower)`
  This resolves RESEARCH Pitfall 3 deterministically: the resolved 5.x path
  (`.model.vision_tower`) is documented at the constraint level.
- The environment has transformers/timm/torch absent (conda base only); the plan
  documents this as the RESEARCH Environment Availability fallback case — all
  downstream tests use stub configs and mini-pyramid fixtures for offline runs.
- `scripts/seg/__init__.py`: package marker (`__all__ = []`) enabling
  `from seg.X import ...` under `pytest.ini pythonpath=scripts`.

**Shared fixtures (Task 2):**
- `mini_pyramid(tmp_path)`: synthesises a 1792x1792 map using the same
  `_make_map_dir` pattern as `test_tiling.py`, calls the real `tiling.tile()`
  producer, returns the first pyramid directory (contains a genuine 21-tile
  `pyramid.json`). Fully offline — Pillow + tiling only.
- `stub_vision_config()`: `SimpleNamespace` with `hidden_size=1152`,
  `patch_size=14`, `num_hidden_layers=27` — SigLIP-So400m/14 values for offline
  shape tests (RESEARCH Assumption A1 fallback).
- All 64 Phase-2 tests remain green (conftest regression check: passed).

**Test scaffolds (Task 3 — TDD RED):**
- 50 tests collected cleanly (`pytest --co -q` exits 0).
- 47 unit tests fail with clear `ImportError` messages (`seg.backbones not yet
  implemented (plan 03-02 pending): No module named 'seg.backbones'`) — live
  Nyquist gates for 03-02..03-05.
- 3 integration tests skip with reason when Phase-1 adapter / Phase-2 data
  absent (OQ1 RESOLVED; skip message names exact CLI flags `--adapter-dir`).

**test_seg_backbones.py (15 tests):**
- `TestBackboneProtocol`: SigLIP/DINOv2/Swin each expose `feature_strides` and
  `feature_channels`; `forward()` returns `list[Tensor]` with matching channels
- `TestSiglipShape`: (B,C,h,w) grids for 224/448/896; no-CLS-token assertion
- `TestNoGemma`: `language_model` parameter + `gemma` module-type absence from
  BOTH Variant A and Variant B (PHASE-03 SC#2)
- `TestVariantBPatchEmbed`: 15-ch input accepted; `patch_embed_weight()` has 15
  input channels; channels 3:15 are zero-init; channels 0:3 match pretrained

**test_seg_decoder.py (9 tests):**
- SegDecoder + LandCoverHead emits (B,9,H,W) for ViT single-scale and Swin 4-level
- SegDecoder + TopographyHead emits (B,3,H,W) at all 3 tile sizes (224/448/896)
- `out_channels` attribute required; batch dimension propagates correctly

**test_seg_recursive.py (8 tests):**
- `crop_prior_to_child`: one-hot blob in parent (0,0) quadrant lands in `448_0`
  and NOT in siblings (Pitfall 5 guard — manifest-box crop, not whole-parent resize)
- `cold_start_prior(batch, tile_size)`: shape (B,12,S,S) all-zeros at 896
- `PriorEncoder(in_channels, out_channels, target_size)`: Variant A decoder-level
  injection; output shape matches target; has trainable parameters

**test_seg_dataset.py (9 tests):**
- `PyramidDataset(pyramid_dir)`: `image`/`land_cover`/`topography`/`sample_weights`
  keys; image is (3,H,W) float32 in [0,1]; label shapes match image
- `tile_path(i)` exposed so split-safety test can inspect path parts
- No `test/` component in any enumerated path (EVAL-01 split-safety, mirrors
  `test_tiling.py:213-215`)
- Raises `ValueError` or `FileNotFoundError` on empty directory

**test_seg_smoke.py (7 tests):**
- Variant A (`SegModelVariantA`): SigLIP backbone + frozen + decoder-level prior
- Variant B (`SegModelVariantB`): SigLIP backbone + 15-ch widened patch-embed
- Both: (B,9,H,W)+(B,3,H,W) correct shapes; no NaN; `torch.no_grad` only (D-06)
- Variants produce distinct outputs (independent architectures)

**tests/integration/test_seg_online.py (3 gated tests):**
- `@pytest.mark.integration` on all 3 tests
- Skips with clear reason when `--adapter-dir` / `--pyramid-dir` absent
- Covers: real LoRA load, Gemma-absence on real model, Variant A + Variant B
  real-tile forward at all 3 pyramid scales

## Decisions Made

1. **Transformers pin tightened to >=5.8** — RESEARCH Pitfall 3 resolved
   deterministically by pinning the major version and documenting the 5.x
   submodule path in requirements.txt as a comment. Alternative (defensive
   `getattr` fallback) deferred to the backbone implementation (03-02).

2. **Import-guard pattern** — `try/except ImportError` at module level stores the
   error; test bodies call `_require_backbones()` which `pytest.fail()`s with a
   clear message. This satisfies both: (a) `--co` collects cleanly (module
   evaluates without raising), (b) running fails assertively (not silently
   skipped/xfailed). Pattern is unique in the codebase.

3. **mini_pyramid returns first pyramid directory, not pyramid root** — Consistent
   with `tiny_geotiff` return-path idiom (returns the usable artifact directly,
   not a parent container). Tests that need the root can call `.parent`.

4. **stub_vision_config as SimpleNamespace** — Lightweight, no class definition
   needed; mirrors the project's preference for simple data holders.

## Deviations from Plan

None — plan executed exactly as written.

Environment deviation (documented, not a code deviation): transformers, timm, and
torch are absent in the execution shell (conda base env). This is the documented
RESEARCH Environment Availability fallback case. The requirements.txt edits are
correct; downstream tests use stub configs + mini_pyramid fixtures as designed.

## TDD Gate Compliance

Wave 0 delivers the RED gate for Phase 3:
- RED (this plan): 50 tests collect; 47 unit tests fail with ImportError; 3
  integration tests skip-with-reason.
- GREEN: plans 03-02..03-05 implement `seg.backbones`, `seg.decoder`, `seg.heads`,
  `seg.model`, `seg.recursive`, `seg.dataset` to pass these tests.
- REFACTOR: post-GREEN cleanup per plan 03-05.

## Known Stubs

None — no seg implementation modules exist yet; all stubs are intentional
TDD RED state, not wired-data gaps. The plan's goal (test infrastructure +
dependency resolution) is fully achieved.

## Threat Surface Scan

No new network endpoints, auth paths, or trust-boundary changes introduced. All
fixture filesystem writes are confined to pytest `tmp_path` (T-03-02 mitigation).
`timm>=1.0.27` supply-chain trust: PyPI-verified version (T-03-01 mitigation).
Integration test only reads user-supplied `--adapter-dir` at test time (T-03-03
accepted).

## Self-Check: PASSED

All files exist and all commits verified:
- `requirements.txt` FOUND
- `scripts/seg/__init__.py` FOUND
- `tests/conftest.py` FOUND (modified)
- `tests/test_seg_backbones.py` FOUND
- `tests/test_seg_decoder.py` FOUND
- `tests/test_seg_recursive.py` FOUND
- `tests/test_seg_dataset.py` FOUND
- `tests/test_seg_smoke.py` FOUND
- `tests/integration/test_seg_online.py` FOUND
- `.planning/phases/03-build-a-dense-semantic-segmentation-pipeline/03-01-SUMMARY.md` FOUND
- Commit `fbc9c1e` (Task 1) FOUND
- Commit `a352327` (Task 2) FOUND
- Commit `cbd1130` (Task 3) FOUND
