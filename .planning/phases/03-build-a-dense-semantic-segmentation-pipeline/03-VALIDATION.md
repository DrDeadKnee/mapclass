---
phase: 3
slug: build-a-dense-semantic-segmentation-pipeline
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-15
---

# Phase 3 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `03-RESEARCH.md` § Validation Architecture + `03-CONTEXT.md`
> (D-03a two prior-injection variants; D-06 construction-only).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest ≥8 (+ pytest-mock ≥3) — already installed from Phase 2 |
| **Config file** | `pytest.ini` (`testpaths = tests`, `pythonpath = scripts`, marker `integration`) — exists |
| **Quick run command** | `pytest tests/test_seg_*.py -x -q -m "not integration"` |
| **Full suite command** | `pytest -q -m "not integration"` (offline) · `pytest -q -m integration` (network/heavy phase gate) |
| **Estimated runtime** | quick < 60s · full offline < 6min · integration gated on Phase-1 adapter + Phase-2 data |

---

## Sampling Rate

- **After every task commit:** `pytest tests/test_seg_*.py -x -q -m "not integration"`
- **After every plan wave:** `pytest -q -m "not integration"` (full offline, incl. Phase-2 tests — must stay green)
- **Before `/gsd-verify-work`:** full offline green; `pytest -q -m integration` green OR explicitly skipped-with-reason if Phase-1/Phase-2 artifacts unavailable
- **Max feedback latency:** 60s (quick) / 360s (wave)

---

## Per-Task Verification Map

> Both D-03a variants must be covered: **A** = frozen backbones + decoder-level
> prior; **B** = all-3 backbones' patch-embed widened to 15-ch + trainable,
> input-level prior. Phase 3 constructs both; neither is trained (D-06/D-06a).

| Plan Area | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-----------|-------------|-----------------|-----------|-------------------|-------------|--------|
| backbones | PHASE-03 | SigLIP backbone yields (B,C,h,w) grids at declared strides for 224/448/896 | unit | `pytest tests/test_seg_backbones.py -x -q` | ❌ W0 | ⬜ pending |
| backbones | PHASE-03 | No Gemma/`language_model` param or module reachable from EITHER variant's seg forward (SC#2) | unit | `pytest tests/test_seg_backbones.py::test_no_gemma -x` | ❌ W0 | ⬜ pending |
| backbones | PHASE-03 | Variant B widened patch-embed = 15-ch, extras zero-init, pretrained RGB weights preserved, all 3 backbones | unit | `pytest tests/test_seg_backbones.py -k variant_b_patch_embed -x` | ❌ W0 | ⬜ pending |
| decoder | PHASE-03 | Shared decoder + 2 heads emit (B,9,H,W)+(B,3,H,W) at full tile res | unit | `pytest tests/test_seg_decoder.py -x -q` | ❌ W0 | ⬜ pending |
| recursive prior | PHASE-03 | Variant A: 12-ch prior-encoder injects at decoder; correct working-res align | unit | `pytest tests/test_seg_recursive.py -k variant_a -x` | ❌ W0 | ⬜ pending |
| recursive prior | PHASE-03 | Prior crops correct parent quadrant per `pyramid.json` box (D-03/D-04), zeros at 896 cold-start | unit | `pytest tests/test_seg_recursive.py -x -q` | ❌ W0 | ⬜ pending |
| backbone protocol | EVAL-03 | DINOv2 + Swin satisfy the same `Backbone` protocol (declared strides/channels; same decoder runs) | unit | `pytest tests/test_seg_backbones.py -k "dino or swin" -x` | ❌ W0 | ⬜ pending |
| dataset | PHASE-03 | Dataloader reads Phase-2 pyramids, surfaces sample_weights, `train/`-only (no `test/` leakage) | unit | `pytest tests/test_seg_dataset.py -x -q` | ❌ W0 | ⬜ pending |
| smoke | PHASE-03 | Both variants: end-to-end forward on a synthetic mini-pyramid — correct shapes, no NaN | smoke | `pytest tests/test_seg_smoke.py -x -q` | ❌ W0 | ⬜ pending |
| integration | PHASE-03 | Real Phase-1 SigLIP-LoRA load + real Phase-2-tile forward (both variants) | integration | `pytest tests/integration/test_seg_online.py -m integration` | ❌ W0 (gated) | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `requirements.txt`: add `timm` (currently missing — backs DINOv2/Swin behind D-05); pin/resolve `transformers` version (4.x `.vision_tower` vs 5.x `.model.vision_tower` attribute path — Pitfall 3)
- [ ] `tests/conftest.py` (or local): synthetic mini-pyramid fixture builder (reuse `tests/test_tiling.py::_make_map_dir` pattern) + tiny-stub-SigLIP-config fixture for offline forward tests
- [ ] `tests/test_seg_backbones.py` — backbone shape/stride contract, Gemma-absence (both variants), Variant-B widened patch-embed, DINOv2/Swin protocol conformance
- [ ] `tests/test_seg_decoder.py` — shared decoder + 2-head output shapes
- [ ] `tests/test_seg_recursive.py` — prior quadrant alignment vs `pyramid.json`; Variant-A decoder-injection path; 896 cold-start zeros
- [ ] `tests/test_seg_dataset.py` — split safety (`train/`-only), weight surfacing, batch shapes
- [ ] `tests/test_seg_smoke.py` — both-variant synthetic mini-pyramid end-to-end forward
- [ ] `tests/integration/test_seg_online.py` — real adapter + real tile (gated; skip-with-reason if Phase-1/2 artifacts absent)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Variant A vs Variant B comparative performance | PHASE-03/EVAL-03 | Requires training + joint-NLL eval — that is **Phase 4** scope (D-06a). Phase 3 only constructs both. | Deferred to Phase 4: train A and B across the backbone matrix, compare joint per-pixel NLL on the held-out synthetic test set. |
| Real Phase-1 adapter availability | PHASE-03 | Phase-1 ran pre-bootstrap; no `checkpoints/` in repo (OQ1) | Provide `--adapter-dir`/`--model-id`; run the gated integration test on a host where the adapter + Phase-2 tiles exist. |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s (quick) / 360s (wave)
- [ ] Both D-03a variants (A frozen+decoder, B trainable+input) independently shape-verified
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
