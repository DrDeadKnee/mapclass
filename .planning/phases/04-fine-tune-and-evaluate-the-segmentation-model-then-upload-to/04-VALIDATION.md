---
phase: 4
slug: fine-tune-and-evaluate-the-segmentation-model-then-upload-to
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-16
---

# Phase 4 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing — `pytest.ini`, used through Phases 2–3) |
| **Config file** | `pytest.ini` (root) |
| **Quick run command** | `python -m pytest -q -m "not integration and not gpu" tests/test_seg_training.py tests/test_seg_eval.py tests/test_seg_gcs.py` |
| **Full suite command** | `python -m pytest -q -m "not integration and not gpu"` |
| **Estimated runtime** | ~60–120 seconds (offline/CPU; GPU + GCS paths are `gpu`/`integration`-marked and skip-with-reason on the planning VM) |

---

## Sampling Rate

- **After every task commit:** Run quick run command
- **After every plan wave:** Run full suite command
- **Before `/gsd-verify-work`:** Full offline suite must be green
- **Max feedback latency:** 120 seconds

---

## Per-Task Verification Map

> Populated by gsd-planner from PLAN.md tasks. Every Phase-4 task that touches
> the training loop, weighted loss, val-carve, eval harness, GCS checkpoint I/O,
> or the cost-probe gate must map to an offline/CPU-runnable test (tiny tensors
> / mini-pyramid fixture, mirroring the Phase-3 `test_seg_*` convention).
> GPU-only and GCS-online behaviors are `gpu`/`integration`-marked and verified
> on the separate GPU host (deferred 04-HUMAN-UAT-style gate, per CONTEXT D-06).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 4-01-01 | 01 | 0 | EVAL-02 | — | N/A | unit | `python -m pytest -q tests/test_seg_training.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_seg_training.py` — training loop, weighted-loss, teacher-forced coarse prior, val-carve zero-leakage (RESEARCH gap set)
- [ ] `tests/test_seg_eval.py` — joint per-pixel NLL harness over mini-pyramid; numerical-stability + pixel-averaging
- [ ] `tests/test_seg_gcs.py` — GCS checkpoint write/auto-resume contract (mocked `gs://`; `gpu`/`integration` skip on planning VM)
- [ ] Reuse existing Phase-3 mini-pyramid fixture / `conftest.py` — no new framework install (pytest already present)

*Wave 0 covers the 9 test gaps identified in 04-RESEARCH.md.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real GPU end-to-end training run + measured GPU-hrs/epoch (cost probe) | EVAL-02 (SC#1, SC#4) | No GPU on planning VM; needs the separate GPU box | Run the probe (SigLIP Variant B) on the GPU host per the deferred 04-HUMAN-UAT gate; record GPU-hrs/epoch + projected grid cost at the `checkpoint:decision` gate |
| Periodic GCS checkpoint + auto-resume against the live bucket | EVAL-02 (SC#3) | Requires real `gs://mapclass-training-northeast1/` + gcsfs on GPU host | On GPU host: interrupt a run mid-training, relaunch, confirm resume from latest GCS checkpoint |
| Joint-NLL numbers on the real held-out synthetic `test/` + comparison report | EVAL-02, EVAL-03 | Requires trained checkpoints (GPU) and full Phase-2 dataset | On GPU host: run the eval harness over real `test/`, produce metrics JSON + written A/B×backbone report |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references (9 RESEARCH gaps)
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
