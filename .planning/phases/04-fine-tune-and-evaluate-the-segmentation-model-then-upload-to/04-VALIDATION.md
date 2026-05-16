---
phase: 4
slug: fine-tune-and-evaluate-the-segmentation-model-then-upload-to
status: draft
nyquist_compliant: true
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
| **Quick run command** | `python -m pytest -q -m "not integration and not gpu" tests/test_seg_training.py tests/test_seg_eval.py tests/test_seg_gcs.py tests/test_seg_report.py` |
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

> Populated by gsd-planner from PLAN.md tasks. Every Phase-4 auto task that
> touches the training loop, weighted loss, val-carve, eval harness, GCS
> checkpoint I/O, the cost-probe gate, or the comparison report maps to an
> offline/CPU-runnable test (tiny tensors / mini-pyramid fixture, mirroring the
> Phase-3 `test_seg_*` convention). The two non-auto tasks (04-05 T1
> `checkpoint:decision` gate, 04-04 T2 deferred GPU-host runbook artifact) are
> listed as gate/manual rows. GPU-only and GCS-online behaviors are
> `gpu`/`integration`-marked and verified on the separate GPU host (deferred
> 04-HUMAN-UAT-style gate, per CONTEXT D-06).

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-T1 | 01 | 0 | EVAL-02, EVAL-03 | — | N/A | scaffold (collect-only) | `python -m pytest -q tests/test_seg_training.py tests/test_seg_eval.py tests/test_seg_gcs.py --collect-only 2>&1 \| grep -E "test session\|error" \| head -5` | ❌ W0 | ⬜ pending |
| 04-01-T2 | 01 | 1 | EVAL-02 | — | sample_weights key validation | unit | `python -m pytest -q tests/test_seg_training.py -x -k "WeightedJointLoss or ValCarve" 2>&1 \| tail -3` | ❌ W0 | ⬜ pending |
| 04-01-T3 | 01 | 1 | EVAL-02 | — | EVAL-01 val-carve guard | unit | `python -m pytest -q tests/test_seg_training.py -x 2>&1 \| tail -3` | ❌ W0 | ⬜ pending |
| 04-02-T1 | 02 | 1 | D-07, D-09 | T-04 GCS path traversal | config-name regex validation | smoke import | `python -c "import sys; sys.path.insert(0,'scripts'); from seg.gcs_checkpoint import validate_config_name, config_prefix, gcs_save_checkpoint, gcs_latest_checkpoint; validate_config_name('siglip-b'); print('ok')"` | ❌ W0 | ⬜ pending |
| 04-02-T2 | 02 | 1 | D-07, D-09 | T-04 corrupt-ckpt DoS | mocked GCS round-trip | unit (mock) | `python -m pytest -q tests/test_seg_gcs.py -x 2>&1 \| tail -3` | ❌ W0 | ⬜ pending |
| 04-03-T1 | 03 | 2 | EVAL-02 | T-04 split.json traversal | test-set enumeration guard | smoke import | `python -c "import sys; sys.path.insert(0,'scripts'); from evaluate_seg import evaluate_joint_nll, load_test_pyramid_dirs, write_nll_metrics, main; print('ok')"` | ❌ W0 | ⬜ pending |
| 04-03-T2 | 03 | 2 | EVAL-02, EVAL-03 | — | EVAL-01 leakage guard | unit | `python -m pytest -q tests/test_seg_eval.py -x 2>&1 \| tail -3` | ❌ W0 | ⬜ pending |
| 04-04-T1 | 04 | 3 | EVAL-02 | — | EVAL-01 train-only loop | unit (probe) | `python -m pytest -q tests/test_seg_training.py -x -k "Probe" 2>&1 \| tail -3` | ❌ W0 | ⬜ pending |
| 04-04-T2 | 04 | 3 | EVAL-02, EVAL-03 | — | D-08 weights-never-committed runbook | manual artifact (+ regression `python -m pytest -q tests/test_seg_training.py -x 2>&1 \| tail -2`) | manual | 04-HUMAN-UAT.md deferred GPU-host runbook — manual content confirmation (8 sections, DEFERRED header, --pretrained callout, D-08 hand-back) | n/a | ⬜ pending (manual) |
| 04-05-T1 | 05 | 4 | EVAL-02, EVAL-03 | T-04 D-05 decision provenance | probe-numbers-pasted-before-choice | gate (`checkpoint:decision`, blocking) | none — autonomous:false human gate; user pastes PROBE RESULT + selects option-a/b/c, recorded in 04-05-SUMMARY | n/a | ⬜ pending (gate) |
| 04-05-T2 | 05 | 4 | EVAL-02, EVAL-03 | T-04 git commit scope (D-08) | metrics/manifest-only commit | unit | `python -m pytest -q tests/test_seg_report.py -x 2>&1 \| tail -3` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

> Coverage: every code-producing auto task (04-01 T1/T2/T3, 04-02 T1/T2,
> 04-03 T1/T2, 04-04 T1, 04-05 T2) maps to an automated command. The remaining
> two tasks are intrinsically non-automatable: 04-04 T2 produces the deferred
> GPU-host runbook artifact (manual content confirmation; carries a regression
> automated test as a secondary check) and 04-05 T1 is the blocking
> `checkpoint:decision` cost-probe gate.

---

## Wave 0 Requirements

- [ ] `tests/test_seg_training.py` — training loop, weighted-loss, teacher-forced coarse prior, val-carve zero-leakage (RESEARCH gap set)
- [ ] `tests/test_seg_eval.py` — joint per-pixel NLL harness over mini-pyramid; numerical-stability + pixel-averaging
- [ ] `tests/test_seg_gcs.py` — GCS checkpoint write/auto-resume contract (mocked `gs://`; `gpu`/`integration` skip on planning VM)
- [ ] `tests/test_seg_report.py` — comparison-report renderer from synthetic metrics JSON fixtures (offline, no torch)
- [ ] Reuse existing Phase-3 mini-pyramid fixture / `conftest.py` — no new framework install (pytest already present)

*Wave 0 covers the test gaps identified in 04-RESEARCH.md plus the offline report-render test.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Real GPU end-to-end training run + measured GPU-hrs/epoch (cost probe) | EVAL-02 (SC#1, SC#4) | No GPU on planning VM; needs the separate GPU box | Run the probe (SigLIP Variant B) on the GPU host per the deferred 04-HUMAN-UAT gate; record GPU-hrs/epoch + projected grid cost at the `checkpoint:decision` gate |
| Periodic GCS checkpoint + auto-resume against the live bucket | EVAL-02 (SC#3) | Requires real `gs://mapclass-training-northeast1/` + gcsfs on GPU host | On GPU host: interrupt a run mid-training, relaunch, confirm resume from latest GCS checkpoint |
| Joint-NLL numbers on the real held-out synthetic `test/` + comparison report | EVAL-02, EVAL-03 | Requires trained checkpoints (GPU) and full Phase-2 dataset | On GPU host: run the eval harness over real `test/`, produce metrics JSON + written A/B×backbone report |
| 04-HUMAN-UAT.md deferred GPU-host runbook content (04-04 T2) | EVAL-02, EVAL-03 | Operational runbook artifact; no automatable content assertion | Confirm DEFERRED header, gcloud ADC auth, gcsfs install, timm-tag verification, exact probe command, PROBE RESULT capture, DINOv2/Swin `--pretrained` callout, resume verification, eval run, D-08 weights-never-committed hand-back |
| D-05 cost-probe grid-breadth decision (04-05 T1) | EVAL-02, EVAL-03 | Blocking `checkpoint:decision` gate; human picks breadth with measured cost in hand | User pastes the GPU-host PROBE RESULT line, selects option-a/b/c; choice + numbers recorded in 04-05-SUMMARY |

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references (RESEARCH gaps + offline report test)
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending (wave 0 not yet executed — `wave_0_complete: false`)
