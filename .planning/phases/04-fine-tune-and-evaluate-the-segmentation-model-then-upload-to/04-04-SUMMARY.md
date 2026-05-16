# 04-04 SUMMARY — finetune_seg.py + offline probe + GPU-host gate

**Plan:** 04-04
**Status:** Complete (recovered after a transient API-500 interrupt; see Deviations)
**Tasks:** 2/2

## What was built

- `scripts/finetune_seg.py` (406 lines) — teacher-forced coarse-to-fine
  training loop with: trainable-only AdamW optimizer, seeded train→val carve
  (via `seg.train_utils.carve_train_val`), per-pyramid `train_step_variant_a/b`,
  GCS checkpoint + auto-resume wiring (`seg.gcs_checkpoint`), and
  `--probe` mode that runs exactly one epoch, prints a `PROBE RESULT |`
  line (measured GPU-hrs/epoch + projected full-grid cost), and `sys.exit(0)`
  without committing to the full grid. CLI flags: `--probe`,
  `--pretrained/--no-pretrained`, `--amp`, `--ckpt-every`, etc.
- `tests/test_seg_training.py::TestProbeMode` — offline contract: probe runs
  one epoch over a stub-backbone `mini_pyramid` fixture (no GPU/GCS/network),
  prints `PROBE RESULT`, exits 0, and opens **no** `test/`-component path
  (EVAL-01 zero-leakage). GCS calls mocked to no-ops.
- `04-HUMAN-UAT.md` — deferred GPU-host gate with exact hand-commands for the
  real probe/train/eval run, pretrained-flag callout, resume verification, and
  push-before-pull discipline (mirrors `02-HUMAN-UAT.md`).

## Commits

- `c1cd306` test(04-04): add failing TestProbeMode for finetune_seg probe mode
- `eaafe64` feat(04-04): implement finetune_seg.py training loop + probe mode
- `ecee16d` docs(04-04): author 04-HUMAN-UAT deferred GPU-host gate

## Verification

- `pytest tests/test_seg_training.py::TestProbeMode` → **1 passed in 27.45s**
  (run under `ulimit -v 45000000`, `-p no:cacheprovider`).
- Memory peak during the probe run: **3.58 GiB** system used / 2.81 GiB
  total RSS — no OOM event. The plan exercises the
  `train_step_variant_a/b` path; the per-tile-backward fix (`229539d`) holds
  on the full finetune loop, not just the unit tests.
- Real GPU training / live GCS / real-test eval remain the **deferred
  GPU-host gate** (`04-HUMAN-UAT.md`), out of scope for autonomous execute by
  design (plan truths 15/19, lines 53–57).

## Deviations

1. **API-500 mid-execution interrupt.** The executor agent hit a transient
   server-side API 500 after committing Tasks 1–2 but before committing the
   uncommitted `04-HUMAN-UAT.md` and writing this SUMMARY. The orchestrator
   salvaged the untracked UAT file (commit `ecee16d`), merged the partial
   branch into `phase4`, verified the probe test GREEN under the diagnostic
   guardrails, and authored this SUMMARY during recovery. No work was lost; no
   re-implementation was needed (both plan tasks' artifacts were already
   committed and correct).
