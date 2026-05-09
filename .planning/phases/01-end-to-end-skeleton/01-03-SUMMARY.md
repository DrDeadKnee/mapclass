---
phase: 01-end-to-end-skeleton
plan: 03
subsystem: eval
tags: [eval-03, pre-registration, pitfall-2, paligemma, bellwether, markdown-protocol]

requires:
  - phase: 01-end-to-end-skeleton
    provides: mapclass/configs/loss_weights.yaml + mapclass/configs/splits.json (Plan 01-01) — the two artifacts whose sha256 are pinned in this protocol
provides:
  - mapclass/configs/EVAL-03_protocol.md — pre-registered bellwether contract for the small-backbone-vs-PaliGemma NLL comparison
  - DECIDE_AT_PHASE_5 literal sentinel — Phase 5's eval_03_compare.py reads this file and refuses to run while the placeholder is still present (D-08 mechanical guard)
  - Pinned sha256 hashes of splits.json (a7b61846...) and loss_weights.yaml (10f7e4dd...) — Phase 5's compare script enforces byte-identity at runtime
affects: [phase-05, eval, paligemma]

tech-stack:
  added: []
  patterns:
    - Two-commit policy (D-08): the threshold replacement is a discrete, auditable git event; the replacement commit message must reference Phase 3 v0 NLL stability
    - Strict fairness statement (D-07): no Permitted-Differences table; only backbone field (+ dtype + 224 input — the one acknowledged deviation under KNOWN TENSION) differs between v0.yaml and v0_paligemma.yaml

key-files:
  created:
    - mapclass/configs/EVAL-03_protocol.md
  modified: []

key-decisions:
  - "Pre-registered the comparison structure in Phase 1 commit history before any real training run starts (PITFALL 2 prevention setup)"
  - "Embedded sha256 hashes of splits.json and loss_weights.yaml inline so Phase 5's compare script can byte-verify the artifacts haven't drifted"
  - "Kept the kill-switch numeric threshold as the literal DECIDE_AT_PHASE_5 placeholder (D-06 calibrated-after-v0-results compromise + D-08 mechanical guard)"

patterns-established:
  - "Pattern: Pre-registered audit-trail Markdown protocol — single page, embedded hashes, deferred-numeric placeholder enforced by a downstream script grep"

requirements-completed: [MODEL-03]

duration: ~5min
completed: 2026-05-09
---

# Phase 01 / Plan 03: EVAL-03 Pre-Registration Protocol Summary

**One-page Markdown protocol pre-registering the small-backbone-vs-PaliGemma NLL bellwether comparison structure, with embedded sha256 hashes of splits.json + loss_weights.yaml and the DECIDE_AT_PHASE_5 deferred-threshold sentinel.**

## Performance

- **Duration:** ~5 min (file authored in failed prior agent's worktree, adopted by orchestrator after hash verification)
- **Started:** 2026-05-09 (Wave 3 of /gsd-execute-phase 1)
- **Completed:** 2026-05-09
- **Tasks:** 1 / 1
- **Files modified:** 1 (one new file)

## Accomplishments

- `mapclass/configs/EVAL-03_protocol.md` authored — 66 lines, 8 `## ` second-level sections, single page of Markdown
- All six required ROADMAP / CONTEXT D-05/D-06 sections present: held-out split hash, loss-weights file hash, augmentation policy, fine-tune budget shape, evaluation seed, kill-switch criterion
- The literal `DECIDE_AT_PHASE_5` appears verbatim — Phase 5's `eval_03_compare.py` (out of Phase 1 scope) will grep for this exact string and refuse to run while present
- Embedded sha256 hex digests:
  - `mapclass/configs/splits.json` → `a7b61846833381501db6b3d42a80575cd28471b62ccc9c21135051a70211d421` (verified at commit time)
  - `mapclass/configs/loss_weights.yaml` → `10f7e4dd4a3e7759152019721644939646f8b20f38dbde2385632b206c9c09ab` (verified at commit time, matches `LossWeights.load(...).source_class_weights_hash`)
- Phase 5 enforcement section documents the D-08 two-commit policy ("MUST be replaced ... in a SEPARATE COMMIT before `eval_03_compare.py` runs")
- Strict fairness statement (D-07) — "There is no 'Permitted Differences' table for EVAL-03"; only backbone, dtype, and 224-vs-384 input resolution differ

## Plan-Level Verification Gates (10/10 PASS)

1. ✅ `test -f mapclass/configs/EVAL-03_protocol.md`
2. ✅ `grep -q 'DECIDE_AT_PHASE_5'` (D-08 literal sentinel preserved verbatim)
3. ✅ `wc -l = 66` (within 30–80 range)
4. ✅ Six required section keyword hits = 10 (≥ 6)
5. ✅ Loss-weights sha256 in file matches `python -c "hashlib.sha256(open('mapclass/configs/loss_weights.yaml','rb').read()).hexdigest()"`
6. ✅ Splits sha256 in file matches `python -c "hashlib.sha256(open('mapclass/configs/splits.json','rb').read()).hexdigest()"`
7. ✅ "Phase 5 enforcement" / "two-commit" present
8. ✅ "separate commit" policy stated
9. ✅ No alternate placeholders (`<TBD>`, `XXX`, `???`) — only `DECIDE_AT_PHASE_5` is the legitimate placeholder
10. ✅ Strict-fairness language present (D-07)

## Files Created/Modified

- `mapclass/configs/EVAL-03_protocol.md` — pre-registered bellwether protocol, one page, six required sections, embedded sha256 hashes, DECIDE_AT_PHASE_5 literal, Phase 5 enforcement + strict fairness sections.

## Decisions Made

- **Adopted the failed-Wave-3-agent's authored file directly** (rather than respawning a fresh executor): the agent had completed the authoring work + correct hash computation before being terminated by an external usage-limit cutoff; the file passed every plan-level verification gate byte-for-byte against current `splits.json` and `loss_weights.yaml`. Respawning would have re-derived the same content while spending another wave of compute. Orchestrator verified the file in-place against the live hashes before adoption.

## Deviations from Plan

None at the file-content level — the document follows the PLAN.md authoring template line-for-line. The execution path deviated (orchestrator adopted the prior agent's untracked output rather than respawning) — this is a workflow-level optimization, not a content deviation. The plan's `<verify><automated>` block was run end-to-end against the adopted file and all gates pass.

## Issues Encountered

- **Wave 3 executor agent (afe7babc06df74d00) terminated mid-flight** by the Anthropic plan's "out of extra usage" cutoff before it could produce its SUMMARY.md or commit. The agent had completed the file authoring (14 tool uses) but never reached the commit step.
- **Resolution:** Orchestrator verified the untracked output against all 10 plan-level acceptance gates, confirmed byte-identity of the embedded hashes vs the live `splits.json` / `loss_weights.yaml`, copied the file into the orchestrator working tree, wrote this SUMMARY.md, and prepared to commit + clean up the failed worktree.

## Next Phase Readiness

- All three Phase 01 plans (01-01 data + config foundation, 01-02 model + CLIs, 01-03 EVAL-03 protocol) are complete with SUMMARY.md.
- The walking skeleton's audit-trail layer is now committed: Phase 5's compare script has a pinned, hash-verified contract to run against.
- The single known phase-level gap (data/renders/ stub data is gitignored, so the smoke train cannot run from a fresh clone) is documented in 01-02-SUMMARY.md and will be surfaced again in the phase verification report.

---
*Phase: 01-end-to-end-skeleton*
*Completed: 2026-05-09*
