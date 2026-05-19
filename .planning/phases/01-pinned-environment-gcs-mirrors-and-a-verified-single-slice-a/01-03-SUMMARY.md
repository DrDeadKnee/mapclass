---
phase: 01-pinned-environment-gcs-mirrors-and-a-verified-single-slice-a
plan: 03
subsystem: attribution
tags: [siglip2, dynamiclrp, lrp, attribution, jupyter, vram, matplotlib]

# Dependency graph
requires:
  - phase: 01-01
    provides: pinned env, vendored dynamicLRP @ SHA 405e7424, src/mapclass package + config
  - phase: 01-02
    provides: manifest reader, idempotent full 1,544 GCS mirror, model/data loaders
provides:
  - "src/mapclass/attribution.py — SigLIP-2 forward + LRP in one scope against logits_per_image[0,0]; coverage_probe helper; peak-VRAM capture (attribute() raises RuntimeError on SigLIP-2 — documented op-coverage finding)"
  - "src/mapclass/overlay.py — signed/magnitude 27x27 patch grid (6-px discard), zero-centered bwr composite, draw_patch_grid (delivered + unit-tested; not exercised end-to-end because no SigLIP-2 relevance is produced)"
  - "notebooks/01_single_slice.ipynb — honest end-to-end SMOKE TEST (forward + peak forward VRAM + caught coverage finding + source map); NOT the original overlay + three-controls finish line"
  - "FINDING: dynamicLRP op-coverage does NOT cover SigLIP-2-so400m (split_with_sizes / SplitWithSizesBackward0 in the MAP-pool head) — recorded per-model result for the reframed multi-model dynamic-LRP comparison"
  - "Peak forward VRAM for the single so400m forward on the locked slice = 4.326 GB (4,644,488,704 bytes) — STATE.md peak-VRAM blocker CLOSED"
affects: [phase-02-sweep, multi-model-comparison, project-md-reconciliation]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Honest-finding notebook: try/except catches the engine RuntimeError and renders a labeled FINDING markdown/stdout cell so the notebook exits 0 as a documented result, not an uncaught traceback"
    - "Generator-authored notebook (notebooks/_build_01_single_slice.py is source of truth; the .ipynb is the committed artifact, regenerated + executed headless via nbconvert with the mapclass kernel)"

key-files:
  created:
    - notebooks/01_single_slice.ipynb
  modified:
    - notebooks/_build_01_single_slice.py
    - src/mapclass/attribution.py

key-decisions:
  - "User DECLINED the entire Fallback Ladder (no custom Promise, no pre-pool/use_attn_lrp, no LXT, no captum IG) at the 01-03 human-verify checkpoint — accepted SigLIP-2 op-coverage gap as a per-model finding for the reframed multi-model comparison"
  - "Phase 1 D-02/D-03 visual-eyeball gate CONSCIOUSLY WAIVED for SigLIP-2 (no relevance is produced, so there is nothing to eyeball) — recorded honestly, not a silent pass"
  - "01_single_slice.ipynb reframed from the overlay + three-controls finish line into an honest end-to-end smoke test"
  - "third_party/dynamicLRP NOT modified — VENDOR_SHA 405e74243ecaa1f615f418fdc8ba24c3c5889b1e intact; the T-01-SC3 vendored-SHA deviation was offered (custom Promise) and NOT taken"

patterns-established:
  - "Caught-finding cell: engine RuntimeError surfaced as a recorded result (notebook exit 0), not a crash"

requirements-completed: [ATTR-01, ATTR-02, ATTR-03, VIZ-01]

# Metrics
duration: ~35min
completed: 2026-05-19
---

# Phase 1 Plan 03: Single-Slice Attribution Summary

**Pipeline + overlay code + 16 unit tests delivered and passing; SigLIP-2 dynamicLRP attribution does NOT work (split_with_sizes op-coverage gap → no heatmap); the three D-02/D-03 sanity controls were NOT visually verified because no relevance renders — a user-WAIVED gate and an accepted FINDING for the reframed multi-model comparison, not a silent pass.**

## Performance

- **Duration:** ~35 min (continuation after the 01-03 human-verify checkpoint)
- **Started:** 2026-05-19 (continuation)
- **Completed:** 2026-05-19
- **Tasks:** Tasks 1-2 previously built+committed (verified, not redone); Task 3 checkpoint resolved by user decision; finalize executed
- **Files modified (this continuation):** 3 (notebook regen+execute, generator, attribution.py docstring)

## What Actually Shipped (honest)

**Delivered and working:**
- `src/mapclass/attribution.py` — `attribute()` runs SigLIP-2 forward + LRP in ONE scope against `output.logits_per_image[0,0]` (detached text by tensor-identity), with the 0-dim → 2-D → 1-D target-form fallback and `torch.cuda.max_memory_allocated` capture. **On SigLIP-2 it raises `RuntimeError`** (documented op-coverage gap) — this is correct, honest behavior, not a bug.
- `src/mapclass/overlay.py` — signed + magnitude 27×27 patch grid (`[:378,:378]` 6-px discard), zero-centered `bwr` `TwoSlopeNorm` composite (D-09, no `.abs()`/no min-max), `draw_patch_grid` edge-exclusion demo. **Code + units delivered; NOT exercised end-to-end** because SigLIP-2 produces no relevance to overlay.
- `tests/test_attribution_shape.py` + `tests/test_overlay_grid.py` — **16/16 pass** (synthetic-graph fixtures; do not depend on the real SigLIP-2 LRP path).
- `notebooks/01_single_slice.ipynb` — honest **end-to-end smoke test**: loads GCS-mirrored SigLIP-2, builds the locked-slice (`manifest[-1]` = `RUMSEY~8~1~344476~90112460`, `"a river"`) `requires_grad` `(1,3,384,384)` tensor, runs the forward, measures peak forward VRAM, runs the coverage probe, calls `attribute()` in a `try/except` that **catches** the `RuntimeError` as a recorded FINDING, and shows the source map inline. **0 cell errors; exit 0.**

**Does NOT work (the central finding):**
- **SigLIP-2 dynamicLRP attribution produces NO heatmap.** `LRPEngine.run` rejects every target form on SigLIP-2-so400m; the failure is at autograd node `SplitWithSizesBackward0` — `split_with_sizes` is intrinsic to SigLIP-2's MAP-pool / attention-pooling head and is outside the current engine's covered ops for a contrastive MAP-pool encoder. This is exactly the MEDIUM-LOW research risk in `CLAUDE.md`.
- **The three D-02/D-03 sanity controls (query-swap, model-randomization, occlusion) were NOT visually verified.** There is no relevance to overlay, so there is nothing to eyeball. The original overlay + three-controls notebook (committed at `837befd`) cannot execute end-to-end and was replaced by the honest smoke test.

## Measurements Recorded

- **Peak forward VRAM (single so400m forward, locked slice):** 4.326 GB (4,644,488,704 bytes). Measured with `torch.cuda.reset_peak_memory_stats()` + `torch.cuda.max_memory_allocated()` around the forward in notebook Cell 4; `empty_cache()` after. **This CLOSES the STATE.md peak-VRAM blocker.** Caveat: this is the *forward-only* peak; the dynamicLRP relevance-pass peak (Promise activation retention, Pitfall E) is unmeasurable for SigLIP-2 because the relevance pass never completes. Phase 2 sweep sizing should treat 4.326 GB as a forward-only lower bound, not the full attribution peak.
- **dynamicLRP coverage probe op count:** 26 (printed in Cell 3).
- **`logits_per_image[0,0]` similarity for the locked slice + `"a river"`:** -13.5440 (forward works; only the relevance pass fails).

## Task Commits

Tasks 1-2 verified present (not redone):

1. **Task 1: attribution engine + 27x27 signed overlay** — `4ae1cf0` (feat)
2. **data_loader / config fixes** (Rule 1, prior continuation) — `77c328d` (fix)
3. **Empirical op-coverage failure recorded** — `f854894` (docs)
4. **Task 2: notebook + generator (original overlay + 3 controls)** — `837befd` (feat)
5. **Rule 4 architectural block recorded** — `4430981` (docs)

This continuation:

6. **Honest end-to-end smoke test (notebook regen+execute + attribution.py docstring note)** — `09c6789` (feat)

**Plan metadata:** committed with this SUMMARY + STATE.md + ROADMAP.md update.

## Decisions Made

- **User declined the entire Fallback Ladder.** At the 01-03 human-verify checkpoint the user reviewed the smoke test and the op-coverage finding and explicitly declined every rung — no custom dynamicLRP Promise (would have deviated from VENDOR_SHA, T-01-SC3), no pre-pool/`use_attn_lrp` engineering, no vendored LXT, no captum Integrated Gradients baseline. Quote: *"Smoke-test was good, it didn't crash. Let's leave well enough alone and move on."* Recorded in the `attribution.py` docstring and the notebook so it is not re-attempted.
- **D-02/D-03 visual-eyeball gate consciously WAIVED for SigLIP-2.** No relevance is produced, so the three sanity controls cannot be rendered or eyeballed. This is recorded honestly as a waived gate for the reframed multi-model-comparison purpose — explicitly NOT "all controls passed".
- **Notebook reframed to a smoke test.** Honest end-to-end coverage of load → forward → peak VRAM → coverage probe → caught finding → source map, exiting 0.

## Deviations from Plan

### Scope deviation (user-directed, recorded)

**1. [Rule 4 — Architectural, user-resolved] dynamicLRP op-coverage fails on SigLIP-2; Fallback Ladder declined; D-02/D-03 gate waived for SigLIP-2**
- **Found during:** Task 1-2 execution (prior), surfaced at the Task 3 human-verify checkpoint.
- **Issue:** `LRPEngine.run` cannot produce relevance for SigLIP-2-so400m (`split_with_sizes` / `SplitWithSizesBackward0` outside engine coverage). The plan designated the Fallback Ladder as the FAIL response path and the three controls as the correctness gate.
- **Resolution:** Architectural decision (Rule 4) — NOT auto-selected; surfaced to the user. The user declined the entire Fallback Ladder and accepted the gap as a per-model finding for the reframed multi-model dynamic-LRP comparison; the D-02/D-03 visual gate is consciously waived for SigLIP-2. The notebook was reframed into an honest smoke test.
- **Files modified:** `notebooks/01_single_slice.ipynb`, `notebooks/_build_01_single_slice.py`, `src/mapclass/attribution.py` (docstring only).
- **Verification:** Notebook executes headless with the `mapclass` kernel, 0 cell errors, RuntimeError caught, peak forward VRAM recorded; 16/16 unit tests pass; `third_party/dynamicLRP` clean, `VENDOR_SHA` == `405e74243ecaa1f615f418fdc8ba24c3c5889b1e`.
- **Committed in:** `09c6789`.

---

**Total deviations:** 1 (Rule 4 architectural, user-resolved). **No T-01-SC3 vendored-SHA deviation taken** — `third_party/dynamicLRP` is byte-unmodified, the custom-Promise rung was declined.
**Impact on plan:** The plan's central deliverable (a control-verified SigLIP-2 heatmap) is NOT met; SigLIP-2 is recorded as a negative per-model result under the reframed multi-model purpose. ATTR-01/02/03 + VIZ-01 *code* is delivered and unit-tested but is not end-to-end verified for SigLIP-2 (no relevance to render). This is documented, not papered over.

## Issues Encountered

- Original Task 2 notebook (`837befd`) crashes mid-run (`attribute()` raises before the controls). Resolved by reframing into the honest smoke test that catches the RuntimeError and exits 0.

## Threat Flags

None — no new security surface. `T-01-SC3` (custom-Promise patch deviating from the vendored SHA) was offered as a Fallback rung and **declined**; `third_party/dynamicLRP` is unmodified and `VENDOR_SHA` is intact.

## Project-Reframe Tension (flag for next milestone/transition)

`PROJECT.md` currently states SigLIP-2 is fixed and model-swap / multi-model work is out of scope. The user's decision reframes the project as a **multi-model comparison of dynamic-LRP** with SigLIP-2 as one (negative-result) model. This is a real tension between PROJECT.md and the working framing. **Not reconciled here** (executor does not rewrite PROJECT.md). Flagged for the next milestone/phase-transition decision: PROJECT.md scope wording, ROADMAP Phase 1 success criteria 4-5 (which assume a working SigLIP-2 heatmap), and the Phase 2 sweep premise all need reconciliation against the multi-model framing.

## Next Phase Readiness

- **Phase 1 finish line (a control-verified SigLIP-2 heatmap) is NOT reached for SigLIP-2.** It is closed as a documented negative finding under the reframed purpose, with the D-02/D-03 gate waived by the user.
- Pipeline code (attribution/overlay) + 16 unit tests are reusable for other models in the multi-model comparison.
- Peak-VRAM blocker closed (forward-only 4.326 GB; relevance-pass peak unmeasurable for SigLIP-2).
- **Open for the next transition:** PROJECT.md / ROADMAP scope reconciliation against the multi-model framing (do not start Phase 2's SigLIP-2 sweep premise without this).

## Self-Check: PASSED

- Created files verified present: `notebooks/01_single_slice.ipynb`, `notebooks/_build_01_single_slice.py`, `src/mapclass/attribution.py`, `src/mapclass/overlay.py`, `01-03-SUMMARY.md`.
- Commits verified present: `4ae1cf0`, `77c328d`, `f854894`, `837befd`, `4430981`, `09c6789`.
- `third_party/dynamicLRP` clean (no T-01-SC3 deviation); `VENDOR_SHA` == `405e74243ecaa1f615f418fdc8ba24c3c5889b1e`.
- Notebook executed headless with the `mapclass` kernel: 0 cell errors, exit 0; RuntimeError caught as a finding; peak forward VRAM 4.326 GB recorded; source map rendered.
- 16/16 unit tests pass.
- Honest framing confirmed: SUMMARY plainly states SigLIP-2 attribution does NOT work, no heatmap, D-02/D-03 controls NOT visually verified (user-waived), not a silent pass.

---
*Phase: 01-pinned-environment-gcs-mirrors-and-a-verified-single-slice-a*
*Completed: 2026-05-19*
