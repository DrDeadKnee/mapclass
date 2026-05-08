# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-08)

**Core value:** A small, cheap-to-run model that returns reliable per-pixel land-cover + topography probability maps on stylized inputs. Inference must run on a single CPU host or 4–8 GB consumer GPU.
**Current focus:** Phase 1 — End-to-End Skeleton (Synthetic + Mock Backbone)

## Current Position

Phase: 1 of 6 (End-to-End Skeleton — Synthetic + Mock Backbone)
Plan: — of — (planning has not started yet)
Status: Ready to plan
Last activity: 2026-05-08 — ROADMAP.md generated; STATE.md and REQUIREMENTS.md traceability synced.

Progress: [██░░░░░░░░] ~25% (init + research + roadmap; planning + execution pending)

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. End-to-End Skeleton | 0 | — | — |
| 2. Small VL Backbone Slice | 0 | — | — |
| 3. OSM + OCR + Full v0 Training | 0 | — | — |
| 4. Auto-Georef Bootstrap + v1 Retrain | 0 | — | — |
| 5. PaliGemma Bellwether + EVAL-03 | 0 | — | — |
| 6. Ship the Small-Backbone Variant | 0 | — | — |

**Recent Trend:**
- No plans executed yet.

*Updated after each plan completion.*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- App-first; paper attempt is opportunistic in milestone 2.
- v1 model components: small VL backbone (SmolVLM-500M) + docTR-pretrained OCR + dense seg heads. Auto-georef is training-tool-only.
- PaliGemma-3B excluded as ship/inference backbone; included as MODEL-04 benchmark/bellwether (EVAL-03 NLL gap).
- OSM road tiles included as the third v1 dataset source (DATA-05).
- Auto-georef bootstrap loop (v0 → register → v1 retrain) included in v1 with the v0-vs-v1 NLL kill-switch (≥2% margin or ship v0).
- Validation v1: held-out NLL on synthetic + historical (EVAL-01); qualitative spot-checks on four fantasy maps (EVAL-02).
- License: ship weights + code, do not redistribute the dataset.
- v1 ends at model handoff (checkpoint + inference module + benchmark numbers); tests / CI / linting deferred from v1.

### Pending Todos

None captured yet.

### Blockers/Concerns

Carried from `.planning/codebase/CONCERNS.md` and `.planning/research/PITFALLS.md`:

- **PITFALL 2 prevention is a Phase-1 deliverable, not a later artifact** — `mapclass/configs/EVAL-03_protocol.md` must be committed before any MODEL-01 / MODEL-04 training run starts. This is the entry criterion for Phases 2 / 3 / 5.
- **Lockfile (CONCERNS.md 8c, FEATURES D-10) cannot be deferred** — `pyproject.toml` + `.python-version` + `requirements.lock.txt` land in Phase 1.
- **Six untested high-risk areas (TESTING.md):** taxonomy index alignment, duplicated geometric helpers, slope thresholds, network paths, CRS round-tripping, loss-weight schema. Phase 1's `assert_sample_valid` + `LossWeights.load` schema validators are the v1 mitigation.
- **License risk on three of four data sources (Rumsey, Copernicus DEM, Azgaar)** — addressed in v1 by not redistributing the dataset (PROJECT.md decision); attribution strings still must land in checkpoint metadata at Phase 6.
- **No tests, no CI, no enforced linter** — accepted at v1 per PROJECT.md; observability-replaces-tests via TS-7 training-time logging + schema validators + the v0-vs-v1 kill switch.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| *(none — v1 has not closed)* | | | |

## Session Continuity

Last session: 2026-05-08
Stopped at: Phase 1 context gathered. CONTEXT.md captures Mock-backbone fidelity decisions (learnable conv stub, single-stage features, seeded init, metadata-only checkpoint ID) and EVAL-03 protocol decisions (markdown-only, deferred threshold with `DECIDE_AT_PHASE_5` sentinel, strict fairness, script-guarded pre-registration). Ready for `/gsd-plan-phase 1`.
Resume file: .planning/phases/01-end-to-end-skeleton/01-CONTEXT.md
