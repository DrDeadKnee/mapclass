---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: phase-complete
stopped_at: "Session resumed via /gsd-resume-work. Investigated dataset state and confirmed Phase 02 prereq gap: no Azgaar GeoJSON inputs anywhere (synthetic pipeline has render/label code but zero raw inputs); no `data/historical/raw/georeferenced/` (Rumsey `search` not yet run for the registered subset; `unregistered_manifest.json` is only the GCP-needed leftover). Smoke fixtures (100 64×64 in `data/renders/`) and `data/toons/` artwork (13 hex-tile classes on GCS) exist but are not training samples. Decision: **insert Phase 1.5 dedicated to the real v0-thin dataset build** rather than fattening Phase 02 or running Rumsey search blind. Phase 02 stays scoped to the SmolVLM swap."
last_updated: "2026-05-14T20:42:40.002Z"
last_activity: 2026-05-09 -- Phase 01 plan 03 EVAL-03 protocol committed (214f07d)
progress:
  total_phases: 7
  completed_phases: 1
  total_plans: 3
  completed_plans: 3
  percent: 14
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-08)

**Core value:** A small, cheap-to-run model that returns reliable per-pixel land-cover + topography probability maps on stylized inputs. Inference must run on a single CPU host or 4–8 GB consumer GPU.
**Current focus:** Phase 01.1 (INSERTED) — Real v0-thin Dataset Build (prerequisite for Phase 02 SmolVLM training)

## Current Position

Phase: 01 (end-to-end-skeleton) — COMPLETE (3/3 plans summarized)
Next phase: 01.1 (real-v0-thin-dataset-build, INSERTED) — directory created, ready to plan/discuss
Following: 02 (small VL backbone slice) — depends on 01.1's dataset
Last activity: 2026-05-14 -- Phase 01.1 inserted via gsd-sdk phase.insert (after dataset prereq surfaced during resume)

Progress: [█░░░░░░░░░] ~14% (1/7 phases complete; Phase 01.1 ready to plan)

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: —
- Total execution time: —

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1. End-to-End Skeleton | 3/3 | — | — |
| 1.1. Real v0-thin Dataset Build (INSERTED) | 0 | — | — |
| 2. Small VL Backbone Slice | 0 | — | — |
| 3. OSM + OCR + Full v0 Training | 0 | — | — |
| 4. Auto-Georef Bootstrap + v1 Retrain | 0 | — | — |
| 5. PaliGemma Bellwether + EVAL-03 | 0 | — | — |
| 6. Ship the Small-Backbone Variant | 0 | — | — |

**Recent Trend:**

- Phase 01 complete: 01-01 (data + config foundation), 01-02 (model layer + CLIs), 01-03 (EVAL-03 pre-registration protocol).
- Wave 3 of /gsd-execute-phase 1 had a worktree-agent terminate mid-flight (out of usage); orchestrator adopted the agent's authored EVAL-03_protocol.md after byte-identity hash verification. See 01-03-SUMMARY.md "Issues Encountered".

*Updated after each plan completion.*

## Accumulated Context

### Roadmap Evolution

- Phase 01.1 inserted after Phase 1: Real v0-thin Dataset Build (URGENT)

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

Last session: 2026-05-14 (resume — dataset gap surfaced, Phase 01.1 inserted)
Stopped at: Phase 01.1 (INSERTED) created via `gsd-sdk query phase.insert 1 "Real v0-thin Dataset Build"`. Directory `.planning/phases/01.1-real-v0-thin-dataset-build/` exists (empty, .gitkeep only). ROADMAP.md updated: detail section auto-inserted between Phase 1 and Phase 2; top-level bullet list manually patched to include Phase 1.1 (the SDK insert handler only emits the detail block, not the bullet). Roadmap evolution entry recorded. Phase goal is to source real Azgaar GeoJSON inputs + the registered-Rumsey subset and run both build pipelines end-to-end so Phase 02 has training-ready data.
Dataset gap that motivated the insert: no Azgaar GeoJSON inputs anywhere (synthetic pipeline has render/label code but zero raw inputs); no `data/historical/raw/georeferenced/` (Rumsey `search` not yet run for the registered subset; `unregistered_manifest.json` is only the GCP-needed leftover). Smoke fixtures (100 64×64 in `data/renders/`) and `data/toons/` artwork (13 hex-tile classes on GCS) exist but are not training samples.
Next action: `/gsd-discuss-phase 1.1` or `/gsd-plan-phase 1.1` (no CONTEXT.md or PLAN.md yet — discuss recommended given Azgaar-sourcing strategy and N/M sample targets are unresolved).
Recent commits (refactor_paper-notebook):

  - 64e9f35 Ran through the notebooks (models appear to work-ish; data pipeline is Phase 2's problem — now Phase 1.5's)
  - cbdd574 Fix compute_loss to handle resolution mismatch (PaliGemma path)
  - 40fba9d Wire train_playground PaliGemma cells to load from GCS cache
  - a28b994 Set up notebook env + add GPU training playground

PaliGemma snapshot staged: `gs://mapclass-training-northeast1/models/paligemma-3b-mix-224/` (10.91 GiB). See memory: [[gcs-bucket]].
Resume file: none (no .continue-here, no incomplete plan, no HANDOFF.json).
