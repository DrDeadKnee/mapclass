# Project State

## Project Reference

**Building:** MapClass / GeoViLM — a small vision-language backbone with rotationally-invariant OCR and two dense segmentation heads, producing per-pixel land-cover (9-class) + topography (3-class) probability maps on stylized cartographic inputs (synthetic fantasy, historical, road, satellite). Consumed by a separate hex-grid aggregator app.

**Core value:** A small, cheap-to-run model that returns reliable per-pixel land-cover + topography probability maps on stylized inputs. Inference must run on a single CPU host or 4–8 GB consumer GPU.

**Current focus:** Project initialization phase — moving from research artifacts to a roadmap with executable phases.

## Current Position

- **Phase:** — (no roadmap yet)
- **Plan:** —
- **Status:** Init partially complete; ROADMAP.md not yet generated
- **Branch:** `refactor_paper`

## Progress

```
[██░░░░░░░░] ~20% — init artifacts only; roadmap pending
```

**Completed:**
- ✓ Codebase mapping (.planning/codebase/ — 7 docs)
- ✓ Research (.planning/research/ — ARCHITECTURE.md, FEATURES.md)
- ✓ PROJECT.md with locked v1 scope, constraints, decisions, out-of-scope
- ✓ Project config (yolo mode, quality model profile)

**Pending:**
- ROADMAP.md — phase decomposition
- Phase planning, execution

## Recent Decisions

(All from PROJECT.md "Key Decisions" — pending until roadmap+execution validate them.)

- App-first; paper attempt opportunistic in milestone 2
- v1 components: VL backbone + OCR + dense seg heads. Auto-georef is training-tool-only
- PaliGemma-3B excluded as v1 backbone (inference budget)
- OSM road tiles included as third v1 dataset source
- Auto-georef bootstrap loop (v0 → register → v1 retrain) included in v1
- Validation v1: held-out NLL + qualitative spot-checks on four target fantasy maps
- License: ship weights + code, do not redistribute the dataset
- v1 ends at model handoff (checkpoint + inference module + benchmark numbers)
- Tests / CI / linting deferred from v1

## Pending Todos

None captured yet.

## Blockers / Concerns

Carried from `.planning/codebase/CONCERNS.md`:
- No tests, no CI, no enforced linter — accepted at v1, flagged for milestone 2
- Six untested high-risk areas: taxonomy index alignment, duplicated geometric helpers, slope thresholds, network paths, CRS round-tripping, loss-weight schema
- License risk on three of four data sources (Rumsey, Copernicus DEM, Azgaar) — punted in v1 by not redistributing dataset

## Session Continuity

Last session: 2026-05-08 (resume)
Stopped at: Session resumed. STATE.md reconstructed from PROJECT.md + research/ + codebase/. Ready to generate ROADMAP.md.
Resume file: —
