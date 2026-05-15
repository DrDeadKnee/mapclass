---
phase: 02-build-a-dataset-of-pixel-label-pairs
plan: 01
subsystem: test-infrastructure + historical-georeferencing
tags: [pytest, fixtures, allmaps, scope-deferral, wave-0]
requires: []
provides:
  - pytest-runnable tests/ layout (pytest.ini + conftest fixtures)
  - allmaps.lookup() returning list[dict] of all annotations per manifest
  - GEOREF-V2 v2 requirement (PaliGemma + manual georef deferred)
affects:
  - plan 02-02 (historical pipeline — depends on lookup() list contract + fixtures)
  - plan 02-03 (synthetic — depends on conftest fixtures)
  - plan 02-04 (satellite — depends on conftest fixtures)
  - plan 02-05 (tiler — depends on conftest fixtures)
tech-stack:
  added: [pytest>=8, pytest-mock>=3]
  patterns: [pytest.importorskip/skip-marked red skeletons, mocker.patch offline HTTP]
key-files:
  created:
    - pytest.ini
    - tests/conftest.py
    - tests/integration/__init__.py
    - tests/integration/test_allmaps_online.py
    - tests/test_allmaps.py
    - tests/test_rumsey.py
    - tests/test_iiif.py
    - tests/test_georef.py
    - tests/test_label.py
    - tests/test_render.py
    - tests/test_split.py
    - tests/test_stac.py
    - tests/test_satellite.py
    - tests/test_tiling.py
  modified:
    - requirements.txt
    - scripts/historical/allmaps.py
    - .planning/REQUIREMENTS.md
decisions:
  - "A6 / Pitfall 3: allmaps.lookup() returns list[dict] of ALL annotations (not items[0]); [] for 404/no-items/all-malformed"
  - "Red Wave 0 skeletons use @pytest.mark.skip (not bare failing imports) so pytest tests/ collects green"
  - "STATE.md / ROADMAP.md scope-deferral edits NOT applied in-worktree — deferred to orchestrator post-merge per execution objective"
metrics:
  duration: ~12 min
  completed: 2026-05-15
---

# Phase 2 Plan 01: Wave 0 Test Framework + Allmaps Multi-Annotation Fix Summary

Established the Phase 2 pytest scaffold (no test infra previously existed), fixed
the load-bearing Allmaps multi-annotation bug so `lookup()` returns every
georeferenced canvas of a multi-canvas Rumsey atlas instead of only the first
(the ~10,455-vs-337 historical yield fix), and added the `GEOREF-V2` v2
deferral requirement.

## What Was Built

### Task 1 — pytest framework + Wave 0 scaffold (commit `09670e7`)
- `requirements.txt`: appended `pytest>=8` and `pytest-mock>=3` (existing pins
  untouched, no reorder).
- `pytest.ini`: `testpaths = tests`, `pythonpath = scripts` (so
  `import historical.allmaps`, `import label`, `import render` resolve against
  the `scripts/` import root), and a registered `integration` marker.
- `tests/conftest.py`: six offline fixtures —
  `sample_luna_item`, `sample_allmaps_annotation`, `sample_allmaps_multi`
  (3 distinct annotations), `tiny_geotiff` (256×256 3-band uint8 EPSG:4326
  GeoTIFF via rasterio with a valid affine transform), `sample_azgaar_geojson`
  (3 Polygon features with integer `biome`/`height`), `mock_stac_item`
  (`.assets["visual"].href` + `.properties["eo:cloud_cover"]`). The Allmaps
  fixtures are modelled exactly on `_parse_annotation`'s expected shape
  (`body.features[].properties.resourceCoords` / `geometry.coordinates`,
  `target.source.{id,width,height}`).
- `tests/integration/__init__.py` package marker.
- 10 red Wave 0 skeleton modules (`test_rumsey`, `test_iiif`, `test_georef`,
  `test_label`, `test_render`, `test_split`, `test_stac`, `test_satellite`,
  `test_tiling`, plus the `test_allmaps` skeleton) — every downstream-targeting
  test is `@pytest.mark.skip(reason="Wave N — implemented in plan-0X")` so
  `pytest tests/` collects green (no bare failing imports). Skip reasons map
  each test to its owning downstream plan per the 02-VALIDATION per-task map.

### Task 2 — allmaps.lookup() multi-annotation fix (commit `b7c5d44`)
- `scripts/historical/allmaps.py`: `lookup()` signature changed
  `Optional[dict]` → `list[dict]`. Now `[_parse_annotation(ann) for ann in items]`
  with `None` results dropped. 404/500, no-items, and all-malformed all return
  `[]` (never `None`) so Plan 02's `rumsey.py` can treat empty as
  `not_in_allmaps` / `gcps_insufficient` for D-04 drop classification and a
  length-N list as N separate maps.
- `_parse_annotation`, the retry/backoff loop, the 429/503 + status-code
  switch, the typed `AllmapsLookupError`, and `haversine_km` /
  `bbox_diagonal_km` are byte-for-byte unchanged (verified by diff — locked
  W3C contract). Module + `lookup()` docstrings updated to describe the list
  contract and the "200 returns all annotations" outcome.
- `tests/test_allmaps.py`: four offline tests (all `requests.get` mocked via
  `mocker.patch`) — `test_lookup_returns_all_annotations` (len==3),
  `test_lookup_empty_on_404`, `test_lookup_empty_on_no_items`,
  `test_lookup_skips_malformed`.
- `tests/integration/test_allmaps_online.py::test_known_rumsey_manifest_has_gcps`
  (`@pytest.mark.integration`) — asserts `len(result) >= 1` and
  `len(result[0]["gcps"]) >= 3` against a live Rumsey manifest (phase-gate run,
  not per-commit).

### Task 3 — scope-deferral documentation (commit `11b23d3`)
- `.planning/REQUIREMENTS.md`: added `GEOREF-V2` under a new
  "### Georeferencing (deferred from Phase 2)" subsection in v2 Requirements,
  covering both the PaliGemma semi-automatic registration path and the manual
  MapWarper/QGIS GCP fallback. Bullet only — no v2 traceability-table row was
  added because the Traceability table tracks v1 requirements exclusively
  ("v1 requirements: 9 total"); the existing table was left intact.

## Verification Results

- `pytest tests/ --co -q` — **19 tests collected, 0 collection/import errors**
  (15 unit/offline + 4 incl. the online integration module).
- `pytest tests/ -q --ignore=tests/integration` — **4 passed, 14 skipped**
  (green: pass/skip only, no errors or failures).
- `pytest tests/test_allmaps.py -x -q` — **4 passed** (all offline assertions).
- Task 2 source asserts: no `ann = items[0]`; `for ann in items` present
  (comprehension form); return type `list[dict]`; `[]` on 404/no-items;
  `_parse_annotation` signature + `(KeyError, TypeError, IndexError, ValueError)`
  guard intact.
- Task 3: `grep -q GEOREF-V2 .planning/REQUIREMENTS.md` passes.
- Online test `test_known_rumsey_manifest_has_gcps` NOT run here (network /
  phase-gate scope) — deferred to the phase verification gate as the plan
  specifies.

## Deviations from Plan

### Deviation 1 — STATE.md / ROADMAP.md scope-deferral edits NOT applied
- **Found during:** Task 3
- **Reason:** The execution objective (parallel worktree mode) explicitly
  forbids modifying `STATE.md` and `ROADMAP.md` — the orchestrator owns those
  shared-file writes after all wave agents merge. The plan's `files_modified`
  lists them and Task 3's `<automated>` verify greps them, but the objective
  overrides the plan here (orchestrator instruction takes precedence).
- **What was done:** Only the `.planning/REQUIREMENTS.md` GEOREF-V2 edit was
  applied and committed. The intended STATE.md and ROADMAP.md wording is
  recorded below for the orchestrator to apply post-merge.
- **Files modified:** none beyond the planned REQUIREMENTS.md edit.
- **Commit:** `11b23d3`

#### Orchestrator to apply post-merge (verbatim wording)

**`.planning/ROADMAP.md`** — Phase 2 Success Criterion #2. *Already reads*
(lines 73–75): "Registered historical maps come from Allmaps directly;
unregistered maps are emitted to `unregistered_manifest.json` for v2 processing
(manual fallback + PaliGemma semi-auto deferred), all registered output as
georeferenced GeoTIFF in EPSG:4326". **This already satisfies the Task 3
requirement** (contains both "unregistered_manifest.json" and "deferred", and
no longer asserts the PaliGemma semi-auto path as a Phase-2 deliverable). No
edit is actually required — but the orchestrator should confirm SC#1, #3, #4,
#5 remain unchanged. If a stricter wording is desired, replace SC#2 with:
"unregistered maps are emitted to `unregistered_manifest.json` for v2
processing (manual fallback + PaliGemma semi-auto deferred)".

**`.planning/STATE.md`** — append two rows to the `## Deferred Items` table
(after the existing 4 rows, before `## Session Continuity`):

```
| Georeferencing | PaliGemma-driven semi-automatic registration (cross-corr + TPS) | Deferred to v2 | Phase 2 planning (2026-05-15) |
| Georeferencing | Manual MapWarper/QGIS GCP placement fallback | Deferred to v2 | Phase 2 planning (2026-05-15) |
```

Post-apply, `grep -ci "semi-automatic" .planning/STATE.md` ≥ 1 and
`grep -c "Georeferencing" .planning/STATE.md` ≥ 2 (the original Task 3
`<automated>` verify) will pass.

## Authentication Gates

None — all offline tests mock network; no auth surface (Allmaps/LUNA/STAC are
anonymous, per threat register T-02-03 disposition `accept`).

## Threat Model Notes

T-02-01 (DoS via malformed/huge `items`) — mitigation in place: the new
`lookup()` loop applies `_parse_annotation` (which wraps field access in
try/except and returns `None` on malformed) per item and drops `None`, so one
bad annotation cannot crash the build. No new threat surface introduced
(no filenames written in this plan; no new endpoints).

## Known Stubs

The 10 Wave 0 skeleton test modules are intentional `@pytest.mark.skip` stubs
by plan design ("each downstream plan fills its own test bodies per the
02-VALIDATION map"). Each skip reason names the owning plan
(02-02 / 02-03 / 02-04 / 02-05). These are NOT defects — Plan 02-01's goal is
the scaffold itself; the `tests/test_allmaps.py` bodies (this plan's own
behavioural target) are fully implemented and passing.

## Self-Check: PASSED

- All 16 created files present on disk (verified below).
- All 3 task commits exist in git log (`09670e7`, `b7c5d44`, `11b23d3`).
