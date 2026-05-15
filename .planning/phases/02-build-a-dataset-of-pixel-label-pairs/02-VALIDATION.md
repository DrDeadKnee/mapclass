---
phase: 2
slug: build-a-dataset-of-pixel-label-pairs
status: planned
nyquist_compliant: true
wave_0_complete: false
created: 2026-05-15
---

# Phase 2 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `02-RESEARCH.md` § Validation Architecture.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | None currently — no `tests/` directory, no pytest config. Wave 0 installs `pytest>=8` + `pytest-mock>=3`. |
| **Config file** | none — Wave 0 creates `pytest`-runnable layout |
| **Quick run command** | `pytest tests/ -x --ignore=tests/integration` |
| **Full suite command** | `pytest tests/` |
| **Estimated runtime** | quick < 30s · full (offline) < 5min · phase gate adds online integration tests |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x --ignore=tests/integration` (unit only, < 30s)
- **After every plan wave:** Run `pytest tests/` (all unit + offline integration, < 5min)
- **Before `/gsd-verify-work`:** Full suite incl. online integration tests must be green
- **Max feedback latency:** 30 seconds (quick) / 300 seconds (wave)

---

## Per-Task Verification Map

> Task IDs are assigned by the planner. Map below is keyed by requirement +
> behavior from RESEARCH; the planner binds each to its concrete task ID.

| Plan Area | Requirement | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|-----------|-------------|-----------------|-----------|-------------------|-------------|--------|
| historical | PHASE-02 | LUNA search returns N ≥ 1 for 1500–1700 query | integration (online) | `pytest tests/test_rumsey.py::test_search_returns_results -x` | ❌ W0 | ⬜ pending |
| historical | PHASE-02 | Allmaps lookup returns GCPs for a known-georeferenced Rumsey manifest | integration (online) | `pytest tests/test_allmaps.py::test_known_rumsey_manifest_has_gcps -x` | ❌ W0 | ⬜ pending |
| historical | PHASE-02 | IIIF fetch at `!4096,4096` returns ≤ 4096 in both dims | integration (online) | `pytest tests/test_iiif.py::test_max_edge_fetch -x` | ❌ W0 | ⬜ pending |
| historical | PHASE-02 | GCP affine fit + GeoTIFF write round-trips (CRS=4326, transform survives) | unit | `pytest tests/test_georef.py::test_affine_roundtrip -x` | ❌ W0 | ⬜ pending |
| historical | PHASE-02 | `make_labels()` produces all 4 outputs for a 256-px synthetic GeoTIFF | integration (offline) | `pytest tests/test_label.py::test_make_labels_writes_all_outputs -x` | ❌ W0 | ⬜ pending |
| synthetic | PHASE-02 | `render_map` produces matching `land_cover.png` / `topography.png` dims | unit | `pytest tests/test_render.py::test_output_dimensions_match -x` | ❌ W0 | ⬜ pending |
| synthetic | PHASE-02 | Seeded train/test split deterministic across re-invocations | unit | `pytest tests/test_split.py::test_seeded_split_deterministic -x` | ❌ W0 | ⬜ pending |
| satellite | PHASE-02 | STAC search returns ≥1 item for known-coverage bbox, cloud<10 | integration (online) | `pytest tests/test_stac.py::test_known_bbox_returns_results -x` | ❌ W0 | ⬜ pending |
| satellite | PHASE-02 | COG window read returns shape (3, 4096, 4096) for `visual` asset | integration (online) | `pytest tests/test_satellite.py::test_window_fetch_shape -x` | ❌ W0 | ⬜ pending |
| tiler | PHASE-02 | Nested pyramid (1×896 + 4×448 + 16×224) child extent == parent | unit | `pytest tests/test_tiling.py::test_nested_alignment -x` | ❌ W0 | ⬜ pending |
| tiler | PHASE-02 | Edge-policy: pyramid >50% off the source map is dropped | unit | `pytest tests/test_tiling.py::test_edge_drop -x` | ❌ W0 | ⬜ pending |
| split | EVAL-01 | `data/synthetic/test/` ⟂ `train/` — NO intersection of map IDs | integration | `pytest tests/test_split.py::test_no_train_test_intersection -x` | ❌ W0 | ⬜ pending |
| split | EVAL-01 | `split.json` frozen — rebuild does not change test-set ID list | integration | `pytest tests/test_split.py::test_split_manifest_frozen -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/conftest.py` — shared fixtures (sample LUNA item, sample Allmaps annotation, tiny 256-px GeoTIFF, sample Azgaar GeoJSON, mocked-STAC item)
- [ ] `tests/test_rumsey.py` — LUNA search + filter behaviour
- [ ] `tests/test_allmaps.py` — Allmaps lookup + offline-dump intersection
- [ ] `tests/test_iiif.py` — IIIF size syntax + scale-factor logic
- [ ] `tests/test_georef.py` — affine fit + GeoTIFF I/O
- [ ] `tests/test_label.py` — `make_labels` end-to-end on a fixture
- [ ] `tests/test_render.py` — synthetic render dimensions
- [ ] `tests/test_split.py` — train/test deterministic split + frozen manifest
- [ ] `tests/test_stac.py` — STAC client wrapper
- [ ] `tests/test_satellite.py` — satellite fetcher + RGB GeoTIFF write
- [ ] `tests/test_tiling.py` — nested-pyramid tiler
- [ ] `tests/integration/` — subdir for online tests (network required)
- [ ] Framework install: add `pytest>=8` and `pytest-mock>=3` to `requirements.txt`; `pytest tests/` runnable

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| PaliGemma-driven semi-automatic registration produces a visually plausible warp on a real unregistered Rumsey map | PHASE-02 | TPS warp quality against illustrated cartography has no automated ground truth at regional scale; correctness is a human cartographic judgement | Run the semi-auto path on one unregistered manifest; overlay the warped GeoTIFF on the WorldCover/DEM reference grid in QGIS; confirm coastlines/ranges align within visual tolerance |
| Manual MapWarper/QGIS GCP fallback path is usable | PHASE-02 | Interactive GCP placement is inherently manual | Follow the documented fallback procedure on one map flagged for manual registration; confirm output is a warped GeoTIFF in EPSG:4326 |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 30s (quick) / 300s (wave)
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** planner-approved 2026-05-15 (Wave 0 = plan 02-01; all downstream tasks carry <automated> verify or a Wave 0 dependency)
