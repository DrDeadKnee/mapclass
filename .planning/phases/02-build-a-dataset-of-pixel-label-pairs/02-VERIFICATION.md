---
phase: 02-build-a-dataset-of-pixel-label-pairs
verified: 2026-05-15T18:10:00Z
updated: 2026-05-15T18:40:00Z
status: human_needed
score: 5/5 ROADMAP success criteria verified (offline); residual-edge warning RESOLVED
overrides_applied: 0
re_verification:
  previous_status: human_needed
  note: >-
    Residual non-dict-feature edge (original human_verification item 2)
    RESOLVED by commit d72c43b — AttributeError added to all per-feature/
    per-ring except tuples in scripts/label.py + scripts/render.py, plus
    regression tests test_non_dict_feature_skipped_{label,render} feeding
    bare string/int FeatureCollection entries with a valid sibling. Offline
    suite 62→64 passing, no regressions. The CR-01 / T-02-09 per-source
    resilience crash class (CR-03→CR-04→CR-01→non-dict) is now closed.
    Decision: user chose "fix now" (2026-05-15). Sole remaining human item:
    the deferred online integration phase gate (networked host required).
human_verification:
  - test: "Run the 6 ONLINE integration tests in a network-enabled environment: pytest tests/integration -m integration (Allmaps annotations server, IIIF image endpoints, David Rumsey LUNA, Sentinel-2 STAC catalogue, s3://sentinel-cogs)"
    expected: "test_allmaps_online (>=1 annotation, >=3 GCPs), test_iiif_online (both dims <=4096, actual w,h returned), test_rumsey_online (search_maps 1500-1700 returns >=1), test_stac_online (known bbox returns results), test_satellite_online (window fetch shape (3,4096,4096) + end-to-end region produces 4 output files)"
    why_human: "Sandbox has no network; all 6 integration tests hang to SIGTERM offline. Documented in deferred-items.md. This is the online phase gate for the Wave-1/2 fetch/search code (network I/O cannot be verified by grep or offline tests). Deferred to networked-host UAT by user decision 2026-05-15."
---

# Phase 2: Build a Dataset of Pixel-Label Pairs — Verification Report

**Phase Goal:** Produce labelled training data from three source streams — (a) historical illustrated maps (Rumsey, Allmaps-georeferenced, ESA WorldCover + Copernicus DEM labels, class-conditional per-source loss weights, unregistered → unregistered_manifest.json), (b) synthetic Azgaar+Pillow auto-labelled triplets, (c) satellite ESA WorldCover + Copernicus DEM GLO-30 mapped to the canonical 9-class taxonomy.
**Verified:** 2026-05-15T18:10:00Z
**Status:** human_needed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Phase-2 Success Criteria)

| # | Truth (SC) | Status | Evidence |
|---|------------|--------|----------|
| 1 | Historical pipeline produces per map image.png/land_cover.png/topography.png/sample_weights.json with the locked trust-tiered weights | ✓ VERIFIED | `scripts/historical/rumsey.py` runs Allmaps→IIIF→GeoTIFF into `<id>__plate<i>/source.tif`; `scripts/historical/label.py:make_labels` (untouched, `HISTORICAL_LC_WEIGHTS` frozen) emits the 4-file schema; `tests/test_label.py::test_make_labels_writes_all_outputs` passes; weights tiering locked in historical/label.py |
| 2 | Registered maps from Allmaps directly; unregistered → unregistered_manifest.json (PaliGemma/manual DEFERRED v2); registered output as warped GeoTIFF EPSG:4326 | ✓ VERIFIED | WMS code fully deleted (`grep -cE '_wms_url\|_parse_bbox\|_download_wms_geotiff' rumsey.py` == 0); `download_georeferenced` calls `allmaps.lookup` (list[dict]) with 5 D-04 status reasons; `emit_manifest` writes per-reason v2 hand-off to `raw_dir/unregistered_manifest.json` (build_historical_dataset.py:52,76); `georef.write_georeferenced_geotiff` writes `CRS.from_epsg(4326)`; `test_georef.py::test_affine_roundtrip` passes; GEOREF-V2 deferral recorded in REQUIREMENTS.md:77, ROADMAP SC#2, STATE.md |
| 3 | Synthetic pipeline: matched triplets from Azgaar+Pillow, auto ground truth, re-normalised height thresholds flat ≤20 / hilly 20–55 / mountainous >55 over [0,100] | ✓ VERIFIED | `scripts/label.py` + `scripts/render.py` produce per-(source×style) dirs sharing land_cover/topography with one image.png; `tests/test_render.py::test_synthetic_topo_locked_boundaries` passes (concrete assertion at and around the 20/55 cuts, water→None) — it is the SC#3 gate, not a dimension proxy |
| 4 | Satellite pulls ESA WorldCover from s3://esa-worldcover + Copernicus DEM GLO-30 from s3://copernicus-dem-30m directly (no GEE), re-encoded canonical 9/3-class | ✓ VERIFIED | `worldcover.py:38 _WC_BASE = https://esa-worldcover.s3.amazonaws.com/...`; `dem.py:35 _COP_BASE = https://copernicus-dem-30m.s3.amazonaws.com`; no `earthengine`/`ee.` references; `satellite/stac.py` queries `sentinel-2-l2a` with `{"eo:cloud_cover":{"lt":...}}`; `satellite/fetch.py` uses `rasterio.windows.Window` (no full-scene read) + imports the single shared `historical.georef.write_georeferenced_geotiff` (no fork); `WC_REMAP` reuse for 9-class re-encode; `test_satellite.py`/`test_stac.py`/`test_coverage.py` pass |
| 5 | Held-out synthetic subset reserved as Phase-4 test set, guaranteed unseen (EVAL-01: frozen seeded stratified split, no leakage) | ✓ VERIFIED | `build_dataset.py` `_SPLIT_SEED=42`, `_TEST_FRACTION=0.15`, `N_TARGET_V1=100`, `stratified_split` per-template salted RNG, `load_or_create_split` writes once + frozen; `test_split.py::{test_split_manifest_frozen,test_no_train_test_intersection,test_seeded_split_deterministic}` pass; `test_tiling.py::test_split_subtree_preserved` confirms whole-source test/ pyramids never cross to train/ |

**Score:** 5/5 ROADMAP success criteria verified at the offline level.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `scripts/historical/allmaps.py` | lookup() → list[dict], all annotations | ✓ VERIFIED | `for ann in items`, `[]` on 404/500/empty, `_parse_annotation` intact with `(KeyError,TypeError,IndexError,ValueError)` guard + WR-05 positive-dim guard (A6 honored) |
| `scripts/historical/iiif.py` | !4096,4096 best-fit + 200MB guard + scale_gcps | ✓ VERIFIED | `build_iiif_url`, `fetch_iiif_image` Content-Length 200MB abort, `scale_gcps` computes sx/sy from actual fetched dims |
| `scripts/historical/georef.py` | from_gcps + EPSG:4326 writer (+nodata) | ✓ VERIFIED | `GroundControlPoint(row=py,col=px,x=lng,y=lat)`, `from_gcps`, `CRS.from_epsg(4326)`, additive `nodata=` (WR-03, single shared writer — W-1 dedup preserved) |
| `scripts/historical/rumsey.py` | Allmaps path, 5 D-04 statuses, sanitized plate dirs, WMS gone | ✓ VERIFIED | `re.sub(r"[^\w-]","_")` id sanitization, `__plate<i>` dirs, emit_manifest per-reason status; WMS deleted |
| `scripts/build_historical_dataset.py` | per-reason drop counter + loud >50% + unregistered_manifest + tiling | ✓ VERIFIED | `drops` dict, `>0.5` loud warning, `emit_manifest`→`unregistered_manifest.json`, `tiling.tile` after make_labels with missing-file guard |
| `scripts/label.py` / `scripts/render.py` | structural `_xy` chokepoint + per-feature/ring try/except | ⚠️ ORPHANED-EDGE | Chokepoint + element-shape resilience VERIFIED for the tested shapes; one residual non-dict-feature edge bypasses the catch (see Anti-Patterns / Gaps) |
| `scripts/synthetic_weights.py` | SYNTHETIC_LC_WEIGHTS uniform 1.0 | ✓ VERIFIED | 9 canonical keys all 1.0, topo 1.0, source "synthetic" (resolved decision honored) |
| `scripts/satellite/weights.py` | SATELLITE_LC_WEIGHTS balance-tilt | ✓ VERIFIED | trees/water 0.6, cropland/built_up/flooded_wetland 1.3, rest 1.0, topo 1.0, source "satellite" (resolved decision honored) |
| `scripts/satellite/{stac,fetch,coverage}.py` | STAC search / windowed COG / cached coverage picker | ✓ VERIFIED | `eo:cloud_cover` filter, `Window` byte-range read, `_tile_origins`/`WC_REMAP` reuse, cached summary + leap-year-correct season (WR-02) |
| `scripts/tiling.py` | nested 1+4+16 / stride-448 / >50% drop / weight propagation | ✓ VERIFIED | Geometry constants 896/448/224, stride 448, area-based off-fraction `<=0.5`, pyramid.json manifest, byte-identical sample_weights copy; 7 tests pass |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| rumsey.py | allmaps.py | `allmaps.lookup` list[dict] | ✓ WIRED |
| rumsey.py | georef.py | scaled GCPs → affine → GeoTIFF | ✓ WIRED |
| build_historical_dataset.py | rumsey.py | status drives drops dict | ✓ WIRED |
| build_dataset.py | split.json | first-write / frozen-read | ✓ WIRED |
| label.py | synthetic_weights.py | SYNTHETIC_LC_WEIGHTS → sample_weights.json | ✓ WIRED |
| satellite/fetch.py | historical/georef.py | shared write_georeferenced_geotiff (no fork) | ✓ WIRED |
| build_satellite_dataset.py | historical/label.py | make_labels reused verbatim | ✓ WIRED |
| build_{historical,dataset,satellite}.py | tiling.py | tile() after make_labels (3/3) | ✓ WIRED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full offline suite | `pytest tests/ --ignore=tests/integration` | 62 passed | ✓ PASS |
| EVAL-01 split invariants | `pytest tests/test_split.py::{frozen,no_intersection,deterministic} test_tiling.py::test_split_subtree_preserved` | 4 passed | ✓ PASS |
| SC#3 topo boundary gate | `pytest test_render.py::test_synthetic_topo_locked_boundaries` | passed | ✓ PASS |
| Targeted SC artifact tests | `pytest test_split/test_tiling/test_synthetic_weights/test_satellite/test_allmaps/test_georef + topo gate` | 33 passed | ✓ PASS |
| Allmaps A6 multi-annotation | offline `test_allmaps.py` | passed (returns all, [] on 404, skips malformed) | ✓ PASS |
| Online phase gate (6 net tests) | `pytest tests/integration -m integration` | not runnable (no network) | ? SKIP → human |

### Adversarial Sanity-Check — CR-01 Structural Fix (requested)

**Claim under test (02-REVIEW-FIX.md / commit 7bbfc9c):** the `_xy()` chokepoint + per-feature/per-ring try/except in `label.py` and `render.py` make the source "structurally resilient" so that "ANY malformed feature/ring — regardless of element shape (missing z, scalar, string, empty ring) — is skipped + drop-counted, never aborting the whole source."

**Verified holds:** missing-z (3D coords), scalar element, non-numeric string element, empty ring, malformed-Polygon scalar coordinates, MultiPolygon scalar — all neutralised by `_xy`/`_rings`; `test_render.py` regression tests (`test_mixed_malformed_rings_do_not_abort_{label,render}`, `test_malformed_point_does_not_corrupt_bbox`, `test_3d_*`, `test_malformed_polygon_scalar_coords_skipped`) all pass.

**Residual reachable edge found:** A FeatureCollection entry that is itself a **non-dict** (bare string / int / float — not a malformed *element* but a malformed *feature*) reaches `(feat or {}).get("geometry")` / `.get("properties")`, which raises `AttributeError`. `AttributeError` is **not** in the per-feature/per-ring `except (TypeError, ValueError, KeyError, IndexError)` tuple in `_bbox`, `make_label_arrays` (label.py) or `_bbox`, `render_style` (render.py). Reproduced end-to-end: a GeoJSON with `["rogue string", <valid polygon>]` aborts both `make_label_arrays` and `render_one` with `AttributeError("'str' object has no attribute 'get'")` — the whole source is lost including its valid sibling polygon. This is the same crash class CR-01 claimed to have structurally eliminated; the fix neutralised every element shape but not a non-dict whole-feature shape, and no regression test covers it.

**Mitigation in place (why this is WARNING not BLOCKER):** `scripts/build_dataset.py:260-263` wraps `build_one_source` in `except Exception as exc` (per-source resilience T-02-09), so the AttributeError is caught at the batch-orchestration layer — the source is logged FAILED and the batch continues. The historical pipeline does not parse GeoJSON features (it consumes a raster GeoTIFF via `historical/label.py`), so the edge is confined to the synthetic stream. Net effect: a single non-dict-feature source is *entirely dropped* (not just the bad feature) but the batch does not abort. This is a partial regression of the narrower CR-01 per-feature contract, not a phase-failing batch crash.

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|--------------|-------------|--------|----------|
| PHASE-02 | 02-01..05 | 3-family labelled pixel-pair dataset with 4-file per-map schema | ✓ SATISFIED | All 5 SCs verified offline; 3 build pipelines exist + tiler wired; per-map schema enforced by reused label code |
| EVAL-01 | 02-01, 02-03 | Frozen seeded stratified held-out synthetic test set, no leakage | ✓ SATISFIED | Seed 42, frozen split.json, whole-source split, 4 split/subtree tests pass |
| GEOREF-V2 (v2, deferred) | 02-01 | PaliGemma semi-auto + manual MapWarper/QGIS deferred to v2 | ✓ SATISFIED | REQUIREMENTS.md:77 bullet, ROADMAP SC#2 + STATE.md deferral rows present; rumsey emits unregistered_manifest only |

No orphaned requirements: every ID in the five PLAN `requirements:` fields (PHASE-02, EVAL-01) is accounted for; GEOREF-V2 is correctly tracked as a v2 deferral, not a Phase-2 deliverable.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/label.py | 109, 183 | `except (TypeError, ValueError, KeyError, IndexError)` omits `AttributeError` | ⚠️ Warning | Non-dict whole-feature aborts the source (CR-01 contract gap); mitigated by build_dataset.py batch-level `except Exception` |
| scripts/render.py | 167, 216 | same omission in `_bbox` / `render_style` | ⚠️ Warning | Same as above for the render path |
| (none) | — | No TBD/FIXME/XXX debt markers in phase-modified files | ℹ️ Info | Clean — no unreferenced debt-marker gate trip |
| scripts/render.py / label.py | — | IN-01/IN-02 clarity nits (two-band fabrication, obfuscated unpack) | ℹ️ Info | Explicitly out-of-scope INFO in review; not behavior-affecting |

### Human Verification Required

1. **Online phase gate (6 network integration tests)** — run `pytest tests/integration -m integration` in a network-enabled environment. Covers Allmaps annotations, IIIF image fetch, Rumsey LUNA search, Sentinel-2 STAC, s3://sentinel-cogs windowed read. Cannot be verified by grep/offline tests; documented in `deferred-items.md`. This exercises the only code paths (network I/O in Wave-1/2 fetch/search) not covered by the 62-pass offline suite.

2. **Residual non-dict-feature edge — harden or accept** — decide whether to add `AttributeError` to the four per-feature/per-ring `except` tuples in `label.py`/`render.py` (+ a non-dict-feature regression test), OR formally record that the batch-level `except Exception` in `build_dataset.py` is a sufficient mitigation for the CR-01 structural-resilience contract. Either resolves the warning.

### Gaps Summary

No phase-goal-blocking gap was found. All 5 ROADMAP success criteria are structurally implemented, wired, and pass their offline gate tests (62/62). All resolved decisions (A6 all-annotations, A4 per-(source×style) dirs + whole-source split, A7 N=100, synthetic uniform-1.0, satellite balance-tilt) are honored in code. The 19-finding 3-iteration code-review fix history is reflected (WMS removed, leap-year season, satellite NODATA discipline, single shared georef writer).

Two items require a human decision before the phase can be marked fully done:
(1) the documented online integration gate must be run on a networked host (deferred, not a failure);
(2) the requested adversarial sanity-check of the CR-01 structural fix found one reachable residual crash-class edge — a non-dict whole-feature raises an uncaught `AttributeError` and aborts the source. It is bounded to the synthetic stream and mitigated by the batch-level per-source `except Exception`, so it is a WARNING (CR-01 contract gap) rather than a BLOCKER, but it surfaces a developer scope/risk decision: harden the four except tuples or accept the batch-level catch as sufficient.

---

_Verified: 2026-05-15T18:10:00Z_
_Verifier: Claude (gsd-verifier)_
