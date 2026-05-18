# MapClass / GeoViLM

## What This Is

MapClass is a training pipeline and model for **dense per-pixel land cover and topography prediction on stylized maps** (synthetic fantasy, historical, road, satellite). The model — GeoViLM, a small vision-language backbone with a rotationally-invariant OCR head and two dense segmentation heads — outputs per-pixel class probabilities consumed by a separate hex-grid aggregator app. Target use is grand strategy game cartography (EU4 / CK3 / HoI4 modding) at 100–2000 km scale; the framework generalises to any stylized cartographic source.

## Core Value

**A small, cheap-to-run model that returns reliable per-pixel land-cover + topography probability maps on stylized inputs.** If everything else fails, the model must produce usable pixel labels on a fantasy map at single-CPU / small-GPU inference cost.

## Requirements

### Validated

<!-- Inferred from existing code on `refactor_paper` branch (see .planning/codebase/) -->

- ✓ **DATA-01** Synthetic dataset pipeline (Azgaar Fantasy Map Generator + custom Pillow renderer with multi-style outputs) — `scripts/build_dataset.py`, `scripts/render.py`, `scripts/label.py`, `scripts/biome_mapping.py`, `scripts/augment.py` — existing
- ✓ **DATA-02** Historical dataset pipeline (David Rumsey LUNA API + ESA WorldCover S3 + Copernicus DEM S3) — `scripts/build_historical_dataset.py`, `scripts/historical/{rumsey,worldcover,dem,label}.py` — existing
- ✓ **DATA-03** Canonical 9-class land-cover taxonomy (water/trees/shrubland/grassland/cropland/built-up/bare-sparse/flooded-wetland/snow-ice) mapped from ESA WorldCover, with Mangroves→Trees and Moss/Lichen→Bare-sparse fold — existing
- ✓ **DATA-04** 3-class topography taxonomy (flat <2°, hilly 2–15°, mountainous >15°) derived from Copernicus DEM GLO-30 slope — existing

### Active

<!-- v1 hypotheses. Building toward these. -->

- [ ] **DATA-05** OSM road-tile dataset sub-pipeline as the third v1 source
- [ ] **DATA-06** Per-source class-conditional loss weights (topography fully trusted; water mostly trusted; trees/built-up/cropland heavily downweighted on historical maps; bare/sparse and snow/ice trusted)
- [ ] **GEOREF-01** Auto-georeferencing tool (cross-correlation against WorldCover + DEM reference grid → thin-plate-spline warping) — used as a training-data-prep tool, not an inference component
- [ ] **GEOREF-02** Bootstrap loop: train v0 on already-registered historical maps → use v0 to register currently-unregistered Rumsey maps → retrain v1 on the expanded dataset
- [ ] **MODEL-01** Small VL backbone for the v1 ship target — the inference module's backbone (specific candidate selected in research phase; PaliGemma-3B is excluded as the *ship* backbone by inference budget)
- [ ] **MODEL-02** Rotationally-invariant OCR module for curved / rotated map text (CRAFT or ABCNet as candidate starting points; specific choice in research phase)
- [ ] **MODEL-03** Two lightweight dense segmentation heads on the backbone's vision features (one per output attribute: land cover, topography)
- [ ] **MODEL-04** PaliGemma-3B benchmark variant — same GeoViLM stack (OCR + seg heads + auto-georef bootstrap) with PaliGemma-3B swapped in as the backbone in place of MODEL-01. Trained as a bellwether upper bound to validate the small-backbone choice; **not** shipped as inference
- [ ] **TRAIN-01** End-to-end training of GeoViLM v0 with per-source loss weighting on the synthetic + already-registered-historical + OSM dataset, run for **both backbones** (MODEL-01 small VL + MODEL-04 PaliGemma)
- [ ] **TRAIN-02** End-to-end retraining of GeoViLM v1 on the bootstrap-expanded dataset, run for **both backbones**
- [ ] **EVAL-01** Held-out joint per-pixel NLL `−(log p_land_cover + log p_topography)` on synthetic + historical test splits as the v1 ship metric (computed for the small-backbone variant; PaliGemma variant evaluated under EVAL-03)
- [ ] **EVAL-02** Qualitative spot-check renders on Tolkien Middle-Earth, Westeros, Abercrombie First-Law Circle of the World, and Warhammer Old World as ship demos (no ground truth, no metric); rendered for both backbones
- [ ] **EVAL-03** PaliGemma-vs-small-backbone NLL comparison on the same held-out splits — bellwether for the small-backbone choice. If the gap is too large to accept, the small-backbone candidate (MODEL-01) is reconsidered before ship
- [ ] **SHIP-01** Inference Python module + trained checkpoint of the **small-backbone variant**, packaged for the separate hex-grid app to consume; runs on single CPU or 4–8 GB consumer GPU. PaliGemma checkpoint is retained as a research/benchmark artifact, not packaged for the app

### Out of Scope (v1)

<!-- Explicit boundaries with reasoning to prevent re-adding. -->

- **PaliGemma-3B as the v1 ship / inference backbone** — 6 GB FP16 weights alone, breaks the single-CPU / small-GPU inference budget. PaliGemma *training* is in scope as a v1 benchmark/bellwether variant (MODEL-04, EVAL-03); only its use as the packaged inference backbone consumed by the hex-grid app is excluded.
- **Auto-georef as an inference / app feature** — the modder's input is a fantasy map with no real-world coordinates. Auto-georef is a training-data-prep tool only.
- **Hex-grid aggregation, polygon tracing, region delineation** — owned by the separate hex-grid app downstream of this project.
- **EU4 / CK3 / HoI4 engine-specific output formats** (terrain.bmp, heightmap.png, province bitmaps) — output is generic per-pixel class probabilities; the hex-grid app handles engine packaging.
- **Hand-annotated fantasy test set** — qualitative spot-checks only in v1; deferred to milestone 2 as a paper-credibility item.
- **Zero-shot CLIP / SigLIP / OpenCLIP / PaliGemma baselines** — paper-flavored, deferred to milestone 2 (executive_TODO.md item).
- **Dynamic-LRP mechanistic failure analysis** — paper-flavored, deferred to milestone 2.
- **Component ablations** (backbone-only vs +OCR vs +seg-heads vs full system) — paper-flavored, deferred to milestone 2.
- **Paper draft / write-up** — milestone 2, opportunistic.
- **Redistribution of the combined training dataset** — license risk on Rumsey, Copernicus DEM, and Azgaar outputs has not been verified and the executive_TODO.md tracks owner-driven licensing contact. v1 ships weights + code only; modders supply their own input maps.
- **Hand-annotation tooling integration** (Label Studio / CVAT) — only relevant if a real-domain annotated test set is added later.
- **Cloud / docker / VM provisioning** — handled by a separate VM-provisioning repo per existing project boundary; this repo holds training scripts only.
- **Tests, CI, linting / formatting tooling** — single-author research codebase; explicitly accepted as out of scope at v1 (flagged as a real risk in `.planning/codebase/CONCERNS.md` and `TESTING.md`; revisit if collaborators join or if release packaging needs hardening).

## Context

**Project state at initialization:** Mid-refactor on branch `refactor_paper`. The codebase is being reworked from an earlier hex-tile fine-tuning plan to the multi-component system-paper plan in `mockup.md`. Synthetic and historical dataset construction are working; benchmark harness, OCR module, segmentation heads, training, and the auto-georef bootstrap are not yet implemented.

**Codebase findings (see `.planning/codebase/`):**
- No tests, no CI, no enforced linter — `print()` for diagnostics, mixed CLI conventions (bare `sys.argv` for small scripts; `argparse` only in `build_historical_dataset.py`).
- Type hints use modern PEP 604/585 syntax on public surfaces; relaxed on private helpers.
- 10 `try`/`except` blocks total — all in the historical pipeline; synthetic pipeline has zero exception handlers.
- Six untested high-risk areas flagged in `TESTING.md`: taxonomy index alignment across remap tables, duplicated geometric helpers, slope thresholds, network paths, CRS round-tripping, loss-weight schema.
- License risk on three of four data sources (Rumsey, Copernicus DEM, Azgaar) tracked in `executive_TODO.md`; addressed in v1 by punting redistribution.

**Goal hierarchy (in order):**
1. A cheap inference model usable from a downstream hex-grid app — primary, app-track.
2. A way to learn multi-modal modelling — met by training the VL+OCR+heads system.
3. A way to play with [dynamic LRP](https://arxiv.org/pdf/2512.07010) — milestone-2 paper attempt, opportunistic.

**Paper framing reduction:** The original mockup.md claimed three contributions (mechanistic failure analysis, an open multi-source dataset, and the GeoViLM architecture). With the v1 license decision (don't redistribute the dataset), the "fully open-source dataset" contribution is dropped. If milestone 2 attempts publication, the contributions narrow to mechanistic failure analysis (dynamic LRP) + the GeoViLM system-architecture claim, with the dataset being "trained on these sources, here's how" rather than a redistributable artifact. mockup.md will need a small revision when milestone 2 starts.

**Cloud / training workflow:** Cloud training is handled by a separate VM-provisioning repo with an SSH workflow. This repo holds training scripts but does not carry docker or cloud-build infrastructure.

## Constraints

- **Inference budget**: Trained model + inference module must run on a single CPU-only host or a 4–8 GB consumer GPU — Drives backbone size; rules out PaliGemma-3B.
- **Data licensing**: Three of four upstream data sources (Rumsey, Copernicus DEM, Azgaar) have unverified redistribution terms — Forces v1 to ship weights + code only and not redistribute the combined dataset.
- **Single developer**: One owner, no team — Drives toward MVP scope, vertical slices, and avoidance of test/CI infrastructure overhead at v1.
- **Downstream consumer is fixed**: Output is consumed by the user's separate hex-grid app — Output format is generic per-pixel class probabilities; engine-specific packaging is out of scope.
- **Training compute is external**: Heavy training runs on a separate VM repo over SSH — This repo carries training *scripts*, not training *infrastructure*.
- **Two output taxonomies are fixed**: 9-class land-cover (ESA WorldCover-derived) and 3-class topography (Copernicus DEM slope) — These are the contracts; downstream hex-grid app depends on them.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| App-first; paper attempt is opportunistic in milestone 2 | Owner's primary goal is a usable model for the downstream hex-grid app; publication is a stretch | — Pending |
| v1 model components: VL backbone + OCR + dense seg heads. Auto-georef is training-tool-only | Inference budget rules out a heavy VLM for app use; hex-grid downstream consumes pixel probabilities; fantasy maps don't georef | — Pending |
| PaliGemma-3B excluded as the v1 ship/inference backbone but included as a v1 benchmark/bellwether variant | 6 GB FP16 weights break the inference budget for shipping. Training the same GeoViLM stack with PaliGemma swapped in (MODEL-04) gives an upper-bound NLL number that validates whether the small-backbone choice (MODEL-01) is good enough; if the gap is too large, the small-backbone candidate is reconsidered before ship | — Pending |
| OSM road tiles included as the third v1 dataset source | Improves robustness on stylized maps that emphasise roads; small additive sub-pipeline next to the working synthetic and historical pipelines | — Pending |
| Auto-georef bootstrap loop (v0 → register → v1 retrain) included in v1 | Expands the historical training set; the v0/v1 dataset gap also makes a clean ablation if milestone-2 paper happens | — Pending |
| Validation v1: held-out NLL on synthetic + historical, qualitative spot-checks on four target fantasy maps | Real ground truth lives only in the registered sources; fantasy validation is qualitative only at v1 cost | — Pending |
| License: ship weights + code, don't redistribute the dataset | Three sources have unverified redistribution terms; punting redistribution removes the gating risk for v1 ship; drops the "open dataset" contribution claim | — Pending |
| v1 horizon ends at model handoff (checkpoint + inference module + benchmark numbers) | Bounds the roadmap; paper items move to milestone 2 | — Pending |
| Tests / CI / linting deferred from v1 | Single-author research codebase, accepted risk; `.planning/codebase/CONCERNS.md` and `TESTING.md` flag the trade-off | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-08 after initialization*
