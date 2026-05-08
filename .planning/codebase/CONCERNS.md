# Codebase Concerns

**Analysis Date:** 2026-05-08

This document inventories technical debt, known issues, and risk inputs feeding the
GSD roadmap. The repository is mid-refactor on branch `refactor_paper`, transitioning
from an earlier hex-tile fine-tuning plan to the system-paper plan in
`mockup.md`. Many concerns below are direct consequences of that refactor being
incomplete; others are gaps between the README's current promise and the code that
exists. Concerns are ordered roughly by severity; each concern is tagged
**HIGH / MEDIUM / LOW**.

---

## 1. Dataset licensing risk (publication-blocking)

The README claims a "fully open-source" combined dataset (see `mockup.md` line 56:
"full open-source and can be used by anyone for further progress"). Three of the four
upstream sources have unverified or ambiguous redistribution terms. If even one source
forbids redistribution of derived training pairs, the open-source claim cannot survive
peer review and the dataset contribution collapses to a recipe rather than an artefact.
This is the highest-priority concern because it can block publication regardless of
code quality.

### 1a. David Rumsey Map Collection — most uncertain source

- **Severity:** HIGH
- **Location:** `scripts/historical/rumsey.py` (entire file pulls from David Rumsey
  LUNA + Georeferencer WMS); referenced from `executive_TODO.md` lines 20–25.
- **Risk:** Originals (16th–17th c.) are public domain, but Rumsey's digitisations and
  georeferenced GeoTIFFs carry separate Stanford institutional terms (last known:
  personal / educational / non-commercial use). The pipeline produces derivative
  label-pair products (`image.png` + `land_cover.png` + `topography.png`) whose
  redistribution status is undetermined.
- **Recommended action:** Contact David Rumsey Map Center at Stanford for written
  clarification on the derived-dataset redistribution terms and required attribution.
  Until resolved, do not publish or distribute any artefact built from
  `scripts/historical/rumsey.py` outputs. If terms forbid redistribution, fall back to
  publishing only the construction recipe and shipping a download script that pulls
  fresh from the source.

### 1b. Copernicus DEM GLO-30 — derivative-product status unclear

- **Severity:** MEDIUM
- **Location:** `scripts/historical/dem.py`; referenced from `executive_TODO.md`
  lines 14–17.
- **Risk:** Slope-derived `topography.png` rasters are computed from Copernicus DEM
  GLO-30 (issued by ESA / Airbus). It is unclear whether classified slope rasters
  count as derivative products under the Copernicus DEM license, and whether
  redistribution requires attribution only or additional terms.
- **Recommended action:** Review the Copernicus DEM license at
  https://spacedata.copernicus.eu/. Document the attribution string in the eventual
  dataset README. If derivative redistribution is restricted, the pipeline can ship
  the slope-classifier code and require users to fetch DEM tiles themselves.

### 1c. Azgaar's Fantasy Map Generator — generator vs. output license

- **Severity:** MEDIUM
- **Location:** `scripts/build_dataset.py`, `scripts/render.py`, `scripts/label.py`,
  `scripts/biome_mapping.py`; referenced from `executive_TODO.md` lines 26–28.
- **Risk:** The generator code is MIT-licensed, but the license status of GeoJSON
  exports used as derivative training data has not been confirmed. Synthetic outputs
  may inherit different terms than the generator code itself.
- **Recommended action:** Confirm with Azgaar (project maintainer) that GeoJSON
  exports may be redistributed as part of an open training dataset.

### 1d. README claim vs. unresolved licensing

- **Severity:** HIGH
- **Location:** `mockup.md` line 56 ("full open-source"); `README.md` line 52
  ("Four open-source sources covering visual diversity").
- **Risk:** The "fully open-source" framing appears in the academic motivation. If
  any of 1a–1c resolves restrictively, the paper's dataset contribution must be
  re-scoped (recipe rather than artefact). Editing this after submission damages
  credibility.
- **Recommended action:** Once licenses are confirmed (per `executive_TODO.md`
  line 30), pick the most restrictive upstream license as the dataset license and
  update `mockup.md` and `README.md` to reflect actual terms. Drop "fully
  open-source" if any source forbids it.

---

## 2. Refactor debt — abandoned hex-tile artefacts still in `scripts/`

The README explicitly states the repo is being "reworked from an earlier hex-tile
fine-tuning plan to the system-paper plan in mockup.md" (`README.md` line 145). Two
files still reflect the abandoned approach and are not referenced by the new pipeline.

### 2a. `scripts/augment.py` — hex-tile augmentation utilities

- **Severity:** MEDIUM
- **Location:** `scripts/augment.py` (entire file)
- **Why pre-refactor:** Module docstring states augmentations are "designed to bridge
  the visual gap between clean hex tile art and the aged, faded, monochrome appearance
  of real historical maps" (`scripts/augment.py` lines 4–6). The `make_grid` function
  is described as creating "'terrain region' patches" by tiling "multiple variants of
  the same hex class" (`scripts/augment.py` lines 116–118). No module under
  `scripts/historical/` or in the synthetic dataset path imports `augment.py`. It was
  added in commit `ba8b407` alongside a now-deleted `scripts/finetune_paligemma.py`.
- **Risk:** Dead code creates confusion for future contributors and the planner — it
  looks live, has no callers, and embeds the old mental model. Some functions
  (`to_parchment`, `to_faded`) may be salvageable for the new historical-map
  augmentation step but currently are not wired in.
- **Recommended action:** Either delete `scripts/augment.py` outright or scope it to
  the system-paper plan (drop `make_grid`, drop hex-tile language in the docstring,
  and document where the parchment / faded effects are called from). Decide as part
  of the dataset-construction phase.

### 2b. `scripts/toon_mapping.py` — hex tile filename → label index

- **Severity:** MEDIUM
- **Location:** `scripts/toon_mapping.py` (entire file)
- **Why pre-refactor:** Module docstring: "Map hex tile filenames to canonical land
  cover and topography class labels. Filename convention: `hex_{terrain}_{variant}_{style}_{n}.png`"
  (`scripts/toon_mapping.py` lines 2–4). The entire `_RAW_MAP` table on lines 24–81
  enumerates `hex_*` prefixes. Comment on line 119: "Text description utilities for
  PaliGemma fine-tuning" — predates the system-paper decision to benchmark
  PaliGemma zero-shot first rather than fine-tune directly. Added in commit
  `ba8b407` with `augment.py`. Not imported anywhere in the new pipeline.
- **Risk:** Same dead-code confusion as 2a. The `data/toons/` directory is mentioned
  in `README.md` line 117 as "Tile assets — kept as validation set / future
  pre-labelled low-level feature library", which keeps the filename map plausibly
  alive, but no current script consumes it.
- **Recommended action:** Decide explicitly whether toons survive into the system-paper
  pipeline. If yes, document where `toon_mapping.py` is consumed. If no, delete the
  file. Either action removes the ambiguity.

### 2c. `data/toons/` directory referenced but unused

- **Severity:** LOW
- **Location:** `README.md` line 117 (`data/toons/`); `scripts/toon_mapping.py`
  line 171.
- **Risk:** Repo-layout doc lists `toons/` as "kept as validation set / future
  pre-labelled low-level feature library", but no v0 script depends on it. Risk of
  shipping unused data assets and inflating dataset size.
- **Recommended action:** Either wire toons into a v1+ component (low-level feature
  library) or remove from `README.md`. Pair with the 2b decision.

---

## 3. Missing core components — gap between README promise and code

These are not bugs; they are deliberate scope. But they are the gap between the
README's GeoViLM architecture description and what currently exists. Each will
become a planning phase. Ordered roughly by execution dependency.

### 3a. Zero-shot benchmark harness

- **Severity:** HIGH (research-credibility blocker)
- **Location:** Not implemented anywhere; tracked in `executive_TODO.md` lines 34–40.
- **Gap:** README phase 2 ("Failure analysis benchmark", `README.md` lines 79–85) and
  `mockup.md` lines 12–16 commit to zero-shot evaluation of CLIP, SigLIP, OpenCLIP,
  and PaliGemma. No code exists to run this. Without it, the dynamic-LRP failure
  localisation contribution has no measured baseline to localise *against*.
- **Recommended action:** Highest-priority new code phase. Build a small harness over
  HuggingFace `transformers` + the existing `biome_mapping.py` taxonomy. Output:
  per-class accuracy + confusion matrix per model on the held-out stylized test set.
  Required before any of (3b) – (3d) are scientifically meaningful.

### 3b. Rotationally-invariant OCR module

- **Severity:** MEDIUM
- **Location:** Not implemented; mentioned in `README.md` lines 32–34, `mockup.md`
  line 18 ("Incorporation of rotationally invariant OCR needs to take place also at
  an early stage").
- **Gap:** README names CRAFT and ABCNet as candidate starting points. No code yet.
  May need its own training data sub-pipeline (`README.md` line 89: "separate
  sub-pipeline, may need its own training data").
- **Recommended action:** Phase 3a sub-task. Decide between adapting an existing
  scene-text recognition model versus training from scratch. Document the training
  data requirement before committing to an approach.

### 3c. Dense segmentation heads

- **Severity:** MEDIUM
- **Location:** Not implemented; `models/` directory referenced in `README.md`
  line 135 ("Trained model checkpoints (currently empty)") — directory does not even
  exist on disk yet (verified via `find -maxdepth 3 -name models`).
- **Gap:** README phase 3 commits to "dense segmentation heads on the chosen
  backbone" but the backbone choice itself is unresolved (see concern 7a).
- **Recommended action:** Blocked by 7a (backbone decision) and 3a (zero-shot
  baseline). Sequence after both.

### 3d. Auto-georeferencing pipeline

- **Severity:** MEDIUM
- **Location:** Not implemented; described as the v0→v1 bootstrapping component in
  `README.md` lines 37–42 and `README.md` line 93.
- **Gap:** Cross-correlation against a WorldCover + DEM reference grid plus
  thin-plate-spline warping. None of this exists. Currently the pipeline only
  consumes maps that are *already* registered in the Georeferencer service
  (`scripts/historical/rumsey.py` — see `_wms_url` filter).
- **Recommended action:** This is also the v0-vs-v1 ablation for the system-paper
  claim, so it has dual purpose. Sequence after segmentation heads (3c) since the v0
  GeoViLM is needed to bootstrap.

### 3e. Road-map data source

- **Severity:** LOW
- **Location:** Not implemented; `README.md` lines 69–70 marks the source itself as
  "TBD source (OSM tile renders are the obvious candidate). Not yet implemented."
- **Gap:** One of the four data sources promised in `mockup.md` line 53 ("road maps")
  has no source decision, no fetcher, no renderer.
- **Recommended action:** Either commit to OSM tile rendering and build the fetcher,
  or drop the road-map source from the dataset claim and update `mockup.md` and
  `README.md` accordingly. Deferral is acceptable but the README claim should match
  the deferral.

---

## 4. No tests, no CI

- **Severity:** HIGH
- **Location:** Repository root (no `tests/` directory, no `pytest.ini`, no
  `tox.ini`, no `.github/workflows/`, no `*.yml` files anywhere).
  Verified: `find /home/drdreadknee/mapclass -type d -name tests` returns empty;
  `find -name "*.yml"` returns empty.
- **Risk:** Every concern below this one (broad exception swallowing, geometry
  assumptions, network-dependent code paths) lacks any automated regression net.
  Refactoring `scripts/historical/` carries no safety guarantee. The label
  taxonomies in `scripts/biome_mapping.py` and `scripts/historical/worldcover.py`
  intersect but their alignment is unverified by tests.
- **Recommended action:** Stand up minimal pytest suite as soon as one of the
  current scripts is touched. Initial targets — pure-function modules with no
  network or filesystem dependence:
  1. `scripts/biome_mapping.py` — `h_to_landcover`, `h_to_topo`, `normalize_land_h`
     (all deterministic, no I/O).
  2. `scripts/historical/worldcover.py::_remap` — purely table-driven.
  3. `scripts/historical/dem.py::_classify` and `_slope_degrees` — pure NumPy.
  Add a GitHub Actions workflow running `pytest` on push. Defer integration tests
  for the network-dependent fetchers (`rumsey.search_maps`,
  `worldcover.fetch_worldcover`, `dem.fetch_topo`) until VCR-style cassettes or
  fixture GeoTIFFs are added.

---

## 5. Hand-wavy claims pending evidence (research credibility)

- **Severity:** HIGH (publication-credibility risk)
- **Location:** `executive_TODO.md` lines 38–40 ("replace the current hand-wavy
  'CLIP is useless' claim with measured evidence"); `mockup.md` lines 13–15
  ("Contrastive vision-language models like CLIP and SigLIP appear to show poor
  zero-shot transfer to stylized cartography; we benchmark this directly").
- **Risk:** The system-paper's mechanistic-failure-analysis contribution rests on
  measurable failures of existing VLMs. Currently the claim is asserted, not
  measured. A reviewer can immediately reject the framing if the supporting
  benchmark is absent or weak. The contribution distinguishing this work from
  MAPWise / MapIQ (`mockup.md` lines 44–48) is *mechanistic* localisation — but
  localisation requires a measured failure to localise.
- **Recommended action:** Tightly coupled to concern 3a. The zero-shot benchmark
  harness must produce per-class accuracy and confusion matrices on the held-out
  stylized test set *before* any version of the paper is drafted. Treat the
  benchmark numbers themselves (and the dynamic-LRP follow-up) as a gating
  artefact for downstream writing.

---

## 6. In-line TODO / FIXME / HACK comments in source

A repo-wide grep for `TODO|FIXME|HACK|XXX` over `scripts/` returned no matches.
Outside of source code there are two `NOTE`-equivalent flags worth listing:

### 6a. Road-map source marked TBD in README

- **Severity:** LOW (already covered in 3e)
- **Location:** `README.md` line 69
- **Action:** See concern 3e.

### 6b. Reference to `data/TODO.md` from README — file does not exist

- **Severity:** LOW
- **Location:** `README.md` line 119 lists `data/TODO.md` ("Open data-side todos")
  in the repo layout. The `data/` directory is gitignored
  (`.gitignore` line 2: `data/`) and `data/TODO.md` is absent on this checkout.
- **Risk:** Either the file exists locally only and is not version-controlled
  (in which case its content is invisible to the planner), or it is a stale
  reference to a planned file that was never created.
- **Recommended action:** Either commit `data/TODO.md` (un-ignore that one
  filename) or remove the reference from `README.md`. Lean toward the former so
  data-side todos are visible.

---

## 7. Open architectural decisions

These are documented in `notes/architectural_references.md` as deliberate open
questions, not oversights. Listed here so the planner can sequence them.

### 7a. Backbone choice — coarse-to-fine vs. large-kernel CNN

- **Severity:** MEDIUM (blocks 3c)
- **Location:** `notes/architectural_references.md` lines 7–28.
- **Status:** Recursive coarse-to-fine segmentation (DINOv2 or Swin) is preferred,
  large-first-kernel ConvNet (RepLKNet / SLaK / ConvNeXt) is the benchmark
  alternative. Decision unmade. PaliGemma is the *separate* vision-language
  backbone (`README.md` lines 31–32) and is not the backbone meant here.
- **Recommended action:** Run a small coarse-to-fine vs. large-kernel comparison on
  the synthetic test set once the segmentation-head infrastructure exists. Until
  then, default to DINOv2 frozen for the v0 GeoViLM as the architectural references
  doc recommends.

### 7b. Progressive backbone unfreezing — planned but not scoped

- **Severity:** LOW
- **Location:** `notes/architectural_references.md` lines 32–36.
- **Status:** Planned follow-up if illustrated-map performance plateaus. Not yet
  scheduled. Includes attribution analysis (GradCAM / attention rollout) at each
  unfreezing stage.
- **Recommended action:** Park as a v2 phase. Not needed for first paper draft.

### 7c. PaliGemma fine-tuning vs. zero-shot-only commitment

- **Severity:** LOW
- **Location:** `README.md` lines 31–32 ("Benchmarked zero-shot first, then
  fine-tuned in-system"); leftover comment in `scripts/toon_mapping.py` line 119
  references PaliGemma fine-tuning utilities from the abandoned hex-tile era.
- **Status:** README commits to fine-tuning after benchmarking, but the
  fine-tuning loop, dataset format, and PEFT/LoRA configuration are unspecified.
  `requirements.txt` already lists `peft>=0.10`, `accelerate`, and
  `bitsandbytes`, signalling intent.
- **Recommended action:** Defer until zero-shot benchmark numbers (3a) tell us
  whether fine-tuning is needed and on what data slice.

---

## 8. Reproducibility risks

### 8a. No pinned random seeds anywhere in the codebase

- **Severity:** MEDIUM
- **Location:** `scripts/augment.py` line 76 (only random call in repo:
  `np.random.normal(0, 6, arr.shape)` for paper-grain noise). No
  `np.random.seed(...)` or `torch.manual_seed(...)` calls anywhere.
- **Risk:** Even before model training begins, the parchment augmentation produces
  non-reproducible noise. Any downstream training run that incorporates parchment
  augmentation cannot be replicated bit-for-bit. Once segmentation heads are
  added (3c), seedless training will compound the problem across data shuffling,
  weight init, dropout, and augmentation.
- **Recommended action:** Define a single project-wide seeding utility (e.g.
  `scripts/seeding.py::set_global_seed(seed)`) that pins
  `random`, `numpy`, `torch`, `torch.cuda` seeds plus
  `torch.backends.cudnn.deterministic`. Call it from every entry point. Pass the
  seed through CLI args. Apply retroactively to `scripts/augment.py::to_parchment`
  before the synthetic dataset is regenerated for the published version.

### 8b. No dataset versioning

- **Severity:** MEDIUM
- **Location:** Repository root — no `dvc.yaml`, no `.dvc/`, no manifest hashes.
  `data/` is entirely gitignored (`.gitignore` line 2).
- **Risk:** The Rumsey LUNA API and AWS S3 buckets (ESA WorldCover, Copernicus
  DEM) all evolve. A dataset rebuilt six months later from the same scripts may
  yield different label maps if upstream data is updated, and there is no manifest
  recording which tile versions were used. ESA WorldCover specifically has
  versioned releases (currently `v200/2021/`, hard-coded in
  `scripts/historical/worldcover.py` line 38 — but no record of which version a
  given training run used).
- **Recommended action:** At minimum, write a `manifest.json` per built dataset
  containing: source URLs, retrieval timestamps, ETags / Last-Modified for every
  S3 object, the Rumsey item IDs used, and the script commit SHA. Consider DVC or
  Git LFS for the actual artefacts once licensing (concern 1) permits hosting.

### 8c. Environment specification limited to `requirements.txt`

- **Severity:** MEDIUM
- **Location:** `requirements.txt` (10 packages, only one with a version pin —
  `transformers>=4.41`, `peft>=0.10`); no `Pipfile.lock`, no `poetry.lock`, no
  `uv.lock`, no `requirements.lock.txt`, no `pyproject.toml`, no
  `.python-version`. Verified via filesystem search.
- **Risk:** Loose pins on heavyweight ML stack (`torch`, `transformers`, `peft`,
  `accelerate`, `bitsandbytes`) effectively guarantee non-reproducible installs
  across machines and dates. The training environment described in
  `README.md` lines 152–154 is on a "separate VM-provisioning repo with an SSH
  workflow" — that repo's environment is not visible from here, so any
  reproducibility claim across the two repos is unenforced.
- **Recommended action:** Add a fully-pinned lockfile (`uv.lock` is the lightest
  lift given uv is already common in this stack). Pin a Python version
  (`.python-version`). Document the CUDA / driver assumptions for `bitsandbytes`
  and the GPU expectations for `torch`. Cross-link the VM-provisioning repo's
  environment manifest from this repo's README so the chain is visible.

### 8d. Network-dependent fetchers with broad exception swallowing

- **Severity:** MEDIUM
- **Location:**
  - `scripts/historical/dem.py` line 127–128: `except Exception: pass`
    (silently skips every DEM tile failure as if it were an ocean tile).
  - `scripts/historical/worldcover.py` line 131: `except Exception as exc: print(...)`
    (logs but does not fail).
  - `scripts/historical/rumsey.py` lines 228, 314: similar.
  - `scripts/build_historical_dataset.py` line 78: catches all exceptions per
    map and continues.
- **Risk:** The `pass`-on-failure pattern in `dem.py` is correct for genuinely
  oceanic tiles (404 is expected) but indistinguishable from transient S3 errors
  or auth failures, which would silently produce label rasters with missing
  topography. The collected dataset can be silently incorrect, and re-running
  the build will not regenerate failed tiles because there is no failure record.
- **Recommended action:** Distinguish 404 (ocean tile, expected) from other
  errors in `scripts/historical/dem.py::fetch_topo`. Log non-404 failures to a
  per-map manifest entry. Pair with concern 8b — every dataset build should
  emit a record of which tiles were skipped and why.

---

## Summary table

| # | Concern | Severity | Primary location |
|---|---------|----------|------------------|
| 1a | David Rumsey licensing | HIGH | `scripts/historical/rumsey.py`, `executive_TODO.md` |
| 1b | Copernicus DEM licensing | MEDIUM | `scripts/historical/dem.py`, `executive_TODO.md` |
| 1c | Azgaar output licensing | MEDIUM | `scripts/build_dataset.py`, `executive_TODO.md` |
| 1d | "Fully open-source" claim vs. unresolved licensing | HIGH | `mockup.md`, `README.md` |
| 2a | `scripts/augment.py` hex-tile dead code | MEDIUM | `scripts/augment.py` |
| 2b | `scripts/toon_mapping.py` hex-tile dead code | MEDIUM | `scripts/toon_mapping.py` |
| 2c | `data/toons/` referenced but unused | LOW | `README.md`, `scripts/toon_mapping.py` |
| 3a | Missing zero-shot benchmark harness | HIGH | not implemented; `executive_TODO.md` |
| 3b | Missing rotationally-invariant OCR | MEDIUM | not implemented; `README.md` |
| 3c | Missing dense segmentation heads | MEDIUM | not implemented; `models/` empty |
| 3d | Missing auto-georeferencing pipeline | MEDIUM | not implemented; `README.md` |
| 3e | Missing road-map data source | LOW | `README.md` |
| 4 | No tests, no CI | HIGH | repository root |
| 5 | Hand-wavy "CLIP is useless" claim | HIGH | `mockup.md`, `executive_TODO.md` |
| 6a | TBD road-map source flag | LOW | `README.md` |
| 6b | `data/TODO.md` referenced but absent | LOW | `README.md` |
| 7a | Backbone choice unmade | MEDIUM | `notes/architectural_references.md` |
| 7b | Progressive unfreezing experiment unscheduled | LOW | `notes/architectural_references.md` |
| 7c | PaliGemma fine-tuning loop unspecified | LOW | `README.md`, `requirements.txt` |
| 8a | No pinned random seeds | MEDIUM | `scripts/augment.py` and project-wide |
| 8b | No dataset versioning / manifest | MEDIUM | repository root |
| 8c | Loose `requirements.txt`, no lockfile | MEDIUM | `requirements.txt` |
| 8d | Broad exception swallowing in fetchers | MEDIUM | `scripts/historical/*.py` |

---

*Concerns audit: 2026-05-08*
