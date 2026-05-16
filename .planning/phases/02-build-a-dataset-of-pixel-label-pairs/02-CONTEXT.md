# Phase 02: build-a-dataset-of-pixel-label-pairs - Context

**Gathered:** 2026-05-16 (REWORK — supersedes the original 2026-05-15 context)
**Status:** Ready for planning (replan all 5 plans; prior verification VOID)

> **REWORK NOTICE.** Phase 2 was marked `complete`/`passed` but its primary
> deliverable — the synthetic dataset and its raw Azgaar `.geojson` inputs —
> was lost everywhere (local-only persistence; ephemeral GPU boxes destroyed
> it; nothing in GCS). Root cause is an architecture defect, not a bug. The
> prior `02-VERIFICATION.md`/`02-HUMAN-UAT.md` verdicts are void. All 5 prior
> plans assumed local-disk persistence and a frozen `split.json` that no longer
> exists — they are superseded. See project memory
> `synthetic-dataset-loss-phase2-rework`.

<domain>
## Phase Boundary

Produce labelled pixel↔class training data for the three source families
(historical illustrated, synthetic Azgaar+Pillow, satellite ESA WorldCover +
Copernicus DEM), with **GCS as the single canonical store** so disposable
compute can never destroy inputs or outputs again. The dataset taxonomy,
labelling, height re-normalisation, per-source loss weights, and split
*semantics* are unchanged from the original Phase 2 — **only persistence
location, the GCS I/O paths, and the split freeze-epoch change.**

Not in scope: changing the 9/3-class taxonomy, label sources, weighting policy,
or split algorithm; Phase 4 training itself; recovering the lost Azgaar maps
(unrecoverable — full manual regeneration accepted by the user).
</domain>

<decisions>
## Implementation Decisions

### GCS write strategy (RW-01)
- **RW-01: Direct gcsfs streaming.** `build_dataset.py`,
  `build_historical_dataset.py`, `build_satellite_dataset.py`, and `tiling.py`
  write outputs **directly to `gs://mapclass-training-northeast1/data/` via
  gcsfs/fsspec** — no canonical local copy. `split.json` is written directly to
  GCS at split-compute time (naturally durable; no separate early-write step).
- **⚠ Flagged risk for researcher (do not let planner skip):** `tiling.py` is
  deeply local-`Path`-based (nested-pyramid `tiling.tile(out)`, Pillow
  `.save`). Streaming-write requires a path abstraction (fsspec) refactor, and
  many small per-tile object writes to GCS at pyramid scale have real
  throughput/cost implications. Researcher MUST evaluate (a) an fsspec/gcsfs
  path-abstraction approach across `build_*` + `tiling.py`, and (b) streaming
  small-object write throughput/egress cost, BEFORE the planner commits an
  implementation. User chose this over scratch+rsync with the invasiveness
  understood — do not re-litigate, but surface the cost honestly.

### Raw-input naming contract (RW-02)
- **RW-02:** Hand-created Azgaar exports are named
  `<template>_<NN>.geojson` (lowercase, zero-padded `NN`, e.g.
  `europe_01.geojson`) and uploaded to
  `gs://mapclass-training-northeast1/data/synthetic/raw/`.
- A user-authored `raw/manifest.json` maps every filename → continent
  template. `build_dataset.py` **validates filenames against the manifest and
  HARD-FAILS** on any mismatch / missing entry / unlisted file **before**
  computing or freezing the split. Filename convention is primary; the manifest
  is the authoritative cross-check that prevents a single mis-named hand-created
  file from silently skewing the EVAL-01 stratified hold-out.

### Train/eval data access (RW-03)
- **RW-03: Pull-once to local cache, verify, then train.** At job start,
  `finetune_seg.py` / `evaluate_seg.py` bulk-fetch the needed split subset from
  GCS to local scratch, **verify against `split.json`** (membership + presence;
  cheap checksum if feasible), then train/eval from local disk. Egress paid
  once per job; resumable after preemption — consistent with the existing
  `seg/gcs_checkpoint.py` preemption-safety design.

### Scope — source families (RW-04)
- **RW-04: ALL THREE families are GCS-canonical.** synthetic (rebuilt from
  scratch), historical, and satellite. Each build script persists raw inputs
  (where applicable), built `train/`+`test/` outputs, and manifests to
  `gs://mapclass-training-northeast1/data/{synthetic,historical,satellite}/`.
  Local disk is scratch only across all three pipelines.

### Project decisions REVERSED by this rework (planner/executor MUST update PROJECT.md)
- **D-06** ("pre-tile at build time on disk"): still pre-tile, but tiles are
  persisted to GCS via streaming — not canonical-on-disk.
- **D-17** ("sibling dirs `data/synthetic/{train,test}/`, filesystem-level
  separation"): REVERSED → `gs://…/data/synthetic/{train,test}/` + `split.json`
  are canonical; filesystem separation is a transient local-scratch detail
  during the RW-03 pull-once.
- **D-18** ("snapshot at first build, frozen by ID list at
  `data/synthetic/split.json`"): `split.json` now at
  `gs://…/data/synthetic/split.json`; **the prior frozen split is GONE.** A NEW
  split freezes on the regenerated Azgaar sources. **Phase 4 EVAL-01 hold-out
  WILL differ from the original 04 plans — accepted by the user, logged.**

### Carried forward UNCHANGED (do not re-discuss)
- **D-15 / D-16 split semantics:** whole Azgaar source held out end-to-end,
  stratified by continent template, fixed seed, ~15% — algorithm unchanged.
- 9/3-class taxonomy; height re-normalisation (flat ≤20 / hilly 20–55 /
  mountainous >55 over `[0,100]`); class-conditional per-source loss weights;
  `sample_weights.json` schema.

### Claude's Discretion
- Exact gcsfs/fsspec abstraction shape, manifest JSON schema, verification
  granularity (count vs checksum) — researcher/planner decide within RW-01..03.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Decisions / requirements to reconcile
- `.planning/PROJECT.md` — D-06, D-15..D-18 (D-06/D-17/D-18 to be REVERSED;
  D-15/D-16 semantics retained); EVAL-01 metric definition.
- `.planning/REQUIREMENTS.md` — PHASE-02, EVAL-01 (zero-leakage hold-out).
- `.planning/ROADMAP.md` §"Phase 2" — goal + 5 success criteria (criterion 5
  = held-out synthetic guaranteed unseen; now re-frozen on new sources).
- Project memory `synthetic-dataset-loss-phase2-rework` — incident + locked
  user decisions.

### Code to rework / reuse
- `scripts/build_dataset.py` — synthetic orchestrator (local-only today).
- `scripts/build_historical_dataset.py`, `scripts/build_satellite_dataset.py`
  — historical/satellite orchestrators (RW-04 brings them in scope).
- `scripts/tiling.py` — nested-pyramid tiler; **main refactor risk** (RW-01).
- `scripts/finetune_seg.py`, `scripts/evaluate_seg.py` — consumers (RW-03).
- `scripts/seg/gcs_checkpoint.py` — **existing GCS auth (ADC / V4 IAM) +
  gcsfs usage pattern; reuse it as the GCS I/O template.**
- `scripts/render.py`, `scripts/label.py`, `scripts/biome_mapping.py`,
  `scripts/synthetic_weights.py` — pure transforms, logic unchanged.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `seg/gcs_checkpoint.py` already establishes ADC / V4-IAM auth against
  `gs://mapclass-training-northeast1/` and a gcsfs access pattern — the GCS
  I/O for build/eval should reuse this, not reinvent auth.

### Established Patterns
- `build_dataset.py` collision-guard + `load_or_create_split` freeze logic is
  sound and should be preserved — only its read/write target moves to GCS and
  the manifest hard-fail (RW-02) is added before `load_or_create_split`.

### Integration Points
- **Researcher MUST check:** `gs://…/data/toons/<biome>/` already holds the
  toon render assets `render.py` consumes, and `data/historical/raw/` holds
  `unregistered_manifest.json`. Confirm whether `render.py` (and the historical
  pipeline) currently read these from GCS or local — RW-01/RW-04 require GCS
  reads to work end-to-end on a fresh box.
- New code connects via gcsfs at: build output write, split.json read/write,
  raw geojson + manifest read, and the RW-03 pull-once in finetune/evaluate.

</code_context>

<specifics>
## Specific Ideas

- Canonical bucket prefix: `gs://mapclass-training-northeast1/data/`
  (https mirror: `https://storage.googleapis.com/mapclass-training-northeast1/data/`).
  Already contains `historical/raw/unregistered_manifest.json` and
  `toons/<biome>/…`; this rework adds `synthetic/{raw,train,test,split.json}`
  and brings `historical/` + `satellite/` fully under the same convention.
- Long-pole external dependency (user, manual, blocks all downstream): re-create
  ~100 Azgaar maps across ~12 continent templates in the Azgaar Fantasy Map
  Generator, name per RW-02, author `raw/manifest.json`, upload to
  `…/data/synthetic/raw/`.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. The lost Azgaar maps are NOT
deferred (unrecoverable; full regeneration accepted). The differing Phase 4
EVAL-01 hold-out is NOT deferred (accepted consequence, logged).

</deferred>

---

*Phase: 02-build-a-dataset-of-pixel-label-pairs*
*Context gathered: 2026-05-16 (rework)*
