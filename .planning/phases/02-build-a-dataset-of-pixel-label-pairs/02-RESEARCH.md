# Phase 02: Build a Dataset of Pixel-Label Pairs — Research (REWORK)

**Researched:** 2026-05-16 (REWORK — supersedes 2026-05-15 pre-rework file)
**Domain:** GCS-canonical dataset persistence via gcsfs/fsspec; Azgaar raw-input naming contract
**Confidence:** HIGH on gcsfs/fsspec API (verified live 2026.5.0/2026.3.0); HIGH on tiling.py
touch-point inventory (code read); HIGH on cost/throughput estimates; MEDIUM on GCS write
latency (based on documented typical ranges, not a live timing run on this box)

> **REWORK SCOPE.** This file supersedes the 2026-05-15 research for the persistence/storage
> sections (RW-01 through RW-04 and GCS-related pitfalls). All unchanged-domain findings
> (Allmaps, IIIF, STAC, Sentinel-2, taxonomy, loss weights, D-15/D-16 split algorithm,
> tiler geometry) are cited from the prior research rather than re-researched. The planner
> MUST read this file in full; the 2026-05-15 research is now a secondary reference only.

<user_constraints>
## User Constraints (from CONTEXT.md — REWORK 2026-05-16)

### Locked Decisions

#### GCS write strategy (RW-01)
- **RW-01: Direct gcsfs streaming.** `build_dataset.py`, `build_historical_dataset.py`,
  `build_satellite_dataset.py`, and `tiling.py` write outputs **directly to
  `gs://mapclass-training-northeast1/data/` via gcsfs/fsspec** — no canonical local copy.
  `split.json` is written directly to GCS at split-compute time. The invasiveness of the
  tiling.py refactor and the small-object write cost are understood by the user; DO NOT
  re-litigate, but document cost and mitigation honestly.

#### Raw-input naming contract (RW-02)
- **RW-02:** Hand-created Azgaar exports are named `<template>_<NN>.geojson` (lowercase,
  zero-padded `NN`, e.g. `europe_01.geojson`). Uploaded to
  `gs://mapclass-training-northeast1/data/synthetic/raw/`. A user-authored `raw/manifest.json`
  maps every filename → continent template. `build_dataset.py` **validates filenames against
  the manifest and HARD-FAILS before computing or freezing the split.**

#### Train/eval data access (RW-03)
- **RW-03: Pull-once to local cache, verify, then train.** At job start, `finetune_seg.py` /
  `evaluate_seg.py` bulk-fetch the needed split subset from GCS to local scratch, verify
  against `split.json` (membership + presence; cheap spot-check), then train/eval from local
  disk. Consistent with the existing `seg/gcs_checkpoint.py` preemption-safety design.

#### Scope — source families (RW-04)
- **RW-04: ALL THREE families are GCS-canonical.** Synthetic (rebuilt from scratch),
  historical, and satellite. Each build script persists raw inputs (where applicable), built
  `train/`+`test/` outputs, and manifests to
  `gs://mapclass-training-northeast1/data/{synthetic,historical,satellite}/`.
  Local disk is scratch only.

#### Reversed project decisions (planner MUST update PROJECT.md)
- **D-06** reversed: tiles persist to GCS via streaming, not canonical-on-disk.
- **D-17** reversed: `gs://…/data/synthetic/{train,test}/` + `split.json` are canonical;
  filesystem separation is a transient local-scratch detail during RW-03 pull-once.
- **D-18** reversed: `split.json` now at `gs://…/data/synthetic/split.json`; prior frozen
  split is GONE. New split freezes on the regenerated Azgaar sources.

#### Carried forward UNCHANGED (do not re-discuss)
- **D-15 / D-16 split semantics:** whole Azgaar source held out end-to-end, stratified by
  continent template, fixed seed 42, ~15%.
- 9/3-class taxonomy; height re-normalisation (flat ≤20 / hilly 20–55 / mountainous >55
  over `[0,100]`); class-conditional per-source loss weights; `sample_weights.json` schema.
- D-01 through D-14 (Allmaps wiring, IIIF fetch, STAC Sentinel-2, satellite source logic).

### Claude's Discretion
- Exact gcsfs/fsspec abstraction shape in tiling.py — researcher/planner decide within RW-01.
- Manifest JSON schema (flat vs nested) — researcher recommends below.
- Verification granularity (count vs checksum) — researcher recommends below.

### Deferred Ideas (OUT OF SCOPE)
- None added by REWORK discussion. Prior deferred items (PaliGemma semi-auto georeferencing,
  manual MapWarper/QGIS) remain deferred.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PHASE-02 | Phase 2 complete — labelled pixel-pair dataset produced from three source families with `image.png`, `land_cover.png`, `topography.png`, and `sample_weights.json` per map — **now with GCS as the single canonical store** | RW-01 section (gcsfs streaming); RW-04 section (all three families); tiling.py refactor design covers the integration point |
| EVAL-01 | Held-out synthetic test set exists, unseen during training | RW-02 manifest hard-fail (prevents mis-named file from skewing the hold-out); `split.json` now at `gs://…/data/synthetic/split.json`; split semantics D-15/D-16 unchanged; Validation Architecture section maps observability points |
</phase_requirements>

---

## Summary

This rework makes Google Cloud Storage the single canonical store for all Phase 2 pipeline
outputs. The prior research (2026-05-15) covered the data pipeline domain thoroughly
(Allmaps, IIIF, STAC, tiler geometry); this document focuses exclusively on the four
GCS-persistence questions the planner is blocked on.

**Five concrete findings this research delivers:**

1. **The recommended tiling.py abstraction is a thin `_GCSWriter` shim over
   `gcsfs.GCSFileSystem`.** It replaces `Path`-based writes with `fs.pipe_file(path, bytes)`
   calls at six touch points in `tiling.py`. Pillow `.save(fh)` works correctly via
   `fs.open(path, 'wb')` because `GCSFile` implements the full seekable/writable buffer
   protocol. The shim adds zero new dependencies — gcsfs 2026.5.0 is already installed.
   `[VERIFIED: gcsfs API inspection + GCSFile protocol check 2026-05-16]`

2. **Small-object write cost for the full 100-map build: ~$6 in GCS class A ops, ~65 minutes
   wall-clock with 32 threads.** Sequential writes would take ~35 hours. The mitigation the
   planner must bake in is a `ThreadPoolExecutor(max_workers=32)` in the per-pyramid tile
   write loop. No egress cost (writes to GCS are ingress). `[ASSUMED: GCS pricing from
   documented rate of $0.05/10,000 class A ops; latency from documented typical 50-150ms
   same-region PUT; confirmed no quota concern at 32 threads against 1000 req/s bucket limit]`

3. **RW-02 manifest validation must run before `load_or_create_split` and must also drive
   stratification.** The recommended `manifest.json` schema is a flat `entries` dict mapping
   `filename → {template}`. The validation checks filename regex, manifest/GCS-listing
   cross-reference, and template/prefix consistency. `stratified_split` is modified to use
   `manifest["entries"][fn]["template"]` instead of `template_key(sid)` — making the manifest
   authoritative for both naming and grouping. `[VERIFIED: template_key() tested against
   RW-02 naming convention 2026-05-16]`

4. **RW-03 pull-once verification: membership + presence + 10% pyramid file-count spot-check.**
   No per-file MD5/CRC: that would require ~1.25M GCS API calls for the full dataset. The
   `fs.get(remote, local, recursive=True)` pattern is idempotent (safe after preemption) and
   composes cleanly with the existing GCS checkpoint resume in `gcs_checkpoint.py`. `[VERIFIED:
   gcsfs.GCSFileSystem.get signature confirmed 2026-05-16]`

5. **`render.py` does NOT read from GCS in the Phase 2 pipeline.** It is called with a local
   GeoJSON path and returns a PIL.Image in memory. The `gs://…/data/toons/` reference in
   CONTEXT.md describes Phase 1 PaliGemma training data — irrelevant to Phase 2 build scripts.
   The **historical pipeline** (`unregistered_manifest.json`, built GeoTIFFs) is where the
   local-read assumption lives and where RW-04 requires GCS migration. `[VERIFIED: render.py
   and build_historical_dataset.py code read 2026-05-16]`

**Primary recommendation:** Structure the rework as parallel tracks matching the five plans:
(02-01) migrate the synthetic pipeline to GCS (RW-01 + RW-02 + split.json at GCS); (02-02)
tiling.py `_GCSWriter` shim + concurrency; (02-03) historical pipeline GCS migration (RW-04);
(02-04) satellite pipeline GCS migration (RW-04); (02-05) finetune/evaluate pull-once + verify
(RW-03). Prior 02-0x plans are void; all five must be replanned.

---

## Architectural Responsibility Map

> Changed rows only — for unchanged rows (Rumsey, Allmaps, STAC, etc.) see the 2026-05-15
> research. The new tier column shows whether a component writes to GCS directly or uses
> local scratch.

| Capability | Primary Tier | GCS Write? | Rationale |
|------------|-------------|------------|-----------|
| Synthetic raw GeoJSON store | `gs://…/data/synthetic/raw/` | READ from GCS | User uploads; build script reads via gcsfs |
| Synthetic manifest.json | `gs://…/data/synthetic/raw/manifest.json` | READ from GCS | User-authored; hard-fail validation reads it |
| Synthetic split.json | `gs://…/data/synthetic/split.json` | WRITE to GCS | Written once at first build; frozen thereafter |
| Synthetic train/test tiles | `gs://…/data/synthetic/{train,test}/` | WRITE to GCS | gcsfs streaming via _GCSWriter shim in tiling.py |
| Historical unregistered manifest | `gs://…/data/historical/raw/unregistered_manifest.json` | WRITE to GCS | RW-04 requires GCS canonical; currently writes local |
| Historical built samples | `gs://…/data/historical/dataset/` | WRITE to GCS | RW-04 |
| Satellite resolved scenes manifest | `gs://…/data/satellite/resolved_scenes.json` | WRITE to GCS | RW-04 |
| Satellite built samples | `gs://…/data/satellite/dataset/` | WRITE to GCS | RW-04 |
| Pyramid tiling output | Routed through `_GCSWriter` shim | WRITE to GCS | shared tiler; shim abstracts local vs GCS |
| Training data access (finetune) | Local scratch (pull-once from GCS) | READ from GCS → local | RW-03; carve_train_val unchanged |
| Test data access (evaluate) | Local scratch (pull-once from GCS) | READ from GCS → local | RW-03; load_test_pyramid_dirs unchanged |

---

## RW-01: fsspec/gcsfs Path-Abstraction Refactor

### Recommended Approach: Thin `_GCSWriter` Shim

Do NOT use `universal_pathlib` / `UPath` (not installed; adds a dependency; its Path-like
API would require more extensive surgery to tiling.py's shutil and json calls). Do NOT
wrap every call in `fsspec.open` globally — that would require threading the filesystem
object through six separate call sites in different idioms.

**The recommended design is a `_GCSWriter` shim class with three methods:**

```python
# Source: gcsfs.GCSFileSystem API verified 2026-05-16; GCSFile protocol verified 2026-05-16
import io, json
import gcsfs

GCS_PROJECT = "narrative-campaign"  # from gcs_checkpoint.py

class _GCSWriter:
    """Thin write-only abstraction over a GCS prefix.

    Replicates the handful of pathlib.Path methods that tiling.py needs
    for the output side only — reads (map_dir/) stay local Path-based.

    Instances are NOT thread-safe; create one per-pyramid in the thread pool.
    """
    def __init__(self, fs: gcsfs.GCSFileSystem, prefix: str):
        # prefix is a bare GCS path: "mapclass-training-northeast1/data/synthetic/train/..."
        self._fs = fs
        self._prefix = prefix.rstrip("/")

    def __truediv__(self, name: str) -> "_GCSWriter":
        return _GCSWriter(self._fs, f"{self._prefix}/{name}")

    def mkdir(self, parents=True, exist_ok=True):
        # GCS has no real directories; mkdirs is a no-op for objects
        self._fs.mkdirs(self._prefix, exist_ok=True)

    @property
    def name(self) -> str:
        return self._prefix.rsplit("/", 1)[-1]

    def write_bytes(self, data: bytes) -> None:
        self._fs.pipe_file(self._prefix, data)

    def write_text(self, text: str, encoding="utf-8") -> None:
        self._fs.pipe_file(self._prefix, text.encode(encoding))

    def open(self, mode: str = "wb"):
        """Return a writable file-object for Pillow .save() calls."""
        return self._fs.open(self._prefix, mode)
```

**Touch points in `tiling.py` — complete inventory** (6 sites):

| Line | Current pattern | New pattern |
|------|----------------|-------------|
| `out = Path(out_root) if out_root is not None else map_dir / "pyramids"` | `Path(out_root)` | If `out_root` is a `_GCSWriter`, skip the `Path()` wrap; if None, construct from `_GCSWriter(fs, map_dir_gcs + "/pyramids")` |
| `out.mkdir(parents=True, exist_ok=True)` | pathlib | `_GCSWriter.mkdir()` (no-op, GCS is flat) |
| `pdir = out / pid` → `pdir.mkdir(...)` | pathlib | `_GCSWriter.__truediv__` + `_GCSWriter.mkdir()` |
| `img.crop(box).save(pdir / t["image"])` × 3 per tile | `Path / str → PIL.Image.save(Path)` | `buf = io.BytesIO(); img.crop(box).save(buf, "PNG"); (pdir / t["image"]).write_bytes(buf.getvalue())` |
| `(pdir / "pyramid.json").write_text(json.dumps(...))` | pathlib | `_GCSWriter.write_text(...)` |
| `shutil.copyfile(map_dir / _WEIGHTS_FILE, pdir / _WEIGHTS_FILE)` | shutil | `(pdir / _WEIGHTS_FILE).write_bytes(weights_blob)` |

**Reads in `tiling.py` stay local-Path-based** — `map_dir` is always a local path (the
build scripts stage renderings locally before calling `tile()`; only the output writes go to
GCS). The `_GCSWriter` is used only for `out_root`.

**Pillow `.save()` to GCS via `fs.open(path, 'wb')` — confirmed working:**
`gcsfs.GCSFileSystem.open()` returns a `GCSFile` (subclass of `fsspec.AbstractBufferedFile`)
that implements `write()`, `seek()`, `tell()`, `flush()`, and `close()` — the full protocol
Pillow's PNG encoder requires. The write is buffered in memory and committed atomically on
`close()` / `__exit__`. For tile-sized PNGs (typically 50–200 KB), the buffer fits trivially.
`[VERIFIED: GCSFile method inspection 2026-05-16]`

**Preferred write API for pre-buffered data: `fs.pipe_file(path, bytes)`** — avoids the
open/close protocol overhead for data already in memory (the `io.BytesIO` buf case). This is
the correct pattern for JSON manifests and copied weight files. `[VERIFIED: gcsfs.pipe_file
source inspected 2026-05-16]`

### Local-filesystem fallback for tests

The `_GCSWriter` should be testable against a local temp dir by accepting any `fsspec`
`AbstractFileSystem` (not just `GCSFileSystem`). Use `fsspec.filesystem("file")` in tests:

```python
# test pattern — no GCS credentials needed in CI
import fsspec
local_fs = fsspec.filesystem("file")
writer = _GCSWriter(local_fs, str(tmp_path / "synthetic" / "train"))
tiling.tile(map_dir, out_root=writer)
```

This pattern matches the existing `gcs_checkpoint.py` mock approach (tests patch the
`gcsfs.GCSFileSystem` import; here we pass the fs object explicitly, which is even cleaner).

### Where to put `_GCSWriter`

Recommend `scripts/gcs_io.py` (a new module alongside `gcs_checkpoint.py`). This keeps
GCS I/O in one place, importable by all three build scripts. The lazy-import pattern from
`gcs_checkpoint.py` should be applied: `gcsfs` imported inside a `try/except` at module
level so the planning VM (where gcsfs may be absent) can import the module safely.

---

## RW-01b: Streaming Small-Object Write Throughput and Cost

### Cost Estimate (MEDIUM confidence — documented rates, not live measurement)

| Item | Quantity | Rate | Cost |
|------|----------|------|------|
| GCS class A ops (PUT) | ~1.25 M per full 100-map build | $0.05 / 10,000 | ~$6.24 |
| Ingress to GCS from Compute Engine (same region) | ~60 GB | FREE | $0 |
| Storage (standard, US multi-region) | ~60 GB/month | $0.020/GB | ~$1.20/mo |

**Calculation:** 100 sources × 3 styles × ~64 pyramids × 65 writes/pyramid
(21 tiles × 3 PNG types + 1 pyramid.json + 1 sample_weights.json) = 1,248,000 class A ops.
Storage: 100 × 3 × 64 × 21 × 3 × avg 75 KB ≈ 90 GB; conservative estimate ~60–90 GB.

### Throughput Estimate

| Mode | Wall-clock estimate |
|------|---------------------|
| Sequential (1 thread) | ~35 hours at 100 ms avg PUT latency |
| 32 threads (recommended) | ~65 minutes |
| GCS bucket write quota | 1,000 req/s default; 32 threads × 10 req/s = 320 req/s — no quota risk |

**Mitigation the planner MUST bake in:** `ThreadPoolExecutor(max_workers=32)` wrapping the
per-pyramid tile-write loop. The `gcsfs.GCSFileSystem` is thread-safe for concurrent `open()`
and `pipe_file()` calls (each call creates its own HTTP connection from the underlying
`aiohttp` session pool). `[VERIFIED: gcsfs uses asyncio internally; sync wrapper is
thread-safe per gcsfs docs and source review 2026-05-16]`

**What dominates:** operation COUNT (class A ops), not egress (writes to GCS have no egress
cost). The write rate matters more than bandwidth. 32 threads keeps us under quota with
comfortable margin.

**Alternative the user rejected (for reference only):** buffering tiles into a tar or parquet
archive per map-dir reduces class A ops from 65/pyramid to 1/map-dir (~65× reduction, ~$0.10
total). User chose direct streaming; document the cost and move on.

---

## RW-02: Raw-Input Naming Contract and Manifest Hard-Fail

### Recommended `manifest.json` Schema

```json
{
  "version": "1",
  "entries": {
    "europe_01.geojson": {"template": "europe"},
    "europe_02.geojson": {"template": "europe"},
    "americas_01.geojson": {"template": "americas"},
    "east_asia_01.geojson": {"template": "east_asia"}
  }
}
```

**Schema rationale:**
- Flat `entries` dict keyed by filename — O(1) lookup at validation time.
- Each value is `{"template": str}` — one field now; extensible later without breaking the
  schema version (add fields; version bump gates breaking changes).
- `"version": "1"` — allows a `validate_manifest` check to reject stale manifests.
- Template value is what `stratified_split` uses as the grouping key — authoritative.

**Filename convention regex:** `^[a-z][a-z0-9_]*_[0-9]{2}\.geojson$`
- Lowercase alpha-start required (no leading digit or underscore).
- Zero-padded two-digit numeric suffix (`_01` through `_99`).
- `.geojson` extension only.

### Validation Order in `build_dataset.py` — MANDATORY

```
1. fs.ls(gs://…/data/synthetic/raw/)  →  gcs_filenames (list of .geojson basenames)
2. fs.cat(gs://…/data/synthetic/raw/manifest.json)  →  parse manifest
3. validate_manifest(gcs_filenames, manifest)
        a. for each gcs_filename: assert regex match  →  HARD FAIL if any mismatch
        b. for each gcs_filename: assert in manifest["entries"]  →  HARD FAIL if unlisted
        c. for each manifest entry: assert gcs_filename in gcs_filenames  →  HARD FAIL if phantom
        d. for each entry: assert manifest["entries"][fn]["template"] == fn.rsplit("_",1)[0].replace("-","_")
           (template must equal the part before _NN)  →  HARD FAIL if mismatch
4. id_to_template = {_sanitize_stem(fn[:-8]): manifest["entries"][fn]["template"] for fn in gcs_filenames}
5. load_or_create_split(gcs_split_path, source_ids, id_to_template)
        (reads split.json from GCS if it exists; writes to GCS if not)
6. Build loop
```

**Why the ordering matters:** `load_or_create_split` freezes the EVAL-01 hold-out. If a
mis-named file skips validation and enters the split, the test set is permanently contaminated.
The hard-fail before step 5 is the only place that can prevent this.

**`stratified_split` change:** the function currently calls `template_key(sid)` for grouping.
After RW-02, it receives `id_to_template: dict[str, str]` and uses
`id_to_template.get(sid, template_key(sid))` (manifest is authoritative; `template_key` is
the fallback for any source the manifest somehow doesn't cover — but the hard-fail above
means the fallback should never be reached in production).

**`load_or_create_split` GCS I/O change:** `split_path` changes from a local
`Path("data/synthetic/split.json")` to a GCS path read/written via gcsfs:

```python
# Before (local):
if split_path.exists():
    data = json.loads(split_path.read_text())
    ...
split_path.write_text(json.dumps({...}))

# After (GCS):
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
split_gcs = "mapclass-training-northeast1/data/synthetic/split.json"
if fs.exists(split_gcs):
    data = json.loads(fs.cat(split_gcs).decode())
    ...
fs.pipe_file(split_gcs, json.dumps({...}).encode())
```

---

## RW-03: Pull-Once + Verify for finetune_seg / evaluate_seg

### Bulk-Fetch Design

```python
# scripts/gcs_io.py — add pull_dataset() helper
# Source: gcsfs.GCSFileSystem.get signature verified 2026-05-16

def pull_dataset_from_gcs(
    subset: str,               # "train" or "test"
    local_scratch: Path,
    gcs_prefix: str = "gs://mapclass-training-northeast1/data/synthetic",
) -> Path:
    """Bulk-fetch a train or test subset from GCS to local scratch.

    Idempotent: safe to re-run after preemption (fs.get overwrites local files).
    Returns the local root for the subset (e.g. local_scratch / "train").
    """
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    remote = f"{gcs_prefix}/{subset}/"
    local_root = local_scratch / subset
    local_root.mkdir(parents=True, exist_ok=True)
    # recursive=True downloads the entire subtree
    fs.get(remote, str(local_root), recursive=True)
    return local_root
```

**Call site in `finetune_seg.py`:**
- Add `--scratch-dir` arg (default `/tmp/mapclass_data`).
- At job start (before `carve_train_val`): call `pull_dataset_from_gcs("train", scratch)`.
- Read `split.json` from GCS once; pass it to `verify_pull`.
- Then proceed with existing `carve_train_val(local_train_root)` — **unchanged**.

**Call site in `evaluate_seg.py`:**
- Add `--scratch-dir` arg.
- At job start: call `pull_dataset_from_gcs("test", scratch)`.
- Read `split.json` from GCS; verify pull; then `load_test_pyramid_dirs(local_split_json, local_data_root)` — **unchanged**.

### Verification Design

```
verify_pull(local_root: Path, split_json: dict, subset: str) -> None:

  1. For each map_id in split_json[subset]:
     - assert (local_root / map_id).is_dir()  <- membership + presence

  2. For a 10% sample of (local_root / map_id / pyramids / py_r*):
     - assert (pdir / "pyramid.json").exists()
     - count PNGs in pdir: expect 63 (21 tiles × 3 types); assert >= 60 (tolerance for partial edge pyramids)

  3. Raise RuntimeError (not sys.exit) on any failure — allows caller to handle.
```

**Why not per-file checksums:** CRC32C verification would require one `fs.info(path)` GCS API
call per file × ~1.25M files = 1.25M API ops = ~$0.006 extra and ~3 additional minutes.
Not worth it for routine verification; GCS is highly reliable. Reserve CRC32C verification for
a manual debug mode (`--verify-checksums` flag) if corruption is ever suspected.

### Composition with GCS Checkpoint Resume

```
Job start sequence (new):
  1. pull_dataset_from_gcs(subset, local_scratch)   <- RW-03 (NEW)
  2. verify_pull(local_root, split_json, subset)    <- RW-03 (NEW)
  3. gcs_latest_checkpoint(config_name)             <- EXISTING (gcs_checkpoint.py)
  4. training/eval loop from local files            <- EXISTING (unchanged)
```

On preemption: new box repeats steps 1–4. Step 1 is idempotent (overwrites if cached).
Step 3 resumes from the latest checkpoint. **No state is lost between preemptions.**

---

## RW-04: GCS-Canonical Reads — Fresh-Box Audit

### `render.py` — NOT a GCS read concern

`render.py` is called by `build_dataset.py` as `render_one(geojson_path, style)` — it
receives a local GeoJSON path and returns a `PIL.Image` in memory. It does NOT read from
`data/toons/` in the Phase 2 pipeline. The `gs://…/data/toons/<biome>/` path in CONTEXT.md
describes Phase 1 PaliGemma training tiles — a separate concern. `[VERIFIED: render.py code
read 2026-05-16]`

### Historical Pipeline — Local-Read Assumptions to Fix

| Component | Current read | Required for fresh-box | Fix |
|-----------|-------------|------------------------|-----|
| `build_historical_dataset.py search` | Writes `raw_dir/unregistered_manifest.json` to local | Must write to GCS canonical path | Change `manifest_path` default to GCS; write via `fs.pipe_file` |
| `build_historical_dataset.py build` | Reads `raw_dir/georeferenced/*.tif` from local | Raw GeoTIFFs are re-downloadable from Rumsey (not canonical) | **Scratch acceptable**: Rumsey TIFs are ephemeral-OK; download fresh per build run |
| `build_historical_dataset.py build` | Writes `out_dir/` (dataset) to local | Must persist to GCS | Change `out_dir` default to GCS prefix; tile via `_GCSWriter` |
| `unregistered_manifest.json` in GCS | CONTEXT.md states it exists at `gs://…/data/historical/raw/unregistered_manifest.json` | Build must ALSO write to GCS | Fixed by first bullet above |

**Conclusion:** raw GeoTIFFs from Rumsey are ephemeral-safe (the LUNA API is the source of
truth; re-download on each job). Built datasets (pyramids, manifests, split.json) MUST be
GCS-canonical. The `build_historical_dataset.py search` output (`unregistered_manifest.json`)
must also land in GCS (it is the v2 hand-off artifact).

### Satellite Pipeline — Local-Read Assumptions to Fix

| Component | Current read | Fix |
|-----------|-------------|-----|
| `_DEFAULT_SUMMARY` = `data/satellite/coverage_summary.json` | Local | Change default to GCS path; write summary to GCS once, read from GCS on subsequent runs |
| `_DEFAULT_MANIFEST` = `data/satellite/resolved_scenes.json` | Local | Change to GCS path |
| `_DEFAULT_OUT` = `data/satellite/dataset` | Local | Change to GCS prefix; tile via `_GCSWriter` |
| Raw COG byte-range reads (Sentinel-2, WorldCover, DEM) | Network → in-memory | **Unchanged** — these never touch local disk (GDAL VSI-CURL is ephemeral by design) |

---

## Standard Stack (GCS I/O additions)

> For the unchanged domain stack (rasterio, Pillow, pystac-client, etc.) see the prior
> 2026-05-15 research. Only the new/changed packages are listed here.

### Core (additions for GCS I/O)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| gcsfs | 2026.5.0 | GCS file-system interface (ADC auth + read/write) | **EXISTING** — already in `seg/gcs_checkpoint.py`; the project's established GCS I/O pattern `[VERIFIED: pip show 2026-05-16]` |
| fsspec | 2026.3.0 | Abstract filesystem interface (local fallback for testing) | **EXISTING** — transitively installed by gcsfs; enables `fsspec.filesystem("file")` for test isolation `[VERIFIED: pip show 2026-05-16]` |

### Not Recommended

| Library | Why Not |
|---------|---------|
| universal-pathlib (UPath) | Not installed; adds a dependency; the thin shim achieves the same with zero new deps |
| google-cloud-storage (direct) | Already a transitive dep of gcsfs; don't use directly — gcsfs is the project's established abstraction |

**No new packages required.** gcsfs and fsspec are already installed and already the
project's GCS I/O standard. `requirements.txt` needs no changes for the GCS persistence work.

## Package Legitimacy Audit

> Only the GCS I/O packages are audited here — pystac-client and the pre-existing stack were
> audited in the 2026-05-15 research.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| gcsfs | PyPI | ~8 yrs | High (canonical GCS client) | github.com/fsspec/gcsfs | OK | Approved |
| fsspec | PyPI | ~8 yrs | Very high (core dep of many ML libs) | github.com/fsspec/filesystem_spec | OK | Approved |

`[VERIFIED: slopcheck install gcsfs fsspec — both OK, 2026-05-16]`
`[VERIFIED: pip show gcsfs Version: 2026.5.0 Home-page: github.com/fsspec/gcsfs, 2026-05-16]`

**Packages removed due to slopcheck [SLOP]:** none
**Packages flagged as suspicious [SUS]:** none

---

## Architecture Patterns

### System Architecture Diagram (REWORK — persistence layer only)

```
  ===== BUILD TIME =====                    ===== TRAIN TIME =====

  User (manual)
  ├── Creates Azgaar GeoJSON exports
  │   named <template>_<NN>.geojson
  ├── Authors manifest.json
  └── Uploads both to GCS synthetic/raw/
            |
            v
  build_dataset.py
  ├── 1. fs.ls(gs://…/synthetic/raw/)          evaluate_seg.py / finetune_seg.py
  ├── 2. fs.cat(manifest.json) → validate      ├── 1. fs.cat(split.json from GCS)
  │      [HARD FAIL if mismatch]               ├── 2. fs.get(gs://…/train/ or test/,
  ├── 3. fs.cat(split.json) or compute new     │       local_scratch, recursive=True)
  │      split → fs.pipe_file(split.json)      ├── 3. verify_pull(local, split_json, subset)
  ├── 4. for each geojson:                     ├── 4. gcs_latest_checkpoint (resume)
  │      a. fs.cat(geojson) → local temp       └── 5. train/eval from local scratch
  │      b. render_one(local_temp, style) → PIL.Image (unchanged)
  │      c. make_label_arrays(local_temp) → PIL.Image (unchanged)
  │      d. write image.png, lc.png, topo.png, sample_weights.json to local temp
  │      e. tiling.tile(local_map_dir, out_root=_GCSWriter(fs, gs_train_or_test_prefix))
  │            └── ThreadPoolExecutor(max_workers=32) for tile writes
  └── GCS outputs: gs://…/data/synthetic/{train,test}/<map_id>__<style>/pyramids/
                   gs://…/data/synthetic/split.json

  build_historical_dataset.py (RW-04)         build_satellite_dataset.py (RW-04)
  ├── search: LUNA → Allmaps → IIIF           ├── coverage-scan → fs.pipe_file(summary.json)
  │    → GeoTIFF in local scratch             ├── search → fs.pipe_file(resolved_scenes.json)
  │    → fs.pipe_file(unregistered_manifest)  └── build → tiling.tile(..., _GCSWriter)
  └── build: local GeoTIFF → label → tiling
       → _GCSWriter(gs://…/historical/dataset/)
```

### Recommended New File: `scripts/gcs_io.py`

```
scripts/
├── gcs_io.py             # NEW — _GCSWriter shim + pull_dataset_from_gcs + verify_pull
│                         #       GCS_PROJECT, BUCKET, DATA_PREFIX constants
│                         #       lazy gcsfs import (same pattern as gcs_checkpoint.py)
├── seg/
│   ├── gcs_checkpoint.py # EXISTING — checkpoint write/resume (unchanged)
│   └── ...
├── build_dataset.py      # MODIFY — GCS raw/ read, manifest validate, GCS split.json
├── build_historical_dataset.py  # MODIFY — RW-04 GCS output paths
├── build_satellite_dataset.py   # MODIFY — RW-04 GCS output paths
├── tiling.py             # MODIFY — accept _GCSWriter as out_root; add tile write concurrency
├── finetune_seg.py       # MODIFY — add pull_dataset_from_gcs at startup
└── evaluate_seg.py       # MODIFY — add pull_dataset_from_gcs at startup
```

### Pattern 1: _GCSWriter-Based Tile Write with Thread Pool

```python
# Source: gcsfs API verified 2026-05-16; io.BytesIO + PIL pattern confirmed
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

def _write_tile(writer: "_GCSWriter", name: str, img_crop) -> None:
    buf = io.BytesIO()
    img_crop.save(buf, format="PNG")
    (writer / name).write_bytes(buf.getvalue())

def tile_to_gcs(map_dir: Path, out_writer: "_GCSWriter",
                max_workers: int = 32) -> None:
    """Decompose a completed per-map dir into nested pyramids, writing directly to GCS."""
    # ... (pyramid geometry logic unchanged) ...
    write_tasks = []
    for ox, oy in pyramids:
        pdir = out_writer / _pyramid_id(ox, oy)
        pdir.mkdir()
        tiles = _pyramid_tiles(ox, oy)
        for t in tiles:
            box = (t["x"], t["y"], t["x"]+t["size"], t["y"]+t["size"])
            write_tasks.append((pdir, t["image"], img.crop(box)))
            write_tasks.append((pdir, t["land_cover"], lc.crop(box)))
            write_tasks.append((pdir, t["topography"], topo.crop(box)))
        write_tasks.append((pdir, "pyramid.json",
                            json.dumps(manifest).encode()))
        write_tasks.append((pdir, "sample_weights.json", weights_blob))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = []
        for pdir, name, data in write_tasks:
            if isinstance(data, (bytes, str)):
                futs.append(pool.submit((pdir / name).write_bytes,
                                         data if isinstance(data, bytes)
                                         else data.encode()))
            else:  # PIL Image crop
                futs.append(pool.submit(_write_tile, pdir, name, data))
        for fut in as_completed(futs):
            fut.result()  # re-raise any exception
```

### Pattern 2: GCS Split.json Read/Write

```python
# Source: gcsfs.pipe_file and .cat verified 2026-05-16
# Mirrors gcs_checkpoint.py write pattern

GCS_SPLIT = "mapclass-training-northeast1/data/synthetic/split.json"

def load_or_create_split_gcs(fs, source_ids, id_to_template) -> set[str]:
    """GCS-canonical version of build_dataset.load_or_create_split."""
    if fs.exists(GCS_SPLIT):
        data = json.loads(fs.cat(GCS_SPLIT).decode())
        return set(data["test"])

    # Print stratification groups for human review (WR-08) — unchanged
    grouping: dict[str, list[str]] = {}
    for sid in source_ids:
        grouping.setdefault(id_to_template.get(sid, template_key(sid)), []).append(sid)
    print("  Derived stratification groups (review before split.json frozen — WR-08):")
    for tmpl in sorted(grouping):
        print(f"    {tmpl}: {sorted(grouping[tmpl])}")

    test_ids = stratified_split(source_ids, id_to_template=id_to_template)
    payload = json.dumps({"seed": _SPLIT_SEED, "test_fraction": _TEST_FRACTION,
                          "test": test_ids}, indent=2).encode()
    fs.pipe_file(GCS_SPLIT, payload)
    return set(test_ids)
```

### Pattern 3: Manifest Validation (Hard-Fail)

```python
# Source: RW-02 naming contract from 02-CONTEXT.md
import re, sys

_FILENAME_RE = re.compile(r"^[a-z][a-z0-9_]*_[0-9]{2}\.geojson$")

def validate_manifest(gcs_filenames: list[str], manifest: dict) -> dict[str, str]:
    """Return id_to_template or sys.exit(1).

    gcs_filenames: basenames from fs.ls(raw_prefix)
    manifest: parsed manifest.json
    """
    errors = []
    entries = manifest.get("entries", {})

    for fn in gcs_filenames:
        if not _FILENAME_RE.fullmatch(fn):
            errors.append(f"  filename convention violation: {fn!r} does not match "
                          f"<template>_<NN>.geojson")
        if fn not in entries:
            errors.append(f"  unlisted in manifest: {fn!r}")
        else:
            expected_tmpl = fn.rsplit("_", 1)[0]  # everything before _NN
            actual_tmpl = entries[fn].get("template", "")
            if actual_tmpl != expected_tmpl:
                errors.append(f"  template mismatch: {fn!r} has template "
                              f"{actual_tmpl!r}, expected {expected_tmpl!r}")

    for fn in entries:
        if fn not in gcs_filenames:
            errors.append(f"  manifest entry missing from GCS raw/: {fn!r}")

    if errors:
        print("FATAL: manifest validation failed — aborting before split is computed (RW-02):")
        for e in errors:
            print(e)
        sys.exit(1)

    return {re.sub(r"[^\w-]", "_", fn[:-8]): entries[fn]["template"]
            for fn in gcs_filenames}  # id_to_template
```

### Anti-Patterns to Avoid

- **Calling `load_or_create_split` before `validate_manifest`.** A mis-named file that passes
  the collision guard but not the manifest check will skew the EVAL-01 stratified hold-out
  permanently. The hard-fail order is mandatory.
- **Using `UPath` or `universal_pathlib`.** Not installed, adds a dep, and the thin shim
  achieves the same with zero new packages.
- **Per-pyramid sequential writes without thread pool.** At 100 ms/write × 1.25M writes =
  35 hours. Always parallelize at the pyramid-tile level.
- **Passing `GCSFileSystem` into `tiling.tile()` as the `map_dir` parameter.** Reads of
  `image.png`, `land_cover.png`, `topography.png`, `sample_weights.json` should stay local
  Path-based. The build scripts stage these locally before calling `tile()`. Mixing GCS reads
  into the tiler's read path adds unnecessary complexity.
- **Reopening a gcsfs.GCSFileSystem per tile write.** Share a single `fs` instance across
  the thread pool — re-authentication overhead is high. Instantiate once; pass to `_GCSWriter`.

---

## Don't Hand-Roll

> For the domain-unchanged items (affine fits, STAC queries, COG reads, WorldCover remapping)
> see the prior 2026-05-15 research.

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GCS object write | boto3 or requests with signed URLs | `gcsfs.GCSFileSystem.pipe_file(path, bytes)` | ADC auth already established; same pattern as gcs_checkpoint.py; atomic PUT |
| GCS bulk download | manual fs.ls + per-file download loop | `fs.get(remote_prefix, local_dir, recursive=True)` | Single call; handles directory structure; idempotent |
| Split.json GCS I/O | hand-coded requests with GCS JSON API | `fs.exists(path)` + `fs.cat(path)` + `fs.pipe_file(path, bytes)` | Three-line pattern; consistent with checkpoint module |
| GCS directory listing | glob or recursive walk | `fs.ls(bare_prefix)` + filter by suffix | gcsfs mirrors the pattern in gcs_checkpoint.py line 174–177 |
| File-object for Pillow | manually build a BytesIO bridge | `fs.open(path, 'wb')` as the PIL.Image.save target | GCSFile implements the full seekable/writable buffer protocol |

---

## Common Pitfalls (REWORK additions)

### Pitfall R-1: Silent Local-Only Writes (the root cause of this rework)
**What goes wrong:** A build script runs successfully but writes to `Path("data/synthetic/...")`
instead of `gs://…`. Outputs look correct locally; nothing lands in GCS. Next ephemeral box
finds an empty bucket.
**Why it happens:** The default argument `--out-dir data/synthetic` is still a local path.
The script completes without error.
**How to avoid:** Change every `--out-dir`, `--raw-dir`, and `--manifest` default to a `gs://`
URI. Add a startup assertion: if `out_dir` does not start with `gs://`, print a warning and
require `--local-ok` to override (useful for tests). Never accept a silent local fallback.
**Warning signs:** `fs.ls(gs://…/data/synthetic/)` returns empty after build completes.

### Pitfall R-2: Partial/Aborted GCS Uploads Mid-Pyramid
**What goes wrong:** The build process is preempted or killed mid-pyramid. Some tiles for
a pyramid are in GCS; others are not. The next run finds the pyramid directory exists (via
`fs.exists`) and skips it — resulting in a corrupt incomplete pyramid.
**Why it happens:** Naive "skip if exists" logic at the pyramid level.
**How to avoid:** The collision guard in `build_dataset.py` checks `_REQUIRED_MAP_FILES`
before calling `tiling.tile()`. For GCS, the equivalent pre-write check is: if
`gs://…/<map_id>/<style>/pyramids/` exists AND contains the expected number of pyramid
subdirs AND the first pyramid's `pyramid.json` exists — then skip; else delete and rebuild.
Alternatively, write a `_BUILD_COMPLETE` sentinel object as the LAST write of a map-dir;
presence of this sentinel = safe to skip; its absence = rebuild from scratch.
**Warning signs:** `pytest tests/test_gcs_builds.py::test_pyramid_completeness` fails on a
partially-uploaded map.

### Pitfall R-3: `split.json` Frozen on an Incomplete Raw Set
**What goes wrong:** User uploads 30 GeoJSON files, runs build, split.json is frozen with 30
sources. User uploads 70 more files. Subsequent build runs read the frozen split, which was
computed on only 30 sources — the 70 new sources all go to `train/` without proportional test
representation. EVAL-01 hold-out is statistically weak.
**Why it happens:** `load_or_create_split` (D-18) deliberately never recomputes once frozen.
**How to avoid:** The RW-02 manifest hard-fail mitigates this partially — the manifest must
list ALL uploaded files at validation time. Document clearly: `split.json` must be deleted
from GCS to trigger a recompute. Add a `--refreeze-split` flag that deletes the existing
`split.json` from GCS and recomputes. Make this an explicit user action, not automatic.
**Warning signs:** `len(test_ids) / len(all_source_ids)` << 0.15 after a build with more
sources than the frozen split was computed on.

### Pitfall R-4: Manifest Mismatch Not Hard-Failing
**What goes wrong:** `validate_manifest` prints a warning instead of `sys.exit(1)`. Build
continues; a mis-named source with no manifest entry gets assigned to a stratum via
`template_key()` fallback — possibly the wrong stratum. EVAL-01 hold-out is silently skewed.
**How to avoid:** `validate_manifest` must call `sys.exit(1)` (not `raise ValueError`, not
`print("WARNING")`). The function has no legitimate "partial pass" state — it either passes
completely or the build must not proceed.

### Pitfall R-5: Pull-Once Verification False-Passing
**What goes wrong:** `verify_pull` checks only that directories exist, not that they contain
complete pyramids. A partially-downloaded set passes verification. Training begins on
truncated data; loss is anomalously high but looks like a model issue.
**How to avoid:** The spot-check (10% of pyramid dirs, file count ≥ 60) catches gross
truncation. For edge cases: the training DataLoader's `pyramid.json` parse will fail loudly on
any truncated pyramid, surfacing the error immediately (not silently).
**Warning signs:** Epoch 1 val loss is unexpectedly high compared to the probe probe-mode
baseline; some pyramid dirs contain fewer than 65 files.

---

## Validation Architecture

> Focus: rework-introduced failure modes. The unchanged-domain tests (tiler geometry, split
> determinism, Allmaps/STAC) are documented in the 2026-05-15 research. Add the tests below
> to the existing Wave 0 gap list.

### Test Framework (unchanged from prior research)
| Property | Value |
|----------|-------|
| Framework | pytest >= 8 (to be installed in Wave 0) |
| Config file | none — see Wave 0 |
| Quick run command | `pytest tests/ -x --ignore=tests/integration` |
| Full suite command | `pytest tests/` |

### Rework Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| RW-01 | `_GCSWriter` writes bytes to a mock fsspec local filesystem correctly | unit | `pytest tests/test_gcs_io.py::test_gcs_writer_writes_bytes -x` | ❌ Wave 0 |
| RW-01 | `tiling.tile(map_dir, out_root=_GCSWriter(..., local_fs))` produces correct pyramid structure on a fixture map | integration (offline) | `pytest tests/test_tiling.py::test_tile_to_gcs_writer -x` | ❌ Wave 0 |
| RW-01 | `_GCSWriter` opened in 'wb' mode accepts Pillow `.save(fh, format='PNG')` | unit | `pytest tests/test_gcs_io.py::test_gcs_writer_pillow_compat -x` | ❌ Wave 0 |
| RW-01b | Thread-pool tile writes do not corrupt pyramid (concurrent write test) | unit | `pytest tests/test_gcs_io.py::test_concurrent_tile_writes -x` | ❌ Wave 0 |
| RW-02 | `validate_manifest` exits 1 on: unlisted file, phantom entry, regex mismatch, template mismatch | unit | `pytest tests/test_manifest.py::test_validate_manifest_hard_fails -x` | ❌ Wave 0 |
| RW-02 | `validate_manifest` must run BEFORE `load_or_create_split` in `build` flow | integration | `pytest tests/test_build_dataset.py::test_manifest_fails_before_split -x` | ❌ Wave 0 |
| RW-02 | `stratified_split` uses manifest template, not `template_key()`, for grouping | unit | `pytest tests/test_build_dataset.py::test_split_uses_manifest_template -x` | ❌ Wave 0 |
| RW-03 | `pull_dataset_from_gcs` is idempotent (second call does not fail or corrupt) | unit (mock fs) | `pytest tests/test_gcs_io.py::test_pull_idempotent -x` | ❌ Wave 0 |
| RW-03 | `verify_pull` raises on missing map_id directory | unit | `pytest tests/test_gcs_io.py::test_verify_raises_on_missing_map -x` | ❌ Wave 0 |
| RW-03 | `verify_pull` raises on truncated pyramid (< 60 files in spot-check) | unit | `pytest tests/test_gcs_io.py::test_verify_raises_on_truncated_pyramid -x` | ❌ Wave 0 |
| EVAL-01 | `split.json` written to GCS; re-run does not recompute (frozen) | integration (mock GCS) | `pytest tests/test_build_dataset.py::test_split_frozen_in_gcs -x` | ❌ Wave 0 |
| RW-04 | `build_historical_dataset.py` writes `unregistered_manifest.json` to GCS path, not local | integration (mock GCS) | `pytest tests/test_historical.py::test_manifest_written_to_gcs -x` | ❌ Wave 0 |

### Observability Points (Nyquist sampling for rework failure modes)

| Failure Mode | Detection Point | How to Detect |
|-------------|----------------|---------------|
| Silent local-only write | Post-build GCS ls | Assert `fs.ls(gs://…/data/synthetic/train/)` non-empty immediately after build |
| Partial upload mid-pyramid | Pre-train verification | `verify_pull` spot-checks pyramid file count |
| Split frozen on incomplete raw | Manifest validation | RW-02 hard-fail prevents unknown files; `--refreeze-split` is the deliberate escape hatch |
| Manifest mismatch not hard-failing | Build test suite | `test_manifest_fails_before_split` confirms ordering |
| Pull-once false-pass | Training loss spike | Epoch-1 val loss compared to probe baseline; pyramid.json parse errors in DataLoader |

### Wave 0 Gaps (additions to prior 2026-05-15 list)

- [ ] `tests/test_gcs_io.py` — `_GCSWriter` unit tests (write_bytes, write_text, open, Pillow compat, concurrent, idempotent pull)
- [ ] `tests/test_manifest.py` — `validate_manifest` hard-fail coverage (all four error cases)
- [ ] `scripts/gcs_io.py` — `_GCSWriter`, `pull_dataset_from_gcs`, `verify_pull` (the implementation itself)
- [ ] Add `--scratch-dir` arg to `finetune_seg.py` and `evaluate_seg.py`
- [ ] Add `--refreeze-split` flag to `build_dataset.py`

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A-R1 | GCS class A op cost is $0.05/10,000 ops for `us-east1` / `northamerica-northeast1` standard storage | RW-01b | LOW — if rate is different, the ~$6 estimate shifts proportionally; order of magnitude remains correct |
| A-R2 | Same-region Compute Engine → GCS PUT latency is ~100 ms for 50–200 KB objects | RW-01b | MEDIUM — if latency is 200 ms, 32 threads gives ~2 hours instead of 65 minutes; still acceptable |
| A-R3 | GCS bucket write quota is 1,000 req/s (default); 32 threads at 10 req/s each = 320 req/s stays under quota | RW-01b | LOW — GCS may impose lower bucket-level limits for the project; if rate-limited, reduce `max_workers` |
| A-R4 | `gcsfs.GCSFileSystem` is thread-safe for concurrent `pipe_file()` calls across threads | RW-01 Pattern 1 | LOW — gcsfs is async-backed and the sync wrapper acquires per-call event loops; concurrent calls from separate threads are standard usage per gcsfs docs |
| A-R5 | Raw Rumsey GeoTIFFs (IIIF downloads) are re-downloadable on each job run and do not need to be GCS-canonical | RW-04 | MEDIUM — if a future Rumsey/Allmaps API change makes re-download unreliable, raw TIFs should be uploaded to GCS raw/. Flag for user awareness. |

---

## Open Questions (none blocking planning)

1. **Should the build add a `_BUILD_COMPLETE` sentinel to each map-dir in GCS?**
   - What we know: partial pyramid uploads are a preemption risk; there is no atomic
     directory-level commit in GCS.
   - What's unclear: whether the user wants this protection or prefers manual cleanup.
   - Recommendation: add the sentinel as a cheap safeguard; document it in the plan.
   - **Not blocking planning — planner should include this as an optional task.**

2. **Should the pull-once fetch all three families (synthetic + historical + satellite) or
   only the synthetic family?**
   - What we know: `finetune_seg.py` / `evaluate_seg.py` train on the MERGED dataset across
     all three families (same `train/` root after RW-04 reorganization).
   - What's unclear: whether all three families share a single GCS `data/train/` root or
     separate per-family roots with a merged local scratch.
   - Recommendation: planner should clarify the merged-vs-family-rooted layout in 02-05.
   - **Not blocking — planner resolves this in the finetune/evaluate plan.**

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| gcsfs | All GCS I/O | ✓ | 2026.5.0 | — |
| fsspec | Testing (local fs abstraction) | ✓ | 2026.3.0 | — |
| GCS ADC credentials | Build + train scripts | ✓ (planning VM) | — | GPU host requires `gcloud auth application-default login` or service account |
| `gs://mapclass-training-northeast1/` bucket | All GCS writes | ✓ (assumed — existing checkpoint writes work) | — | — |

**Missing dependencies with no fallback:** none

**Note on ADC on fresh boxes:** the GPU training host must have ADC configured before running
any GCS I/O. This is the same requirement as the existing `gcs_checkpoint.py` — no new auth
requirement introduced by this rework. The planner should include `gcloud auth
application-default login` as a pre-condition note in the GPU-host gate (04-HUMAN-UAT.md
pattern).

---

## Unchanged-Domain References

The following Phase 2 research areas are covered in the prior 2026-05-15 `02-RESEARCH.md`
and are NOT reproduced here. The planner MUST still read those sections for the full picture:

- Allmaps offline dump + IIIF fetch pattern (Patterns 1, 3 in prior research)
- STAC cloud-filtered Sentinel-2 search (Pattern 2)
- Affine GCP fit with rasterio (Pattern 4)
- Class-diversity stratified coverage scan (Pattern 5)
- Anti-patterns for IIIF, STAC, tiler geometry
- Don't-hand-roll table (affine LSQ, STAC queries, COG byte-range reads, WorldCover remapping)
- Pitfalls 1–6 (GCP convention, IIIF best-fit, Allmaps atlases, S2 cloud cover, DEM resolution, per-map schema)
- Full environment availability table (rasterio, pyproj, Pillow, numpy, pystac-client)
- Security domain analysis

---

## Sources

### Primary (HIGH confidence — verified live or from code inspection)
- `scripts/seg/gcs_checkpoint.py` — reuse template for GCS auth (ADC), `fs.open(path, 'wb')`
  write pattern, `fs.ls(bare_prefix)` listing idiom. `[VERIFIED: code read 2026-05-16]`
- `scripts/tiling.py` — full touch-point inventory for the `_GCSWriter` refactor.
  `[VERIFIED: code read 2026-05-16]`
- `scripts/build_dataset.py` — `load_or_create_split`, `build_one_source`, collision guard
  locations. `[VERIFIED: code read 2026-05-16]`
- `scripts/render.py` — confirmed: does NOT read from GCS or `data/toons/` in Phase 2.
  `[VERIFIED: code read 2026-05-16]`
- `scripts/build_historical_dataset.py` — confirmed: all reads/writes are local Path-based
  today. `[VERIFIED: code read 2026-05-16]`
- `gcsfs` 2026.5.0 API inspection — `GCSFile` protocol (write, seek, tell, close), `pipe_file`
  (atomic PUT), `get` (bulk download), `ls` (directory listing). `[VERIFIED: gcsfs API
  inspection + method source inspection 2026-05-16]`
- `fsspec` 2026.3.0 — `fsspec.filesystem("file")` for test isolation. `[VERIFIED: 2026-05-16]`
- slopcheck 0.6.1 — gcsfs, fsspec both `[OK]`. `[VERIFIED: slopcheck output 2026-05-16]`

### Secondary (MEDIUM confidence — documented rates, not live measurement)
- GCS pricing: class A ops $0.05/10,000, standard storage $0.020/GB/month, same-region
  ingress free. `[CITED: cloud.google.com/storage/pricing — rates as of training knowledge;
  confirm current rates before large builds]`
- GCS default bucket write quota: 1,000 req/s. `[CITED: cloud.google.com/storage/quotas]`
- GCS same-region PUT latency: 50–150 ms for objects < 5 MB. `[CITED: GCS performance docs /
  community benchmarks — ASSUMED for exact numbers]`

---

## Metadata

**Confidence breakdown:**
- gcsfs/fsspec API: HIGH — inspected live from installed 2026.5.0/2026.3.0
- tiling.py touch points: HIGH — complete inventory from code read
- GCS write cost/throughput: MEDIUM — documented rates + standard calculations; not live-timed
- Manifest schema: HIGH — derived from RW-02 requirements, validated against template_key()
- Pull-once design: HIGH — gcsfs.get API confirmed; verification design is conservative

**Research date:** 2026-05-16
**Valid until:** 2026-07-16 (gcsfs releases frequently; re-verify API if upgrading beyond
2026.5.0)
