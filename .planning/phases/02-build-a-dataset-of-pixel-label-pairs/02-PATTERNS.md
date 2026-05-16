# Phase 02: build-a-dataset-of-pixel-label-pairs — Pattern Map (REWORK)

**Mapped:** 2026-05-16
**Files analyzed:** 7 (1 new, 6 modified)
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/gcs_io.py` | utility (GCS I/O shim) | file-I/O (streaming write + bulk read) | `scripts/seg/gcs_checkpoint.py` | exact — same bucket, same ADC auth, same gcsfs version |
| `scripts/tiling.py` | utility (pyramid tiler) | file-I/O (write) | `scripts/tiling.py` (self — modify) | self-modify — 6 call sites identified |
| `scripts/build_dataset.py` | service (orchestrator) | CRUD + file-I/O | `scripts/build_dataset.py` (self — modify) | self-modify — collision-guard + split-freeze logic preserved |
| `scripts/build_historical_dataset.py` | service (orchestrator) | CRUD + file-I/O | `scripts/build_dataset.py` | role-match — identical tiling.tile() call site pattern |
| `scripts/build_satellite_dataset.py` | service (orchestrator) | CRUD + file-I/O | `scripts/build_historical_dataset.py` | role-match — same ThreadPoolExecutor + tiling.tile() pattern |
| `scripts/finetune_seg.py` | service (training loop) | request-response + file-I/O | `scripts/finetune_seg.py` (self) + `scripts/seg/gcs_checkpoint.py` | self-modify; GCS resume composing pattern from gcs_checkpoint.py |
| `scripts/evaluate_seg.py` | service (eval harness) | request-response + file-I/O | `scripts/evaluate_seg.py` (self) + `scripts/seg/gcs_checkpoint.py` | self-modify; GCS fs pattern from gcs_checkpoint.py |

---

## Pattern Assignments

### `scripts/gcs_io.py` (NEW — utility, file-I/O streaming)

**Analog:** `scripts/seg/gcs_checkpoint.py`

This is the single most important analog. `gcs_io.py` reuses the ADC auth, lazy import, `GCS_PROJECT` constant, `fs.open()`, and `fs.pipe_file()` patterns from `gcs_checkpoint.py` verbatim.

**Lazy-import pattern** (`gcs_checkpoint.py` lines 36–39):
```python
try:
    import gcsfs  # type: ignore[import]
except ModuleNotFoundError:
    gcsfs = None  # type: ignore[assignment]
```
Apply identically in `gcs_io.py`. Keeps the module importable on the planning VM (no gcsfs). Tests patch `gcs_io.gcsfs` the same way tests patch `seg.gcs_checkpoint.gcsfs`.

**GCS project constant** (`gcs_checkpoint.py` line 45):
```python
GCS_PROJECT = "narrative-campaign"
```
Copy verbatim into `gcs_io.py`. Add `BUCKET = "mapclass-training-northeast1"` and `DATA_PREFIX = "mapclass-training-northeast1/data"` alongside it.

**fs instantiation pattern** (`gcs_checkpoint.py` lines 131, 171):
```python
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
```
`_GCSWriter.__init__` receives an already-instantiated `fs` (passed in, not created inside the shim). Build scripts instantiate once and pass to all `_GCSWriter` instances — do NOT re-instantiate per-tile (mirrors `gcs_checkpoint.py`'s single-instance-per-function discipline).

**Atomic write pattern** (`gcs_checkpoint.py` lines 127–133):
```python
buf = io.BytesIO()
torch.save(state, buf)
buf.seek(0)
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
with fs.open(path, "wb") as fh:
    fh.write(buf.read())
```
`_GCSWriter.open("wb")` delegates to `self._fs.open(self._prefix, mode)` — same `fs.open(path, "wb")` idiom. `_GCSWriter.write_bytes(data)` delegates to `self._fs.pipe_file(self._prefix, data)` for pre-buffered blobs.

**fs.ls + bare-prefix pattern** (`gcs_checkpoint.py` lines 174–179):
```python
bare_prefix = f"mapclass-training-northeast1/models/{config_name}/"
try:
    files = fs.ls(bare_prefix)
except FileNotFoundError:
    return 0, None
```
`pull_dataset_from_gcs` and `verify_pull` use `fs.ls(bare_prefix)` (no `gs://`) following this exact idiom. All bare GCS paths in `gcs_io.py` strip the `gs://` prefix just as `gcs_checkpoint.py` does.

**Test mock pattern** (`tests/test_seg_gcs.py` lines 76–121):
```python
class _MockGCSFileSystem:
    _store: dict[str, bytes] = {}
    def open(self, path, mode="rb"): ...
    def ls(self, prefix): ...

def _patch_gcsfs():
    return mock.patch("seg.gcs_checkpoint.gcsfs", _MockGCSModule)
```
`tests/test_gcs_io.py` must use the identical pattern: a `_MockGCSFileSystem` with `pipe_file`, `open`, `ls`, `exists`, `cat`, and `get` methods; patched via `mock.patch("gcs_io.gcsfs", _MockGCSModule)`.

---

### `scripts/tiling.py` (MODIFY — utility, file-I/O write)

**Analog:** `scripts/tiling.py` (self-modification)

**All 6 current write/save call sites** — these are the exact lines the refactor must replace:

**Touch point 1 — `out_root` Path wrap** (`tiling.py` line 146):
```python
out = Path(out_root) if out_root is not None else map_dir / "pyramids"
```
New pattern: if `out_root` is already a `_GCSWriter` instance, use it directly; if `out_root` is a `str` starting with `gs://`, construct `_GCSWriter(fs, bare_path)`; if `None`, keep `map_dir / "pyramids"` as a local `Path`. The type-check branches on `isinstance(out_root, _GCSWriter)`.

**Touch point 2 — top-level mkdir** (`tiling.py` line 159):
```python
out.mkdir(parents=True, exist_ok=True)
```
`_GCSWriter.mkdir()` is a no-op (`self._fs.mkdirs(self._prefix, exist_ok=True)`). Call site unchanged syntactically.

**Touch point 3 — per-pyramid mkdir** (`tiling.py` lines 165–166):
```python
pdir = out / pid
pdir.mkdir(parents=True, exist_ok=True)
```
`_GCSWriter.__truediv__` returns a new `_GCSWriter`. Call site unchanged syntactically.

**Touch point 4 — three Pillow saves per tile** (`tiling.py` lines 172–174):
```python
img.crop(box).save(pdir / t["image"])
lc.crop(box).save(pdir / t["land_cover"])
topo.crop(box).save(pdir / t["topography"])
```
New pattern (must buffer then write_bytes, NOT pass `_GCSWriter` directly to PIL.save for the batch path):
```python
buf = io.BytesIO()
img.crop(box).save(buf, format="PNG")
(pdir / t["image"]).write_bytes(buf.getvalue())
```
Repeat for `land_cover` and `topography`. Alternatively use `_GCSWriter.open("wb")` as the save target — confirmed working because `GCSFile` implements the full seekable/writable buffer protocol.

**Touch point 5 — pyramid.json write_text** (`tiling.py` line 185):
```python
(pdir / "pyramid.json").write_text(json.dumps(manifest, indent=2))
```
`_GCSWriter.write_text(text)` delegates to `self._fs.pipe_file(self._prefix, text.encode())`. Call site unchanged syntactically.

**Touch point 6 — sample_weights.json shutil.copyfile** (`tiling.py` lines 189–195):
```python
shutil.copyfile(map_dir / _WEIGHTS_FILE, pdir / _WEIGHTS_FILE)
if (pdir / _WEIGHTS_FILE).read_bytes() != weights_blob:
    raise RuntimeError(...)
```
New pattern: `(pdir / _WEIGHTS_FILE).write_bytes(weights_blob)`. The integrity check `read_bytes()` comparison cannot be done against a GCS path without a round-trip read. Replace with an in-process check: `weights_blob` is already loaded at line 160 via `(map_dir / _WEIGHTS_FILE).read_bytes()` — the data written is exactly `weights_blob`, so the check becomes a no-op for the GCS path (the write is atomic via `pipe_file`; there is no partial-write risk). Keep the check only when `pdir` is a local `Path`.

**Concurrency addition** (new code in `tiling.py`):
The research mandates `ThreadPoolExecutor(max_workers=32)` wrapping the per-pyramid tile-write loop (touch points 4, 5, 6). The existing sequential loop at `tiling.py` lines 163–197 is the exact scope to parallelize. The `gcsfs.GCSFileSystem` instance is thread-safe for concurrent `pipe_file()` / `open()` calls.

---

### `scripts/build_dataset.py` (MODIFY — service/orchestrator, CRUD + file-I/O)

**Analog:** `scripts/build_dataset.py` (self-modification) + `scripts/seg/gcs_checkpoint.py`

**Preserve unchanged:**
- `_sanitize_stem`, `template_key`, `stratified_split` (lines 80–128) — logic unchanged
- Collision guard (lines 238–247) — HARD FAIL before any split/build
- `build_one_source` core render/label logic (lines 168–211)

**`load_or_create_split` — current local pattern** (`build_dataset.py` lines 131–165):
```python
split_path = Path(out_dir) / _SPLIT_FILENAME
if split_path.exists():
    data = json.loads(split_path.read_text())
    return set(data["test"])
# ... compute ...
split_path.write_text(json.dumps({...}))
```
New GCS pattern (copy from `gcs_checkpoint.py` `fs.exists` / `fs.cat` / `fs.pipe_file` idiom):
```python
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
split_gcs = f"{DATA_PREFIX}/synthetic/split.json"
if fs.exists(split_gcs):
    data = json.loads(fs.cat(split_gcs).decode())
    return set(data["test"])
# ... compute ...
fs.pipe_file(split_gcs, json.dumps({...}).encode())
```

**`build` function — raw dir listing, current pattern** (`build_dataset.py` line 219):
```python
geojsons = sorted(raw_dir.glob("*.geojson"))
```
New GCS pattern (from `gcs_checkpoint.py` `fs.ls` bare-prefix idiom, line 177):
```python
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
raw_gcs = f"{DATA_PREFIX}/synthetic/raw"
all_objects = fs.ls(raw_gcs)
gcs_filenames = [o.rsplit("/", 1)[-1] for o in all_objects if o.endswith(".geojson")]
```

**New validation insertion point** — mandatory ordering (`build_dataset.py` lines 238–250):
Insert `validate_manifest(gcs_filenames, manifest)` call AFTER the collision guard but BEFORE `load_or_create_split`. The collision guard is at lines 238–247; `load_or_create_split` is at line 250. The manifest hard-fail must occupy the slot between them.

**`tiling.tile()` call site** (`build_dataset.py` line 207):
```python
tiling.tile(out)
```
New pattern — pass a `_GCSWriter` as `out_root`:
```python
from gcs_io import _GCSWriter
gcs_map_prefix = f"{DATA_PREFIX}/synthetic/{split_name}/{src_id}__{style}"
tiling.tile(out, out_root=_GCSWriter(fs, gcs_map_prefix + "/pyramids"))
```
`out` (local Path) stays as `map_dir` for the read-side; only `out_root` changes.

**CLI arg defaults — Pitfall R-1 guard** (`build_dataset.py` lines 277–285):
Change `--raw-dir` default from `Path("data/synthetic/raw")` to `"gs://mapclass-training-northeast1/data/synthetic/raw"`. Change `--out-dir` default to `"gs://mapclass-training-northeast1/data/synthetic"`. Add a startup assertion: if `out_dir` does not start with `gs://`, require `--local-ok` flag.

---

### `scripts/build_historical_dataset.py` (MODIFY — service/orchestrator, CRUD + file-I/O)

**Analog:** `scripts/build_dataset.py` (role-match — identical tiling + write patterns)

**`tiling.tile()` call site** (`build_historical_dataset.py` line 110):
```python
tiling.tile(sample_dir)
```
New pattern — same as `build_dataset.py`:
```python
tiling.tile(sample_dir, out_root=_GCSWriter(fs, gcs_out_prefix + "/pyramids"))
```
`sample_dir` stays as a local Path (scratch) for reads.

**`manifest_path` write** (`build_historical_dataset.py` line 76):
```python
rumsey.emit_manifest(unregistered, manifest_path)
```
`rumsey.emit_manifest` currently writes to a local `Path`. RW-04 requires writing to `gs://…/data/historical/raw/unregistered_manifest.json`. Two options: (a) update `rumsey.emit_manifest` to accept a gcsfs `fs` + GCS path, or (b) write locally then `fs.pipe_file` the result. Option (b) follows the existing `gcs_checkpoint.py` write-to-BytesIO-then-pipe pattern and avoids modifying `rumsey.py`.

**`out_dir.mkdir` pattern** (`build_historical_dataset.py` line 127):
```python
out_dir.mkdir(parents=True, exist_ok=True)
```
When `out_dir` is a GCS prefix string, replace with `_GCSWriter(fs, gcs_out_prefix).mkdir()` (no-op for GCS; kept for local scratch compatibility).

**CLI arg defaults** (`build_historical_dataset.py` lines 169, 179):
Change `--raw-dir` default to a GCS path for build subcommand outputs. Note: raw GeoTIFF downloads from Rumsey remain local scratch (ephemeral-safe per RW-04 research finding). Change `--out-dir` default to `"gs://mapclass-training-northeast1/data/historical/dataset"`.

**ThreadPoolExecutor already present** (`build_historical_dataset.py` lines 138–144):
```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(_process_one, tif, out_dir): tif for tif in tifs}
    for fut in as_completed(futures):
        tif, exc = fut.result()
```
This is the correct concurrency pattern. The `_process_one` function will need the `fs` + GCS prefix passed through; thread safety relies on the same `gcsfs.GCSFileSystem` instance being passed (thread-safe, as confirmed for `build_dataset.py`).

---

### `scripts/build_satellite_dataset.py` (MODIFY — service/orchestrator, CRUD + file-I/O)

**Analog:** `scripts/build_historical_dataset.py` (role-match — structurally identical)

**`tiling.tile()` call site** (`build_satellite_dataset.py` line 219):
```python
tiling.tile(sample_dir)
```
New pattern — identical to historical:
```python
tiling.tile(sample_dir, out_root=_GCSWriter(fs, gcs_out_prefix + "/pyramids"))
```

**`manifest_path.write_text` call site** (`build_satellite_dataset.py` line 154):
```python
manifest_path.write_text(json.dumps({"scenes": resolved}, indent=2))
```
New GCS pattern:
```python
fs.pipe_file(gcs_manifest_path, json.dumps({"scenes": resolved}, indent=2).encode())
```
Follows `gcs_checkpoint.py` `pipe_file` pattern (lines 128–133 by analogy).

**Coverage summary write** (`build_satellite_dataset.py` `cmd_coverage_scan` — currently delegates to `coverage.build_summary(summary_path)`): The `--summary` default `_DEFAULT_SUMMARY = Path("data/satellite/coverage_summary.json")` must change to the GCS path. The write inside `coverage.build_summary` currently uses `Path.write_text`; add a GCS write-out step after it returns.

**CLI arg defaults** (`build_satellite_dataset.py` lines 57–59):
```python
_DEFAULT_SUMMARY = Path("data/satellite/coverage_summary.json")
_DEFAULT_MANIFEST = Path("data/satellite/resolved_scenes.json")
_DEFAULT_OUT = Path("data/satellite/dataset")
```
Change all three to `gs://mapclass-training-northeast1/data/satellite/...` strings.

**ThreadPoolExecutor already present** (`build_satellite_dataset.py` lines 248–255):
```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(_process_one, scene, out_dir): scene for scene in scenes}
    for fut in as_completed(futures):
        _, status = fut.result()
```
Same pattern as historical; same thread-safety approach applies.

---

### `scripts/finetune_seg.py` (MODIFY — service/training loop, request-response + file-I/O)

**Analog:** `scripts/finetune_seg.py` (self) + `scripts/seg/gcs_checkpoint.py`

**Existing GCS resume sequence** (`finetune_seg.py` lines 220–231):
```python
cfg = config_prefix(args.backbone, args.variant)
try:
    resume_step, resume_state = gcs_latest_checkpoint(cfg)
except ImportError:
    resume_step, resume_state = 0, None
```
The new pull-once step inserts BEFORE this block (RW-03 research, step 1 of job-start sequence). The `gcs_latest_checkpoint` call at line 222 is step 3 in the new sequence — its position unchanged.

**New job-start sequence** — insert before `carve_train_val` at line 206:
```python
# Step 1: pull-once (RW-03)
from gcs_io import pull_dataset_from_gcs, verify_pull
local_root = pull_dataset_from_gcs("train", args.scratch_dir)

# Step 2: read split.json from GCS + verify pull
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
split_json = json.loads(fs.cat(f"{DATA_PREFIX}/synthetic/split.json").decode())
verify_pull(local_root, split_json, "train")

# Step 3: existing GCS checkpoint resume (unchanged)
cfg = config_prefix(args.backbone, args.variant)
...

# Step 4: existing carve_train_val — point at local scratch root
train_ds, val_ds = carve_train_val(local_root, val_frac=0.2, seed=42)
```

**`--scratch-dir` arg addition** — follow the existing `--train-root` arg pattern (wherever it is defined in the argparse block):
```python
parser.add_argument("--scratch-dir", type=Path, default=Path("/tmp/mapclass_data"),
                    help="Local scratch dir for pull-once dataset cache (RW-03)")
```

**`ImportError` guard pattern** (already present at lines 221–225) — apply to `pull_dataset_from_gcs` and `verify_pull` the same way: wrap in `try/except ImportError` so offline CI (no gcsfs) falls back to `args.train_root` as before.

---

### `scripts/evaluate_seg.py` (MODIFY — service/eval harness, request-response + file-I/O)

**Analog:** `scripts/evaluate_seg.py` (self) + `scripts/seg/gcs_checkpoint.py`

**Existing `load_test_pyramid_dirs` call** (`evaluate_seg.py` lines 103–138 — reads local `split_json: Path` and `data_root: Path`):
```python
split = json.loads(split_json.read_text())
test_ids: List[str] = split["test"]
candidate = (data_root / "test" / map_id).resolve()
```
These local-Path reads continue to work post-RW-03 because the pull-once step downloads the test subset to local scratch before `load_test_pyramid_dirs` is called. The function itself is unchanged.

**New job-start sequence** — insert before `load_test_pyramid_dirs`:
```python
# Step 1: pull-once (RW-03)
from gcs_io import pull_dataset_from_gcs, verify_pull
local_root = pull_dataset_from_gcs("test", args.scratch_dir)

# Step 2: read split.json from GCS + verify pull
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
split_json_dict = json.loads(fs.cat(f"{DATA_PREFIX}/synthetic/split.json").decode())
verify_pull(local_root, split_json_dict, "test")

# Write split.json locally so load_test_pyramid_dirs can read it as a Path
local_split = args.scratch_dir / "split.json"
local_split.write_text(json.dumps(split_json_dict))

# Step 3: existing load_test_pyramid_dirs — unchanged
pdirs = load_test_pyramid_dirs(local_split, local_root)
```

**`--scratch-dir` arg addition** — same argparse pattern as `finetune_seg.py`.

**`--split-json` and `--data-root` arg defaults** (currently `data/synthetic/split.json` and `data/synthetic/` in the CLI):
Change defaults to the GCS paths; or alternatively, document that when `--scratch-dir` is set, both are derived from scratch. The planner should resolve this in plan 02-05.

---

## Shared Patterns

### GCS Auth / fs Instantiation
**Source:** `scripts/seg/gcs_checkpoint.py` lines 36–39, 45, 131, 171
**Apply to:** `scripts/gcs_io.py`, all three build scripts, `scripts/finetune_seg.py`, `scripts/evaluate_seg.py`
```python
try:
    import gcsfs  # type: ignore[import]
except ModuleNotFoundError:
    gcsfs = None  # type: ignore[assignment]

GCS_PROJECT = "narrative-campaign"

# Instantiate once per script entrypoint; pass instance to _GCSWriter
fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
```
Never re-instantiate `GCSFileSystem` per tile or per file. Never inline auth credentials — ADC only.

### fs.ls Bare-Prefix Idiom
**Source:** `scripts/seg/gcs_checkpoint.py` lines 174–179
**Apply to:** `scripts/gcs_io.py` (`pull_dataset_from_gcs`, `verify_pull`), `scripts/build_dataset.py` raw-dir listing
```python
bare_prefix = "mapclass-training-northeast1/data/synthetic/raw/"
try:
    files = fs.ls(bare_prefix)
except FileNotFoundError:
    return 0, None  # or handle appropriately
```
`fs.ls()` returns bare paths (no `gs://` prefix). Strip `gs://` when constructing bare_prefix.

### Atomic GCS Write (pipe_file for pre-buffered data)
**Source:** `scripts/seg/gcs_checkpoint.py` lines 127–133 (BytesIO variant)
**Apply to:** `scripts/gcs_io.py` `_GCSWriter.write_bytes`, `scripts/build_dataset.py` split.json write, all manifest writes
```python
# For pre-buffered bytes (JSON manifests, weights blobs):
fs.pipe_file(bare_gcs_path, data_bytes)

# For streaming data (Pillow PNG, torch checkpoint):
with fs.open(gs_path_or_bare_path, "wb") as fh:
    fh.write(data)
```

### Hard-Fail Pattern (sys.exit(1), not raise)
**Source:** `scripts/build_dataset.py` lines 243–247 (collision guard):
```python
if dupes:
    print("FATAL: source stems collide after sanitization — ...")
    for sid in sorted(dupes):
        print(f"  {sid!r} <- {sorted(dupes[sid])}")
    sys.exit(1)
```
**Apply to:** `validate_manifest` in `scripts/gcs_io.py` or `scripts/build_dataset.py`. Use `sys.exit(1)` (not `raise ValueError`, not `print("WARNING")`). Identical loudness to the collision guard.

### ThreadPoolExecutor + as_completed Error Re-raise
**Source:** `scripts/build_historical_dataset.py` lines 138–144
**Apply to:** `scripts/tiling.py` tile-write loop (new addition), all three build scripts (already present in historical + satellite)
```python
with ThreadPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(fn, *args): args for args in tasks}
    for fut in as_completed(futures):
        result = fut.result()  # re-raises any exception from the worker
```
For `tiling.py` tile writes: `max_workers=32` (research-verified safe against 1000 req/s quota). Each `_GCSWriter` instance passed to the pool must be a per-pyramid instance (not shared across threads — `_GCSWriter` is not thread-safe; the underlying `gcsfs.GCSFileSystem` is).

### Missing-file SKIP + Log Guard
**Source:** `scripts/build_dataset.py` lines 201–208, `scripts/build_historical_dataset.py` lines 102–110
**Apply to:** All three build scripts' tiling call sites (unchanged logic, only `tiling.tile` call changes)
```python
missing = [f for f in _REQUIRED_MAP_FILES if not (out / f).exists()]
if missing:
    print(f"  SKIP tiling {out.name}: missing {', '.join(missing)}")
else:
    tiling.tile(out, out_root=_GCSWriter(fs, gcs_prefix))
```

### Test Mock for gcsfs
**Source:** `tests/test_seg_gcs.py` lines 52–120
**Apply to:** `tests/test_gcs_io.py` (new), any test touching GCS in build scripts
```python
class _MockGCSFileSystem:
    _store: dict[str, bytes] = {}

    def open(self, path, mode="rb"): ...   # BytesIO-backed context manager
    def ls(self, prefix): ...              # filter _store keys by prefix
    # Add for gcs_io.py tests:
    def pipe_file(self, path, data): self._store[path.lstrip("gs://")] = data
    def cat(self, path): return self._store[path.lstrip("gs://")]
    def exists(self, path): return path.lstrip("gs://") in self._store
    def get(self, remote, local, recursive=False): ...  # write files to local dir
    def mkdirs(self, path, exist_ok=True): pass

mock.patch("gcs_io.gcsfs", _MockGCSModule)
```

---

## No Analog Found

No files in this phase lack a close analog. All 7 files have strong analogs as documented above.

---

## Metadata

**Analog search scope:** `scripts/`, `scripts/seg/`, `tests/`
**Files scanned:** 9 (gcs_checkpoint.py, tiling.py, build_dataset.py, build_historical_dataset.py, build_satellite_dataset.py, finetune_seg.py, evaluate_seg.py, test_seg_gcs.py, conftest.py)
**Pattern extraction date:** 2026-05-16

**Key constraint:** `gcs_checkpoint.py` is the single canonical GCS I/O template for this project. Every new GCS operation in Phase 2 must mirror its lazy-import, ADC-only-auth, bare-prefix `fs.ls`, and `pipe_file`/`fs.open` write idioms. No deviations.
