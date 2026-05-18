# Technology Stack

**Analysis Date:** 2026-05-08

## Languages

**Primary:**
- Python (>= 3.10) — Entire codebase. Modern PEP 604 union syntax (`int | None`) is used throughout `scripts/toon_mapping.py`, `scripts/label.py`, `scripts/render.py`, and `scripts/historical/*.py`, which mandates 3.10 or later. PEP 585 generics (`list[tuple[...]]`, `dict[str, float]`) reinforce this floor. No upper bound is pinned; the developer machine observed during analysis runs CPython 3.12.3.

**Secondary:**
- Markdown — Project documentation (`README.md`, `mockup.md`, `executive_TODO.md`, `notes/architectural_references.md`).
- JSON / GeoJSON — Data interchange (Azgaar exports consumed by `scripts/render.py` and `scripts/label.py`; LUNA API responses parsed in `scripts/historical/rumsey.py`; manifests written to `unregistered_manifest.json`; per-sample metadata in `sample_weights.json`).
- GeoTIFF / PNG — Raster I/O (input and output of the pipeline; not a "language" but the dominant on-disk binary format and a defining stack characteristic).

**Not used (despite being repository-adjacent):**
- No JavaScript/TypeScript, despite Azgaar's Fantasy Map Generator being a JS web app — the upstream tool is invoked manually by the developer; only its GeoJSON export is consumed.
- No SQL — no database in the repo.
- No shell scripts in version control.

## Runtime

**Environment:**
- CPython 3.10+ (3.12.3 verified locally).
- No `.python-version` file, no `.nvmrc`, no `runtime.txt`. Python version floor is implicit in the type-hint syntax.

**Package Manager:**
- pip — `requirements.txt` is the only manifest. No `pyproject.toml`, no `setup.py`, no `setup.cfg`, no `Pipfile`, no `poetry.lock`, no `uv.lock`, no `pixi.toml`.
- Lockfile: **missing.** `requirements.txt` uses bare names (no exact pins) for all but two packages. Builds are not reproducible.

**Virtual Environment:**
- Not committed. `.gitignore` lists `.venv/`, `venv/`, `env/`, `ENV/` — convention is venv-based isolation, but no name is enforced.

## Frameworks

**Core (in active use):**
- **Pillow (PIL)** — All synthetic rendering and label rasterisation. Used in `scripts/render.py` (`PIL.Image`, `ImageDraw`, `ImageFilter`), `scripts/label.py` (`Image`, `ImageDraw`), `scripts/augment.py` (`Image`, `ImageFilter`, `ImageEnhance`), `scripts/historical/label.py` (`Image.fromarray` for PNG export of label rasters).
- **NumPy** — Numerical work behind augmentation (`scripts/augment.py`) and remote-sensing label generation (`scripts/historical/worldcover.py`, `scripts/historical/dem.py`, `scripts/historical/label.py`). Slope computation uses `np.gradient`, NaN-aware arithmetic, and `np.uint8` raster buffers.
- **rasterio** — GeoTIFF I/O and reprojection, including streaming reads from public S3 buckets via GDAL VSI-CURL. Used in `scripts/historical/worldcover.py`, `scripts/historical/dem.py`, `scripts/historical/label.py`. Specific submodules: `rasterio.crs.CRS`, `rasterio.warp.reproject`, `rasterio.warp.Resampling`, `rasterio.warp.calculate_default_transform`, `rasterio.band`.
- **pyproj** — CRS construction and coordinate transforms. Used in `scripts/historical/label.py` (`Transformer.from_crs` for converting native dataset CRS → WGS84 envelope) and `scripts/historical/dem.py` (`pyproj.CRS.from_dict` to build a Lambert Azimuthal Equal Area metric CRS for slope calculation).
- **requests** — HTTP client for the David Rumsey LUNA search API and Georeferencer WMS endpoints (`scripts/historical/rumsey.py`). Includes manual exponential-backoff retry logic against HTTP 429/503.

**Pinned but not yet imported (scaffolding for upcoming work):**
- **PyTorch** (`torch`) — Will host the GeoViLM backbone, OCR module, and dense segmentation heads. No `import torch` exists in the repo as of this snapshot.
- **Hugging Face transformers** (`transformers>=4.41`) — Intended for loading PaliGemma-3B / SigLIP / CLIP weights for the zero-shot benchmark and fine-tuned backbone. Version pin matches PaliGemma's introduction window in the HF library.
- **PEFT** (`peft>=0.10`) — LoRA / adapter fine-tuning for the backbone. Not yet wired up.
- **accelerate** — HF distributed-training launcher.
- **bitsandbytes** — 8-bit / 4-bit quantisation for fitting PaliGemma-3B on smaller GPUs.

**Testing:**
- None. There is no `tests/` directory, no `pytest.ini`, no `conftest.py`, no `tox.ini`, no `nox` config. `.gitignore` contains test-tooling entries (`.pytest_cache/`, `.coverage`, etc.) by convention only.

**Build/Dev:**
- None. No build system (no `Makefile`, no `tasks.py`, no `tox.ini`). All scripts are run directly with `python <script>.py`.
- No linter / formatter configuration in the repo (no `.ruff.toml`, no `.flake8`, no `pyproject.toml [tool.black]`). `.gitignore` references `.ruff_cache/` and `.mypy_cache/` defensively but neither is configured.

## Key Dependencies

Every dependency declared in `requirements.txt` and its role in the codebase:

| Package | Pin | In-tree usage | Role |
|---|---|---|---|
| `Pillow` | unpinned | `scripts/render.py`, `scripts/label.py`, `scripts/augment.py`, `scripts/historical/label.py` | Raster image creation, polygon rasterisation, PNG I/O, sepia/parchment augmentation. |
| `numpy` | unpinned | `scripts/augment.py`, `scripts/historical/{worldcover,dem,label}.py` | Array math, slope computation, class remapping, normalisation. |
| `rasterio` | unpinned | `scripts/historical/{worldcover,dem,label}.py` | GeoTIFF read/write, CRS-aware reprojection, S3 streaming via GDAL. |
| `pyproj` | unpinned | `scripts/historical/{label,dem}.py` | CRS objects (LAEA construction, WGS84 transforms). |
| `requests` | unpinned | `scripts/historical/rumsey.py` | HTTP GET against LUNA API and Georeferencer WMS, with retry + custom User-Agent. |
| `torch` | unpinned | (none yet) | Reserved for ML work. |
| `transformers` | `>=4.41` | (none yet) | Reserved for VLM / segmentation backbones. |
| `peft` | `>=0.10` | (none yet) | Reserved for adapter fine-tuning. |
| `accelerate` | unpinned | (none yet) | Reserved for distributed launch. |
| `bitsandbytes` | unpinned | (none yet) | Reserved for quantised inference / training. |

**Transitive system dependencies (not in requirements.txt but required at runtime):**
- **GDAL** — `rasterio` and `pyproj` are thin Python bindings over GDAL/PROJ. The codebase relies on GDAL's VSI-CURL driver to stream COG / GeoTIFF tiles directly from `https://*.s3.amazonaws.com/...` without first downloading them. This requires GDAL to be available with HTTPS support; on Linux, this typically comes via the `rasterio` wheel which ships its own GDAL.
- **CUDA** (future) — implied by `torch` + `bitsandbytes` once ML training begins; not required for any code currently executable.

## Configuration

**Environment Variables:**
- `AWS_NO_SIGN_REQUEST=YES` — set as a default in `scripts/historical/worldcover.py:36` and `scripts/historical/dem.py:33` via `os.environ.setdefault`. This tells GDAL to make anonymous (unsigned) requests against the public ESA WorldCover and Copernicus DEM S3 buckets. No AWS credentials are required.
- No other environment variables are read.
- No `.env` file exists in the repo and none is referenced. No `python-dotenv` dependency.

**Module-level constants used as configuration** (`scripts/historical/rumsey.py`):
- `_LUNA_SEARCH = "https://www.davidrumsey.com/luna/servlet/as/search"` — David Rumsey LUNA endpoint.
- `_LUNA_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}` — fixed UA string.
- `_MIN_DIAG_KM = 100`, `_MAX_DIAG_KM = 2000` — bbox extent filter (regional-scale maps only).
- `_DATE_START = 1500`, `_DATE_END = 1700` — historical date window (16th–17th century).
- `_BATCH_SIZE = 50`, `_MAX_RETRIES = 4`, `_BACKOFF_BASE = 2.0` — pagination + retry tuning.
- `_MAP_TYPES` — set of LUNA `Type` field values that count as "actual maps" (filters out atlas covers, text pages, etc.).

**Module-level constants in `scripts/historical/dem.py`:**
- `FLAT_MAX_DEG = 2.0`, `HILLY_MAX_DEG = 15.0` — slope thresholds for topography classification.
- `WATER_TOPO = 255` — uint8 sentinel for "no topography class" (also used in `scripts/label.py`).

**Per-source loss weights** (`scripts/historical/label.py:40`): `HISTORICAL_LC_WEIGHTS` dictionary encodes temporal-reliability weights per land cover class for the historical-map source (e.g. `built_up: 0.1`, `cropland: 0.15`, `topography: 1.0`). These are written to each sample's `sample_weights.json` and are intended to be picked up by the (not-yet-implemented) training loop.

**Build:**
- No build configuration. Scripts run directly via `python scripts/<name>.py` or `python -m historical.label`.

## Platform Requirements

**Development:**
- Linux or macOS recommended (rasterio + GDAL wheels available on both; Windows needs care with GDAL).
- Python 3.10+ on PATH.
- Sufficient disk for `data/` (gitignored — Azgaar GeoJSONs, rendered PNGs, downloaded historical GeoTIFFs, and label rasters can grow large).
- Network access to `davidrumsey.com`, `maps.georeferencer.com`, `esa-worldcover.s3.amazonaws.com`, `copernicus-dem-30m.s3.amazonaws.com` for dataset construction.

**Production / training:**
- Cloud training is delegated to a separate VM-provisioning repo over SSH (per `README.md` "Cloud / training workflow"). This repository deliberately ships no Docker, no Terraform, no cloud-init, and no CI configuration.
- A GPU with ≥16 GB VRAM is implied for PaliGemma-3B fine-tuning once that work lands; quantisation via `bitsandbytes` may relax this.

## Run Conventions

**Synthetic dataset build:**
```bash
python scripts/build_dataset.py <raw_dir> <output_dir> [flat illustrated satellite]
```
Reads `*.geojson` from `<raw_dir>` (Azgaar exports), writes `flat.png`, `illustrated.png`, `satellite.png`, `land_cover.png`, `topography.png` per map under `<output_dir>/<name>/`.

**Historical dataset build:**
```bash
python scripts/build_historical_dataset.py search [--raw-dir DIR] [--max-maps N]
python scripts/build_historical_dataset.py build  [--raw-dir DIR] [--out-dir DIR] [--workers N]
python scripts/build_historical_dataset.py full   [...]
```
The `search` subcommand calls David Rumsey's LUNA API and downloads georeferenced GeoTIFFs; `build` consumes those GeoTIFFs and produces `image.png`, `land_cover.png`, `topography.png`, `sample_weights.json`. Default paths: `data/historical/raw/` and `data/historical/dataset/`.

**Single-file utilities (importable or runnable):**
- `scripts/render.py <geojson> <output_dir>` — render one Azgaar GeoJSON in chosen styles.
- `scripts/label.py <geojson> <output_dir>` — produce `land_cover.png` + `topography.png` for one Azgaar GeoJSON.
- `python -m historical.label <map.tif> <output_dir>` — process a single georeferenced historical GeoTIFF.
- `scripts/toon_mapping.py [toon_dir]` — when run as `__main__`, prints class-distribution stats over hex tiles in `data/toons/`.

**Import-path quirk:**
`scripts/build_historical_dataset.py:33-35` mutates `sys.path` to insert its own directory so that `from historical import ...` works when the script is executed from either the repo root or `scripts/`. The `scripts/` directory is **not a Python package** (no `__init__.py`); only `scripts/historical/` is a package — but its `__init__.py` is empty (1 line, no exports).

## Things deliberately not in the stack

These are called out because their absence is load-bearing for the system-paper plan:

- **No Docker / containerisation** in this repo (per `README.md`, handled by a sibling VM-provisioning repo).
- **No CI/CD config** (no `.github/`, no `.gitlab-ci.yml`, no `bitbucket-pipelines.yml`).
- **No experiment tracker** (no `wandb`, `mlflow`, `comet_ml`, `clearml`, `tensorboard` import or dependency).
- **No model checkpoints in tree** — `models/` exists per `README.md` but is currently empty.
- **No data versioning** (no `dvc.yaml`, no `lakeFS`, no `git-lfs`). `data/` is gitignored entirely.
- **No annotation tooling** in tree — `README.md` mentions Label Studio / CVAT as candidates if a hand-annotated real-domain test set is added later.
- **No OCR library yet** (no `pytesseract`, no `easyocr`, no `craft-text-detector`) — the rotationally-invariant OCR module is unimplemented; CRAFT and ABCNet are named in `README.md` as starting points.
- **No segmentation framework** (no `segmentation_models_pytorch`, no `mmsegmentation`) — segmentation heads are unimplemented.
- **No georeferencing library** beyond `pyproj` / `rasterio` — the auto-georeferencing pipeline (cross-correlation + thin-plate-spline) is unimplemented; no `scikit-image`, no `OpenCV`, no `gdal_translate` wrapper present.

---

*Stack analysis: 2026-05-08*
