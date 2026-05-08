# Coding Conventions

**Analysis Date:** 2026-05-08

This codebase has **no enforced style tooling** — no `pyproject.toml`, `setup.cfg`,
`ruff.toml`, `.flake8`, `tox.ini`, `Makefile`, `.pre-commit-config.yaml`, or
`.github/workflows/`. Conventions are observational, derived from the eight
Python modules under `scripts/` and `scripts/historical/`. Style is consistent
across files, suggesting a single author, and broadly aligns with PEP 8 with a
relaxed ~110-character line budget.

## Naming Patterns

**Files / modules:**
- `snake_case.py` throughout (`build_dataset.py`, `biome_mapping.py`,
  `build_historical_dataset.py`, `toon_mapping.py`, `worldcover.py`).
- Top-level orchestrators are prefixed `build_*` (`scripts/build_dataset.py`,
  `scripts/build_historical_dataset.py`).
- Source-specific historical adapters live as flat modules under
  `scripts/historical/` named after the data source (`rumsey.py`, `dem.py`,
  `worldcover.py`).

**Functions:**
- Public API: `snake_case`, often verb-first (`render_map`,
  `make_labels`, `fetch_worldcover`, `fetch_topo`, `search_maps`,
  `download_georeferenced`, `emit_manifest`, `label_for_filename`).
- Private/helper: leading underscore (`_bbox`, `_rings`, `_field`, `_wms_url`,
  `_parse_bbox`, `_richness_score`, `_get_json`, `_haversine_km`,
  `_tile_name`, `_tile_origins`, `_remap`, `_classify`, `_slope_degrees`,
  `_laea_crs`, `_read_rgb`, `_to_wgs84_bbox`, `_process_one`).
- Subcommand handlers in CLI entry points are prefixed `cmd_`
  (`cmd_search`, `cmd_build` in `scripts/build_historical_dataset.py:45,82`).

**Variables:**
- `snake_case` for locals and parameters (`raw_dir`, `output_dir`, `geojson_path`,
  `wgs84_bbox`, `water_mask`, `dem_intermediate`, `metric_transform`).
- Loop indices use intuitive short names (`feat`, `geom`, `ring`, `tif`, `fut`).
- Module-level constants: `UPPER_SNAKE_CASE` and frequently leading-underscored
  when treated as private to the module:
  - Public constants (re-exported / referenced by sibling modules):
    `LANDCOVER_CLASSES`, `LANDCOVER_IDX`, `TOPO_CLASSES`, `TOPO_IDX`,
    `H_SEA_LEVEL`, `BIOME_TO_LANDCOVER_NAME`, `AZGAAR_BIOMES` in
    `scripts/biome_mapping.py:10-82`; `PALETTES`, `BORDER_STYLE`, `BACKGROUND`
    in `scripts/render.py:65-83`; `NODATA`, `WATER_TOPO` in
    `scripts/label.py:24-25`; `WC_REMAP`, `NODATA` in
    `scripts/historical/worldcover.py:42-56`; `FLAT_MAX_DEG`, `HILLY_MAX_DEG`,
    `WATER_TOPO` in `scripts/historical/dem.py:41-43`;
    `HISTORICAL_LC_WEIGHTS`, `HISTORICAL_TOPO_WEIGHT` in
    `scripts/historical/label.py:40-51`; `BG_WHITE`, `BG_PARCHMENT`, `BG_DARK`
    in `scripts/augment.py:20-22`; `TERRAIN_DESCRIPTIONS`, `PROMPT_TEMPLATES`
    in `scripts/toon_mapping.py:124-151`.
  - Private (leading underscore): tuning parameters and URL bases —
    `_FLAT`, `_ILLUSTRATED`, `_SATELLITE` (`scripts/render.py:29-63`),
    `_LUNA_SEARCH`, `_LUNA_HEADERS`, `_MIN_DIAG_KM`, `_MAX_DIAG_KM`,
    `_DATE_START`, `_DATE_END`, `_BATCH_SIZE`, `_MAX_RETRIES`, `_BACKOFF_BASE`,
    `_MAP_TYPES` (`scripts/historical/rumsey.py:40-56`),
    `_WC_BASE`, `_WC_FILENAME`, `_WGS84` (`scripts/historical/worldcover.py:38-57`),
    `_COP_BASE`, `_COP_PATH`, `_WGS84` (`scripts/historical/dem.py:35-44`),
    `_PREFIX_MAP`, `_RAW_MAP` (`scripts/toon_mapping.py:24-86`),
    `_HERE` (`scripts/build_historical_dataset.py:33`).

**Types / type aliases:**
- No custom classes are defined anywhere in `scripts/`. The codebase is
  entirely function-oriented. `dataclass`, `TypedDict`, `Protocol`, `Enum`,
  and `NamedTuple` are not used. Tuples and dicts carry structured data.

## Code Style

**Formatting:**
- No formatter (Black, Ruff format, autopep8, yapf) configured.
- 4-space indentation, consistent across all files.
- Line length: most lines are well under 100 characters; the longest line
  observed is 118 characters (`scripts/historical/label.py`). The hand-aligned
  comment columns in `scripts/render.py:29-63`, `scripts/biome_mapping.py:56-79`,
  and `scripts/toon_mapping.py:24-81` push some lines into the 90–105 range.
  This implies a soft budget around 100–110 characters, not the PEP 8 79.
- Manual columnar alignment is preferred for tabular constants. Two
  `# fmt: off` / `# fmt: on` markers exist in `scripts/biome_mapping.py:9,80`
  — these target Black/Ruff but no formatter is actually run, so they are
  defensive against future tooling.
- Trailing commas are used in multi-line literals
  (`scripts/historical/rumsey.py:400-423`).

**Linting:**
- No linter (Ruff, Flake8, Pylint, Pyflakes) configured.
- No `mypy` / `pyright` / `pytype` config — type hints are present but
  not type-checked.

## Type Hints

Type hints are **partially adopted, modern-syntax** (PEP 604 union
`X | None`, PEP 585 builtin generics `list[...]`, `dict[...]`, `tuple[...]`).

**Where present:**
- All public function signatures have parameter and return annotations:
  `def render_map(geojson_path: str | Path, output_dir: str | Path, styles=...) -> None`
  (`scripts/render.py:138`),
  `def fetch_worldcover(map_ds: "rasterio.DatasetReader", wgs84_bbox: tuple[float, float, float, float]) -> np.ndarray`
  (`scripts/historical/worldcover.py:94`),
  `def make_labels(map_geotiff: Path, output_dir: Path) -> None`
  (`scripts/historical/label.py:95`).
- Module-level dict/list constants are annotated where the structure is
  non-trivial (`_PREFIX_MAP: list[tuple[str, int, int | None]]`,
  `scripts/toon_mapping.py:24,84`; `WC_REMAP: dict[int, int]`,
  `scripts/historical/worldcover.py:42`;
  `HISTORICAL_LC_WEIGHTS: dict[str, float]`,
  `scripts/historical/label.py:40`).
- Forward references for third-party types appear as string literals
  (`"rasterio.DatasetReader"` in `scripts/historical/worldcover.py:95`,
  `scripts/historical/dem.py:96`).

**Where absent:**
- Internal helpers in `scripts/historical/rumsey.py` skip annotations on
  primitive parameters:
  `_haversine_km(lon1, lat1, lon2, lat2) -> float` (`:63`),
  `_field(item: dict, name: str, default=None)` (`:77`),
  `_download_wms_geotiff(wms_url: str, bbox: tuple, output_path: Path, size: int = 4096) -> bool`
  (`:283`, note the bare `tuple`).
- The `default=None` parameter pattern is preferred over
  `default: object = None`.
- Iterator/generator return types are not always annotated
  (`_tile_origins`, `scripts/historical/worldcover.py:67`).

**Convention:** annotate the public surface; relax for short private helpers.

## Docstring Style

**Module docstrings:**
- Every non-empty `.py` file opens with a triple-quoted module docstring.
  `scripts/historical/__init__.py` is intentionally empty (package marker).
- Module docstrings explain purpose, usage, and key invariants. Several use
  ASCII section dividers (`# --- Helpers ---`) below the docstring to
  segment files (e.g. `scripts/historical/rumsey.py:59-61, 188-190, 279-281,
  359-361`; `scripts/build_historical_dataset.py:41-43, 69-71, 120-122`;
  `scripts/render.py:25-27, 71, 78`; `scripts/augment.py:15-17, 40-42, 97-99`).

**Function docstrings:**
- Mostly **plain-prose, single- or multi-paragraph** docstrings rather than
  structured Google/Sphinx format.
- A handful adopt **NumPy-style** parameter/return blocks where the contract
  is non-obvious — see `fetch_worldcover` (`scripts/historical/worldcover.py:99-110`),
  `fetch_topo` (`scripts/historical/dem.py:101-114`), and
  `make_labels` in `scripts/historical/label.py:96-103`. Section headers
  use `Parameters`/`Returns` followed by a dashed underline
  (`Parameters\n----------`).
- Short helpers often use a single-line docstring
  (`_haversine_km`, `_bbox_diagonal_km`, `_tile_name`).
- Inline `#` comments are used liberally for tabular constants and to
  annotate non-obvious magic numbers (e.g. backoff seconds, sentinel values,
  WorldCover→canonical class remapping).

**Convention:** prose docstrings for everything; NumPy-style sections only
for functions whose parameters are easy to misuse.

## Import Organization

A consistent three-block layout is used throughout, separated by blank lines:

1. **Stdlib** (`json`, `sys`, `math`, `os`, `re`, `time`, `argparse`,
   `pathlib`, `concurrent.futures`).
2. **Third-party** (`numpy`, `rasterio`, `pyproj`, `requests`, `PIL`).
3. **First-party / local** (sibling modules in `scripts/` or
   `historical.*`).

**Style:**
- `from X import a, b` is preferred over qualified `X.a` calls when the
  symbol is used multiple times.
- Multiple imports from the same package are split onto separate `from`
  lines if they come from sub-modules
  (`from rasterio.crs import CRS`, `from rasterio.warp import Resampling, reproject`
  in `scripts/historical/worldcover.py:32-33`).
- One inline `from PIL import ImageEnhance` inside a function
  (`scripts/augment.py:89`) — used only inside `to_faded()` to keep the
  module-top imports trimmer. Treated as an exception, not a pattern.
- One late `from rasterio.warp import calculate_default_transform`
  inside `fetch_topo()` (`scripts/historical/dem.py:145`) for the same
  localisation reason.
- No `import *`, no aliased imports beyond conventional ones (`numpy as np`,
  `pyproj.CRS as ProjCRS` to disambiguate from `rasterio.crs.CRS` —
  `scripts/historical/dem.py:28-30`).

**Path Aliases:**
- No absolute package layout. Modules under `scripts/` import each other
  by bare name (`from label import make_labels`,
  `scripts/build_dataset.py:21`; `from biome_mapping import ...`,
  `scripts/render.py:23`, `scripts/label.py:22`, `scripts/toon_mapping.py:17`).
- `scripts/build_historical_dataset.py:33-35` mutates `sys.path` at import
  time so that `historical` resolves as a package whether you invoke from
  the repo root or from `scripts/`. This is the only sys.path manipulation in
  the codebase.

## Error Handling

Error handling is **minimal and print-based**. There is no logging framework
and no custom exception hierarchy.

**Patterns observed:**
- **Fail-fast at CLI entry**: missing required args trigger `print(...)` +
  `sys.exit(1)` (`scripts/build_dataset.py:31-32, 48-50`,
  `scripts/render.py:163-165`, `scripts/label.py:92-94`,
  `scripts/historical/label.py:153-155`,
  `scripts/build_historical_dataset.py:88-90`).
- **Per-item resilience in batch loops**: the orchestrator
  `scripts/build_historical_dataset.py:73-117` wraps each map's processing
  in `try/except Exception`, prints `ERROR <name>: <exc>`, accumulates
  `(tif, exc)` tuples in an `errors` list, and reports a summary at the end.
  Single failures never abort the batch.
- **Network resilience**: `_get_json` (`scripts/historical/rumsey.py:161-185`)
  implements explicit exponential backoff (`_BACKOFF_BASE ** attempt`) on
  HTTP 429/503 with `_MAX_RETRIES = 4`, and re-raises `RequestException`
  on the final attempt.
- **Silent skip for expected misses**: `scripts/historical/dem.py:124-128`
  uses bare `except Exception: pass` for ocean DEM tiles (404s are normal).
  This is the only `pass`-on-exception in the codebase, and the comment
  documents it.
- **Print-and-return-None for soft failures**: download/parse helpers in
  `scripts/historical/rumsey.py` (`_download_wms_geotiff:316`,
  `download_georeferenced:336-352`) return `None` or `False` so callers can
  branch without try/except.
- **Raise on contract violations**: empty inputs raise `ValueError`
  (`scripts/augment.py:120`); missing CRS in a GeoTIFF raises `ValueError`
  in `_to_wgs84_bbox` (`scripts/historical/label.py:57`) but is downgraded
  to `print + return` at the public `make_labels` boundary
  (`scripts/historical/label.py:111-113`).

**Total `try` / `except` blocks across all modules:** 10 — concentrated in
`historical/rumsey.py` (7), with single instances in `dem.py`,
`worldcover.py`, and `build_historical_dataset.py`. The synthetic pipeline
(`build_dataset.py`, `render.py`, `label.py`, `biome_mapping.py`,
`augment.py`, `toon_mapping.py`) contains zero exception handlers.

**No use of:**
- `logging` module (every diagnostic is a `print()`).
- Custom exception classes.
- `warnings.warn`.
- Context-rich error wrapping (`raise ... from exc`).

## Logging

**Framework:** none — `print()` is used uniformly.

**Patterns:**
- Top-level progress: `=== <phase> ===` banners
  (`scripts/build_historical_dataset.py:49,92`).
- Per-item indented status with `  ` two-space prefix
  (`scripts/historical/rumsey.py:354`,
  `scripts/historical/label.py:108-149`).
- Tail summaries: blank line + counts + optional error list
  (`scripts/build_historical_dataset.py:112-117`,
  `scripts/historical/rumsey.py:64-66`).
- Unicode ellipses (`…`) in user-facing progress messages
  (`scripts/historical/rumsey.py:168, 261`,
  `scripts/historical/label.py:126, 133`).
- Right-arrow (`→`) used to associate input/output in messages
  (`scripts/render.py:158`, `scripts/historical/label.py:149`,
  `scripts/build_historical_dataset.py:66`).

This makes pipeline output human-skimmable but unstructured — there is no
machine-parseable log stream.

## CLI Argument Parsing

**Mixed pattern, by complexity tier:**

- **Single-purpose scripts use raw `sys.argv`**: `build_dataset.py:46-53`,
  `render.py:161-168`, `label.py:91-95`, `historical/label.py:152-156`,
  `toon_mapping.py:169-177`. Each emits a usage string and `sys.exit(1)`
  on the wrong arg count. Defaults are inlined as tuples
  (`("flat", "illustrated", "satellite")`).
- **Multi-subcommand orchestrator uses `argparse`**:
  `scripts/build_historical_dataset.py:124-164` — `argparse.ArgumentParser`
  with `RawDescriptionHelpFormatter`, the module docstring as `epilog`, a
  required subparser group, and a nested `add_common(p)` helper to share
  `--raw-dir` across `search`/`build`/`full`. This is the only `argparse`
  usage in the repo.

**No use of:** `click`, `typer`, `fire`, `docopt`.

**Convention:** if you need flags, switch to `argparse`; otherwise positional
`sys.argv` is fine for ≤2 args.

## Configuration / Path Sharing

There is **no central config module, no YAML, no `.env` loader, no Pydantic
Settings class.** Configuration lives in two forms:

- **Module-level constants** for tunables (palettes in
  `scripts/render.py:65-83`, search caps in
  `scripts/historical/rumsey.py:42-56`, slope thresholds in
  `scripts/historical/dem.py:41-43`, loss weights in
  `scripts/historical/label.py:40-51`).
- **CLI defaults via argparse** for paths
  (`--raw-dir data/historical/raw`, `--out-dir data/historical/dataset` in
  `scripts/build_historical_dataset.py:134, 145, 152`).

The single shared taxonomy module is `scripts/biome_mapping.py`:
`LANDCOVER_CLASSES`, `LANDCOVER_IDX`, `TOPO_CLASSES`, `TOPO_IDX`, and
`H_SEA_LEVEL` are imported by `render.py`, `label.py`, `toon_mapping.py`,
and indirectly informed by `historical/worldcover.py` and
`historical/label.py` (the `WC_REMAP` and `HISTORICAL_LC_WEIGHTS` indices
must match `LANDCOVER_CLASSES`). Coupling is by convention; nothing
enforces alignment.

`os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")` is set at module top
in `scripts/historical/worldcover.py:36` and `scripts/historical/dem.py:33`
to enable anonymous S3 access via GDAL VSI-CURL.

## Comments

**Patterns:**
- Tabular `#` comments on data structures, often hand-aligned to a column
  (`scripts/render.py:29-39`, `scripts/biome_mapping.py:56-79`,
  `scripts/toon_mapping.py:24-81`,
  `scripts/historical/worldcover.py:42-54`,
  `scripts/historical/label.py:40-50`).
- ASCII section dividers — a row of dashes following `# ---` — used to
  break large modules into named regions
  (`scripts/historical/rumsey.py:59, 188, 279, 359`;
  `scripts/build_historical_dataset.py:41, 69, 120`;
  `scripts/render.py:25-27, 71, 78`;
  `scripts/augment.py:15-17, 40-42, 97-99`;
  `scripts/toon_mapping.py:19-22, 118-120`).
- Inline rationale comments are common, especially for non-obvious geometric,
  numeric, or remote-API behaviour (e.g. "ocean tile or missing — expected"
  in `scripts/historical/dem.py:128`; "deduplicate, preserve order" in
  `scripts/build_historical_dataset.py:86`).

**JSDoc/TSDoc:** N/A (Python project).

## Function Design

**Size:** Most functions are 5–25 lines. The longest functions are
`search_maps` (`scripts/historical/rumsey.py:192-276`, ~85 lines, dense with
inline filtering logic) and `fetch_topo` (`scripts/historical/dem.py:95-191`,
~95 lines, multi-stage geospatial pipeline). Both exceed comfortable
single-screen length but stay readable thanks to ASCII section comments
inside.

**Parameters:**
- Path-like parameters accept `str | Path` at public entry points and are
  immediately normalised via `Path(...)`
  (`scripts/render.py:139-140`, `scripts/label.py:52-53`,
  `scripts/build_dataset.py:26-27`).
- Default arguments are tuples or simple immutables. No mutable defaults.
- Keyword-only arguments are not used (no `*` separators in signatures).

**Return Values:**
- Fixed-arity tuples for bbox-like data:
  `tuple[float, float, float, float]`
  (`scripts/label.py:28`, `scripts/historical/label.py:54`,
  `scripts/historical/worldcover.py:96`).
- `Optional` returns indicate "soft miss"
  (`-> int | None`, `-> str | None`, `-> Path | None`,
  `-> tuple[...] | None`) — see
  `scripts/biome_mapping.py:97`,
  `scripts/historical/rumsey.py:93,109,322`,
  `scripts/toon_mapping.py:89`.

## Module Design

**Exports:**
- No `__all__` declarations anywhere. Everything not prefixed `_` is
  effectively public; the underscore convention is the only access control.

**Barrel Files:**
- `scripts/historical/__init__.py` is empty. Sibling modules under
  `historical/` are imported individually as `from historical import rumsey`
  or `from historical.dem import fetch_topo` rather than re-exported through
  the package.

**Package vs. flat scripts:**
- `scripts/` itself is **not** a Python package — it has no `__init__.py`.
  Cross-script imports rely on whatever runs the entry point being invoked
  with `scripts/` on `sys.path`, either implicitly (running
  `python scripts/build_dataset.py` adds `scripts/`) or explicitly (the
  `sys.path.insert` shim at `scripts/build_historical_dataset.py:33-35`).

## Tooling Inventory (Absent)

The following common Python tooling artefacts are **not present**:

- `pyproject.toml`, `setup.cfg`, `setup.py`
- `pre-commit-config.yaml`
- `ruff.toml`, `.flake8`, `.pylintrc`, `.bandit`
- `mypy.ini`, `pyrightconfig.json`
- `tox.ini`, `nox.py`
- `Makefile` or `justfile`
- `.github/workflows/` (no GitHub Actions; no CI of any kind)
- `requirements-dev.txt`, `requirements-test.txt`
- `.editorconfig`

`requirements.txt` exists with **10 unpinned dependencies** (only `transformers>=4.41`
and `peft>=0.10` carry version specifiers; the other eight — `Pillow`, `numpy`,
`rasterio`, `pyproj`, `requests`, `torch`, `accelerate`, `bitsandbytes` — are
unbounded). There is no lockfile (`requirements.lock`, `uv.lock`, `poetry.lock`,
`Pipfile.lock`, or `pixi.lock`). Reproducibility relies on whatever pip resolves
at install time.

## Summary: What "the convention" is, in practice

When adding code to this repo:

1. Use 4-space indent, snake_case modules and functions, leading-underscore
   private helpers, UPPER_SNAKE constants (leading-underscore for private
   tuning knobs), trailing commas in multi-line literals.
2. Open every module with a triple-quoted docstring describing purpose and
   usage. Use ASCII `# --- Section ---` dividers if the file grows past
   ~150 lines.
3. Annotate the public function surface with PEP 604/585 syntax
   (`X | None`, `list[X]`). Skip annotations on tiny private helpers if
   they hurt readability. Don't add a type checker — none is configured.
4. Use `print()` for diagnostics, with `===` banners for phases and
   two-space-indented per-item lines.
5. For ≤2 positional args, parse `sys.argv` directly. For anything with
   subcommands or flags, use `argparse` with a `RawDescriptionHelpFormatter`
   and the module docstring as `epilog`.
6. Wrap per-item batch operations in `try/except Exception` and accumulate
   errors for an end-of-run summary; let the synthetic single-shot scripts
   crash on uncaught errors.
7. Return `None` (or `False`) from helpers for soft failures, `raise
   ValueError` for contract violations, downgrade to `print + return` at
   public boundaries when a partial result is acceptable.
8. Put shared taxonomy constants in `biome_mapping.py`. Don't introduce a
   config framework; module-level constants and argparse defaults are the
   convention.

---

*Convention analysis: 2026-05-08*
