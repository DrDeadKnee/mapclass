# Testing Patterns

**Analysis Date:** 2026-05-08

## Status: No Test Suite Exists

The MapClass / GeoViLM codebase **has no automated tests of any kind.** This
is the most consequential finding of the quality audit and is flagged
explicitly here because the project's plan calls for a benchmark and a
training phase, both of which materially benefit from regression coverage on
the data pipeline.

### Evidence

- No files matching `test_*.py`, `*_test.py`, or `conftest.py` anywhere
  under `/home/drdreadknee/mapclass/` (verified via `find`).
- No `tests/` or `test/` directory exists at the repo root, under
  `scripts/`, or under `scripts/historical/`.
- `requirements.txt` (`/home/drdreadknee/mapclass/requirements.txt`)
  contains 10 dependencies; **none** are testing tools (no `pytest`,
  `unittest` is stdlib but not invoked, no `hypothesis`, no `coverage`,
  no `pytest-cov`, no `tox`, no `nox`, no `responses`, no `vcrpy`,
  no `freezegun`, no `pytest-mock`).
- No test runner configuration: no `pyproject.toml`, no `setup.cfg`, no
  `pytest.ini`, no `tox.ini`, no `noxfile.py`. The `.gitignore` does
  reference `.pytest_cache/`, `htmlcov/`, `.tox/`, `.nox/`, `.coverage`,
  `.hypothesis/`, but these are boilerplate Python `.gitignore` entries
  (the file mirrors GitHub's standard Python template) — they do not
  imply that any of those tools have ever run.
- No CI/CD config: `.github/workflows/` does not exist, nor `.gitlab-ci.yml`,
  `.circleci/`, `azure-pipelines.yml`, `Jenkinsfile`, `.travis.yml`, or
  `bitbucket-pipelines.yml`. There is no automated build or test run on
  push or pull request.
- No pre-commit hooks: `.pre-commit-config.yaml` is absent.
- No `Makefile`, `justfile`, or comparable task runner that might define a
  `test` target.
- No `__main__` self-test blocks in any script run anything that could be
  called a test — the `if __name__ == "__main__":` blocks present in
  `scripts/build_dataset.py:46`, `scripts/render.py:161`,
  `scripts/label.py:91`, `scripts/historical/label.py:152`,
  `scripts/toon_mapping.py:169`, and `scripts/build_historical_dataset.py:167`
  are all CLI entry points that invoke the production code on user-supplied
  paths.

### What This Means

Every module in `scripts/` and `scripts/historical/` is **completely untested**
in any automated sense. Correctness depends entirely on:

1. The author manually running the pipeline end-to-end on real GeoJSON /
   GeoTIFF inputs and visually inspecting the output PNGs.
2. The print-statement diagnostics emitted by each script (counts,
   bounding boxes, per-tile warnings) being read and judged by eye.

There is no safety net for refactoring. Any change to `biome_mapping.py`'s
class indices, `worldcover.py`'s remap table, `dem.py`'s slope thresholds,
or any geometric helper (`_bbox`, `_rings`, `_haversine_km`,
`_tile_origins`, `_tile_urls`, `_to_wgs84_bbox`) silently affects every
downstream artefact, with no way to detect a regression except by
re-running the full data build.

## Test Framework

**Runner:** None. Neither `pytest` nor `unittest` (despite the latter being
stdlib) is used.

**Assertion Library:** None.

**Run Commands:** Not applicable — no test command exists. The closest
analogues are the production CLI invocations:

```bash
python scripts/build_dataset.py <raw_dir> <output_dir>             # synthetic
python scripts/build_historical_dataset.py search                  # historical: download
python scripts/build_historical_dataset.py build                   # historical: label
python scripts/build_historical_dataset.py full                    # both
python scripts/render.py <geojson> <out>                           # render only
python scripts/label.py <geojson> <out>                            # label only
python -m historical.label <map.tif> <out>                         # historical label only
python scripts/toon_mapping.py <toon_dir>                          # diagnostic: list+count
```

`python scripts/toon_mapping.py <toon_dir>` (`scripts/toon_mapping.py:169-177`)
is the only script whose `__main__` block exists primarily to print a
diagnostic summary — it lists labelled tiles per land-cover class — but it
does not assert any expected values; it is a smoke check, not a test.

## Test File Organization

**Location:** N/A — no test files exist.

**Naming:** N/A.

**Structure:** N/A.

## Test Structure

**Suite Organization:** N/A.

**Patterns:** N/A.

## Mocking

**Framework:** None configured (`unittest.mock` is stdlib but not imported
anywhere).

**Patterns:** N/A.

**What to Mock:** N/A.

**What NOT to Mock:** N/A.

## Fixtures and Factories

**Test Data:** N/A — no test fixtures exist. The pipeline operates on
real input data under `data/` (gitignored — see
`/home/drdreadknee/mapclass/.gitignore:2`):

- `data/raw/*.geojson` (Azgaar Fantasy Map Generator exports) for the
  synthetic build.
- `data/historical/raw/georeferenced/*.tif` (Rumsey downloads) and
  `data/historical/raw/*.tif` (manually QGIS-registered) for the
  historical build.
- `data/toons/hex_*.png` for `toon_mapping.py`.

None of these are committed; none of them are minimised "fixture-sized"
copies suitable for fast tests.

## Coverage

**Requirements:** None enforced. No `coverage.py`, no `pytest-cov`, no
coverage gate.

**View Coverage:** N/A.

## Test Types

**Unit Tests:** None.

**Integration Tests:** None automated. The two `build_*` orchestrators are
de-facto manual integration tests when the author runs them on real data.

**E2E Tests:** None.

## Common Patterns

**Async Testing:** N/A — codebase is synchronous except for a
`ThreadPoolExecutor` in `scripts/build_historical_dataset.py:104-110`,
which is also untested.

**Error Testing:** N/A.

## Continuous Integration

**Status:** Absent. There is no CI of any kind — no GitHub Actions, GitLab
CI, CircleCI, Azure Pipelines, Travis, or Jenkins configuration in the
repository.

**Implications:** Pull requests, branch merges, and pushes to
`refactor_paper` (the current working branch) trigger no automated checks.
The `requirements.txt` file is not exercised against any runtime; an
unpinned dependency could break the pipeline silently the next time
`pip install -r requirements.txt` runs in a new environment.

## Gap Analysis (relevant to upcoming benchmark/training phases)

This section flags untested behaviour that is most likely to bite during
the planned benchmark and training work. It is intentionally specific so
it can be lifted into CONCERNS.md or future test plans.

1. **Taxonomy index alignment.** `LANDCOVER_CLASSES`
   (`scripts/biome_mapping.py:38-48`) is the single source of truth for
   class indices, but three independent remap tables must stay in lockstep:
   `BIOME_TO_LANDCOVER_NAME` (synthetic, `:56-79`),
   `WC_REMAP` (ESA WorldCover, `scripts/historical/worldcover.py:42-54`),
   and the `_RAW_MAP` prefix table (toon hex tiles,
   `scripts/toon_mapping.py:24-81`). A silent off-by-one in any of these
   poisons every downstream label. No test asserts that the keys/values
   line up.

2. **Geometric helpers.** `_bbox`, `_rings`, and the polygon→raster code
   in `scripts/render.py` and `scripts/label.py` are duplicated between
   the two files (compare `scripts/render.py:86-105` to
   `scripts/label.py:28-48`) and silently drift. No test pins their
   behaviour on a known GeoJSON.

3. **Slope classification thresholds.** `FLAT_MAX_DEG = 2.0`,
   `HILLY_MAX_DEG = 15.0` (`scripts/historical/dem.py:41-42`) define the
   topography boundary; the synthetic side uses
   `normalize_land_h` cutoffs at 20/55 (`scripts/biome_mapping.py:97-107`).
   These two definitions of "hilly" are not numerically reconciled and
   nothing tests that they produce comparable distributions on matched
   regions.

4. **Network-dependent code paths.** `scripts/historical/rumsey.py` makes
   live HTTPS calls to `davidrumsey.com` and `maps.georeferencer.com`,
   and `scripts/historical/worldcover.py` and `scripts/historical/dem.py`
   open S3 URLs through `rasterio`. None of these are mocked. The retry
   logic in `_get_json` (`scripts/historical/rumsey.py:161-185`) and the
   silent-skip-on-404 behaviour in
   `scripts/historical/dem.py:124-128` cannot be exercised offline and
   regress invisibly.

5. **CRS round-tripping.** `_to_wgs84_bbox`
   (`scripts/historical/label.py:54-73`) reprojects the four corners of a
   GeoTIFF to WGS84 and takes the enclosing envelope. This is correct only
   for non-degenerate, non-pole-crossing extents. There is no test asserting
   correct behaviour at the date line, near the poles, or on a square
   raster in a non-conformal projection.

6. **Loss weight schema.** `HISTORICAL_LC_WEIGHTS`
   (`scripts/historical/label.py:40-50`) is dumped as JSON to
   `sample_weights.json`. No test pins the schema; a downstream training
   loop that consumes a key change will fail at runtime, not at commit time.

### Recommended minimum to add later (not done yet, documented as a gap)

- A `tests/` directory with `pytest` as the runner.
- Pure-function unit tests for `biome_mapping.h_to_landcover` /
  `h_to_topo`, `worldcover._remap`, `worldcover._tile_origins`,
  `dem._tile_urls`, `dem._classify`, `toon_mapping.label_for_filename`,
  `augment.composite`, and the geometric `_bbox` / `_rings` helpers.
- One end-to-end smoke test that runs `build_dataset` against a
  miniature GeoJSON fixture and asserts file existence + image dimensions.
- HTTP mocks (`responses` or `vcrpy`) for the Rumsey LUNA API.
- A GitHub Actions workflow running `pytest` on push.

None of this exists today. This document records the absence; addressing
it is out of scope for the codebase mapper.

---

*Testing analysis: 2026-05-08*
