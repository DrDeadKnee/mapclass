---
phase: 01-end-to-end-skeleton
plan: 01
subsystem: data + packaging foundation
tags: [phase-1, walking-skeleton, data-layer, packaging, contracts]
requires: []
provides:
  - mapclass.data.taxonomy.taxonomy_hash
  - mapclass.data.contract.assert_sample_valid
  - mapclass.data.contract.SampleContractError
  - mapclass.data.loss_weights.LossWeights
  - mapclass.data.loss_weights.LossWeightsSchemaError
  - mapclass.data.splits.build_splits
  - mapclass.data.splits._bucket_for_sample_id
  - mapclass.data.splits.SplitsContaminationError
  - mapclass.data.dataset.MapClassDataset
  - mapclass.seeding.set_global_seed
  - mapclass/configs/loss_weights.yaml
  - mapclass/configs/v0.yaml
  - pyproject.toml installable via `pip install -e .`
  - requirements.lock.txt (uv-compiled, exact pins)
affects:
  - scripts/__init__.py (new empty package marker — fixes brownfield sys.path mutation)
tech-stack:
  added:
    - "torch>=2.4,<3.0"
    - "transformers>=4.47,<5.0"
    - "accelerate>=0.34"
    - "peft>=0.11"
    - "bitsandbytes>=0.43"
    - "safetensors>=0.4.5"
    - "omegaconf>=2.3"
    - "tensorboard>=2.18"
    - "pyyaml>=6.0.2"
    - "tqdm>=4.66"
  patterns:
    - Pattern A — module docstring on every .py
    - Pattern D — uniform ValueError-subclass exception hierarchy (SampleContractError, LossWeightsSchemaError, SplitsContaminationError)
    - Pattern F — three-block import organisation (stdlib / third-party / first-party)
    - Pattern G — PEP 604/585 hints on public surfaces
    - Pattern H — json.dump(indent=2) on writes
key-files:
  created:
    - scripts/__init__.py
    - mapclass/__init__.py
    - mapclass/data/__init__.py
    - mapclass/model/__init__.py
    - mapclass/data/taxonomy.py
    - mapclass/seeding.py
    - mapclass/data/contract.py
    - mapclass/data/loss_weights.py
    - mapclass/data/splits.py
    - mapclass/data/dataset.py
    - mapclass/configs/loss_weights.yaml
    - mapclass/configs/v0.yaml
    - pyproject.toml
    - requirements.lock.txt
    - .python-version
  modified: []
decisions:
  - "Lockfile generated via `uv pip compile pyproject.toml --python-version 3.10 -o requirements.lock.txt` (uv was available; no fallback needed)"
  - "`mapclass/data/contract.py` hoisted `import numpy as np` to the top-level imports block (Pattern F three-block discipline) instead of the inline import shown in the plan example — purely stylistic, behavior-equivalent"
metrics:
  duration: ~25 minutes (active execution time, excluding the prior partial run that landed Task 1)
  completed: 2026-05-09
  tasks_completed: 3
  files_created: 15
hashes:
  taxonomy_hash: "052624d2022aa6650b4ef09a519d86c3d4c8eb4a0fe8d7cbc683f3947a489e22"
  source_class_weights_hash: "10f7e4dd4a3e7759152019721644939646f8b20f38dbde2385632b206c9c09ab"
  requirements_lock_sha256: "209400deb62176187184920d7fd36ab0af1976825a9414fe85cd62d9b2a8ce66"
---

# Phase 1 Plan 01: Data + Config + Packaging Foundation Summary

Walking-skeleton data layer landed: an importable `mapclass.data.*` API with strict on-disk
sample contract, hash-stamped per-source loss-weights validator, deterministic
sample-id-hash splits, torch Dataset adapter, plus `pyproject.toml` + uv-compiled
`requirements.lock.txt` + `.python-version=3.10` so `pip install -e .` from a fresh clone
of `refactor_paper` succeeds. No model code yet — that lands in Plan 02.

## Execution Notes

This plan resumed an interrupted prior run. Mapping commits → tasks:

| Task | Files | Status | Commit(s) |
| ---- | ----- | ------ | --------- |
| Task 1: package markers + taxonomy + seeding | `scripts/__init__.py`, `mapclass/__init__.py`, `mapclass/data/__init__.py`, `mapclass/model/__init__.py`, `mapclass/data/taxonomy.py`, `mapclass/seeding.py` | resumed (verified pre-existing) | `42503d8` (carried via merge `d3f3752`) |
| Task 2: pyproject.toml + lockfile + .python-version | `pyproject.toml`, `requirements.lock.txt`, `.python-version` | executed in this run | `8c682d7` |
| Task 3: contract + loss_weights + splits + dataset + configs | `mapclass/data/contract.py`, `mapclass/data/loss_weights.py`, `mapclass/data/splits.py`, `mapclass/data/dataset.py`, `mapclass/configs/loss_weights.yaml`, `mapclass/configs/v0.yaml` | executed in this run | `ac309b0` |

Task 1 artifacts were verified against the plan spec before proceeding: file sizes (all 4
markers are zero-byte), `taxonomy.py` re-imports (no redefinition, `grep -c 'LANDCOVER_CLASSES = \['` returned `0`),
and `seeding.py` pins all five RNGs + cudnn deterministic. The plan's Task 1 automated
verification command was re-run end-to-end and passed.

## Per-Task Outcomes

### Task 1 — Package markers, taxonomy re-export, seeding utility (`42503d8`, resumed)

- Four empty package marker files exist and are zero bytes: `scripts/__init__.py`,
  `mapclass/__init__.py`, `mapclass/data/__init__.py`, `mapclass/model/__init__.py`. No
  `__all__` in any of them (PATTERNS gotcha #7).
- `mapclass/data/taxonomy.py` re-exports `LANDCOVER_CLASSES`, `LANDCOVER_IDX`, `TOPO_CLASSES`,
  `TOPO_IDX`, `h_to_landcover`, `h_to_topo` from `scripts.biome_mapping`, plus
  `WATER_TOPO=255` and `NODATA=255` from `scripts.label`. No constants are redefined.
- `taxonomy_hash()` returns
  `052624d2022aa6650b4ef09a519d86c3d4c8eb4a0fe8d7cbc683f3947a489e22` (sha256 over
  `{"land_cover": LANDCOVER_CLASSES, "topography": TOPO_CLASSES}` with `sort_keys=True`).
- `mapclass/seeding.py` defines `set_global_seed(seed)` that pins random + numpy + torch CPU/CUDA
  + cudnn deterministic + benchmark-off + `PYTHONHASHSEED`. Verified post-call: `torch.backends.cudnn.deterministic == True`, `torch.backends.cudnn.benchmark == False`.

### Task 2 — `pyproject.toml` + `requirements.lock.txt` + `.python-version` (`8c682d7`)

- `pyproject.toml` declares `setuptools>=68` build, `requires-python = ">=3.10"`, all minimum-version
  pins per `research/STACK.md` ("Recommended Stack" minimums) plus the existing data-pipeline deps
  preserved from `requirements.txt`. No `[tool.black]` / `[tool.ruff]` / `[tool.mypy]` /
  `optional-dependencies` sections (PROJECT.md "Out of Scope (v1)" — linters deferred).
- `requirements.lock.txt` generated via `uv pip compile pyproject.toml --python-version 3.10 -o requirements.lock.txt`.
  Compiled targeting Python 3.10 even though the host runs 3.12 — the `.python-version` file is the
  binding contract. Lockfile contains exact pins for `torch==`, `transformers==`, `safetensors==`
  (sha256 = `209400deb62176187184920d7fd36ab0af1976825a9414fe85cd62d9b2a8ce66`).
- `.python-version` is one line containing exactly `3.10` (no comment, no patch version).
- `pip install --dry-run -e .` (no-deps) succeeds against the new pyproject.toml, confirming
  metadata is consumable. Closes PITFALL 19 + CONCERNS.md item 8c.

### Task 3 — Sample contract + loss-weights + splits + dataset + configs (`ac309b0`)

- `mapclass/data/contract.py` exports `SampleContractError(ValueError)` and `assert_sample_valid(path)`.
  Validates required files (`land_cover.png`, `topography.png`, `sample_weights.json`,
  `manifest.json`, plus one of the four image candidates), image-dim consistency across image
  + land_cover + topography, label values in `[0, n_classes) ∪ {255}` (honoring WATER_TOPO/NODATA
  sentinels), manifest required keys, sample_weights required keys. Two negative tests run inline
  during verification:
    - tampered land_cover (value=200, out of [0,9)) → `SampleContractError` raised ✓
    - missing manifest.json → `SampleContractError` with "manifest" in message ✓
- `mapclass/data/loss_weights.py` exports `LossWeights` (frozen dataclass) and
  `LossWeightsSchemaError(ValueError)`. `LossWeights.load(path)` parses YAML, validates that EVERY
  source row lists ALL 9 LANDCOVER_CLASSES (explicit floats, no implicit defaults), rejects unknown
  class names, requires `topography` ∈ `[0.0, 1.0]`. Computes
  `source_class_weights_hash = sha256(yaml_text)` =
  `10f7e4dd4a3e7759152019721644939646f8b20f38dbde2385632b206c9c09ab`. `lookup('rumsey_registered', 'trees')`
  returns `(0.3, 1.0)` matching `scripts/historical/label.py:HISTORICAL_LC_WEIGHTS`.
- `mapclass/data/splits.py` exports `_bucket_for_sample_id(id)` (sha256 first-32-bits → bucket 0..9,
  pure function), `build_splits(ids)` (returns `{version: 1, by_split: {train, val, test}, by_id_hash_bucket}`),
  and `SplitsContaminationError(ValueError)`. Bucket policy: 0–7 train, 8 val, 9 test (80/10/10).
  `write_splits` / `load_splits` round-trip JSON with `indent=2` (Pattern H).
- `mapclass/data/dataset.py` exports `MapClassDataset(torch.utils.data.Dataset)`. Calls
  `assert_sample_valid` once per sample at `__init__` (when `validate=True`, default), NOT per
  `__getitem__`. Returns `{image: (3, H, W) float [0,1], land_cover: (H, W) int64,
  topography: (H, W) int64, source_subtype: str, sample_id: str}`.
- `mapclass/configs/loss_weights.yaml` contains `synthetic_azgaar` and `rumsey_registered` rows.
  Both rows enumerate all 9 land-cover classes with explicit floats; `rumsey_registered.land_cover`
  values are verbatim from `scripts/historical/label.py:HISTORICAL_LC_WEIGHTS` (water=1.0,
  trees=0.3, ..., snow_ice=1.0). Synthetic uses 0.0 for absent classes (cropland, built_up,
  flooded_wetland) — explicit, not implicit.
- `mapclass/configs/v0.yaml` is the Phase 1 Mock-backbone smoke-run config: `seed: 42`,
  `model.backbone: mock`, `data.num_samples: 100`, `data.image_size: 384` (matches SmolVLM input
  for trivial Phase-2 swap), `data.splits_path: mapclass/configs/splits.json` (forward-pointing —
  splits.json itself is generated by Plan 04's smoke run, not this plan).

## Verification

### Per-task automated checks

All three tasks' `<verify><automated>` blocks executed and exited 0:

```
$ python -c "from mapclass.data.taxonomy import LANDCOVER_CLASSES, ..., taxonomy_hash; ..."
052624d2022aa6650b4ef09a519d86c3d4c8eb4a0fe8d7cbc683f3947a489e22
$ python -c "from mapclass.seeding import set_global_seed; set_global_seed(42); ..."
seeded
$ test -f pyproject.toml && grep -q 'transformers>=4\.47' pyproject.toml && ... && python -c "import tomllib; ..."
pyproject ok
$ python -c "from mapclass.data.contract import ..."
contract ok
$ python -c "from mapclass.data.loss_weights import ..."
loss_weights ok hash=10f7e4dd4a3e
$ python -c "from mapclass.data.splits import ..."
splits ok
$ python -c "from mapclass.data.dataset import MapClassDataset"
dataset import ok
$ python -c "import yaml; cfg = yaml.safe_load(open('mapclass/configs/v0.yaml')); ..."
v0 config ok
```

### Plan-level verification (`<verification>` block)

1. `pip install --dry-run -e .` exits 0 against the new `pyproject.toml` (metadata consumable). ✓
2. `python -c "import mapclass; import mapclass.data.taxonomy; import mapclass.data.contract; import mapclass.data.loss_weights; import mapclass.data.splits; import mapclass.data.dataset; import mapclass.seeding"` exits 0. ✓
3. `LossWeights.load('mapclass/configs/loss_weights.yaml')` parses + validates strict schema + returns 64-char hash. ✓
4. `taxonomy_hash() != source_class_weights_hash` and both are 64 chars. ✓

### Negative tests (tamper rejection — Task 3 acceptance)

Both inline negative tests pass:
- Tampered `land_cover.png` with value 200 (out of `[0, 9)`) → `SampleContractError` raised ✓
- Missing `manifest.json` → `SampleContractError` with "manifest" in error message ✓

## Decisions Made

- **Lockfile tooling: `uv pip compile`.** `uv` was available on the host (`/home/drdreadknee/.local/bin/uv`); the plan's listed fallback (`pip-compile` or `pip freeze`) was not needed. Compiled targeting Python 3.10 (`--python-version 3.10`) so the lockfile binds to the version pinned in `.python-version`, not the host's Python 3.12.
- **`import numpy as np` placement in `contract.py`.** Hoisted to the top-level imports block (matching Pattern F three-block discipline and `dataset.py`'s top-level numpy import) instead of the inline import shown in the plan's code example. Stylistic, behavior-equivalent.
- **`mapclass/configs/splits.json` deferred.** Per the plan note, `splits.json` itself is generated at smoke-run time (Plan 04) from discovered sample IDs via `mapclass.data.splits.build_splits()`. This plan ships the builder but not the artifact.

## Deviations from Plan

None functionally. Two minor stylistic notes (already in Decisions):
- Lockfile generated via `uv` (the plan's preferred path; only documented because the plan listed fallbacks).
- `import numpy as np` hoisted to top-level in `contract.py` (matches Pattern F and `dataset.py` style).

No auto-fixes (Rules 1–3) triggered. No checkpoints encountered. No deferred items.

## Hashes for Downstream Plans

- `taxonomy_hash`: `052624d2022aa6650b4ef09a519d86c3d4c8eb4a0fe8d7cbc683f3947a489e22`
- `source_class_weights_hash` (sha256 of `mapclass/configs/loss_weights.yaml`): `10f7e4dd4a3e7759152019721644939646f8b20f38dbde2385632b206c9c09ab`
- `requirements.lock.txt` sha256: `209400deb62176187184920d7fd36ab0af1976825a9414fe85cd62d9b2a8ce66`

Plan 02's `mapclass.train` will embed `taxonomy_hash` and `source_class_weights_hash` in the
safetensors checkpoint metadata (TS-2, TS-8). The lockfile sha256 is captured here so a future
Plan 02 executor can confirm install reproducibility against this exact lockfile content.

## Threat Flags

None. All new on-disk surface (the two YAML configs and the contract/loss-weights validators) was
already enumerated in the plan's `<threat_model>` (T-01-01 through T-01-06) and mitigated as
specified. No new endpoints, auth paths, network calls, or schema changes at trust boundaries
beyond what the plan anticipated.

## Known Stubs

None. All modules ship with full functionality matching their plan specs. `mapclass/configs/splits.json`
is intentionally absent (deferred to Plan 04 smoke run, as noted in the plan's Task 3 action) — this
is documented in `mapclass/configs/v0.yaml` via the forward-pointing `splits_path` and is not a stub.

## Self-Check: PASSED

Files verified to exist:
- FOUND: scripts/__init__.py
- FOUND: mapclass/__init__.py
- FOUND: mapclass/data/__init__.py
- FOUND: mapclass/model/__init__.py
- FOUND: mapclass/data/taxonomy.py
- FOUND: mapclass/seeding.py
- FOUND: mapclass/data/contract.py
- FOUND: mapclass/data/loss_weights.py
- FOUND: mapclass/data/splits.py
- FOUND: mapclass/data/dataset.py
- FOUND: mapclass/configs/loss_weights.yaml
- FOUND: mapclass/configs/v0.yaml
- FOUND: pyproject.toml
- FOUND: requirements.lock.txt
- FOUND: .python-version

Commits verified to exist:
- FOUND: 42503d8 (Task 1, resumed from prior partial run)
- FOUND: 8c682d7 (Task 2)
- FOUND: ac309b0 (Task 3)
