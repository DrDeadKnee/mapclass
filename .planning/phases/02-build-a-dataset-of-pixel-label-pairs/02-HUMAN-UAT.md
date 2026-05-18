# 02-HUMAN-UAT: Azgaar Regeneration + Manifest + GCS Upload Gate

**Status:** BLOCKING — Phase 2 pipeline is fully implemented (offline-verified,
Waves 0-3 complete). The end-to-end build CANNOT run until the user re-creates
the lost Azgaar source maps and uploads them to the canonical bucket.

**Accepted consequence:** The new EVAL-01 hold-out WILL differ from the original
Phase 4 plans. This is a logged, accepted consequence: the original synthetic
dataset + raw .geojson inputs were destroyed by ephemeral compute preemption and
are unrecoverable (see `02-CONTEXT.md` for full rework rationale).

---

## Pre-conditions

**ADC authentication:** On the build host, run:
```
gcloud auth application-default login
```
This is the same ADC requirement as the existing `gcs_checkpoint.py` path — no new
secret or service account is needed.

---

## Step-by-Step Gate

### Step 1: Re-create ~100 Azgaar maps

In the [Azgaar Fantasy Map Generator](https://azgaar.github.io/Fantasy-Map-Generator/):

- Target: **N=100 Azgaar source maps** across ~12 continent templates (≈15–16 per
  template), e.g. `europe`, `africa`, `northamerica`, `southamerica`, `asia`,
  `oceania`, etc.
- Each map MUST be exported as **GeoJSON**, not SVG or PNG.
- Use varied seeds and settings to create geographic diversity within each template.

### Step 2: Name each file correctly (RW-02 convention)

Filename must match: `^[a-z][a-z0-9_]*_[0-9]{2}\.geojson$`

Examples:
```
europe_01.geojson
europe_02.geojson
...
europe_16.geojson
africa_01.geojson
...
northamerica_12.geojson
```

Rules:
- Lowercase only.
- Template name (part before `_NN`) must be consistent and alphanumeric + underscores.
- `NN` is zero-padded two-digit counter per template (01, 02, …).
- **No spaces**, no uppercase, no special characters other than `_`.

### Step 3: Author `raw/manifest.json`

Create a JSON file named `manifest.json` with this schema:
```json
{
  "version": "1",
  "entries": {
    "europe_01.geojson": {"template": "europe"},
    "europe_02.geojson": {"template": "europe"},
    "africa_01.geojson": {"template": "africa"},
    ...
  }
}
```

Rules:
- **Every** uploaded .geojson file MUST have an entry.
- The `template` value MUST equal the part of the filename before `_NN`
  (e.g. `europe_01.geojson` → `"template": "europe"`).
- No phantom entries (entries whose file is not uploaded will cause a hard fail).

`validate_manifest` in `build_dataset.py` performs a 4-way cross-check and calls
`sys.exit(1)` with a `FATAL: manifest validation failed` message on ANY mismatch.

### Step 4: Upload all files to GCS

```bash
gsutil -m cp *.geojson manifest.json \
  gs://mapclass-training-northeast1/data/synthetic/raw/
```

Or using the Google Cloud console browser upload.

Target prefix: `gs://mapclass-training-northeast1/data/synthetic/raw/`

### Step 5: Verify ADC on the build host

```bash
gcloud auth application-default login
gcloud config set project narrative-campaign
```

### Step 6: Run the synthetic build

```bash
python scripts/build_dataset.py build
```

(Defaults: `--raw-dir gs://mapclass-training-northeast1/data/synthetic/raw`
and `--out-dir gs://mapclass-training-northeast1/data/synthetic`)

**Expected outcomes:**

- **CLEAN run:** The build streams pyramid objects to GCS for each
  `<azgaar_id>__<style>/` directory, writes `_BUILD_COMPLETE` sentinels per
  map-dir, freezes `gs://.../data/synthetic/split.json` (first build only),
  and exits 0.
  
- **LOUD failure:** If any filename/manifest mismatch is detected,
  `FATAL: manifest validation failed — aborting before split is computed (RW-02)`
  is printed and the process exits 1. Fix the mismatch (rename files or update
  `manifest.json`) and re-run — the split.json is NOT written until validation passes.

- **Note:** The build is preemption-safe. If interrupted, re-run the same command.
  Maps with a `_BUILD_COMPLETE` sentinel are skipped; partial map-dirs (sentinel
  absent) are rebuilt from scratch (OQ1 / Pitfall R-2).

### Step 7: Confirm non-empty output

Run this one-liner to verify `train/` is non-empty and `split.json` exists:

```bash
python -c "
import sys; sys.path.insert(0, 'scripts')
import gcsfs, gcs_io
fs = gcsfs.GCSFileSystem(project=gcs_io.GCS_PROJECT)
train_count = len(fs.ls(gcs_io.DATA_PREFIX + '/synthetic/train'))
split_exists = fs.exists(gcs_io.DATA_PREFIX + '/synthetic/split.json')
print(f'train objects: {train_count}')
print(f'split.json exists: {split_exists}')
assert train_count > 0, 'FAIL: train/ is empty'
assert split_exists, 'FAIL: split.json not found'
print('PASS: non-empty bucket + split.json confirmed')
"
```

---

## Resume Signal

Type **"approved"** once the build completed cleanly and
`gs://mapclass-training-northeast1/data/synthetic/{train,test}/` and `split.json`
are populated. Alternatively, describe the manifest/upload error to fix.

---

## Downstream Impact

Once this gate is complete:
- Phase 4 (`finetune_seg.py`) can pull the train subset via `--scratch-dir` (RW-03).
- Phase 4 (`evaluate_seg.py`) can pull the synthetic test subset via `--scratch-dir` (RW-03).
- The EVAL-01 hold-out is frozen in the new `split.json` (differs from the original
  lost split — this is accepted and logged; see `02-CONTEXT.md`).
