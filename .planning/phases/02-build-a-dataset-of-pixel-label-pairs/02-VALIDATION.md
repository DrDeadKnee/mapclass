---
phase: 02
slug: build-a-dataset-of-pixel-label-pairs
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-16
---

# Phase 02 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> **REWORK scope:** rework-introduced failure modes (GCS-canonical persistence).
> Unchanged-domain tests (tiler geometry, split determinism, Allmaps/STAC) are
> documented in the 2026-05-15 research and remain in the Wave 0 gap list.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest >= 8 (installed in Wave 0) |
| **Config file** | none — Wave 0 installs |
| **Quick run command** | `pytest tests/ -x --ignore=tests/integration` |
| **Full suite command** | `pytest tests/` |
| **Estimated runtime** | ~60 seconds (offline; mock fsspec/GCS) |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/ -x --ignore=tests/integration`
- **After every plan wave:** Run `pytest tests/`
- **Before `/gsd:verify-work`:** Full suite must be green
- **Max feedback latency:** 60 seconds

---

## Per-Task Verification Map

> Mapped from RESEARCH.md "Rework Requirements → Test Map". Plan/task IDs are
> assigned by the planner — Threat Ref filled by /gsd:secure-phase.

| Req ID | Wave | Behavior | Test Type | Automated Command | File Exists |
|--------|------|----------|-----------|-------------------|-------------|
| RW-01 | 0 | `_GCSWriter` writes bytes to a mock fsspec local fs | unit | `pytest tests/test_gcs_io.py::test_gcs_writer_writes_bytes -x` | ❌ W0 |
| RW-01 | 0 | `tiling.tile(..., out_root=_GCSWriter)` produces correct pyramid | integration | `pytest tests/test_tiling.py::test_tile_to_gcs_writer -x` | ❌ W0 |
| RW-01 | 0 | `_GCSWriter` 'wb' accepts Pillow `.save(fh, format='PNG')` | unit | `pytest tests/test_gcs_io.py::test_gcs_writer_pillow_compat -x` | ❌ W0 |
| RW-01b | 0 | Thread-pool tile writes do not corrupt pyramid | unit | `pytest tests/test_gcs_io.py::test_concurrent_tile_writes -x` | ❌ W0 |
| RW-02 | 0 | `validate_manifest` exits 1 on unlisted/phantom/regex/template mismatch | unit | `pytest tests/test_manifest.py::test_validate_manifest_hard_fails -x` | ❌ W0 |
| RW-02 | 0 | `validate_manifest` runs BEFORE `load_or_create_split` | integration | `pytest tests/test_build_dataset.py::test_manifest_fails_before_split -x` | ❌ W0 |
| RW-02 | 0 | `stratified_split` uses manifest template, not `template_key()` | unit | `pytest tests/test_build_dataset.py::test_split_uses_manifest_template -x` | ❌ W0 |
| RW-03 | 0 | `pull_dataset_from_gcs` is idempotent | unit | `pytest tests/test_gcs_io.py::test_pull_idempotent -x` | ❌ W0 |
| RW-03 | 0 | `verify_pull` raises on missing map_id directory | unit | `pytest tests/test_gcs_io.py::test_verify_raises_on_missing_map -x` | ❌ W0 |
| RW-03 | 0 | `verify_pull` raises on truncated pyramid | unit | `pytest tests/test_gcs_io.py::test_verify_raises_on_truncated_pyramid -x` | ❌ W0 |
| EVAL-01 | 0 | `split.json` frozen in GCS; re-run does not recompute | integration | `pytest tests/test_build_dataset.py::test_split_frozen_in_gcs -x` | ❌ W0 |
| RW-04 | 0 | `build_historical_dataset.py` writes `unregistered_manifest.json` to GCS | integration | `pytest tests/test_historical.py::test_manifest_written_to_gcs -x` | ❌ W0 |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_gcs_io.py` — `_GCSWriter` unit tests (write_bytes, write_text, open, Pillow compat, concurrent, idempotent pull, verify_pull)
- [ ] `tests/test_manifest.py` — `validate_manifest` hard-fail coverage (all four error cases)
- [ ] `tests/test_build_dataset.py` — manifest-before-split ordering, manifest-driven stratification, frozen-split-in-GCS
- [ ] `tests/test_tiling.py` — `tile()` to `_GCSWriter` pyramid structure
- [ ] `tests/test_historical.py` — `unregistered_manifest.json` written to GCS
- [ ] `tests/conftest.py` — shared mock fsspec/local-filesystem fixtures
- [ ] pytest >= 8 install (no framework currently present)

*Plus the prior 2026-05-15 unchanged-domain Wave 0 gaps (tiler geometry, split determinism, Allmaps/STAC) — carry forward.*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| ~100 Azgaar maps re-created across ~12 templates, named per RW-02, `raw/manifest.json` authored, uploaded to `gs://…/data/synthetic/raw/` | PHASE-02 | External human authoring in Azgaar Fantasy Map Generator — not automatable | GPU-host gate (02-HUMAN-UAT.md): user regenerates maps, authors manifest, uploads; build then hard-fails or proceeds |
| Live GCS write/throughput on the GPU host with real ADC | RW-01b | Requires real bucket + same-region compute; offline suite uses mock fs | Run build on GPU host; observe wall-clock + GCS object count vs the ~$6 / ~65 min estimate |

---

## Observability Points (Nyquist sampling for rework failure modes)

| Failure Mode | Detection Point | How to Detect |
|-------------|----------------|---------------|
| Silent local-only write | Post-build GCS ls | Assert `fs.ls(gs://…/data/synthetic/train/)` non-empty immediately after build |
| Partial upload mid-pyramid | Pre-train verification | `verify_pull` spot-checks pyramid file count |
| Split frozen on incomplete raw | Manifest validation | RW-02 hard-fail; `--refreeze-split` is the deliberate escape hatch |
| Manifest mismatch not hard-failing | Build test suite | `test_manifest_fails_before_split` confirms ordering |
| Pull-once false-pass | Training loss spike | Epoch-1 val loss vs probe baseline; pyramid.json parse errors in DataLoader |

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 60s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
