---
phase: 02
slug: build-a-dataset-of-pixel-label-pairs
status: accepted_risks
threats_open: 0
asvs_level: 1
created: 2026-05-15
---

# Phase 2 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
>
> **Audit mode:** All 17 plan-time threats were **accepted as documented risks by
> explicit user decision (2026-05-15)** without spawning the code-verification
> auditor. The 13 `mitigate` threats therefore have plan-time mitigation *plans*
> on record but were **NOT independently verified to exist in the shipped code**
> by this run. `threats_open: 0` reflects that every threat has a disposition,
> NOT that mitigations were code-verified. A future `/gsd-secure-phase 2` (choosing
> "Verify all open threats") can upgrade the `mitigate` rows from accepted to
> code-verified-closed.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Allmaps annotations API → parser | Untrusted third-party JSON into `_parse_annotation` | Annotation JSON (untrusted) |
| LUNA / Allmaps / IIIF servers → pipeline | Untrusted third-party HTTP (JSON + binary image) | Search JSON, IIIF JPEG bytes (untrusted) |
| Remote JPEG/COG → Pillow/rasterio decode | Untrusted image/raster bytes parsed locally | Image/GeoTIFF bytes (untrusted) |
| Element84 STAC API → pipeline | Untrusted third-party STAC item JSON | STAC items (untrusted) |
| s3://sentinel-cogs / esa-worldcover / copernicus-dem-30m → rasterio | Anonymous public-S3 remote COG/GeoTIFF parsed locally | Public open-data rasters (untrusted bytes, non-secret) |
| Externally-derived ids → output filenames/dirs | LUNA/Allmaps/Azgaar/region ids used in filesystem paths | Path components (untrusted strings) |
| Synthetic train/test boundary → tiler output | The leakage-critical boundary EVAL-01 depends on | Split assignment (integrity-critical) |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation (plan-time) | Status |
|-----------|----------|-----------|-------------|------------------------|--------|
| T-02-01 | DoS (self) | `allmaps.lookup()` parsing malformed/huge third-party `items` | mitigate | Per-item `_parse_annotation` try/except → None; loop drops None | closed (accepted, unverified) |
| T-02-02 | Tampering | Annotation/LUNA id → downstream filename | mitigate | Non-`[\w-]` → `_` before path join | closed (accepted, unverified) |
| T-02-03 | Info Disclosure | Test fixtures committing real API tokens | accept | No auth surface — anonymous APIs; synthetic fixtures | closed (accepted) |
| T-02-04 | DoS | Oversized IIIF response exhausting memory/disk | mitigate | `Content-Length` check; abort > 200 MB before decode | closed (accepted, unverified) |
| T-02-05 | DoS (self) | Malformed Allmaps/LUNA JSON crashing the batch | mitigate | Per-item drop + worker returns `download_failed` | closed (accepted, unverified) |
| T-02-06 | Tampering | Decompression-bomb JPEG from hostile IIIF server | accept | Curated David Rumsey surface; Content-Length ceiling; Pillow MAX_IMAGE_PIXELS active | closed (accepted) |
| T-02-07 | Tampering | EVAL-01 leakage — held-out tiles reaching train/ | mitigate | Filesystem split (D-17); `test_no_train_test_intersection` + `test_split_manifest_frozen` | closed (accepted, unverified) |
| T-02-08 | Tampering | Azgaar id stem → split dir path | mitigate | Sanitize non-`[\w-]` → `_` before path join | closed (accepted, unverified) |
| T-02-09 | DoS (self) | Malformed Azgaar GeoJSON crashing the batch | mitigate | Per-source try/except returns failed; structural per-feature/ring resilience (CR-01/CR-03/CR-04 + non-dict fix `d72c43b`) | closed (accepted, unverified) |
| T-02-10 | DoS | Downloading a full Sentinel-2 scene vs. 4096-px window | mitigate | rasterio Window byte-range reads; acceptance grep forbids windowless read | closed (accepted, unverified) |
| T-02-11 | DoS (self) | Malformed STAC item / missing `visual` asset crashing batch | mitigate | StacLookupError + per-item try/except → fetch_failed | closed (accepted, unverified) |
| T-02-12 | Tampering | Region id → path traversal | mitigate | Sanitize non-`[\w-]` → `_` before path join | closed (accepted, unverified) |
| T-02-13 | Spoofing | Anonymous S3/STAC over plain HTTP MITM | accept | All endpoints HTTPS; public open-data, no integrity-critical secrets | closed (accepted) |
| T-02-14 | DoS | Malformed/decompression-bomb remote COG | accept | Curated AWS open-data registries; windowed reads bound decoded extent | closed (accepted) |
| T-02-15 | Tampering | Tiler writing held-out source pyramids into train/ (EVAL-01) | mitigate | Tiler writes into same train/ or test/ subtree; `test_split_subtree_preserved` | closed (accepted, unverified) |
| T-02-16 | DoS (self) | Partially-failed per-map dir → corrupt pyramid tree | mitigate | Pre-tiling guard skips + logs map dir missing a required file | closed (accepted, unverified) |
| T-02-17 | Tampering | Pyramid id collisions overwriting tiles | mitigate | Deterministic grid row/col ids; per-pyramid subdir isolation | closed (accepted, unverified) |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-02-03 | No auth surface — Allmaps/LUNA/STAC anonymous; fixtures synthetic | user | 2026-05-15 |
| AR-02 | T-02-06 | Curated David Rumsey IIIF; Content-Length ceiling; Pillow bomb guard active | user | 2026-05-15 |
| AR-03 | T-02-13 | All endpoints HTTPS; public open-data, no integrity-critical secrets | user | 2026-05-15 |
| AR-04 | T-02-14 | Curated AWS open-data registries; windowed reads bound decoded extent | user | 2026-05-15 |
| AR-05 | T-02-01,02,04,05,07,08,09,10,11,12,15,16,17 | The 13 `mitigate` threats accepted **without code verification** by explicit user decision — plan-time mitigation plans on record; not independently confirmed present in shipped code this run. Re-run `/gsd-secure-phase 2` → "Verify all open threats" to upgrade to code-verified-closed. | user | 2026-05-15 |

*Accepted risks do not resurface in future audit runs.*

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-15 | 17 | 17 (4 accept-by-rationale + 13 accepted-unverified) | 0 | user decision (no auditor agent spawned) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed (every threat has a disposition)
- [ ] `status: verified` — NOT set: 13 `mitigate` threats accepted unverified, not code-verified

**Approval:** accepted-risks 2026-05-15 (user) — code verification deferred
