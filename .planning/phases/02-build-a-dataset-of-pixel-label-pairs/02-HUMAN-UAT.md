---
status: partial
phase: 02-build-a-dataset-of-pixel-label-pairs
source: [02-VERIFICATION.md]
started: 2026-05-15T18:40:00Z
updated: 2026-05-15T18:40:00Z
---

## Current Test

[awaiting human testing on a network-enabled host]

## Tests

### 1. Online integration phase gate

expected: On a networked host, `pytest tests/integration -m integration`
passes all 6 tests — test_allmaps_online (≥1 annotation, ≥3 GCPs),
test_iiif_online (both dims ≤4096, actual w,h returned), test_rumsey_online
(search_maps 1500–1700 returns ≥1), test_stac_online (known bbox returns
results), test_satellite_online (window fetch shape (3,4096,4096) +
end-to-end region produces 4 output files). Exercises Wave-1/2 network I/O
(Allmaps / IIIF / David Rumsey LUNA / Sentinel-2 STAC / s3://sentinel-cogs)
that cannot be verified offline or by grep.
result: [pending — sandbox has no network; deferred to networked-host run by user decision 2026-05-15]

## Summary

total: 1
passed: 0
issues: 0
pending: 1
skipped: 0
blocked: 0

## Gaps
