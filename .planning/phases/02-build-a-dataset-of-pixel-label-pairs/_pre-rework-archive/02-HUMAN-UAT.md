---
status: complete
phase: 02-build-a-dataset-of-pixel-label-pairs
source: [02-VERIFICATION.md]
started: 2026-05-15T18:40:00Z
updated: 2026-05-16T00:00:00Z
verdict: passed
---

## Current Test

[testing complete]

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
result: pass — networked-host run 2026-05-16. test_rumsey_online,
test_stac_online, test_satellite_online (window fetch + end-to-end region)
all passed on the first run. test_allmaps_online and test_iiif_online
initially failed on a stale hardcoded David Rumsey fixture
(RUMSEY~8~1~24694~890095, retired upstream — manifest now returns
{"error":"Invalid Identifier"}); fixtures repointed to a live
Allmaps-georeferenced record (RUMSEY~8~1~292315~90066993, 19 GCPs,
11651×14997 source) in commit dc4f585, after which both passed. All 6
integration tests green. Pipeline/Allmaps/IIIF/LUNA/STAC/S3 paths were
healthy throughout — the only defect was the test fixture.

## Summary

total: 1
passed: 1
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
