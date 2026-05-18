# Phase 02 — Deferred / Out-of-Scope Items

## From plan 02-05 (nested-pyramid tiler)

- **Integration phase gate (`pytest tests/integration -m integration`) not runnable
  in this executor's sandbox.** All 6 integration tests are *online* network tests
  (`test_allmaps_online`, `test_iiif_online`, `test_rumsey_online`,
  `test_satellite_online`, `test_stac_online`) hitting David Rumsey LUNA, the
  Allmaps annotation server, IIIF image endpoints, the Sentinel-2 STAC catalogue,
  and `s3://sentinel-cogs`. With no network they hang until SIGTERM (exit 143).
  These exercise the Wave-1/2 fetch/search code, NOT the 02-05 tiler (which is
  pure offline geometry and modifies no network code path). Out of scope for this
  plan per the executor scope boundary — flagged here for the verifier to run the
  online phase gate in a network-enabled environment.
