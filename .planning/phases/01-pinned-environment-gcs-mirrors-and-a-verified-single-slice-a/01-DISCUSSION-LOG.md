# Phase 1: Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-18
**Phase:** 1-Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution
**Areas discussed:** Ingestion scope, Phase 1 gate rigor, Verified slice (map + query), dynamicLRP vendoring, Slice query strings, Ingest ordering

---

## Ingestion Scope

| Option | Description | Selected |
|--------|-------------|----------|
| Bounded high-index subset | Mirror only top-N by manifest index (~200). Fast, polite to Rumsey host. | |
| Full 1,544 manifest | Mirror the entire manifest now. Slowest, exercises Pitfall 6 at real scale. | ✓ |
| Tiny smoke set + bounded subset | ~5 ids to validate pipeline, then ~200. | |

**User's choice:** Full 1,544 manifest ("all of the rumsey images should be in gcp")
**Notes:** Phase 2 sweep then has zero ingest dependency; outcome manifest complete at gate time.

---

## Phase 1 Gate Rigor

| Option | Description | Selected |
|--------|-------------|----------|
| Printed cheap scalars + eyeball | Print a number per control + show overlays; gate = numbers cross written threshold. | |
| Pure visual eyeball | Display overlays side-by-side; judge pass/fail purely by looking. | ✓ |
| Scalars as hard CI-style asserts | Controls become asserts with fixed thresholds that fail the run. | |

**User's choice:** Pure visual eyeball
**Notes:** Conflicts with PITFALLS.md Pitfall 4 (eyeball is weakest defense against silent-wrong attribution on un-benchmarked SigLIP-2-as-contrastive). Tradeoff was surfaced in the option description and explicitly accepted → recorded as accepted risk D-04. The three controls themselves remain mandatory; only the judgment modality is visual.

---

## Verified Slice (map + query)

| Option | Description | Selected |
|--------|-------------|----------|
| You pick a landmark map + I choose query | User names a manifest id with a sharp known feature. | |
| Highest-index map, generic query | Use highest manifest index; generic query; zero user input. | ✓ |
| I propose 2-3 candidates, you confirm | Claude proposes triples, user picks. | |

**User's choice:** Highest-index map, generic query
**Notes:** No guaranteed landmark → Pitfall-3 alignment check is best-effort; explicit patch-grid overlay still required to demonstrate 27×27 / 6-px / un-squash correctness.

---

## dynamicLRP Vendoring

| Option | Description | Selected |
|--------|-------------|----------|
| Git submodule pinned to SHA | Submodule at fixed commit; sys.path. | |
| Vendored copy + recorded SHA | Copy src/lrp_engine/ into repo, record source SHA. | ✓ |
| Clone-at-setup pinned to SHA | Setup step git clone + checkout SHA into external/. | |

**User's choice:** Vendored copy + recorded SHA
**Notes:** Self-contained on ephemeral VM rebuilds; reproducibility carried by recorded SHA; updates are deliberate manual re-vendor.

---

## Slice Query Strings

| Option | Description | Selected |
|--------|-------------|----------|
| "a mountain range" / "a xylophone" | Control + unrelated swap. | |
| "a river" / "a xylophone" | Control 'a river' (near-ubiquitous on historical maps) + swap 'a xylophone'. | ✓ |
| "a coastline" / "a piano keyboard" | Control 'a coastline' + swap 'a piano keyboard'. | |

**User's choice:** "a river" / "a xylophone"
**Notes:** Strong expected signal for the control query; clearly unrelated swap for a strong query-swap contrast.

---

## Ingest Ordering

| Option | Description | Selected |
|--------|-------------|----------|
| Independent — slice unblocks early | Slice work proceeds once the one image is mirrored; full ingest in background. | |
| Full ingest gates everything | Phase 1 done requires full manifest mirrored AND slice verified. | ✓ |

**User's choice:** Full ingest gates everything
**Notes:** Plan ingestion as a long-running resumable step the phase must complete; not a background race.

---

## Claude's Discretion

- Internal module layout / filenames / signatures (within research-prescribed package-of-modules + thin-notebook architecture).
- Ingestion concurrency / backoff / validation mechanics (bounded by PITFALLS.md Pitfall 6).
- Build sequencing within the phase (follow research-prescribed order).

## Deferred Ideas

None — discussion stayed within phase scope. Quantitative/printed control metrics were offered and explicitly declined (recorded as accepted risk D-04, not deferred). Genuine faithfulness metrics remain v2 / out of scope per REQUIREMENTS.md.
