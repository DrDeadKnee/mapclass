# Phase 1: Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution - Context

**Gathered:** 2026-05-18
**Status:** Ready for planning

<domain>
## Phase Boundary

Stand up the pinned toolchain (mirroring dynamicLRP's `requirements.txt`) and
idempotent, validated GCS mirrors (Rumsey images + SigLIP-2 weights), then build
and **prove correct** the single-slice attribution primitive: one map + one
query → a properly-aligned heatmap overlaid on the source map, displayed inline
in JupyterLab, that has passed query-swap, model-randomization, and occlusion
sanity controls.

Covers requirements ENV-01, DATA-01, DATA-02, DATA-03, DATA-04, MODEL-01,
ATTR-01, ATTR-02, ATTR-03, VIZ-01.

**Not in this phase:** the configurable maps × queries sweep, run caching, and
contact-sheet browse (Phase 2 — pure orchestration over this verified
primitive).

</domain>

<decisions>
## Implementation Decisions

### Ingestion Scope & Ordering
- **D-01:** Mirror the **full 1,544-entry Rumsey manifest** into
  `gs://mapclass-training-northeast1/data/` — not a bounded subset. The
  idempotent / validated / per-id-outcome-manifest robustness (PITFALLS.md
  Pitfall 6) is exercised at full scale.
- **D-02:** **Full ingest gates the phase.** Phase 1 is not "done" until the
  entire manifest is mirrored (with its outcome manifest complete) *and* the
  single slice is verified. Phase 2's sweep then has zero ingest dependency.
  Plan ingestion as a long-running, resumable step the phase must complete, not
  a background task the slice work races ahead of.

### Phase 1 Correctness Gate
- **D-03:** The three sanity controls (query-swap, model-randomization,
  occlusion) are judged by **pure visual eyeball** — overlays displayed
  side-by-side, human judges pass/fail by looking. No printed scalars, no
  asserts.
- **D-04 (ACCEPTED RISK):** This conflicts with PITFALLS.md Pitfall 4, which
  argues pure-eyeball judgment is the weakest defense against silent-wrong
  attribution on SigLIP-2-as-contrastive-encoder (faithfulness is unvalidated
  in the paper). The user explicitly accepted this tradeoff. **The three
  controls themselves remain mandatory** — only the judgment modality is
  visual. Downstream agents must still implement all three controls and present
  them clearly for visual comparison; they must NOT silently add quantitative
  thresholds, but SHOULD lay the controls out to make a wrong result visually
  obvious (e.g. query-swap heatmaps directly adjacent, randomized-weights
  overlay next to trained, occluded-vs-random side by side).

### Verified Single Slice
- **D-05:** The slice is the **highest manifest index** map (richest per the
  index semantics in PROJECT.md) — chosen programmatically, no per-id input
  from the user.
- **D-06:** Locked control query: **`"a river"`** (near-ubiquitous on
  historical maps → strong expected signal). Locked query-swap query:
  **`"a xylophone"`** (clearly unrelated → strong query-swap contrast).
- **D-07:** No known sharp landmark is guaranteed on the highest-index map, so
  the Pitfall-3 patch-grid alignment check cannot rely on a hand-picked
  landmark. The overlay/alignment correctness (27×27 grid, 6-px edge discard,
  aspect un-squash) must still be demonstrated explicitly — e.g. overlay the
  raw patch grid on the slice map and confirm cell placement and right/bottom
  6-px exclusion — but visual landmark alignment is best-effort given the
  programmatic map choice.

### dynamicLRP Vendoring
- **D-08:** **Vendor a copy** of `keeinlev/dynamicLRP`'s `src/lrp_engine/`
  into the repo and **record the source commit SHA** in a tracked file and in
  run metadata. Not a git submodule, not clone-at-setup. Rationale:
  self-contained on ephemeral VM rebuilds (no clone/init step, no GitHub
  reachability dependency), reproducibility carried by the recorded SHA.
  Updating the engine is a deliberate manual re-vendor.

### Claude's Discretion
- Internal module layout under the package-of-modules + thin-notebook
  architecture (research-prescribed in SUMMARY.md §Architecture) — follow that
  shape; specific filenames/signatures are planner/researcher territory.
- Ingestion concurrency, backoff, validation mechanics — bounded by PITFALLS.md
  Pitfall 6; exact implementation is Claude's.
- Build sequencing within the phase — research prescribes
  env → mirrors → loaders → de-risk spike (reproduce `ViT.ipynb`, then swap
  SigLIP-2 + similarity target) → overlay → controls. Follow it.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project scope & requirements
- `.planning/PROJECT.md` — core value, out-of-scope boundaries, key decisions
- `.planning/REQUIREMENTS.md` — ENV-01 / DATA-01..04 / MODEL-01 / ATTR-01..03 /
  VIZ-01 exact requirement text and Phase 1 finish line
- `.planning/ROADMAP.md` §"Phase 1" — goal and 5 success criteria (the
  goal-backward check target)

### Attribution correctness (the central technical risk)
- `.planning/research/PITFALLS.md` — all 8 pitfalls; **Pitfalls 1–5, 7–8 are
  in-scope for this phase**. Pitfall 1 (contrastive target =
  `logits_per_image[0,0]`, detached text, keep L2-norm in graph), Pitfall 2
  (extract relevance at pre-MAP-pool patch tokens), Pitfall 3 (27×27 / 6-px
  discard / aspect un-squash), Pitfall 4 (eyeball-gate risk — see D-04),
  Pitfall 7 (vendoring/provenance — see D-08), Pitfall 8 (official SigLIP-2
  processor, [-1,1] rescale, no ImageNet norm).
- `.planning/research/SUMMARY.md` §"Architecture Approach" + §"Critical
  Pitfalls" + §"Research Flags" — package-of-modules architecture, the
  `requires_grad_()` tensor-identity invariant, the de-risk spike sequence.
- `.planning/research/ARCHITECTURE.md` — component breakdown (manifest reader,
  ingest/mirror, loaders, attribution engine, overlay) — read before planning.
- `.planning/research/STACK.md` — exact pins (mirror dynamicLRP
  `requirements.txt` verbatim, then project additions).
- `.planning/research/FEATURES.md` — table-stakes feature list for Phase 1.

### Stack & integration detail
- `CLAUDE.md` (Technology Stack section) — full pinned stack table, the
  SigLIP-2 + dynamicLRP integration sketch, "What NOT to use".
- `README.md` — runtime model (remote GCP VM, JupyterLab over SSH), dataset
  layout, GCS bucket layout.
- `metadata/rumsey_manifest.json` — 1,544 entries (`id`, `image_url`,
  `thumbnail_url`, metadata, `richness_score`); index semantics: higher index ≈
  richer map. Source of the highest-index slice map (D-05).

### External (record SHA when vendoring — D-08)
- `https://github.com/keeinlev/dynamicLRP` (`src/lrp_engine/`, `requirements.txt`,
  `src/experiments/ViT.ipynb`, `src/lrp_engine/model_specific/mosaicbert.py` as
  the custom-promise pattern) — vendor `src/lrp_engine/` at a pinned commit SHA.
- `arXiv 2512.07010` — dynamicLRP paper (Table A.4 SigLIP-2 coverage; §3.1–3.2
  Promise-system VRAM; §5 occlusion test).

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- None — greenfield. Repo currently contains only `.planning/`, `CLAUDE.md`,
  `README.md`, and `metadata/rumsey_manifest.json`. No source code, no
  `.planning/codebase/` maps (the `01-end-to-end-skeleton/` directory is from an
  abandoned earlier roadmap structure and is being deleted — not legacy to
  preserve).

### Established Patterns
- None yet. The architecture to *establish* is research-prescribed:
  single-responsibility Python modules called from thin notebook cells, GCS as
  the only persistence layer (SUMMARY.md §Architecture).

### Integration Points
- GCS bucket `gs://mapclass-training-northeast1` (`data/`, `models/`) — already
  exists; VM has ADC. All persistence connects here.
- JupyterLab on the GCP VM over SSH tunnel — the only inspection surface.

</code_context>

<specifics>
## Specific Ideas

- Locked verified slice: highest-index manifest map, control query `"a river"`,
  query-swap query `"a xylophone"` (D-05, D-06).
- Phase 1 "done" = full 1,544 mirror complete (with outcome manifest) **and**
  the slice verified by visual eyeball of all three controls (D-02, D-03).
- dynamicLRP engine lives vendored in-tree with its source SHA recorded
  (D-08).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Quantitative/printed control
metrics were offered and explicitly declined in favor of visual eyeball; this
is recorded as accepted risk D-04, not a deferred idea. Genuine
metrics/faithfulness scoring remains v2 per REQUIREMENTS.md, out of scope.)

</deferred>

---

*Phase: 1-Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution*
*Context gathered: 2026-05-18*
