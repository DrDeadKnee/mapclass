# Phase 4: Fine-tune and evaluate the segmentation model - Context

**Gathered:** 2026-05-16
**Status:** Ready for planning

<domain>
## Phase Boundary

**RESHAPED (user decision 2026-05-16) — narrower than the original ROADMAP
Phase 4.** Phase 4 trains the Phase-3 segmentation model end-to-end on the
Phase-2 dataset (honouring per-sample `sample_weights.json` weights) and
computes the joint per-pixel NLL evaluation on the held-out synthetic test set.

**Phase 4 final deliverable:** trained checkpoints for whatever slice of the
A/B × SigLIP/DINOv2/Swin matrix the cost-probe gate selects + the joint-NLL
metrics on `test/` + a **written A/B × backbone comparison report**.

**Phase 4 STOPS there.** It does NOT select "the" model and does NOT publish
anything. Model selection, hands-on model/dataset exploration, and the
HuggingFace publication (DELIV-01, DELIV-02, and the "reported with the
published model" / "load published model" half of EVAL-02) move to a **new
Phase 5**. Rationale: the user needs to test and explore the trained models
and dataset by hand before choosing and publishing a deliverable.

**ROADMAP delta (must be applied before planning):** original Phase 4 success
criteria #3 (uploaded to HuggingFace) and #4 (load published model) relocate to
a new Phase 5. Phase 4 retains SC #1 (trained end-to-end honouring weights) and
SC #2 (joint NLL computed on held-out synthetic test, primary + benchmarks
side by side). EVAL-02 is split: *computation* stays in Phase 4; *"reported
with the published model"* is Phase 5. EVAL-03 (DINOv2/Swin compared on the
same metric) stays in Phase 4. Requirements PHASE-04, DELIV-01, DELIV-02 →
Phase 5. See `confirm_creation` next-steps for the `/gsd-phase` action.

</domain>

<decisions>
## Implementation Decisions

### Phase 4 / Phase 5 scope split
- **D-01: Phase 4 = train + evaluate + report only.** Ends with trained
  checkpoints + joint-NLL numbers on `test/` + a written A/B × backbone
  comparison report. No "which model is the deliverable" choice, no packaging,
  no HuggingFace anything.
- **D-02: New Phase 5 owns delivery.** Hands-on model/dataset exploration by
  the user → deliverable selection → HuggingFace publish (public repo, license,
  model card with inference instructions + joint-NLL, optional benchmark
  numbers). DELIV-01/DELIV-02 and EVAL-02's "reported-with-published-model"
  clause live there. (Original discuss areas "Deliverable & model selection"
  and "HuggingFace publish shape" were deliberately deferred here — NOT
  discussed in this session per user instruction.)

### Training matrix scope (cost-probe-gated)
- **D-03: Cost-probe-first; grid breadth is NOT committed upfront.** The plan
  runs a short calibration training first and decides how much of the
  A/B × SigLIP/DINOv2/Swin grid to train from the *measured* cost.
- **D-04: Probe config = SigLIP Variant B** (trainable widened 15-ch
  patch-embed). Chosen because it upper-bounds SigLIP-path cost, so the
  projection is conservative.
- **D-05: Manual `checkpoint:decision` gate after the probe.** The plan halts
  (task `autonomous: false`, mirroring the Phase-2 gated-checkpoint pattern),
  reports measured GPU-hrs/epoch + projected full-grid cost, and the **user**
  chooses grid breadth at that gate. Candidate breadths to present at the gate:
  full 6-config grid · SigLIP A+B full + DINOv2/Swin single-variant · SigLIP
  A+B full + DINOv2/Swin short benchmark runs.

### GPU execution & artifact handoff
- **D-06: Training executes on a separate GPU box; planning is CPU-safe.**
  All `discuss`/`plan` work is environment-independent (no GPU). The training
  + eval execution is an explicitly gated wave run on a GPU host, using the
  same hand-commands / push-before-pull / target-the-subset discipline
  established for the Phase-2 networked-host gate. Phase 4 needs its own
  deferred GPU-host gate artifact analogous to `02-HUMAN-UAT.md`.
- **D-07: Checkpoints → `gs://mapclass-training-northeast1/models/<config>/`.**
  Per-config prefix under that bucket (project: narrative-campaign). This GCS
  convention does NOT currently exist anywhere in this repo (data/models are
  local `data/`) — Phase 4 introduces it.
- **D-08: Only lightweight artifacts return to git.** A manifest of the
  `gs://` checkpoint URIs + the joint-NLL metrics (JSON) + the written
  comparison report are committed. Model weights are NEVER committed to git.
- **D-09: Periodic GCS checkpoints + auto-resume.** Write
  `{weights, optimizer, step}` to the config's `gs://` prefix every N
  steps/epochs; on (re)launch auto-detect the latest checkpoint and resume.
  Preemption-safe by design — GPU VMs may be interrupted/preempted.

### Claude's Discretion (planner/researcher decide within the above)
- Training hyperparameters, optimizer, LR schedule, epoch budget, and the
  loss formulation that consumes per-sample `sample_weights.json` (locked dict
  shape from Phase 2) — within DECISION-class-conditional-loss-weights.
- **Validation-signal methodology (constrained):** the cost probe's "epoch",
  any early-stopping, and training-progress signal need a validation slice.
  That slice MUST be carved from `train/` only — `test/` must never be
  enumerated during training (EVAL-01 zero-leakage, locked). Researcher to
  recommend the train→val carve; do NOT re-ask the user.
- Exact checkpoint cadence `N` (D-09), GCS I/O mechanism, resume-detection
  logic.
- How recursive coarse-to-fine inference (Phase-3 D-03/D-04) is invoked at
  eval time, and the comparison-report format/structure.
- Pyramid-aware batch sampling and dataloader internals (Phase-3 D-06).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Locked constraints & requirements
- `.planning/PROJECT.md` — locked SPEC constraints: joint per-pixel NLL is the
  single evaluation metric; SigLIP locked primary backbone; Gemma NEVER in the
  segmentation forward pass; DINOv2/Swin non-locked benchmark alternatives;
  per-source class-conditional loss weights persisted in `sample_weights.json`.
- `.planning/REQUIREMENTS.md` — PHASE-04, EVAL-02, EVAL-03, DELIV-01, DELIV-02
  (note the Phase 4/5 split in `<domain>`: DELIV-01/02 + EVAL-02 reporting →
  Phase 5; EVAL-02 computation + EVAL-03 stay in Phase 4).
- `.planning/ROADMAP.md` §"Phase 4" — original goal + success criteria
  (criteria #3/#4 relocate to Phase 5 per the ROADMAP delta above).

### Phase-3 model under test (what Phase 4 trains)
- `.planning/phases/03-build-a-dense-semantic-segmentation-pipeline/03-CONTEXT.md`
  §decisions — D-03a (Variant A frozen+decoder-prior vs Variant B
  trainable-widened-15ch-patch-embed), D-05 (unified `Backbone` protocol:
  SigLIP/DINOv2/Swin), D-06/D-06a (Phase-3 = construction only; Phase-4 owns
  ALL training + the joint-NLL eval), D-01/D-02 (shared conv decoder + 2 thin
  heads), D-04 (recursive c2f walks `pyramid.json` parent→child literally).
- `scripts/seg/model.py`, `scripts/seg/backbones.py`, `scripts/seg/decoder.py`,
  `scripts/seg/heads.py`, `scripts/seg/recursive.py`, `scripts/seg/dataset.py`
  — the assembled, untrained Phase-3 model + dataloader Phase 4 fine-tunes.
- `.planning/phases/03-build-a-dense-semantic-segmentation-pipeline/03-UAT.md`
  — confirms both variants emit correct dense tensors and no Gemma is reachable
  (the contract Phase 4 trains against).

### Phase-1 backbone weights (reused, frozen for Variant A)
- `scripts/finetune_paligemma.py` — Phase-1 PEFT adapter production/load
  contract; SigLIP backbone path loads these, Gemma never in forward pass.

### Phase-2 dataset & zero-leakage contract
- `.planning/phases/02-build-a-dataset-of-pixel-label-pairs/02-CONTEXT.md`
  §decisions — D-15..D-18 (frozen seeded stratified `data/synthetic/split.json`,
  `train/` vs `test/` filesystem separation — the EVAL-01 leakage guarantee
  Phase 4 must preserve; any val slice comes from `train/` only).
- `scripts/tiling.py` — `pyramid.json` schema + per-map `image.png` /
  `land_cover.png` / `topography.png` / `sample_weights.json` contract the
  training loop and eval consume.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/seg/` (model/backbones/decoder/heads/recursive/dataset) — the
  complete assembled-but-untrained Phase-3 pipeline. Phase 4 adds a training
  loop + eval harness ON TOP; it does not re-architect the model.
- `scripts/finetune_paligemma.py` — established LoRA/PEFT load + Gemma-isolation
  pattern; Phase 4's SigLIP path reuses it so the trained seg model provably
  never touches Gemma.
- Phase-2 gated `checkpoint:decision` pattern (`autonomous: false` halts in
  02-03/02-04) — the proven template for D-05's post-probe cost gate.

### Established Patterns
- Frozen `data/synthetic/split.json` + sibling `train/`/`test/` dirs: training
  + any val carve point ONLY at `train/`; `test/` enumerated solely by the
  final joint-NLL eval. End-to-end EVAL-01 zero-leakage.
- Per-source `sample_weights.json` (historical uniform-trust, synthetic
  uniform-1.0, satellite balance-tilt) surfaced unchanged through the Phase-3
  dataloader (Phase-3 D-06) — Phase 4's loss multiplies by these.
- Networked/GPU-host gate workflow (from the Phase-2 online-integration gate):
  separate box, hand exact commands, push-before-tell-pull, target the changed
  subset, deferred *-HUMAN-UAT-style gate artifact.

### Integration Points
- Training loop ← `scripts/seg` model + Phase-2 pyramid dataloader + Phase-1
  PEFT adapter (SigLIP) / pretrained DINOv2/Swin weights.
- Checkpoints → `gs://mapclass-training-northeast1/models/<config>/` (new GCS
  convention; not present in repo today).
- Eval harness ← held-out `data/synthetic/.../test/` via frozen `split.json`;
  recursive c2f inference from Phase-3 D-04.
- git ← URI manifest + joint-NLL metrics JSON + comparison report only.

</code_context>

<specifics>
## Specific Ideas

- The cost-probe gate is treated like the Phase-2 loss-weight gates: a real
  blocking `checkpoint:decision` the user answers with measured numbers in
  hand — not a Claude-discretion auto-pick.
- Phase 5 is explicitly a *hands-on* phase: the user wants to play with the
  models and dataset directly before any deliverable is chosen or published.
  Phase 4 should leave artifacts in a shape that makes that exploration easy
  (clear per-config GCS prefixes, a readable comparison report).

</specifics>

<deferred>
## Deferred Ideas

- **Deliverable / model selection** (original discuss area #3) → **Phase 5.**
  Which trained config is "the" model, and the selection criterion. Not
  discussed this session by user instruction.
- **HuggingFace publish shape** (original discuss area #4) → **Phase 5.**
  Public repo/org naming, license, model-card contents (inference instructions
  + joint-NLL), whether DINOv2/Swin benchmark numbers ship alongside.
  DELIV-01/DELIV-02 + EVAL-02 "reported-with-published-model" clause.
- **OCR / place-name reading** — carried from Phase 3 deferred; still a future
  v2 / GEOREF-V2 concern, not Phase 4 or Phase 5.

### Reviewed Todos (not folded)
None — `todo.match-phase 4` returned zero matches.

</deferred>

---

*Phase: 4-Fine-tune and evaluate the segmentation model*
*Context gathered: 2026-05-16*
