# MapClass

## What This Is

MapClass is exploratory research piping for pixel-level labeling of artistic map
images. It runs **dynamic LRP attribution** on a pre-trained vision-language model
(**SigLIP-2-so400m-patch14-384**) over historical maps from the **David Rumsey**
collection, producing per-query attribution heatmaps that a human eyeballs in
JupyterLab on a remote GCP VM. The deliverable is the *piping that lets the
researcher explore* — not a polished dataset or labeller (yet).

## Core Value

A working, repeatable loop: pick a map + a text query → get a dynamic-LRP
attribution heatmap overlaid on that map → judge it visually — that scales to a
configurable sweep of maps × queries browsable in a notebook. If everything else
fails, this loop must work.

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. All hypotheses until shipped. -->

- [ ] Mirror Rumsey map images from manifest `image_url`s into `gs://mapclass-training-northeast1/data/`
- [ ] Mirror SigLIP-2-so400m-patch14-384 weights from HuggingFace into `gs://mapclass-training-northeast1/models/`
- [ ] Image loader reads a map by ID from the GCS data mirror
- [ ] Run SigLIP-2 + dynamic LRP (from keeinlev/dynamicLRP) for a single map + single text query
- [ ] Produce an attribution heatmap overlaid on the source map, viewable in JupyterLab
- [ ] Eyeball the single slice (one map, one query) and judge it visually — **Phase 1 done**
- [ ] Sweep a configurable subset of maps × a set of queries, counting down from high manifest index N
- [ ] Browse all attribution maps from a sweep inside the notebook (contact-sheet style) — **v1 done**

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Fine-tuning / model training — that is the abandoned prior approach this restart deliberately replaces; dynamic LRP is "better and less work"
- Quantitative attribution metrics — v1 judgment is purely visual; metrics add scope without payoff until the loop is trusted
- Open-source labeller integration — README's "phase 2"; depends on the exploration loop existing first
- Polished mask/dataset export — pixel-label dataset generation is a downstream milestone, not v1 piping
- Model swapping as a first-class explore axis — model is fixed to SigLIP-2 in v1 to keep the loop small
- LRP method/parameter tuning as a first-class explore axis — fixed in v1; vary map + query only
- Other dataset categories (Toons, Fantasy, MapMaker, Satellite) — Rumsey historical only for v1

## Context

- **Fresh repo.** All prior fine-tuning pipeline work is abandoned and irrelevant; nothing in the current tree is legacy to preserve.
- **Reference implementation:** dynamic LRP comes from https://github.com/keeinlev/dynamicLRP (paper: arXiv 2512.07010). This is the load-bearing external dependency — adapting it to SigLIP-2 is the central technical risk.
- **Dataset:** `metadata/rumsey_manifest.json` — 1,544 entries, each with `id`, `image_url`, `thumbnail_url`, and rich metadata (date, author, region, `richness_score`). Images are public Rumsey URLs to be mirrored to GCS for stability.
- **Index semantics:** higher manifest index ≈ richer/more complete maps; low index (near 0) ≈ more abstract/incomplete. Sweeps should count *down from a high index N*, not up from 0.
- **Runtime:** remote GCP VM with GPU, JupyterLab accessed via SSH tunnel (per README). GCS bucket `gs://mapclass-training-northeast1` already exists; the VM has `gcloud`/ADC credentials to read and write it.
- **Explore axes for v1:** input map and text query are the swappable knobs. Model and LRP config are held fixed.

## Constraints

- **Tech stack**: Python, PyTorch-based VLM (SigLIP-2 via HuggingFace), JupyterLab for inspection
- **Dependency**: dynamic LRP behavior is bounded by what keeinlev/dynamicLRP supports and how cleanly it adapts to SigLIP-2's architecture
- **Infrastructure**: GCS bucket `gs://mapclass-training-northeast1` (data/ and models/); GCP VM with GPU; auth assumed present
- **Environment**: must run on a remote VM with results inspected through a Jupyter notebook (no local-first assumption)
- **Model size**: SigLIP-2-so400m is the chosen size; dynamic LRP fidelity/overhead degrades with larger models, so this is a deliberate ceiling

## Key Decisions

<!-- Decisions that constrain future work. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Restart with dynamic LRP instead of fine-tuning | "Better if it works, and less work" than the dataset-troubled fine-tuning pipeline | — Pending |
| SigLIP-2-so400m-patch14-384 as the v1 model | Image-text model supports query-driven attribution; sized for LRP fidelity | — Pending |
| Adopt keeinlev/dynamicLRP as reference impl | Avoids re-deriving dynamic LRP from the paper | — Pending |
| Mirror data + model to GCS, load from GCS | Stability/reproducibility on ephemeral VMs; avoids HF/Rumsey runtime dependency | — Pending |
| v1 explore axes = map + query only | Keep the loop small; defer model/LRP-tuning knobs | — Pending |
| Purely visual evaluation in v1 | Trust the loop visually before investing in metrics | — Pending |
| Sweep counts down from high manifest index | Higher index = richer maps; low index too abstract to be useful early | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-05-18 after initialization*
