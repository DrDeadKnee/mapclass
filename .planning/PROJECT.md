# MapClass

## What This Is

MapClass is exploratory research piping that **compares dynamic-LRP attribution
across multiple pre-trained vision models** on the same historical map from the
**David Rumsey** collection. For a single locked map + text query it runs
dynamic LRP against **SigLIP-2, CLIP, PaliGemma, and a plain ViT**, rendering
their attribution heatmaps side-by-side in JupyterLab on a remote GCP VM so a
human can eyeball how the method behaves model-to-model. The deliverable is the
*cross-model comparison piping* — including the honest negative result when
dynamic LRP cannot traverse a given model's architecture.

## Core Value

A working, repeatable cross-model comparison: pick the locked map + a text query
→ run dynamic-LRP attribution through each model → see their heatmaps
side-by-side (and, per model, whether dynamic LRP covers that architecture at
all) → judge visually. The comparison itself is the product. A model dynamic LRP
cannot attribute is a *result*, not a failure. If everything else fails, this
side-by-side comparison must work.

## Current Milestone: v1.1 Multi-Model Dynamic-LRP Comparison

**Goal:** Run dynamic-LRP attribution for the same locked map + text query across
SigLIP-2, CLIP, PaliGemma, and a plain ViT, and render their attribution
heatmaps side-by-side in a notebook — so the cross-model comparison is the
deliverable. A model dynamic LRP cannot traverse simply shows no heatmap
(op-coverage gap) rather than crashing.

**Target features:**
- Model-agnostic attribution adapter (per-model load + forward + query-conditioned target)
- CLIP, PaliGemma, and a plain ViT added alongside SigLIP-2, weights mirrored to GCS
- One notebook: locked slice + query → side-by-side LRP overlays for all 4 models
- Reuse v1.0 infra unchanged: pinned env, GCS image mirror, loaders, D-09 signed/zero-centered overlay

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. All hypotheses until shipped. -->

- [ ] Model-agnostic attribution adapter: given a model id, load it + run forward + expose the query-conditioned attribution target
- [ ] CLIP added as a comparison model (weights mirrored to GCS, loadable)
- [ ] PaliGemma added as a comparison model (weights mirrored to GCS, loadable)
- [ ] A plain ViT added as a comparison model (weights mirrored to GCS, loadable)
- [ ] SigLIP-2 retained as a comparison model (reusing v1.0's path; its op-coverage gap is a recorded result)
- [ ] Run dynamic LRP for the locked map + a text query across all 4 models in one notebook
- [ ] Render the per-model attribution heatmaps side-by-side; a model with an LRP op-coverage gap shows "no heatmap" instead of crashing the notebook — **milestone done**

### Carried from v1.0 (shipped infrastructure, reused as-is)

- [x] Pinned reproducible environment + vendored dynamicLRP (Phase 1, v1.0)
- [x] Rumsey image mirror to GCS + idempotent ingestion (Phase 1, v1.0)
- [x] Image-by-ID loader from the GCS data mirror (Phase 1, v1.0)
- [x] SigLIP-2 attribution + D-09 signed/zero-centered overlay + 27×27 grid reconstruction (Phase 1, v1.0 — code shipped; SigLIP-2 LRP op-coverage gap recorded as the first comparison datum)

### Out of Scope

<!-- Explicit boundaries. Includes reasoning to prevent re-adding. -->

- Fine-tuning / model training — that is the abandoned prior approach this restart deliberately replaces; dynamic LRP is "better and less work"
- Quantitative attribution metrics — v1 judgment is purely visual; metrics add scope without payoff until the loop is trusted
- Open-source labeller integration — README's "phase 2"; depends on the exploration loop existing first
- Polished mask/dataset export — pixel-label dataset generation is a downstream milestone, not v1 piping
- **Maps × queries sweep + contact-sheet browse** — dropped in v1.1. The comparison axis is *model*, not *map count*; the locked single slice is sufficient to compare models. (Was v1.0 Phase 2; never built.)
- LRP method/parameter tuning as a first-class explore axis — fixed; vary model + query only in v1.1
- Fixing per-model dynamic-LRP op-coverage gaps (custom Promises / pre-pool / LXT / captum fallback) — a model the engine can't traverse is a *recorded comparison result*; engineering around it defeats the comparison's purpose (SigLIP-2 Fallback Ladder declined, v1.0)
- Other dataset categories (Toons, Fantasy, MapMaker, Satellite) — Rumsey historical only

## Context

- **Fresh repo.** All prior fine-tuning pipeline work is abandoned and irrelevant; nothing in the current tree is legacy to preserve.
- **Reference implementation:** dynamic LRP comes from https://github.com/keeinlev/dynamicLRP (paper: arXiv 2512.07010). The load-bearing external dependency. v1.0 established it does NOT cover SigLIP-2's MAP-pool `split_with_sizes` op — adapting/comparing it across model architectures is the central technical risk, and per-model coverage is itself the data v1.1 produces.
- **Dataset:** `metadata/rumsey_manifest.json` — 1,544 entries, each with `id`, `image_url`, `thumbnail_url`, and rich metadata (date, author, region, `richness_score`). Images are public Rumsey URLs to be mirrored to GCS for stability.
- **Index semantics:** higher manifest index ≈ richer/more complete maps; low index (near 0) ≈ more abstract/incomplete. Sweeps should count *down from a high index N*, not up from 0.
- **Runtime:** remote GCP VM with GPU, JupyterLab accessed via SSH tunnel (per README). GCS bucket `gs://mapclass-training-northeast1` already exists; the VM has `gcloud`/ADC credentials to read and write it.
- **Explore axes for v1.1:** **model** and text query are the swappable knobs. The map is locked to one slice and LRP config is held fixed.
- **Models under comparison:** SigLIP-2-so400m-patch14-384, CLIP, PaliGemma, a plain ViT. Each needs its own load + query-conditioned attribution target; PaliGemma (generative VLM) and CLIP (contrastive) differ from SigLIP-2's MAP-pool head — distinct op-coverage outcomes are expected and are the point.

## Constraints

- **Tech stack**: Python, PyTorch-based vision models (SigLIP-2 / CLIP / PaliGemma / ViT via HuggingFace), JupyterLab for inspection; pinned stack unchanged from v1.0
- **Dependency**: dynamic LRP behavior is bounded by what keeinlev/dynamicLRP supports per model architecture; per-model coverage gaps are recorded as results, not engineered around
- **Infrastructure**: GCS bucket `gs://mapclass-training-northeast1` (data/ and models/); GCP VM with GPU (L4); auth assumed present
- **Environment**: must run on a remote VM with results inspected through a Jupyter notebook (no local-first assumption)
- **Model size**: each comparison model kept within the dynamic-LRP fidelity/overhead ceiling (SigLIP-2-so400m class); no models larger than necessary for the comparison

## Key Decisions

<!-- Decisions that constrain future work. -->

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Restart with dynamic LRP instead of fine-tuning | "Better if it works, and less work" than the dataset-troubled fine-tuning pipeline | — Pending |
| Adopt keeinlev/dynamicLRP as reference impl | Avoids re-deriving dynamic LRP from the paper | ✓ Shipped (v1.0, vendored SHA-pinned) |
| Mirror data + model to GCS, load from GCS | Stability/reproducibility on ephemeral VMs; avoids HF/Rumsey runtime dependency | ✓ Shipped (v1.0 Phase 1) |
| Purely visual evaluation | Trust the comparison visually before investing in metrics | — Active |
| **Pivot to multi-model comparison (v1.1)** | The real purpose is comparing dynamic-LRP across models, not a SigLIP-2-only loop; one model is one data point | — Active |
| **Drop the maps × queries sweep (v1.1)** | Comparison axis is *model*, not *map count*; locked single slice suffices | — Active |
| **Per-model LRP coverage gap = recorded result, not a bug (v1.1)** | SigLIP-2's `split_with_sizes` MAP-pool gap is the first datum; engineering around gaps defeats the comparison (Fallback Ladder declined) | — Active |
| **v1.1 explore axes = model + query; map locked** | Isolate the model variable for a clean cross-model comparison | — Active |
| SigLIP-2-so400m-patch14-384 as a comparison model | Was the v1.0 fixed model; retained as one of four under comparison | — Superseded by the multi-model pivot |

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
*Last updated: 2026-05-19 — milestone v1.1 (Multi-Model Dynamic-LRP Comparison) started*
