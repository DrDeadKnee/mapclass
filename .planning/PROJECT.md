# MapClass

## What This Is

MapClass is exploratory research piping that runs **dynamic-LRP attribution
across multiple pre-trained vision models against multiple historical maps**
from the **David Rumsey** collection. One notebook runs dynamic LRP for
**SigLIP-2, CLIP, ViT-b-16, and PaliGemma-3B** over the last 50 Rumsey maps and
renders the attribution heatmaps in JupyterLab on a remote GCP VM for a human to
eyeball. Models and maps both come from `gs://mapclass-training-northeast1/`.
The deliverable is that working notebook — including the honest "no heatmap"
result when dynamic LRP cannot traverse a given model.

## Core Value

One notebook that runs dynamic-LRP attribution for several models against
several maps and shows the heatmaps so a human can eyeball them. Models + maps
both loaded from the GCS buckets in the README. A model dynamic LRP cannot
attribute just shows "no heatmap" — a result, not a failure. Keep it simple:
if everything else fails, this one notebook must work.

## Current Milestone: v1.1 Multi-Model × Multi-Map Attribution Notebook

**Goal:** One notebook runs dynamic-LRP attribution for SigLIP-2, CLIP,
ViT-b-16, and PaliGemma-3B against the last 50 Rumsey maps and renders the
heatmaps for visual eyeballing. A model dynamic LRP cannot traverse shows a
captioned "no heatmap" tile rather than crashing. Deliberately small — no
adapter framework.

**Target features:**
- Load the 4 models from `gs://.../models/` (SigLIP-2 mirrored; PaliGemma already at `models/paligemma-3b-mix-224/`; CLIP + ViT-b-16 mirrored as needed)
- Plain per-model attribution-target functions (no Protocol/dataclass scaffold)
- One notebook: last-50 maps × 4 models → labeled overlays grouped per map; failures as captioned tiles; headless exit 0
- Reuse v1.0 infra unchanged: pinned env, GCS image mirror, loaders, D-09 signed/zero-centered overlay; only generalize model load + target + patch geometry

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

(None yet — ship to validate)

### Active

<!-- Current scope. Building toward these. All hypotheses until shipped. -->

- [ ] Load SigLIP-2, CLIP, ViT-b-16, PaliGemma-3B from `gs://.../models/` (plain per-model functions, no adapter framework)
- [ ] Wire each model's query-conditioned attribution target (SigLIP-2/CLIP `logits_per_image`; ViT class logit; PaliGemma answer-token)
- [ ] Generalize the v1.0 overlay patch-grid per model's patch size (keep D-09 signed/zero-centered)
- [ ] One notebook iterates the last 50 Rumsey maps × the 4 models, rendering labeled overlays grouped per map
- [ ] A failing (map, model) shows a captioned "no heatmap" tile; notebook completes headless at exit 0 — **milestone done**

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
- **Configurable sweep machinery + run caching + contact-sheet styling** — v1.1 just iterates a fixed last-50 maps in one notebook; the configurable-sweep/caching/contact-sheet build (old v1.0 Phase 2) is not needed.
- **Adapter Protocol / PatchGeometry dataclass / conformance-test & pre-flight-guard infrastructure** — over-engineered for one notebook (user: "getting over complicated"); use plain per-model functions.
- LRP method/parameter tuning as a first-class explore axis — fixed; vary model + map only.
- Fixing per-model dynamic-LRP op-coverage gaps (custom Promises / pre-pool / LXT / captum fallback) — a model the engine can't traverse just shows "no heatmap"; engineering around it defeats the purpose (SigLIP-2 Fallback Ladder declined, v1.0).
- Other dataset categories (Toons, Fantasy, MapMaker, Satellite) — Rumsey historical only

## Context

- **Fresh repo.** All prior fine-tuning pipeline work is abandoned and irrelevant; nothing in the current tree is legacy to preserve.
- **Reference implementation:** dynamic LRP comes from https://github.com/keeinlev/dynamicLRP (paper: arXiv 2512.07010). The load-bearing external dependency. v1.0 established it does NOT cover SigLIP-2's MAP-pool `split_with_sizes` op — adapting/comparing it across model architectures is the central technical risk, and per-model coverage is itself the data v1.1 produces.
- **Dataset:** `metadata/rumsey_manifest.json` — 1,544 entries, each with `id`, `image_url`, `thumbnail_url`, and rich metadata (date, author, region, `richness_score`). Images are public Rumsey URLs to be mirrored to GCS for stability.
- **Index semantics:** higher manifest index ≈ richer/more complete maps. v1.1 uses the **last 50** manifest entries (`manifest[-50:]` — the richest).
- **Runtime:** remote GCP VM with GPU (L4), JupyterLab via SSH tunnel (per README). GCS bucket `gs://mapclass-training-northeast1` exists; VM has `gcloud`/ADC.
- **Explore axes for v1.1:** **model** and **map** are the swappable knobs (4 models × last-50 maps). LRP config and text query held fixed.
- **Models:** SigLIP-2-so400m-patch14-384, CLIP, ViT-b-16, PaliGemma-3B. Loaded from `gs://.../models/`: SigLIP-2 mirrored (v1.0); PaliGemma already at `models/paligemma-3b-mix-224/` (full code+weights, **NOT gated**); CLIP + ViT-b-16 to be mirrored. Each needs its own load + query-conditioned target; distinct op-coverage outcomes expected and recorded.

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
| **Per-model LRP coverage gap = recorded result, not a bug (v1.1)** | SigLIP-2's `split_with_sizes` MAP-pool gap is the first datum; engineering around gaps defeats the comparison (Fallback Ladder declined) | — Active |
| **Simplify to one notebook, plain functions (v1.1)** | User: "getting over complicated — just want a notebook that runs multiple models against multiple maps." Collapsed 5-phase adapter/guard roadmap → 1 phase; no Protocol/dataclass/guard scaffold | — Active |
| **v1.1 axes = model × map (4 models × last-50 Rumsey maps)** | "Multiple maps" reinstated as the second axis (not a locked single slice, not a configurable sweep — a fixed last-50) | — Active |
| **PaliGemma served from self-hosted GCS copy** | Full code+weights already at `models/paligemma-3b-mix-224/`; treat as NOT gated | — Active |
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
*Last updated: 2026-05-19 — v1.1 simplified to a single-phase Multi-Model × Multi-Map Attribution Notebook (per user)*
