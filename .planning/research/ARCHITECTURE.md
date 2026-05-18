# Architecture Research

**Domain:** Single-researcher ML attribution pipeline (dynamic LRP on SigLIP-2 over GCS-hosted historical maps, inspected in JupyterLab on a remote GCP VM)
**Researched:** 2026-05-18
**Confidence:** MEDIUM — pipeline structure and the DynamicLRP integration mechanism are HIGH confidence (verified against the keeinlev/dynamicLRP source tree and the ViT example notebook); the SigLIP-2-specific adaptation is MEDIUM/LOW (no published CLIP/SigLIP example exists in the repo — this is the flagged risk).

## Standard Architecture

This is not a multi-user service. It is a **batch attribution pipeline with a notebook inspection front-end**. The right shape is a small Python package of single-responsibility modules called from thin notebook cells, with GCS as the only persistence layer. "Scale" here means "number of maps × queries in a sweep", not concurrent users.

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                  INSPECTION LAYER (JupyterLab on VM)                   │
├──────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐         ┌────────────────────────────────────┐  │
│  │ single-slice cell│         │ sweep cell (contact-sheet display) │  │
│  │ (1 map × 1 query)│         │ (N maps × Q queries, count down)   │  │
│  └────────┬─────────┘         └──────────────────┬─────────────────┘  │
│           │  calls thin Python API               │                    │
├───────────┴──────────────────────────────────────┴───────────────────┤
│                       ORCHESTRATION LAYER                              │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │  sweep orchestrator  (iterate manifest subset × query set,       │  │
│  │                       count DOWN from high index N, collect)     │  │
│  └───────┬─────────────────────────┬─────────────────────┬─────────┘  │
├──────────┼─────────────────────────┼─────────────────────┼────────────┤
│          │            CORE PIPELINE COMPONENTS            │            │
│  ┌───────▼──────┐  ┌───────────────▼─────┐  ┌─────────────▼────────┐  │
│  │ data loader  │  │  attribution engine │  │  overlay / viz       │  │
│  │ (map by ID   │→ │  (SigLIP-2 forward  │→ │  (heatmap → RGBA     │  │
│  │  → PIL/tensor│  │   + DynamicLRP)     │  │   over source image) │  │
│  └───────┬──────┘  └──────┬──────────────┘  └──────────────────────┘  │
│          │                │ uses                                       │
│  ┌───────▼──────┐  ┌──────▼──────────────┐                            │
│  │ model loader │  │ manifest reader     │                            │
│  │ (GCS mirror  │  │ (rumsey_manifest    │                            │
│  │  → HF model) │  │  → entry by index)  │                            │
│  └───────┬──────┘  └──────┬──────────────┘                            │
├──────────┼─────────────────┼──────────────────────────────────────────┤
│          │   SETUP / INGESTION (run-once, idempotent)                  │
│  ┌───────▼─────────────────▼──────────────────────────────────────┐   │
│  │  image ingestion (manifest image_url → download → GCS data/)    │   │
│  │  model mirror    (HF google/siglip2-... → GCS models/)          │   │
│  └────────────────────────────┬───────────────────────────────────┘   │
├───────────────────────────────┼────────────────────────────────────────┤
│                         STORAGE LAYER                                  │
│   gs://mapclass-training-northeast1/data/    (mirrored map images)     │
│   gs://mapclass-training-northeast1/models/  (mirrored SigLIP-2)       │
│   metadata/rumsey_manifest.json              (local, 1,544 entries)    │
└────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Typical Implementation |
|-----------|----------------|------------------------|
| **manifest reader** | Parse `metadata/rumsey_manifest.json`; resolve entry by manifest index; expose "count down from index N" iteration | Plain function returning dicts; index = list position (higher = richer per PROJECT.md) |
| **image ingestion** | One-time: for each manifest entry, download `image_url`, upload to `gs://.../data/<id>.jpg`. Idempotent (skip if blob exists) | `requests` + `google-cloud-storage`; resumable; logs failures, never blocks the loop |
| **model mirror** | One-time: pull `google/siglip2-so400m-patch14-384` from HF, push the snapshot to `gs://.../models/siglip2-so400m-patch14-384/` | `huggingface_hub.snapshot_download` then recursive GCS upload |
| **model loader** | Load SigLIP-2 + processor from the **GCS mirror** (not HF) onto GPU; force a known attention impl; cache the singleton | `AutoModel`/`AutoProcessor.from_pretrained(local_path, attn_implementation="eager")` after pulling mirror to local disk |
| **data loader** | Given a map `id`, fetch the image from `gs://.../data/`, return a preprocessed tensor (with `requires_grad_()`) plus the original PIL for overlay | GCS download → PIL → SigLIP-2 processor → tensor on device |
| **attribution engine** | The load-bearing component. Run SigLIP-2 forward for (image, text query), select the scalar similarity target, run DynamicLRP, return a per-patch/pixel relevance map | HF forward + `LRPEngine` from keeinlev/dynamicLRP (see "DynamicLRP ↔ SigLIP-2 Integration") |
| **overlay / viz** | Reshape patch relevance → image grid, upsample to image size, render as a colored heatmap composited over the source map | numpy reshape + `matplotlib` (bwr/jet colormap, alpha~0.5), as in the repo's ViT notebook |
| **sweep orchestrator** | Iterate `subset × queries`, count DOWN from high manifest index N, call data loader → attribution → overlay per pair, collect results for contact-sheet display | Nested loop + list of (id, query, overlay) tuples; resumable/skippable on per-item failure |

## Recommended Project Structure

```
mapclass/
├── metadata/
│   └── rumsey_manifest.json        # local source of truth (1,544 entries)
├── src/mapclass/
│   ├── config.py                   # bucket name, model id, GCS prefixes, device
│   ├── manifest.py                 # manifest reader + count-down-from-N iteration
│   ├── ingest_images.py            # run-once: image_url → GCS data/  (idempotent)
│   ├── mirror_model.py             # run-once: HF model → GCS models/  (idempotent)
│   ├── model_loader.py             # GCS models/ → loaded SigLIP-2 + processor (cached)
│   ├── data_loader.py              # id → GCS data/ → (tensor.requires_grad_(), PIL)
│   ├── attribution.py              # SigLIP-2 forward + DynamicLRP → relevance map
│   ├── overlay.py                  # relevance map → heatmap composited on source
│   └── sweep.py                    # orchestrate subset × queries, collect overlays
├── notebooks/
│   ├── 01_single_slice.ipynb       # Phase 1: one map × one query, eyeball it
│   └── 02_sweep.ipynb              # v1: contact-sheet of a configurable sweep
├── third_party/dynamicLRP/         # vendored keeinlev/dynamicLRP (pin a commit)
├── requirements.txt
└── README.md
```

### Structure Rationale

- **`src/mapclass/` flat module package, not a deep hierarchy:** one researcher, ~8 small modules. Each module = one box in the diagram. Notebooks import and call; they hold no logic beyond parameters and display so the loop is reproducible from `.py` and testable without a kernel.
- **Run-once setup separated from the loop (`ingest_images.py`, `mirror_model.py`):** these are slow, network-bound, idempotent, and must not be re-run every notebook session. They are scripts, not loop dependencies.
- **`third_party/dynamicLRP/` vendored and commit-pinned:** it is "the load-bearing external dependency" (PROJECT.md), default branch `master`, packaged as `src/lrp_engine`, research-grade (Jupyter-Notebook repo, no PyPI release). Vendoring + pinning prevents an upstream change from silently breaking the only loop that must work. Add its `requirements.txt` deps to the project's.
- **`config.py` centralizes the GCS layout and model id:** bucket/prefixes/model name appear in ingestion, mirror, model loader, and data loader; one place prevents drift.

## DynamicLRP ↔ SigLIP-2 Integration (the central technical risk)

This is the make-or-break boundary. Findings are verified against the repo source tree and the ViT example notebook (`src/experiments/ViT.ipynb`, `src/lrp_engine/lrp.py`).

**How DynamicLRP attaches (HIGH confidence):**
- **No model surgery, no forward hooks, no module replacement, no wrapping.** It operates *post-hoc on PyTorch's autograd computation graph*. `LRPEngine` walks the `grad_fn` chain of the model output, builds a topological graph (`make_graph_iter`), and back-propagates relevance through ~47 primitive tensor ops, using a "Promise System" to recover intermediate activations autograd discarded. (Source: `lrp_engine/lrp.py`, paper arXiv 2512.07010.)
- **Required workflow (from the ViT notebook, verbatim pattern):**
  1. Preprocess image → `img_tensor.unsqueeze(0).to(device).requires_grad_()`
  2. Standard HF forward: `output = model(...)`
  3. `engine = LRPEngine(use_gamma=True, no_recompile=True)`
  4. `engine.params_to_interpret = [img_tensor]`
  5. `lrp_output = engine.run(<scalar/target tensor>)`
  6. Pixel attribution comes back via the `params_to_interpret` relevance; reshape to the patch grid and upsample for overlay.
- **Implication for module boundaries:** the attribution engine must *own both the forward pass and the LRP call together* — they share the live autograd graph and the exact input tensor object. Do not split "run model" and "run LRP" across modules; the `grad_fn` chain and `params_to_interpret` identity must be preserved in one scope. The data loader must hand back the *same* `requires_grad_()` tensor that goes into both the forward pass and `params_to_interpret`.

**The SigLIP-2-specific adaptation risk (MEDIUM/LOW — flagged, unverified):**
- The repo's vision example is `ViTForImageClassification` — attribution target is **classification logits**. SigLIP-2 is a **dual-encoder contrastive model**: `Siglip2Model` forward returns `logits_per_image` / `logits_per_text` (image↔text similarity). The attribution target must be the **scalar image-text similarity for the chosen query** (e.g. `logits_per_image[0, 0]`), not class logits. The autograd graph from that scalar must trace back through the vision tower to `img_tensor`. This routing is the adaptation work; no published SigLIP/CLIP DynamicLRP example exists.
- **Op-coverage uncertainty:** DynamicLRP claims 99.92% node coverage over 15 architectures incl. multimodal (DePlot), but SigLIP-2 was not in the tested set. SigLIP's attention pooling head, the sigmoid-loss similarity, and the text-tower contribution may hit ops needing a `model_specific/` shim (cf. existing `model_specific/mosaicbert.py`).
- **Attention implementation:** load SigLIP-2 with `attn_implementation="eager"` rather than SDPA/flash. The ViT notebook ran with an SDPA backward and the engine lists SDPA among covered ops, but eager attention produces a plainer, fully-traced graph and is the lower-risk default for a first integration; revisit only if eager is too slow.
- **Mitigation:** treat "single map × single query attribution actually traces SigLIP-2 → pixels" as a **dedicated de-risking spike at the very start of Phase 1**, before any sweep or overlay polish. Reproduce the repo's ViT notebook first to confirm the toolchain, *then* swap in SigLIP-2 and the similarity target.

## Data Flow

### Run-once Setup Flow (prerequisite, idempotent)

```
rumsey_manifest.json ──┐
                       ├─► ingest_images: for each entry → GET image_url → PUT gs://.../data/<id>.jpg
HF google/siglip2-... ─┴─► mirror_model: snapshot_download → PUT gs://.../models/siglip2-so400m-patch14-384/
```

### Single-Slice Flow (Phase 1 — the loop that must work)

```
manifest index N ──► manifest reader ──► entry{id}
                                            │
gs://.../models/ ──► model loader ──► SigLIP-2 + processor (GPU, eager attn)
                                            │
entry.id ──► data loader ──► GCS data/<id>.jpg ──► PIL ──► processor ──► img_tensor.requires_grad_()
                                            │
text query ────────────────────────────────┤
                                            ▼
                    attribution engine: model(img_tensor, query)
                                  → similarity scalar logits_per_image[0,0]
                                  → LRPEngine(params_to_interpret=[img_tensor]).run(scalar)
                                  → per-patch relevance
                                            ▼
                    overlay: relevance → grid reshape → upsample → heatmap over PIL
                                            ▼
                    notebook cell: display(overlay)  ← human eyeballs it  [PHASE 1 DONE]
```

### Sweep Flow (v1 — adds breadth, reuses the same core)

```
config{ subset_count, query_set, start_index N }
        │
        ▼
sweep orchestrator:
   for idx in range(N, N - subset_count, -1):          # count DOWN from high index
       entry = manifest[idx]
       for query in query_set:
           try:
               (relevance) = attribution_engine(entry.id, query)   # SAME core as single-slice
               overlay = overlay(relevance, entry)
               results.append((entry.id, query, overlay))
           except Exception: log + continue              # one bad map ≠ dead sweep
        │
        ▼
notebook cell: contact-sheet grid of all overlays      [v1 DONE]
```

**Key data flows:**
1. **Tensor identity flow:** the *same* `img_tensor` object flows from data loader → model forward → `params_to_interpret`. Breaking this identity (e.g. cloning, re-tensoring between modules) breaks attribution. This is the single most important invariant in the system.
2. **GCS-first flow:** nothing in the inner loop touches HF Hub or davidrumsey.com. After setup, all reads are from `gs://mapclass-training-northeast1` — the stability/reproducibility goal in PROJECT.md.
3. **Sweep = single-slice in a loop:** the sweep adds *no new attribution logic*; it only iterates and collects. This is the core build-order lever.

## Scaling Considerations

"Scale" = maps × queries in a sweep on one GPU VM, not users.

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 1 map × 1 query (Phase 1) | No adjustments. Load model once per kernel; everything inline. |
| ~10 maps × ~5 queries (early v1) | Cache the loaded model singleton; cache GCS image downloads to local disk; reuse forward pass across queries for the same image if feasible. |
| ~100+ maps × queries (larger v1 sweep) | Local disk image cache keyed by id; consider batching the vision forward; persist overlays to disk so notebook re-renders don't recompute LRP. LRP backward is the bottleneck — it is per-(image,query) and not trivially batchable. |

### Scaling Priorities

1. **First bottleneck — DynamicLRP backward pass per (image, query).** Operation-level relevance + Promise activation recovery on a 400M-param model is the dominant cost and is essentially serial per pair. Mitigate by: keeping the model fixed (already a PROJECT.md decision), caching nothing-changes results to disk, and keeping sweep subsets small (count down from N, stop early).
2. **Second bottleneck — repeated GCS image downloads.** Cache to local VM disk on first fetch; the data loader checks local cache before GCS.
3. **Non-bottleneck — model load.** Slow once per kernel; amortized via a module-level singleton. Do not reload per slice.

## Anti-Patterns

### Anti-Pattern 1: Splitting forward pass and LRP across module boundaries

**What people do:** A "model service" returns logits; a separate "attribution service" later tries to run LRP on them.
**Why it's wrong:** DynamicLRP needs the live `grad_fn` graph and the exact `params_to_interpret` tensor object. Once the forward scope ends or tensors are detached/serialized, the graph is gone and attribution is impossible.
**Do this instead:** The attribution engine owns forward + LRP in one function call, receiving the `requires_grad_()` input tensor and returning a finished relevance map.

### Anti-Pattern 2: Loading the model (or pulling from HF/Rumsey) inside the loop

**What people do:** `from_pretrained(...)` or `requests.get(image_url)` inside the per-slice/per-sweep-iteration code.
**Why it's wrong:** Re-pays multi-GB load / network latency every iteration; reintroduces the HF/Rumsey runtime dependency the GCS mirror exists to remove (PROJECT.md key decision).
**Do this instead:** Run-once idempotent ingestion/mirror scripts; cached singleton model loader; data loader reads only from GCS (+ local disk cache).

### Anti-Pattern 3: Building the sweep before the single slice is visually trusted

**What people do:** Generalize to N×Q sweep + contact sheet before one (map, query) attribution has been eyeballed and judged correct.
**Why it's wrong:** If SigLIP-2↔DynamicLRP integration is subtly wrong, a sweep produces 100 plausible-looking-but-wrong heatmaps and hides the bug. PROJECT.md explicitly gates Phase 1 on eyeballing one slice.
**Do this instead:** Single-slice must be visually judged correct first; the sweep then wraps the *unchanged* core in a loop.

### Anti-Pattern 4: Re-deriving / re-implementing dynamic LRP from the paper

**What people do:** Reimplement relevance propagation instead of using keeinlev/dynamicLRP.
**Why it's wrong:** PROJECT.md explicitly adopts the reference impl to avoid re-deriving from the paper; reimplementation is enormous scope and the project's stated bet is "less work."
**Do this instead:** Vendor + pin keeinlev/dynamicLRP; treat SigLIP-2 adaptation as a thin `model_specific` shim if needed, not a rewrite.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| Google Cloud Storage (`mapclass-training-northeast1`) | `google-cloud-storage` SDK via ADC on the VM | Only persistence layer. data/ and models/ prefixes. Idempotent writes (skip-if-exists) for ingestion/mirror. |
| HuggingFace Hub (`google/siglip2-so400m-patch14-384`) | `huggingface_hub.snapshot_download`, **setup-only** | Touched once by mirror_model; never in the loop. |
| davidrumsey.com (`image_url` per entry) | `requests` GET, **setup-only** | Touched once by ingest_images; public URLs; expect some failures → log & continue, never block the loop. |
| keeinlev/dynamicLRP (master) | **Vendored** into `third_party/`, commit-pinned, imported as `lrp_engine` | Research code, no PyPI release, Jupyter-Notebook repo. Pin to insulate the load-bearing dep. |

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| manifest reader ↔ orchestrator | Direct call: index → entry dict | Index semantics: higher = richer; sweep counts DOWN from N. |
| data loader ↔ attribution engine | Direct call: returns `(img_tensor.requires_grad_(), pil_image)` | Tensor identity must be preserved into both forward and `params_to_interpret`. |
| model loader ↔ attribution engine | Direct call: cached singleton model+processor | Loaded once per kernel; eager attention. |
| attribution engine ↔ overlay | Direct call: per-patch relevance array + source PIL | Overlay owns reshape/upsample/colormap; engine owns nothing visual. |
| orchestrator ↔ {data loader, attribution, overlay} | Direct calls in a loop; per-item try/except | Sweep adds iteration only; zero new attribution logic. |

## Suggested Build Order (for the roadmap)

Dependency-ordered. Phase 1 = the slice that must work; v1 = the sweep wrapped around it.

1. **De-risk spike (start of Phase 1):** vendor+pin dynamicLRP; reproduce its `ViT.ipynb` end-to-end to confirm the toolchain works on the VM. *Gate: ViT heatmap reproduced.*
2. **Setup scripts (parallelizable, prerequisites):** `mirror_model` (HF→GCS), `ingest_images` (manifest→GCS). Idempotent. *Gate: model + at least the high-index map images present in GCS.*
3. **Loaders:** `manifest`, `model_loader` (GCS→model), `data_loader` (id→requires_grad tensor). *Gate: a high-index map loads as a tensor + PIL from GCS.*
4. **Attribution engine — the risk:** SigLIP-2 forward → similarity scalar → DynamicLRP → relevance. Adapt the ViT pattern to the dual-encoder target; add a `model_specific` shim only if op coverage fails. *Gate: relevance array returned without graph errors.*
5. **Overlay + single-slice notebook:** reshape→upsample→composite; `01_single_slice.ipynb`. *Gate: human eyeballs one (map, query) overlay and judges it — **PHASE 1 DONE**.*
6. **Sweep + contact-sheet notebook:** `sweep.py` wrapping the unchanged core; count down from N; per-item try/except; `02_sweep.ipynb` grid. *Gate: configurable subset × queries browsable — **v1 DONE**.*

Phase 1 needs steps 1–5 (single tensor through the whole pipe). v1 adds only step 6 — pure orchestration/iteration over the proven core, no new attribution logic.

## Sources

- keeinlev/dynamicLRP repository tree, `master` branch (`src/lrp_engine/lrp.py`, `src/lrp_engine/__init__.py`, `src/experiments/ViT.ipynb`, `src/lrp_engine/model_specific/mosaicbert.py`, `README.md`) — HIGH confidence on the integration mechanism. https://github.com/keeinlev/dynamicLRP
- "Always Keep Your Promises: A Model-Agnostic Attribution Algorithm for Neural Networks", arXiv 2512.07010 (v4) — HIGH on the Promise System / op-level autograd-graph approach; coverage claims do not include SigLIP-2. https://arxiv.org/html/2512.07010v4
- HuggingFace Transformers SigLIP2 model docs (`Siglip2Model`, `logits_per_image`/`logits_per_text`, `attn_implementation` configurable) — HIGH. https://huggingface.co/docs/transformers/model_doc/siglip2
- `google/siglip2-so400m-patch14-384` model card — MEDIUM (checkpoint uses SiglipVisionTransformer / Conv2d patch embedding). https://huggingface.co/google/siglip2-so400m-patch14-384
- Project inputs: `.planning/PROJECT.md`, `README.md`, `metadata/rumsey_manifest.json` (entry shape, GCS layout, count-down-from-N index semantics) — HIGH (authoritative project constraints).

---
*Architecture research for: dynamic-LRP map-attribution pipeline on SigLIP-2 (GCS + JupyterLab on GCP VM)*
*Researched: 2026-05-18*
