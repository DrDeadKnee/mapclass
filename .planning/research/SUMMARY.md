# Project Research Summary

**Project:** MapClass
**Domain:** Research/exploratory attribution pipeline — dynamic LRP over a vision-language model (SigLIP-2) for query-driven pixel heatmaps on historical map images; GCS-backed data, JupyterLab on a remote GCP VM
**Researched:** 2026-05-18
**Confidence:** MEDIUM-HIGH overall (stack and architecture HIGH; SigLIP-2 faithfulness as contrastive encoder MEDIUM)

## Executive Summary

MapClass is a single-researcher exploratory pipeline that applies dynamic LRP attribution to SigLIP-2-so400m-patch14-384 over historical Rumsey map images, producing query-driven heatmaps for visual inspection in JupyterLab. The entire project rests on one external dependency: `keeinlev/dynamicLRP` (arXiv 2512.07010), a PyTorch autograd-graph-level LRP engine that is genuinely model-agnostic. It has no pip package — it is installed by `git clone` and `sys.path` — and its `requirements.txt` pins (`torch==2.7.1`, `transformers==4.52.3`) are the hard constraints that every other dependency must satisfy, built outward from. The stack is therefore fixed: mirror that `requirements.txt` verbatim, add project-specific libraries on top, and never float the torch or transformers versions.

The central risk framing has been revised by direct inspection of the DynamicLRP paper (Table A.4): SigLIP-2-so400m-patch14-384 achieves 100% node coverage (2,178/2,178 nodes) in the paper's own coverage tables, so the engine is very likely to run without error. The real risk is **silent wrong attribution** — plausible-looking heatmaps that are actually query-independent because the attribution target was mis-specified. The correct target is the image-text similarity scalar (`logits_per_image[0,0]` — the dot product of the normalized image embedding with a detached, frozen text embedding for the query). Attributing from a raw image embedding, an embedding norm, or `model.get_image_features()` produces visually convincing but query-independent heatmaps that invalidate the entire deliverable. Additionally, SigLIP-2 has no CLS token — it uses a MAP (multi-head attention pooling) head — and 384/14 is not an integer, producing a 27×27 = 729 patch grid with the final 6 pixels of each edge discarded. Both facts must be explicitly handled in the overlay logic or all visual judgment is invalid.

Phase 1 must be gated on three near-zero-cost sanity controls — query-swap (different query, heatmap must change), model-randomization (randomized weights, heatmap must collapse to noise), and occlusion (top-relevance patches must reduce similarity more than random patches) — not on a heatmap "looking reasonable." These controls are correctness gates, not metrics, and do not violate the project's purely-visual-evaluation scope. The recommended build order is: pinned environment → idempotent GCS mirrors (images and model) → loaders → single-slice attribution (the central de-risk spike: reproduce the repo's `ViT.ipynb` first, then swap SigLIP-2 and the similarity target) → pass the three controls (Phase 1 gate) → sweep wrapper (pure orchestration, count down from high manifest index, v1 gate).

## Key Findings

### Recommended Stack

The load-bearing constraint is `keeinlev/dynamicLRP`. Its `requirements.txt` pins `torch==2.7.1` and `transformers==4.52.3`. These are non-negotiable: dynamicLRP operates on PyTorch's autograd graph internals, so a torch version change risks silent graph-shape drift; transformers 4.52.3 is verified to include SigLIP-2 (`siglip2` model landed at v4.49.0) and must not be floated to 5.x which has breaking API changes. The canonical project `requirements.txt` should be produced by copying `external/dynamicLRP/requirements.txt` verbatim and appending project-specific additions. The repo has no `setup.py` or `pyproject.toml` — `pip install git+...` will fail; use `git clone` pinned to a commit SHA, with the engine directory on `sys.path`.

**Core technologies:**
- `keeinlev/dynamicLRP` @ pinned commit SHA: the LRP engine — git clone only (no pip package); import `src/lrp_engine/` via sys.path or as a vendored submodule
- `torch==2.7.1` + `torchvision==0.22.1`: DL framework — hard pin from dynamicLRP requirements.txt; autograd Node names must match
- `transformers==4.52.3`: loads SigLIP-2 and provides `logits_per_image` — pinned by LRP repo; verified compatible with SigLIP-2
- `google/siglip2-so400m-patch14-384`: the fixed VLM — load via `AutoModel`/`AutoProcessor` (fixed-resolution variant, not NaFlex); load from GCS mirror, never from HF at runtime
- `google-cloud-storage>=3.0,<4`: GCS mirror access — uses existing VM ADC credentials; no extra auth
- `JupyterLab>=4.4,<5`: inspection surface — SSH-tunnelled; already prescribed by README

Supporting: `einops==0.8.1`, `timm==1.0.20`, `scikit-learn==1.7.0`, `omegaconf==2.3.0`, `tqdm==4.66.4`, `matplotlib==3.8.0`, `seaborn==0.13.2` (all pinned by dynamicLRP), `Pillow>=10.3,<12`, `accelerate>=0.30,<2`, `ipywidgets>=8.1,<9`, `captum>=0.7,<0.9`, `datasets>=3,<4`.

### Expected Features

**Must have (table stakes — Phase 1 and v1 scope):**
- GCS image mirror (idempotent, validated, per-id outcome manifest) — ingestion foundation
- GCS model mirror (idempotent, HF snapshot to GCS) — reproducibility on ephemeral VMs
- Manifest access with ordered index resolution — sweep semantics depend on it
- Image loader by ID from GCS with local byte cache — every attribution run starts here
- SigLIP-2 load from GCS mirror (once per kernel, eager attention, GPU) — central model
- Model-aligned preprocessing via official SigLIP-2 AutoProcessor (384x384 squash, [-1,1] rescale, NOT ImageNet normalization) that records the resize transform — overlay correctness
- Single forward + dynamicLRP attribution via `logits_per_image[0,0]` similarity scalar (detached text embedding) with `params_to_interpret=[img_tensor.requires_grad_()]` — the Core Value primitive
- Patch-relevance (729 tokens) reshaped to 27x27 grid accounting for 6-px edge discard and original-aspect un-squash — spatial interpretability
- Heatmap overlay (patch-resolution primary, labeled-interpolated secondary) on source map, displayed inline in JupyterLab — Phase 1 finish line
- Three sanity controls before Phase 1 sign-off: query-swap, model-randomization, occlusion — correctness gates
- Configurable sweep (maps counting down from index N x query set) — v1 scaling
- Contact-sheet notebook browse of all sweep overlays — v1 finish line
- Run caching of relevance grids keyed by (map id, query) as `.npy` — makes sweeps usable

**Should have (v1.x, after loop is trusted):**
- Provenance metadata sidecar per cached run (map id, query, model version, dynamicLRP commit SHA, LRP config)
- Resumable sweep with tqdm progress and skip-if-cached
- Side-by-side per-map query comparison view (layout variant of contact sheet)
- Overlay tuning params (alpha, colormap, percentile clip) as function arguments

**Defer (v2+):**
- Quantitative attribution metrics (faithfulness/ABPC, pixel-flipping, IoU) — needs trusted loop and ground-truth masks
- Model swapping as a first-class axis — each model is a separate LRP adaptation risk
- LRP parameter tuning as a first-class axis — explodes sweep space
- Open-source labeller integration (CVAT/Label Studio) — README phase 2
- Polished mask/dataset export — downstream milestone
- Other dataset categories — Rumsey only for v1

### Architecture Approach

The correct shape is a **small Python package of single-responsibility modules called from thin notebook cells**, with GCS as the only persistence layer. There is no service layer, no concurrent users, no HTTP API — "scale" means maps x queries in a sweep on one GPU VM. The single most important architectural invariant is tensor identity: the same `requires_grad_()` `img_tensor` object must flow from the data loader through the model forward and into `params_to_interpret` without being cloned or detached between modules. Breaking this identity destroys the autograd graph that dynamicLRP traverses. Consequently, the attribution engine must own both the forward pass and the LRP call together in one function; never split them across module boundaries.

**Major components:**
1. **manifest reader** (`manifest.py`) — parse `rumsey_manifest.json`; resolve entry by index; expose descending-index iteration; sweep over successfully-mirrored ids only
2. **ingest_images / mirror_model** (`ingest_images.py`, `mirror_model.py`) — run-once, idempotent, never re-enter the attribution loop; validate each download before writing to GCS
3. **model loader** (`model_loader.py`) — pull GCS model mirror to local disk; load SigLIP-2 with `attn_implementation="eager"`; singleton cached per kernel
4. **data loader** (`data_loader.py`) — GCS fetch to PIL to SigLIP-2 AutoProcessor to `img_tensor.requires_grad_()`; returns tensor AND original PIL; local disk cache on first fetch
5. **attribution engine** (`attribution.py`) — owns forward + dynamicLRP in one scope; target = `logits_per_image[0,0]` with detached text embedding; returns per-patch relevance; free graph + `empty_cache()` after each call
6. **overlay / viz** (`overlay.py`) — reshape 729 to 27x27 accounting for 6-px discard and aspect un-squash; patch-resolution primary view; labeled-interpolated secondary view
7. **sweep orchestrator** (`sweep.py`) — iterate subset x queries counting DOWN from high index N; per-item try/except; collect (id, query, overlay) tuples; no new attribution logic

### Critical Pitfalls

1. **Wrong attribution target (contrastive encoder, not classifier)** — do NOT attribute from a raw image embedding, embedding norm, or `model.get_image_features()`; attribute from `logits_per_image[0,0]` = dot product of normalized image embedding with a DETACHED, frozen text embedding for the query; keep L2-normalization inside the traced graph; sanity-test: swap to an unrelated query — the heatmap must change substantially or the target is wrong.

2. **Silent wrong attribution mistaken for correct because it looks plausible** — "looks reasonable" is not a correctness gate; run all three controls before Phase 1 sign-off: (a) query-swap: heatmap changes with query; (b) model-randomization: randomized vision weights collapse heatmap to noise; (c) occlusion: masking top-relevance patches reduces image-text similarity more than masking random patches.

3. **MAP attention-pooling head routing relevance incorrectly** — SigLIP-2 has no CLS token; the image embedding comes from a MAP head that attends over 729 patch tokens; the paper validates SigLIP-2 graph coverage but not faithfulness for this head specifically; extract relevance at patch tokens BEFORE the MAP head, not at the final embedding; verify with the occlusion control.

4. **Patch-to-pixel overlay misaligned due to 384/14 non-integer and aspect squash** — 384/14 = 27.4, floor-div gives 27x27 = 729 patches, last 6 pixels of right and bottom edges are DISCARDED (confirmed by model author as an unfixed bug); naive bilinear 27x27 to 384x384 upsample produces a 6-px spatial offset and misleading smooth blobs; map the heatmap onto the exact 378x378 valid region, then un-squash back to original aspect ratio; use nearest-neighbor or block upsample for the primary honest view.

5. **Unpinned dynamicLRP and ephemeral-VM reproducibility** — the repo is pre-publication, actively changing (last pushed 2026-05-06); `git clone @master` means the attribution algorithm can change between VM rebuilds; pin to a specific commit SHA; load SigLIP-2 only from the GCS model mirror, never HF at runtime; stamp every heatmap with map id, query, model mirror version, dynamicLRP SHA, and LRP config.

6. **Non-idempotent or unvalidated GCS image ingestion** — 1,544 Rumsey URLs against a third-party server with no SLA from an ephemeral VM; validate each download (HTTP 200 + content-type + byte length + decodable) before writing to GCS; write to temp then atomic upload; skip-if-present-and-valid on re-run; maintain a per-id outcome manifest; sweep must only iterate successfully-mirrored ids.

7. **GPU memory blowup under LRP on so400m** — dynamicLRP's Promise system retains/recovers forward activations, materially increasing VRAM vs. inference alone; batch size 1 for attribution always; measure peak VRAM on the first single-image attribution; free the LRP graph and call `torch.cuda.empty_cache()` between iterations; assert VRAM returns to baseline each sweep iteration.

## Implications for Roadmap

Based on combined research, the recommended phase structure has two phases driven by a strict dependency chain: the attribution primitive must be verified correct (with controls) before any sweep has value.

### Phase 1: Pinned Environment, Mirrors, and Verified Single-Slice Attribution

**Rationale:** Everything downstream depends on the attribution primitive being correct. The de-risk spike (reproduce ViT.ipynb, then swap SigLIP-2 + similarity target) must happen before any sweep or overlay polish. The three sanity controls are the gate, not a heatmap looking reasonable. Pinned environment and idempotent GCS mirrors are prerequisites for any other step and are fast to validate.

**Delivers:** A verified, correct single-slice attribution pipeline — one map + one query produces a properly-aligned heatmap that has passed query-swap, model-randomization, and occlusion controls. All critical pitfalls (target spec, overlay alignment, memory) are addressed at this stage.

**Addresses (features from FEATURES.md):**
- Pinned environment (dynamicLRP SHA, requirements.txt mirrored from LRP repo)
- Idempotent GCS image mirror + model mirror (with validation and per-id outcome manifest)
- Manifest reader + image loader + model loader
- Model-aligned preprocessing via official AutoProcessor (records resize transform)
- Single forward + dynamicLRP attribution with correct similarity target
- 27x27 patch-relevance reconstruction accounting for 6-px edge discard + aspect un-squash
- Heatmap overlay on source map, inline notebook display
- Three sanity controls passing (query-swap, model-randomization, occlusion)

**Avoids (pitfalls from PITFALLS.md):** Pitfalls 1-5, 7-8 — all the ways the core primitive can be silently wrong, misaligned, or non-reproducible.

**Research flag:** NEEDS DEEPER RESEARCH. dynamicLRP's SigLIP-2 faithfulness as a contrastive encoder is unvalidated in the paper. The de-risk spike must reproduce the repo's ViT.ipynb end-to-end first, then swap in SigLIP-2 and the similarity target. If op coverage fails on SigLIP-2's ops, add a custom promise/rule using `model_specific/mosaicbert.py` as the pattern. If MAP-head relevance is distorted, extract at pre-pool patch tokens. Last resort: fall back to vendored LXT or captum Integrated Gradients.

### Phase 2: Configurable Sweep and Contact-Sheet Browse (v1 Done)

**Rationale:** The sweep is pure orchestration over the proven Phase 1 primitive — zero new attribution logic. It must only begin once Phase 1 controls have passed. Building the sweep before the slice is trusted hides adaptation bugs behind 100 plausible-looking-but-wrong heatmaps.

**Delivers:** A configurable maps x queries sweep counting down from high manifest index N, with all results browsable as a contact-sheet grid in the notebook. Run caching of relevance grids makes re-renders cheap. This is the v1 finish line.

**Addresses (features from FEATURES.md):**
- Configurable sweep (map subset x query set, descending index from N, over successfully-mirrored ids only)
- Contact-sheet notebook browse of all sweep overlays
- Run caching of relevance grids as `.npy` keyed by (map id, query)
- Per-item try/except to avoid one bad map killing the sweep

**Avoids (pitfalls from PITFALLS.md):** Pitfall 6 (partial mirror producing silent sweep gaps), Pitfall 5 (GPU memory leak across sweep iterations — empty_cache + baseline assert per iteration), Pitfall 7 (provenance stamped on each tile).

**Research flag:** STANDARD PATTERNS — no additional research needed. The sweep is a nested loop over the Phase 1 function. The only implementation care is per-iteration memory cleanup (Pitfall 5) and sweeping only successfully-mirrored ids from the ingestion outcome manifest (Pitfall 6).

### Phase Ordering Rationale

- The attribution primitive is the single dependency that gates all other value; verifying it correct (not just running) is the primary risk and must be resolved first.
- The three sanity controls are not optional scope — they are the mechanism by which "visually trusted" acquires meaning; running them costs near-zero engineering effort and prevents scaling a broken loop.
- The build order within Phase 1 is itself dependency-ordered: (1) pinned env, (2) idempotent mirrors, (3) loaders, (4) de-risk spike (ViT.ipynb reproduction then SigLIP-2 + similarity target), (5) overlay with correct 27x27/6px/aspect handling, (6) three controls pass = Phase 1 gate.
- Phase 2 adds only iteration and caching; it cannot and should not begin until Phase 1 is gated.
- v1.x features (provenance sidecars, resumable sweep, comparison views, overlay tuning) fall out of the sweep phase almost for free once caching exists; they are not a separate phase.

### Research Flags

**Phase 1 — needs hands-on research during planning:**
- **SigLIP-2 + dynamicLRP de-risk spike:** no published CLIP/SigLIP dynamicLRP example exists; paper tests SigLIP-2 as a vision-only classifier, not as a contrastive encoder; the similarity-target wiring and MAP-head relevance extraction are novel; plan for the possibility of needing a custom promise/rule; allocate time for the ViT.ipynb reproduction step before SigLIP-2 swap.
- **MAP attention-pooling head relevance topology:** the paper does not validate faithfulness for MAP-pooling architectures; the occlusion control is the only faithfulness check available; plan to extract relevance at pre-pool patch tokens.
- **27x27 / 6-px / aspect-ratio overlay chain:** non-trivial coordinate math; plan an explicit alignment check (overlay the patch grid on one map, confirm cell placement and right/bottom discard) before trusting any visual judgment.

**Phase 2 — standard patterns (skip research-phase):**
- Sweep orchestration is a simple nested loop over the Phase 1 function; contact-sheet is a matplotlib subplot grid; run caching is keyed `.npy` writes; no novel integration risk.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | dynamicLRP repo, requirements.txt, and ViT.ipynb inspected directly; PyPI versions verified 2026-05-18; transformers/SigLIP-2 compatibility verified against HF release notes |
| Features | MEDIUM-HIGH | Table-stakes/anti-feature split is well-grounded in PROJECT.md scope; dynamicLRP API is README-level (not full source walk for every detail) |
| Architecture | HIGH (integration mechanism); MEDIUM (SigLIP-2 adaptation specifics) | Integration mechanism verified against lrp_engine/lrp.py source and ViT.ipynb; SigLIP-2 dual-encoder adaptation has no published example to validate against |
| Pitfalls | MEDIUM-HIGH | LRP VRAM, contrastive-target, and reproducibility pitfalls verified against paper/repo; MAP-head faithfulness and 6-px discard verified against HF model author confirmation; contrastive attribution semantics from XAI literature (MEDIUM) |

**Overall confidence:** MEDIUM-HIGH

### Gaps to Address

- **dynamicLRP faithfulness on SigLIP-2 as a contrastive encoder:** the paper's SigLIP-2 row is graph coverage only (it runs), not faithfulness (it's correct for query-driven attribution); cannot be resolved by research alone — requires running the three sanity controls on actual output during Phase 1; plan for and DO NOT skip the controls.
- **Custom promise/rule for SigLIP-2-specific ops (if needed):** if `LRPEngine.get_model_operations(out)` reports uncovered ops on SigLIP-2's attention pooling or sigmoid loss head, a shim will be needed; use `model_specific/mosaicbert.py` as the pattern; this is a real but bounded contingency.
- **Peak VRAM budget for so400m + LRP:** actual peak for so400m is unknown until first run; measure on the first single-image attribution and verify it fits the VM's GPU before planning sweep sizes.
- **Rumsey URL health:** some of the 1,544 manifest URLs may be dead or rate-limited; the per-id outcome manifest from ingestion is the only way to know how many maps are actually available; sweep planning should budget for a fraction of the manifest being unavailable.

## Sources

### Primary (HIGH confidence)
- `github.com/keeinlev/dynamicLRP` (repo tree, `README.md`, `requirements.txt`, `src/lrp_engine/lrp.py`, `src/experiments/ViT.ipynb`, `src/lrp_engine/model_specific/mosaicbert.py`) — fetched 2026-05-18
- arXiv 2512.07010 ("Always Keep Your Promises: A Model-Agnostic Attribution Algorithm for Neural Networks") — full paper including Table A.4 (SigLIP-2 100% node coverage), Appendix F (covered ops), sections 3.1-3.2 (Promise system VRAM), section 5 (occlusion faithfulness test)
- `huggingface.co/google/siglip2-so400m-patch14-384` model card + HF Transformers SigLIP2 docs + HF transformers v4.49.0-SigLIP-2 release notes — SigLIP-2 architecture, `logits_per_image`, AutoProcessor, MAP head, min transformers version
- `huggingface.co/google/siglip-so400m-patch14-384/discussions/4` — model author (@giffmana) confirmation of 384/14 floor-div, 27x27 patches, 6-px edge discard, unfixed
- `.planning/PROJECT.md`, `README.md`, `metadata/rumsey_manifest.json` — authoritative project constraints

### Secondary (MEDIUM confidence)
- Captum (`captum.ai`) — overlay visualization pattern (`visualize_image_attr`), IG as a fallback baseline
- AttnLRP arXiv 2402.05602 + LXT docs — context on transformer LRP and attention decomposition; basis for MAP-head risk framing
- Adebayo et al. sanity-checks for saliency methods — model-randomization control rationale
- arXiv 2502.14786 (SigLIP-2 paper), emergentmind SigLIP-2 vision encoder description — 27-layer ViT, MAP head, sigmoid loss details

### Tertiary (LOW confidence)
- Zennit (hook-based LRP) — referenced as the alternative that dynamicLRP exists to avoid; not used directly
- pypi.org JSON API versions (queried 2026-05-18) — version availability only

---
*Research completed: 2026-05-18*
*Ready for roadmap: yes*
