# Feature Research

**Domain:** Exploratory model-attribution pipeline (dynamic LRP over a VLM, SigLIP-2, for query-driven heatmaps on historical maps; visual inspection in JupyterLab on a remote GCP VM)
**Researched:** 2026-05-18
**Confidence:** MEDIUM-HIGH

The domain is well-understood from established tooling: Captum (PyTorch interpretability), Zennit (hook-based LRP), LXT / AttnLRP (transformer LRP), and the specific reference dependency keeinlev/dynamicLRP (arXiv 2512.07010). These tools converge on a common shape of capabilities, which makes the table-stakes / differentiator split clear. Confidence is HIGH for the general feature landscape and MEDIUM for dynamicLRP-specific behavior (its README documents an `LRPEngine().run(logits)` API but does not ship visualization, batching, or caching helpers — those are the project's own piping to build).

## Feature Landscape

The "user" here is a single researcher running a notebook on a remote VM. "Table stakes" = the loop physically cannot deliver its Core Value without it. "Differentiators" = makes the loop materially better but the v1 finish line is reachable without it. "Anti-features" = explicitly out of scope per PROJECT.md; building them is scope creep that delays trusting the loop.

### Table Stakes (Without These the Loop Doesn't Work)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Image loader by ID from GCS data mirror | Every attribution run starts from a map; mirror exists for VM stability (PROJECT.md) | LOW | Resolve manifest `id` → GCS object → PIL/tensor. Local disk cache of fetched bytes avoids repeated GCS reads in a sweep. **Phase 1.** |
| Manifest access + index→entry resolution | Sweep semantics ("count down from high index N") require ordered manifest indexing | LOW | Load `metadata/rumsey_manifest.json` (1,544 entries); expose ordered list so index N maps to a specific map. **Phase 1.** |
| Model load (SigLIP-2-so400m) from GCS model mirror | The fixed VLM; mirrored to avoid HF auth at runtime (PROJECT.md) | MEDIUM | HF `transformers` load from local path synced from GCS; preprocessor for patch14-384. Load once, reuse across sweep (do not reload per map). **Phase 1.** |
| Preprocessing aligned to model (resize/normalize to 384, patchify) | Attribution map only makes sense if input is the model's actual input; misalignment silently corrupts overlays | MEDIUM | Must match SigLIP-2 image processor exactly; record the resize transform so the heatmap can be mapped back to source-image coordinates. **Phase 1.** |
| Single forward + dynamic-LRP attribution for one (map, query) | This *is* the Core Value primitive | HIGH | Adapt keeinlev/dynamicLRP's `LRPEngine().run(logits)` to SigLIP-2's image-text architecture. PROJECT.md names this the central technical risk. Reduce image-token relevance to a per-patch grid. **Phase 1.** |
| Patch-relevance → 2D heatmap reconstruction | Raw relevance is a token vector; researcher needs a spatial map | MEDIUM | Reshape patch relevances to the patch grid (e.g. 27×27 for 384/14), upsample to image size. Decide a fixed normalization (per-image min-max or signed) and keep it constant so sweeps are comparable. **Phase 1.** |
| Heatmap overlaid on source map | Judgment is "does the hot region match the queried feature on *this* map" — requires the map underneath | LOW | matplotlib `imshow` map + alpha-blended colormap (Captum's `visualize_image_attr` overlay mode is the standard pattern). **Phase 1.** |
| Inline notebook display of the overlay | Inspection happens in JupyterLab over SSH tunnel; no GUI/file-export path in v1 | LOW | Return/display a matplotlib figure in the notebook cell. **Phase 1 (single slice).** |
| Configurable sweep: map subset × query set | v1 finish line is a sweep, not a single slice | MEDIUM | Iterate maps counting *down* from a high manifest index N (configurable count + start), cross with a configurable query list. Pure orchestration over the Phase 1 primitive. **Later in v1.** |
| Contact-sheet browse of all sweep overlays in notebook | v1 "done" = browse all heatmaps contact-sheet style (PROJECT.md) | MEDIUM | Grid of subplots (rows=maps, cols=queries) with map/query labels. matplotlib subplot grid; mind figure size / memory for larger sweeps. **Later in v1 — the v1 finish line.** |
| Run caching / artifact persistence | A sweep over many maps × queries is slow (GPU forward + LRP backward each); recomputing on every notebook re-run makes iteration painful and is the difference between a usable and an unusable loop | MEDIUM | Cache keyed by (map id, query, fixed model/LRP config) → store the relevance grid (small `.npy`) to local disk or GCS. Lets the notebook re-render contact sheets without recompute. **Later in v1** (Phase 1 can run uncached; caching pays off once sweeps exist). |

### Differentiators (Better Loop, Not Required for v1 Finish Line)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Interactive query widget (ipywidgets text box + map selector) | Re-query without editing cells; tightens the explore loop | MEDIUM | Nice for ad-hoc exploration; the sweep + contact sheet already satisfies the stated v1 goal. Likely v1.x. |
| Side-by-side query comparison for one map | Seeing "ocean" vs "mountains" vs "text label" attribution on the same map side-by-side is highly informative for judging fidelity | LOW-MEDIUM | Largely a layout variant of the contact sheet (fix map, vary query). Cheap if contact-sheet exists; reasonable v1.x. |
| Overlay tuning controls (alpha, colormap, signed vs abs, percentile clipping) | Attribution maps are often noisy; clipping/colormap choices change whether a human can read them | LOW | Expose as function params with sane fixed defaults for v1; surface as widgets in v1.x. Keep defaults fixed during a sweep for comparability. |
| Per-run provenance metadata (map id, query, model/LRP config, timestamp) recorded with artifact | Reproducibility; lets the researcher trust "this heatmap came from these exact settings" | LOW | Small JSON sidecar next to each cached relevance grid. Pairs naturally with run caching. |
| Sweep progress / resumability | Long sweeps on a remote VM benefit from progress feedback and skip-if-cached resume | LOW-MEDIUM | `tqdm` + cache-hit skip. Becomes valuable only once sweeps are large; falls out almost free from run caching. |
| Token/query-string attribution sanity readout (top patches, score range) | Quick numeric gut-check that LRP produced non-degenerate output before eyeballing 50 overlays | LOW | A debugging aid, not a metric (no faithfulness scoring). Helps detect the dynamicLRP-on-SigLIP-2 adaptation failing silently. |

### Anti-Features (Deliberately Out of Scope for This Research Spike)

Aligned 1:1 with PROJECT.md "Out of Scope." These look reasonable but delay trusting the loop and contradict the v1 charter.

| Feature | Why Requested | Why Problematic | Alternative |
|---------|---------------|-----------------|-------------|
| Quantitative attribution metrics (faithfulness/ABPC, pixel-flipping, IoU vs ground truth) | "Numbers feel rigorous"; the dynamicLRP paper reports such metrics | PROJECT.md: v1 judgment is purely visual; metrics add scope before the loop is trusted, and need ground-truth masks that don't exist yet | Eyeball overlays; revisit metrics only after the loop is visually trusted |
| Model swapping as a first-class axis (Florence-2, PaliGemma, ViT, etc.) | README lists many candidate models | PROJECT.md: model fixed to SigLIP-2 in v1 to keep the loop small; each model needs its own LRP adaptation (the central risk multiplied) | Hardcode SigLIP-2; design loaders so a swap is *possible* later but not a v1 knob |
| LRP method/parameter tuning as a first-class axis (rule choice, gamma, filtering thresholds) | Tuning often improves heatmap readability | PROJECT.md: LRP config fixed in v1; vary map + query only. Tuning explodes the sweep space and confounds visual judgment | Pick one reasonable dynamicLRP config, freeze it, document it |
| Open-source labeller integration (CVAT/Label Studio export) | README's "phase 2"; ultimate goal is pixel labels | PROJECT.md: depends on the exploration loop existing and being trusted first | Defer to phase 2 milestone after v1 loop validated |
| Polished mask/dataset export (binary masks, COCO/PNG label maps) | "We'll need the labels eventually" | PROJECT.md: dataset export is a downstream milestone, not v1 piping; premature format choices ossify | Keep relevance grids as raw `.npy`; convert later when label spec is known |
| Fine-tuning / any model training | The abandoned prior approach | PROJECT.md: explicitly the thing this restart replaces; dynamic LRP is "better and less work" | None — strictly forbidden in v1 |
| Other dataset categories (Toons, Fantasy, MapMaker, Satellite) | README enumerates 5 categories | PROJECT.md: Rumsey historical only for v1; other categories add manifest/loader variance with no loop payoff | Rumsey manifest only; loader can generalize later |
| Web app / dashboard / served UI | "Nicer than a notebook" | Contradicts the remote-VM + JupyterLab-over-SSH-tunnel constraint; adds infra with zero research payoff | Notebook display only; ipywidgets is the ceiling of "interactivity" for v1 |
| Multi-GPU / distributed sweep orchestration | "Sweeps are slow" | Premature optimization; 1,544 maps and a single GPU VM is the stated runtime; LRP overhead is bounded by model size choice | Serial sweep + run caching + resumable skip-if-cached |

## Feature Dependencies

```
Manifest access (index→entry)
    └──requires──> nothing (local file)

Image loader (GCS)              Model load (GCS)
    └──requires──> nothing          └──requires──> nothing
            \                          /
             \                        /
              v                      v
        Preprocessing (model-aligned, records resize transform)
                        |
                        v
        Single forward + dynamic-LRP attribution   <-- CENTRAL RISK
                        |
                        v
        Patch-relevance → 2D heatmap reconstruction
                        |
                        v
        Heatmap overlay on source map ──> Inline notebook display   [PHASE 1 DONE]
                        |
                        v
        Configurable sweep (map subset × query set)
                        |
            +-----------+-----------+
            v                       v
   Run caching / artifacts   Contact-sheet browse   [v1 DONE]
            |                       ^
            +--enhances-------------+
            |
            v
   Provenance metadata, resumable sweep  (v1.x, fall out of caching)

Interactive query widget ──enhances──> single-slice + sweep loop  (v1.x)
Side-by-side comparison  ──is-a-layout-of──> contact-sheet         (v1.x)
```

### Dependency Notes

- **Everything requires the attribution primitive.** The dynamicLRP→SigLIP-2 adaptation gates the entire project; if it fails, no downstream feature has value. This is why Phase 1 is a single slice — prove the risky primitive before any orchestration.
- **Preprocessing must record its resize transform.** The overlay step needs to map a patch-grid heatmap back onto the original (non-384) image; if the resize is lossy and unrecorded, overlays misalign and visual judgment is invalid. Cheap to do correctly in Phase 1, expensive to retrofit.
- **Sweep requires the single-slice primitive working end-to-end.** Sweep is pure orchestration over Phase 1's function; building sweep before the slice works wastes effort on the wrong risk.
- **Contact-sheet and run caching co-evolve.** Caching makes re-rendering the contact sheet cheap; the contact sheet is what makes a sweep worth caching. They are the v1 finish line pairing.
- **Provenance and resumability fall out of run caching almost free** — same cache key, add a JSON sidecar and a skip-if-exists check.
- **Comparison view conflicts with nothing** — it is the contact sheet with map fixed and query varied; do not build a separate code path.

## MVP Definition

### Launch With (v1)

- [ ] Manifest access with ordered index resolution — sweep semantics depend on it
- [ ] Image loader by ID from GCS mirror (with local byte cache) — every run needs the map
- [ ] SigLIP-2-so400m load from GCS model mirror (loaded once) — the fixed model
- [ ] Model-aligned preprocessing that records the resize transform — overlay correctness
- [ ] Single forward + dynamic-LRP attribution for one (map, query) — **the Core Value primitive; central risk**
- [ ] Patch-relevance → 2D heatmap with fixed normalization — spatial interpretability
- [ ] Heatmap overlay on source map, displayed inline in JupyterLab — **Phase 1 finish line**
- [ ] Configurable sweep: maps counting down from index N × query set — v1 scaling
- [ ] Contact-sheet notebook browse of all sweep overlays — **v1 finish line**
- [ ] Run caching of relevance grids keyed by (map id, query) — makes sweeps and re-render usable

### Add After Validation (v1.x)

- [ ] Provenance metadata sidecar per cached run — trigger: loop trusted, reproducibility matters
- [ ] Resumable / skip-if-cached sweep with `tqdm` progress — trigger: sweeps grow large enough to interrupt
- [ ] Side-by-side per-map query comparison view — trigger: researcher wants focused multi-query reads
- [ ] Overlay tuning params surfaced (alpha, colormap, percentile clip) — trigger: default overlays hard to read
- [ ] ipywidgets interactive query/map selector — trigger: cell-editing friction during exploration
- [ ] Attribution sanity readout (top patches, score range) — trigger: silent adaptation failures suspected

### Future Consideration (v2+ / later milestones)

- [ ] Quantitative attribution metrics — defer: needs trusted loop + ground truth, explicitly out of v1 scope
- [ ] Model swapping axis — defer: each model is its own LRP adaptation risk
- [ ] LRP parameter tuning axis — defer: explodes sweep space, confounds visual judgment
- [ ] Open-source labeller integration — defer: README phase 2, depends on trusted loop
- [ ] Polished mask/dataset export — defer: downstream milestone, label spec not yet known
- [ ] Other dataset categories — defer: Rumsey-only for v1

## Feature Prioritization Matrix

| Feature | Researcher Value | Implementation Cost | Priority |
|---------|------------------|---------------------|----------|
| Single forward + dynamic-LRP attribution (SigLIP-2 adaptation) | HIGH | HIGH | P1 |
| Model-aligned preprocessing (records resize transform) | HIGH | MEDIUM | P1 |
| Patch-relevance → 2D heatmap reconstruction | HIGH | MEDIUM | P1 |
| Heatmap overlay on source map | HIGH | LOW | P1 |
| Inline notebook display | HIGH | LOW | P1 |
| Image loader by ID from GCS mirror | HIGH | LOW | P1 |
| Manifest access / index resolution | HIGH | LOW | P1 |
| Model load from GCS mirror | HIGH | MEDIUM | P1 |
| Configurable sweep (map subset × query set) | HIGH | MEDIUM | P1 |
| Contact-sheet notebook browse | HIGH | MEDIUM | P1 |
| Run caching of relevance grids | HIGH | MEDIUM | P1 |
| Provenance metadata sidecar | MEDIUM | LOW | P2 |
| Resumable / progress sweep | MEDIUM | LOW-MEDIUM | P2 |
| Side-by-side comparison view | MEDIUM | LOW-MEDIUM | P2 |
| Overlay tuning params | MEDIUM | LOW | P2 |
| ipywidgets interactive selector | MEDIUM | MEDIUM | P2 |
| Attribution sanity readout | MEDIUM | LOW | P2 |
| Quantitative metrics | LOW (v1) | HIGH | P3 |
| Model swapping axis | LOW (v1) | HIGH | P3 |
| LRP parameter tuning axis | LOW (v1) | MEDIUM | P3 |
| Labeller integration | LOW (v1) | HIGH | P3 |
| Mask/dataset export | LOW (v1) | MEDIUM | P3 |

**Priority key:** P1 = required for v1 launch · P2 = add after the loop is visually trusted (v1.x) · P3 = future milestone, explicitly out of v1 scope per PROJECT.md.

## Competitor Feature Analysis

These are reference toolkits, not competitors — they define the standard shape of an attribution exploration tool and show which capabilities are conventional vs custom.

| Feature | Captum | Zennit / LXT (AttnLRP) | keeinlev/dynamicLRP | Our Approach |
|---------|--------|------------------------|---------------------|--------------|
| Attribution engine | Many methods (IG, saliency, LRP-ish) | Hook-based LRP, transformer rules | Op-level model-agnostic LRP (`LRPEngine().run`) | Use dynamicLRP, adapt to SigLIP-2 |
| Image overlay viz | `visualize_image_attr` (overlay mode, alpha) | Minimal; user-built | Notebook examples only; no shipped viz | Build matplotlib overlay (Captum-style) |
| Batching/sweep | User-orchestrated | User-orchestrated | Not provided | Build configurable map×query sweep |
| Run caching | Not provided | Not provided | Not provided | Build `.npy` cache keyed by (map, query) |
| Notebook contact sheet | Insights tool (interactive, heavier) | Not provided | Not provided | Build lightweight subplot grid |
| Interactive widgets | Captum Insights (web UI) | Not provided | Not provided | Defer; ipywidgets at most (v1.x) |
| Metrics/faithfulness | Some (infidelity, sensitivity) | Faithfulness in papers | Reports ABPC/accuracy in paper | Explicitly excluded from v1 |

Takeaway: visualization, batching, and caching are *always* the consumer's own piping — no toolkit ships them turnkey. That is exactly what this project builds. The risky, non-standard part is adapting op-level LRP to SigLIP-2's dual-encoder image-text architecture.

## Sources

- keeinlev/dynamicLRP — https://github.com/keeinlev/dynamicLRP (reference dependency; `LRPEngine` API, no shipped viz/batch/cache) — MEDIUM confidence (README-level)
- "Always Keep Your Promises: A Model-Agnostic Attribution Algorithm for Neural Networks" — https://arxiv.org/abs/2512.07010 (dynamicLRP paper; op-level LRP, multimodal coverage incl. DePlot) — HIGH
- Captum — https://github.com/meta-pytorch/captum and https://captum.ai/api/saliency.html (standard PyTorch interpretability; `visualize_image_attr` overlay pattern, Insights interactive tool) — HIGH
- AttnLRP — https://arxiv.org/abs/2402.05602 and LXT https://lxt.readthedocs.io/ (transformer-aware LRP; attention decomposition context) — HIGH
- Zennit (hook-based LRP in PyTorch) — referenced via AttnLRP/LXT lineage — MEDIUM
- Matplotlib heatmap/overlay docs — https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html (contact-sheet/grid + imshow overlay pattern) — HIGH
- Project context: .planning/PROJECT.md (Out of Scope, v1 explore axes, finish lines) and README.md (environment, datasets, candidate models) — HIGH

---
*Feature research for: exploratory dynamic-LRP-over-VLM attribution pipeline (SigLIP-2, historical maps, JupyterLab)*
*Researched: 2026-05-18*
