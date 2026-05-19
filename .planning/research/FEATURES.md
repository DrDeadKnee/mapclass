# Feature Research

**Domain:** Cross-model dynamic-LRP attribution comparison harness (research piping, notebook-only) — MapClass v1.1
**Researched:** 2026-05-19
**Confidence:** HIGH (grounded in PROJECT.md v1.1 scope, v1.0 01-03 shipped code/SUMMARY, and direct inspection of the vendored dynamicLRP engine)

> Supersedes the 2026-05-18 v1.0 single-model / sweep feature research. v1.1
> drops the maps×queries sweep and contact-sheet entirely; the comparison axis
> is now **model**, not map count.

## Scope Anchor

This is **not** a market-facing product. "Table stakes / differentiator /
anti-feature" here means: *what the v1.1 comparison harness must do to be a
usable research artifact* vs. *what makes the comparison genuinely informative*
vs. *what is scope creep the user has already ruled out*. The user's explicit
deliverable is **"notebook overlays only — NO structured op-coverage report
table; coverage is implicit in whether a heatmap renders; map locked; sweep
dropped."** Every categorization below is tied to that boundary.

## The Feature-Defining Question: Per-Architecture Attribution Target

This is the single most important design decision in v1.1 and it is **not yet
resolved by existing scope.** "Run dynamic LRP for one map + a text query across
4 models" silently assumes the four models accept the same attribution target.
**They do not.** The "query" means a structurally different thing per
architecture:

| Model | Architecture class | Query-conditioned attribution target | Notes |
|-------|--------------------|--------------------------------------|-------|
| SigLIP-2-so400m-patch14-384 | Contrastive, **MAP-pool** (attention-pooling) head | `output.logits_per_image[0,0]` (image–text similarity scalar) | v1.0 path; **known to fail** at `SplitWithSizesBackward0` in the MAP-pool head — a recorded negative datum, *not to be fixed* |
| CLIP (ViT-B/32 or ViT-L/14) | Contrastive, **CLS-token** pooling | `logits_per_image[0,0]` — same contrastive API as SigLIP-2 | The cleanest analogue to SigLIP-2; differs *precisely* in the pooling head (CLS, no `split_with_sizes`) → the single most informative comparison point |
| PaliGemma (3B-class) | **Generative** VLM (autoregressive prefix-LM) | **No `logits_per_image` exists.** Target must be a token logit: the logit of a chosen answer/next token given image+prompt | The "query" is a generated-token logit, not a similarity. Largest semantic gap; largest model (size-ceiling tension vs PROJECT.md so400m ceiling) |
| Plain ViT (`vit-base-patch16-224`-class) | **Classification-only** (ImageNet head), no text encoder | **No text query at all.** Target is a class logit `logits[0, class_idx]` | The "query" is a class label, not free text. Either fix a class, or treat ViT as a *method-sanity reference* — it is the architecture dynamicLRP's own paper/README demonstrably covers |

**Roadmap implication:** the adapter contract cannot be "given model id →
`logits_per_image`". It must be **"given model id → (load, forward,
select_target, declare_target_semantics)"** where `select_target` is
*per-architecture* and each adapter states in plain words what its "query" binds
to. Rendering that statement as a per-panel label is itself table stakes —
without it the side-by-side silently compares incommensurable quantities (a
cosine similarity vs. a token probability vs. a class logit).

A second consequence: dynamicLRP coverage is **per-target-op, not per-model.**
The vendored engine ships generic op promises (`split_backward`, `cat_backward`,
`softmax_backward`, `unbind`, `stack`, …) and exactly **one** model-specific
shim (`model_specific/mosaicbert.py`) — there is **no per-architecture code for
ViT, CLIP, SigLIP, or PaliGemma.** Coverage success/failure is therefore a
property of *which ops the chosen target's autograd subgraph traverses*. This is
exactly why CLIP (CLS pooling) may traverse where SigLIP-2 (MAP-pool
`split_with_sizes`) provably does not — and that contrast *is the experiment*.

## Feature Landscape

### Table Stakes (The Harness Is Incomplete Without These)

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Model-agnostic adapter contract: `model_id → {load, forward, select_target, declare_semantics}` | "Across 4 models" is impossible without a uniform per-model entry point; v1.0 `attribute()` hardwires SigLIP-2's `logits_per_image` and must be generalized | MEDIUM | Reuse v1.0 Pattern 2 (forward + `LRPEngine.run` in ONE scope; SAME `img_tensor` object into forward AND `params_to_interpret`) per adapter. Variability is confined to `select_target` + load class. ~4 thin adapters. Depends on v1.0 `model_loader`/`data_loader`/`attribution` patterns |
| Per-architecture query-conditioned target selection (the feature-defining question above) | Contrastive similarity, generated-token logit, and class logit are different targets; wrong choice silently yields image-saliency or nonsense | MEDIUM–HIGH | The genuine design risk. CLIP=`logits_per_image`; PaliGemma=answer-token logit; ViT=class logit; SigLIP-2=v1.0 path. Each adapter must declare its target in words |
| Graceful per-model degradation: catch coverage/engine failure → labeled "no heatmap (op-coverage gap)" tile → notebook exit 0 | The explicit v1.1 done-criterion. v1.0 already proved the caught-`RuntimeError`→FINDING-cell→exit-0 pattern for SigLIP-2 | LOW | Generalize v1.0's caught-finding pattern over a per-model loop. A failing model must not abort the other 3 panels. Catch broadly — the engine fails as `IndexError`/`TypeError`/`RuntimeError` depending on the op |
| Side-by-side rendering surface: 4 panels in one figure, source map shown once | "Side-by-side" is the literal deliverable; a comparison the human can't see at a glance is not a comparison | LOW–MEDIUM | One matplotlib figure: source-map-once (as a reference panel or header) + 4 overlay panels. Each panel titled: model + target semantics + render status |
| Consistent D-09 signed/zero-centered colormap reused across all panels | v1.0 established signed relevance on `TwoSlopeNorm(vmin=-M,vcenter=0,vmax=M)`, `bwr`, no `.abs()`, no min-max. Mixing conventions across panels makes the comparison lie | LOW | Reuse `src/mapclass/overlay.py` `to_patch_grid`/`composite` unchanged. The only *new* decision is the cross-model **scale** (see differentiators) |
| Per-model patch-grid reconstruction (grid size differs per model) | v1.0's 27×27 (`floor(384/14)`, 6-px discard) is SigLIP-2-specific. CLIP ViT-B/32@224→7×7; ViT-B/16@224→14×14; PaliGemma's SigLIP tower patches differently. A hardcoded 27×27 reshape crashes or misaligns | MEDIUM | Grid (patch count, edge discard, aspect un-squash) must be derived per adapter from its processor/patch config — not hardcoded. The most error-prone reuse point |
| Reproducible load of all 4 models from the GCS model mirror (never HF at runtime) | v1.0 invariant (Pitfall G): weights only from `gs://mapclass-training-northeast1`. CLIP/PaliGemma/ViT weights must be mirrored too | LOW–MEDIUM | Extends the existing `_download_model_mirror` cache-first pattern; mostly a one-time `gsutil`/SDK mirror + per-model `from_pretrained(local_dir)`. PaliGemma is multi-GB (size/VRAM watch) |
| Notebook runs top-to-bottom headless, exit 0, even with N models failing coverage | The v1.1 milestone is *done* on exactly this behavior; matches v1.0's nbconvert verification discipline | LOW | nbconvert `--execute`; assert exit 0 and ≥1 rendered figure regardless of how many models produced heatmaps |

### Differentiators (Make The Comparison Genuinely Informative)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Explicit per-panel target-semantics label ("image–text similarity" vs "P(token 'yes')" vs "class logit 'map'") | Turns an apples-to-oranges grid into an *honest* comparison; the reader knows *what* each heatmap explains. The intellectual core of the deliverable | LOW | A panel subtitle string from each adapter's `declare_semantics`. Tiny effort, large honesty payoff |
| Visible "no heatmap (op-coverage gap)" tile that names the failing op/node | The negative result *is the product* (PROJECT.md). "dynamicLRP could not traverse — failed at `SplitWithSizesBackward0`" is far more informative than a blank panel — without being a structured report | LOW | Render the caught exception's salient node name *inside* the failed tile as caption text. NOT a separate table (anti-feature) — coverage stays implicit-in-the-tile per the user's framing |
| Deliberate, stated relevance-scale choice (per-panel self-scaled) | Cross-model relevance magnitudes are not comparable in absolute terms (different heads/scales). A shared scale visually flattens weaker models; per-panel symmetric `TwoSlopeNorm` is the honest default. Stating which is used prevents a silent misread | LOW–MEDIUM | Recommend **per-panel self-scaled** D-09 norm (each panel `M=max(abs(grid))`) + a one-line figure note: "each panel scaled independently; magnitudes not cross-comparable" |
| CLIP-vs-SigLIP-2 framed as the designed informative contrast | Same contrastive `logits_per_image` API; differ only in pooling head (CLS vs MAP-pool `split_with_sizes`). If CLIP traverses where SigLIP-2 fails, the comparison *localizes the dynamicLRP gap to the pooling head* — a real finding | (design, near-zero code) | Choose the model set so this contrast is legible; ViT as the "method works here at all" reference (dynamicLRP's own example architecture) |
| Same locked map + same query string across panels, shown once | A controlled comparison: only the model varies. Reuses v1.0's locked slice (`manifest[-1]`, `RUMSEY~8~1~344476~90112460`, query `"a river"`) for continuity | LOW | Reuse v1.0 locked-slice + locked-query constants. ViT's class-logit and PaliGemma's token target are the unavoidable exceptions to "same query" — label them as such, don't hide them |

### Anti-Features (Explicitly Out Of Scope — Do Not Build)

| Feature | Why Requested | Why Problematic Here | Alternative |
|---------|---------------|----------------------|-------------|
| Structured op-coverage report table (per-model covered/uncovered op matrix) | Feels like the rigorous way to present "which models dynamicLRP supports" | User explicitly chose "overlays only — NO structured op-coverage report; coverage implicit in whether a heatmap renders". A table is a different deliverable and a documentation rabbit hole | A captioned "no heatmap (op-coverage gap), failed at <node>" tile — coverage visible *in the grid itself* |
| Fixing per-model coverage gaps (custom dynamicLRP Promise, pre-pool/`use_attn_lrp`, vendored LXT, captum IG fallback) | "A model with no heatmap looks like a bug — make it work" | PROJECT.md Out-of-Scope **and** the v1.0 01-03 user decision *explicitly declined the entire Fallback Ladder*. A non-traversable model is a **recorded comparison result**; engineering around it destroys the experiment's point and deviates from the vendored SHA (T-01-SC3). The `attribution.py` docstring forbids re-attempting this without a new user decision | Render the gap honestly as a tile. The gap *is data* |
| Maps × queries sweep / contact-sheet browse | Was v1.0 Phase 2; "more maps = more coverage" intuition | PROJECT.md Key Decision: comparison axis is *model*, not *map count*; one locked slice suffices. Dropped in v1.1 by explicit decision | Single locked slice; the axis is the 4 models |
| Quantitative attribution metrics (faithfulness, occlusion deltas, IoU vs labels) | "Visual eyeballing isn't rigorous" | PROJECT.md Out-of-Scope: v1 judgment purely visual; metrics add scope before the loop is trusted. v1.0 D-04 explicitly prohibited quantitative pass/fail gates | Visual side-by-side eyeball, exactly as v1.0 |
| LRP method/parameter tuning as an explore axis (gamma, relevance_filter, attn-LRP, dtype sweeps) | "Maybe a knob makes more models work" | PROJECT.md Out-of-Scope for v1.1: LRP config held fixed; vary *model + query* only. Per-model tuning confounds the cross-model comparison | Fixed v1.0 knobs (`use_gamma=False`, `no_recompile=True`, `relevance_filter=0.5`) for every adapter |
| The three sanity controls (query-swap, model-randomization, occlusion) per model | They were the v1.0 correctness gate; instinct to carry forward | v1.0's gate validated *one model's faithfulness*; v1.1's deliverable is the *cross-model render*, and the gate was consciously waived once SigLIP-2 produced no relevance. 4 models × 3 controls = a sweep-shaped scope explosion the user dropped | One overlay per model. (One query-swap on the single best-traversing model is a *stretch* v1.x item, not table stakes) |
| Interactive widgets / model or query pickers (ipywidgets) | "Let the researcher explore live" | Interactivity is the dropped sweep/browse in disguise; the deliverable is a static side-by-side render. Adds JupyterLab-state fragility against the headless exit-0 requirement | Static figure; change the locked query by editing one constant and re-running |
| Other Rumsey dataset categories / non-historical maps | "More data variety" | PROJECT.md Out-of-Scope: Rumsey historical only; map is locked | The single locked slice |

## Feature Dependencies

```
[Model-agnostic adapter contract]
    └──requires──> [Per-architecture target-selection]   (the feature-defining question)
    └──requires──> [Per-model GCS weight mirror + reproducible load]
    └──requires──> [Per-model patch-grid reconstruction]  (grid size per processor)

[Side-by-side rendering surface]
    └──requires──> [Model-agnostic adapter contract]   (needs N results to lay out)
    └──requires──> [Graceful per-model degradation]    (a failed model still needs a tile)
    └──requires──> [Consistent D-09 colormap]          (reused from v1.0 overlay.py)

[Graceful per-model degradation]
    └──requires──> [Model-agnostic adapter contract]   (the per-model loop it wraps)

[Per-panel target-semantics label] ──enhances──> [Side-by-side rendering surface]
[Failing-op-named tile]            ──enhances──> [Graceful per-model degradation]

[Per-architecture target-selection] ──conflicts──> ["same query across all models"]
    (ViT has no text query; PaliGemma's query is a token, not a similarity —
     the comparison must LABEL this divergence, not pretend it away)
```

### Dependency Notes

- **Adapter contract requires target-selection:** the contract's whole purpose
  is to abstract the per-architecture target. Resolve the
  contrastive/generative/classification target semantics *before* writing the
  loop, or the harness silently compares incommensurable quantities. The one
  item warranting a deeper roadmap research/decision step.
- **Rendering requires graceful degradation:** the layout must reserve a tile
  for every model up front and fill it with either an overlay or a "no heatmap"
  caption — degradation is a *render state*, not an error path bolted on after.
- **Per-model patch-grid is a hidden dependency:** v1.0's 27×27 is SigLIP-2-only.
  Each adapter must derive its grid (patch count, edge discard, aspect un-squash)
  from its own processor/patch size, or panels crash/misalign. Easy to overlook
  because v1.0 "already has overlay.py".
- **Target-selection conflicts with "same query":** unavoidable. The honest
  resolution is *labeling* the divergence per panel, not forcing a fake uniform
  query. This conflict is why the per-panel semantics label is a differentiator,
  not a nicety.

## MVP Definition

### Launch With (v1.1 — the milestone-done set)

- [ ] Model-agnostic adapter contract (`model_id → load/forward/select_target/declare_semantics`) — without it "across 4 models" is impossible
- [ ] Per-architecture target selection resolved & documented (CLIP=`logits_per_image`; PaliGemma=answer-token logit; ViT=class logit; SigLIP-2=v1.0 path) — the feature-defining question; decide it, do not defer it
- [ ] CLIP, PaliGemma, plain ViT weights mirrored to GCS + reproducible per-model load (SigLIP-2 reuses v1.0 path)
- [ ] Per-model patch-grid reconstruction derived from each model's processor (generalize v1.0 overlay; do not hardcode 27×27)
- [ ] Graceful per-model degradation: caught failure → labeled "no heatmap (op-coverage gap)" tile → exit 0 (generalize v1.0's caught-finding pattern over the 4-model loop) — **the milestone done-criterion**
- [ ] One notebook: locked slice + query → 4 panels side-by-side + source map shown once, D-09 colormap reused, each panel labeled with model + target semantics + render status, runs headless exit 0

### Add After Validation (v1.x — only if the comparison proves valuable)

- [ ] A second locked query string as a second side-by-side row (query as the second axis, still no sweep) — trigger: the single comparison is informative and the researcher wants the query axis
- [ ] One query-swap contrast on the single best-traversing model as a faithfulness sanity check — trigger: a model produces a heatmap and its faithfulness is in doubt

### Future Consideration (v2+ — deferred, currently Out-of-Scope)

- [ ] Anything from the Anti-Features table — deferred by explicit PROJECT.md decisions; revisit only on a new milestone with a fresh user decision (especially: the Fallback Ladder was explicitly declined and must not be re-attempted without one)

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| Model-agnostic adapter contract | HIGH | MEDIUM | P1 |
| Per-architecture target selection (decided + documented) | HIGH | MEDIUM (mostly a design decision) | P1 |
| Graceful per-model degradation → labeled tile, exit 0 | HIGH (the done-criterion) | LOW | P1 |
| Side-by-side 4-panel render, source once, D-09 reused | HIGH | LOW–MEDIUM | P1 |
| Per-model patch-grid reconstruction | HIGH (panels break without it) | MEDIUM | P1 |
| CLIP/PaliGemma/ViT GCS mirror + load | HIGH | LOW–MEDIUM | P1 |
| Per-panel target-semantics label | HIGH (comparison honesty) | LOW | P1 (cheap, do it) |
| Failing-op named in the "no heatmap" tile | MEDIUM–HIGH | LOW | P1 |
| Deliberate + stated relevance-scale choice (per-panel self-scaled) | MEDIUM | LOW | P2 |
| Second query row / one query-swap sanity | LOW (post-validation) | LOW | P3 |
| Op-coverage report table / metrics / sweep / fixing gaps / tuning | NEGATIVE (scope creep) | — | Anti-feature (do not build) |

## Comparator Feature Analysis (attribution toolkits, for context)

These are reference toolkits, not competitors — they bound what is conventional
vs. custom for a cross-model attribution harness.

| Feature | Captum (LayerLRP/IG) | Zennit / LXT (AttnLRP) | Our v1.1 harness |
|---------|----------------------|------------------------|------------------|
| Cross-model uniformity | Per-model hooks, manual | Per-architecture canonizers / model-specific files | One adapter contract; dynamicLRP is op-level so variability is *only* target + grid |
| Negative-result handling | N/A (you wire it or it errors) | N/A | **First-class: a non-traversable model is a rendered datum, exit 0** — the distinctive thing here |
| Comparison surface | DIY | DIY | Built-in 4-panel side-by-side, one locked stimulus |
| Quantitative metrics | Yes (faithfulness suites) | Some | **Deliberately none** (visual only — anti-feature here) |

Takeaway: the harness is not competing on attribution sophistication; its
distinctive feature is treating *whether dynamicLRP can attribute a given
architecture at all* as the comparison's primary observable, rendered honestly
side-by-side. Visualization/orchestration is always the consumer's own piping —
no toolkit ships a cross-model honest-negative comparison surface turnkey.

## Sources

- `.planning/PROJECT.md` (v1.1 scope, Active/Out-of-Scope, Key Decisions, 4-model set, locked slice) — HIGH
- `.planning/archive/v1.0-milestone/.../01-03-PLAN.md` & `01-03-SUMMARY.md` (v1.0 single-slice + D-09 + caught-finding pattern to generalize; Fallback Ladder declined; SigLIP-2 `SplitWithSizesBackward0` finding) — HIGH (read directly)
- `src/mapclass/attribution.py` (v1.0 `attribute()`/`coverage_probe`; Pattern 2 tensor-identity invariant; the empirical SigLIP-2 op-coverage finding + the explicit "do not re-attempt the Fallback Ladder" user resolution) — HIGH (read directly)
- `src/mapclass/model_loader.py` (GCS-mirror reproducible cache-first load pattern to extend per model) — HIGH (read directly)
- `third_party/dynamicLRP/src/lrp_engine/{promises,model_specific}/` (engine is op-level/model-agnostic: generic op promises + a single `mosaicbert` shim; no ViT/CLIP/SigLIP/PaliGemma-specific code → coverage is per-target-op) — HIGH (inspected directly)
- Model-architecture knowledge: SigLIP-2 MAP-pool/`split_with_sizes`; CLIP CLS-token contrastive pooling; PaliGemma autoregressive prefix-LM (token-logit target, no `logits_per_image`); plain ViT classification-only (class-logit target, no text) — HIGH (corroborated by the v1.0 empirical finding + the transformers API surface documented in CLAUDE.md)

---
*Feature research for: cross-model dynamic-LRP attribution comparison harness (MapClass v1.1)*
*Researched: 2026-05-19*
