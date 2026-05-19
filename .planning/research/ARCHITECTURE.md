# Architecture Research

**Domain:** Multi-model dynamic-LRP attribution comparison harness (PyTorch VLMs + vendored dynamicLRP, JupyterLab on a remote GCP VM) — v1.1 milestone
**Researched:** 2026-05-19
**Confidence:** HIGH — grounded in the actual v1.0 source (`attribution.py`, `overlay.py`, `data_loader.py`, `model_loader.py`, `config.py`, `manifest.py`, `mirror_model.py`), the vendored `dynamicLRP/src/lrp_engine/lrp.py` engine contract, the 01-03 PLAN/SUMMARY findings, and the `_build_01_single_slice.py` notebook generator — all read directly 2026-05-19.

> Supersedes the v1.0 single-model architecture in this file's prior revision (2026-05-18). v1.0's pipeline shape is preserved; v1.1 introduces exactly one new seam (a model-agnostic adapter) and parameterizes two invariant-bearing modules.

## Standard Architecture

v1.0 is a **linear single-model pipeline hard-wired to SigLIP-2 at four points**: `model_loader.py` (SigLIP-only `AutoModel` singleton), `data_loader.py` (SigLIP `padding="max_length", max_length=64` tokenization), `attribution.py` (the literal `output.logits_per_image[0,0]` target), and `overlay.py` (module-level `PATCH_SIZE=14, IMG_DIM=384, GRID=27`). v1.1 must add **one model-agnostic adapter seam** that absorbs all four model-specific concerns, while the two load-bearing invariants — (a) the `requires_grad` pixel-tensor object identity from loader → forward → `params_to_interpret`, and (b) the D-09 signed / no-min-max / zero-centered `TwoSlopeNorm` overlay — survive unchanged, and `manifest`, `ingest_images`, and the vendored engine are reused verbatim.

### System Overview (v1.1 target)

```
┌──────────────────────────────────────────────────────────────────────┐
│                  notebooks/02_multimodel.ipynb                         │
│  locked slice (manifest[-1]) + locked query → loop 4 adapters →        │
│  per-model try/except → 2x2 grid of overlays (or "no heatmap" tile)    │
└───────────────────────────────┬──────────────────────────────────────┘
                                 │  for adapter in REGISTRY:
                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│        src/mapclass/adapters/   (THE NEW MODEL-AGNOSTIC SEAM)          │
│  ┌──────────┐ ┌────────┐ ┌────────────┐ ┌──────────┐                  │
│  │ siglip2  │ │  vit   │ │   clip     │ │paligemma │   (ordered list)  │
│  │ adapter  │ │adapter │ │  adapter   │ │ adapter  │                   │
│  └────┬─────┘ └───┬────┘ └─────┬──────┘ └────┬─────┘                   │
│       │  each implements ModelAdapter:                                 │
│       │  load() · build_inputs() · forward()                           │
│       │  attribution_target() · patch_geometry()                       │
└───────┼───────────┼────────────┼─────────────┼───────────────────────┘
        │           │            │             │
 ┌──────▼────┐ ┌────▼──────┐ ┌───▼─────────────▼────┐
 │model_loader│ │data_loader│ │ attribution.attribute │   (REUSED core,
 │(per-model_ │ │(image IO  │ │ forward + LRPEngine in │    modified API)
 │ id singleton)│ cache;    │ │ ONE scope; identity inv)│
 └────────────┘ │ generic)  │ └───────────┬───────────┘
                └───────────┘  ┌───────────▼───────────┐
                               │ overlay.to_patch_grid  │  (PARAMETERIZED
                               │ / composite — geometry │   by adapter
                               │ injected, not 27 const)│   PatchGeometry)
                               └───────────────────────┘
        │ all 4 forwards traverse the SAME vendored engine
 ┌──────▼────────────────────────────────────────────────────────┐
 │ third_party/dynamicLRP/src/lrp_engine  (UNCHANGED, SHA-pinned)  │
 └─────────────────────────────────────────────────────────────────┘
 ┌─────────────────────────────────────────────────────────────────┐
 │ GCS gs://mapclass-training-northeast1  models/<id>/ + data/      │
 └─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | New / Modified / Reused |
|-----------|----------------|-------------------------|
| `adapters/base.py` (`ModelAdapter` protocol + `PatchGeometry`) | The seam contract: `model_id`, `load()`, `build_inputs(pil,query,...)`, `forward(model,inputs)`, `attribution_target(output)`, `patch_geometry()` | **NEW** |
| `adapters/registry.py` | Ordered list of the 4 adapter instances the notebook iterates (SigLIP-2 → ViT → CLIP → PaliGemma) | **NEW** |
| `adapters/siglip2.py` | SigLIP-2 specifics extracted from v1.0: `AutoModel` eager, `padding=max_length/64`, `logits_per_image[0,0]`, `PatchGeometry(14,384)` → 27×27/6-px. Carries the known op-coverage gap as a recorded result | **NEW** (lifts v1.0 logic) |
| `adapters/vit.py` | Plain `ViTForImageClassification`, **no text path** — query-conditioned via the query-mapped class logit; patch16/224 geometry | **NEW** |
| `adapters/clip.py` | `CLIPModel`, CLIP processor, `logits_per_image[0,0]`, CLIP patch geometry | **NEW** |
| `adapters/paligemma.py` | `PaliGemmaForConditionalGeneration`, query-conditioned target = answer-token logit (generative, non-contrastive); SigLIP-vision patch geometry | **NEW** |
| `model_loader.py` | Cache-first GCS-mirror download + `from_pretrained` singleton, **keyed by model_id** (dict of singletons, not one global) | **MODIFIED** |
| `mirror_model.py` | Mirror N checkpoints to `models/<id>/` (loop registry repo ids) instead of one hard-coded id | **MODIFIED** |
| `data_loader.py` | Disk cache → GCS-miss → PIL decode (model-agnostic, kept); the SigLIP processor/tokenize call is **delegated to the adapter** | **MODIFIED** |
| `attribution.py` | Forward + `LRPEngine` in one scope; target via `adapter.attribution_target(output)` not literal `logits_per_image[0,0]` | **MODIFIED** |
| `overlay.py` | `to_patch_grid` / `composite` / `draw_patch_grid` take `PatchGeometry` instead of module-level `14/384/27`. D-09 rendering untouched | **MODIFIED** |
| `config.py` | Single `MODEL_REPO_ID`/`MODEL_GCS_DIR` → a per-model registry map; `VENDOR_SHA`/cache paths unchanged | **MODIFIED** |
| `manifest.py`, `ingest_images.py` | Locked-slice resolution / image GCS — model-agnostic | **REUSED unchanged** |
| `third_party/dynamicLRP` | LRP engine traversed by all 4 forwards; coverage gaps recorded, never patched (Fallback Ladder declined, `VENDOR_SHA` intact) | **REUSED unchanged (SHA-pinned)** |

## Recommended Project Structure

```
src/mapclass/
├── config.py                 # MODIFIED: MODELS = {id: ModelSpec(repo_id, gcs_dir)}
├── manifest.py               # REUSED (locked slice = entry_by_index(m, -1))
├── ingest_images.py          # REUSED (GCS bucket + image mirror)
├── mirror_model.py           # MODIFIED: loop registry repo ids → models/<id>/
├── model_loader.py           # MODIFIED: {model_id: (model, processor)} singletons
├── data_loader.py            # MODIFIED: image IO kept; tokenize → adapter
├── attribution.py            # MODIFIED: target via adapter.attribution_target
├── overlay.py                # MODIFIED: PatchGeometry injected (D-09 untouched)
└── adapters/                 # NEW: the model-agnostic seam
    ├── __init__.py
    ├── base.py               # ModelAdapter Protocol + PatchGeometry dataclass
    ├── registry.py           # ordered [siglip2, vit, clip, paligemma]
    ├── siglip2.py
    ├── vit.py
    ├── clip.py
    └── paligemma.py
notebooks/
├── 01_single_slice.ipynb     # REUSED (SigLIP-2 smoke test / negative finding)
├── _build_02_multimodel.py   # NEW: generator (mirrors _build_01 pattern)
└── 02_multimodel.ipynb       # NEW: 4-model side-by-side comparison
tests/
├── test_attribution_shape.py # MODIFIED: parametrize over a synthetic adapter
├── test_overlay_grid.py      # MODIFIED: parametrize geometry (27x27, 14x14, …)
└── test_adapters.py          # NEW: protocol conformance + PatchGeometry math
```

### Structure Rationale

- **`adapters/` as a package, one module per model.** Each model's quirks are noisy and divergent: PaliGemma is generative, ViT has no text path, CLIP/SigLIP are contrastive with different patch grids. Co-locating each model's full vertical slice in one file keeps the cross-cutting `attribution.py`/`overlay.py`/`data_loader.py` small and model-agnostic. This is the Strategy + registry pattern — the standard answer for "one pipeline, swappable backends."
- **Registry is an ordered list, not a name→adapter dict.** The notebook iterates a fixed build/comparison order (SigLIP-2 first as the known-negative oracle, then ViT/CLIP/PaliGemma); ordering is part of the comparison contract, so encode it as a list.
- **Geometry travels with the adapter, not as `overlay.py` constants.** v1.0's `GRID=27` is `floor(384/14)`. A plain ViT-base and CLIP-base are `224/16=14`; PaliGemma's SigLIP vision tower differs again. Geometry must be a per-model value object the adapter computes and the overlay consumes.
- **`data_loader.py` keeps image IO, loses tokenization.** The disk-cache → GCS-miss → PIL decode logic (DATA-04) is fully model-agnostic and battle-tested; only the `processor(text=…, images=…, padding=…)` call is SigLIP-specific. Split there: `data_loader` yields the PIL; the adapter turns `(pil, query)` into model-specific tensors.

## Architectural Patterns

### Pattern 1: ModelAdapter protocol (the seam)

**What:** A `typing.Protocol` (structural — no inheritance required) plus a frozen `PatchGeometry` value object. One class per model implements it.
**When to use:** Always — this is *the* v1.1 architectural change; every model-specific decision lives behind it.
**Trade-offs:** One indirection layer; pays for itself the instant there is >1 model. The protocol must be wide enough for a generative VLM (PaliGemma) and a text-less classifier (ViT) without leaking model types into `attribution.py`.

```python
# src/mapclass/adapters/base.py
from dataclasses import dataclass
from typing import Protocol

@dataclass(frozen=True)
class PatchGeometry:
    patch_size: int
    img_dim: int
    @property
    def grid(self) -> int:          # floor(img_dim / patch_size)   (SigLIP: 27)
        return self.img_dim // self.patch_size
    @property
    def valid_dim(self) -> int:     # grid * patch_size              (SigLIP: 378)
        return self.grid * self.patch_size
    @property
    def edge_discard(self) -> int:  # img_dim - valid_dim (padding="valid"; SigLIP: 6)
        return self.img_dim - self.valid_dim

class ModelAdapter(Protocol):
    model_id: str                    # GCS models/<model_id>/ key + tile label

    def load(self) -> tuple[object, object]:
        """(model, processor) — cache-first from GCS, .to(DEVICE).eval()."""
    def build_inputs(self, pil, query, processor, device) -> dict:
        """PIL + query → {'pixel_values': <requires_grad tensor>, ...}.
        MUST call .requires_grad_() on pixel_values and NOT
        clone/detach/re-.to() it afterward (identity invariant)."""
    def forward(self, model, inputs: dict):
        """model(**inputs) → raw output object."""
    def attribution_target(self, output):
        """Query-conditioned scalar/2-D tensor for LRPEngine.run.
        SigLIP-2/CLIP: output.logits_per_image[0, 0].
        PaliGemma: answer-token logit at the query position.
        ViT: logit of the query-mapped class."""
    def patch_geometry(self) -> PatchGeometry:
        """Vision-tower patch grid for overlay reconstruction."""
```

### Pattern 2: Tensor-identity invariant preserved across heterogeneous forwards

**What:** The `requires_grad_()` pixel tensor object built in `build_inputs` must be the *same Python object* passed to `model(**inputs)` and to `engine.params_to_interpret`. Verified in `lrp.py`: `run()` calls `make_graph_iter(root_nodes, params_to_interpret, …)` which resolves `params_to_interpret` by walking the live `grad_fn` chain and matching by **tensor object identity**. Any `.clone()`/`.detach()`/re-`.to()`/scope-close between build and `engine.run()` destroys the graph — exactly what v1.0's `attribution.py` and `data_loader.py` docstrings warn about ("no clone/detach/re-.to() after this line").
**When to use:** Every adapter, non-negotiable. This is why `build_inputs` returns the dict containing the *live* tensor and `attribution.attribute()` (not the adapter) owns the `engine.run` call in one scope.
**Trade-offs:** Constrains the adapter API — `build_inputs` may not post-process `pixel_values` after `requires_grad_()` (no normalize-after, no device move after). The discipline that worked for SigLIP-2 in v1.0 is generalized verbatim, not redesigned.

```python
# src/mapclass/attribution.py  (MODIFIED — adapter-driven; identity preserved)
def attribute(adapter, model, inputs):           # inputs from adapter.build_inputs
    img_tensor = inputs["pixel_values"]          # SAME object, requires_grad already set
    output = adapter.forward(model, inputs)      # forward in THIS scope
    target = adapter.attribution_target(output)  # per-model query-conditioned target
    # 0d→2d→1d target-form fallback (v1.0 Empirical Risk 1) is retained here,
    # generalized: SigLIP's IndexError-on-0d may not apply to ViT/CLIP logits.
    engine = LRPEngine(use_gamma=False, no_recompile=True, relevance_filter=0.5)
    engine.params_to_interpret = [img_tensor]    # SAME object as the forward input
    _ckpt, param_vals = engine.run(target)
    return param_vals[0]                          # input-shaped relevance
```

The forbidden-token grep guard in v1.0's `attribution.py` (asserting `get_image_features` / image-embedding-norm never appear) **moves into each adapter's `attribution_target`** — the cross-cutting module no longer names a target, so the Pitfall-B guard belongs where the target is now chosen.

### Pattern 3: Geometry-parameterized overlay (kill the hard-coded 27)

**What:** `to_patch_grid` / `composite` / `draw_patch_grid` accept a `PatchGeometry` and derive `reshape(grid, patch, grid, patch).mean((1,3))` from it. The D-09 signed-sum-no-`.abs()` / no-min-max / zero-centered `TwoSlopeNorm` `bwr` rendering is **unchanged** — only the grid dimensions are parameterized.
**When to use:** All overlay calls in v1.1. SigLIP-2 stays `PatchGeometry(14,384)` → 27×27/6-px; ViT/CLIP supply their own.
**Trade-offs:** Signature churn (callers pass geometry). The alternative — per-model overlay copies — duplicates the load-bearing D-09 signed-relevance logic and will drift; the v1.0 16 unit tests already lock the SigLIP numbers and just need parametrizing.

```python
# src/mapclass/overlay.py  (MODIFIED — geometry injected, D-09 untouched)
def to_patch_grid(relevance, geom):                    # geom: PatchGeometry
    r = relevance.detach()[0]                          # (3, H, W)
    r_signed = r.sum(0); r_mag = r.abs().sum(0)        # D-09: signed (no abs) + magnitude
    v, g, p = geom.valid_dim, geom.grid, geom.patch_size
    grid_signed = r_signed[:v, :v].reshape(g, p, g, p).mean((1, 3))
    grid_mag    = r_mag[:v, :v].reshape(g, p, g, p).mean((1, 3))
    return grid_signed.cpu().numpy(), grid_mag.cpu().numpy()   # NO min-max (D-09)
```

### Pattern 4: Per-model fault isolation in the notebook ("no heatmap" tile is a result)

**What:** The notebook loop wraps each model's `load → build_inputs → attribute → to_patch_grid → composite` in a `try/except Exception`. On failure it renders a labeled "NO HEATMAP — <op-coverage finding>" tile in that model's grid cell instead of aborting. This generalizes v1.0's "caught-finding cell" pattern (`01_single_slice.ipynb` Cell 5: catch the engine `RuntimeError`, render a FINDING cell, exit 0) to a 2×2 layout.
**When to use:** The comparison notebook only. Per `PROJECT.md` Key Decisions, a model dynamicLRP cannot traverse is *the data*, not a bug — the harness must keep running and record it visually.
**Trade-offs:** A broad `except` can mask unrelated bugs; mitigate by capturing `repr(exc)` + `traceback` into the tile caption (as v1.0 does) and running the per-model coverage probe (`LRPEngine.get_model_operations`) first as a recorded artifact.

```python
# notebooks/02_multimodel.ipynb  (loop skeleton)
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
for ax, adapter in zip(axes.flat, REGISTRY):           # SigLIP-2, ViT, CLIP, PaliGemma
    try:
        model, proc = adapter.load()
        inputs = adapter.build_inputs(pil, QUERY, proc, DEVICE)  # requires_grad tensor
        relevance = attribute(adapter, model, inputs)            # forward+LRP one scope
        gs, gm = to_patch_grid(relevance, adapter.patch_geometry())
        composite(ax, pil, gs, gm, title=adapter.model_id)
    except Exception as exc:
        ax.imshow(pil); ax.set_axis_off()
        ax.set_title(f"{adapter.model_id}: NO HEATMAP")
        ax.text(0.5, 0.5, f"op-coverage finding:\n{type(exc).__name__}",
                transform=ax.transAxes, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8))
    finally:
        del model                                       # release before next model
        torch.cuda.empty_cache()                        # Pitfall E — per-model free
```

## Data Flow

### Comparison Flow (v1.1)

```
manifest[-1] (locked slice id)              QUERY (locked text, e.g. "a river")
        │                                            │
        └──────────────┬─────────────────────────────┘
                        ▼
        data_loader cached_image_path(id) → PIL   (GCS-miss disk cache; model-agnostic)
                        │
        ┌───────────────┴─── for adapter in REGISTRY ───────────────┐
        ▼ (per model, sequential)                                    │
  adapter.load()  ── cache-first GCS models/<id>/ → (model, processor)
        ▼
  adapter.build_inputs(pil, QUERY, processor, DEVICE)
        → {pixel_values: TENSOR.requires_grad_(), ...}   ← identity origin
        ▼
  attribution.attribute(adapter, model, inputs)   [ONE scope]
     ├ adapter.forward(model, inputs)          → output
     ├ adapter.attribution_target(output)      → query-conditioned scalar
     ├ LRPEngine.params_to_interpret=[pixel_values]  ← SAME object
     └ engine.run(target)                      → relevance (input-shaped)
        ▼
  overlay.to_patch_grid(relevance, adapter.patch_geometry())  → signed+mag grids
        ▼
  overlay.composite(ax, pil, …)                → one tile  (or except → "no heatmap")
        ▼
  del model; torch.cuda.empty_cache()          → free before next adapter
        └───────────────────────────────────────────────────────────┘
                        ▼
        2×2 figure: SigLIP-2 | ViT | CLIP | PaliGemma   (eyeball comparison)
```

### State / lifecycle

- **Model singletons keyed by `model_id`.** `model_loader` becomes `{model_id: (model, processor)}` (drop-in extension of the existing single global + lock). But four ~400M-class VLMs do **not** co-reside in L4 VRAM — SigLIP-2's *forward-only* peak alone was measured at 4.326 GB (01-03), and the dynamicLRP relevance pass retains forward activations on top (Pitfall E). The notebook loop must be **strictly sequential**: load → attribute → `del model` → `torch.cuda.empty_cache()` before the next adapter. Sequential load/free is a *required lifecycle*, not an optimization. (Implication: the per-model_id singleton cache mostly serves re-runs of the same model in one kernel; the loop deliberately evicts.)
- **Coverage probe per model.** Run `LRPEngine.get_model_operations(target)` per adapter and print the op count as a recorded comparison artifact (v1.0 Cell-3 pattern, now inside the loop).

## Scaling Considerations

Single-user exploratory research piping on one GPU VM; "scale" = number of comparison models and VRAM, not users.

| Scale | Architecture Adjustments |
|-------|--------------------------|
| 4 models (v1.1 scope) | Sequential load → attribute → free per model; `try/except` per tile; one locked slice + one query. Fits L4 only if models are released between iterations. |
| +models / +queries (future) | Registry already supports N adapters; wrap a query loop outside the model loop → N×M grid. No architectural change, pure iteration. |
| Larger models | Bounded by `PROJECT.md` constraint (so400m-class ceiling for dynamic-LRP fidelity/overhead) and L4 VRAM — deliberately out of scope. |

### Scaling Priorities

1. **First bottleneck — VRAM with multiple resident models.** Fix: strictly sequential per-model load/free in the loop (`del model; torch.cuda.empty_cache()`); never hold two VLMs. The relevance pass also retains activations (Pitfall E) — free aggressively in a `finally`.
2. **Second bottleneck — per-model GCS mirror cost on a cold VM.** Fix: `model_loader`'s existing size-checked cache-first download already handles this per-model; `mirror_model.py` just loops the registry repo ids once (idempotent skip-if-same-size, same pattern as the SigLIP-2 mirror).

## Anti-Patterns

### Anti-Pattern 1: Letting the adapter own the `LRPEngine.run()` call

**What people do:** Put `forward + engine.run` inside each adapter for "encapsulation."
**Why it's wrong:** The tensor-identity invariant requires the forward and `engine.run` in **one scope** with the *same* `pixel_values` object as `params_to_interpret`. Spreading it across the adapter boundary risks a scope close / re-`.to()` / clone that silently breaks graph traversal — the exact failure v1.0's `attribution.py` docstring (Pattern 2) warns about.
**Do this instead:** Adapter supplies `forward()` and `attribution_target()` only; `attribution.attribute()` owns the engine and the one-scope discipline.

### Anti-Pattern 2: Re-hardcoding geometry in copied overlay code

**What people do:** Copy `overlay.py` to `overlay_clip.py` with `GRID=14`.
**Why it's wrong:** Duplicates the load-bearing D-09 signed-relevance / zero-centered `TwoSlopeNorm` logic; a fix to the signed-rendering invariant would have to be applied N times and will drift.
**Do this instead:** One `overlay.py`, geometry injected via `PatchGeometry`. The D-09 reduction is identical across models; only `grid/patch/valid_dim` differ.

### Anti-Pattern 3: A query-independent attribution target on the text-less ViT

**What people do:** For the plain ViT (no text encoder), attribute the pooled image embedding or its norm.
**Why it's wrong:** That yields image-saliency, not a *query-conditioned* map — it breaks cross-model comparability (every other tile answers "where is the query") and is exactly the Pitfall B that v1.0's `attribution.py` forbids by grep.
**Do this instead:** Map the query string to the nearest ViT class logit (ImageNet class, or a fixed agreed class for the locked query) and attribute *that* logit. Record the query→class mapping as a documented adapter decision so the ViT tile stays query-conditioned and comparable.

### Anti-Pattern 4: Patching vendored dynamicLRP to make a model "work"

**What people do:** Add a custom Promise so PaliGemma/CLIP traverses cleanly.
**Why it's wrong:** `PROJECT.md` Key Decisions and 01-03-SUMMARY record that the entire Fallback Ladder was **explicitly declined** — a model the engine cannot traverse is *the recorded comparison result*. Patching defeats the deliverable and breaks `VENDOR_SHA` provenance (threat T-01-SC3).
**Do this instead:** Catch the engine error per model, render the "no heatmap" tile with the op-coverage finding, move on. The negative result is data.

## Integration Points

### External Services

| Service | Integration Pattern | Notes |
|---------|---------------------|-------|
| GCS `models/<id>/` | Per-model cache-first download (extend `model_loader._download_model_mirror` to a model_id-keyed prefix) | One subdir per model id; `mirror_model.py` loops registry repo ids; idempotent size-check skip already implemented |
| GCS `data/` (Rumsey images) | Unchanged — `data_loader._cached_image_path` | Model-agnostic; the locked slice image is loaded once and reused across all 4 adapters |
| HuggingFace Hub | Only at mirror time (`mirror_model.snapshot_download` per repo id); never at runtime | Reproducibility (Pitfall G) — runtime loads strictly from GCS |
| dynamicLRP (vendored, SHA-pinned) | In-process autograd traversal via `LRPEngine`; unchanged | Coverage gaps per model are recorded results, never engine patches (Fallback Ladder declined) |

### Internal Boundaries

| Boundary | Communication | Considerations |
|----------|---------------|----------------|
| notebook ↔ `adapters/registry` | Import ordered list, iterate | Order is the comparison contract: SigLIP-2 → ViT → CLIP → PaliGemma |
| adapter ↔ `data_loader` | `data_loader` returns PIL; adapter does `build_inputs` (processor/tokenize) | Splits image IO (generic, reused) from tokenization (model-specific, moved into adapter) |
| `attribution.attribute` ↔ adapter | Calls `adapter.forward` + `adapter.attribution_target`; owns `engine.run` | One-scope identity invariant lives here, not in the adapter |
| `overlay` ↔ adapter | `adapter.patch_geometry()` → `PatchGeometry` passed to `to_patch_grid`/`composite` | D-09 signed/zero-centered rendering invariant unchanged; only dimensions parameterized |
| `attribution` ↔ vendored engine | `engine.params_to_interpret=[pixel_values]`; `engine.run(target)` | `params_to_interpret` resolved by tensor object identity in `make_graph_iter` — the invariant's root cause |

## Suggested Build Order

Dependency-driven; each step independently verifiable and respects both load-bearing invariants.

1. **Adapter contract first** — `adapters/base.py` (`ModelAdapter` Protocol + `PatchGeometry`) + `registry.py` skeleton + `test_adapters.py`. Unit-test that a trivial fake adapter satisfies the protocol and `PatchGeometry(14,384).grid==27`, `.edge_discard==6` (locks the SigLIP numbers the v1.0 tests assert). No model yet. *Verify: tests pass, zero behavior change to v1.0.*
2. **Parameterize `overlay.py` by `PatchGeometry`; rework `attribution.py` to take an adapter.** Keep SigLIP numbers as the default-equivalent; parametrize `test_overlay_grid.py` over geometry. *Verify: existing 16 unit tests still pass with geometry injected.*
3. **SigLIP-2 adapter (extract v1.0 logic verbatim).** Wrap the exact v1.0 `model_loader`/`data_loader`/`logits_per_image[0,0]` path behind the adapter. Re-run `01_single_slice` semantics through it — it must reproduce the *same recorded op-coverage finding* (no heatmap, caught `RuntimeError`). This proves the seam is behavior-preserving on the one model fully understood. *Verify: same finding as 01-03 (regression oracle).*
4. **ViT adapter (simplest new model; no text path).** Mirror a small ViT to GCS; implement the query→class-logit target. Plain transformer, no MAP-pool `split_with_sizes` → most likely to traverse cleanly → first *positive* heatmap, validating the parameterized overlay end-to-end with non-SigLIP geometry. *Verify: a rendered heatmap tile + coverage artifact.*
5. **CLIP adapter (contrastive like SigLIP-2 but different head/geometry).** `logits_per_image[0,0]` target, CLIP patch geometry. Tests whether dynamicLRP traverses CLIP's pooling — a real comparison datum either way. *Verify: heatmap or recorded "no heatmap" finding.*
6. **PaliGemma adapter (hardest: generative VLM, non-contrastive target).** Query-conditioned target = answer-token logit. Highest op-coverage risk (generative decoder); a recorded finding is acceptable per `PROJECT.md`. *Verify: heatmap or recorded finding, harness does not crash.*
7. **`mirror_model.py` + `config.py` registry, then `notebooks/02_multimodel.ipynb` last.** Generalize the mirror loop / config map incrementally alongside 4–6 (each model needs weights). Build the comparison notebook *last* via a `_build_02_multimodel.py` generator (mirror the v1.0 `_build_01` pattern: script is source of truth, ipynb is the committed artifact, headless nbconvert verifies 0 cell errors and ≥4 tiles). *Verify: notebook runs end-to-end exit 0, 2×2 grid renders, failed models show "no heatmap" tiles not tracebacks — milestone done.*

**Why this order:** The contract and the two invariant-bearing modules (`overlay`, `attribution`) must be generalized before any model can plug in (steps 1–2). SigLIP-2 first (step 3) because it is the *only* model whose end-to-end behavior is already known (the recorded negative finding) — it is the regression oracle proving the seam preserves behavior. ViT before CLIP/PaliGemma (steps 4→6) ascends difficulty: text-less plain transformer → contrastive-with-different-head → generative VLM, so the first *positive* heatmap (validating the parameterized overlay) arrives as early as possible. The notebook is last because it only composes already-verified parts and is the milestone's done-gate.

## Sources

- `src/mapclass/{attribution,overlay,data_loader,model_loader,config,manifest,mirror_model}.py` — read directly 2026-05-19 — HIGH (the authoritative current architecture; the four SigLIP hard-wire points and the two invariants are quoted from these docstrings)
- `third_party/dynamicLRP/src/lrp_engine/lrp.py` (`LRPEngine.__init__`, `run`, `get_model_operations`, `make_graph_iter(root_nodes, params_to_interpret, …)`) — read directly — HIGH (confirms `params_to_interpret` resolved by tensor identity — the invariant's mechanism; confirms `run()` accepts tensor or tuple[tensor])
- `.planning/PROJECT.md` (v1.1 goal, Key Decisions: Fallback Ladder declined, per-model gap = recorded result, model + query are the only knobs) — HIGH
- `.planning/archive/v1.0-milestone/.../01-03-SUMMARY.md` + `01-03-PLAN.md` — HIGH (SigLIP-2 `split_with_sizes` op-coverage finding; D-09 signed-overlay invariant; 4.326 GB forward VRAM; the "caught-finding cell" notebook pattern)
- `notebooks/_build_01_single_slice.py` — read directly — HIGH (the generator pattern the v1.1 notebook should mirror)

---
*Architecture research for: multi-model dynamic-LRP attribution comparison harness (v1.1)*
*Researched: 2026-05-19*
