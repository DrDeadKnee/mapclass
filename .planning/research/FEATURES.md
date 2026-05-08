# Feature Research

**Domain:** Multi-source-trained dense-prediction VL+OCR+heads model (model-handoff package, not an app)
**Researched:** 2026-05-08
**Confidence:** HIGH (boundaries are tightly specified by `PROJECT.md`; only feasibility/cost ratings carry MEDIUM confidence)

## Scope and framing

This document covers features **beyond** the v1 scope already locked in `PROJECT.md` (DATA-01..06, GEOREF-01..02, MODEL-01..03, TRAIN-01..02, EVAL-01..02, SHIP-01). Those locked items are out of scope here — they are not "new features" to evaluate, they are the v1 contract. This document answers: *given that locked set, what additional capabilities are table-stakes for a model-handoff package, what are differentiators worth their cost, and what are tempting features the codebase or external pressure might suggest but PROJECT.md explicitly excludes?*

Three audiences consume this:
1. **The roadmap** — uses table-stakes to derive REQUIREMENTS, differentiators to score phase add-ons, anti-features to gate scope creep.
2. **The hex-grid app integration** — needs the inference module's API surface defined (sketched below).
3. **Future Claude sessions** — needs the anti-feature list to refuse re-promoting deferred work.

## Feature Landscape

### Table Stakes — without these, the model handoff is broken

These are features the locked v1 list (in `PROJECT.md`) implies but does not name explicitly. Every one of these supports `SHIP-01` ("Inference Python module + trained checkpoint, packaged for the separate hex-grid app to consume"); none are in the "Out of Scope (v1)" list of `PROJECT.md` — verified.

| # | Feature | Why Expected | Complexity | Notes |
|---|---------|--------------|------------|-------|
| TS-1 | **Reproducible training entrypoint** (single `train.py` or `mapclass-train` console script + config file) | Roadmap and bootstrap loop (`TRAIN-01`/`TRAIN-02`) are useless if v0 → v1 cannot be re-run from a config | M | YAML or dataclass config; must take `--seed`, `--source-weights`, `--checkpoint-dir`. Reproducibility is a known gap (`CONCERNS.md` 8a — no seeds anywhere) |
| TS-2 | **Checkpoint format = safetensors** (not pickle .bin) | Hugging Face has standardised on safetensors; pickle .bin executes arbitrary code on load and the hex-grid app would inherit that risk | S | `safetensors.torch.save_file` / `load_file`. Free win — `transformers` already writes both. |
| TS-3 | **Inference Python module with stable single-call API** (the hex-grid app's only entry point) | `SHIP-01` says "packaged for the separate hex-grid app to consume" — that consumption needs a documented function signature, not "import whatever from train.py" | M | API sketch below. This is the contract; pin it in v1, never break it. |
| TS-4 | **Probability output, not argmax** | Stated explicitly in the prompt's downstream-consumer block; the hex-grid app aggregates per-pixel probabilities into hexes, so argmax discards information it needs | S | Already implied by EVAL-01 (NLL needs probabilities) — guarantee the inference path also returns them, not just the eval path |
| TS-5 | **Held-out evaluation script** that reports NLL on synthetic + historical splits and saves a JSON report | `EVAL-01` is named but the *runnable* artefact (`mapclass-eval` or `python -m mapclass.eval`) needs to exist for the v1 ship metric to be checkable | S | Must split deterministically by sample-id hash, not by glob order. Output: `eval_report.json` with per-source per-class NLL + global. |
| TS-6 | **Calibration / confidence sanity** — a one-shot reliability-diagram + ECE check appended to the eval script | Without this, "probability map" might mean "argmax with 1.0 wherever the argmax went". The hex-grid app *will* be misled by overconfident outputs. CNN segmentation models are documented to be miscalibrated by default (Mask-TS Net 2024). | S | Cheap: bin predictions by confidence, plot accuracy vs confidence, compute ECE. Don't need full temperature-scaling fitting in v1 — just the sanity diagnostic. |
| TS-7 | **Training-time logging** (loss curves + sample prediction images at fixed intervals) | The owner accepted "no tests / no CI" at v1 (`PROJECT.md` Out-of-Scope); this stands in for the missing observability layer | S | tensorboard or wandb (latter requires an account; tensorboard is in-tree). Sample preds should include each source so distribution shift is visible. |
| TS-8 | **Model + dataset versioning** (semver on the checkpoint + a manifest JSON) | The hex-grid app needs to know which checkpoint it's running ("model v1.2 trained on dataset commit abc123"); without this, a checkpoint mismatch silently changes the hex-grid app's outputs | S | Embed `{model_version, dataset_manifest_sha, training_seed, source_class_weights, taxonomy_hash}` inside the safetensors metadata header. `CONCERNS.md` 8b flags the absence of any versioning. |
| TS-9 | **CPU and 4–8 GB GPU inference paths** (both, tested) | `PROJECT.md` Constraints: "single CPU-only host **or** a 4–8 GB consumer GPU". "Or" not "and" in the constraint, but the inference module must support both modes via a `device='cpu'/'cuda'/'auto'` flag. | S | Just plumb `model.to(device)`; the fact that it works on 4–8 GB is owned by `MODEL-01`'s backbone choice, not by the inference module. |
| TS-10 | **Deterministic data splits** (train/val/test split is reproducible from a seed) | Without this, "held-out" cannot survive re-running the dataset build — `EVAL-01` numbers stop being comparable across runs | S | One canonical split file (e.g. `splits.json`) committed to git, generated from a seed + sample-id hash. |
| TS-11 | **Output shape + dtype contract documented** (e.g. `land_cover: float16, [9, H, W], softmax over axis 0`) | The hex-grid app needs to know whether to expect logits or probabilities, fp16 or fp32, channels-first or last, before integrating | S | One markdown section in the inference module's README. Sources for fp16 prob storage as a sane default: standard practice for chunked array stores ([Zarr v2 driver](https://google.github.io/tensorstore/driver/zarr/index.html)). |

**Total table-stakes additions: 11.** None duplicate items already in `PROJECT.md` Active. None are in `PROJECT.md` Out-of-Scope — verified item-by-item.

### Inference module API surface (TS-3 detail)

The roadmap will need to encode this contract; sketching it here so it can flow through to REQUIREMENTS.md. Adjust function/module names at implementation time, but the *shape* of the contract is what the downstream app integrates against.

```python
# mapclass/inference.py

from pathlib import Path
import numpy as np
import torch

def load_model(
    checkpoint: str | Path,
    device: str = "auto",          # "cpu" | "cuda" | "auto"
    dtype: str = "float16",        # "float16" | "float32"
) -> "GeoViLM":
    """
    Load a trained GeoViLM checkpoint (safetensors).

    Reads model metadata (version, dataset manifest sha, taxonomy hash) from
    the safetensors header and exposes them on the returned object.
    """

def predict(
    model: "GeoViLM",
    image: "np.ndarray | PIL.Image | str | Path",
    *,
    return_logits: bool = False,
    batch: bool = False,           # if True, image is a stack [N, H, W, 3]
) -> dict[str, np.ndarray]:
    """
    Returns:
        {
          "land_cover": float16 array, shape [9, H, W] (or [N, 9, H, W] if batch),
                         softmax probabilities over axis -3, sums to 1 per pixel,
          "topography": float16 array, shape [3, H, W] (or [N, 3, H, W] if batch),
                         softmax probabilities; pixels where land_cover argmax==water
                         carry the WATER_TOPO sentinel (255) policy decision —
                         the hex-grid app must agree on this.
          "model_version": str,
          "taxonomy": {"land_cover": [9 class names], "topography": [3 class names]},
        }

    Image can be RGB H×W×3 uint8 or any PIL-loadable thing. Internal preprocessing
    handles resize/pad to model's expected input. The output spatial shape (H, W)
    is the *input* H, W after de-padding/de-resize — the caller does not have to
    track preprocessing.
    """

def predict_batched(
    model: "GeoViLM",
    images: list,
    *,
    batch_size: int = 4,
) -> list[dict[str, np.ndarray]]:
    """Convenience batcher; preferred entrypoint for the hex-grid app's
    multi-tile workflow."""
```

**Contract guarantees the roadmap must protect:**
- Output spatial shape matches input spatial shape after de-preprocessing.
- `land_cover` and `topography` arrays sum to 1.0 along the class axis (within fp16 epsilon) when `return_logits=False`.
- Class index → name mapping is stable across versions of the checkpoint as long as `model.taxonomy_hash` is unchanged. A new taxonomy hash is a breaking change.
- `WATER_TOPO = 255` sentinel policy on the topography head's water pixels matches the training-time convention in `scripts/biome_mapping.py` and `scripts/historical/dem.py`.

### Differentiators — affordable, not blockers

These are features that improve the v1 deliverable but the roadmap can drop any of them under time pressure without violating the v1 contract.

| # | Feature | Value Proposition | Complexity | Notes |
|---|---------|-------------------|------------|-------|
| D-1 | **Multi-style synthetic rendering** (flat / illustrated / satellite) | Already implemented (`scripts/render.py`); confirm coverage at v1 ship | DONE (M) | Already in tree. Confirm v1 doesn't regress this. |
| D-2 | **Map-artifact augmentation** (parchment, sepia, faded, grayscale) | Bridges the synthetic→historical visual gap, which is the dominant distribution shift in v1 training | DONE in `scripts/augment.py` (S to wire in) | `CONCERNS.md` 2a flags `augment.py` as wired to *no* current pipeline — it's dead code today. Wiring it back into the synthetic and OSM training paths is a S-cost differentiator. Drop the `make_grid` / hex-tile language while doing so. |
| D-3 | **Random rotation + small-angle jitter at training time** | Real maps in the target use case (fantasy maps) appear at arbitrary rotations; in-domain robustness is essentially free if added to the augmentation stack | S | One Pillow rotate-and-resize + same rotation applied to label rasters. NB this also supports the rotationally-invariant OCR module (`MODEL-02`) by giving it rotated-text exposure. |
| D-4 | **OSM road-tile augmentation diversity** (multiple zoom levels, multiple render styles e.g. Standard, Carto Positron, Mapnik B&W) | Three different OSM stylesheets is much cheaper than three different data sources; cheap robustness on the third v1 source (`DATA-05`) | M | Adds rendering pre-step to `DATA-05`. Not blocking — single-style OSM still ships v1. |
| D-5 | **Bootstrap-loop quality monitoring** (does v1 actually beat v0 on held-out, or did georef noise tank it?) | Early-warning if `GEOREF-02` produces dirty registrations; gives the owner a kill switch on the bootstrap loop | S | Run `EVAL-01` on v0 *and* v1 against the same held-out set; report delta. If v1 < v0, abort the bootstrap and ship v0. This is also a paper-credibility item if milestone 2 happens (clean ablation). |
| D-6 | **Per-class loss weighting hooks in the training loop** | `DATA-06` is locked: per-source class-conditional weights from `sample_weights.json`. The implementation choice — table-driven vs configurable hooks — is a differentiator. Hooks let v1 sweep weights without code edits if the bootstrap loop reveals a class that's drifting. | S | Already-locked feature; this entry just confirms it's table-stakes-not-differentiator at the **functionality** level. The differentiator is **configurability** of the hook — pass a YAML or class-weight dict in via CLI. |
| D-7 | **Inference quantisation (INT8 dynamic, optionally INT4 via bitsandbytes)** | Tightens the "single CPU / 4–8 GB GPU" constraint with margin; a 4-bit checkpoint may run on 2 GB VRAM and CPU latency drops 2–4× | M | `bitsandbytes` is already in `requirements.txt`. INT8 dynamic quant on the seg heads is essentially free; INT4 needs validation against the EVAL-01 NLL number. Worth doing as a *post-ship* artefact rather than blocking v1 ship on it. |
| D-8 | **Batched inference helper** (TS-3's `predict_batched`) | The hex-grid app processes a map by tiling — without batching, every tile is its own `forward()` and the wrapper overhead dominates | S | One `for batch in chunked(images, batch_size): ...` wrapper; the win is large for the downstream consumer. Promote to table-stakes only if the hex-grid app explicitly asks for it; otherwise treat as P2 differentiator. |
| D-9 | **Tile-and-stitch helper** for inputs larger than the model's native input size | Fantasy maps are often 4096×4096 or larger; the backbone almost certainly accepts something smaller. The hex-grid app shouldn't have to know the model's native input size. | M | Overlap-and-feather tiling at the inference-module level. NB if the backbone is a ViT with patch16 at 224, native input is small enough that this is required-not-optional for any realistic input. Worth re-classifying as table-stakes if the chosen backbone (`MODEL-01`) has a native input ≤ 512. |
| D-10 | **`requirements.lock.txt` or `uv.lock`** (pinned deps) | The training-environment reproducibility gap (`CONCERNS.md` 8c) is a real risk; pinning is cheap | S | uv is already common in this stack; `uv pip compile requirements.txt > requirements.lock.txt` is one line. |
| D-11 | **Per-source class-conditional output dropout / noise on training labels with `sample_weights.json` weight 0** | The `HISTORICAL_LC_WEIGHTS` table downweights cropland/built-up/trees on historical maps — but downweighting is not the same as masking. If a class is essentially unreliable, masking it from the loss is cleaner than weighting it ε. Differentiator because it's a v1 quality lever the locked-DATA-06 spec doesn't pin. | S | One mask in the loss function; only trips when the implementation lands. |

**Differentiator cost summary:** D-1, D-5, D-10 are essentially free (≤ a few hours). D-2, D-3, D-6, D-8 are small (< a day each). D-4, D-7, D-9, D-11 are medium (1–3 days each). The roadmap can take all the S items and pick among the M items based on phase budget.

### Anti-Features — explicitly excluded from v1

These are features the codebase, README, or external pressure (paper framing, modder community, similar projects) might suggest but `PROJECT.md` Out-of-Scope explicitly excludes. Listed here to prevent re-promotion in subsequent planning sessions.

| Feature | Why Tempting | Why Excluded | Where Excluded | What to Do Instead |
|---------|--------------|--------------|----------------|--------------------|
| **Polygon tracing / region delineation** | Natural progression from per-pixel labels; common ask from grand-strategy modders; the README mentions it as a downstream step | Owned by the separate hex-grid app downstream | `PROJECT.md` Out-of-Scope: "Hex-grid aggregation, polygon tracing, region delineation" | Output per-pixel probabilities and stop. The hex-grid app handles tracing. |
| **Hex-grid aggregation** | The whole point of the system from the modder's POV | Lives in the separate hex-grid app | `PROJECT.md` Out-of-Scope | Output per-pixel probabilities; the hex-grid app aggregates. |
| **EU4 / CK3 / HoI4-specific output formats** (terrain.bmp, heightmap.png, province bitmaps, definitions.csv) | Modders want plug-and-play files; the README explicitly names these games | Engine packaging is the hex-grid app's job; v1 ships generic per-pixel probabilities only | `PROJECT.md` Out-of-Scope: "EU4 / CK3 / HoI4 engine-specific output formats" | Generic float16 prob arrays per the TS-11 contract. |
| **Hand-annotated fantasy test set** (Tolkien / Westeros / Abercrombie / Warhammer with ground truth) | Would give a defensible "real-domain" metric; matches the four target images named in `mockup.md` | Deferred to milestone 2 (paper-credibility item); `EVAL-02` is qualitative spot-check only at v1 | `PROJECT.md` Out-of-Scope: "Hand-annotated fantasy test set" | Use the four target images as `EVAL-02` qualitative renders only — no metric on them in v1. |
| **Hand-annotation tooling integration** (Label Studio, CVAT, Labelbox) | Naturally pairs with the previous item; `README.md` line 107 names them as candidates | Only relevant if a real-domain annotated test set is added later (milestone 2) | `PROJECT.md` Out-of-Scope: "Hand-annotation tooling integration" | Skip in v1. If milestone 2 picks this up, evaluate Label Studio (open-source, self-host) vs CVAT then. |
| **Auto-georef as an inference / app feature** | The codebase is building auto-georef (`GEOREF-01`) — the leap to "expose it as part of the shipped model" is small from a code-distance perspective | Modder's input is a fantasy map with no real-world coords; auto-georef has nothing to bind to | `PROJECT.md` Out-of-Scope: "Auto-georef as an inference / app feature" | Auto-georef is a training-data-prep tool only. Do not export it from the inference module. |
| **PaliGemma-3B as the inference backbone** | Largest, best-known multimodal model in the codebase's intended stack; named throughout `mockup.md` and `notes/architectural_references.md` | 6 GB FP16 weights break the single-CPU / 4–8 GB GPU inference budget | `PROJECT.md` Out-of-Scope: "PaliGemma-3B as the inference backbone" | Pick a small VL backbone in research phase (`MODEL-01`); PaliGemma-3B may be a paper-time comparison only (milestone 2). |
| **Zero-shot CLIP / SigLIP / OpenCLIP / PaliGemma baselines** | `mockup.md` and `executive_TODO.md` flag these prominently; they're the entry point for the dynamic-LRP failure analysis contribution | Paper-flavoured, deferred to milestone 2 | `PROJECT.md` Out-of-Scope: "Zero-shot CLIP / SigLIP / OpenCLIP / PaliGemma baselines" | Skip in v1. Milestone 2 picks this up if/when paper attempt happens. `executive_TODO.md` already tracks the owner-driven decision. |
| **Dynamic-LRP mechanistic failure analysis** | Listed as goal #3 in `mockup.md` ("a way to play with dynamic LRP") | Paper-flavoured, deferred to milestone 2; v1 is app-track | `PROJECT.md` Out-of-Scope: "Dynamic-LRP mechanistic failure analysis" | Park until milestone 2. The arXiv:2512.07010 link stays in `mockup.md` as the entry point. |
| **Component ablations** (backbone-only vs +OCR vs +seg-heads vs full system) | `mockup.md` line 105 explicitly names the ablation suite as a paper contribution | Paper-flavoured, deferred to milestone 2 | `PROJECT.md` Out-of-Scope: "Component ablations" | Skip in v1. The `D-5` bootstrap-loop quality monitor incidentally produces the v0-vs-v1 ablation data, which is a freebie if milestone 2 happens. |
| **Paper draft / write-up** | Goal #2/#3 for the owner | Milestone 2, opportunistic | `PROJECT.md` Out-of-Scope: "Paper draft / write-up" | Don't write any paper-shaped artefacts in v1 except as side-effects of EVAL-01/EVAL-02. `mockup.md` stays as a planning doc, not a draft. |
| **Open-dataset redistribution** (combined Rumsey + ESA + Copernicus + Azgaar set as a downloadable artefact) | Fourth contribution claim in `mockup.md`; would help paper credibility and modder community building | License risk on three of four sources unverified; redistribution not approved | `PROJECT.md` Out-of-Scope: "Redistribution of the combined training dataset" | Ship weights + code + recipe; modders supply their own input maps. `executive_TODO.md` tracks the licensing-decision item. |
| **Tests / CI / linting** (pytest, GitHub Actions, ruff, black, mypy) | Best-practice for any codebase about to grow; `CONCERNS.md` flags this as HIGH severity | Single-author research codebase, accepted risk at v1 | `PROJECT.md` Out-of-Scope: "Tests, CI, linting / formatting tooling" | Replaced at v1 by training-time logging (TS-7) as the observability surface. Revisit at milestone 2 or if collaborators join. |
| **Cloud / docker / VM provisioning** (Dockerfile, terraform, CI build) | Standard ML-engineering hygiene | Handled by a separate VM-provisioning repo per the existing project boundary | `PROJECT.md` Out-of-Scope: "Cloud / docker / VM provisioning" | This repo holds training scripts only. Cross-link the VM-provisioning repo from the README. |
| **Live georef-as-app-feature** | See "auto-georef as inference" above; redundant with that entry but worth re-stating | See above | See above | See above |
| **Real-time / streaming / live-edit inference** | Common modern ML expectation, especially for "interactive" tools | The use case is batch — a modder loads a fantasy map, gets a probability map, the hex-grid app digests it. No streaming need. | Implicit in `PROJECT.md` Constraints (downstream consumer is a separate hex-grid app, not a UI) | Stick with synchronous `predict()`. If interactive use ever matters, `D-7` quantisation is the lever. |
| **A web UI / Gradio demo** | One of the most common ways small ML projects get attention | Out of scope per goal hierarchy: v1 is "model handoff", not "demo app" | Implicit in `PROJECT.md` What-This-Is ("training pipeline and model... consumed by a separate hex-grid aggregator app — NOT a UI app") | The qualitative renders for `EVAL-02` are static pngs, not a live demo. A Gradio demo can ship at milestone 2 if useful for paper supplementary material. |
| **Multi-language OCR support** (Cyrillic, CJK, Arabic) | Real-world maps include non-Latin scripts | OCR module's `MODEL-02` scope is rotationally-invariant Latin reading; multi-script is a milestone 3+ extension | Not in `PROJECT.md` Active; not explicitly excluded but implicit from `MODEL-02` scope (CRAFT/ABCNet are Latin-trained starting points) | Latin-only in v1. Document the limitation in the inference module README. |
| **Confidence-calibrated probability output via temperature scaling** (full fit, not just diagnostic) | Mask-TS-Net 2024 and similar work show segmentation models are systematically miscalibrated; temperature-scaling fitting is an obvious next step | Diagnostic-only (TS-6) is enough for v1; full calibration is a quality-improvement move that requires a calibration set | Not in `PROJECT.md` either way — placing here as a "tempting addition that is actually milestone 2 work" | TS-6 sanity check only in v1. Full local-temperature-scaling fitting becomes a candidate if the v1 ECE is bad. |

## Feature Dependencies

```
TS-1 (training entrypoint)
    └─requires─> seed plumbing (CONCERNS.md 8a)
    └─requires─> TS-10 (deterministic splits)
    └─requires─> D-6 (per-class weight hook config)

TS-3 (inference module API)
    └─requires─> TS-2 (safetensors checkpoint format)
    └─requires─> TS-4 (probability output guarantee)
    └─requires─> TS-8 (model versioning, embedded in safetensors metadata)
    └─requires─> TS-11 (output shape contract)
    └─enhances──> D-8 (batched helper)
    └─enhances──> D-9 (tile-and-stitch helper)

TS-5 (eval script)
    └─requires─> TS-10 (deterministic splits) — without this, "held-out" is undefined
    └─enhances──> TS-6 (calibration sanity is a one-section addition to eval)
    └─enhances──> D-5 (bootstrap-loop quality monitor reuses the eval pipeline)

TS-6 (calibration sanity)
    └─requires─> TS-5
    └─requires─> TS-4

TS-8 (model versioning)
    └─requires─> TS-2 (safetensors metadata header is the natural place)
    └─requires─> dataset-build manifest (CONCERNS.md 8b)

D-2 (artifact augmentation wiring)
    └─requires─> reviving / repurposing scripts/augment.py (CONCERNS.md 2a)
    └─enhances──> D-3 (rotation augmentation)

D-5 (bootstrap-loop quality)
    └─requires─> TS-5 (uses the same eval script on v0 and v1)
    └─requires─> GEOREF-02 (PROJECT.md Active)

D-7 (quantisation)
    └─requires─> TS-3 + TS-2
    └─requires─> TS-5 (to validate post-quant NLL doesn't degrade)
    └─enhances──> Constraint: "single CPU or 4–8 GB GPU" (PROJECT.md)

D-8 (batched inference)
    └─requires─> TS-3
    └─enhances──> hex-grid app integration (downstream consumer)

D-9 (tile-and-stitch)
    └─requires─> TS-3
    └─may-be-required-by─> MODEL-01 backbone choice (if native input ≤ 512)

D-10 (lockfile)
    └─enhances──> TS-1 (reproducibility)
    └─enhances──> CONCERNS.md 8c
```

### Dependency notes

- **TS-3 sits at the centre.** Six other features either feed it or extend it. The roadmap should treat the inference module as a phase boundary — most other v1 work converges on it.
- **TS-5 → TS-6 → D-5 form a chain.** Build the eval script with calibration as a section, then reuse the script for the bootstrap-loop monitor. Three features, one shared codepath.
- **D-9 may quietly become table-stakes once `MODEL-01` is decided.** Almost any ViT or hierarchical segmentation backbone has a native input < 1024; fantasy maps are routinely larger. Re-evaluate this at the end of `MODEL-01` research.
- **D-2 is the cleanup lane for `CONCERNS.md` 2a (`scripts/augment.py` is dead code).** Wiring the parchment/sepia/faded augmentations into the synthetic and OSM training paths simultaneously serves a v1 differentiator and removes refactor debt — two birds.

## MVP Definition

### Launch With (v1)

The locked v1 set from `PROJECT.md` Active **plus** the eleven table-stakes features above, **plus** the cheap differentiators where doing them is essentially free.

Locked from `PROJECT.md` Active (named for completeness, not re-decided here):
- DATA-05, DATA-06, GEOREF-01, GEOREF-02, MODEL-01, MODEL-02, MODEL-03, TRAIN-01, TRAIN-02, EVAL-01, EVAL-02, SHIP-01

Adding from this research:
- [ ] **TS-1** Reproducible training entrypoint with config + seeding
- [ ] **TS-2** Safetensors checkpoint format
- [ ] **TS-3** Inference module with stable API per the sketch above
- [ ] **TS-4** Probability output (not argmax) on both heads
- [ ] **TS-5** Held-out eval script reporting NLL per source per class + JSON report
- [ ] **TS-6** Calibration sanity (reliability diagram + ECE) appended to eval
- [ ] **TS-7** Training-time logging (loss curves + sample preds; tensorboard or wandb)
- [ ] **TS-8** Model + dataset versioning embedded in checkpoint metadata
- [ ] **TS-9** CPU and 4–8 GB GPU inference paths plumbed through `device='auto'`
- [ ] **TS-10** Deterministic train/val/test split from a seed + sample-id hash, committed
- [ ] **TS-11** Output shape + dtype contract documented in inference module README
- [ ] **D-1** Multi-style synthetic rendering — confirm v1 doesn't regress (already in tree)
- [ ] **D-2** Map-artifact augmentation wired into synthetic + OSM training paths
- [ ] **D-3** Random rotation + jitter at training time
- [ ] **D-5** Bootstrap-loop quality monitor (run eval on both v0 and v1)
- [ ] **D-6** Per-source class weights as configurable hook (not hardcoded)
- [ ] **D-10** `uv.lock` or `requirements.lock.txt`

### Add After v1 (v1.x — only if budget allows post-ship)

- [ ] **D-4** OSM road-tile augmentation diversity (multiple stylesheets) — adds robustness, costs M
- [ ] **D-7** Inference quantisation (INT8 first, INT4 if needed) — relaxes the inference budget
- [ ] **D-8** Batched inference helper — only promote if the hex-grid app explicitly asks for it
- [ ] **D-9** Tile-and-stitch — promote to v1 table-stakes once `MODEL-01` is decided and confirmed to need it
- [ ] **D-11** Per-source class masking (vs weighting) — quality lever, only if EVAL-01 shows weighted-not-masked is hurting

### Future Consideration (milestone 2+)

Everything in the Anti-Features list above. Plus:

- [ ] Full temperature-scaling calibration fit (not just diagnostic) — quality of probability outputs
- [ ] Component ablations (backbone-only vs +OCR vs +heads vs full) — paper-side
- [ ] Hand-annotated fantasy test set — paper-side
- [ ] Dynamic-LRP failure analysis — paper-side
- [ ] Open-dataset redistribution — license-decision-gated
- [ ] Tests + CI + linting — accepted-risk at v1; revisit if collaborators or hardening
- [ ] Web UI / Gradio demo — only as paper supplementary material if needed

## Feature Prioritization Matrix

| Feature | User Value | Implementation Cost | Priority |
|---------|------------|---------------------|----------|
| TS-1 training entrypoint | HIGH | M | P1 |
| TS-2 safetensors | MEDIUM (security baseline) | S | P1 |
| TS-3 inference API | HIGH | M | P1 |
| TS-4 probability output | HIGH | S | P1 |
| TS-5 eval script | HIGH | S | P1 |
| TS-6 calibration sanity | MEDIUM | S | P1 |
| TS-7 training logging | MEDIUM | S | P1 |
| TS-8 versioning | HIGH | S | P1 |
| TS-9 CPU/GPU paths | HIGH | S | P1 |
| TS-10 deterministic splits | HIGH | S | P1 |
| TS-11 output contract docs | MEDIUM | S | P1 |
| D-1 multi-style synthetic | HIGH | DONE | P1 (regression-protect) |
| D-2 artifact aug wiring | HIGH | S | P1 |
| D-3 rotation aug | MEDIUM | S | P1 |
| D-4 OSM stylesheet diversity | MEDIUM | M | P2 |
| D-5 bootstrap quality | HIGH | S | P1 |
| D-6 per-source weights as hook | MEDIUM | S | P1 |
| D-7 quantisation | MEDIUM | M | P2 |
| D-8 batched helper | MEDIUM | S | P2 |
| D-9 tile-and-stitch | MEDIUM-to-HIGH | M | P2 (re-eval after MODEL-01) |
| D-10 lockfile | MEDIUM | S | P1 |
| D-11 class masking | LOW | S | P3 |

**Priority key:**
- P1: Must have for v1 launch
- P2: Should have, post-ship if budget
- P3: Nice to have, milestone 2+

## "Competitor" / reference feature analysis

This isn't a competitive product space (the hex-grid app is the only consumer; there is no peer model shipping for grand-strategy modders). But adjacent ML packages give us a rubric for "what does a serious model-handoff package look like":

| Feature | `transformers` (HF) | `segmentation_models_pytorch` | `mmsegmentation` | MapClass v1 plan |
|---------|---------------------|-------------------------------|------------------|------------------|
| Checkpoint format | safetensors (default) | torch .pth | mmengine custom | safetensors (TS-2) |
| Single-call inference API | `pipeline(...)` | `model(input)` (raw) | `inference_segmentor(...)` | `predict(model, image)` (TS-3) |
| Probability output | logits, user does softmax | logits, user does softmax | logits, user does softmax | **probabilities by default** (TS-4 — opinionated for the hex-grid app) |
| Versioning | `model.config.transformers_version` + revision hash | none | `cfg` dump | embedded in safetensors metadata (TS-8) |
| Quantisation | `bitsandbytes` integration | not directly | not directly | INT8/INT4 via bitsandbytes (D-7) |
| Calibration | none | none | none | **opinionated: ECE diagnostic in eval** (TS-6) |
| Training entry point | `Trainer` class | user's choice | `tools/train.py` | single `train.py` (TS-1) |
| Dataset versioning | none (HF datasets has revision) | none | none | manifest JSON (TS-8) |

**Where MapClass differs from the reference packages:**
- **Probabilities by default, not logits.** The hex-grid app's input is probabilities; we don't make every consumer redo `softmax`.
- **Calibration diagnostic is in the eval script.** The reference packages punt this; we don't, because over-confident outputs would silently corrupt hex-grid aggregation.
- **Embedded checkpoint metadata.** None of the references do this consistently; we do because the hex-grid app needs it and it's cheap with safetensors.

## Quality Gate Verification

- [x] **No table-stakes feature is in `PROJECT.md` Out-of-Scope.** Verified item-by-item — TS-1..TS-11 each cross-checked against the Out-of-Scope list. None hit.
- [x] **Differentiator costs are explicit.** Each D-N has a complexity estimate (S/M/L) and the cost summary at the end of the differentiator table tells the roadmap exactly which ones are essentially free.
- [x] **Anti-features list everything tempting but excluded.** 18 anti-features documented, each with the *why-tempting* hook and the explicit `PROJECT.md` Out-of-Scope reference (or implicit-from-Constraints reference).
- [x] **Inference module API surface sketched.** `load_model`, `predict`, `predict_batched` signatures provided with input/output shape and dtype contracts. The hex-grid app integration depends on this — captured at the precision the roadmap can convert into REQUIREMENTS items.

## Sources

- [Hugging Face Safetensors Support in PyTorch Distributed Checkpointing — pytorch.org blog](https://pytorch.org/blog/huggingface-safetensors-support-in-pytorch-distributed-checkpointing/)
- [Safetensors vs PyTorch — codegenes.net](https://www.codegenes.net/blog/safetensors-vs-pytorch/)
- [Mask-TS Net: Mask Temperature Scaling Uncertainty Calibration for Polyp Segmentation (2024) — arXiv:2405.05830](https://arxiv.org/abs/2405.05830)
- [Local Temperature Scaling for Probability Calibration (2020) — arXiv:2008.05105](https://ar5iv.labs.arxiv.org/html/2008.05105)
- [Neural Network Calibration — geoffpleiss.com](https://geoffpleiss.com/blog/nn_calibration.html)
- [Zarr v2 driver — TensorStore](https://google.github.io/tensorstore/driver/zarr/index.html)
- [TIA Toolbox nucleus instance segmentation (probability map output convention)](https://tia-toolbox.readthedocs.io/en/stable/_notebooks/jnb/08-nucleus-instance-segmentation.html)
- `/home/drdreadknee/mapclass/.planning/PROJECT.md` — locked v1 scope (Active + Out of Scope sections, both binding)
- `/home/drdreadknee/mapclass/.planning/codebase/ARCHITECTURE.md` — implemented vs planned component split
- `/home/drdreadknee/mapclass/.planning/codebase/CONCERNS.md` — items 2a (`augment.py` dead code), 8a (no seeds), 8b (no dataset versioning), 8c (no lockfile) directly inform table-stakes
- `/home/drdreadknee/mapclass/.planning/codebase/STACK.md` — confirms `bitsandbytes`, `safetensors`-via-`transformers`, `peft` are already in `requirements.txt`
- `/home/drdreadknee/mapclass/scripts/augment.py` — confirms parchment/sepia/faded augmentations exist but are unwired (informs D-2)
- `/home/drdreadknee/mapclass/notes/architectural_references.md` — DINOv2 / Swin frozen-backbone direction informs D-9 (tile-and-stitch likely table-stakes)
- `/home/drdreadknee/mapclass/executive_TODO.md` — confirms zero-shot baselines and licensing are owner-tracked, not v1 scope (informs anti-features)

---
*Feature research for: model-handoff package (multi-source-trained dense-prediction VL+OCR+heads model)*
*Researched: 2026-05-08*
