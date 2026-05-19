# Phase 1: Pinned Environment, GCS Mirrors, and a Verified Single-Slice Attribution - Research

**Researched:** 2026-05-18
**Domain:** dynamic LRP (keeinlev/dynamicLRP) adapted to SigLIP-2-so400m-patch14-384 as a contrastive encoder, for query-driven pixel attribution on Rumsey maps; idempotent GCS mirroring; JupyterLab inspection on an ephemeral GCP VM
**Confidence:** HIGH on stack/API/architecture/mirror mechanics; MEDIUM-LOW on whether dynamicLRP produces *faithful* (not just runnable) attribution for SigLIP-2's MAP-pool head as a contrastive encoder — this is unverifiable by research and is exactly what the three sanity controls exist to test.

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Mirror the **full 1,544-entry Rumsey manifest** into `gs://mapclass-training-northeast1/data/` — not a bounded subset. Idempotent/validated/per-id-outcome-manifest robustness exercised at full scale.
- **D-02:** **Full ingest gates the phase.** Phase 1 is not "done" until the entire manifest is mirrored (with its outcome manifest complete) *and* the single slice is verified. Plan ingestion as a long-running, resumable step the phase must complete — not a background task the slice work races ahead of.
- **D-03:** The three sanity controls (query-swap, model-randomization, occlusion) are judged by **pure visual eyeball** — overlays side-by-side, human judges pass/fail by looking. No printed scalars, no asserts.
- **D-04 (ACCEPTED RISK):** Pure-eyeball judgment conflicts with PITFALLS.md Pitfall 4 (weakest defense against silent-wrong attribution). User explicitly accepted this. **All three controls remain mandatory** — only the judgment modality is visual. Downstream agents MUST implement all three controls and lay them out to make a wrong result visually obvious (query-swap heatmaps directly adjacent, randomized-weights overlay next to trained, occluded-vs-random side by side). MUST NOT silently add quantitative thresholds.
- **D-05:** The slice is the **highest manifest index** map (last list position, programmatic — no per-id user input).
- **D-06:** Locked control query: **`"a river"`**. Locked query-swap query: **`"a xylophone"`**.
- **D-07:** No guaranteed sharp landmark on the highest-index map. Overlay/alignment correctness (27×27 grid, 6-px edge discard, aspect un-squash) must still be demonstrated explicitly (e.g. overlay the raw patch grid on the slice map, confirm cell placement and right/bottom 6-px exclusion) — but visual landmark alignment is best-effort given the programmatic choice.
- **D-08:** **Vendor a copy** of `keeinlev/dynamicLRP`'s `src/lrp_engine/` into the repo and **record the source commit SHA** in a tracked file and in run metadata. Not a submodule, not clone-at-setup. Updating the engine is a deliberate manual re-vendor.

### Claude's Discretion
- Internal module layout under the package-of-modules + thin-notebook architecture — follow the research-prescribed shape; specific filenames/signatures are planner/researcher territory.
- Ingestion concurrency, backoff, validation mechanics — bounded by PITFALLS.md Pitfall 6; exact implementation is Claude's.
- Build sequencing within the phase — research prescribes env → mirrors → loaders → de-risk spike (reproduce `ViT.ipynb`, then swap SigLIP-2 + similarity target) → overlay → controls. Follow it.

### Deferred Ideas (OUT OF SCOPE)
None. Quantitative/printed control metrics were offered and explicitly declined (recorded as accepted risk D-04, not deferred). Genuine metrics/faithfulness scoring remains v2, out of scope.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ENV-01 | Reproducible Python env pinned to dynamicLRP's `requirements.txt` (torch==2.7.1, transformers==4.52.3), dynamicLRP vendored at a pinned commit SHA | Verbatim `requirements.txt` retrieved (Standard Stack). Pinned SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` (master HEAD 2026-05-04). Vendor target `third_party/dynamicLRP/` is NOT gitignored — tracked. Python 3.10 (`.python-version`). |
| DATA-01 | Idempotently mirror Rumsey images from manifest `image_url`s into `gs://.../data/` (skip-if-exists, records failures, originals preserved) | Manifest = 1,544-entry JSON list; each has `id`, `image_url` (Code Examples §Ingestion). Idempotency/validation pattern in Don't Hand-Roll + Common Pitfalls. |
| DATA-02 | Mirror SigLIP-2-so400m weights from HF into `gs://.../models/` | `huggingface_hub.snapshot_download` → recursive GCS upload (Code Examples §Model mirror). Repo id `google/siglip2-so400m-patch14-384`. |
| DATA-03 | Manifest access: ordered index → entry, selection counting *down* from a high index N | Manifest is an ordered Python `list`; index = list position. Highest index = `len-1` = `manifest[-1]` (D-05). `richness_score` is NOT monotonic with index — see Common Pitfalls. |
| DATA-04 | Image loader: manifest `id` → GCS object → usable image + local byte cache | GCS download → local disk cache keyed by id → PIL (Code Examples §Data loader). |
| MODEL-01 | Load SigLIP-2-so400m once from GCS mirror, model-aligned preprocessing (resize/normalize 384, patchify), records the resize transform | Checkpoint is the **fixed-res** variant (`SiglipImageProcessor`, not NaFlex). Load via `AutoModel`/`SiglipModel` + `AutoProcessor`, `attn_implementation="eager"`. Processor: 384×384 square resize, rescale 1/255, normalize mean/std 0.5 → [-1,1]. (Code Examples §Model loader.) |
| ATTR-01 | Single forward + dynamic-LRP for one (map, query), attributing `logits_per_image` similarity (detached text), reduce image-token relevance to per-patch grid | `LRPEngine` API verified (Architecture Patterns). Target = `output.logits_per_image[0,0]`. Forward + LRP in one scope (tensor-identity invariant). |
| ATTR-02 | Reconstruct patch relevance into 2D heatmap, fixed normalization, handle 27×27 grid + 384÷14 non-integer edge discard | floor(384/14)=27 → 27×27=729 patches; right/bottom 6 px discarded (`padding="valid"`). Reshape/un-squash math in Code Examples §Overlay. |
| ATTR-03 | Three sanity controls (query-swap, model-randomization, occlusion) as the Phase 1 correctness gate | Concrete pass/fail recipes in §Sanity Controls. Visual-eyeball per D-03/D-04. |
| VIZ-01 | Heatmap overlaid on source map (alpha-blended), inline in JupyterLab — Phase 1 finish line | matplotlib `imshow` overlay, patch-resolution primary + labeled-interpolated secondary (Code Examples §Overlay). |
</phase_requirements>

## Summary

This phase stands up the pinned toolchain, idempotently mirrors the full 1,544-map Rumsey manifest and the SigLIP-2 weights to GCS, then builds and **proves correct** the single-slice attribution primitive. All of the genuine technical risk lives in one boundary: wiring `keeinlev/dynamicLRP`'s `LRPEngine` to attribute SigLIP-2's `logits_per_image[0,0]` similarity scalar back to the input pixel tensor, then folding the per-patch relevance onto the 27×27 grid with correct 6-px-discard and aspect un-squash handling.

The `LRPEngine` API is verified from source: it operates post-hoc on the autograd graph (no model surgery), the input tensor must carry `requires_grad_()` and be the *same object* passed to both the forward and `engine.params_to_interpret`, and `engine.run(scalar)` returns a 2-tuple `(checkpoint_vals, param_node_vals)` where `param_node_vals[i]` is the relevance for `params_to_interpret[i]`. The chosen checkpoint `google/siglip2-so400m-patch14-384` is the **fixed-resolution** variant — it carries `model_type: "siglip"` and a plain `SiglipImageProcessor`, so it loads via `AutoModel`/`SiglipModel` + `AutoProcessor` (NOT the NaFlex `Siglip2*` path with `pixel_attention_mask`/`spatial_shapes`). The processor squashes to 384×384 and rescales to [-1,1] via mean/std 0.5.

The paper reports SigLIP-2-so400m at 100% node coverage, so the engine will very likely *run*. It does NOT validate faithfulness for SigLIP-2 as a contrastive dual-encoder with a MAP attention-pooling head — that is unverifiable by research and is precisely why the three sanity controls (query-swap, model-randomization, occlusion) are the mandatory Phase 1 gate, judged visually per the user's accepted risk (D-03/D-04). The recommended fallback ladder if op coverage or faithfulness fails is documented and concrete.

**Primary recommendation:** Vendor `dynamicLRP/src/lrp_engine/` at SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e`, mirror its `requirements.txt` verbatim, build the package-of-modules per ARCHITECTURE.md, sequence the build as a de-risk spike (reproduce ViT.ipynb → swap SigLIP-2 + `logits_per_image[0,0]` target → 27×27 overlay → three visual controls). Keep forward + LRP in one function scope; never clone/detach the input tensor between modules.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Manifest indexing / count-down-from-N | Local process (in-memory) | — | `metadata/rumsey_manifest.json` is a local tracked file; pure list-index logic |
| Rumsey image ingest → GCS | Setup script (run-once, network) | GCS storage | Third-party fetch + validation + idempotent upload; never in the loop |
| SigLIP-2 weights mirror → GCS | Setup script (run-once, network) | GCS storage | HF snapshot → GCS; never in the loop |
| Model load (SigLIP-2 + processor) | Compute / GPU (singleton per kernel) | GCS storage (source) | Multi-GB load; amortize via module-level singleton |
| Image load by id | Compute (with local disk cache) | GCS storage (source) | Cache-first; GCS only on cache miss |
| Forward + dynamic LRP | Compute / GPU (one function scope) | — | Tensor-identity + live `grad_fn` graph must not cross module boundaries |
| Patch→pixel overlay | Compute (CPU/numpy + matplotlib) | — | Pure transform on the returned relevance array; owns all visual logic |
| Inspection / eyeball | JupyterLab on VM (SSH-tunnelled) | — | Only inspection surface; thin cells, no logic |

## Standard Stack

The stack is fully determined by `keeinlev/dynamicLRP`'s `requirements.txt` (mirrored verbatim) plus project additions. CLAUDE.md already contains the HIGH-confidence pinned table; this section records only what was re-verified this session.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `keeinlev/dynamicLRP` | vendored @ SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` | The LRP engine (`LRPEngine`) | The whole project premise. No PyPI package — vendor `src/lrp_engine/` (D-08). `[VERIFIED: github.com/keeinlev/dynamicLRP master HEAD, fetched 2026-05-18]` |
| `torch` | `2.7.1` | Autograd graph dynamicLRP traverses | Hard pin from LRP `requirements.txt` `[VERIFIED: requirements.txt fetched]` |
| `transformers` | `4.52.3` | Loads SigLIP-2; provides `logits_per_image` | Pinned by LRP repo; ≥4.49.0 needed for siglip2; SigLIP-2 also loadable via the older `Siglip*` classes since this checkpoint is `model_type: siglip` `[VERIFIED: requirements.txt + HF docs v4.52.3]` |
| `google/siglip2-so400m-patch14-384` | HF snapshot → GCS mirror | The VLM under attribution | Fixed by PROJECT.md. Fixed-res variant (`SiglipImageProcessor`, `model_type: siglip`) `[CITED: huggingface.co/google/siglip2-so400m-patch14-384 config.json + preprocessor_config.json]` |
| `google-cloud-storage` | `>=3.0,<4` | GCS mirror access via VM ADC | Official GCP client `[ASSUMED — version range from CLAUDE.md prior research, not re-verified this session]` |
| `huggingface_hub` | (transitive of transformers) | `snapshot_download` for the model mirror | Standard HF download path `[CITED: HF docs]` |
| JupyterLab | `>=4.4,<5` | Inspection surface | Per README `[ASSUMED — from CLAUDE.md prior research]` |

### Supporting (from dynamicLRP requirements.txt — pin verbatim)
| Library | Version | Notes |
|---------|---------|-------|
| `einops` | `0.8.1` | LRP pin `[VERIFIED: requirements.txt]` |
| `scikit_learn` | `1.7.0` | LRP pin (note underscore spelling in the file) `[VERIFIED: requirements.txt]` |
| `tqdm` | `4.66.4` | LRP pin `[VERIFIED: requirements.txt]` |
| `omegaconf` | `2.3.0` | LRP pin `[VERIFIED: requirements.txt]` |
| `matplotlib` | `3.8.0` | LRP pin; also the overlay tool `[VERIFIED: requirements.txt]` |
| `seaborn` | `0.13.2` | LRP pin `[VERIFIED: requirements.txt]` |
| `timm` | `1.0.20` | LRP pin `[VERIFIED: requirements.txt]` |
| `datasets` | unpinned in LRP → pin `>=3,<4` | LRP leaves unpinned; pin conservatively to protect torch 2.7.1 `[VERIFIED: requirements.txt shows unpinned; range is CLAUDE.md recommendation, ASSUMED]` |
| `captum` | unpinned in LRP → pin `>=0.7,<0.9` | Same; also IG fallback baseline `[VERIFIED: requirements.txt shows unpinned; range ASSUMED]` |

### Project additions (not in LRP requirements.txt)
| Library | Version | Purpose |
|---------|---------|---------|
| `torchvision` | `0.22.1` | Must match torch 2.7.1 exactly `[ASSUMED — CLAUDE.md prior research; verify torchvision↔torch 2.7.x pairing at install]` |
| `Pillow` | `>=10.3,<12` | Read/resize Rumsey JPEGs `[ASSUMED]` |
| `google-cloud-storage` | `>=3.0,<4` | (see Core) |
| `requests` | latest stable | Rumsey image download (setup-only) `[ASSUMED]` |
| `ipykernel` | latest stable | Register venv kernel (README) `[ASSUMED]` |

**Verbatim `dynamicLRP/requirements.txt` (fetched 2026-05-18, master):**
```
einops==0.8.1
scikit_learn==1.7.0
torch==2.7.1
tqdm==4.66.4
transformers==4.52.3
omegaconf==2.3.0
matplotlib==3.8.0
seaborn==0.13.2
timm==1.0.20
datasets
captum
```

**Installation (per README + D-08 vendoring):**
```bash
# Python 3.10 (.python-version pins 3.10; CLAUDE.md confirms 3.10–3.12 OK, avoid 3.13)
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt   # LRP pins verbatim + project additions
.venv/bin/python -m ipykernel install --user --name mapclass --display-name "mapclass (.venv)"
# dynamicLRP is vendored in-tree at third_party/dynamicLRP/ (D-08) — NOT pip-installed, NOT cloned at setup
```

**Version verification note:** `requirements.txt`, the master HEAD SHA, the SigLIP-2 `config.json`/`preprocessor_config.json`, and the `transformers` SigLIP/SigLIP2 docs were re-verified this session (2026-05-18). `google-cloud-storage`, `torchvision`, `Pillow`, JupyterLab ranges are carried from CLAUDE.md prior research and tagged `[ASSUMED]` — re-verify at install with `pip index versions <pkg>` and confirm the torchvision↔torch 2.7.1 pairing.

## Package Legitimacy Audit

slopcheck was not run (no network package-audit tooling invoked this session). All pip dependencies originate from an authoritative source — `dynamicLRP/requirements.txt` fetched directly from the pinned GitHub repo — except the project additions carried from CLAUDE.md prior research.

| Package | Registry | Provenance | Disposition |
|---------|----------|-----------|-------------|
| torch, transformers, einops, scikit_learn, tqdm, omegaconf, matplotlib, seaborn, timm, datasets, captum | PyPI | Verbatim from `keeinlev/dynamicLRP/requirements.txt` @ pinned SHA | Approved (authoritative source) |
| torchvision, Pillow, google-cloud-storage, requests, ipykernel | PyPI | CLAUDE.md prior research, well-known mainstream packages | `[ASSUMED]` — planner SHOULD gate the first `pip install` behind a human-verify checkpoint and confirm versions resolve cleanly against torch 2.7.1 |
| `keeinlev/dynamicLRP` | GitHub (not PyPI) | Vendored at SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` (D-08) | Approved — `pip install git+...` deliberately NOT used (no packaging metadata); vendored copy is the trust anchor, SHA recorded |

**Packages removed due to slopcheck [SLOP]:** none (slopcheck not run).
**Packages flagged [SUS]:** none. All names are mainstream and the core set comes verbatim from the pinned upstream requirements file.

## Architecture Patterns

Follow ARCHITECTURE.md verbatim — package-of-modules + thin notebooks, GCS as the only persistence layer. Recommended structure (from ARCHITECTURE.md §Recommended Project Structure):

```
mapclass/
├── metadata/rumsey_manifest.json        # local source of truth (1,544 entries)
├── src/mapclass/
│   ├── config.py                        # bucket, model id, GCS prefixes, device, vendored-SHA constant
│   ├── manifest.py                      # ordered-index reader + count-down-from-N
│   ├── ingest_images.py                 # run-once idempotent: image_url → GCS data/
│   ├── mirror_model.py                  # run-once idempotent: HF → GCS models/
│   ├── model_loader.py                  # GCS models/ → SigLIP-2 + processor (singleton, eager attn)
│   ├── data_loader.py                   # id → GCS data/ → (tensor.requires_grad_(), PIL)
│   ├── attribution.py                   # SigLIP-2 forward + LRPEngine in ONE scope → relevance
│   └── overlay.py                       # 729 → 27×27 → un-squash → composite on source
├── notebooks/01_single_slice.ipynb      # Phase 1 finish line
├── third_party/dynamicLRP/              # vendored src/lrp_engine/ + recorded SHA (tracked, NOT gitignored)
├── requirements.txt
└── README.md
```

### System Architecture Diagram (Single-Slice — the Phase 1 loop)

```
metadata/rumsey_manifest.json
        │  manifest.py: index = -1 (highest, D-05)
        ▼
   entry{id, image_url}
        │
        ├──────────────► (setup, run-once, idempotent) ──────────────┐
        │  ingest_images.py: GET image_url → validate → GCS data/<id> │
        │  mirror_model.py:  HF snapshot → GCS models/siglip2-so400m  │
        │                                                             ▼
        │                                          gs://mapclass-training-northeast1
        │                                            /data/<id>.jpg   /models/...
        ▼                                                  │           │
data_loader.py: GCS data/<id> ─(local disk cache)─► PIL ───┘           │
        │  AutoProcessor (384 squash, [-1,1]) → img_tensor             │
        │  .unsqueeze(0).to(device).requires_grad_()  ◄── SAME OBJECT  │
        │                                                              ▼
        │                                  model_loader.py (singleton, eager attn)
        │                                       SigLIP-2 (SiglipModel) + processor
        ▼                                                  │
   text query "a river" (D-06) ──► processor(text=..., padding="max_length", max_length=64)
        │                                                  │
        ▼                                                  ▼
attribution.py  (ONE function scope — forward + LRP share the live graph):
   output = model(pixel_values=img_tensor, input_ids=..., attention_mask=...)
   target = output.logits_per_image[0, 0]              # detached-text similarity scalar
   engine = LRPEngine(use_gamma=False, no_recompile=True, relevance_filter=0.5)
   engine.params_to_interpret = [img_tensor]           # SAME object as forward input
   ckpt_vals, param_vals = engine.run(target)
   relevance = param_vals[0]                            # shape == img_tensor shape
        │  sum/abs over channel → 729-vector (or per-patch reduce)
        ▼
overlay.py: 729 → reshape(27,27) → map onto valid 378×378 region
        │   → un-squash to original PIL aspect → alpha-blend (patch-res primary)
        ▼
notebooks/01_single_slice.ipynb: display(overlay) + 3 controls side-by-side
        ▼
   human eyeballs (D-03) → PHASE 1 DONE
```

### Pattern 1: dynamicLRP `LRPEngine` integration (VERIFIED from source)

**What:** Post-hoc attribution on the autograd graph. No hooks, no module replacement.
**When to use:** Always — this is the core primitive.

`LRPEngine.__init__` signature (verbatim from `src/lrp_engine/lrp.py`, master, 2026-05-18):
```python
def __init__(self,
             params_to_interpret=None,
             no_recompile=False,
             use_gamma=False,
             conv_gamma=100.0,
             mm_gamma=1.0,
             use_z_plus=False,
             use_attn_lrp=False,
             use_bilinear_mm=False,
             relevance_filter=1.0,
             starting_relevance=None,
             topk=1,
             dtype=torch.float32,
             with_grad=False)
```

`run()` signature and return (verbatim):
```python
def run(self, output_tuple_or_tensor: Union[tuple[torch.Tensor], torch.Tensor])
# Returns a 2-tuple of lists:
#   checkpoint_vals  – relevances from hooked LRPCheckpoint nodes
#   param_node_vals  – relevances positionally matching params_to_interpret
```

Static graph-inspection helper (verbatim):
```python
@staticmethod
def get_model_operations(model_output)   # returns op-set names, count, graph structure
```

**Canonical usage (adapted from `src/experiments/ViT.ipynb`, verbatim pattern):**
```python
# ViT.ipynb does, in order:
img_tensor = imgs_list[0].unsqueeze(0).to(device).requires_grad_()
output = vit_model(img_tensor)
lrp_engine = LRPEngine(use_gamma=True, no_recompile=True)
lrp_engine.params_to_interpret = [img_tensor]
lrp_output = lrp_engine.run(output.logits)        # ViT: classification logits
# lrp_output == ([], [relevance_tensor])  → relevance_tensor matches img_tensor shape
# ViT reshapes:  relevance_map = patch_relevance[-1].reshape(14,14)   (224/16=14)
# overlay:       plt.imshow(patch_relevance, cmap='bwr', alpha=0.5)
```

**SigLIP-2 adaptation (the one change vs. ViT):** swap the target from `output.logits` to `output.logits_per_image[0, 0]`. Everything else (tensor identity, engine construction, `params_to_interpret`, `run`, relevance extraction from `param_node_vals[0]`) is the same mechanism.

### Pattern 2: Forward + LRP in ONE function scope (the tensor-identity invariant)

**What:** `attribution.py` owns both the model forward AND `engine.run` in a single function.
**Why:** `LRPEngine` walks the live `grad_fn` chain of the output and resolves `params_to_interpret` by tensor object identity. If the input tensor is cloned, detached, re-`.to()`'d, or the forward scope ends before `run()`, the graph is gone and relevance cannot be computed. The data loader must hand back the *exact same* `requires_grad_()` tensor object that flows into both the forward and `params_to_interpret`.
**Anti-pattern:** a "model service" returning logits + a separate "attribution service" running LRP later. Forbidden (ARCHITECTURE.md Anti-Pattern 1).

### Pattern 3: Extending op coverage via a Promise (the fallback surface)

**Extension surface (verified directory listing of `src/lrp_engine/`):**
- `promises/` — backward-promise classes, base in `promises/promise.py`
- `lrp_prop_fcns.py` — propagation functions
- `model_specific/` — `torch.autograd.Function` subclasses for model-specific custom ops (e.g. `mosaicbert.py` defines `IndexFirstAxis`/`IndexPutFirstAxis` as `torch.autograd.Function` with static `forward`/`backward`, exposed via `.apply`)

`promises/promise.py` base class (verbatim signatures):
```python
def __init__(self, promise, traversal_ind, bucket, *args, **kwargs)
# Abstract members a subclass must implement:
@property @abstractmethod arg
@property @abstractmethod op_result      # forward result of the op on the promise args
@property @abstractmethod rin            # incoming relevance
@abstractmethod _setarg(self, value)
@abstractmethod set_rin(self, new_rin)
@abstractmethod compute_rins(self)       # the propagation function for this Node
# Instantiation entry point (registration happens via promise_type string):
def instantiate_promise(self, r, promise_type, num_branches, origin_node_ind,
                         branch_shapes=None, extra_args={})
```
Concrete example shape (`promises/softmax_backward_promise.py`): `class SoftmaxBackwardPromise(DummyPromise)` overriding `op_result`, `compute_rins(self)` (AttnLRP logic), `refresh_metadata(self, node)`, constructed as `__init__(self, promise, traversal_ind, bucket, saved_result, saved_dim)`.

**When to use:** ONLY if `LRPEngine.get_model_operations(output)` reports an uncovered op on SigLIP-2's forward, OR if the occlusion control fails (MAP-head relevance distortion). See §Fallback Ladder.

### Anti-Patterns to Avoid
- **Splitting forward and LRP across modules** — breaks the live graph (Pattern 2).
- **Loading the model / fetching from HF or Rumsey inside the loop** — re-pays multi-GB load; reintroduces the runtime dependency the GCS mirror exists to remove. Singleton model loader; loop reads only GCS + local cache.
- **Building the sweep before the slice is visually trusted** — Phase 2 scope; a wrong integration hides behind plausible heatmaps.
- **Re-deriving dynamic LRP from the paper** — vendor + use the reference impl; SigLIP adaptation is a thin shim at most, never a rewrite.
- **Using `model.get_image_features()` / image-embedding norm as the target** — produces query-independent heatmaps. Target MUST be `logits_per_image[0,0]`.
- **Using the NaFlex `Siglip2ImageProcessor` / `pixel_attention_mask` / `spatial_shapes` path** — this checkpoint is fixed-res `model_type: siglip`; use `AutoProcessor`/`SiglipImageProcessor` square-384 path.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LRP relevance propagation | Custom backward rules from the paper | vendored `LRPEngine` | Project's explicit bet is "less work"; 47-op engine, Promise system |
| SigLIP-2 preprocessing | Hand transform / reused CLIP/ImageNet norm | official `AutoProcessor` (`SiglipImageProcessor`) | Wrong normalization → OOD inputs → plausible-but-meaningless heatmaps (Pitfall 8) |
| HF model download | `requests` to HF URLs | `huggingface_hub.snapshot_download` | Handles sharding, resume, revision pinning |
| GCS upload/download | Raw HTTP | `google-cloud-storage` blob API via ADC | ADC on VM; idempotent `blob.exists()` check is one call |
| Idempotent resumable mirror | Bespoke state file | `blob.exists()` + size check + per-id outcome JSON | Standard skip-if-present-and-valid pattern |
| Image decode safety | Trust raw bytes | `PIL.Image.open(...).verify()` then re-open + decode; cap dimensions | Rumsey is third-party; decompression-bomb / truncated-file risk |

**Key insight:** Every custom solution in this domain (custom LRP, hand transform, bespoke mirror state) is a known silent-wrong-result vector documented in PITFALLS.md. The reference implementations exist precisely to avoid them.

## Runtime State Inventory

Greenfield repo (only `.planning/`, `CLAUDE.md`, `README.md`, `metadata/`, `.gitignore`, `.python-version`). No prior runtime state to migrate. Two persistence stores will be *created* this phase:

| Category | Items | Action |
|----------|-------|--------|
| Stored data | GCS `data/` (1,544 mirrored images), GCS `models/` (SigLIP-2 snapshot), local disk image cache, per-id ingestion outcome manifest | Created fresh; idempotent so re-runs converge |
| Live service config | None — GCS bucket exists, ADC present per PROJECT.md | None |
| OS-registered state | JupyterLab kernel `mapclass` registered via `ipykernel install` (per README) | Re-registered on each VM rebuild — part of setup, not migration |
| Secrets/env vars | GCS ADC (assumed present on VM); no secret names embedded in code | None — verify ADC at job start (fail-fast) |
| Build artifacts | `.venv/` (gitignored), vendored `third_party/dynamicLRP/` (tracked, NOT gitignored) | Vendored copy + recorded SHA is the reproducibility anchor (D-08) |

**Nothing found requiring data migration:** confirmed — greenfield, verified by `ls` of repo root and `.gitignore` inspection.

## Common Pitfalls

(Full treatment in `.planning/research/PITFALLS.md`; this section surfaces the planning-critical specifics.)

### Pitfall A: `richness_score` is NOT monotonic with manifest index
**What goes wrong:** D-05 says "highest manifest index map (richest per index semantics)". Verified directly: `manifest[0].richness_score == 18`, `manifest[1543].richness_score == 0`. The index does **not** track `richness_score`. The locked decision D-05 is "highest *list index*" = `manifest[-1]` (id `RUMSEY~8~1~344476~90112460`) — a programmatic choice, NOT a richness-max choice.
**How to avoid:** Slice selection = `manifest[-1]` (last list position). Do NOT sort by or filter on `richness_score`. The "richer = high index" framing in PROJECT.md/CONTEXT does not hold in the actual data — follow the literal D-05 rule (highest list index), and note this discrepancy in the plan so no one "fixes" it by sorting.
**Warning sign:** A task that sorts the manifest by `richness_score` or asserts index↔richness correlation.

### Pitfall B: Wrong attribution target (contrastive, not classifier)
Target MUST be `output.logits_per_image[0, 0]` (single image, single query). Detach the text path so relevance flows only through pixels. Keep image-embedding L2-norm inside the traced graph. Drop sigmoid/temperature (monotonic). Query-swap is the detector.

### Pitfall C: MAP attention-pooling head routes relevance away from patches
SigLIP-2 has no CLS token; image embedding comes from `SiglipMultiheadAttentionPoolingHead`. The paper validates *graph coverage*, not *faithfulness*, for this head. Mitigation: prefer extracting relevance at `params_to_interpret=[img_tensor]` (input-level, which the engine supports directly) so the per-pixel relevance is read at the input, not at the post-pool embedding; the occlusion control is the only available faithfulness check.

### Pitfall D: 27×27 grid + 6-px edge discard + aspect un-squash
`floor(384/14)=27` → `27×27=729` patches; `padding="valid"` discards the **last 6 px of the right and bottom edges**. The processor *squashes* (not letterboxes) the non-square Rumsey scan to 384×384. Overlay chain: 729 → `reshape(27,27)` → place on the valid 378×378 region of the 384 tensor → invert the square-squash back to the original PIL aspect ratio → alpha-blend. Primary view = nearest-neighbour / block upsample (honest patch resolution); secondary smoothed view explicitly labeled "interpolated" (D-07 requires demonstrating the grid placement + 6-px exclusion explicitly).

### Pitfall E: LRP VRAM blowup on so400m
Promise system retains forward activations → VRAM far above inference. Batch size 1 always. **Measure peak VRAM on the first single-image attribution** before any sweep sizing (this is a Phase 1 deliverable per STATE.md Blockers). Free graph + `torch.cuda.empty_cache()` after each attribution call.

### Pitfall F: Non-idempotent / partial GCS ingest at 1,544 scale
Download to temp → validate (HTTP 200 + image content-type + byte length matches `Content-Length` + PIL-decodable) → atomic upload. Skip if blob exists AND size valid. Per-id outcome manifest (`ok` / `dead-url` / `decode-fail` / `skipped`). Throttle + exponential backoff on 429/5xx. Re-run must converge. ADC verified at job start (fail-fast).

### Pitfall G: Ephemeral-VM irreproducibility
Load SigLIP-2 ONLY from the GCS mirror, never HF at runtime. dynamicLRP vendored at recorded SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` (D-08). Set + record seeds (torch/numpy/cuda); note SDPA kernels may be nondeterministic — `attn_implementation="eager"` both lowers integration risk and improves determinism.

## Code Examples

> Signatures verified from upstream source / official docs as cited. Bodies are skeletal — the planner turns these into tasks.

### Manifest reader (DATA-03, D-05)
```python
# manifest.py  — manifest is a JSON LIST of 1,544 dicts; index = list position
import json
def load_manifest(path="metadata/rumsey_manifest.json") -> list[dict]:
    return json.load(open(path))                    # [VERIFIED: inspected, type=list len=1544]

def entry_by_index(m: list[dict], idx: int) -> dict:
    return m[idx]                                   # idx = -1 → highest index (D-05)

def count_down_from(m: list[dict], start_idx: int, count: int):
    for i in range(start_idx, max(start_idx - count, -1), -1):
        yield i, m[i]                               # descending; Phase 2 uses this
```

### Image ingest → GCS (DATA-01, Pitfall F)
```python
# ingest_images.py  — run-once, idempotent
# entry keys verified: 'id', 'image_url'  (e.g. id="RUMSEY~8~1~344476~90112460")
from google.cloud import storage
import requests, io
from PIL import Image
def mirror_one(bucket, entry) -> str:               # returns status string for outcome manifest
    blob = bucket.blob(f"data/{entry['id']}.jpg")
    if blob.exists() and blob.size and blob.size > 0:
        return "skipped"
    r = requests.get(entry["image_url"], timeout=60, stream=True)
    if r.status_code != 200: return "dead-url"
    if "image" not in r.headers.get("Content-Type", ""): return "bad-content-type"
    data = r.content
    try:
        Image.open(io.BytesIO(data)).verify()       # decodable check before write
    except Exception:
        return "decode-fail"
    blob.upload_from_string(data, content_type="image/jpeg")   # atomic; originals preserved
    return "ok"
# Loop with bounded concurrency + exponential backoff on 429/5xx; write per-id outcome JSON.
```

### Model mirror → GCS (DATA-02)
```python
# mirror_model.py  — run-once, idempotent
from huggingface_hub import snapshot_download
local = snapshot_download("google/siglip2-so400m-patch14-384")   # [CITED: HF]
# recursively upload local/* → gs://.../models/siglip2-so400m-patch14-384/<relpath>,
# skipping blobs that already exist with matching size.
```

### Model loader (MODEL-01) — singleton, GCS-only, eager attn
```python
# model_loader.py
from transformers import AutoModel, AutoProcessor
# 1. download gs://.../models/siglip2-so400m-patch14-384/ → local dir (cache-first)
# 2. load (fixed-res checkpoint: model_type "siglip", SiglipImageProcessor):
model = AutoModel.from_pretrained(local_dir, attn_implementation="eager").to(device).eval()
processor = AutoProcessor.from_pretrained(local_dir)
# [CITED: HF docs v4.52.3 — logits_per_image shape (image_bs, text_bs);
#         preprocessor_config.json: 384x384 square, rescale 1/255, mean/std 0.5 → [-1,1]]
```

### Data loader (DATA-04) — returns the SAME requires_grad tensor + original PIL
```python
# data_loader.py
from PIL import Image
def load_slice(bucket, processor, entry, query, device):
    # local-cache-first GCS fetch of data/<id>.jpg
    pil = Image.open(local_cached_path).convert("RGB")
    inputs = processor(text=[query], images=[pil],
                       padding="max_length", max_length=64,   # REQUIRED for SigLIP [CITED: HF]
                       return_tensors="pt").to(device)
    img_tensor = inputs["pixel_values"].requires_grad_()      # SAME object downstream
    return img_tensor, inputs["input_ids"], inputs["attention_mask"], pil
```

### Attribution (ATTR-01) — forward + LRP in ONE scope
```python
# attribution.py
import torch
from lrp_engine import LRPEngine     # vendored: third_party/dynamicLRP/src on sys.path
def attribute(model, img_tensor, input_ids, attention_mask):
    output = model(pixel_values=img_tensor, input_ids=input_ids,
                    attention_mask=attention_mask)
    target = output.logits_per_image[0, 0]            # detached-text similarity scalar (Pitfall B)
    engine = LRPEngine(use_gamma=False, no_recompile=True, relevance_filter=0.5)  # CLAUDE.md knobs
    engine.params_to_interpret = [img_tensor]         # SAME object as forward input
    ckpt_vals, param_vals = engine.run(target)        # 2-tuple of lists [VERIFIED: lrp.py]
    relevance = param_vals[0]                         # shape == img_tensor (1,3,384,384)
    torch.cuda.empty_cache()                          # Pitfall E
    return relevance
# Optional coverage probe before first run: LRPEngine.get_model_operations(output)
```

### Overlay (ATTR-02, VIZ-01, Pitfall D, D-07)
```python
# overlay.py
import numpy as np, torch
def to_patch_grid(relevance):                          # relevance: (1,3,384,384)
    r = relevance.detach().abs().sum(1)[0]             # → (384,384) signed→abs reduce
    valid = r[:378, :378]                              # discard last 6 px R/B (padding="valid")
    grid = valid.reshape(27, 14, 27, 14).mean((1, 3))  # → (27,27) per-patch mean
    g = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)  # FIXED normalization
    return g.cpu().numpy()                             # 27x27 in [0,1]
# Primary view: nearest-neighbour upsample 27x27 → original PIL aspect (un-squash the
#   384x384 square back to pil.size), plt.imshow(pil); plt.imshow(grid, cmap='bwr',
#   alpha=0.5, interpolation='nearest'). Secondary smoothed view labeled "interpolated".
# D-07: also render the bare 27x27 patch grid over the slice map to demonstrate cell
#   placement + the right/bottom 6-px exclusion explicitly.
```

## Sanity Controls (ATTR-03 — the Phase 1 gate)

All three are **mandatory** (D-04) and judged by **visual eyeball** (D-03) — no printed scalars, no asserts. Lay them out to make a wrong result visually obvious.

| Control | Procedure | Visual PASS (eyeball) | Visual FAIL |
|---------|-----------|-----------------------|-------------|
| **Query-swap** | Same slice map. Run with control query `"a river"` (D-06) and swap query `"a xylophone"` (D-06). Render both overlays **directly adjacent**. | The two heatmaps differ **substantially** — different regions highlighted. | The two heatmaps look essentially the same → target is image-saliency, not query-driven (Pitfall B). |
| **Model-randomization** | Re-init the SigLIP-2 **vision tower** weights randomly (e.g. `model.vision_model.apply(reinit_fn)`), keep text tower; re-run attribution on the slice with `"a river"`. Render randomized overlay **next to** the trained overlay. | Randomized overlay collapses to structureless noise vs. the structured trained overlay. | Randomized overlay still looks structured/similar → attribution is not reading the model (Adebayo sanity-check failure). |
| **Occlusion** | On the slice + `"a river"`: occlude (zero / mean-fill) the **top-k** highest-relevance 14×14 patches, recompute `logits_per_image[0,0]`; separately occlude **k random** patches, recompute. Show the original map, top-k-occluded map, random-occluded map **side by side** with their similarity values *rendered into the displayed figure as part of the visual* (this is a rendered overlay caption, not a printed scalar/assert — consistent with D-03 "no printed scalars"). | Occluding top-relevance patches visibly drops image-text similarity **more** than occluding random patches. | top-k drop ≈ random drop → MAP-pool relevance not faithful (Pitfall C). |

Phase 1 is **DONE** only when all three are rendered side-by-side in `01_single_slice.ipynb` and the human judges all three PASS by eye (D-02: AND full 1,544 mirror complete with outcome manifest).

## Fallback Ladder (op-coverage / faithfulness failure)

Concrete decision procedure (CLAUDE.md "Stack Patterns by Variant" + ARCHITECTURE.md):

1. **Probe first:** `LRPEngine.get_model_operations(output)` on the SigLIP-2 forward → does it report uncovered ops? If clean, run as-is.
2. **Engine erroring / uncovered op:** add a custom backward Promise in `third_party/dynamicLRP/src/lrp_engine/promises/` subclassing `Promise`/`DummyPromise` (pattern: `softmax_backward_promise.py`); if a model-specific custom autograd op is needed, follow `model_specific/mosaicbert.py` (`torch.autograd.Function` + `.apply`). Document the patch (deviates from the vendored SHA — record it).
3. **Runs but MAP-head distorts relevance** (occlusion control fails, sparse/flat map): try `use_attn_lrp=True`; and/or attribute on the **pre-pool** image-patch logits (raw `image_embeds @ text_embeds` before the MAP head) instead of post-pool `logits_per_image`.
4. **Memory knobs:** start `use_gamma=False`, `relevance_filter=0.5` (cut memory/noise); so400m fp32 single-image should fit a mid-tier GPU — if not, bf16 forward but keep `LRPEngine(dtype=torch.float32)` for the relevance pass.
5. **Last resort:** vendored LXT (`external/LRP-eXplains-Transformers` in the dynamicLRP repo) with a ViT-style patch; or `captum` Integrated Gradients on `logits_per_image[0,0]` as a degraded but trivially-correct baseline (captum is already a dep).

## State of the Art

| Old assumption | Current reality | Impact |
|----------------|-----------------|--------|
| "ViT coverage ⇒ SigLIP-2 coverage" | Paper Table A.4 = SigLIP-2 *graph coverage* (vision-only), not contrastive-encoder faithfulness | Three controls are non-negotiable |
| `manifest` ordered by richness | `richness_score` NOT monotonic with index (verified: idx0=18, idx1543=0) | Slice = `manifest[-1]` by position; do not sort by richness (Pitfall A) |
| Checkpoint is NaFlex `siglip2` | This checkpoint is fixed-res `model_type: "siglip"`, `SiglipImageProcessor` | Use `AutoModel`/`SiglipModel` + square-384 processor; NOT the NaFlex `pixel_attention_mask`/`spatial_shapes` path |
| dynamicLRP master HEAD unknown | HEAD = `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` (2026-05-04) | Pin/vendor at this SHA (D-08) |

**Deprecated/avoid:** `transformers>=5.0`, `torch≠2.7.1`, `pip install git+...dynamicLRP`, NaFlex `Siglip2ImageProcessor` for this checkpoint, `model.get_image_features()` as target.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `google-cloud-storage>=3.0,<4`, `torchvision==0.22.1`, `Pillow>=10.3,<12`, JupyterLab `>=4.4,<5` ranges | Standard Stack | Carried from CLAUDE.md prior research, not re-verified this session. Re-verify at install; torchvision MUST pair with torch 2.7.x or silent segfault |
| A2 | The NaFlex `Siglip2Model` API page's mean/std 0.5 / `padding="max_length", max_length=64` requirements apply identically to the fixed-res `SiglipModel` path | Code Examples | If `max_length` differs for the fixed-res tokenizer, text encoding is wrong → bad similarity. Verify against the loaded processor's config at runtime (low risk — SigLIP tokenizer default `model_max_length=64`, confirmed in docs) |
| A3 | `param_vals[0]` returns relevance with the same shape as `img_tensor` `(1,3,384,384)` | Code Examples / Pattern 1 | Inferred from ViT.ipynb (`patch_relevance[-1].reshape(14,14)` from a `[tensor]` return). If the engine returns a different layout, the overlay reshape changes — verify empirically in the de-risk spike |
| A4 | SigLIP-2 vision tower uses `SiglipMultiheadAttentionPoolingHead` (no CLS) | Pitfall C | HIGH-confidence from SigLIP-2 paper + ARCHITECTURE.md, but the generic transformers SigLIP doc page was vague. Confirm by inspecting the loaded `model.vision_model` — affects the model-randomization reinit target and pre-pool fallback |
| A5 | `LRPEngine.run(target)` accepts a 0-dim scalar tensor (`logits_per_image[0,0]`) | Code Examples | ViT passed `output.logits` (2-D). If `run` requires a 1-D/2-D tensor, wrap as `target.reshape(1)` or pass `logits_per_image[:1,:1]`. Resolve in the de-risk spike |
| A6 | GCS ADC is present and valid on the VM | Pitfall F/G | PROJECT.md asserts it; verify fail-fast at job start. If absent, ingestion + all loaders block |

## Empirical Risks (handled by plans — not planning blockers)

These three items are answered only at execution time (they require running the pinned stack on the real model/GPU — research cannot resolve them). They are **not open planning questions**: each is already handled concretely by a specific plan/task, with an explicit fallback path. They are listed here for traceability, not as blockers.

1. **Does `LRPEngine.run` accept a 0-dim scalar, or must the target be ≥1-D?**
   - Known: ViT.ipynb passes `output.logits` (2-D classification logits).
   - Unclear: whether `logits_per_image[0,0]` (0-dim) works directly.
   - Empirical handling: in the de-risk spike, first reproduce ViT.ipynb verbatim, then try `output.logits_per_image[0,0]`; if it errors on dimensionality, fall back to `output.logits_per_image[:1,:1]` or `.reshape(1)`. Cheap to resolve empirically.
   - **Resolved by:** Plan 01-03, Task 1 (`src/mapclass/attribution.py`) — the explicit 0-dim-vs-≥1-d target fallback path is implemented and unit-asserted (`tests/test_attribution_shape.py`); the de-risk spike in Plan 01-01, Task 2 first exercises the verbatim ViT 2-D path. Assumption A5.

2. **At what node does `params_to_interpret=[img_tensor]` relevance get read — does it already bypass MAP-pool distortion?**
   - Known: relevance is resolved by tensor identity at the *input* (`img_tensor`), so it is the relevance that survived all the way back through the MAP head to the pixels.
   - Unclear: whether MAP-head relevance smearing still corrupts the *input-level* map (it can — distortion happens during backprop, not only at the embedding).
   - Empirical handling: the occlusion control is the arbiter; if it fails, apply Fallback Ladder step 3 (pre-pool logits / `use_attn_lrp`).
   - **Resolved by:** Plan 01-03, Task 2 (occlusion control in `notebooks/01_single_slice.ipynb`) is the faithfulness arbiter; Plan 01-03, Task 1 documents the Fallback Ladder in `attribution.py` as the response path; the blocking human-verify checkpoint (Plan 01-03, Task 3) refuses approval unless the occlusion control visually PASSes. Pitfall C / §Fallback Ladder.

3. **Peak VRAM for so400m + LRP graph retention vs. the VM GPU.**
   - Known: paper shows multi-× growth on tiny models; so400m is ~400M params, 27 layers, 729 tokens.
   - Unclear: actual peak — unmeasurable without running.
   - Empirical handling: measure peak on the first single-image attribution; this is a Phase 1 deliverable that gates any Phase 2 sweep sizing (STATE.md Blocker).
   - **Resolved by:** Plan 01-03, Task 1 (`attribution.py` captures `torch.cuda.max_memory_allocated`) and Plan 01-03, Task 2 (the notebook prints and records peak VRAM for the single attribution; acceptance criterion gates Phase 2 sizing). Pitfall E / STATE.md Blocker.

## Environment Availability

| Dependency | Required By | Available | Notes / Fallback |
|------------|------------|-----------|------------------|
| GPU + CUDA matching torch 2.7.1 cu wheels | Attribution (LRP pass) | Assumed (GCP VM w/ GPU per PROJECT.md) | No fallback — LRP on so400m on CPU is impractical. Verify `torch.cuda.is_available()` at kernel start |
| GCS ADC | All GCS I/O | Assumed present (PROJECT.md) | No fallback — verify fail-fast at job start (A6) |
| Network to davidrumsey.com | `ingest_images` (run-once) | Assumed | Per-id outcome manifest absorbs dead URLs; budget for a fraction unavailable (STATE.md Blocker) |
| Network to huggingface.co | `mirror_model` (run-once) | Assumed | One-time; after mirror, never touched again |
| GitHub reachability | NOT required at runtime | — | D-08: dynamicLRP vendored in-tree; no clone/init step |
| Python 3.10 | Everything | `.python-version` = 3.10 | torch 2.7.1 supports 3.10–3.12; avoid 3.13 |

**No missing dependency with a viable code fallback** — all are infrastructure assumptions to verify fail-fast, not things the plan can route around.

## Validation Architecture

`.planning/config.json` had no `workflow.nyquist_validation` key → treat as enabled. **However**, this phase's correctness gate is, by explicit user decision (D-03/D-04), **visual eyeball of three controls — NOT automated asserts**. This is an accepted risk, not an oversight.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | None present (greenfield; no test/, no pytest.ini, no config.json test config) |
| Config file | none — see Wave 0 |
| Quick run command | n/a until Wave 0 |
| Full suite command | n/a until Wave 0 |

### Phase Requirements → Validation Map
| Req | Behavior | Validation Type | Mechanism |
|-----|----------|-----------------|-----------|
| ENV-01 | Env builds from pinned reqs; ViT.ipynb reproduces its heatmap | smoke (manual) | Run `ViT.ipynb` end-to-end; visually confirm heatmap reproduced (de-risk spike gate) |
| DATA-01 | Ingest idempotent, outcome manifest complete, no zero-byte objects | automatable | A re-run script that asserts: 2nd run = all "skipped", outcome JSON has 1,544 rows, no GCS blob with size 0 |
| DATA-02 | Model mirror present + idempotent | automatable | Assert key GCS blobs exist (config.json, model shards); 2nd run skips |
| DATA-03 | `entry_by_index(-1)` resolves the last entry | unit | `assert load_manifest()[-1]["id"] == "RUMSEY~8~1~344476~90112460"` |
| DATA-04 | Loader returns a usable image + cache works | unit/smoke | Load slice id; assert PIL size > 0; 2nd call hits local cache |
| MODEL-01 | Model loads from GCS; processor [-1,1] range | smoke | Assert tensor min≈-1, max≈+1, shape `(1,3,384,384)` |
| ATTR-01 | LRP returns relevance shaped like input | smoke | Assert `param_vals[0].shape == img_tensor.shape` |
| ATTR-02 | 27×27 grid, 6-px discard applied | unit | Assert grid shape `(27,27)`; assert valid-region slice is `[:378,:378]` |
| ATTR-03 | Three controls implemented + rendered side-by-side | **manual visual (D-03)** | Human eyeballs `01_single_slice.ipynb` — the gate; NOT an assert (accepted risk D-04) |
| VIZ-01 | Overlay renders inline aligned to source | **manual visual** | Human eyeballs the overlay |

### Sampling Rate
- **Per task commit:** the relevant unit/smoke check above (where automatable).
- **Phase gate:** full 1,544 mirror complete with outcome manifest (D-02) AND human visual PASS on all three controls (D-03) in `01_single_slice.ipynb`.

### Wave 0 Gaps
- [ ] `requirements.txt` — LRP pins verbatim + project additions (ENV-01)
- [ ] Vendor `third_party/dynamicLRP/src/lrp_engine/` @ SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` + a tracked `VENDOR_SHA` file (D-08, ENV-01)
- [ ] `tests/` + minimal `pytest` for the automatable rows above (DATA-03, ATTR-02, MODEL-01, DATA-01 idempotency) — framework: add `pytest` (not in LRP reqs; project addition, planner decides pin). The visual controls (ATTR-03/VIZ-01) are NOT automated by explicit user decision.
- [ ] `notebooks/01_single_slice.ipynb` — the visual gate surface

*The visual-eyeball gate (D-03/D-04) means automated coverage is intentionally partial. The planner MUST NOT add quantitative thresholds to the three controls (D-04 explicit prohibition). Automatable rows above are scaffolding/regression aids, not the phase gate.*

## Security Domain

`.planning/config.json` has no `security_enforcement` key → treat as enabled. This is a single-researcher exploratory pipeline on a private VM with no served surface, so most ASVS categories are N/A. Relevant ones:

| ASVS Category | Applies | Control |
|---------------|---------|---------|
| V5 Input Validation | yes | Validate every Rumsey download (HTTP 200 + image content-type + byte length + PIL-decodable) before writing to GCS; cap max image dimensions/bytes (decompression-bomb defense) |
| V6 Cryptography | no | No crypto; ADC handled by GCS SDK |
| V2/V3/V4 Auth/Session/Access | no | No users, no sessions, no API; GCS ADC is the only credential, verified fail-fast at job start |

| Threat | STRIDE | Mitigation |
|--------|--------|------------|
| Malformed/oversized third-party (Rumsey) image | Tampering / DoS | Content-type + decodability + size cap before mirror; quarantine failures in the outcome manifest |
| Decompression bomb (huge Rumsey scan exhausts VM RAM during ingest) | DoS | Validate `Content-Length` before full fetch; cap pixel dimensions on decode |
| Credentials assumed always present | Info / Availability | Verify GCS ADC at job start; fail fast with a clear message, not 1,000 items in |

## Sources

### Primary (HIGH confidence) — re-verified this session 2026-05-18
- `github.com/keeinlev/dynamicLRP` @ master: `requirements.txt` (verbatim), `src/lrp_engine/lrp.py` (`LRPEngine.__init__`/`run`/`get_model_operations` signatures), `src/lrp_engine/__init__.py`, `src/experiments/ViT.ipynb` (usage pattern), `src/lrp_engine/model_specific/mosaicbert.py`, `src/lrp_engine/promises/promise.py` + `softmax_backward_promise.py`, directory listings of `src/lrp_engine/` and `promises/`. Latest commit SHA `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` (2026-05-04) via GitHub API.
- `huggingface.co/google/siglip2-so400m-patch14-384`: `config.json` (vision: image_size 384, patch_size 14, 27 layers, hidden 1152, 16 heads; model_type "siglip"), `preprocessor_config.json` (`SiglipImageProcessor`, 384×384, rescale 1/255, mean/std 0.5).
- `huggingface.co/docs/transformers/v4.52.3` SigLIP & SigLIP2 model docs (forward returns `logits_per_image` shape `(image_bs, text_bs)`; `padding="max_length"`, `model_max_length=64`; attention-pooling not CLS).
- `metadata/rumsey_manifest.json` inspected directly: list of 1,544 dicts; keys incl. `id`, `image_url`, `richness_score`; idx0 score 18, idx1543 score 0 (non-monotonic).
- Repo `.gitignore` / `.python-version` / `README.md` / `ls` — vendor path `third_party/dynamicLRP/` not gitignored; Python 3.10.

### Secondary (MEDIUM confidence)
- `.planning/research/PITFALLS.md`, `SUMMARY.md`, `ARCHITECTURE.md`; `CLAUDE.md` Technology Stack section (prior HIGH-confidence project research — treated as authoritative per task instructions).

### Tertiary (LOW / ASSUMED)
- `google-cloud-storage`, `torchvision`, `Pillow`, JupyterLab version ranges — from CLAUDE.md prior research, not re-verified this session (Assumptions A1).

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — `requirements.txt` + SHA + checkpoint configs re-verified from source.
- `LRPEngine` API & integration pattern: HIGH — signatures + ViT.ipynb usage read from source.
- SigLIP-2 forward target & processor: HIGH — config/docs verified; checkpoint is fixed-res `siglip`.
- Patch-grid / 6-px / un-squash math: HIGH — derived from verified `patch_size=14`, `image_size=384`.
- GCS mirror idempotency pattern: HIGH — standard, fully prescriptive.
- **dynamicLRP *faithfulness* on SigLIP-2 as a contrastive MAP-pool encoder: MEDIUM-LOW** — unverifiable by research; this is exactly what the three controls test (the residual project risk, by design).

**Research date:** 2026-05-18
**Valid until:** ~2026-06-17 for the stack (dynamicLRP is a fast-moving pre-publication repo — re-check the vendored SHA before any deliberate re-vendor; the SHA pin itself does not expire).
</content>
</invoke>
