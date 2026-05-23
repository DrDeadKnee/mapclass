<!-- GSD:project-start source:PROJECT.md -->
## Project

**MapClass**

MapClass is exploratory research piping for pixel-level labeling of artistic map
images. It runs **dynamic LRP attribution** on a pre-trained vision-language model
(**SigLIP-2-so400m-patch14-384**) over historical maps from the **David Rumsey**
collection, producing per-query attribution heatmaps that a human eyeballs in
JupyterLab on a remote GCP VM. The deliverable is the *piping that lets the
researcher explore* — not a polished dataset or labeller (yet).

**Core Value:** A working, repeatable loop: pick a map + a text query → get a dynamic-LRP
attribution heatmap overlaid on that map → judge it visually — that scales to a
configurable sweep of maps × queries browsable in a notebook. If everything else
fails, this loop must work.

### Constraints

- **Tech stack**: Python, PyTorch-based VLM (SigLIP-2 via HuggingFace), JupyterLab for inspection
- **Dependency**: dynamic LRP behavior is bounded by what keeinlev/dynamicLRP supports and how cleanly it adapts to SigLIP-2's architecture
- **Infrastructure**: GCS bucket `gs://mapclass-training-northeast1` (data/ and models/); GCP VM with GPU; auth assumed present
- **Environment**: must run on a remote VM with results inspected through a Jupyter notebook (no local-first assumption)
- **Model size**: SigLIP-2-so400m is the chosen size; dynamic LRP fidelity/overhead degrades with larger models, so this is a deliberate ceiling
<!-- GSD:project-end -->

## Hardware (HARD CONSTRAINT — measured 2026-05-23)

This is the actual VM and the binding limit on every plan. **Treat the GPU VRAM
ceiling as a hard constraint: only ONE model is held in VRAM at a time.**

| Resource | Spec | Implication |
|----------|------|-------------|
| GPU | 1× **NVIDIA L4 — 23.66 GB total VRAM** (~22.5 GB usable at rest), driver 580.126.20 | The binding limit. The dynamic-LRP relevance pass retains the full forward-activation graph, so peak VRAM ≫ plain inference (ViT-b-16 already peaks **~19.3 GB**). Load + attribute **one** model, free it, then the next. |
| CUDA | build **12.6** (torch `2.7.1+cu126`, transformers `4.52.3`) | Matches the pinned stack — do not change. |
| CPU | 16 vCPU (Intel Xeon @ 2.20 GHz, 8c×2t) | Not a bottleneck. |
| RAM | 62 GB | Not a bottleneck; safe to offload a model to CPU between runs. |
| Disk | 96 GB total, **~24 GB free** | Tight. Each mirrored model dir competes for this — don't mirror models you won't run. |

**Hard rules that follow from the L4:**
- One model resident in VRAM at any moment. Free (`model.to("cpu")` → `del` → `gc.collect()` → `torch.cuda.empty_cache()`) before loading the next.
- A ~3B model (PaliGemma) may load on an empty GPU but its relevance pass can still OOM — that is a recorded result, not a bug to engineer around.
- The vendored dynamicLRP engine orphans retained graph on its failure path (VENDOR_SHA frozen, do not patch); never call a known-failing model in a tight loop.

<!-- GSD:stack-start source:research/STACK.md -->
## Technology Stack

## Executive Finding (read this first)
## Recommended Stack
### Core Technologies
| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| Python | 3.11 (3.10–3.12 OK) | Runtime | torch 2.7.1 + transformers 4.52.3 both support 3.10–3.12; 3.11 is the safe middle. Avoid 3.13 (wheel coverage for the pinned torch was incomplete at that version). |
| PyTorch (`torch`) | **2.7.1** (pinned by LRP repo) | DL framework; autograd graph that dynamicLRP traverses | Hard pin from `keeinlev/dynamicLRP/requirements.txt`. dynamicLRP runs *on the autograd graph*; a torch version mismatch risks autograd Node-name / graph-shape drift that breaks the engine. Do not float this. |
| `transformers` | **4.52.3** (pinned by LRP repo) | Loads SigLIP-2 (`Siglip2Model`/`AutoModel`, `AutoProcessor`) and provides `logits_per_image` | Pinned by LRP repo AND satisfies SigLIP-2's `>=4.49.0` requirement. Verified: SigLIP-2 landed in transformers v4.49.0-SigLIP-2 tag; 4.52.3 includes the `siglip2` model. |
| `keeinlev/dynamicLRP` | git clone @ `master` (commit pinned; pushed 2026-05-06) | The dynamic LRP engine (`LRPEngine`) | The whole project premise. No PyPI package, no `setup.py`/`pyproject.toml` — install by cloning and importing from `src/lrp_engine/`. |
| SigLIP-2 weights | `google/siglip2-so400m-patch14-384` | The pre-trained VLM under attribution | Fixed by PROJECT.md. Fixed-resolution variant → usable via `AutoModel`/`SiglipModel`; exposes `logits_per_image` directly (the query-driven attribution target). ~400M params — within the LRP fidelity ceiling stated in PROJECT.md. |
| `google-cloud-storage` | 3.x (latest 3.10.1; pin `>=3.0,<4`) | Mirror Rumsey images + SigLIP-2 weights to/from `gs://mapclass-training-northeast1` | Official GCP Python client; uses the VM's ADC credentials (already present per PROJECT.md). Idiomatic for blob up/download with no extra auth code. |
| JupyterLab | 4.x (latest 4.5.7; pin `>=4.4,<5`) | Human inspection surface (SSH-tunnelled) | Already the chosen inspection tool (README). 4.x is the current line and matches the README's `jupyter lab` invocation. |
### Supporting Libraries
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `einops` | 0.8.1 (LRP pin) | Tensor reshaping inside dynamicLRP | Required transitively by dynamicLRP. Install at its pin. |
| `timm` | 1.0.20 (LRP pin) | Vision backbones used by dynamicLRP experiments | Required by dynamicLRP `requirements.txt`. Install even if SigLIP-2 doesn't need it — the engine imports may assume the full requirements set. |
| `scikit-learn` | 1.7.0 (LRP pin) | Metrics in dynamicLRP experiment code | Transitive LRP dependency. Pin to avoid import-time surprises. |
| `omegaconf` | 2.3.0 (LRP pin) | Config objects in dynamicLRP | Transitive LRP dependency. |
| `tqdm` | 4.66.4 (LRP pin) | Progress bars (LRP + your sweep loop) | Reuse for the maps×queries sweep. |
| `captum` | unpinned by LRP (`captum`) → use `>=0.7,<0.9` | Baseline attributions inside dynamicLRP | LRP repo lists it unpinned; pin yourself to a recent stable to avoid resolver pulling something incompatible with torch 2.7.1. |
| `datasets` | unpinned by LRP (`datasets`) → use `>=2.19,<3` or `>=3,<4` | HF datasets in LRP experiments | LRP lists it unpinned. Not needed for the core loop (you load maps from GCS, not HF datasets) but it's an LRP import — pin conservatively. |
| `matplotlib` | 3.8.0 (LRP pin) | Heatmap rendering / overlay in the notebook | Pinned by LRP repo; also your primary visualization tool. The ViT notebook uses `plt.imshow` for the relevance map — overlay the map RGB with a semi-transparent `cmap` relevance map. |
| `seaborn` | 0.13.2 (LRP pin) | Optional nicer plotting | Transitive LRP pin; available for contact-sheet styling. |
| `Pillow` (PIL) | `>=10.3,<12` | Read/resize Rumsey JPEGs before the SigLIP processor | Standard image IO; `AutoProcessor` for SigLIP-2 consumes PIL images. Already a transitive dep of torchvision/transformers; pin explicitly for clarity. |
| `torchvision` | matched to torch 2.7.1 → `0.22.1` | Image tensor transforms (the ViT notebook uses `torchvision.transforms`) | Must match torch exactly (torchvision 0.22.x ↔ torch 2.7.x). A mismatch here is a classic silent breakage. |
| `numpy` | `<2.3`, let torch/transformers resolve (likely 1.26.x or 2.x) | Array glue between relevance tensors and matplotlib | Do not hard-pin blindly; let the torch 2.7.1 wheel constrain it. NumPy 2.x is fine with torch 2.7.1. |
| `accelerate` | `>=0.30,<2` | Convenience for `device_map`/dtype when loading SigLIP-2 | Optional but smooths `from_pretrained(..., device_map="auto")`. Not strictly required for a single-GPU VM. |
| `ipywidgets` | `>=8.1,<9` | Interactive contact-sheet browsing in JupyterLab | For the v1 "browse all attribution maps" requirement; works with JupyterLab 4.x out of the box. |
### Development Tools
| Tool | Purpose | Notes |
|------|---------|-------|
| `venv` + `pip` | Environment isolation | README already prescribes `python3 -m venv .venv` + `requirements.txt`. Keep this; do not introduce conda/poetry/uv mid-stream — the LRP repo assumes plain pip. |
| `ipykernel` | Register the venv kernel for JupyterLab | README already does `ipykernel install --name mapclass`. Keep. |
| `gcloud` CLI / ADC | GCS auth on the VM | Already present per PROJECT.md; `google-cloud-storage` picks up ADC automatically. No service-account JSON juggling needed. |
| Git submodule **or** vendored clone of dynamicLRP | Pin the LRP engine reproducibly | Recommended: add dynamicLRP as a git submodule pinned to a specific commit, or vendor `src/lrp_engine/` with the source commit recorded. `pip install git+...` will NOT work (no packaging metadata). |
## Installation
# 0. System: Python 3.11, NVIDIA driver + CUDA matching torch 2.7.1 cu wheels (cu126/cu128)
# 1. Clone the load-bearing dependency (NO pip package exists)
# pin it: cd external/dynamicLRP && git checkout <commit-from-2026-05-06> && cd -
# Engine import path: external/dynamicLRP/src/lrp_engine  (sys.path.append in notebook)
# 2. Core stack — MIRROR dynamicLRP/requirements.txt pins exactly, then add ours
# 3. Project-specific additions
# 4. Kernel registration (per README)
## SigLIP-2 + dynamicLRP integration sketch (verified pattern from repo's ViT.ipynb)
# param_rels[pixel_values] -> reshape to patch grid (384/14 ≈ 27x27) -> upsample
# -> matplotlib overlay on the original map (alpha-blended cmap)
## Alternatives Considered
| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `keeinlev/dynamicLRP` | LXT / LRP-eXplains-Transformers (Achtibat et al., AttnLRP) | If dynamicLRP cannot cover SigLIP-2's ops. LXT is *vendored inside dynamicLRP's repo* (`external/LRP-eXplains-Transformers`) and is the academic predecessor — a real fallback, but it requires model-specific patches (it ships `vit_torch.py`, `bert.py`, etc.) which is exactly the work dynamicLRP avoids. Only fall back if op coverage fails. |
| `keeinlev/dynamicLRP` | Zennit | Zennit is mature and well-documented but is *layer/module*-level (needs canonizers per architecture). SigLIP-2 has no off-the-shelf Zennit canonizer; you'd write one. dynamicLRP exists specifically to avoid this. |
| `keeinlev/dynamicLRP` | `captum` (LayerLRP / GradientShap / IG) | If LRP fidelity is poor and you only need *a* heatmap. captum is already a transitive dep. Integrated Gradients is trivial to run on `logits_per_image`. Good cheap sanity-check baseline, not the project goal. |
| transformers 4.52.3 (pinned) | transformers 5.x (latest 5.8.1) | Never for this project. 5.x has breaking API changes and would diverge from dynamicLRP's pin. |
| torch 2.7.1 (pinned) | torch 2.12 (latest) | Never unless dynamicLRP is re-tested against it. Engine traverses autograd internals; newer torch can change Node names/graph structure. |
| `google-cloud-storage` SDK | `gcsfs` + fsspec | If you want pandas/`open()`-style path transparency. Fine, but adds an abstraction layer; the SDK's explicit `blob.upload_from_filename`/`download_to_filename` is clearer for a one-time mirror script. |
| `gsutil`/`gcloud storage` CLI from notebook | — | Acceptable and simple for the *one-time* image/weight mirror step. Use the Python SDK for anything programmatic in the loop. |
## What NOT to Use
| Avoid | Why | Use Instead |
|-------|-----|-------------|
| `transformers>=5.0` | Breaking API changes; diverges from dynamicLRP's `==4.52.3` pin; SigLIP-2 loading code may differ | `transformers==4.52.3` |
| `torch` ≠ 2.7.1 | dynamicLRP runs *on the autograd graph*; Node-name/graph drift across torch versions can silently break LRP propagation | `torch==2.7.1` (+ `torchvision==0.22.1`) |
| `pip install git+https://github.com/keeinlev/dynamicLRP` | Repo has **no** `setup.py`/`pyproject.toml`/packaging metadata — pip install will fail or install nothing usable | `git clone` + `sys.path` / submodule |
| Floating ("unpinned") `datasets`/`captum` | LRP repo leaves these unpinned; a fresh resolver may pull versions incompatible with torch 2.7.1 | Pin to conservative ranges (see table) |
| conda / poetry / uv lock as the primary env | dynamicLRP assumes plain `pip install -r requirements.txt`; README prescribes `venv`+pip; mixing resolvers invites the torch/CUDA wheel mismatch class of bug | `venv` + `pip` (per README) |
| SigLIP-2 *NaFlex* variant via `SiglipModel` | NaFlex variants need `Siglip2Model`; mismatched class → load/shape errors | `so400m-patch14-384` is *fixed-resolution* → `AutoModel`/`SiglipModel` is correct here |
| `model.get_image_features()` as the attribution target | Returns pooled embeddings, not the query-conditioned similarity — attribution would not be query-driven | Attribute through `output.logits_per_image` (image–text similarity) for query-driven heatmaps |
| Fine-tuning / training libs (DeepSpeed, PEFT, TRL) | Explicitly out of scope (PROJECT.md: the abandoned approach this restart replaces) | Inference-only; nothing to add |
| TensorFlow / JAX SigLIP ports | dynamicLRP is PyTorch-only (operates on torch autograd) | PyTorch `transformers` SigLIP-2 |
## Stack Patterns by Variant
- First, inspect `src/lrp_engine/promises/` and `lrp_prop_fcns.py` — add a promise/rule for the missing op (the repo is *designed* for this; that's its thesis).
- If the gap is the attention-pooling head or sigmoid loss head specifically, try `use_attn_lrp=True` / attribute on raw `image_embeds @ text_embeds` logits before the pooling head.
- Only if both fail: fall back to the vendored LXT (`external/LRP-eXplains-Transformers`) with a ViT-style patch, or captum Integrated Gradients as a degraded baseline.
- Start with `use_gamma=False`, `relevance_filter` < 1.0 (e.g. 0.5) to cut memory/noise — both are documented `LRPEngine` knobs.
- so400m at fp32 on a single mid-tier GPU should fit; if not, load model in bf16 for the forward pass but note `LRPEngine` defaults to `dtype=torch.float32` for the relevance pass (keep it).
- Use `gcloud storage cp` / `gsutil -m` for the bulk image mirror (parallel), reserve the Python SDK for the in-loop "load map by ID from GCS" reader.
## Version Compatibility
| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `torch==2.7.1` | `torchvision==0.22.1` | Must match minor line exactly. torchvision 0.22.x ↔ torch 2.7.x. A mismatch is a classic silent/segfault failure. |
| `torch==2.7.1` | Python 3.10–3.12 | 3.11 recommended. Avoid 3.13. |
| `transformers==4.52.3` | SigLIP-2 (`siglip2`) | Verified: SigLIP-2 added at transformers `v4.49.0-SigLIP-2`; 4.52.3 includes the model. ✅ No conflict with the LRP pin. |
| `transformers==4.52.3` | `torch==2.7.1` | Both in dynamicLRP's own `requirements.txt` — tested together by the LRP authors. |
| `dynamicLRP @ master` | `torch==2.7.1`, `transformers==4.52.3` | The repo's `requirements.txt` is the source of truth; mirror it verbatim. |
| `google-cloud-storage 3.x` | Python 3.11, ADC | Independent of the torch/transformers stack; no shared transitive conflicts observed. |
| `numpy` | `torch==2.7.1`, `matplotlib==3.8.0` | Let torch wheel constrain numpy (2.x OK with torch 2.7.1). Do not hard-pin numpy 1.x — matplotlib 3.8 + torch 2.7.1 both support numpy 2. |
## Confidence Assessment
| Claim | Confidence | Basis |
|-------|------------|-------|
| dynamicLRP is PyTorch, op-level, model-agnostic, no per-arch code | HIGH | README + arXiv abstract + read `src/lrp_engine/lrp.py` source directly |
| dynamicLRP has no pip packaging (clone-only) | HIGH | Inspected full repo file tree via GitHub API — no `setup.py`/`pyproject.toml` |
| dynamicLRP `requirements.txt` pins (torch 2.7.1, transformers 4.52.3, etc.) | HIGH | Fetched the actual `requirements.txt` from `master` |
| transformers 4.52.3 supports SigLIP-2 (no conflict with LRP pin) | HIGH | SigLIP-2 needs ≥4.49.0 (HF release notes); 4.52.3 > 4.49.0 and ships `siglip2` |
| Usage pattern (`LRPEngine.run(logits)`, `params_to_interpret=[input]`) | HIGH | Read `src/experiments/ViT.ipynb` cells + `LRPEngine.__init__`/`run` signatures |
| `logits_per_image` is the right query-driven attribution target | HIGH | SigLIP2 docs/source confirm `logits_per_image` = image–text similarity scores |
| SigLIP-2 so400m-patch14-384 works via `AutoModel` (fixed-res, not NaFlex) | HIGH | HF docs: fixed-resolution variants usable with `SiglipModel`/`AutoModel` |
| **Whether dynamicLRP covers 100% of SigLIP-2's specific ops out of the box** | **MEDIUM-LOW** | Could NOT verify — repo has no SigLIP/CLIP example; paper lists ViT/VGG/RoBERTa/T5/Mamba/Whisper/DePlot but not SigLIP. Engine is *designed* to extend, but a custom promise/rule for SigLIP's pooling/sigmoid head may be needed. **This is the flagged risk.** |
| PyPI latest versions (gcs 3.10.1, jupyterlab 4.5.7, torch latest 2.12) | HIGH | Queried pypi.org JSON API on 2026-05-18 |
## Sources
- `github.com/keeinlev/dynamicLRP` — repo metadata, full file tree, `README.md`, `requirements.txt`, `src/lrp_engine/__init__.py`, `src/lrp_engine/lrp.py` (LRPEngine API), `src/experiments/ViT.ipynb` (usage + viz pattern) — fetched directly 2026-05-18 — HIGH
- arXiv 2512.07010 abstract ("Dynamic LRP") — model-agnostic claim, 99.92% node coverage / 15 architectures, 47 tensor ops, Promise System — HIGH
- huggingface.co/google/siglip2-so400m-patch14-384 + HF transformers SigLIP2 docs + transformers `v4.49.0-SigLIP-2` release notes — load class, `logits_per_image`, NaFlex vs fixed-res, min transformers version — HIGH
- pypi.org JSON API (torch, transformers, google-cloud-storage, jupyterlab) — latest versions + pin existence checks, 2026-05-18 — HIGH
- `.planning/PROJECT.md`, `README.md`, `metadata/rumsey_manifest.json` — project constraints, GCS bucket, model/dataset shape — HIGH
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->
## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
