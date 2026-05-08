# Stack Research

**Domain:** Multi-source-trained dense-prediction VL+OCR+heads model (model-handoff package, brownfield mid-refactor)
**Researched:** 2026-05-08
**Confidence:** HIGH for backbone choice, training framework, config/logging, checkpoint format. MEDIUM for specific OCR module starting point (active research area, no clear 2025 winner for *map* text specifically). MEDIUM-HIGH for PaliGemma 2 swap (architecture verified loadable; segmentation feature surface trivially exposes vision encoder hidden states via HF transformers).

## Scope and framing

This document answers three sub-questions tied to the locked v1 scope in `PROJECT.md`:

1. **MODEL-01 ship backbone** — small VL with public weights, fits 4–8 GB GPU + CPU envelope, exposes dense vision features for two seg heads to attach to.
2. **MODEL-04 benchmark backbone** — PaliGemma swap target; same stack with the model id substituted; bellwether NLL upper bound (`EVAL-03`).
3. **Surrounding stack** — training framework, checkpoint format, config system, logging, OCR module, training entry-point design.

It does **not** redecide already-locked stack items: `Pillow + rasterio + pyproj + requests + numpy` for the data pipelines stays as-is (per `.planning/codebase/STACK.md`). The new additions are model-side and training-side only.

It is aligned with `.planning/research/ARCHITECTURE.md`'s **Backbone interface as a swap surface** (Pattern 3) — the chosen backbone implements `Backbone.extract_features(image) -> dict[str, Tensor]`, returning per-stage feature maps consumed by the two seg heads and (optionally) the OCR head. Both MODEL-01 and MODEL-04 must satisfy this contract; neither choice changes the surrounding training, eval, or inference plumbing.

## Recommended Stack

### Core Technologies

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| **Python** | 3.10+ (3.12 verified locally) | Project language | Already pinned by existing PEP 604/585 type-hint usage in tree; no reason to change. |
| **PyTorch** | `torch>=2.4,<3.0` (2.5 if CUDA toolchain on the VM is 12.4+) | Tensor / autograd / model serialization | Already in `requirements.txt`; transformers 4.47+ requires `torch>=2.4`. PyTorch 2.5 added cuDNN SDPA backend speedups for SDPA-heavy ViTs (which both candidates are). |
| **Hugging Face transformers** | `>=4.47,<5.0` | Backbone loader, processor, autoclass surface | 4.47 is the version that introduced `PaliGemma 2` support (release notes). Existing repo pin is `>=4.41`; bump to `>=4.47` lets MODEL-04 load `paligemma2-3b-pt-224` cleanly. transformers 5.x exists but has breaking changes on `pipeline` and processor signatures — pin below 5.0 to defer that migration. |
| **safetensors** | `>=0.4.5` | Checkpoint format | Comes transitively with `transformers`; trainer must explicitly call `safetensors.torch.save_file(state_dict, path, metadata={...})` to embed `model_version`, `dataset_manifest_sha`, `taxonomy_hash`, `training_seed`, `source_class_weights_hash` per `FEATURES.md` TS-8. **Drop pickle .bin entirely** — it executes arbitrary code on load, which would propagate into the hex-grid app. |
| **HuggingFace accelerate** | `>=0.34` | Mixed-precision + device placement + `accelerator.prepare()` | Already in `requirements.txt`. The right default for this codebase: keeps the training loop hand-written (research-grade), abstracts only the multi-GPU / fp16 boilerplate. **Not Lightning** — see "What NOT to Use" below. |
| **PEFT** | `>=0.11` | LoRA adapters on the backbone (esp. PaliGemma 2 — full fine-tune is expensive) | Already in `requirements.txt`. For MODEL-01 small backbone, PEFT is optional (full fine-tune of a 500M / 2B model on a 16 GB+ training GPU is fine). For MODEL-04 PaliGemma 2 it's the realistic path (full FT of 3B model needs >32 GB). |
| **bitsandbytes** | `>=0.43` | INT8 / NF4 quantisation for inference budget headroom | Already in `requirements.txt`. Used for two purposes: (a) INT4 PaliGemma 2 *training* on a smaller VM (NF4 + LoRA = ~9 GB VRAM for a 3B model — see PaliGemma 2 fine-tune walkthroughs), (b) post-ship INT8 dynamic quant on the small-backbone ship checkpoint to extend the 4–8 GB GPU envelope. |
| **OmegaConf + dataclass-typed configs** | `omegaconf>=2.3` | Training config system | Lighter than full Hydra. Config files live at `mapclass/configs/{v0,v1}.yaml` and `mapclass/configs/loss_weights.yaml` (per ARCHITECTURE.md). Typed via `@dataclass` Python configs that OmegaConf merges with the YAML — gives runtime validation without Hydra's launcher / multirun framing, which this single-developer codebase doesn't need. |
| **TensorBoard** | `tensorboard>=2.18` | Training-time logging | TS-7 in FEATURES.md. Local files in `runs/`, no network, no account setup. Single-developer research codebase doesn't need wandb's collaboration features; tensorboard's local-files model also works seamlessly with the cloud-VM SSH workflow (rsync `runs/` back to the laptop, run `tensorboard --logdir runs/`). |

### Backbone — MODEL-01 (ship target)

| Backbone (HF model ID) | Params | License | FP16 weights | INT4 (NF4) | CPU throughput (single 1024×1024 input, est.) | Decision |
|-------------------------|--------|---------|--------------|-----------|------------------------------------------------|----------|
| **`HuggingFaceTB/SmolVLM-500M-Instruct`** | ~500M (93M SigLIP-B/16 vision encoder + ~360M SmolLM2 text decoder) | Apache-2.0 | ~1 GB | ~250 MB | ~2–4 s on a modern laptop CPU for the vision encoder pass (text-decoder unused for our use case, see below) | **PRIMARY** |
| `vikhyatk/moondream2` (`moondream/moondream-2b-2025-04-14`) | ~1.86B (SigLIP vision encoder + Phi-1.5 text decoder) | Apache-2.0 | ~3.7 GB | ~950 MB | ~5–8 s CPU; ~150–300 ms on a 4–8 GB consumer GPU | **FALLBACK** |
| `HuggingFaceTB/SmolVLM-256M-Instruct` | ~256M | Apache-2.0 | ~512 MB | ~130 MB | ~1–2 s CPU | Reserved as floor option if 500M proves too slow on CPU |

**Picking criteria (PRIMARY = SmolVLM-500M-Instruct):**

- **Inference budget — comfortably inside.** `transformers` reports ~1.23 GB FP16; with two seg heads (~30–50 MB each) plus image-tile activations during inference, total fits with multiple GB to spare on a 4–8 GB GPU and is the only candidate that genuinely runs on CPU at usable throughput.
- **License — Apache-2.0** (no carve-outs, no Gemma-style "prohibited uses" rider). Trained weights from this base may be redistributed under the same terms or any compatible license, satisfying `PROJECT.md` SHIP-01 ("Inference Python module + trained checkpoint…packaged for the separate hex-grid app").
- **Vision encoder is SigLIP-B/16 (93M)** — the same family of encoder that the existing ARCHITECTURE.md research already cited as the right shape for dense-prediction features (SigLIP 2 paper: "self-supervised + decoder losses → better dense features for segmentation and depth estimation"). Hidden-state outputs from the SigLIP encoder are accessible through the standard `model.vision_tower(image, output_hidden_states=True)` HF transformers pattern. The `Backbone.extract_features()` swap-surface implementation is straightforward.
- **Pure-architecture compatibility with the seg-head attach plan.** The vision encoder is a stand-alone ViT; we don't need the text decoder for MODEL-01 at all (the OCR head is *separate* — it's its own module, see below). We can load the full SmolVLM checkpoint, pull the SigLIP encoder out via `model.model.vision_model`, freeze the rest, and attach the seg heads. This is a clean 50-line `Backbone` subclass.
- **HF transformers compatibility — first-class.** Supported via `AutoProcessor` + `AutoModelForVision2Seq` (or `Idefics3ForConditionalGeneration` since SmolVLM is the Idefics3 architecture under the hood). No `trust_remote_code=True` needed.

**Why Moondream2 is the fallback, not the primary:**

- Slightly higher quality on detection-flavoured tasks (Moondream is *designed* for detection) might matter once seg heads start training. If the quality gap shows up in `EVAL-01`, swap.
- BUT: Moondream2 requires `trust_remote_code=True` to load (custom modeling code in the repo, not in transformers proper). For a model that ships in a downstream consumer's package, this is a meaningful operational tax — the hex-grid app would need to allow remote-code execution on import.
- Also: Moondream2's SigLIP encoder is larger and the wrapper modeling code is more entangled with the text decoder than SmolVLM's. Pulling out a clean vision-tower handle is harder.

**Why SmolVLM-256M is reserved, not primary:**

- 256M may be too small for the dense-prediction quality target. The vision encoder (93M SigLIP-B) is identical between 256M and 500M variants — only the text decoder shrinks. So for our use case (we only consume vision features), 500M and 256M are *equivalent in vision quality* and the only reason to use 256M is if the FP16 wrapper-load cost actually matters. It does not, on the inference budget. Stick with 500M as primary.

**Why Florence-2 was considered and rejected:**

- Florence-2-base (~230M) and -large (~770M) use a DaViT vision encoder (not SigLIP) and an encoder-decoder transformer for output. License is MIT (excellent), HF integration via `trust_remote_code=True` (acceptable but worse than SmolVLM's no-trust-needed path).
- The fatal issue: Florence-2 is built around prompted output for specific task tokens (`<OD>`, `<CAPTION>`, `<DENSE_REGION_CAPTION>`, etc.). Adapting it as a feature extractor for two new dense seg heads requires either bypassing the encoder-decoder entirely (defeats the point of using Florence) or training new task tokens (significant new infrastructure). Cleaner to use a model whose vision encoder is a plain ViT.
- Florence-2 stays a credible *alternative* if SmolVLM-500M dense-feature quality is poor — but that's a v1.x reconsideration, not a v1 recommendation.

**Why Idefics3-8B was considered and rejected:**

- Even at 4-bit (NF4), an 8B model is ~4.5 GB just for the language model; with vision-tower activations and two seg heads the inference graph balloons past the 4–8 GB GPU envelope's lower half. Not a CPU candidate at all.
- It's a fine *training-time* upgrade path — but PaliGemma 2 already plays that role as MODEL-04 with stronger model-card and more transformers integration polish.

**Why nanoVLM / MobileVLM / LLaVA-1.5-7B-quantized were rejected:**

- nanoVLM is a pedagogical/reference repo, not a maintained checkpoint — no production-grade weights to ship.
- MobileVLM is on the older CLIP family (not SigLIP) and predates the dense-feature improvements that SigLIP 2 / SmolVLM bring. Its model cards have not been updated since early 2024.
- LLaVA-1.5-7B even quantized is 4–5 GB at INT4; same problem as Idefics3 with less of the upside.

### Backbone — MODEL-04 (benchmark variant; bellwether for `EVAL-03`)

| Backbone (HF model ID) | Params | License | FP16 weights | NF4 (LoRA) | Decision |
|-------------------------|--------|---------|--------------|------------|----------|
| **`google/paligemma2-3b-pt-224`** | ~3B (SigLIP-So400m vision + Gemma-2-2B text) | Gemma Terms of Use (commercial OK, with prohibited-use rider) | ~6 GB | ~9 GB total during training (NF4 weights + LoRA adapters + activations) | **PRIMARY MODEL-04 CHECKPOINT** |
| `google/paligemma2-3b-mix-224` | ~3B | Gemma | ~6 GB | n/a (mix is for direct task use) | Use only if we want a stronger initialization (already partially fine-tuned on segmentation) |
| `google/paligemma-3b-pt-224` (v1, original) | ~3B | Gemma | ~6 GB | similar | **Do NOT use as MODEL-04** — superseded by PaliGemma 2; keep only as a paper appendix if v1 vs v2 comparison is interesting |

**Use `google/paligemma2-3b-pt-224`** — the PaliGemma 2 (not 1) pre-trained checkpoint at 224×224 resolution.

**Why PaliGemma 2, not PaliGemma 1:**
- PaliGemma 2 is the December 2024 release; uses Gemma 2 as the language decoder (improved architecture, longer context, better tokenizer). Available in 3B / 10B / 28B; we want the 3B for the bellwether-not-ship role.
- PaliGemma 1 (`google/paligemma-3b-*`) is still loadable but is the older Gemma decoder. No reason to use the older variant when the newer one is a drop-in replacement at the same parameter count.

**Why pt-224, not mix-224 or pt-448:**
- **pt-224 is the right starting point for fine-tuning** — it's the unfine-tuned base. We are doing our own fine-tune on dense seg heads, so we want the pretraining-only weights. The HF docs are explicit: "PT checkpoints are pre-trained and intended for further fine-tuning. Mix checkpoints are fine-tuned on a diverse set of tasks and are ready to use out of the box."
- **pt-224 fits training budget for benchmark.** 224 input × patch16 = 14×14 vision tokens + multi-tile if needed. Larger resolution (448, 896) doubles or quadruples activation memory and we only need a benchmark, not SOTA — keeping it at 224 lets MODEL-04 train on a 24 GB GPU with NF4+LoRA.
- mix-224 would *reduce* the benchmark's value: fine-tuning from a model already fine-tuned on segmentation conflates "Gemma-as-backbone gives X NLL" with "the mix fine-tune transferred something". For a clean bellwether, we want the same hyperparameters and data path as MODEL-01, just with the backbone class swapped — and that means starting from pt.

**License flag (loud):**
- PaliGemma 2 ships under the **Gemma Terms of Use**, not Apache-2.0. This is *not* OSI-approved. It permits commercial use and redistribution, **but** carries prohibited-use riders (no use to "harm minors", "facilitate critical-infrastructure attacks", etc.) and an indemnification clause.
- **Practical consequence for `SHIP-01`:** PaliGemma 2 fine-tuned weights inherit the Gemma Terms. We are explicitly *not shipping* the PaliGemma checkpoint as the inference backbone — `PROJECT.md` puts it in v1 as MODEL-04 benchmark only. The Gemma license still applies to the *training artifact* even if it's only a research-internal benchmark, but it does not contaminate the SmolVLM-based ship checkpoint (independent base, independent license). **Document this split in the eventual model card.**
- **Action item for the roadmap:** when MODEL-04 trains, the resulting `models/geovilm_paligemma_v0.pt` and `_v1.pt` checkpoints should NOT be uploaded to HuggingFace alongside the small-backbone ship checkpoint — keep the PaliGemma artifact in `models/benchmark/` clearly separated and apply the Gemma license verbatim if it ever leaves the lab. The shipped MODEL-01 checkpoint can be Apache-2.0 (or any compatible license you choose).

**Backbone-interface compatibility:**
- PaliGemma 2 in transformers exposes the SigLIP-So400m vision tower at `model.vision_tower` (just like PaliGemma 1). The `Backbone.extract_features()` implementation is parallel to the SmolVLM one — load the model, freeze most of it, hook the vision-tower output. Standard pattern, well-documented, has been done in dozens of public fine-tune walkthroughs for object detection and segmentation.
- ARCHITECTURE.md's swap surface (Pattern 3) is satisfied without modification. The seg heads written for MODEL-01 reuse the exact same forward signature for MODEL-04 — only the `feature_channels` config and the input image size (224 vs SmolVLM's 384) change.

**ARCHITECTURE.md alignment note:** The architecture document was written before MODEL-04 was added to v1 scope. It does not change the backbone-interface pattern, the loss-weights pattern, the bootstrap-loop pattern, the data-flow diagram, or the entry-point contract. **The only thing that gains a new sub-bullet is the build order**: MODEL-01 and MODEL-04 are *parallel training runs* on the same dataset, sharing all infrastructure. The `Backbone` ABC just gets a second concrete subclass (`PaliGemma2Backbone`) alongside `SmolVLMBackbone`. No swap-surface change.

### OCR module — MODEL-02

| Library / model | Version | Approach | Curved/rotated map text? | Decision |
|------------------|---------|----------|---------------------------|----------|
| **`docTR` (mindee/doctr)** | `>=1.0.1` | Two-stage detection + recognition; default detection `fast_base`, recognition `parseq` or `vitstr` | Yes — explicit support for skew/rotation; ParSeq recognition handles curved layouts | **PRIMARY OCR STARTING POINT** |
| `easyocr` | latest | Two-stage CRAFT + CRNN | Partial — CRAFT detection is rotation-tolerant; CRNN recognition is rectilinear-biased | Fallback if docTR fine-tuning blocks |
| Original CRAFT (`clovaai/CRAFT-pytorch`) | unmaintained since 2020 | Character-region detection + affinity | Yes for detection only; no recognition stage | **Use only as detection sub-component** if docTR detection plateaus |
| ABCNet (`Yuliang-Liu/bezier_curve_text_spotting`) | unmaintained since 2022 | Bezier-curve aligned spotting | Designed for curved text | Not actively maintained; ABCNet v2 lives in MMOCR which adds a heavy dependency tree. **Skip unless docTR fails.** |
| DBNet++ | available in MMOCR / standalone | Differentiable binarization for arbitrary shapes | Yes | Strong alternative if recognition isn't needed (just bounding polygons). |

**Pick docTR (mindee).** Reasoning:

1. **Active maintenance.** Released v1.0.1 in 2025 (transitioned to a 1.x series, which is a strong signal for production-readiness). Releases 0.10.0 and 1.0.0 in 2024-2025 added newer recognition heads (ViTSTR, ParSeq) in PyTorch.
2. **End-to-end, not just detection.** Unlike CRAFT (detection only) and DBNet (detection only), docTR ships a complete pipeline. For our purpose (read place names off rotated map text and inject them as auxiliary supervision into the GeoViLM training loss), we need both detection bboxes AND recognition strings.
3. **PyTorch backend native.** docTR supports both TF and PyTorch; we use the PyTorch path so it lives next to `torch` and `transformers` without adding a TF dep.
4. **Apache-2.0 license** — clean for `SHIP-01`.
5. **Curved/rotated text is a first-class concern in their docs.** The "How can I detect rotated images/documents?" discussion is on the project's GitHub; the page-orientation classifier in their pipeline is configurable; the ParSeq recognition head is robust to non-rectilinear text.

**LOW-confidence flag — what could still go wrong:**

- docTR is trained on *document* OCR, not *map* OCR. Map text is curved-along-paths (rivers labeled in a wavy line, country names arching across a region) in ways that document-OCR training data does not contain. The recognition head will likely need fine-tuning, which means we need a *map-text training set*. The existing repo has nothing on disk for this (no map-text label data, no annotation tooling).
- **Honest assessment:** there is no off-the-shelf 2025 SOTA model that solves "rotated, curved, often-stylised text on stylized fantasy and historical maps." This is a genuine gap.
- **Mitigation path for v1:** start by *running docTR pretrained as-is* on the map images. Use whatever recognition strings come back (with a confidence threshold) as auxiliary text labels for the OCR auxiliary loss. The ARCHITECTURE.md OCR head is described as auxiliary and conditional — `λ * ocr_loss` "scheduled / off if labels absent" (data flow section). So a noisy or partial OCR signal is fine for v1; the seg heads carry the primary supervision.
- **Defer to phase-specific research:** if docTR-pretrained fails badly on the synthetic + historical training images, the roadmap can spawn a phase-specific OCR research block to evaluate map-text fine-tuning datasets (DocBank, MapText competition data, synthetic curved-text generation).

### Supporting Libraries

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `omegaconf` | `>=2.3` | YAML config loading + dataclass validation | All training, eval, infer configs |
| `tensorboard` | `>=2.18` | Loss curves + sample-prediction images per source per N steps | Any training run; no setup beyond `--logdir` |
| `python-doctr[torch]` | `>=1.0.1` | OCR module starting point | MODEL-02 implementation |
| `scikit-image` | `>=0.24` | Phase-correlation, image registration helpers (auto-georef tool component, per ARCHITECTURE.md) | GEOREF-01 / GEOREF-02 only |
| `torchmetrics` | `>=1.4` | Calibration / ECE computation for the eval script (FEATURES.md TS-6) | Eval entry point only |
| `tqdm` | `>=4.66` | Progress bars in training loop and dataset iterators | All long-running scripts |
| `pyyaml` | `>=6.0.2` | YAML reading (transitive via omegaconf, but pin for safety) | Config files |
| `opencv-python-headless` | `>=4.10` | Image rotation / TPS warp helpers (auto-georef tool) | GEOREF-01 only — headless flavour to avoid GUI deps on the cloud VM |

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `pyproject.toml` | Make `mapclass/` `pip install -e .`-able for `python -m mapclass.train` to work on the cloud VM | New file; **required** per ARCHITECTURE.md "Cloud-VM SSH workflow contract". Build backend: `setuptools` (smallest viable choice; this is not a publishable PyPI package, so hatchling/poetry are overkill). |
| `uv` (recommended) | Fast resolver + lockfile generation | Per FEATURES.md D-10 / CONCERNS.md 8c — `uv pip compile requirements.in -o requirements.lock.txt`. Doesn't have to be the install tool on the VM (can keep using `pip install -r requirements.lock.txt`); just the lockfile generator. |
| `.python-version` | Pin Python to 3.10+ | One-line file; per CONCERNS.md 8c. |

## Installation

```bash
# Pin first
echo "3.10" > .python-version

# Generate lockfile from requirements.in
uv pip compile requirements.in -o requirements.lock.txt

# Install (on the cloud VM, after cloning)
pip install -r requirements.lock.txt
pip install -e .   # makes mapclass/ importable as a package

# Verify the candidate backbones load
python -c "from transformers import AutoProcessor, AutoModelForVision2Seq; \
           AutoProcessor.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct'); \
           AutoModelForVision2Seq.from_pretrained('HuggingFaceTB/SmolVLM-500M-Instruct', torch_dtype='auto')"

python -c "from transformers import AutoProcessor, PaliGemmaForConditionalGeneration; \
           AutoProcessor.from_pretrained('google/paligemma2-3b-pt-224'); \
           PaliGemmaForConditionalGeneration.from_pretrained('google/paligemma2-3b-pt-224', torch_dtype='auto')"
```

**`requirements.in` (canonical source for the lockfile):**

```text
# Core ML
torch>=2.4,<3.0
transformers>=4.47,<5.0
accelerate>=0.34
peft>=0.11
bitsandbytes>=0.43
safetensors>=0.4.5

# Existing data pipeline
Pillow
numpy
rasterio
pyproj
requests

# New for v1
python-doctr[torch]>=1.0.1
omegaconf>=2.3
tensorboard>=2.18
torchmetrics>=1.4
scikit-image>=0.24
opencv-python-headless>=4.10
tqdm>=4.66
```

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| SmolVLM-500M-Instruct (MODEL-01) | Moondream2 (`vikhyatk/moondream2`) | If `EVAL-01` shows SmolVLM dense-feature quality plateaus; the cost is `trust_remote_code=True` which complicates downstream packaging |
| SmolVLM-500M-Instruct | Florence-2-base (`microsoft/Florence-2-base`) | If a DaViT encoder unexpectedly outperforms the SigLIP-B/16 vision tower on map imagery — unlikely but worth a one-off comparison if EVAL-01 disappoints. Cost: heavier integration to bypass the encoder-decoder; license is MIT (better than Gemma) |
| SmolVLM-500M-Instruct | DINOv2-S/14 (`facebook/dinov2-small`) frozen + own seg heads | If we decide we don't actually need any text/OCR-flavoured features from a VLM and want a purer dense-prediction backbone. DINOv2 is the cleanest dense-feature ViT in the open ecosystem and was the original architectural-references.md preference. **Worth keeping in mind as a sanity baseline alongside MODEL-01.** |
| PaliGemma 2 pt-224 (MODEL-04) | PaliGemma 2 mix-224 | Only if you want a stronger init at the cost of a less-clean bellwether comparison |
| PaliGemma 2 pt-224 | PaliGemma 2 pt-448 | If MODEL-04 results suggest 224 input resolution is the bottleneck (unlikely for our use case where the seg-head upsamples to image resolution) — note the VRAM/compute cost is 4× |
| accelerate | PyTorch Lightning | If multi-node distributed training becomes important and the team grows; for a single-developer research codebase Lightning is over-abstraction |
| accelerate | bare PyTorch with hand-rolled AMP | If accelerate causes issues on the specific cloud-VM CUDA/driver combo. Honestly fine — the loss in productivity is small |
| OmegaConf | Full Hydra (`hydra-core`) | If multirun hyperparameter sweeps become valuable. We don't need them in v1. |
| OmegaConf | Plain `argparse` + dataclass | What the existing codebase does. Acceptable for `scripts/` entry points (per CONVENTIONS.md), but for the training loop we need typed nested configs that argparse handles poorly. OmegaConf is the lightest upgrade path. |
| TensorBoard | Weights & Biases (wandb) | If a collaborator joins and centralised tracking matters. Currently a single-developer repo with no team — local files are simpler |
| docTR | EasyOCR | If docTR fine-tuning fails on map text and EasyOCR's CRAFT detection is enough on its own (recognition stage may need replacement) |
| safetensors | torch.save (.pt / .bin) | Never. .pt is pickle and executes arbitrary code on load. The hex-grid app must not be exposed to that. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| **PaliGemma 1 (`google/paligemma-3b-*`)** as MODEL-04 | Superseded by PaliGemma 2 at the same parameter count and identical HF integration; no benefit to running the older variant | `google/paligemma2-3b-pt-224` |
| **PaliGemma 2 / any Gemma-licensed checkpoint as MODEL-01 ship target** | Ships under Gemma Terms of Use; not Apache-2.0; carries prohibited-use riders that the hex-grid app's downstream redistribution can't cleanly inherit. Also fails the inference-budget constraint at 6 GB FP16 | SmolVLM-500M-Instruct (MODEL-01); PaliGemma 2 stays in MODEL-04 benchmark scope only |
| **PyTorch Lightning** | Wraps the training loop in a `LightningModule` that hides exactly the knobs we need to tune (per-source loss weights, multi-task balancing, freeze schedules). For a research codebase the loss is high; for a single developer the multi-GPU benefits Lightning ostensibly provides are equally available via `accelerate` with less ceremony | `accelerate` + hand-written training loop |
| **Full Hugging Face `Trainer`** | Same problem as Lightning, more so. `Trainer` hides loss computation, gradient accumulation, evaluation hooks; `PROJECT.md` constraints (per-source × per-class loss weighting, joint NLL on two heads, OCR auxiliary loss) require explicit control of all three. Patching `Trainer.compute_loss` is a known pain point | Bare `for batch in dataloader: ...` loop with `accelerator.backward()` |
| **pickle .bin checkpoints (`torch.save(model.state_dict(), 'foo.pt')`)** | Loads execute arbitrary Python; the hex-grid app must not inherit that risk. Also, embedding model versioning + dataset hashes per FEATURES.md TS-8 is much cleaner with safetensors metadata header | `safetensors.torch.save_file(state_dict, path, metadata={...})` |
| **wandb (in v1)** | Requires account setup, network during training, and adds a deploy surface (API keys) that a single-developer project doesn't need. Also: anything wandb logs can be re-derived from tensorboard logs trivially; the converse is harder | TensorBoard, local files only |
| **Hydra full launcher** | Brings in plugin loaders, multirun framing, working-directory mutations that a single-config-file training loop doesn't need. The `@hydra.main` decorator also imposes a CLI shape that conflicts with the existing `python -m mapclass.train --config foo.yaml` convention from ARCHITECTURE.md | OmegaConf with hand-rolled CLI parser |
| **MMOCR / MMSegmentation / mmcv** | Open-MMLab toolboxes pull in a deep dependency tree (mmengine, mmcv-full with custom CUDA ops); installation alone takes ~30min on a clean VM. Brings huge config systems we don't need | docTR (OCR) and home-rolled seg heads (per ARCHITECTURE.md Pattern 3) |
| **TensorFlow / JAX paths in any library that offers a choice (e.g. docTR's TF backend)** | Existing codebase is PyTorch-only; mixing frameworks in one repo is operational tax for zero benefit | Always pick the PyTorch flavour |
| **Idefics3-8B** as MODEL-01 (even at INT4) | 8B at INT4 ≈ 4.5 GB just for the LM weights, before activations or the seg heads. Pushes past the 4–8 GB GPU lower half; not a CPU candidate | SmolVLM-500M-Instruct |
| **Pickling / arbitrary Python in `mapclass.infer`** (e.g. `cloudpickle`) | Same arbitrary-code-execution risk as pickle .bin; doubly bad in the inference path which is the consumer's import surface | Pure tensor I/O via safetensors |
| **PyTorch Lightning's `LightningCLI` + jsonnet config** | Two layers of config indirection on top of an already-elaborate framework | OmegaConf + dataclass |

## Stack Patterns by Variant

**For MODEL-01 (small VL ship target):**
- Backbone: `HuggingFaceTB/SmolVLM-500M-Instruct` loaded via `AutoModelForVision2Seq`
- Vision-feature extraction: `model.model.vision_model` (the SigLIP-B/16 encoder); freeze, pull `output_hidden_states=True`
- Fine-tuning: full fine-tune is feasible (~500M params on a 16 GB VM is comfortable). PEFT/LoRA is optional and only needed if VRAM is tight or training data is small enough to overfit.
- Quantization for inference: optional INT8 dynamic quant on the full `GeoViLM` module post-training (`torch.ao.quantization.quantize_dynamic`)
- Inference budget: comfortable on 4 GB GPU at FP16; usable on CPU at ~2–4 s per 1024×1024 image

**For MODEL-04 (PaliGemma 2 benchmark variant):**
- Backbone: `google/paligemma2-3b-pt-224` loaded via `PaliGemmaForConditionalGeneration`
- Vision-feature extraction: `model.vision_tower` (SigLIP-So400m); freeze most parameters; the multi-modal projector after the vision tower is *not* needed (we attach our own seg heads directly to the vision-tower output)
- Fine-tuning: NF4 quantisation + LoRA adapters on the language tower (which we don't actually use, but `PaliGemmaForConditionalGeneration` keeps it loaded). Actual gradient flow only into the seg heads + a thin LoRA on the vision tower's last few blocks. Trainable parameters: ~5–10% of total → fits on a 24 GB GPU.
- Quantization for inference: not relevant; MODEL-04 is not shipped
- Inference budget: explicitly violated (this is the "bellwether upper bound" — the *point* is that it's bigger than MODEL-01)

**If a fully-CPU deployment becomes the priority (no GPU at all):**
- Drop SmolVLM-500M to SmolVLM-256M (drop is in the text decoder which we don't use; vision encoder is identical at 93M)
- INT8 dynamic quant on the seg heads
- Tile-and-stitch inference per FEATURES.md D-9 (becomes table-stakes if input resolution is large)

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `transformers>=4.47` | `torch>=2.4` | transformers 4.47 added PaliGemma 2; lower torch versions break some SDPA paths |
| `transformers>=4.47,<5.0` | `accelerate>=0.34` | accelerate 0.34+ matches transformers' 4.47 expectation for `accelerator.prepare_model` semantics |
| `peft>=0.11` | `transformers>=4.45` | LoRA on PaliGemma 2 vision tower needs peft 0.11+ for the SigLIP target-module recognition |
| `bitsandbytes>=0.43` | CUDA 12.1+ | Required for NF4 quantisation. The cloud-VM repo's `nvidia-smi` output drives this pin |
| `safetensors>=0.4.5` | `torch>=2.4` | Older safetensors had a memory leak with `torch.save_file` from CUDA tensors; 0.4.5 fixed it |
| `python-doctr[torch]>=1.0.1` | `torch>=2.0`, `Pillow>=9.0` | docTR 1.0 dropped some legacy PyTorch <2.0 paths |
| `omegaconf>=2.3` | Python 3.10+ | Older Python 3.8/3.9 paths in 2.2 are deprecated |

## Quality Gate Verification

- [x] **HF model IDs verified loadable in 2025–2026.** `HuggingFaceTB/SmolVLM-500M-Instruct` (Apache-2.0, current), `vikhyatk/moondream2` (Apache-2.0, current), `google/paligemma2-3b-pt-224` (Gemma TOU, current — confirmed in transformers 4.47 release notes and the official PaliGemma 2 blog post).
- [x] **License confirmed for shipping trained weights publicly:** SmolVLM-500M-Instruct → Apache-2.0; trained weights are redistributable under the project's choice of compatible license, satisfying SHIP-01. PaliGemma 2 → Gemma TOU; **MODEL-04 artifact must NOT be packaged with the SmolVLM-based ship checkpoint** because the licenses do not align — kept separate per `PROJECT.md` decision that PaliGemma is a research/benchmark artifact only.
- [x] **Inference budget honoured at fp16:** SmolVLM-500M FP16 ≈ 1 GB weights + ~50 MB seg heads + ~200–400 MB activations on a 1024×1024 input ≈ ~1.5–2 GB total. Fits 4 GB GPU envelope with multiple GB to spare. CPU usable.
- [x] **Each recommendation has confidence level:** Backbone choice — HIGH; PaliGemma 2 swap — HIGH; OCR module starting point — MEDIUM (active research area, fine-tuning likely needed); training framework — HIGH; checkpoint format — HIGH; config system — HIGH (Hydra-vs-OmegaConf is well-trodden ground); logging — HIGH.
- [x] **Aligned with `.planning/research/ARCHITECTURE.md`:** Backbone interface (Pattern 3) is satisfied by both MODEL-01 (`SmolVLMBackbone`) and MODEL-04 (`PaliGemma2Backbone`) implementations; OCR-head wiring (`mapclass/model/ocr_head.py`) is satisfied by docTR-as-starting-point; sample-contract / data-flow / bootstrap-loop patterns are unchanged. **No contradiction with ARCHITECTURE.md.** The MODEL-04 addition introduces parallel training but not parallel infrastructure — same trainer, eval, and config layer for both backbones, which is exactly what ARCHITECTURE.md's swap-surface design enables.

## Roadmap-facing summary (one paragraph)

Bare PyTorch + HF transformers + accelerate + safetensors + OmegaConf + tensorboard, with `HuggingFaceTB/SmolVLM-500M-Instruct` as the MODEL-01 ship backbone (Apache-2.0, ~500M params, ~1 GB FP16, comfortably inside the 4–8 GB inference envelope, CPU-usable, vision encoder is the well-validated 93M SigLIP-B/16 with first-class HF integration and no `trust_remote_code`), `google/paligemma2-3b-pt-224` as the MODEL-04 benchmark backbone (Gemma TOU; trained with NF4+LoRA on a 24 GB GPU; explicitly *not* shipped), `python-doctr[torch]>=1.0.1` as the OCR module starting point (active maintenance, end-to-end pipeline, rotation-aware — though map-text fine-tuning may be needed and is flagged as a v1.x concern). Avoid Lightning, full HF Trainer, Hydra, wandb, MMOCR, and any pickle-based checkpoint format. The cloud-VM SSH workflow gains exactly one new artifact: a minimal `pyproject.toml` + `.python-version` + `requirements.lock.txt` so `pip install -e .` works for `python -m mapclass.train`.

## Sources

### Verified at HIGH confidence (Hugging Face official model cards / blog posts / transformers release notes)

- [HuggingFaceTB/SmolVLM-500M-Instruct (model card)](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct) — Apache-2.0, params, processor/model classes
- [HuggingFaceTB/SmolVLM-256M-Instruct (model card)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct) — Apache-2.0, smallest variant
- [SmolVLM blog post](https://huggingface.co/blog/smolvlm) — architecture (SigLIP encoder + SmolLM2), Apache-2.0 license confirmation
- [SmolVLM 256M / 500M release blog](https://huggingface.co/blog/smolervlm) — vision encoder shrunk from 400M SigLIP to 93M SigLIP-B/16
- [SmolVLM2 paper (arXiv:2504.05299)](https://arxiv.org/html/2504.05299v1) — token efficiency (81 tokens/384×384 patch)
- [vikhyatk/moondream2 (model card)](https://huggingface.co/vikhyatk/moondream2) — Apache-2.0; SigLIP + Phi-1.5; trust_remote_code requirement
- [moondream/moondream-2b-2025-04-14-4bit (model card)](https://huggingface.co/moondream/moondream-2b-2025-04-14-4bit) — INT4 variant
- [google/paligemma2-3b-pt-224 (model card)](https://huggingface.co/google/paligemma2-3b-pt-224) — pt-vs-mix distinction, fine-tuning intent
- [google/paligemma2-3b-mix-224 (model card)](https://huggingface.co/google/paligemma2-3b-mix-224) — mix variant for direct task use
- [PaliGemma 2 blog post (HF)](https://huggingface.co/blog/paligemma2) — 3B/10B/28B sizes; 224/448/896 resolutions; transformers 4.47 minimum
- [transformers v4.47.0 release notes](https://github.com/huggingface/transformers/releases/tag/v4.47.0) — confirms PaliGemma 2 introduced in this release
- [PaliGemma in transformers docs](https://huggingface.co/docs/transformers/model_doc/paligemma) — `PaliGemmaForConditionalGeneration` API
- [Gemma Terms of Use](https://ai.google.dev/gemma/terms) — distribution, commercial use, prohibited uses, indemnification
- [microsoft/Florence-2-base (model card)](https://huggingface.co/microsoft/Florence-2-base) — MIT; DaViT encoder; trust_remote_code
- [microsoft/Florence-2-large (model card)](https://huggingface.co/microsoft/Florence-2-large) — same architecture, larger
- [Florence-2 transformers docs](https://huggingface.co/docs/transformers/model_doc/florence2) — HF integration
- [SigLIP 2 paper (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — dense-feature improvements over SigLIP 1; ViT-B/16 (86M), ViT-L/16 (303M), ViT-So400m/14 (400M), ViT-g/16 (1B)
- [SigLIP 2 blog (HF)](https://huggingface.co/blog/siglip2) — segmentation/depth-estimation evaluation
- [DINOv2 transformers docs](https://huggingface.co/docs/transformers/model_doc/dinov2) — backbone interface, ViT-S/14 etc.
- [facebook/dinov2-small (model card)](https://huggingface.co/facebook/dinov2-small) — sanity-baseline candidate
- [docTR project (mindee)](https://mindee.github.io/doctr/) — OCR pipeline overview
- [docTR releases](https://github.com/mindee/doctr/releases) — confirms 2025 v1.0+ active maintenance, ParSeq/ViTSTR additions
- [docTR rotation discussion](https://github.com/mindee/doctr/discussions/1283) — rotation-tolerant pipeline configuration
- [Safetensors documentation (HF)](https://huggingface.co/docs/safetensors/index) — metadata header, `save_file`/`load_file` API
- [PyTorch 2.5 release blog](https://pytorch.org/blog/pytorch2-5/) — cuDNN SDPA backend speedups; relevant for ViT inference
- [accelerate (huggingface/accelerate)](https://github.com/huggingface/accelerate) — minimum-boilerplate training-loop wrapper

### Verified at MEDIUM confidence (community blog posts / aggregator articles / forum discussions)

- [Spheron blog: SmolVLM/SmolVLA edge deployment guide (2026)](https://www.spheron.network/blog/smolvlm-smolvla-gpu-cloud-edge-ai-robotics/) — practical VRAM/CPU numbers
- [PyImageSearch: SmolVLM to SmolVLM2 multi-image VQA (2025)](https://pyimagesearch.com/2025/06/23/smolvlm-to-smolvlm2-compact-models-for-multi-image-vqa/) — vision-encoder details across SmolVLM versions
- [Hugging Face VLMs 2025 retrospective blog](https://huggingface.co/blog/vlms-2025) — SmolVLM 2.0 and detection-task limitations
- [Roboflow Moondream2 review](https://blog.roboflow.com/moondream-2/) — VRAM and inference characteristics for Moondream2
- [Lightning vs Accelerate comparison (Restack)](https://www.restack.io/p/pytorch-lightning-answer-accelerate-vs-pytorch-lightning-cat-ai) — boilerplate trade-off analysis
- [Hydra/OmegaConf article (Neural Bits)](https://medium.com/neuralbits/master-ml-configuration-files-using-omegaconf-and-hydra-48d35c5dfd2c) — when each is needed
- [ABCNet repo (Yuliang-Liu/bezier_curve_text_spotting)](https://github.com/Yuliang-Liu/bezier_curve_text_spotting) — confirms unmaintained status
- [CRAFT repo (clovaai/CRAFT-pytorch)](https://github.com/clovaai/CRAFT-pytorch) — confirms unmaintained status

### Internal context (anchoring)

- `/home/drdreadknee/mapclass/.planning/PROJECT.md` — locked v1 scope (MODEL-01 / MODEL-04 / SHIP-01 constraints binding)
- `/home/drdreadknee/mapclass/.planning/research/ARCHITECTURE.md` — backbone-interface swap surface (Pattern 3); training/eval/infer entry-point contracts; `pyproject.toml` requirement
- `/home/drdreadknee/mapclass/.planning/research/FEATURES.md` — TS-1..TS-11 cross-reference for framework / checkpoint / config / logging choices
- `/home/drdreadknee/mapclass/.planning/codebase/STACK.md` — existing pinned-but-unused `torch / transformers / peft / accelerate / bitsandbytes` confirms intent direction; absence of `pyproject.toml` flagged
- `/home/drdreadknee/mapclass/.planning/codebase/ARCHITECTURE.md` — sample-contract on disk (`image*.png + land_cover.png + topography.png [+ sample_weights.json]`) — both candidate backbones consume this unchanged
- `/home/drdreadknee/mapclass/.planning/codebase/CONCERNS.md` — items 8a (no seeds), 8b (no dataset versioning), 8c (no lockfile) drove the OmegaConf + uv.lock + safetensors-with-metadata stack choices

---

*Stack research for: MapClass / GeoViLM (multi-source-trained dense-prediction VL+OCR+heads model, model-handoff package)*
*Researched: 2026-05-08*
