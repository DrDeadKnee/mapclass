# Stack Research

**Domain:** Adding CLIP, PaliGemma, and a plain ViT as comparison models for cross-model dynamic-LRP attribution, under a FROZEN transformers==4.52.3 / torch==2.7.1 / vendored-dynamicLRP stack (MapClass v1.1)
**Researched:** 2026-05-19
**Confidence:** HIGH on model ids / transformers classes / gating / attribution targets (verified against transformers v4.52.3 docs + HF model cards); MEDIUM on exact parameter counts; the dynamic-LRP op-coverage outcome per model is intentionally UNKNOWN (it is the data v1.1 produces, not a thing to pre-resolve).

> Supersedes the 2026-05-18 v1.0 SigLIP-2-only stack research. The v1.0 pins
> (torch==2.7.1+cu126, torchvision==0.22.1, transformers==4.52.3, vendored
> dynamicLRP @ SHA 405e7424…, google-cloud-storage, JupyterLab) are FROZEN and
> NOT re-researched here. This document covers ONLY the v1.1 model additions.

## Executive Finding (read this first)

**No stack pins change. No new hard dependency is required for CLIP or ViT.** Both
`CLIPModel`/`CLIPProcessor` and `ViTForImageClassification`/`ViTImageProcessor` are
native to transformers 4.52.3 and load with the existing torch 2.7.1 stack and the
existing `huggingface_hub` (transitive) + `google-cloud-storage` mirror path — the
exact same code shape already shipped for SigLIP-2.

**PaliGemma is the one that carries cost and risk:**
1. **Gated weights** — every `google/paligemma*` repo requires accepting Google's
   Gemma license while logged into a Hugging Face account *before* the weights can
   be downloaded (even though the repo is "publicly listed"). The v1.0 mirror step
   used an anonymous `snapshot_download`; PaliGemma needs an authenticated HF token
   on the (one-time) mirror VM. Once mirrored to GCS this is moot — the runtime
   loader reads GCS, not HF — but the **mirror step must be run with an HF token
   tied to an account that has accepted the Gemma terms**, and the GCS bucket then
   holds Gemma-licensed weights (a license/redistribution consideration to flag,
   not a blocker for a private single-researcher bucket).
2. **Size** — the smallest PaliGemma is **~3B params**, ~7.5× the SigLIP-2-so400m
   (~400M) dynamic-LRP fidelity ceiling stated in PROJECT.md/CLAUDE.md. It is
   included because the *comparison* is the point (a model the engine cannot
   traverse, or that blows VRAM, is a recorded result), but it should be loaded in
   `bf16` and the LRP relevance pass on a 3B generative decoder may exceed the L4's
   24 GB — that is itself a recordable comparison datum, consistent with the
   "coverage gap = result, not bug" decision.
3. **Different attribution target** — PaliGemma is generative; there is no
   `logits_per_image`. The query-conditioned target is the **logit of the answer
   token** at the generated position (`logits[0, answer_pos, answer_token_id]`),
   which is a genuinely different attribution wiring from CLIP/SigLIP-2 and must be
   implemented per-model in the adapter.

**Plain ViT is NOT image-text.** `google/vit-base-patch16-224` has no text tower
and no `logits_per_image`. Its only query-conditionable scalar is an **ImageNet
class logit** (`logits[0, class_id]`). The notebook's free-text query ("a river")
cannot condition a plain ViT the way it conditions CLIP/SigLIP-2 — this asymmetry
must be flagged in the adapter and the side-by-side UI (e.g. map the text query to
the nearest ImageNet class, or fix a class, and label the ViT panel as
"class-conditioned, not text-conditioned"). This is a genuine semantic caveat, not
an implementation detail.

## Recommended Stack

### Core Technologies (additions for v1.1 — all already satisfiable under the frozen pins)

| Technology | Version | Purpose | Why Recommended |
|------------|---------|---------|-----------------|
| `CLIPModel` + `CLIPProcessor` (transformers) | from `transformers==4.52.3` (no change) | CLIP comparison model: contrastive image-text, exposes `logits_per_image` | Native to 4.52.3; `logits_per_image` is the *exact same* query-conditioned target already used for SigLIP-2 — the v1.0 `attribute()` path applies almost verbatim. `[VERIFIED: HF transformers v4.52.3 CLIP docs — CLIPModel/CLIPProcessor, logits_per_image shape (image_bs, text_bs)]` |
| `openai/clip-vit-large-patch14` weights | HF snapshot → GCS mirror | The CLIP checkpoint under attribution | ViT-L/14 vision tower (~304M vision params, ~428M total) — the closest CLIP to the SigLIP-2-so400m (~400M) fidelity ceiling without exceeding it. Patch 14 / image 224 → 16×16=256 patch grid. CLIP uses a CLS token (NO MAP-pool `split_with_sizes`, *unlike* SigLIP-2) → a deliberately different LRP path = a real comparison datum. NOT gated. `[VERIFIED: HF model card openai/clip-vit-large-patch14; transformers CLIP docs default image_size 224]` |
| `ViTForImageClassification` + `ViTImageProcessor` (transformers) | from `transformers==4.52.3` (no change) | Plain-ViT comparison model: image classifier, exposes class `logits` | Native to 4.52.3; this is the architecture closest to dynamicLRP's own verified `ViT.ipynb` example — highest a-priori chance the engine traverses it cleanly, making it the cross-model "engine works at all" control. `[VERIFIED: HF transformers v4.52.3 ViT docs — ViTForImageClassification, logits shape (batch, num_labels)]` |
| `google/vit-base-patch16-224` weights | HF snapshot → GCS mirror | The plain-ViT checkpoint under attribution | ~86M params (well under the ceiling), patch 16 / image 224 → 14×14=196 patches + 1 CLS = 197 tokens; ImageNet-1k head (`num_labels=1000`). Matches the dynamicLRP paper's ViT family. NOT gated. `[VERIFIED: HF transformers v4.52.3 ViT docs — patch 16, image 224, 12 layers, hidden 768, 197 tokens]` |
| `PaliGemmaForConditionalGeneration` + `PaliGemmaProcessor` (transformers) | from `transformers==4.52.3` (no change) | PaliGemma comparison model: generative VLM, exposes vocab `logits` | Native to 4.52.3 (both `paligemma` and `paligemma2` model types are present in 4.52.3 — verified). Generative → attribution target = logit of the answer token, a deliberately different wiring. `[VERIFIED: HF transformers v4.52.3 PaliGemma docs — PaliGemmaForConditionalGeneration, logits shape (batch, seq_len, vocab_size); PaliGemma2 supported in 4.52.3]` |
| `google/paligemma2-3b-pt-224` weights | **authenticated** HF snapshot (Gemma-license-accepted token) → GCS mirror | The PaliGemma checkpoint under attribution | Smallest PaliGemma family member (~3B). 224-res `pt` variant keeps the image grid small (224/14=16 → 256 image tokens) to bound the LRP graph. **GATED**: requires HF login + accepted Gemma terms to download (one-time mirror only). `[VERIFIED: HF model card google/paligemma2-3b-pt-224 — "you have to accept the conditions to access its files", Gemma license, 3B params]` |

### Supporting Libraries (dependency delta — what must / must not be added)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `huggingface_hub` (already transitive of `transformers==4.52.3`) | unchanged — do NOT pin/upgrade separately | `snapshot_download(..., token=...)` for the gated PaliGemma mirror; anonymous for CLIP/ViT | Already used for the v1.0 SigLIP-2 mirror. For PaliGemma, pass an HF token (env `HF_TOKEN`) on the **one-time mirror VM only**; runtime never touches HF. No version bump — use whatever `transformers==4.52.3` already resolved. |
| `sentencepiece` | **NOT required** — do not add | (would be the Gemma tokenizer backend) | transformers 4.52.3's `PaliGemmaProcessor` uses `GemmaTokenizerFast` (Rust/`tokenizers`-backed, already a transformers transitive dep). Adding `sentencepiece` is unnecessary and risks pulling a build the frozen resolve did not vet. `[VERIFIED: HF transformers v4.52.3 PaliGemma docs — PaliGemmaProcessor wraps GemmaTokenizerFast]` |
| `accelerate` | **OPTIONAL**, only if added: `>=0.26,<1.1` | Enables `device_map="auto"` / low-mem sharded load for the 3B PaliGemma | NOT required: a single-GPU L4 can `.to("cuda")` the 3B bf16 model with a plain `from_pretrained(...).to(device)`, exactly like v1.0's SigLIP-2 loader. Add `accelerate` ONLY if the 3B load OOMs host RAM during materialization; pin conservatively so the resolver cannot disturb `torch==2.7.1`/`transformers==4.52.3`. Fallback, not a baseline addition. |
| `google-cloud-storage` (already pinned `>=3.0,<4`) | unchanged | Mirror the 3 new model snapshots to `gs://mapclass-training-northeast1/models/` exactly like SigLIP-2 | Reuse the v1.0 `mirror_model.py` recursive-upload pattern verbatim; only the repo id (and, for PaliGemma, the auth token) changes. |

**Net dependency delta: ZERO new hard requirements.** CLIP and ViT add no
packages. PaliGemma adds no *runtime* package; it adds an *operational* requirement
(an HF token with Gemma terms accepted, used once at mirror time). `accelerate` is a
conditional fallback only. `requirements.txt` is UNCHANGED.

### Development Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| `HF_TOKEN` env var (mirror VM only) | Authenticate the one-time PaliGemma `snapshot_download` | Required because PaliGemma repos are gated. The account behind the token must have visited the model page and accepted the Gemma license once. Never needed at runtime (GCS-only loaders). Do NOT commit the token; do NOT bake it into the GCS-loaded model dir. |
| existing `mirror_model.py` (v1.0) | Recursive HF-snapshot → GCS upload | Generalize to `(repo_id, gcs_prefix, token=None)`; CLIP/ViT pass `token=None`, PaliGemma passes the env token. Idempotent skip-if-exists logic unchanged. |
| existing `model_loader.py` (v1.0) | GCS → local → `from_pretrained` | Generalize to dispatch on model id → correct class (`AutoModel`/`CLIPModel`/`ViTForImageClassification`/`PaliGemmaForConditionalGeneration` + matching processor). The model-agnostic adapter is a v1.1 feature, not a stack concern. |

## Installation

```bash
# NOTHING TO INSTALL for CLIP or ViT — both are in the already-frozen
# transformers==4.52.3.  requirements.txt is UNCHANGED.

# One-time PaliGemma mirror (on the mirror VM only — NOT a dependency change):
#   1. Accept Gemma terms at https://huggingface.co/google/paligemma2-3b-pt-224 (web, once)
#   2. export HF_TOKEN=hf_xxx   # account that accepted the terms
#   3. run the (generalized) mirror_model.py for the three new repo ids:
#        openai/clip-vit-large-patch14        (token=None)
#        google/vit-base-patch16-224          (token=None)
#        google/paligemma2-3b-pt-224          (token=$HF_TOKEN)
#   -> uploads to gs://mapclass-training-northeast1/models/<repo_basename>/

# OPTIONAL fallback ONLY if the 3B PaliGemma OOMs host RAM at load:
#   pip install 'accelerate>=0.26,<1.1'   # gate behind a human-verify checkpoint;
#   confirm `pip` does NOT propose torch/transformers changes before accepting.
```

## Per-Model Attribution Target (the load-bearing ambiguity to flag)

| Model | transformers class | Query-conditioned target | Same as SigLIP-2? | Flag |
|-------|--------------------|--------------------------|-------------------|------|
| SigLIP-2 (v1.0, shipped) | `AutoModel`/`SiglipModel` | `output.logits_per_image[0,0]` | — (reference) | Known: engine does NOT cover its `split_with_sizes` MAP-pool op (recorded result) |
| CLIP | `CLIPModel` | `output.logits_per_image[0,0]` | **YES — verbatim** | Different head from SigLIP-2 (CLIP has a CLS token + projection, no MAP-pool `split_with_sizes`) → genuinely different op-coverage outcome expected. This is the point. |
| plain ViT | `ViTForImageClassification` | `output.logits[0, class_id]` (an ImageNet-1k class logit) | **NO — not text-conditioned** | **MUST FLAG**: no text tower; the free-text query cannot condition it. Either fix a class id or map the query → nearest ImageNet label, and label the ViT panel "class-conditioned, not text-conditioned" so the side-by-side is not misread as apples-to-apples. |
| PaliGemma | `PaliGemmaForConditionalGeneration` | `output.logits[0, answer_pos, answer_token_id]` (logit of the answer token for a "<image> {query}?" prompt) | **NO — generative** | **MUST FLAG**: target is a single vocab logit at a generated position; choosing `answer_pos`/`answer_token_id` is a per-query modeling decision (e.g. teacher-forced single-token answer). Different wiring; ~3B size may exceed the LRP/VRAM ceiling — an expected, recordable comparison result. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|-------------|-------------|-------------------------|
| `openai/clip-vit-large-patch14` (CLIP) | `openai/clip-vit-base-patch16` (~150M) | If the L/14 LRP graph OOMs the L4 or you want CLIP *under* (not at) the so400m ceiling for a cleaner size-matched comparison. base-patch16 → 14×14 grid, smaller VRAM. Equally non-gated, same `CLIPModel`/`logits_per_image` path. A reasonable swap; L/14 chosen to size-match SigLIP-2-so400m. |
| `openai/clip-vit-large-patch14` | `openai/clip-vit-large-patch14-336` | Only if 336-res alignment with SigLIP-2's 384 grain matters; larger token grid = more LRP memory for marginal comparison value. Not worth it for v1.1. |
| `google/vit-base-patch16-224` | `timm` ViT (e.g. `timm.create_model('vit_base_patch16_224')`) | If you want the *exact* model the dynamicLRP `ViT.ipynb` uses. `timm==1.0.20` is already a frozen dep, so a timm ViT adds nothing and is the most faithful reproduction of the engine's known-good path. Acceptable substitute; HF `ViTForImageClassification` preferred only for loader/processor uniformity with the other three HF models. |
| `google/paligemma2-3b-pt-224` | `google/paligemma-3b-pt-224` (PaliGemma **1**, Gemma-1 decoder) | If PaliGemma 2's Gemma-2 decoder ops trip the engine in a way that obscures the comparison, PaliGemma 1 is a simpler decoder. Both gated, both 3B, both `PaliGemmaForConditionalGeneration` in 4.52.3. PaliGemma 2 chosen as the current/representative family. |
| `google/paligemma2-3b-pt-224` | `google/paligemma2-3b-mix-224` | `mix` is instruction-tuned/ready-to-use; `pt` is the pretrained base. For a single-token answer-logit target the `pt` base is cleaner and avoids chat-template confounds. Use `mix` only if you want natural-language answers in the prompt loop. |
| Plain `from_pretrained(...).to(device)` for PaliGemma | `device_map="auto"` via `accelerate` | Only if 3B bf16 materialization OOMs host RAM on the mirror/runtime VM. Adds `accelerate` (conditional dep) — gate behind a checkpoint. |

## What NOT to Use

| Avoid | Why | Use Instead |
|-------|-----|-------------|
| Bumping `transformers` above 4.52.3 "to get newer PaliGemma" | Both `paligemma` and `paligemma2` are ALREADY in 4.52.3. A bump violates the HARD dynamicLRP autograd-graph pin (CLAUDE.md "What NOT to Use", D-08) | `transformers==4.52.3` as-is |
| Bumping `torch`/`torchvision` for any new model | Engine traverses torch autograd internals; node-name drift silently breaks LRP. None of CLIP/ViT/PaliGemma need a newer torch | `torch==2.7.1` + `torchvision==0.22.1` (unchanged) |
| Adding `sentencepiece` for PaliGemma | 4.52.3's `PaliGemmaProcessor` uses `GemmaTokenizerFast` (tokenizers-backed); `sentencepiece` is unnecessary and an unvetted resolver perturbation | Nothing — `GemmaTokenizerFast` ships with transformers |
| Adding `accelerate` unconditionally | Not needed for single-GPU `.to(device)` load; an unpinned add can disturb the frozen resolve | Plain `from_pretrained(...).to(device)`; add `accelerate>=0.26,<1.1` ONLY as a measured OOM fallback |
| Anonymous `snapshot_download("google/paligemma2-3b-pt-224")` | PaliGemma is GATED — anonymous download 401/403s | Authenticated `snapshot_download(..., token=HF_TOKEN)` on the one-time mirror, account having accepted Gemma terms |
| `model.get_image_features()` / pooled embedding as the CLIP target | Produces query-INDEPENDENT heatmaps (same mistake called out for SigLIP-2 in v1.0) | `output.logits_per_image[0,0]` (query-conditioned) |
| Treating the plain-ViT panel as text-query-conditioned | ViT has no text tower; silently equating its class-logit map with the CLIP/SigLIP-2 text-conditioned maps misrepresents the comparison | Explicitly label ViT as class-conditioned; document the query→class mapping |
| `Siglip2*` NaFlex classes for CLIP/ViT, or `AutoModel` for the ViT classifier | Wrong class → load/shape errors or pooled (non-logit) output | `CLIPModel`, `ViTForImageClassification`, `PaliGemmaForConditionalGeneration` exactly as tabled |
| Fixing a PaliGemma/ViT/CLIP op-coverage gap (custom Promise, LXT, captum) | Per-model coverage gap is a RECORDED RESULT, not a bug (PROJECT.md Out of Scope; Fallback Ladder declined in v1.0) | Let the engine fail honestly; render "no heatmap" for that model |

## Stack Patterns by Variant

**If the model is contrastive image-text (SigLIP-2, CLIP):**
- Target `output.logits_per_image[0,0]`; detach the text path; reuse the v1.0
  `attribute()` scope almost verbatim.
- CLIP differs from SigLIP-2 by having a CLS token + linear projection (no MAP-pool
  `split_with_sizes`) — expect a *different* op-coverage outcome (the comparison datum).

**If the model is a plain image classifier (ViT):**
- Target `output.logits[0, class_id]`; there is NO text conditioning.
- This is the architecture closest to dynamicLRP's verified `ViT.ipynb` — treat it as
  the cross-model "engine works at all" control.
- Flag the query→class semantic gap in the adapter and the notebook UI.

**If the model is generative (PaliGemma):**
- Build a "<image> {query}" prompt; target the logit of the chosen answer token at
  its generated/teacher-forced position: `output.logits[0, answer_pos, answer_token_id]`.
- Load in `bf16`; size (~3B) is ~7.5× the so400m ceiling — VRAM/coverage failure on
  the L4 is an EXPECTED, recordable result, not a defect to engineer around.

## Version Compatibility

| Package A | Compatible With | Notes |
|-----------|-----------------|-------|
| `transformers==4.52.3` | `CLIPModel`/`CLIPProcessor` | Native since long before 4.49; `logits_per_image` shape `(image_bs, text_bs)`. `[VERIFIED: v4.52.3 CLIP docs]` |
| `transformers==4.52.3` | `ViTForImageClassification`/`ViTImageProcessor` | Native; logits `(batch, num_labels)`; `google/vit-base-patch16-224` → 197 tokens. `[VERIFIED: v4.52.3 ViT docs]` |
| `transformers==4.52.3` | `paligemma` AND `paligemma2` model types | Both present in 4.52.3 (PaliGemma 2 uses Gemma-2 decoder); `PaliGemmaForConditionalGeneration` logits `(batch, seq, vocab)`. `[VERIFIED: v4.52.3 PaliGemma docs explicitly cover PaliGemma 2]` |
| `transformers==4.52.3` | `GemmaTokenizerFast` (PaliGemma tokenizer) | Ships with transformers; `tokenizers`-backed → no `sentencepiece` needed. `[VERIFIED: v4.52.3 PaliGemma docs]` |
| `torch==2.7.1` | all three new models | Standard PyTorch modules; no model needs a newer torch. bf16 supported on the L4. `[ASSUMED — standard transformers/torch contract; no model-specific torch floor above 2.7.1 known]` |
| `huggingface_hub` (transitive) | gated `snapshot_download(token=...)` | Token-auth download is a long-stable hub feature; whatever 4.52.3 resolved is sufficient — do NOT pin/upgrade hub separately. `[ASSUMED — stable hub API; not re-verified at exact resolved version]` |
| `google-cloud-storage>=3.0,<4` | the 3 new model mirrors | Identical to the SigLIP-2 mirror; independent of the torch/transformers stack. `[VERIFIED: reused v1.0 pattern]` |
| `accelerate>=0.26,<1.1` (IF added) | `torch==2.7.1`, `transformers==4.52.3` | Conservative range so the resolver cannot pull a build that re-pins torch/transformers; conditional fallback only. `[ASSUMED — safety band; confirm `pip` plan at install before accepting]` |

## Sizing vs. the dynamic-LRP fidelity ceiling

| Model | ~Params | Image grid (tokens) | Vs. so400m (~400M) ceiling |
|-------|---------|---------------------|-----------------------------|
| `google/vit-base-patch16-224` | ~86M | 14×14 + CLS = 197 | Well under — lowest LRP/VRAM risk; the "control" |
| `openai/clip-vit-large-patch14` | ~428M total (~304M vision) | 16×16 + CLS = 257 | At/near the ceiling — size-matched to SigLIP-2 by design |
| SigLIP-2-so400m (v1.0) | ~400M | 27×27 = 729 (no CLS) | the reference ceiling |
| `google/paligemma2-3b-pt-224` | ~3B | 16×16 = 256 image + text | **~7.5× over** — included for the comparison; VRAM/coverage failure on the L4 is an expected recordable datum, not a bug (consistent with PROJECT.md "coverage gap = result") |

## Sources

- HF transformers v4.52.3 docs — CLIP (`https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/clip`): `CLIPModel`/`CLIPProcessor`, `logits_per_image` shape `(image_bs, text_bs)`, default vision image_size 224 — HIGH
- HF transformers v4.52.3 docs — ViT (`https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/vit`): `ViTForImageClassification`/`ViTImageProcessor`, logits `(batch, num_labels)`, `google/vit-base-patch16-224` (patch16/224/12L/768h/197 tokens) — HIGH
- HF transformers v4.52.3 docs — PaliGemma (`https://huggingface.co/docs/transformers/v4.52.3/en/model_doc/paligemma`): `PaliGemmaForConditionalGeneration`/`PaliGemmaProcessor`, logits `(batch, seq, vocab)`, `GemmaTokenizerFast` (no sentencepiece), PaliGemma 2 supported in 4.52.3 — HIGH
- HF model card `google/paligemma2-3b-pt-224`: GATED ("you have to accept the conditions to access its files"), Gemma license, 3B params, `AutoProcessor`/`AutoModelForImageTextToText` example — HIGH
- HF model card `openai/clip-vit-large-patch14` / community size references: ViT-L/14, ~428M total params, not gated — MEDIUM (param count from secondary aggregators; class/grid from primary docs)
- arXiv 2512.07010 ("Always Keep Your Promises", dynamicLRP) abstract + vendored `third_party/dynamicLRP` tree (promises: add/cat/dummy/softmax/split/stack/sub/sum/unbind; `model_specific/mosaicbert.py` only) — the engine has NO CLIP/PaliGemma/ViT-specific model module; per-model coverage is empirical and is the v1.1 data — HIGH on what is/isn't vendored; coverage outcome intentionally UNKNOWN
- `.planning/PROJECT.md`, `requirements.txt`, `01-03-SUMMARY.md`, v1.0 `01-RESEARCH.md`, `CLAUDE.md` — frozen pins, SigLIP-2 `split_with_sizes` recorded gap, "coverage gap = result not bug" decision, mirror/loader patterns — HIGH

---
*Stack research for: multi-model dynamic-LRP comparison additions (CLIP / PaliGemma / plain ViT) under frozen transformers==4.52.3 / torch==2.7.1 / vendored dynamicLRP*
*Researched: 2026-05-19*
</content>
