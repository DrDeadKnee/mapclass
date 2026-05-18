# Pitfalls Research

**Domain:** Adapting dynamic LRP (keeinlev/dynamicLRP) to SigLIP-2-so400m for query-driven pixel attribution on historical maps; GCS-backed ingestion + JupyterLab exploration on an ephemeral GCP VM
**Researched:** 2026-05-18
**Confidence:** MEDIUM-HIGH (LRP repo + arXiv 2512.07010 inspected directly; SigLIP-2 architecture verified against HuggingFace; contrastive-attribution semantics from XAI literature, MEDIUM)

## Key Upfront Finding (changes the central risk framing)

`PROJECT.md` calls "adapting dynamicLRP to SigLIP-2" the central technical risk. The paper's own coverage tables (arXiv 2512.07010, Table A.4) report **SigLIP-2-So400m-base14-384 at 100% node coverage: 2,178/2,178 nodes, 19/19 unique ops** — and SigLIP-2's full op set (Conv, Mm/Bmm, SDPA, NativeLayerNorm, GELU, Softmax, Embedding) is entirely within DynamicLRP's covered-operations list (Appendix F). So the operation-level adapter is very likely to *run without erroring*.

**The real risk is not "will it run" — it is "will it run silently and produce a wrong but plausible heatmap."** The paper tests SigLIP-2 only as a **Vision** model (image-classification-style coverage), **not** as a contrastive image-text dual encoder, and reports no faithfulness benchmark for it (no entry in Table 1). This reframes every pitfall below: the failure mode is *silent wrong attribution*, not crashes.

---

## Critical Pitfalls

### Pitfall 1: Backpropagating relevance from the wrong scalar in a contrastive dual encoder

**What goes wrong:**
DynamicLRP needs a single scalar `target_node` to propagate from (paper Algorithm, line 278: "Relevance attribution target target_node"). For a classifier this is "logit of class k." For SigLIP-2 there is **no class logit** — the model produces an image embedding and a text embedding, and the score is a cosine/dot similarity passed through a learned temperature + bias into a sigmoid. People default to propagating from (a) the raw image-encoder pooled output, (b) one image-embedding dimension, or (c) `logits_per_image` after sigmoid. Only the *similarity to the specific text query* is the semantically correct target, and even that has subtleties (the sigmoid + bias is monotonic so it can be dropped; the L2-normalization of embeddings is **not** a no-op for relevance and must be in the graph).

**Why it happens:**
Every DynamicLRP example in the repo/paper is a classifier or generative LM with an obvious target logit. There is no documented dual-encoder example (DePlot is the only Vision/Language model tested and it is a generative captioner, not a contrastive encoder — different target semantics entirely). The "obvious" target (image embedding norm) produces a *visually convincing but query-independent* heatmap — it highlights whatever is salient to the vision tower regardless of the query.

**How to avoid:**
- Define the target explicitly as the **dot product of the (normalized) image embedding with the (frozen, detached) text embedding for the query**. Treat text-tower outputs as constants (detach) so relevance flows only through pixels.
- Drop the sigmoid and temperature/bias scaling (monotonic — does not change relative attribution) but **keep image-embedding L2-normalization inside the traced graph** so the Promise system handles it; removing it changes attributions.
- Sanity test: swap the query to an unrelated word ("xylophone"). If the heatmap barely changes, you are attributing image saliency, not query relevance — wrong target.

**Warning signs:**
Heatmaps look the same across very different queries; heatmap is identical whether the query is present in the map or not; attributions concentrate on high-frequency texture regardless of semantics.

**Phase to address:** Attribution (must be settled before any sweep; a wrong target invalidates the entire deliverable)

---

### Pitfall 2: SigLIP-2's attention-pooling head silently routes relevance away from patches

**What goes wrong:**
SigLIP-2-so400m does **not** use a CLS token. Its image embedding comes from a learned **MAP (multi-head attention pooling) head** — a probe query attends over the 729 patch tokens. LRP relevance for the image embedding flows backward *through this attention-pool*. If the SDPA rule distributes relevance through the pooling attention in a way that collapses onto a few patch tokens (or onto the learned probe rather than the patches), the resulting per-patch relevance map is structurally distorted before you ever upsample it. DynamicLRP covers SDPA generically but the paper never validates faithfulness on an attention-*pooling* head specifically (its ViT-b-16 result uses a CLS token; VGG has no attention).

**Why it happens:**
People assume "ViT coverage = SigLIP-2 coverage." But ViT-b-16's faithfulness number in Table 1 comes from a CLS-token architecture. The attention-pool head is a different relevance topology. The paper's SigLIP-2 row is *graph coverage* (it runs), not *faithfulness* (it's correct).

**How to avoid:**
- Identify the patch-token tensor *before* the attention-pool head and extract relevance there, not at the final embedding, so you map relevance at the patch grid rather than after pooling mixing.
- Run the perturbation/occlusion sanity check the paper itself uses (Section 5: occlude top-k highest-relevance 14×14 patches, measure similarity drop vs. occluding lowest-k / random). If MoRF≈LeRF≈random, the pooled relevance is not faithful for this head.
- Compare against a cheap independent baseline (e.g., gradient of similarity w.r.t. patch tokens, or attention-rollout) on 2–3 maps. They should *roughly* agree on coarse regions.

**Warning signs:**
Relevance map is extremely sparse (1–3 hot patches) or uniformly flat; perturbing top-relevance patches doesn't reduce image-text similarity more than random patches.

**Phase to address:** Attribution

---

### Pitfall 3: Patch→pixel upsampling artifacts mistaken for model behavior (the 27×27, 6-px-discard trap)

**What goes wrong:**
SigLIP-2-so400m-patch14-384 has a documented design quirk: **384 is not divisible by 14**. The model uses floor division with `padding="valid"` → a **27×27 = 729 patch grid**, and the **last 6 pixels of the right and bottom edges are discarded** (verified: HF model author confirmed this is an inattention bug they chose not to fix). Naively reshaping 729 relevance values to 27×27 and bilinearly upsampling to 384×384 (and then to the original Rumsey image, which may be thousands of px) produces: (a) a 6-px spatial offset/crop that misaligns the heatmap with map features near the right/bottom edge, and (b) smooth blob artifacts from 27×→full-res upsampling that a human will over-interpret as "the model found a soft region" when it is interpolation.

**Why it happens:**
Everyone writes `relevance.reshape(27,27)` then `F.interpolate(..., 'bilinear')` and overlays on the *full original image*. The 6-px discard and the aspect-ratio change from non-square Rumsey scans (resize-to-384 squashing) are invisible until you check alignment on a map with a sharp known landmark.

**How to avoid:**
- Map the heatmap onto the **exact 378×378 valid region** the model actually saw (or onto the preprocessed 384 tensor), not the raw original image. Document the resize/crop chain explicitly and invert it deliberately when overlaying on the original.
- Account for the SigLIP-2 preprocessor's resize (it squashes to 384×384, not letterbox) — the heatmap must be un-squashed back to original aspect ratio for the overlay to align.
- Prefer nearest-neighbor or block upsampling for an honest "this is patch-resolution" overlay in v1; offer a separate smoothed view but label it as interpolated.
- Alignment check: overlay the patch grid itself on one map; confirm grid cells land where expected and the right/bottom 6px are excluded.

**Warning signs:**
Heatmap consistently appears shifted toward top-left; hot regions don't sit on the map feature they should; everything looks like soft Gaussian blobs regardless of query.

**Phase to address:** Attribution (resolution/overlay logic), verified again in Sweep (contact-sheet makes misalignment visually obvious across many maps)

---

### Pitfall 4: Trusting eyeballed heatmaps with no negative control (the core deliverable risk)

**What goes wrong:**
The entire v1 deliverable is "human eyeballs a heatmap and judges it." Attribution methods are notorious for producing **confident, structured, plausible-looking maps that fail sanity checks** — Adebayo et al.'s model-randomization test shows many methods produce near-identical maps even with random weights. With purely visual evaluation (explicitly chosen in PROJECT.md), a broken pipeline (wrong target, distorted pooling, misaligned upsampling) yields heatmaps that *look* meaningful and get accepted. The project then scales a wrong loop across 1,544 maps.

**Why it happens:**
PROJECT.md scopes out quantitative metrics ("metrics add scope without payoff until the loop is trusted") — but the loop cannot be *trusted visually* without at least cheap controls. Humans pattern-match heatmaps onto map features even when attribution is noise.

**How to avoid:**
Add three near-zero-cost sanity controls to the single-slice loop *before* declaring Phase 1 done (these are not "metrics," they are correctness gates):
1. **Query-swap control:** same map, unrelated query → heatmap must change substantially.
2. **Model-randomization control (run once):** re-init SigLIP-2 vision weights randomly → heatmap should collapse to noise. If it doesn't, the attribution is not reading the model.
3. **Occlusion check (paper's own metric, on ~3 maps):** masking top-relevance patches must drop image-text similarity more than masking random patches.
These are the minimum bar for "trust the loop visually."

**Warning signs:**
Heatmap looks "reasonable" but you can't articulate why a *different* query would look different; no one has run a single negative control; "it highlights the city so it works" with no comparison case.

**Phase to address:** Attribution (gate Phase 1 completion on the three controls; do not scope these out)

---

### Pitfall 5: GPU memory blowup from LRP graph retention on so400m

**What goes wrong:**
DynamicLRP's Promise system works by **retrieving forward activations that autograd normally discards** — it deliberately retains/recovers tensors along the graph (paper §3.1–3.2). Peak VRAM rises sharply vs. a normal forward: the paper's own tables show ViT-b-16 going from ~1GB to ~1.6GB and VGG from ~0.9GB to ~2.3GB *for tiny models*. SigLIP-2-so400m is ~400M params with 27 transformer layers and 729 tokens — materially larger. A single-image LRP pass can OOM on a modest VM GPU even though a normal inference forward fits comfortably, leading people to wrongly conclude "the adaptation failed" when it's just memory.

**Why it happens:**
People size the VM for SigLIP-2 *inference* (small), not SigLIP-2 + full LRP graph retention (much larger). The paper's note that "GPU memory limits caused thrashing and prevented observation of true speed" for some methods is a direct warning.

**How to avoid:**
- Provision GPU memory for the *LRP pass*, not inference. Budget several× normal forward VRAM; measure peak on one image early.
- Process **one image at a time** (batch size 1) for attribution; never batch the sweep through LRP.
- Free the LRP graph/promises explicitly between maps in the sweep loop; call `torch.cuda.empty_cache()` and assert memory returns to baseline each iteration (a slow leak across 1,544 maps will OOM mid-sweep otherwise).
- so400m is the deliberate model ceiling (PROJECT.md) precisely because LRP overhead grows with size — do not let scope creep raise the model.

**Warning signs:**
Single-image works but sweep dies after N maps; VRAM never returns to baseline between iterations; `CUDA out of memory` only under LRP, not inference.

**Phase to address:** Attribution (single-image memory budget), Sweep (per-iteration cleanup + leak assertion)

---

### Pitfall 6: GCS ingestion of 1,544 Rumsey URLs treated as a happy-path script

**What goes wrong:**
The manifest has 1,544 `image_url`s pointing at the public David Rumsey collection. A naive "loop and download to GCS" fails on: dead/changed Rumsey URLs, rate-limiting / throttling on bulk hits to the Rumsey host, partial/truncated downloads written as if complete, and non-idempotent re-runs that re-download everything (or worse, overwrite good files with failed ones). The mirror is the foundation of reproducibility (Key Decision: "Mirror data to GCS for stability on ephemeral VMs"); a silently partial mirror means later sweeps quietly skip or error on maps and the "count down from high index N" semantics break.

**Why it happens:**
Ingestion feels trivial ("just copy files") so it gets the least design attention, but it is a 1,544-item network operation against a third-party server with no SLA, run from an ephemeral VM that can die mid-job.

**How to avoid:**
- **Idempotent + resumable:** check object exists in GCS (with expected size/content-length) before downloading; skip if present and valid. Re-running must converge, not restart.
- **Validate each download:** verify HTTP 200, content-type is an image, byte length matches `Content-Length`, file is a decodable image — *before* writing to GCS. Never write partial bytes to the canonical path (download to temp, validate, then atomic move/upload).
- **Throttle + retry with backoff:** polite concurrency and exponential backoff on 429/5xx; don't hammer the Rumsey host.
- **Manifest of outcomes:** record per-id status (ok / dead-url / decode-fail / skipped). The sweep must read this and only iterate over successfully-mirrored ids, preserving the descending-index ordering over *available* maps.

**Warning signs:**
Re-running ingestion re-downloads everything; sweep throws FileNotFound on some ids; image count in GCS ≠ manifest count with no record of which are missing; corrupt/zero-byte objects in the bucket.

**Phase to address:** Ingestion (idempotency, validation, outcome manifest are acceptance criteria, not nice-to-haves)

---

### Pitfall 7: Ephemeral-VM non-reproducibility (model/version/seed drift)

**What goes wrong:**
The VM is ephemeral; it can be recreated at any time. If SigLIP-2 weights are pulled from HuggingFace at runtime (instead of the GCS model mirror), or the DynamicLRP repo is `pip install git+...@main` without a pin, or no seeds are set, then two runs of "the same map + query" produce different heatmaps — and there is no way to tell whether a heatmap change came from a different map, the model updating, the LRP code changing, or nondeterminism. This destroys the core loop's value (compare maps/queries) because the baseline isn't fixed.

**Why it happens:**
keeinlev/dynamicLRP is an actively moving research repo (default branch `master`, last pushed 2026-05-06 — very recent, pre-publication, "under review"). Pulling `@master` means the attribution algorithm can change under you between VM rebuilds. HF model files can also change. PROJECT.md already mandates mirroring model+data to GCS for exactly this reason — but the LRP *code* dependency is the unpinned one.

**How to avoid:**
- Load SigLIP-2 **only from the GCS model mirror** (already a Key Decision) — never fall back to HF at runtime.
- **Pin DynamicLRP to a specific commit SHA**, not a branch; vendor it or record the SHA in requirements and in run metadata.
- Set and record seeds (torch, numpy, cuda) and `torch.use_deterministic_algorithms` where feasible; SDPA may have nondeterministic kernels — record this caveat.
- Stamp every produced heatmap with: map id, query, model mirror version, DynamicLRP commit SHA, LRP config. The contact sheet should make provenance inspectable.

**Warning signs:**
Same (map, query) gives visibly different heatmaps across runs; no commit SHA recorded anywhere; "it worked last week" with no way to reconstruct last week's environment.

**Phase to address:** Ingestion (model mirror + pinned deps in environment setup), enforced in Attribution and Sweep (provenance stamping)

---

### Pitfall 8: Resize/normalization mismatch between ingestion and the SigLIP-2 processor

**What goes wrong:**
Rumsey scans are large, varied-aspect-ratio images. SigLIP-2's image processor expects a specific pipeline (resize to 384×384 — squashing, not letterboxing — and SigLIP-specific normalization, typically rescale to [-1,1], *not* ImageNet mean/std). If ingestion pre-resizes/crops images "to save space," or if the loader applies the wrong normalization, the model sees out-of-distribution inputs and attributions are meaningless — but the pipeline runs and produces plausible-looking maps (Pitfall 4 again).

**Why it happens:**
People reuse a generic CLIP/ImageNet transform out of habit; SigLIP uses different normalization constants. Pre-resizing during ingestion seems like a harmless optimization but discards information and bakes in a wrong transform.

**How to avoid:**
- Mirror **original-resolution** images to GCS; do all resizing/normalization at load time via the **official SigLIP-2 `AutoProcessor`/image processor**, never a hand-rolled transform.
- Assert preprocessing matches the model card (input range, size, no letterbox) on one image; print the tensor min/max/shape and confirm against SigLIP-2 expectations.

**Warning signs:**
Image-text similarities are near-constant across very different (map, query) pairs (sign of OOD inputs); tensor value range isn't what the SigLIP-2 card specifies; attributions don't respond to query at all.

**Phase to address:** Ingestion (store originals, don't pre-transform), Attribution (use official processor, assert input contract)

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Propagate from image-embedding norm instead of query-similarity | "Heatmap appears" fast | Entire deliverable is query-independent and wrong; invalidates sweep | **Never** |
| Skip sanity controls, judge purely by eye | Faster Phase 1 sign-off | Scale a broken loop to 1,544 maps; weeks lost | **Never** (these are correctness gates, not metrics) |
| `pip install git+...dynamicLRP@master` | One-line setup | Attribution algorithm changes under you between VM rebuilds; irreproducible | Only for a throwaway first spike, must pin before Phase 1 done |
| Bilinear upsample 27×27→full-res, overlay on raw original | Pretty heatmap | 6-px misalignment + interpolation mistaken for model behavior | Acceptable as a *labeled* secondary view; never as the primary judged artifact |
| Pre-resize images during ingestion to save GCS space | Smaller bucket | Wrong/lossy transform baked in; can't change preprocessing later | Never (storage is cheap; originals are the asset) |
| Non-idempotent download loop | Simplest code | Ephemeral VM dies → restart from zero; partial mirror undetected | Never at 1,544 scale |
| Batch images through LRP for sweep speed | Faster sweep | OOM on so400m + LRP graph retention | Never; batch size 1 for attribution |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| keeinlev/dynamicLRP | Assume "SigLIP-2 in coverage table = validated/correct" | Coverage = it runs; faithfulness on SigLIP-2 *as contrastive encoder* is unvalidated — verify with occlusion + randomization controls |
| keeinlev/dynamicLRP | Track `master` branch | Pin to commit SHA; record SHA in run metadata (repo is pre-publication, actively changing) |
| SigLIP-2 (HuggingFace) | Pull weights at runtime; hand-rolled CLIP/ImageNet transform | Load from GCS mirror only; use official SigLIP-2 processor (rescale to [-1,1], 384 squash) |
| SigLIP-2 attention-pool head | Extract relevance at final image embedding | Extract at patch tokens *before* the MAP pooling head |
| David Rumsey image URLs | Loop-and-download, assume all 200 OK | Validate (200 + image content-type + byte length + decodable), throttle, backoff, record per-id outcome |
| GCS bucket | Overwrite canonical path with in-progress download | Download to temp, validate, atomic upload; check-exists-and-valid before re-downloading (idempotent) |
| GCP VM (ephemeral) | Assume environment persists; no provenance | Pin everything; stamp heatmaps with map id, query, model version, LRP SHA, config |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| LRP graph/promise VRAM retention | OOM under LRP but not inference | Size GPU for LRP pass; batch size 1; free graph per iteration | Single so400m image, or partway through sweep |
| Memory leak across sweep iterations | VRAM never returns to baseline; dies after N maps | `empty_cache()` + assert baseline memory each iteration | ~tens-to-hundreds of maps into a 1,544 sweep |
| Serial ingestion of 1,544 URLs, no concurrency | Ingestion takes hours; VM idle-billed | Bounded concurrent downloads with backoff | At full manifest scale |
| Re-running ingestion redownloads everything | Hours wasted on every VM rebuild | Idempotent skip-if-present-and-valid | Every ephemeral VM recreation |
| LRP recompute per (map,query) with shared image | Sweep is queries×maps slow | Cache image-tower forward; vary only text target where the LRP method allows | Sweep grid grows (maps × queries) |

## Security Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Decoding arbitrary downloaded images without limits | Decompression bomb / huge Rumsey scans exhaust VM memory during ingestion | Cap max image dimensions/bytes; validate Content-Length before fetch; use a hardened image lib |
| Trusting third-party (Rumsey) content blindly | Malformed/unexpected content treated as valid map | Validate content-type + decodability before mirroring; quarantine failures |
| Credentials/ADC assumed always present | Sweep dies mid-run on token expiry on long jobs | Verify GCS auth at job start; fail fast with a clear message, not 1,000 lines in |

## UX Pitfalls (notebook / human-in-the-loop)

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Heatmap shown with no provenance | Researcher can't tell which map/query/model produced it; can't trust comparisons | Caption every tile: map id, query, model version, LRP SHA |
| No side-by-side query control in the view | Human pattern-matches noise onto map; false confidence | Show same-map/different-query pairs so query-sensitivity is visually obvious |
| Smoothed heatmap presented as ground truth | Interpolation artifacts read as model findings | Default to patch-resolution view; smoothed view explicitly labeled "interpolated" |
| Contact sheet with no ordering semantics | Loses the "rich maps = high index" signal | Sweep counts *down from high N*; label index on each tile |

## "Looks Done But Isn't" Checklist

- [ ] **LRP adaptation:** Runs without error — but verify it passes occlusion (MoRF vs LeRF vs random) and model-randomization controls, not just "produces a map"
- [ ] **Query-driven attribution:** Produces a heatmap — but verify the heatmap *changes substantially* when the query changes (else it's image saliency, not query relevance)
- [ ] **Patch→pixel overlay:** Heatmap overlays the map — but verify alignment on a sharp known landmark and that the 6-px right/bottom discard + aspect-ratio un-squash are handled
- [ ] **GCS ingestion:** "1,544 images copied" — but verify a per-id outcome manifest exists, re-run is idempotent, and no zero-byte/corrupt objects
- [ ] **Reproducibility:** "It works on the VM" — but verify same (map,query) → same heatmap across a fresh VM, with DynamicLRP pinned to a SHA
- [ ] **Sweep:** "Runs over the subset" — but verify memory returns to baseline each iteration and it only iterates successfully-mirrored ids in descending index order
- [ ] **Preprocessing:** "Image loads" — but verify it uses the official SigLIP-2 processor with correct ([-1,1]) normalization, not an ImageNet transform

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Wrong backprop target discovered after sweep | HIGH | Redefine target as query-similarity; re-run entire sweep; all prior heatmaps discarded |
| Attention-pool relevance distortion | MEDIUM-HIGH | Extract at pre-pool patch tokens; re-validate with occlusion; partial re-run |
| Upsampling misalignment | LOW-MEDIUM | Fix overlay/un-squash logic; re-render from cached relevance arrays (no model re-run if relevance saved) |
| Partial GCS mirror | LOW | Re-run idempotent ingestion; outcome manifest fills gaps; no model work lost |
| Unpinned LRP drift | MEDIUM | Pin SHA, rebuild env, re-run sweep to restore comparability |
| Sweep OOM mid-run | LOW | Add per-iteration cleanup + checkpoint/resume from last completed id |
| OOD preprocessing | MEDIUM | Switch to official processor, re-run (relevance must be recomputed) |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Wrong backprop target (contrastive) | Attribution | Query-swap test: unrelated query changes heatmap substantially |
| 2. Attention-pool relevance distortion | Attribution | Occlusion test (top-k vs random) drops similarity; extract at pre-pool tokens |
| 3. Patch→pixel upsample/discard artifacts | Attribution (verified in Sweep) | Patch-grid overlay aligns on a known landmark; 6-px + aspect handled |
| 4. Trusting eyeballed heatmaps | Attribution (gates Phase 1) | Query-swap + model-randomization + occlusion controls all pass before sign-off |
| 5. LRP VRAM blowup | Attribution (budget), Sweep (cleanup) | Single-image peak VRAM measured; memory returns to baseline each sweep iteration |
| 6. Non-idempotent / partial GCS ingest | Ingestion | Re-run converges; per-id outcome manifest; no corrupt objects |
| 7. Ephemeral-VM irreproducibility | Ingestion (env), Attribution+Sweep (provenance) | Fresh VM reproduces same (map,query) heatmap; LRP commit SHA recorded |
| 8. Resize/normalization mismatch | Ingestion (store originals), Attribution (official processor) | Tensor range matches SigLIP-2 card; attributions respond to query |

## Sources

- **arXiv 2512.07010** (DynamicLRP paper, "Accepted to Principled Design for Trustworthy AI workshop at ICLR 2026", under review) — inspected full 27-page PDF. Table A.4: SigLIP-2-So400m-base14-384 100% node coverage as *Vision* modality only; no faithfulness entry in Table 1. Appendix F: covered-operations list (SDPA, NativeLayerNorm, GELU, Softmax, etc.). §3.1–3.2: Promise system retains/recovers forward activations (VRAM implication). §5/Tables A.1–A.2: peak-VRAM growth under LRP; "GPU memory limits caused thrashing." Line 278: single `target_node` requirement. [HIGH]
- **github.com/keeinlev/dynamicLRP** — repo metadata: default branch `master`, last pushed 2026-05-06, description confirms operation-level autograd-graph LRP. README not directly readable (404 on raw path) — config flags (`use_gamma`, `use_z_plus`, `relevance_filter`) from GitHub landing summary. [MEDIUM — pin-to-SHA recommendation stands regardless]
- **huggingface.co/google/siglip-so400m-patch14-384 discussions/4** — model author (@giffmana) confirms 384 not divisible by 14, floor-div → 27×27=729 patches, `padding="valid"` discards 6 edge px, "inattention mistake," not fixed. [HIGH]
- **huggingface.co/google/siglip2-so400m-patch14-384**, **arxiv.org/html/2502.14786** (SigLIP-2 paper), emergentmind SigLIP-2 vision encoder — 27-layer ViT, MAP attention-pooling head (no CLS token), sigmoid (not softmax) image-text loss. [HIGH]
- **Chefer et al. CVPR 2021 (Transformer Interpretability Beyond Attention)**, **AttnLRP arXiv 2402.05602**, **Adebayo et al. sanity-checks** (via search) — basis for model-randomization + occlusion controls and contrastive-target caveats. [MEDIUM]

---
*Pitfalls research for: dynamic-LRP-on-SigLIP-2 query-driven map attribution*
*Researched: 2026-05-18*
