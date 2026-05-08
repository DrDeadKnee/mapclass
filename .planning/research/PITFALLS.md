# Pitfalls Research

**Domain:** Multi-source-trained dense-prediction VL+OCR+heads model with auto-georef bootstrap loop
**Researched:** 2026-05-08
**Confidence:** HIGH on the ML-literature pitfalls (well-documented), MEDIUM on the project-management pitfalls (single-author specific), MEDIUM-HIGH on OSM licensing (clear OSMF guidance, MapClass-specific application is the new bit)

## Scope and framing

This document complements `.planning/codebase/CONCERNS.md` (data-pipeline layer, brownfield) by covering the **new components** in v1: the small-VL backbone + dense seg heads + rotation-invariant OCR + auto-georef bootstrap loop + PaliGemma benchmark variant + OSM third source. Data-pipeline pitfalls (licensing, taxonomy alignment in remap tables, broad exception swallowing in fetchers) are in CONCERNS.md and are not duplicated here — when relevant they are cross-referenced.

Each pitfall maps to one or more `PROJECT.md` REQ-IDs and is tagged **CRITICAL / HIGH / MEDIUM**. The downstream-consumer block at the top of this prompt asks for a concrete **warning sign** (something visible in logs, eval output, or spot-checks) and a **prevention strategy** (specific, actionable). Both are included.

---

## Critical Pitfalls

### Pitfall 1: Bootstrap-loop error compounding (v0 → v1 gets worse, not better)

**Severity:** CRITICAL
**REQ-IDs:** GEOREF-02, TRAIN-02, EVAL-01, also overlaps EVAL-03

**What goes wrong:**
v0 trained on the small registered set produces noisy predictions. Auto-georef uses those noisy predictions to register additional Rumsey maps. The newly-registered maps' labels (derived from WorldCover/DEM via `make_labels()`) are systematically biased toward whatever v0 already gets right — coastlines if v0 is good at water, but pixel-aligned only where v0's mistake-pattern coincides with the WorldCover reference. v1 trains on `v0-correct + v0-mistake-aligned` data and amplifies its own errors, exactly the [confirmation-bias dynamic Arazo et al. 2019 documented for pseudo-labeling](https://arxiv.org/abs/1908.02983). Result: v1 NLL is **worse** than v0 NLL on the held-out set, and the bootstrap was net-negative.

**Why it happens:**
- Self-training generates labels from the model itself; errors are correlated with the model's blind spots, so retraining doesn't fix them, it entrenches them.
- Auto-georef quality on heavily-stylized 17th-century maps is weakest exactly where forest cover and built-up have changed most since 1700 — those are also the classes the loss-weight schema downweights, so the bootstrap silently re-introduces noise that the weighting was designed to suppress.
- The `confidence: 0.5` multiplier on `rumsey_bootstrapped` weights (architecture Pattern 2) is a guess; it might not be aggressive enough.

**Warning signs:**
- v1 held-out NLL > v0 held-out NLL on the **same** held-out split (this is the kill-switch metric).
- Per-class NLL on `trees`, `cropland`, `built_up` regresses from v0 to v1, while `water` and `bare_sparse` improve — that's the signature of bias amplification on the classes WorldCover disagrees with for pre-modern maps.
- Spot-check renders on Tolkien/Westeros: v1 hallucinates more confidently in regions v0 was uncertain about (the model "decides" on a wrong answer instead of staying diffuse). Visible as overconfident-but-wrong colored regions where v0 was a noisy fade.
- Auto-georef GCP-residual histogram for the bootstrap-registered set has a long right tail (high-residual maps got registered anyway because no kill-threshold was set).

**Prevention strategy:**
1. **Mandatory v0-vs-v1 NLL comparison gate** — implement D-5 from `.planning/research/FEATURES.md` (bootstrap-loop quality monitor): run `mapclass.eval` on **both** v0 and v1 against the **same** held-out split (the one frozen by `TS-10` deterministic splits). If v1 NLL is not better than v0 NLL by at least 2% (a small but non-zero margin), abort the bootstrap and ship v0. This is a 30-line addition to the eval script.
2. **GCP-residual hard threshold** in `scripts/georef.py bootstrap` — reject any auto-registration whose mean GCP residual exceeds a configurable threshold (e.g., 5 km at the source resolution). Log the rejection rate; if >50% of unregistered maps reject, the bootstrap is degenerate and should not feed v1.
3. **Per-source bootstrap-confidence multiplier** — keep `rumsey_bootstrapped` weights at ≤0.5 of `rumsey_registered` (already in architecture Pattern 2); make this configurable so the multiplier can be reduced further if D-5 shows regression on specific classes.
4. **No bootstrap-on-bootstrap** — explicitly forbid v1 from auto-registering more maps for a hypothetical v2. Bootstrap depth = 1 by contract; reconsider only if v1 demonstrably beats v0.

**Recovery if it occurs:**
- Ship v0. The whole point of the kill-switch is that v0 is a valid v1 ship target.
- Diagnose with per-class NLL deltas to identify which classes the bootstrap poisoned, then either drop those classes from the bootstrapped weight row or reduce the bootstrap-confidence multiplier and retrain.

---

### Pitfall 2: PaliGemma-vs-small-backbone comparison contamination (EVAL-03 invalid)

**Severity:** CRITICAL
**REQ-IDs:** EVAL-03, MODEL-04, TRAIN-01, TRAIN-02

**What goes wrong:**
EVAL-03 is the **bellwether**: it answers "is the small backbone good enough, or do we need to reopen MODEL-01?" If the PaliGemma run sees different data, different loss weighting, a different held-out split, a different evaluation seed, or a different preprocessing pipeline than the small-backbone run, the NLL-gap is no longer a pure backbone comparison — it's a confound. The decision the gap drives (re-open MODEL-01 or ship as planned) is then based on a contaminated signal, and the worst case is that PaliGemma "looks much better" purely because it saw a more favourable preprocessing pipeline, leading to an unjustified backbone re-evaluation that wastes weeks.

**Why it happens:**
- The two backbones have **different native input sizes** (PaliGemma = 224/448/896 mix, small VL likely 384). The instinct is to use each backbone's native size — but that means each model sees different effective receptive fields and patch grids, conflating backbone-size with backbone-quality.
- The two backbones have **different preprocessing pipelines** (PaliGemma's `processor` does specific resize+normalize+pad; SigLIP's does a different normalize). It's tempting to use each model's "default" processor, but these are not identical interventions on the data.
- LoRA vs full fine-tune is a separate axis; running PaliGemma with LoRA-only and the small backbone full-finetune mixes capacity and adaptation budget into the comparison.
- Evaluation seed drift: re-running `mapclass.eval` with a re-built held-out set (because someone re-ran the dataset build and `splits.json` was regenerated) silently changes the evaluation set between the two runs.

**Warning signs:**
- Diff of `eval_report.json` between the two runs shows different sample-IDs in the held-out set (or different counts).
- The two `loss_weights.yaml` files (or git history thereof) diverged between the two training runs.
- Sample-prediction images at fixed validation iterations look at *different* validation images for the two runs.
- NLL-gap is suspiciously large (e.g., > 1.0 nats per pixel) — that's a level of difference that's almost never just backbone, suggesting the comparison is contaminated.
- Per-class NLL gap is dominated by one class (e.g., PaliGemma is 0.5 nats better on `built_up` and indistinguishable elsewhere) — points to a preprocessing or augmentation difference rather than a backbone-quality difference.

**Prevention strategy:**
1. **Single eval contract** — both runs MUST use the *same* `mapclass.eval --split heldout` invocation against the same `splits.json` (committed file, hashed in checkpoint metadata per TS-8) and the same `loss_weights.yaml` (also hashed). Write a one-line script that asserts the splits-file hash and the weights-file hash match between the two checkpoints before printing the comparison.
2. **Same data, same augmentation, same loss weighting** — the only thing that varies is `Backbone` (per architecture Pattern 3). The training config differs in the `backbone:` field only; `data:`, `loss_weights:`, `augmentation:`, `optimizer:` should be byte-identical between v0/v1 small-backbone configs and v0/v1 PaliGemma configs (4 configs, 2 axes: backbone × bootstrap-stage).
3. **Same fine-tune budget** — pick LoRA-or-full once, apply to both. The LoRA-PaliGemma vs full-small-backbone comparison is *not* an EVAL-03; it would be a confounded ablation. If asymmetric is unavoidable for VRAM (PaliGemma must be QLoRA, small backbone can be full-finetune), document this loudly as the **one** axis where the comparison is not pure, and report it in the eval header.
4. **Pre-register the comparison protocol** — before either training run starts, write `EVAL-03_protocol.md` (one page) listing the held-out split hash, loss-weights hash, augmentation hash, fine-tune approach, evaluation seed, and the kill-switch criterion ("if PaliGemma is more than X nats better than small, MODEL-01 is reconsidered"). Commit it. Reviewers (the owner, future-Claude) can then verify the comparison wasn't post-hoc rationalised.
5. **Resolution normalisation** — if backbones have different native input sizes, the `Predictor`'s output spatial shape (TS-11) is what matters; both NLLs are computed at the *output* resolution, which is the input image's pre-pad shape. Verify both heads emit the same H,W per sample. The CONCERNS.md TESTING gap-item 6 (loss-weight schema) is one place this could silently drift.

**Recovery if it occurs:**
- Re-run both backbones under the corrected protocol. The cost is mostly compute; the small backbone is cheap, PaliGemma is the expensive half.

---

### Pitfall 3: Per-source loss-weight schema silently misaligned with canonical taxonomy index

**Severity:** CRITICAL
**REQ-IDs:** DATA-03, DATA-06, MODEL-03, TRAIN-01, TRAIN-02

**What goes wrong:**
`mapclass/configs/loss_weights.yaml` is keyed by class **name** (`water`, `trees`, ...) but the loss in `mapclass.model.losses.weighted_nll` consumes per-pixel labels by class **index** (0..8 + 255). The mapping name→index lives in `scripts/biome_mapping.py:LANDCOVER_CLASSES`. Three places hold parallel taxonomy state: `LANDCOVER_CLASSES`, `BIOME_TO_LANDCOVER_NAME` (synthetic remap), `WC_REMAP` (WorldCover remap). If anyone renames a class, reorders the list, or adds a class without updating all parallel tables, the loss silently weights `built_up` predictions using the `flooded_wetland` weight (or worse, a key-miss falls through to a default of 0.0 or 1.0). The model trains for hours and the loss-curve looks fine, but the per-class outputs are systematically wrong.

This is `.planning/codebase/TESTING.md` gap-item 6 made worse by adding two new sources (OSM, bootstrapped-historical) and the YAML schema. Each new source row is one more place a typo can creep in.

**Why it happens:**
- Three independent remap tables (TESTING.md gap-item 1) maintained in three files.
- Adding the YAML adds a fourth table (`loss_weights.yaml`) which itself depends on the canonical names.
- No tests (PROJECT.md accepted risk).
- Cross-source schema drift: synthetic uses `cropland: 0.0` (absent), OSM uses `cropland: 0.7` — typo turns one into the other and nothing fails.

**Warning signs:**
- Per-class NLL in eval output has a class with NLL = 0.0 exactly (loss was masked entirely; weight is 0 instead of the intended value).
- Per-class NLL has a class with NLL much higher than its `valid` neighbours (weight is 1.0 instead of intended downweight, model is being pushed toward wrong labels).
- Training loss curve is smooth but eval qualitative renders show a class that is "absent" — model never predicts it (weight was 0 by accident).
- KeyError at training startup is the *good* case; silent fallback to default is the bad case.
- Per-source confusion matrix on eval: a class is confidently mis-predicted in only one source (the one whose weights are misaligned).

**Prevention strategy:**
1. **Strict schema validator at training startup**, in `mapclass.data.loss_weights.LossWeights.load()`:
   - Assert that every class name in the YAML is in `LANDCOVER_CLASSES`. Fail loudly with a list of unknown / missing names.
   - Assert that every source mentioned has all 9 land-cover classes (no implicit defaults). If a class is intentionally absent from a source, require explicit `0.0` (don't allow KeyError-as-zero).
   - Assert that the `inherit:` directive on `rumsey_bootstrapped` resolves to a known parent.
   - This is ~30 lines of code, replaces the gap-item-6 "no test" risk with one schema check.
2. **Single canonical taxonomy module exports a frozen dict** — `LANDCOVER_CLASSES` becomes the single source; loss-weights, eval reporter, and the inference module's `taxonomy` field (TS-11) all read from it. Renames touch one file.
3. **Hash the taxonomy** — embed a sha256 of `(LANDCOVER_CLASSES, TOPO_CLASSES, loss_weights.yaml)` into the checkpoint metadata (TS-8). At `mapclass.eval` time, refuse to evaluate if the in-memory taxonomy hash doesn't match the checkpoint's. This catches the "you renamed a class between train and eval" case.
4. **Per-source weight-table lint** — a one-screen script (`scripts/lint_loss_weights.py`) that prints the full (source × class) matrix to stdout. Eyeballing 4 rows × 9 columns once before each training run is enough to catch a typo.

**Recovery if it occurs:**
- Detect via per-class NLL in eval. Fix the YAML or the canonical list. Retrain.
- Cost: one full training run wasted (worst case for v0; v1 is cheaper because v0 catches it earlier).

---

### Pitfall 4: Tokenizer / processor mismatch when swapping backbones

**Severity:** CRITICAL
**REQ-IDs:** MODEL-01, MODEL-04, TRAIN-01

**What goes wrong:**
Each VL backbone ships with its own image processor (resize policy, normalize stats, padding behaviour, channel order). A common bug: `Predictor.predict()` uses `transforms.Compose` from torchvision (with ImageNet stats) but the backbone was trained with `AutoProcessor.from_pretrained()` (with SigLIP-specific stats — mean ≈ 0.5, std ≈ 0.5, not ImageNet's 0.485/0.456/0.406). The model trains fine because the training loop happens to use the right processor; inference uses the wrong stats and outputs are systematically biased — most often manifesting as washed-out probability maps where everything looks "vaguely possible" and argmax wanders.

PaliGemma's processor is even more particular: it normalises to (-1, 1), uses bicubic resize, and pads to a square (448 or 896). Swapping in PaliGemma without using its `PaliGemmaProcessor` produces silently wrong inputs.

The HuggingFace forums document this as a recurring fine-tuning error: ["ValueError: Input image size doesn't match model"](https://discuss.huggingface.co/t/fine-tuning-image-transformer-on-higher-resolution/22623) is the *visible* failure; silently-wrong-normalize is the *invisible* failure that's harder to debug.

**Why it happens:**
- The Backbone ABC (`mapclass.model.backbone`) is a swap surface, but it doesn't enforce that the swap also includes the processor. If someone instantiates `Backbone` and then writes `image = (PIL_to_tensor(img) / 255.0)` in the training loop, the backbone's expected input space is violated and there's no exception.
- Mixed-precision adds a further trap: SigLIP-2 expects fp32 input that gets cast to bf16/fp16 internally; if the user pre-casts to fp16 *before* the processor, normalize stats compose with the dtype cast and produce subtly different outputs than `processor + autocast`.

**Warning signs:**
- v0 training loss is fine but eval NLL is implausibly bad (e.g., > 3 nats per pixel where ~1 nat is expected).
- Sample prediction images from training-time logging look fine, but `mapclass.infer.predict()` on the same image gives different output than the training-time logger did. This indicates the inference path uses different preprocessing than training.
- Argmax of land_cover output is uniform across the image (one class everywhere), or wanders chaotically pixel-to-pixel — both signatures of inputs being far outside the backbone's training distribution.
- PaliGemma checkpoint inference produces sensible outputs only when image is exactly 448×448; small changes in input shape produce nonsense — indicates the processor's pad-to-square wasn't run.

**Prevention strategy:**
1. **Backbone owns the processor.** Extend `Backbone` ABC to require `preprocess(image: PIL.Image) -> Tensor` returning the fully-prepped tensor in the right dtype/shape. Concrete subclasses wrap `AutoProcessor.from_pretrained(...)`. The training loop and inference path both call `backbone.preprocess()` — never re-implement.
2. **Single integration test at the swap boundary** — even though the project punts tests, write *one* assertion: load the trained checkpoint via `mapclass.infer.load_model`, push a fixed reference image through, hash the output. Compare against a reference hash committed alongside the checkpoint. If anyone changes the preprocessing, the hash diverges and the load_model call fails. ~10 lines.
3. **Preprocessor identity in checkpoint metadata** (TS-8) — embed `model.config.image_processor_dict` (or equivalent) into the safetensors metadata header. At inference time, refuse to load a checkpoint whose preprocessor config doesn't match the runtime processor's config.
4. **Mixed-precision discipline** — use `torch.autocast` around the **forward pass only**. Don't cast inputs manually before `backbone.preprocess()`. Log the dtype of the tensor entering the backbone at startup.

**Recovery if it occurs:**
- Identify the discrepancy (compare training-time vs inference-time preprocessing call stacks).
- Fix the inference path to use `backbone.preprocess()`.
- No retraining needed if the bug is on the inference side; full retrain if the bug was on the training side and no checkpoint exists with the correct preprocessor.

---

### Pitfall 5: Training-eval contamination via held-out split drift

**Severity:** CRITICAL
**REQ-IDs:** EVAL-01, EVAL-03, TRAIN-01, TRAIN-02

**What goes wrong:**
The bootstrap loop expands the training set. Without a deterministic, frozen split policy, the held-out set can leak: a Rumsey map that was held out in v0 evaluation gets auto-registered during the bootstrap, lands in `data/historical/raw/georeferenced/`, gets picked up by `build_historical_dataset.py build`, and silently becomes a *training* sample for v1. v1 then evaluates on a smaller-than-expected held-out set, possibly including some samples that are now also in training. EVAL-01's NLL number is contaminated; EVAL-03's PaliGemma comparison inherits the contamination.

This is the segmentation-equivalent of the [LLM benchmark contamination](https://arxiv.org/html/2402.03927) problem, except the threat vector is internal (pipeline reshuffling), not external (training-corpus crawls).

**Why it happens:**
- The bootstrap loop adds new samples to the training pool at the same time as the held-out set is defined.
- If the split is computed by `glob` order over the dataset directory, adding samples to the directory shuffles the split.
- If the split is computed at training start (deterministic seed but seed not pinned to *content*), the split changes when content changes.

**Warning signs:**
- The held-out sample-ID list in `splits.json` for v1 differs from v0's by more than the bootstrap-set size (it should differ by zero — bootstrap adds train-only).
- Some of the newly-bootstrapped sample-IDs are also in the held-out list (overlap > 0, should be 0).
- v1 eval NLL improves dramatically over v0 NLL on a per-sample basis but only on samples that happen to also be in the bootstrap set.

**Prevention strategy:**
1. **Split by sample-ID hash, not glob order.** Each sample has a stable ID (e.g., `luna_4567` for Rumsey, `synth_0001` for Azgaar); compute `hash(sample_id) mod 10` and use buckets 0-7 for train, 8 for val, 9 for test. The split is a pure function of sample-ID, immune to dataset-build reordering. (TS-10 in FEATURES.md; this is the implementation.)
2. **Held-out IDs locked at v0 build time.** Generate `splits.json` once when v0 training starts; commit it. v1 training reads the *same* `splits.json`. The bootstrap-added samples either land in train (bucket 0-7) or are rejected from the dataset entirely if their hash bucket is 8 or 9 — implement a "bootstrap can only add to train" filter in `scripts/georef.py bootstrap` that consults `splits.json` and skips samples destined for val/test.
3. **Eval-time assertion** — `mapclass.eval` asserts that every held-out sample-ID in its iteration is *not* in the dataset's train-source manifest. Fail loudly on overlap.
4. **Source-subtype matters too** — `rumsey_registered` and `rumsey_bootstrapped` are *not* the same source. A sample that exists in `rumsey_registered` must not also exist in `rumsey_bootstrapped` (it would appear in both training mixes). Enforce: a sample-ID is in exactly one source-subtype.

**Recovery if it occurs:**
- Recompute the held-out split by hash, identify which samples leaked into training, drop them from training, retrain. Cost: one full v1 training run. (Not recoverable post-hoc by re-evaluation — the model has seen the held-out samples.)

---

## High Pitfalls

### Pitfall 6: PaliGemma-3B VRAM blowup during training

**Severity:** HIGH
**REQ-IDs:** MODEL-04, TRAIN-01, TRAIN-02

**What goes wrong:**
PaliGemma-3B is ~6 GB FP16 weights + ~6 GB FP16 gradients + ~12 GB FP32 Adam moments + activations. A naive full-finetune blows past 40 GB on a single A100. Even LoRA can OOM if the seg-head adds aggressive feature-pyramid skip-connections that hold large activations. The cloud VM provisioning lives in a separate repo (`PROJECT.md` Constraints), so an OOM during a long training run is a remote failure with high diagnostic latency — by the time you SSH in to investigate, the run has been dead for hours.

The [Modal blog](https://modal.com/blog/how-much-vram-need-fine-tuning) and [unsloth issue tracker](https://github.com/unslothai/unsloth/issues/4504) document that "advertised" VRAM numbers (e.g., "QLoRA fits 7B in 12 GB") often understate by 2–3x once activations, gradient accumulation, and the segmentation head's feature-map intermediates are added.

**Why it happens:**
- Activation memory scales with input resolution × batch size. PaliGemma at 448x448 with batch 4 holds ~10x the activations of the same model at 224x224 batch 1.
- The two seg heads + OCR head all run forward simultaneously; their feature pyramids hold activations during backprop.
- Mixed-precision `autocast` doesn't free activations early; gradient checkpointing does but adds compute.
- bitsandbytes quantization to 4-bit reduces weight memory but **not** activation memory.

**Warning signs:**
- VRAM usage at training start (before backward pass) already > 30 GB on the target VM.
- First backward pass OOMs ~5 minutes into training.
- Training succeeds with batch_size=1 but fails at batch_size=2 with the same gradient accumulation — pure activation issue, not weights.
- Forward pass succeeds, backward pass OOMs — indicates activations rather than weights.

**Prevention strategy:**
1. **Run a one-batch dry-run on the target VM before launching the full training.** Forward + backward on a single batch at the configured input resolution. If it OOMs, fix before scheduling the long run. Cost: 5 minutes of VM time saves 5 hours of wasted run time.
2. **Default to QLoRA + gradient checkpointing for PaliGemma.** Start conservative; if VRAM has slack, relax. Per the [VRAM guidance for 2026](https://vrlatech.com/how-much-vram-do-you-need-for-llm-fine-tuning-in-2026/), QLoRA can fit 3B + activations in 16 GB if checkpointing is on; full finetune is 40+ GB territory.
3. **Smaller input resolution for PaliGemma initially** — train at 224 first, scale to 448 only after validating the loss curve is sane. Position embeddings interpolate (per [HF ViT docs](https://discuss.huggingface.co/t/fine-tuning-image-transformer-on-higher-resolution/22623)).
4. **Log VRAM every 100 steps** (`torch.cuda.memory_allocated()`) — captures slow leaks; an upward trend across iterations indicates a memory leak (e.g., a tensor accidentally retaining graph). The training-time logging requirement (TS-7) covers this.
5. **The PaliGemma run is a *benchmark*, not the ship target.** If VRAM is tight, accept lower batch size or smaller input. The bellwether (EVAL-03) needs a converged number, not the optimal number — a slightly under-trained PaliGemma still tells you whether the small backbone is leaving signal on the table.

**Recovery if it occurs:**
- Reduce batch / resolution / fine-tune scope. The QLoRA baseline is a safe fallback.
- The non-PaliGemma small-backbone path is unaffected.

---

### Pitfall 7: Frozen-vs-LoRA-vs-full-finetune trade-off mis-pick for small VL backbone

**Severity:** HIGH
**REQ-IDs:** MODEL-01, TRAIN-01

**What goes wrong:**
The small VL backbone (SigLIP-2-B / DINOv2-S — STACK choice deferred) was pretrained on natural images. Stylized cartography is an extreme out-of-distribution shift. **Fully frozen** backbone means the seg heads have to do all the adaptation work on features that were never trained to discriminate "fantasy parchment forest" vs "synthetic shrubland" — under-fits. **Full fine-tune** has the capacity but a 86M-param ViT trained on a few thousand stylized images is destined to overfit to the synthetic style and forget the natural-image priors that help on satellite-style maps. **LoRA** is the middle path but the rank choice is non-trivial — too low (rank 4) is effectively-frozen, too high (rank 64) approaches full-finetune.

The [recent VLA fine-tuning literature](https://arxiv.org/html/2512.19219) and [LoRA empirical work](https://pmc.ncbi.nlm.nih.gov/articles/PMC12730038/) suggest small VLMs benefit from LoRA on attention layers only, not MLPs, with rank 8–16 as a typical sweet spot for domain-shift fine-tuning.

**Why it happens:**
- Frozen-backbone is the default for "ML hygiene" but is wrong when domain shift is large.
- Full-finetune is tempting for "we have a small model, why not", but the dataset is small relative to the backbone's pretraining.
- LoRA's rank/alpha defaults are often 8/16 from natural-image papers; cartography may need different.

**Warning signs:**
- Frozen baseline: training loss plateaus at a high value within 2–3 epochs; eval NLL is bounded above ~1.5 nats per pixel and won't go lower regardless of seg-head capacity.
- Full-finetune: training loss decreases nicely but eval NLL on `historical` source diverges upward after epoch ~5 while `synthetic` keeps improving — classic catastrophic forgetting / overfit-to-dominant-source.
- LoRA: training loss is smooth but eval NLL gap between sources is large and stable — backbone isn't adapting enough to bridge the synthetic-historical gap.
- Held-out spot-check renders on Tolkien Middle-Earth: frozen produces "satellite-flavored" outputs (uses ImageNet priors); full-finetune produces "synthetic-flavored" outputs (overfits to Azgaar style); LoRA at correct rank should look "in between" / coherent across styles.

**Prevention strategy:**
1. **Default to LoRA on attention-Q/V only, rank=16, alpha=32** for the small backbone. Documented as the default in the v0.yaml config; deviation requires a one-line note in the experiment log.
2. **Run a 30-epoch frozen-backbone v0 as a baseline** before any LoRA experiments. If frozen NLL is good enough (within 10% of LoRA NLL), ship frozen — it's simpler, smaller, faster, and avoids the LoRA rank-tuning rabbit hole.
3. **Track per-source NLL separately** in eval; the "synthetic improving while historical regressing" pattern is the early-warning signal for overfit.
4. **Hand-pick LoRA target modules** — apply LoRA to the backbone's attention `q_proj` and `v_proj` only. Do NOT add LoRA to the seg heads (which are trained from scratch and don't need adapters).
5. **Don't try to compete with PaliGemma on capacity** — the small backbone is constrained by inference budget. If LoRA isn't enough, increase data, augmentation diversity, and seg-head capacity *before* widening LoRA — those scale within the constraint.

**Recovery if it occurs:**
- Retraining cost. The small backbone is cheap (the cloud-VM SSH workflow handles repeats). Iterate on rank / target modules in 4–8 hours per attempt.

---

### Pitfall 8: OCR module false positives on textured map elements (mountains, forests)

**Severity:** HIGH
**REQ-IDs:** MODEL-02

**What goes wrong:**
Stylized maps are full of pen-and-ink hatching, contour lines, mountain shading, and forest stippling — visual textures that share statistical properties (high local edge density, repeating patterns) with text. CRAFT's character-region heatmap can fire on parchment hatching; ABCNet's curve-fitting can produce Bezier curves around tree-stippled regions. The OCR head's loss term then "supervises" the backbone toward features that distinguish text from non-text in *natural* scenes, which is the wrong inductive bias for cartographic textures. Worst case: the OCR head produces hundreds of spurious detections per fantasy map, the seg heads co-train against features that are now warped toward "text vs not-text-but-actually-a-tree", and land-cover NLL regresses.

The [ABCNet paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf) acknowledges that segmentation-based detectors are "easily affected by nearby text" — and cartographic clutter is the dual problem (background that *looks* like text).

**Why it happens:**
- CRAFT/ABCNet were trained on natural-scene text (signs, books, posters), not maps.
- Maps have **dense overlapping** text + texture co-occurrence (a place name written across hatched mountains).
- The decision in `MODEL-02` of "joint training" vs "separate sub-pipeline" matters here. Joint training propagates OCR false positives into the backbone; separate sub-pipeline isolates the damage.

**Warning signs:**
- OCR detection count per image is implausibly high (e.g., > 200 detections on a Tolkien map that has < 30 visible labels).
- Detected "text" boxes cluster on terrain features (mountains, forests) rather than near settlements/roads.
- Land-cover NLL on synthetic (which has minimal text) regresses when the OCR loss term is added; baselines without OCR loss outperform.
- Backbone attention maps (if you visualise them) attend to mountain hatching when prompted with a place name from an unrelated region.

**Prevention strategy:**
1. **Treat OCR as a separate sub-pipeline first, joint-train only after isolation.** Architecture Pattern 4 (filesystem-mediated, not in-process) extends here: train an OCR head independently on a small map-text-annotated set; only after it produces sane outputs on a held-out map fold the OCR loss into the joint backbone training.
2. **Use a low OCR loss weight (λ ≈ 0.1) when joint training begins.** The OCR signal is most useful if it's a small regularizer, not a co-equal task. Increase only after evidence.
3. **Mine hard negatives from cartographic textures** — generate synthetic training crops of "hatched mountain, no text" and "stippled forest, no text" and add them to the OCR training data with empty annotation. This is the standard fix for false positives in detection.
4. **Detection-count sanity diagnostic** — log the number of OCR detections per training image; if the median > 5x the median *expected* labels per image (roughly inferable from synthetic-map text density), the head is over-detecting. Add this to TS-7 (training-time logging).
5. **Confidence threshold tuning** — at inference, use a high detection-score threshold (e.g., 0.5+) rather than the default; fantasy maps have few labels, so high-precision is preferred over high-recall in v1. (Anti-features list already excludes multi-language OCR; this is the matched policy on detection too.)

**Recovery if it occurs:**
- Lower λ on the OCR loss term, retrain.
- Drop OCR for v1 ship (FEATURES.md treats MODEL-02 as P1 but the dense seg-head NLL is the actual ship metric — OCR can be deferred if it's net-negative).

---

### Pitfall 9: OCR language-model bias hallucinating geographic names (Tolkien → Tolstoy)

**Severity:** HIGH
**REQ-IDs:** MODEL-02

**What goes wrong:**
[Recent OCR research](https://arxiv.org/html/2604.12978v1) documents that modern OCR systems "rely on language-model pretraining as much as visual recognition" — and on unfamiliar scripts/fonts, "models either produce random noise or hallucinate characters from similar scripts." Fantasy maps use names like *Mordor*, *Erebor*, *Rohan*, *Ulthuan*, *Vermilion Sea*. An OCR model with a strong English-language prior reads `Mordor` and outputs `Modern` (it's a more probable English word). For paper figures (EVAL-02 qualitative renders) this is embarrassing; for downstream consumers (the hex-grid app's optional name extraction) it's actively wrong.

**Why it happens:**
- Encoder-decoder OCR models are trained with a language-model loss; they autoregressively decode tokens biased by frequency in the training corpus.
- Fantasy names are out-of-distribution.
- CRAFT detects, but the *recognizer* downstream (TPS-Net, CRNN, or whatever follows) does the LM-biased decoding.

**Warning signs:**
- Spot-check on Tolkien/Westeros/Abercrombie: detected text is real-English-word-like (`Modern`, `Robin`, `Vermin`) instead of fantasy-like (`Mordor`, `Rohan`, `Vermilion`).
- Edit-distance from the actual map text (manually transcribed for spot-check) is large but the decoded strings are themselves valid English words — sign of LM rewriting.
- Detection boxes are correct but recognized text is wrong; lowering the LM weight in beam search restores the correct text.

**Prevention strategy:**
1. **Use a fixed-vocabulary recognizer trained on character-level outputs**, or use beam search with a near-zero LM weight. Don't use a recognizer with a full English LM head.
2. **Augment training data with synthetic fantasy text** — generate synthetic crops of pseudo-fantasy names rendered in cartographic fonts (cheap with PIL + a Tolkien-flavored unigram generator) and include them in the OCR training set.
3. **Include training-time labels for *invented* names** — if joint-training data has fantasy maps, transcribe their names as ground truth even though the words aren't real. This is the strongest signal that "out-of-vocab is allowed."
4. **Document the limitation** in the inference module README (per TS-11). Multi-language and fantasy-name OCR is in the anti-features list (FEATURES.md); v1 is Latin-only English-trained OCR with a known LM-bias caveat. Owners shipping a downstream app should know the recognized text is unreliable for fantasy names and prefer detected boxes (no recognition) where exact text matters.
5. **De-prioritise OCR if it doesn't help the dense seg-head NLL** — see Pitfall 8 mitigation 2. The ship metric is per-pixel NLL, not text accuracy. If OCR is degrading rather than helping, ship without it.

**Recovery if it occurs:**
- Document, ship, defer multi-language / fantasy-text OCR to milestone 2.

---

### Pitfall 10: Class imbalance dominates segmentation training (water/grassland > cropland/built-up)

**Severity:** HIGH
**REQ-IDs:** DATA-06, MODEL-03, TRAIN-01

**What goes wrong:**
ESA WorldCover globally is dominated by `water`, `trees`, `grassland`. Synthetic Azgaar maps are dominated by `water`, `grassland`. OSM tiles render at low zoom over land are dominated by `built_up` only in urban-centred extents — at random global tiles, `built_up` is < 5% of pixels. A naive cross-entropy loss optimizes mean NLL by being good at the dominant classes; rare classes (`cropland`, `built_up`, `flooded_wetland`, `snow_ice`) get ignored. Per-class NLL is fine on dominant classes, terrible on rare classes, and the **mean** NLL looks acceptable, hiding the failure.

The [Unified Focal Loss paper](https://arxiv.org/abs/2102.04525) and [Loss Functions Survey 2024](https://arxiv.org/html/2312.05391v1) cover this extensively for medical segmentation; cartography has the same dynamic.

**Why it happens:**
- Real-world geography is class-imbalanced; uniform per-pixel NLL maximises overall accuracy, not per-class accuracy.
- Per-source loss weights (DATA-06) operate at the source × class level (not at the pixel-frequency level within an image), so they don't compensate for imbalance *within* a class.
- The current loss-weights schema downweights *unreliable* classes (e.g., `built_up: 0.1` on historical) — but this is the wrong direction for imbalance: rare classes need *more* weight, not less, where they appear and are reliable.

**Warning signs:**
- Eval per-class NLL (which TS-5 reports) shows `cropland`, `built_up`, `flooded_wetland`, `snow_ice` at 3-5x the NLL of `water`, `trees`, `grassland`.
- Confusion matrix: rare classes are predicted as the most common neighbouring class (e.g., `cropland` always predicted as `grassland`).
- Argmax outputs almost never include rare classes — argmax frequency of `built_up` < 0.1% across the held-out set.

**Prevention strategy:**
1. **Use focal cross-entropy or unified-focal loss** (gamma=2 default) instead of vanilla CE. Focal CE downweights well-predicted easy pixels and effectively shifts the model's attention to rare/hard pixels. This is the modern standard for imbalanced segmentation.
2. **Compute pixel-frequency class weights once at dataset-build time** and bake them into a *global* class-weight vector that multiplies the per-pixel CE. Combine multiplicatively with the per-source × per-class weights from DATA-06 (so the final per-pixel loss weight is `source_weight × class_pixel_weight`).
3. **Track per-class NLL in eval, not just mean.** TS-5 already does this; gate v0/v1 ship on the worst-class NLL being acceptable, not just mean NLL.
4. **Sample-balance at dataloader level** — if a class is < 1% of pixels globally but is critical (e.g., `built_up` for the modder use case), oversample images that contain it. Implementation: a `WeightedRandomSampler` parameterised by per-image rare-class fraction.
5. **Don't conflate "rare" with "downweighted by source-trust"** — `cropland` on historical is downweighted for *trust* reasons (16th-c cropland ≠ modern cropland). `cropland` on synthetic is *zero* because it's not in the synthetic taxonomy at all. Be explicit in the YAML about which is which.

**Recovery if it occurs:**
- Add focal CE, retrain. Cost: one full training run.

---

### Pitfall 11: Segmentation-head miscalibration — overconfident wrong predictions

**Severity:** HIGH
**REQ-IDs:** MODEL-03, EVAL-01, EVAL-02, SHIP-01

**What goes wrong:**
CNN/ViT segmentation heads are documented to be **systematically miscalibrated** (the [Mask-TS Net paper, 2024](https://arxiv.org/abs/2405.05830) is the most recent evidence; this is FEATURES.md TS-6). They produce probabilities that are over-confident: a 0.95-probability output is actually correct ~80% of the time. The hex-grid app downstream **aggregates per-pixel probabilities into hexes** — overconfident probabilities propagate cleanly into wrong hex labels, with no uncertainty signal that something was iffy. Worse: argmax-confidence outputs *look correct* on spot-checks (sharp colored regions) and only fail on the metric (NLL) and on downstream aggregation.

**Why it happens:**
- Cross-entropy training maximises log-likelihood, which is minimised by over-confident-when-correct *and* over-confident-when-wrong. The objective doesn't care about calibration.
- Batch normalization, label smoothing absence, and class imbalance all contribute.
- The hex-grid app's aggregation step is a downstream consumer that the training-time eval doesn't see — calibration matters more for them than for the training-loss number.

**Warning signs:**
- Reliability diagram (predicted-confidence vs empirical-accuracy) is below the diagonal — model says 0.9, gets 0.7 right.
- Expected Calibration Error (ECE) > 0.05.
- Hex-grid app's outputs (when integration testing happens) have crisp boundaries that don't reflect actual model uncertainty.
- High-entropy pixels (where the model "should be unsure") are concentrated only at class boundaries; interior-of-region pixels are uniformly 0.99-confident even on out-of-distribution images.

**Prevention strategy:**
1. **TS-6 from FEATURES.md (already P1 in roadmap-feeding research)** — append a reliability-diagram + ECE check to `mapclass.eval`. One-shot diagnostic; ~30 lines.
2. **Apply temperature scaling at eval time** — the simplest post-hoc calibration: fit a single temperature `T` on the validation set such that softmax(logits/T) is calibrated; report both calibrated and uncalibrated NLL. Keep T as a checkpoint metadata field; the inference module applies T at predict time.
3. **Use label smoothing during training (smoothing=0.05)** — well-documented to improve calibration at marginal cost to accuracy.
4. **Report calibrated probabilities in the inference module by default** — the hex-grid app receives temperature-scaled probs unless the caller explicitly requests raw. The contract in TS-3 already says "probabilities, not logits"; calibrate them.
5. **Reliability diagrams in EVAL-02 spot-checks** — for each of Tolkien/Westeros/Abercrombie/Warhammer, render the entropy map alongside the argmax. Visible high-confidence-but-out-of-distribution regions are exactly the uncertainty signal the hex-grid app needs.

**Recovery if it occurs:**
- Add temperature scaling at eval time (5 minutes of compute on validation set, no retraining).
- If ECE is still high after T-scaling, add label smoothing and retrain.

---

### Pitfall 12: Tile-boundary edge effects in dense segmentation inference

**Severity:** HIGH
**REQ-IDs:** MODEL-03, SHIP-01

**What goes wrong:**
Fantasy maps are often 4096×4096+. The backbone's native input is much smaller (224 / 384 / 448). The inference module must tile-and-stitch (FEATURES.md D-9 promotes-to-table-stakes-if-backbone-input-≤-512). Naive tiling produces visible **seam artifacts** at tile boundaries: pixels at the edge of a tile have less context than pixels at the centre, so their predictions are systematically lower-quality. When tiles are stitched, the seams show as 1-pixel-wide stripes of class-disagreement, which the hex-grid app's aggregation will pick up as spurious region boundaries.

The [arXiv 2503.19545 paper on tiling artifacts](https://arxiv.org/html/2503.19545v1) and the [biomedical tiling tutorial](https://buglakova.github.io/tiling_artifacts_tutorial/) both document this — even with overlap, sliding-window inference can produce abrupt discrepancies between predictions in neighbouring tiles, especially when feature normalisation (instance-norm / batch-norm) is computed per-tile.

**Why it happens:**
- Pixels at the edge of a tile receive less context than pixels in the centre.
- Per-tile feature normalisation (LayerNorm in ViT, BatchNorm in CNN heads) computes statistics over the tile, so the same pixel in two different tiles gets different normalised features.
- Even with overlap, naive averaging at the seam can produce hard transitions.

**Warning signs:**
- Output probability map has visible 1-2 pixel wide stripes at predictable spacing (the tile size).
- Argmax map has hard discontinuities at tile edges (a `forest` region abruptly becomes `grassland` at the seam).
- Hex-grid app reports many tiny single-pixel "regions" along regular grid lines.
- High-entropy band along tile edges in the entropy map.

**Prevention strategy:**
1. **Overlap with feathered weighted blending.** Tile with stride < tile-size (e.g., 50% overlap). At stitch time, weight each pixel by a 2D cosine window centred on the tile centre — pixels at the tile centre have weight 1, edge pixels have weight 0. Sum weighted predictions and normalise by total weight. Implementation: ~20 lines in `mapclass.infer`.
2. **Don't use per-tile feature normalisation at inference if avoidable.** Use global (pretrained) statistics, not running statistics computed per-tile. For ViT, this means using the model in eval mode where LayerNorm uses learned scale/shift; for CNNs, use frozen running BN stats.
3. **Fixed-size tiles, padded inputs.** Pad the input image to a multiple of tile-size with reflection padding. Crop the output back to the original image size. Avoids variable-sized edge tiles whose predictions are even more degraded.
4. **Visualise the seam map as a diagnostic.** During eval, take a large held-out image, run the tiled inference, and overlay a grid showing the tile boundaries; spot-check whether predictions at boundaries match interiors.
5. **Promote D-9 (tile-and-stitch helper) to v1 table-stakes** if the chosen backbone has native input < 512 (FEATURES.md flagged this as conditional). The hex-grid app's input distribution makes this near-certain to be required.

**Recovery if it occurs:**
- Add feathered blending. No retraining needed; this is an inference-side fix.

---

### Pitfall 13: TPS-warping pathologies on auto-georeferencing

**Severity:** HIGH
**REQ-IDs:** GEOREF-01, GEOREF-02

**What goes wrong:**
Thin-plate-spline warping is documented to produce **overshoot oscillations** and ill-conditioned matrices when GCPs are clustered, near-collinear, or numerous. [GDAL's TPS implementation](https://gdal.org/en/stable/programs/gdalwarp.html) and [Wikipedia's TPS overview](https://en.wikipedia.org/wiki/Thin_plate_spline) document that the system matrix becomes ill-conditioned with many close points; [computational TPS literature](https://www.academia.edu/29654283/Warping_digital_images_using_thin_plate_splines) flags severe overshoot in 2D. On a 17th-c Rumsey map where v0's predicted-coastline GCPs cluster around one well-known peninsula and are sparse elsewhere, TPS warps the dense region beautifully and the sparse region wildly — the resulting GeoTIFF has a "good" coast and a "fictional" interior. The historical labeller (which doesn't know any better) then produces label rasters that align with WorldCover for the coast and are catastrophically wrong inland.

**Why it happens:**
- TPS minimises bending energy globally; sparse-GCP regions have to balance the high-density regions and overshoot.
- Cross-correlation against WorldCover finds matches at high-contrast features (coastlines), not low-contrast ones (forest boundaries) — so GCPs are intrinsically clustered.
- "Many close points" is exactly what happens when v0 is good at coastlines: it produces 50 GCPs along a single coast, all near-collinear.

**Warning signs:**
- TPS residuals (after warping, distance from GCP to its expected location) are < 1 pixel for some GCPs and > 100 pixels for others on the same map.
- Visual diff of warped map vs. WorldCover reference shows the coast aligned well and inland features wildly displaced.
- The auto-georef tool produces a GeoTIFF whose pixel-to-coordinate mapping near the centre of the map disagrees with the corner mapping by > 10 km.
- Label rasters from `make_labels()` on the bootstrapped TIF show the coast correctly but show "WorldCover-style modern cropland" in regions the historical map clearly draws as "forest" — that's the warping-induced spatial mis-registration showing as label-domain noise.

**Prevention strategy:**
1. **Reject GCP configurations that are degenerate.** Before TPS, compute the GCP convex hull; if the hull covers < 30% of the image area, reject (sparse-GCP-regions cannot be safely TPS'd). Fall back to a global affine fit + an explicit "low-confidence" flag on the manifest.
2. **Use regularised TPS (smoothing splines).** Most TPS implementations have a smoothing parameter (`s` in scipy, regularisation in scikit-image). A small non-zero value damps overshoot at the cost of GCP-residual exactness — for noisy v0 predictions, this is the right trade.
3. **Cap the number of GCPs.** With n > ~50, the system matrix conditioning degrades noticeably. Sub-sample a uniform spatial distribution from the v0-detected GCP set rather than using all of them.
4. **Validate warped output before accepting.** After TPS, run a sanity check: re-extract a small set of high-contrast features from the warped image and see whether they align with WorldCover within tolerance. If not, mark the map as "auto-registered with low confidence" and either downweight it (`confidence: 0.25` instead of `0.5` in the bootstrapped weight row) or reject it from the bootstrap set.
5. **Sanity-check mode is for this** — `python scripts/georef.py sanity-check` (architecture build-order step C) on already-registered Rumsey maps gives a measured residual distribution before any model exists. If residuals are bad on known-good maps, the TPS / cross-correlation pipeline itself is broken; if they're good, the bootstrap-mode failure mode is contained to v0's prediction quality.

**Recovery if it occurs:**
- Drop the worst-residual auto-registered maps from the v1 training set.
- Retrain v1 without them.
- Or skip the bootstrap and ship v0 (Pitfall 1's kill-switch).

---

### Pitfall 14: Auto-georef silent "no good match" failure

**Severity:** HIGH
**REQ-IDs:** GEOREF-01, GEOREF-02

**What goes wrong:**
Cross-correlation against WorldCover assumes the input map *has* coastline / water features that match the reference. Some Rumsey maps are landlocked country interiors with no coast; others are at scales where WorldCover's class boundaries don't show up at all (a 1:5,000,000 continental map of "Asia" has only continental-scale features, mostly invisible at WorldCover's 10 m resolution). The cross-correlation peak is then noise — a shallow local maximum with no significance. If the auto-georef tool doesn't have a confidence threshold, it silently registers the map at the noise-peak location, producing a wildly wrong registration.

**Why it happens:**
- Cross-correlation always returns a peak; whether the peak is meaningful is a separate question (peak value vs. peak-to-second-peak ratio).
- The unregistered-Rumsey set is heterogeneous; many maps are out-of-domain for WorldCover-based registration.
- The default behaviour ("register everything you can") is aspirational; in practice, "register only the ones we can do well" is the goal.

**Warning signs:**
- Many auto-registered maps in the bootstrap set have GCPs concentrated in implausible regions of the map (e.g., GCPs cluster in the corner of the map but the map's *content* isn't in that corner).
- v1 training loss is unstable and high on the bootstrapped subset — the model can't fit randomly-registered samples.
- Spot-checking 10 randomly-bootstrapped maps reveals 3+ that are visibly mis-registered (the warped map's coastlines don't trace WorldCover's coastlines).
- The bootstrap "added 200 maps" but D-5's v0-vs-v1 NLL comparison shows v1 worse than v0 — silent low-quality additions are masking the bootstrap value.

**Prevention strategy:**
1. **Peak-significance threshold.** Compute the ratio of the cross-correlation primary peak to the second-highest peak. Reject if the ratio is below a threshold (e.g., 1.5 — the primary peak must be at least 50% taller than the next-best alternative).
2. **Spatial coverage check.** As in Pitfall 13, demand that GCPs cover > 30% of the image area. A registration with all GCPs in one corner is rejected.
3. **Domain-applicability filter.** Pre-classify Rumsey maps by metadata (LUNA tags include scale and region); only attempt auto-registration on maps whose scale and content domain match WorldCover's strengths (regional scale, with coast or major water features). Continental-scale maps and landlocked interiors are pre-filtered out.
4. **Manifest-as-rejection-record.** Every map attempted: succeed → write to `data/historical/raw/georeferenced/`; reject → write to `data/historical/raw/rejected_register.json` with the reason. Visible reject rate is a health metric; if reject rate is < 10% the threshold is too lax.
5. **Sample the rejected set.** Manually inspect 10 maps from the rejected set; if some look obviously registrable, the threshold is too strict and the bootstrap is leaving signal on the table. Iterate on threshold value at sanity-check stage (architecture step C), before the bootstrap is called for real.

**Recovery if it occurs:**
- Tighten thresholds, re-run bootstrap, retrain v1. The recovery cost is mostly the v1 training run.

---

## Medium Pitfalls

### Pitfall 15: Reproducibility — no seeds, single run, can't replicate the EVAL-03 number

**Severity:** MEDIUM (HIGH for paper-credibility, MEDIUM for ship)
**REQ-IDs:** TRAIN-01, TRAIN-02, EVAL-01, EVAL-03 (also CONCERNS.md 8a)

**What goes wrong:**
`.planning/codebase/CONCERNS.md` 8a flags that no seeds exist anywhere. Adding training and bootstrap on top: every step has a non-deterministic component (data shuffling, weight init, dropout, augmentation, GCP sampling). A v0 → v1 → eval run produces an NLL number; re-running produces a different number. The EVAL-03 PaliGemma-vs-small-backbone gap could be explained by seed variance rather than backbone difference. There's no way to know without running each variant ≥3 times — but the project has compute budget for one run per variant.

**Why it happens:**
- Pre-existing technical debt (`np.random.normal(...)` without a seed in `scripts/augment.py`).
- Adding `torch.manual_seed`, `np.random.seed`, `random.seed`, `torch.backends.cudnn.deterministic = True` requires touching every entry point.
- Even with seeds, certain ops (CUDA atomic adds, multi-threaded DataLoader) are non-deterministic — full bit-reproducibility costs ~10% throughput.

**Warning signs:**
- Re-running the exact same training config produces different eval NLL.
- Per-class NLL changes by > 0.05 nats between two seeds — that's a reproducibility issue, not seed variance.
- The cloud-VM SSH workflow runs over hours; a transient cuDNN nondeterminism produces a result you can't replicate even with the seed set.

**Prevention strategy:**
1. **`mapclass.utils.seeding.set_global_seed(seed)`** — the utility CONCERNS.md 8a recommended. Call from every entry point (`mapclass.train`, `mapclass.eval`, `scripts/georef.py bootstrap`). Pin `random`, `numpy`, `torch`, `torch.cuda`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`. ~15 lines.
2. **Pass the seed via CLI** — `--seed 42` everywhere. Default to 42 explicitly. Embed the seed in `splits.json` and in checkpoint metadata.
3. **Three-seed budget for the bellwether.** EVAL-03 *is* the high-stakes comparison; the other 11 metrics can be single-seed. Run PaliGemma + small backbone with 3 seeds each (6 runs total). Report mean ± stddev. If the gap is < 1 stddev, the bellwether is null and MODEL-01 is fine.
4. **Document non-determinism sources.** Even with seeds, `cudnn.deterministic` doesn't capture every op. Note in the eval JSON that the number is reproducible to ~0.01 nat (typical cudnn-deterministic noise floor).

**Recovery if it occurs:**
- Add seeds, re-run. Cost: depends on which run you need to re-do. Flag explicitly that pre-seed numbers are not directly comparable to post-seed numbers.

---

### Pitfall 16: OSM ODbL share-alike triggers on bootstrap-expanded dataset

**Severity:** MEDIUM
**REQ-IDs:** DATA-05 (also overlaps CONCERNS.md 1)

**What goes wrong:**
PROJECT.md handles dataset licensing by **not redistributing the combined dataset** (ship weights + code only). The OSM ODbL adds a layer: per [OSMF guidance](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines) and the [Collective Database Guideline](https://wiki.openstreetmap.org/wiki/Collective_Database_Guideline), training datasets that contain "substantial extractions" from OSM are themselves Derivative Databases and must be ODbL-licensed if publicly distributed. The OSM Wiki explicitly clarifies that **predictions from a model are not implicated by ODbL**, but the training dataset is. Combined with the "don't redistribute" PROJECT.md decision, OSM compliance for v1 ship is contained — but milestone 2 (the paper-track open-dataset claim) was already dropped, and adding OSM strengthens the reason it was dropped.

The new pitfall: **OSM attribution is required even for predictions** (the OSMF guidance: "models that have been trained with such training sets must be attributed in documentation"). If the inference module's README or the checkpoint metadata don't credit OSM, the v1 ship is technically non-compliant even though no dataset is distributed.

**Why it happens:**
- Attribution is an active obligation, not a passive one.
- OSM is mentioned in the architecture but might not propagate to the inference module's docs or checkpoint metadata.
- The "we don't redistribute the dataset" decision can be misread as "we have no OSM obligation at all."

**Warning signs:**
- The inference module's README does not mention OSM.
- The checkpoint metadata does not include "trained on data derived from OpenStreetMap (© OpenStreetMap contributors, ODbL)."
- The hex-grid app downstream redistributes the model checkpoint but doesn't pass through OSM attribution.

**Prevention strategy:**
1. **Embed OSM attribution in checkpoint metadata.** TS-8 already requires versioning metadata in safetensors header; add a `data_attribution` field listing OSM, Rumsey, ESA WorldCover, Copernicus DEM, and Azgaar with their respective notice strings. The inference module reads and exposes this; the hex-grid app's own README inherits it.
2. **README of the inference module** explicitly credits OSM. One-line obligation; trivial to satisfy.
3. **License-review item on `executive_TODO.md`** explicitly covers OSM (currently lists three sources; add OSM as the fourth). Verify `executive_TODO.md` is updated when DATA-05 is implemented.
4. **Don't publish the OSM-derived training data** even via accident (e.g., a dataset-build log file that contains raw OSM tags). The "don't redistribute" stance covers this if it's enforced.

**Recovery if it occurs:**
- Add attribution to README and checkpoint metadata. Single-commit fix.

---

### Pitfall 17: OSM tile renderer style consistency / projection mismatches

**Severity:** MEDIUM
**REQ-IDs:** DATA-05

**What goes wrong:**
OSM tiles are typically rendered in Web Mercator (EPSG:3857). WorldCover and Copernicus DEM are in EPSG:4326 (geographic) at 10 m / 30 m resolution. The road-tile sub-pipeline must render OSM at a chosen extent, then project the same extent into WorldCover/DEM coords for label generation — and both must agree. A unit mismatch (tile rendered in Mercator, label fetched in WGS84-degrees) produces a sample where image and labels are off by tens of meters at high latitudes (Mercator distorts horizontally with latitude).

A second variant: OSM data is **continuously updated**. WorldCover is a snapshot (currently `v200/2021/`, hardcoded — TESTING.md gap-item, also CONCERNS.md 8b). An OSM tile rendered today might show a road built in 2024; the WorldCover label for that pixel is `built_up` from 2021 or could still be `cropland` if the road's right-of-way wasn't urbanised. Time-domain mismatch is small per-pixel but accumulates across the dataset.

**Why it happens:**
- Mixing CRS systems is a perennial source of bugs (TESTING.md gap-item 5).
- The historical pipeline's `_to_wgs84_bbox` reproject helper is reused for OSM, but OSM tiles aren't GeoTIFFs — they're PNGs with implicit Mercator coordinates.
- WorldCover's hardcoded version suggests the pipeline is freshness-blind.

**Warning signs:**
- Spot-check OSM samples: roads in the rendered image don't align with built-up label pixels. Visible offset > 5 pixels at the image scale.
- The offset varies systematically with latitude (small at equator, large at high lat) — that's Mercator-vs-geographic distortion.
- WorldCover label has `cropland` or `bare_sparse` exactly where the OSM render shows roads / urban — temporal mismatch.

**Prevention strategy:**
1. **CRS round-trip verification.** Before saving each OSM sample, project the image's bounding box to both CRSs (WGS84 for label fetch, Mercator for image render) and check that re-projecting back yields the original within tolerance. The `pyproj.Transformer` loses ~0.1 m at typical latitudes; tolerances looser than that indicate a bug.
2. **Use a tile renderer that emits geographic coordinates natively** — `osmnx`-driven render with explicit WGS84 bbox specification, rather than a Mercator-tile-stitcher. Architecture's "STACK research phase decides" between Overpass and PBF is the place this is locked down.
3. **Document the WorldCover snapshot date** in the per-sample manifest (`worldcover_version: "v200/2021/"`). If a future bootstrap retrains on samples dated 2026, the time-domain mismatch is quantifiable.
4. **Downweight `built_up` and `cropland` for OSM** in the loss-weights schema where temporal mismatch is biggest. The `road_osm` weights row in architecture Pattern 2 already has `cropland: 0.7` (not 1.0); validate this is enough by per-source NLL spot-check.
5. **Reject OSM samples in regions with high recent built-up change** — NDVI-change-detection or "recent OSM edit density" can flag tiles where the WorldCover snapshot is stale. Out of v1 scope; document as a known limitation in the inference module README.

**Recovery if it occurs:**
- Fix the CRS handling in the OSM sub-pipeline. Re-build the OSM source.
- Time-domain noise is irreducible without a same-year label source; downweight or accept.

---

### Pitfall 18: Single-author bus factor + no tests + new ML on a brownfield codebase

**Severity:** MEDIUM (HIGH if owner takes a 4-week break; LOW if continuous attention)
**REQ-IDs:** All (cross-cutting; CONCERNS.md 4 already flags)

**What goes wrong:**
Six new components (OSM source, auto-georef offline, auto-georef bootstrap, OCR head, two seg heads, training, eval, inference) on a codebase with no tests, no CI, no formatter, and one author. The "yolo mode" velocity is real but every silent bug from above (Pitfalls 3, 4, 5, 17) is one-degree-of-protection-removed. A typical pattern: owner ships a working v0 → takes a vacation → returns and can't remember which YAML row matched which class index → makes a "harmless" rename → loss-curve on v1 looks fine → eval NLL is quietly worse → owner attributes it to "model variance" and ships → hex-grid app integration reveals a class is unused → 3 days of debugging traces back to the rename.

The accepted-risk position in PROJECT.md is correct *given* the time-budget tradeoff, but the **mitigations** for that risk are weaker than tests. The mitigations need to be deliberate, not hopeful.

**Why it happens:**
- "It works for me" verification only.
- Print-statement diagnostics (CONCERNS.md notes mixed `print()` for output) drown out signal.
- Cross-script schema drift (the loss-weights / canonical taxonomy / WC_REMAP triangle, TESTING.md gap-item 1).

**Warning signs:**
- Eval NLL changes between two runs of "the same" config (config drifted, you didn't notice).
- Spot-check renders look "fine" but per-class metrics show a class is silently never predicted (Pitfall 3 / Pitfall 10's signature).
- Owner discovers a 30-line script does what a 200-line script was claimed to do (ad-hoc duplication; smaller blast radius from one bug fix).

**Prevention strategy:**
1. **Embrace observability over tests.** TS-7 (training-time logging) + TS-5 (eval JSON report) + TS-8 (model versioning) collectively replace the regression-test net the project doesn't have. Treat these three as non-negotiable v1 features (FEATURES.md already does).
2. **One-line schema validators are cheap**, even without a test framework. Pitfall 3's `LossWeights.load()` validator is 30 lines; Pitfall 4's hash check is 10 lines; Pitfall 5's split-overlap assertion is 5 lines. None require pytest. Just `assert` statements at startup.
3. **Commit-message discipline as a lightweight changelog.** The owner is also the only consumer; commits like "rename `built_up` to `urban` (HISTORICAL_LC_WEIGHTS row also updated, WC_REMAP also updated)" function as a memory of which-files-must-be-touched-together.
4. **Pre-merge-into-main checklist** — even without CI, a personal checklist (in `.planning/MAINTENANCE.md` or similar) that the owner runs through before a major change: "did I touch the canonical taxonomy? if yes, also touch [files X, Y, Z]." Costs 30 seconds; saves 3 days.
5. **The bus-factor is real for the owner alone.** Document architectural decisions in `.planning/PROJECT.md` (already done) and `.planning/research/ARCHITECTURE.md` (already done). The "what does this do and why" for each new component is your future-self's safety net.

**Recovery if it occurs:**
- Re-derive intent from `.planning/` docs.
- Bisect the regression by reverting commits one by one.
- The cost is owner-time; mitigations above reduce probability and per-incident cost.

---

### Pitfall 19: Training-script-repo / VM-provisioning-repo drift

**Severity:** MEDIUM
**REQ-IDs:** TRAIN-01, TRAIN-02 (also CONCERNS.md 8c)

**What goes wrong:**
PROJECT.md Constraints: "Training compute is external — heavy training runs on a separate VM repo over SSH. This repo carries training scripts, not training infrastructure." Two-repo split means: (a) the training-script repo (this one) evolves dependencies, (b) the VM-provisioning repo evolves Python/CUDA/driver versions. They drift independently. A working `requirements.txt` here might fail on the VM because `bitsandbytes` requires CUDA 12.1 and the VM has 11.8. Failure manifests at training start on the VM, far from the local-machine context where the change was made.

CONCERNS.md 8c documents the no-lockfile issue; FEATURES.md D-10 (lockfile) addresses it. The two-repo split makes it more important, not less.

**Why it happens:**
- Two repos with no cross-link.
- `requirements.txt` is unpinned (`transformers>=4.41` etc.).
- The VM is provisioned ahead of training; mismatches are discovered at run-time, not provision-time.

**Warning signs:**
- `pip install -r requirements.txt` succeeds on the VM but `python -m mapclass.train` fails with a CUDA mismatch / missing `bitsandbytes` symbol / `peft` API change.
- Two consecutive cloud runs of "the same" code produce different behaviour because pip resolved a different version.
- `bitsandbytes` import errors that disappear on `pip install --upgrade`.

**Prevention strategy:**
1. **Lockfile (D-10) is mandatory, not optional.** `uv.lock` or `requirements.lock.txt` committed to this repo. The VM-provisioning repo's setup script reads this lockfile and pins to those versions.
2. **Embed Python + CUDA expectations in the README.** Cross-link the VM-provisioning repo from this repo's README (CONCERNS.md 8c recommendation). Cross-link this repo from the VM-provisioning repo's README. Both should be discoverable from each other.
3. **Integration smoke-test in the cloud-VM workflow.** First step of every cloud training run: `python -c "import mapclass; mapclass.preflight_check()"` — a 30-line script that imports every dependency, checks CUDA availability, checks the dataset directory exists, and exits 0 or 1. Catches the env mismatch in 30 seconds instead of after a 2-hour data load.
4. **Pin a Python version** (`.python-version` or `pyproject.toml requires-python`). Floating Python is also a drift source.
5. **Training script reads its own dependency hash** — the preflight check above can compare the current installed-package versions against the lockfile and warn if they differ.

**Recovery if it occurs:**
- Update lockfile, sync VM, re-run.
- Cost: minutes, but only if the failure is caught early.

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skipping the `LossWeights` schema validator | -30 lines of code | A class silently has weight 0 or 1; one wasted training run | Never — the validator is too cheap to skip |
| Using `glob` order for dataset splits | Simpler than hashing | Pitfall 5 — held-out leaks during bootstrap | Never for v1 ship |
| Hardcoding LoRA rank=8 without validating | -1 hyperparameter to think about | Pitfall 7 — frozen-or-overfit failure mode | Acceptable as a first-pass default; revisit if NLL plateaus |
| Skipping the v0-vs-v1 NLL kill-switch (D-5) | -50 lines of code | Pitfall 1 — bootstrap silently regresses, you ship v1 anyway | Never — D-5 is the kill-switch for the bootstrap loop |
| Skipping seeds | +0% throughput vs +5–10% with full determinism | Pitfall 15 — EVAL-03 gap can't be attributed | Never for the bellwether (EVAL-03); acceptable for ad-hoc spot-checks |
| Hardcoding tile size in the inference module | Simpler API | Pitfall 12 amplified — non-overlap tiling produces seams | Only for specific known image sizes; otherwise use feathered overlap |
| Not embedding preprocessing config in checkpoint | -5 lines of code | Pitfall 4 — invisible inference bug at swap | Never; safetensors metadata is free |
| Single-seed PaliGemma run | -2x compute on the most expensive run | Pitfall 15 — bellwether comparison has no error bar | Acceptable if compute budget is genuinely tight; document the limitation |
| In-process bootstrap loop | Simpler orchestration | Architecture Anti-Pattern 2 + Pitfall 1 less detectable | Never; architecture explicitly rejects |
| Flat OCR loss weight λ=1.0 at training start | Treats all heads equally | Pitfall 8/9 — OCR damage propagates to seg heads | Never; start at λ=0.1 |

---

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HuggingFace `AutoProcessor` | Use `transforms.Compose` instead — different normalize stats | Always use the model's own `AutoProcessor`; wrap in `Backbone.preprocess()` |
| `bitsandbytes` 4-bit | CUDA version mismatch; symbol-not-found at import | Pin CUDA + bitsandbytes versions in lockfile; preflight check |
| `safetensors` metadata | String-only values; complex objects need JSON-encoding | JSON-encode dataclasses before saving; JSON-decode on load |
| OSM tile renderer | Mercator render + WGS84 label = ~tens-of-meter offset at high lat | Round-trip CRS verification per-sample; reject if tolerance breaks |
| Auto-georef + WorldCover | Use raw cross-correlation peak as registration | Peak-significance ratio threshold + GCP convex-hull coverage threshold + TPS regularization |
| `torch.autocast` mixed precision | Pre-cast inputs to fp16 manually | `autocast` around forward only; let it manage dtypes |
| `torch.compile` | Apply to backbone but recompiles per-input-shape | Don't compile in v1; revisit for inference if it helps |
| External hex-grid app integration | Pass logits, expect downstream to softmax | Deliver calibrated probabilities (TS-3 + TS-6) |
| Bootstrap-loop manifest | Forget to record `registered_via: auto_georef_v0` | Required field in `manifest.json` (architecture Pattern 1) |

---

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| OOM during PaliGemma backward pass | First backward step fails | One-batch dry-run on target VM before scheduling long training | Always for full-finetune; sometimes for QLoRA |
| Activation memory from seg-head feature pyramid | OOM when adding seg-head; forward alone is fine | Gradient checkpointing on backbone; smaller seg-head feature dim | Backbone input ≥ 384 with batch ≥ 4 |
| TPS computation O(n³) in GCP count | Auto-georef takes minutes per map for n>1000 | Cap GCPs at 50–200; sub-sample uniformly | When v0 is "very good" and produces many GCPs |
| Tile-and-stitch wall-clock cost | 4096x4096 input takes minutes per inference | Batch tiles within an image; precompute feather window once | Always for fantasy-map-scale inputs |
| Bootstrap-then-rebuild pipeline serial cost | Bootstrap + label rebuild + train v1 = >24 h | Embarrassingly parallel; bootstrap many maps in parallel | When the unregistered set has > ~100 maps |

---

## "Looks Done But Isn't" Checklist

- [ ] **Loss weights YAML:** validates name→index mapping at startup? Check by deliberately renaming a class in YAML — should `KeyError` at startup, not silently use default
- [ ] **Held-out splits:** verify `splits.json` content is identical between v0 and v1 evaluation runs
- [ ] **Inference preprocessing:** push a fixed reference image through `mapclass.infer` and `mapclass.train`'s validation step; outputs must be byte-identical (pre-softmax) within fp16 epsilon
- [ ] **Bootstrap kill-switch:** v1 NLL > v0 NLL produces an error or warning, not just a higher number in a log
- [ ] **OSM attribution:** appears in inference module README *and* checkpoint metadata
- [ ] **Reliability diagram:** present in eval output for both v0 and v1, both backbones
- [ ] **Tile-stitching seams:** zoom into a 4096x4096 inference output at predicted tile boundaries — should not see hard transitions
- [ ] **OCR detection count:** training-log includes per-batch detection count; values are sane (median < expected text density)
- [ ] **Preflight check:** runs in <30s on the cloud VM and exits 0 before any heavy step
- [ ] **PaliGemma checkpoint:** has Gemma-license attribution string in metadata; not redistributed under inconsistent license
- [ ] **Spot-check renders for EVAL-02:** are produced for *both* backbones (small + PaliGemma), with the same input images, for direct visual comparison
- [ ] **Per-class NLL:** reported for every source × class pair, not just mean
- [ ] **GCP residuals:** logged per auto-registered map; high-residual maps flagged in manifest

---

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| 1 (bootstrap regression) | LOW (ship v0) | Detect via D-5, ship v0 checkpoint, document |
| 2 (EVAL-03 contamination) | MEDIUM (re-run both) | Diff configs, fix the divergence, re-run both backbones |
| 3 (loss-weight schema mis-align) | HIGH (one wasted training run) | Detect via per-class NLL anomaly, fix YAML, retrain |
| 4 (tokenizer/processor mismatch) | LOW–HIGH (depends which side) | Detect via inference-vs-training diff; fix preprocessing path; retrain only if training side |
| 5 (held-out leak) | HIGH (full retrain) | Recompute split by hash; identify leaked samples; remove from train; retrain |
| 6 (PaliGemma OOM) | LOW (config tweak) | Reduce batch / resolution; QLoRA + checkpointing; re-run |
| 7 (LoRA rank wrong) | MEDIUM (re-run small backbone) | Try frozen baseline + 2 LoRA ranks; pick best by NLL |
| 8 (OCR false positives) | LOW–MEDIUM | Lower λ, mine hard negatives, re-run; or drop OCR for v1 |
| 9 (OCR LM bias) | LOW | Document, defer multi-language to milestone 2 |
| 10 (class imbalance) | MEDIUM (one retrain) | Add focal CE; retrain; verify per-class NLL |
| 11 (calibration) | LOW (post-hoc) | Fit temperature on validation set; embed in checkpoint; no retrain |
| 12 (tile seams) | LOW | Add feathered blending; inference-side fix only |
| 13 (TPS pathology) | MEDIUM | Reject high-residual registrations; re-run bootstrap; possibly retrain v1 |
| 14 (silent no-match) | MEDIUM | Add peak-significance threshold; re-run bootstrap |
| 15 (no seeds) | MEDIUM (re-run with seeds) | Add seeding utility; document pre-seed numbers as not-comparable |
| 16 (OSM attribution) | LOW | One-commit fix to README + metadata |
| 17 (OSM CRS / freshness) | LOW–MEDIUM | Fix CRS handling; document freshness limitation |
| 18 (single-author drift) | LOW per incident, accumulates | Restore from `.planning/` docs; bisect to find regression |
| 19 (repo drift) | LOW | Add lockfile; preflight check; sync VM |

---

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls. Phase numbers are placeholders; the roadmapper will assign actual phase IDs.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1 (bootstrap regression) | GEOREF-02 + EVAL-01 phase | D-5 monitor implemented; gate v1 ship on v1 NLL > v0 NLL |
| 2 (EVAL-03 contamination) | EVAL-03 phase / TRAIN-01+TRAIN-02 phase | Pre-registered protocol committed before either training run; hash-equality assertion at compare time |
| 3 (loss-weight schema) | DATA-06 / "sample contract" phase | `LossWeights.load()` validator passes startup test; per-class NLL in eval shows no anomalies |
| 4 (processor mismatch) | MODEL-01 / "Backbone interface" phase | Backbone ABC requires `preprocess()`; reference-image hash check |
| 5 (held-out leak) | "Sample contract" phase + TRAIN-01 | `splits.json` hash-stable; eval-time overlap assertion |
| 6 (PaliGemma OOM) | MODEL-04 / TRAIN-01 PaliGemma phase | One-batch dry-run on cloud VM passes |
| 7 (LoRA rank) | MODEL-01 / TRAIN-01 phase | Frozen baseline + LoRA r=16 are both run; pick by NLL |
| 8 (OCR false positives) | MODEL-02 phase | OCR detection-count diagnostic; λ=0.1 default; sub-pipeline-first integration |
| 9 (OCR LM bias) | MODEL-02 phase | Spot-check on 4 fantasy maps; documented limitation |
| 10 (class imbalance) | MODEL-03 + DATA-06 phase | Focal CE used; per-class NLL reported in eval |
| 11 (calibration) | EVAL-01 / MODEL-03 phase | TS-6 reliability diagram in eval; temperature scaling metadata in checkpoint |
| 12 (tile seams) | SHIP-01 / inference-module phase | Feathered overlap implemented; visual seam-map diagnostic |
| 13 (TPS pathology) | GEOREF-01 phase | Sanity-check mode passes on known-good maps; GCP-coverage threshold |
| 14 (silent no-match) | GEOREF-01 phase | Peak-significance threshold tuned on sanity-check set |
| 15 (no seeds) | TRAIN-01 phase + cross-cutting | `set_global_seed()` utility called from every entry point; 3-seed budget for EVAL-03 |
| 16 (OSM attribution) | DATA-05 phase + SHIP-01 | Inference module README + checkpoint metadata both contain OSM credit |
| 17 (OSM CRS / freshness) | DATA-05 phase | Per-sample CRS round-trip assertion; per-source NLL spot-check |
| 18 (bus factor) | Cross-cutting; addressed by TS-7 / TS-8 / planning docs | TS-7/TS-8 features implemented; `.planning/` docs current |
| 19 (repo drift) | TRAIN-01 / SHIP-01 (lockfile + preflight) | `uv.lock` committed; preflight script passes on cloud VM |

---

## Sources

### ML literature on the pitfall mechanisms

- [Pseudo-Labeling and Confirmation Bias in Deep Semi-Supervised Learning (Arazo et al., 2019, arXiv:1908.02983)](https://arxiv.org/abs/1908.02983) — the canonical bootstrap-error-compounding paper (Pitfall 1)
- [Mask-TS Net: Mask Temperature Scaling Uncertainty Calibration (2024, arXiv:2405.05830)](https://arxiv.org/abs/2405.05830) — recent evidence that segmentation models are systematically miscalibrated (Pitfall 11)
- [Loss Functions in the Era of Semantic Segmentation: A Survey and Outlook (2024, arXiv:2312.05391)](https://arxiv.org/html/2312.05391v1) — focal CE, generalised dice, class imbalance (Pitfall 10)
- [Unified Focal Loss (Yeung et al., 2022, arXiv:2102.04525)](https://arxiv.org/abs/2102.04525) — class-imbalanced segmentation losses (Pitfall 10)
- [Tiling artifacts and trade-offs of feature normalization (2025, arXiv:2503.19545)](https://arxiv.org/html/2503.19545v1) — tile-boundary effects (Pitfall 12)
- [Tile-Based Segmentation Inference for Images Larger than GPU Memory (NIST 2024)](https://nvlpubs.nist.gov/nistpubs/jres/126/jres.126.009.pdf) — overlap and stitching strategies (Pitfall 12)
- [Tiling artifact tutorial (Buglakova 2024)](https://buglakova.github.io/tiling_artifacts_tutorial/) — practical sliding-window inference fixes (Pitfall 12)
- [GlotOCR Bench: OCR Models Still Struggle Beyond a Handful of Unicode Scripts (2026, arXiv:2604.12978)](https://arxiv.org/html/2604.12978v1) — OCR LM-bias / hallucination on unfamiliar scripts (Pitfall 9)
- [ABCNet: Real-time Scene Text Spotting with Bezier-Curve Network (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_ABCNet_Real-Time_Scene_Text_Spotting_With_Adaptive_Bezier-Curve_Network_CVPR_2020_paper.pdf) — curved text limitations (Pitfall 8)
- [CRAFT-pytorch (clovaai)](https://github.com/clovaai/CRAFT-pytorch) — the canonical CRAFT implementation (Pitfall 8)
- [Towards Escaping from Language Bias and OCR Error (arXiv:2203.12929)](https://arxiv.org/abs/2203.12929) — OCR + LM-bias interaction (Pitfall 9)
- [Leak, Cheat, Repeat: Data Contamination in LLMs (2024, arXiv:2402.03927)](https://arxiv.org/html/2402.03927) — analogue for held-out leakage (Pitfall 5)
- [Image-LoRA / Towards Minimal Fine-Tuning of VLMs (2025, arXiv:2512.19219)](https://arxiv.org/html/2512.19219) — LoRA target-module selection for small VLMs (Pitfall 7)
- [Empirical Evaluation of LoRA on Vision-Language Models (PMC12730038)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12730038/) — practical LoRA configurations (Pitfall 7)

### Practical / VRAM / framework

- [How much VRAM do I need for fine-tuning (Modal)](https://modal.com/blog/how-much-vram-need-fine-tuning) — empirical VRAM numbers (Pitfall 6)
- [How Much VRAM for LLM Fine-Tuning in 2026 (VRLA)](https://vrlatech.com/how-much-vram-do-you-need-for-llm-fine-tuning-in-2026/) — current-year practical guidance (Pitfall 6)
- [PaliGemma fine-tuning tutorial (DigitalOcean)](https://www.digitalocean.com/community/tutorials/finetune-paligemma) — concrete VRAM and config numbers
- [unsloth issue #4504 — fine-tuning OOM despite advertised numbers](https://github.com/unslothai/unsloth/issues/4504) — VRAM advertised vs. actual gap
- [HF discussion: fine-tuning ViT at higher resolution](https://discuss.huggingface.co/t/fine-tuning-image-transformer-on-higher-resolution/22623) — image-size / patch-grid mismatches (Pitfall 4)
- [LLaVA fine-tuning image-token mismatch (HF transformers issue #36002)](https://github.com/huggingface/transformers/issues/36002) — concrete swap-time bug (Pitfall 4)
- [Thin plate spline (Wikipedia)](https://en.wikipedia.org/wiki/Thin_plate_spline) — TPS overshoot, ill-conditioning (Pitfall 13)
- [GDAL gdalwarp documentation, TPS option](https://gdal.org/en/stable/programs/gdalwarp.html) — practical TPS GCP bounds (Pitfall 13)
- [Warping digital images using thin plate splines (academia.edu)](https://www.academia.edu/29654283/Warping_digital_images_using_thin_plate_splines) — pathological 2D overshoot (Pitfall 13)
- [Rasterio TPS GCP transformer discussion](https://github.com/rasterio/rasterio/discussions/2981) — TPS in the project's own toolchain

### Licensing

- [PaliGemma license blog (HF)](https://huggingface.co/blog/paligemma) — Gemma license, PT vs FT vs Mix checkpoint distinctions (Pitfall on training the bellwether checkpoint)
- [PaliGemma 2 license blog (HF)](https://huggingface.co/blog/paligemma2) — confirms commercial-use rights for fine-tuned PT-derived checkpoints under Gemma terms
- [Gemma terms of use HN discussion](https://news.ycombinator.com/item?id=40371631) — community awareness of non-FOSS terms
- [OSM Foundation Licence/Attribution Guidelines](https://osmfoundation.org/wiki/Licence/Attribution_Guidelines) — attribution obligation for ML predictions (Pitfall 16)
- [OSM Wiki Machine Learning page](https://wiki.openstreetmap.org/wiki/Machine_learning) — explicit guidance on training-set ODbL status and prediction-side exemption
- [OSM Foundation Collective Database Guideline](https://osmfoundation.org/wiki/License/Community_Guidelines/Collective_Database_Guideline_Guideline) — derivative-database vs collective-database distinction
- [OSM Wiki Open Database License](https://wiki.openstreetmap.org/wiki/Open_Database_License) — ODbL share-alike

### Project-internal sources

- `/home/drdreadknee/mapclass/.planning/PROJECT.md` — locked v1 scope (REQ-IDs cited throughout)
- `/home/drdreadknee/mapclass/.planning/research/ARCHITECTURE.md` — integration architecture for the new components
- `/home/drdreadknee/mapclass/.planning/research/FEATURES.md` — TS-6 calibration explicitly called out; D-5 bootstrap quality monitor; TS-3 inference API contract
- `/home/drdreadknee/mapclass/.planning/codebase/CONCERNS.md` — items 1 (licensing), 4 (no tests), 8a (no seeds), 8b (no dataset versioning), 8c (no lockfile), 8d (broad exception swallowing) referenced rather than duplicated
- `/home/drdreadknee/mapclass/.planning/codebase/TESTING.md` — six untested high-risk areas; gap-items 1 (taxonomy alignment), 5 (CRS round-tripping), 6 (loss-weight schema) cited
- `/home/drdreadknee/mapclass/scripts/biome_mapping.py` — canonical `LANDCOVER_CLASSES` is the linchpin for Pitfall 3
- `/home/drdreadknee/mapclass/scripts/historical/label.py` — `HISTORICAL_LC_WEIGHTS` is the seed of the YAML loss-weights schema (Pitfall 3)

---

*Pitfalls research for: multi-source-trained dense-prediction VL+OCR+heads model with auto-georef bootstrap (new components only; data-pipeline pitfalls in CONCERNS.md)*
*Researched: 2026-05-08*
