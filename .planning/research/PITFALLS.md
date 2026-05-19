# Pitfalls Research

**Domain:** Adding CLIP, PaliGemma, and a plain ViT as comparison models to an EXISTING SigLIP-2 dynamic-LRP harness on a FROZEN pinned stack (torch==2.7.1 / transformers==4.52.3 / vendored dynamicLRP SHA 405e742) — MapClass v1.1
**Researched:** 2026-05-19
**Confidence:** HIGH — grounded in the v1.0 01-03 shipped behavior + recorded decisions (Fallback Ladder declined, VENDOR_SHA intact, caught-finding pattern, 4.326 GB forward VRAM), the v1.1 STACK/ARCHITECTURE/FEATURES research, the `lrp.py` tensor-identity mechanism, and CLAUDE.md "What NOT to Use" pins.

> These are **integration** pitfalls — mistakes specific to generalizing THIS
> working single-model harness across heterogeneous model APIs without breaking
> the two load-bearing v1.0 invariants or the frozen stack. Phase numbers refer
> to ARCHITECTURE.md "Suggested Build Order" (1: adapter contract; 2:
> parameterize overlay/attribution; 3: SigLIP-2 adapter / regression oracle; 4:
> ViT adapter; 5: CLIP adapter; 6: PaliGemma adapter; 7: mirror/config registry +
> comparison notebook).

## Critical Pitfalls

### Pitfall 1: Breaking the requires_grad tensor-identity invariant when generalizing the forward across heterogeneous model APIs

**What goes wrong:**
The `pixel_values` tensor that `build_inputs` marks `requires_grad_()` must be the *exact same Python object* that flows into `model(**inputs)` and into `engine.params_to_interpret`. `lrp.py`'s `make_graph_iter(root_nodes, params_to_interpret, …)` resolves `params_to_interpret` by walking the live `grad_fn` chain and matching by **tensor object identity** — not by value or shape. Any `.clone()`, `.detach()`, re-`.to(device)`, dtype cast, or in-place normalize *after* `requires_grad_()` substitutes a new object; the engine then cannot find the input in the graph and produces wrong/empty relevance or an opaque error. v1.0 solved this for SigLIP-2 with a single-scope `attribute()` and explicit "no clone/detach/re-.to() after this line" docstrings. Generalizing across 4 model APIs is exactly where this regresses, because each model's processor returns differently-shaped/typed tensors and the "natural" per-model fix is a `.to()` / cast / reshape *after* the grad flag is set.

**Why it happens:**
- CLIP/ViT/PaliGemma processors differ from SigLIP-2's; a developer "normalizes" the per-model output (cast to bf16 for PaliGemma, `.to(device)` after building inputs, `.unsqueeze`/reshape) and unknowingly does it *after* `requires_grad_()`.
- Moving `forward + engine.run` *into the adapter* for "encapsulation" (Architecture Anti-Pattern 1) lets a scope close or device move slip between build and run.
- bf16 PaliGemma especially tempts a post-hoc `.to(torch.bfloat16)` on `pixel_values`.

**How to avoid:**
- Adapter contract makes `build_inputs` the **last** place `pixel_values` is touched: do device move + dtype cast + reshape *before* `.requires_grad_()`, and `.requires_grad_()` is the final statement on that tensor (encode this in the `ModelAdapter` docstring and a `test_adapters.py` assertion).
- `attribution.attribute()` — NOT the adapter — owns `forward + engine.run` in ONE scope; the adapter supplies only `forward()` and `attribution_target()` (Architecture Pattern 2 / Anti-Pattern 1).
- Add an identity guard in `attribute()`: assert `inputs["pixel_values"] is the_tensor_passed_to_forward` and `engine.params_to_interpret[0] is inputs["pixel_values"]` (object identity `is`, not `==`).
- Carry v1.0's forbidden-token grep into each adapter's `attribution_target` (the cross-cutting module no longer names a target).

**Warning signs:**
- Relevance is all-zero, uniform, or shaped unlike `pixel_values` while the forward succeeds (logit prints fine).
- Engine raises a "param not found in graph" / node-resolution error only for the new models, not SigLIP-2.
- Any `.to()`, `.clone()`, `.detach()`, `.half()/.bfloat16()`, `.contiguous()`, reshape on `pixel_values` appearing *after* `requires_grad_()` in a diff.

**Phase to address:**
Phase 1 (bake the constraint into the `ModelAdapter` contract + protocol-conformance test) and Phase 3 (SigLIP-2 regression oracle must reproduce the *identical* recorded finding, proving the seam preserved identity). Re-verify per adapter in Phases 4–6.

---

### Pitfall 2: Hardcoded 27×27 / 14 / 384 grid geometry silently misaligning or crashing overlays for CLIP/ViT/PaliGemma

**What goes wrong:**
v1.0's `overlay.py` has module-level `PATCH_SIZE=14, IMG_DIM=384, GRID=27` and a literal `r[:378,:378].reshape(27,14,27,14).mean((1,3))`. Each new model has different geometry: plain `vit-base-patch16-224` → 224/16 = 14×14 (+CLS); CLIP `clip-vit-large-patch14`@224 → 16×16 (+CLS); PaliGemma's SigLIP vision tower @224/14 → 16×16. Feeding a non-SigLIP relevance map through the hardcoded 27/14/384 reshape either **crashes** (`RuntimeError: shape '[27,14,27,14]' is invalid for input of size …`) or, worse, **silently "works"** on a tensor whose size happens to factor, producing a heatmap that is spatially scrambled relative to the map — a comparison that *looks* fine but is geometrically lying. CLIP/ViT also have a CLS token (SigLIP-2 MAP-pool does not); naïvely reshaping `num_tokens` that includes the +1 CLS row off-by-one-shifts every patch.

**Why it happens:**
- "We already have overlay.py" — the most-overlooked dependency (FEATURES.md dependency note). Geometry feels like a solved problem because it is, *for SigLIP-2 only*.
- The CLS token: SigLIP-2 so400m has *no* CLS (729 = 27×27 pure patches); CLIP/ViT prepend a CLS → token count is `grid²+1`, and the relevance must drop the CLS index before the square reshape.
- A wrong reshape that doesn't crash is invisible without a known-answer fixture.

**How to avoid:**
- Geometry is a per-adapter `PatchGeometry(patch_size, img_dim)` value object (Architecture Pattern 3); `overlay.py` derives `grid/valid_dim/edge_discard` from it — zero geometry literals remain in `overlay.py`.
- The adapter (not overlay) is responsible for **stripping the CLS token** from the relevance/patch sequence before handing a pure `grid²` map to `to_patch_grid`; document per model whether a CLS exists (SigLIP-2: no; CLIP/ViT: yes; PaliGemma vision tower: confirm from its processor config, do not assume).
- Parametrize `test_overlay_grid.py` over geometry: lock the v1.0 SigLIP numbers (`PatchGeometry(14,384).grid==27`, `.edge_discard==6`) *and* add 14×14 (ViT) and 16×16 (CLIP) cases with a synthetic gradient whose post-reshape pattern is analytically known (detects scrambling, not just crashes).
- Derive `patch_size`/`img_dim` from the model's own processor/config at adapter-load time and assert it against the adapter's declared `PatchGeometry` (catches a model card vs. checkpoint mismatch).

**Warning signs:**
- `RuntimeError: shape '[...]' is invalid for input of size N` in `to_patch_grid` for a new model only.
- A heatmap that renders but whose hot region is rotated/offset/tiled relative to obvious map features (a river overlay nowhere near the river).
- Relevance token count is `grid²+1` (CLS present) but you reshaped `grid²`.

**Phase to address:**
Phase 2 (parameterize `overlay.py` by `PatchGeometry`; existing 16 tests must still pass with geometry injected). Phase 4 first exercises non-SigLIP geometry end-to-end (ViT 14×14) — the earliest point a *positive* heatmap validates alignment; Phases 5–6 add CLIP/PaliGemma geometry.

---

### Pitfall 3: Wrong or ambiguous attribution target per architecture (pooled embeds, no-text ViT, PaliGemma answer-token choice)

**What goes wrong:**
The harness silently assumes all 4 models accept SigLIP-2's `logits_per_image[0,0]`. They do not (FEATURES.md "Feature-Defining Question"):
- **ViT has no text tower** — there is no query-conditioned similarity at all. Attributing the pooled image embedding or its norm yields generic image-saliency, not "where is the query", breaking cross-model comparability and silently re-committing the exact Pitfall-B v1.0 forbids by grep.
- **PaliGemma is generative** — no `logits_per_image`; the target is a single vocab logit `logits[0, answer_pos, answer_token_id]`. Choosing `answer_pos`/`answer_token_id` is a per-query modeling decision (teacher-forced single-token answer); picking it wrong (e.g. the prompt's last token, or argmax of an unconditioned position) produces a heatmap explaining the wrong thing.
- **CLIP** *does* take `logits_per_image[0,0]` verbatim — but `model.get_image_features()` / pooled embedding is the seductive wrong target (query-independent), the identical mistake called out for SigLIP-2 in v1.0.

The deeper trap: even when every target "works", the four panels then compare a cosine similarity vs. a token logit vs. a class logit **without saying so** — an apples-to-oranges grid the reader misreads as apples-to-apples.

**Why it happens:**
- The v1.0 mental model ("attribute `logits_per_image`") generalizes silently and incorrectly.
- ViT's missing text tower is discovered late; the path of least resistance is to attribute *something* image-level (pooled embed) to "make the panel render".
- PaliGemma's answer-token choice has no obvious default; a guess gets hardcoded.

**How to avoid:**
- The adapter contract is `model_id → {load, forward, select_target, declare_semantics}` (FEATURES.md) — `declare_semantics` returns a plain-words string ("image–text similarity" / "logit of answer token 'yes'" / "ImageNet class logit 'map'") **rendered as the panel subtitle**. The honesty label is table stakes, not a nicety.
- ViT: map the locked query string to a fixed agreed ImageNet class (document the query→class mapping as an adapter decision); attribute `logits[0, class_id]`. Label the panel "class-conditioned, not text-conditioned".
- PaliGemma: build a `"<image> {query}"` prompt, teacher-force a single-token answer, attribute `logits[0, answer_pos, answer_token_id]` with `answer_pos`/`answer_token_id` explicit and documented per query.
- Keep v1.0's forbidden-token guard (no `get_image_features` / image-embed-norm) but move it into each adapter's `attribution_target` (Architecture Pattern 2 note).
- Resolve and document all four targets *before* writing the comparison loop — this is the one item warranting a roadmap decision step (FEATURES.md dependency note), not a discover-while-coding choice.

**Warning signs:**
- A ViT/PaliGemma heatmap that does not change when the query string changes (query-independent → wrong target).
- Any adapter calling `get_image_features`, `.pooler_output`, or attributing an embedding norm.
- A panel with no `declare_semantics` subtitle, or all panels labeled identically despite different target types.

**Phase to address:**
Phase 1 (the `select_target`/`declare_semantics` slots in the contract). The *decision* of each target is resolved up-front (roadmap decision step, before Phase 4). Verified per model: Phase 3 (SigLIP-2 = v1.0 path), Phase 4 (ViT class logit + query→class doc), Phase 5 (CLIP `logits_per_image`), Phase 6 (PaliGemma answer-token).

---

### Pitfall 4: PaliGemma gated weights / HF token / Gemma license tripping the GCS mirror

**What goes wrong:**
Every `google/paligemma*` repo is GATED — `snapshot_download("google/paligemma2-3b-pt-224")` returns 401/403 anonymously, even though the repo is "publicly listed". v1.0's `mirror_model.py` used an anonymous download (CLIP and ViT are fine anonymous). The mirror step silently fails *only for PaliGemma*, or — worse — the generalized mirror loop aborts mid-registry and leaves an incomplete `models/paligemma2-3b-pt-224/` in GCS that the runtime loader then half-loads with a confusing `from_pretrained` error far from the real cause. A second-order issue: the mirrored Gemma-licensed weights now live in the GCS bucket (a license/redistribution consideration to flag for a shared bucket; acceptable for a private single-researcher bucket but worth recording).

**Why it happens:**
- CLIP/ViT mirror anonymously and work; PaliGemma's gating is invisible until the download 401s.
- The HF account behind the token must have **visited the model page and accepted the Gemma license once** — a token alone is insufficient if terms were never accepted.
- Token handling temptations: committing `HF_TOKEN`, baking it into the GCS-loaded model dir, or needing it at runtime (it must be mirror-time only — runtime is GCS-only, Pitfall 7).

**How to avoid:**
- Generalize `mirror_model.py` to `(repo_id, gcs_prefix, token=None)`; CLIP/ViT pass `token=None`, PaliGemma passes `os.environ["HF_TOKEN"]` (STACK.md). The token is consumed on the **one-time mirror VM only**.
- Operational precondition (document in the mirror runbook): accept Gemma terms at the model page while logged into the HF account, then `export HF_TOKEN=hf_…` for that account, *before* running the PaliGemma mirror.
- Make the mirror loop per-repo fault-isolated and idempotent: a PaliGemma auth failure must not corrupt or half-write its GCS prefix, and must not abort the CLIP/ViT mirrors; verify with a size/manifest check after upload.
- Never commit the token; never write it into the model dir uploaded to GCS; runtime loaders must not read HF at all (reproducibility invariant).
- Flag the Gemma-license-in-bucket consideration in the milestone notes (not a blocker for a private bucket; a real consideration if the bucket is shared).

**Warning signs:**
- `401 Client Error` / `Cannot access gated repo` / `Repository Not Found` only for `google/paligemma*`.
- `from_pretrained` at runtime failing with missing-config/missing-weight for PaliGemma while SigLIP-2/CLIP/ViT load (→ a partial mirror from a failed gated download).
- `HF_TOKEN` appearing in git, in the GCS model dir, or being required by the runtime notebook.

**Phase to address:**
Phase 7 (`mirror_model.py` registry generalization). The token operational requirement is a one-time mirror-VM concern; the runtime-from-GCS invariant is verified in the Phase 7 notebook end-to-end run.

---

### Pitfall 5: VRAM blowup loading 4 models — must sequential-load + del + empty_cache; PaliGemma-3B likely OOMs (degrade gracefully, do not crash)

**What goes wrong:**
The L4 has 24 GB. v1.0 measured the SigLIP-2 *forward-only* peak at 4.326 GB — and that is a lower bound: the dynamicLRP relevance pass retains forward activations *on top* (Pitfall E / Promise activation retention), and that peak was never measurable for SigLIP-2 because its relevance pass never completes. Holding 2+ ~400M-class VLMs resident, or loading all four up front into a `{model_id: model}` cache, OOMs. PaliGemma-3B (~7.5× the so400m ceiling) loaded in bf16 plus its LRP activation retention very plausibly exceeds 24 GB *by itself* — and that OOM is an **expected, recordable comparison datum** (consistent with "coverage/VRAM failure = result, not bug"), but ONLY if it degrades into a labeled "no heatmap" tile instead of an uncaught CUDA OOM that aborts the notebook and kills the other three panels.

**Why it happens:**
- The natural `model_loader` extension is `{model_id: (model, processor)}` singletons — which, if the notebook loads all four before the loop, co-resides them and OOMs.
- `torch.cuda.empty_cache()` after `del model` is easy to forget in a `try/except`; without it freed-but-cached memory accumulates across iterations.
- CUDA OOM raises *outside* a narrow `except` if the loop's try/except doesn't catch broadly, or frees in `except` but not `finally`.
- PaliGemma OOM is treated as a failure to fix (bigger GPU, quantize) rather than a recordable datum.

**How to avoid:**
- The notebook loop is **strictly sequential**: `load → build_inputs → attribute → render → del model → torch.cuda.empty_cache()` per adapter, freeing in a `finally:` (Architecture Pattern 4 / data-lifecycle note). Never hold two VLMs.
- Sequential load/free is a *required lifecycle*, not an optimization — the per-model_id singleton cache mostly serves same-model re-runs in one kernel; the loop deliberately evicts.
- The per-model `try/except Exception` must catch CUDA OOM too (it surfaces as `RuntimeError`/`torch.cuda.OutOfMemoryError`); the `finally:` frees regardless so the *next* model still has VRAM.
- Treat a PaliGemma OOM as a rendered "no heatmap (OOM — model exceeds L4)" tile — a recorded result, exactly like an op-coverage gap. Do NOT escalate to quantization/bigger GPU (that is Pitfall 7 scope creep).
- Order the registry SigLIP-2 → ViT → CLIP → PaliGemma so the riskiest (3B) runs last; a late OOM cannot starve earlier panels (they already rendered).

**Warning signs:**
- `torch.cuda.OutOfMemoryError` / `CUDA out of memory` — especially escalating across loop iterations (→ missing `empty_cache()` in `finally`).
- Notebook aborts at model 3/4 with the earlier panels already drawn (→ OOM not caught, or freed in `except` not `finally`).
- `nvidia-smi` showing two model footprints simultaneously resident.

**Phase to address:**
Phase 7 (the comparison notebook loop — sequential load/free + broad catch + `finally` free is the loop skeleton). The single-model VRAM characteristic is known from Phase 3 (v1.0 4.326 GB). Each adapter (Phases 4–6) loads in the appropriate dtype (PaliGemma bf16).

---

### Pitfall 6: Accidentally violating the frozen pins to satisfy a new model

**What goes wrong:**
A new model appears to "need" a newer dependency and someone bumps it — the single most dangerous integration mistake. The engine traverses torch autograd internals; a `torch`/`torchvision` bump silently drifts autograd Node names/graph structure and breaks LRP propagation invisibly (not a clean error — wrong/empty relevance). A `transformers` bump to "get newer PaliGemma" is both unnecessary (both `paligemma` and `paligemma2` are already in 4.52.3 — STACK.md verified) and diverges from dynamicLRP's hard pin. Adding `sentencepiece` for Gemma's tokenizer (unnecessary — 4.52.3's `PaliGemmaProcessor` uses `GemmaTokenizerFast`, tokenizers-backed) or adding `accelerate` unconditionally perturbs the frozen resolve and can re-pin torch/transformers transitively.

**Why it happens:**
- "PaliGemma 2 is newer, surely it needs newer transformers" — false; 4.52.3 ships it (STACK.md verified against v4.52.3 docs).
- A `pip install accelerate`/`sentencepiece` with no version cap lets the resolver pull a build that re-pins torch/transformers.
- The pin's danger is invisible: a torch bump does not error, it produces *subtly wrong* relevance — the worst failure for a comparison whose entire output is "does the heatmap look right".

**How to avoid:**
- `requirements.txt` is UNCHANGED for v1.1 — net dependency delta is ZERO (STACK.md Executive Finding). CLIP and ViT add no packages; PaliGemma adds an *operational* HF-token requirement, not a runtime package.
- Hard rule from CLAUDE.md "What NOT to Use" (D-08): never bump `transformers` (use 4.52.3 as-is — it already has paligemma2), never bump `torch`/`torchvision`, do not add `sentencepiece` (GemmaTokenizerFast ships with transformers), do not add `accelerate` unconditionally.
- `accelerate` is a *conditional fallback only* (if 3B bf16 OOMs host RAM at materialization), pinned `>=0.26,<1.1`, gated behind a human-verify checkpoint that confirms `pip` proposes NO torch/transformers change before accepting.
- Add a CI/pre-flight assertion: `transformers.__version__ == "4.52.3"`, `torch.__version__` startswith `2.7.1`, and `VENDOR_SHA == 405e74243ecaa1f615f418fdc8ba24c3c5889b1e` — fail loudly if drifted.

**Warning signs:**
- Any diff to `requirements.txt`, or a `pip install` without an exact/capped version.
- SigLIP-2 (the regression oracle) producing a *different* finding than the v1.0 recorded one (→ stack drift, not a model issue).
- `pip`'s install plan listing torch/torchvision/transformers as "would change".

**Phase to address:**
Phase 1 (add the pin-assertion pre-flight to the test suite). Enforced continuously; specifically re-verified at Phase 3 (SigLIP-2 regression oracle = the canary for stack drift) and at the Phase 6 PaliGemma adapter (the model most likely to tempt a bump).

---

### Pitfall 7: "Fixing" a coverage gap — the scope violation the user explicitly declined

**What goes wrong:**
A model produces no heatmap (op-coverage gap) and it *looks like a bug*, so someone adds a custom dynamicLRP Promise, tries `use_attn_lrp`/pre-pool, vendors LXT, or falls back to captum IG to "make it work". This destroys the deliverable: a model dynamicLRP cannot traverse is **the recorded comparison result** (PROJECT.md Out of Scope; 01-03 Key Decisions). The user reviewed exactly this for SigLIP-2 at the 01-03 human-verify checkpoint and **explicitly declined the entire Fallback Ladder** (no custom Promise, no pre-pool/use_attn_lrp, no LXT, no captum). A custom Promise also requires editing `third_party/dynamicLRP`, breaking the byte-intact `VENDOR_SHA` provenance (threat T-01-SC3). Patching to get CLIP/PaliGemma/ViT "working" repeats the declined mistake on new models and defeats the experiment (CLIP-vs-SigLIP-2's pooling-head contrast is *the finding* only if neither is engineered around).

**Why it happens:**
- A "no heatmap" panel viscerally reads as broken; the engineering instinct is to fix it.
- The Fallback Ladder is documented in the codebase (`attribution.py` docstring) and looks like an available, sanctioned path — it is the *declined* path.
- "Just one small Promise for CLIP" feels harmless but is the exact T-01-SC3 vendored-SHA deviation that was offered and not taken.

**How to avoid:**
- Per-model coverage gap = recorded result, never an engine patch (Architecture Anti-Pattern 4). Catch the engine error, render the labeled "no heatmap (op-coverage gap), failed at <node>" tile, move on.
- `third_party/dynamicLRP` is byte-unmodified; `VENDOR_SHA` must equal `405e74243ecaa1f615f418fdc8ba24c3c5889b1e` — add a hard assertion (shared with Pitfall 6's pre-flight) and a `git diff --quiet third_party/dynamicLRP` check.
- The Fallback Ladder must not be re-attempted without a *new explicit user decision* (recorded in `attribution.py` docstring and the notebook from v1.0 — preserve those notes).
- Anti-features are out of scope by explicit decision: custom Promise, pre-pool/use_attn_lrp, LXT, captum IG fallback, op-coverage report table, metrics, sweep, LRP tuning (FEATURES.md Anti-Features).

**Warning signs:**
- Any edit under `third_party/dynamicLRP/`, or `VENDOR_SHA` ≠ the pinned value, or `git diff` non-empty there.
- New code paths invoking `use_attn_lrp`, a custom Promise registration, LXT imports, or `captum`.
- A "no heatmap" model being treated as a task to resolve rather than a datum to render.

**Phase to address:**
Phases 3–6 (each model adapter — when its coverage gap surfaces, it is rendered as a finding, not patched). The VENDOR_SHA + no-`third_party`-diff assertion lands in Phase 1's pre-flight and is enforced continuously.

---

### Pitfall 8: Notebook not reaching exit 0 when a model fails — uncaught traceback vs. caught FINDING tile

**What goes wrong:**
The v1.1 milestone is *done* on exactly this behavior: the notebook runs headless top-to-bottom, exit 0, with the 2×2 grid rendered, and any failing model showing a labeled "no heatmap" tile — **not** an uncaught traceback that aborts nbconvert and kills the remaining panels. The dynamicLRP engine fails as different exception types depending on the uncovered op (`RuntimeError` for SigLIP-2's `SplitWithSizesBackward0`, but `IndexError`/`TypeError` for other targets/ops); a too-narrow `except RuntimeError` lets a different model's `IndexError` escape and abort the notebook. CUDA OOM (Pitfall 5) and a missing/partial PaliGemma mirror (Pitfall 4) are *also* failure modes that must land in the caught-tile path, not crash. The v1.0 caught-finding pattern (Cell 5: catch the engine error, render a FINDING cell, exit 0) is proven for *one* model and must be generalized over a 4-model loop *without* a broad `except` masking unrelated real bugs.

**Why it happens:**
- The v1.0 pattern caught `RuntimeError` specifically; generalizing without widening to `except Exception` lets other models' different exception types escape.
- Freeing in `except` but not `finally` means a model that OOMs and is caught still starves the next model (Pitfall 5 interaction).
- A broad `except Exception` *can* mask an unrelated bug (e.g. a real overlay shape error) as if it were an op-coverage finding.
- nbconvert non-zero exit on any uncaught cell exception — easy to not verify headless.

**How to avoid:**
- Per-model loop wraps `load → build_inputs → attribute → to_patch_grid → composite` in `try/except Exception` (broad — the engine's failure type varies), rendering a labeled tile with `type(exc).__name__` + the salient node/op from `repr(exc)` as caption text (Architecture Pattern 4). Free in `finally:` (ties to Pitfall 5).
- Mitigate the broad-except masking risk: run the per-model coverage probe (`LRPEngine.get_model_operations`) first and print the op count as a recorded artifact (v1.0 Cell-3 pattern, now per-model); capture full `traceback` into the tile caption so a real bug is visible, not silently swallowed.
- The notebook is generator-authored (`_build_02_multimodel.py` is source of truth; `.ipynb` is the committed artifact) and verified headless via nbconvert with the `mapclass` kernel: assert exit 0, 0 cell errors, ≥4 tiles rendered (the milestone done-gate, mirroring v1.0's `_build_01` discipline).
- Distinguish the two states explicitly in the tile: an *expected* finding (op-coverage gap / OOM, the data) vs. an *unexpected* error — the caption should make a real bug recognizable, not disguised as a finding.

**Warning signs:**
- nbconvert exits non-zero, or a panel is blank with the traceback in cell output instead of a captioned tile.
- `except RuntimeError:` (too narrow) instead of `except Exception:`.
- A "no heatmap" tile whose caption is a generic message that would hide a genuine shape/identity bug (Pitfalls 1–2) as if it were an op-coverage result.
- The notebook only verified interactively, never headless.

**Phase to address:**
Phase 7 (the comparison notebook is built last and is the milestone done-gate; the headless exit-0 + ≥4-tiles nbconvert check IS the milestone completion criterion). The pattern is proven in Phase 3 (SigLIP-2 reproduces the v1.0 caught-finding).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Copy `overlay.py` → `overlay_clip.py` with a new `GRID` constant per model | Fast — no signature churn | Duplicates the load-bearing D-09 signed/zero-centered logic; a fix drifts across N copies; geometry bugs multiply | **Never** — geometry must be `PatchGeometry`-injected (Architecture Anti-Pattern 2) |
| Put `forward + engine.run` inside the adapter for "encapsulation" | Cleaner-looking adapter | Risks a scope-close/clone/`.to()` between build and run → breaks the tensor-identity invariant invisibly | **Never** — `attribute()` owns the one-scope run (Architecture Anti-Pattern 1) |
| Hardcode PaliGemma's `answer_pos`/`answer_token_id` to a guess to "get a panel" | Panel renders now | Attributes the wrong thing; a query-independent map silently corrupts the comparison | **Never** — resolve + document the target before the loop (Pitfall 3) |
| `except RuntimeError:` reusing the exact v1.0 catch verbatim | Matches the known SigLIP-2 failure | Other models fail as `IndexError`/`TypeError`/OOM and escape → notebook aborts | **Never** — broaden to `except Exception` + full-traceback caption |
| Load all 4 models into the singleton cache up front | Simple loop body | Co-resident VLMs OOM the L4 | **Never** — strictly sequential load/`del`/`empty_cache` (Pitfall 5) |
| `pip install accelerate` (uncapped) for PaliGemma | Smooth 3B load | Resolver may re-pin torch/transformers → silent autograd drift | Only as a *measured* OOM fallback, pinned `>=0.26,<1.1`, behind a human-verify of the `pip` plan |
| Skip headless nbconvert verification (looks fine interactively) | Faster iteration | The milestone done-gate (exit 0 + ≥4 tiles, failing models as tiles) is unverified | **Never** — headless nbconvert IS the done-criterion |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| HuggingFace Hub (PaliGemma) | Anonymous `snapshot_download` → 401/403; or token without accepted Gemma terms | Authenticated `snapshot_download(token=HF_TOKEN)` on the one-time mirror VM, account having accepted Gemma terms; CLIP/ViT stay anonymous |
| GCS `models/<id>/` mirror | Generalized mirror loop aborts on PaliGemma auth fail, leaving a partial prefix the runtime half-loads | Per-repo fault-isolated + idempotent mirror; post-upload size/manifest check; PaliGemma failure does not corrupt its prefix nor abort CLIP/ViT |
| Runtime model load | A new model loaded from HF at runtime (reproducibility break) | Runtime loads strictly from `gs://mapclass-training-northeast1/models/<id>/`; HF touched only at mirror time |
| vendored dynamicLRP | Adding a custom Promise / editing `third_party/` to make a model traverse | Never edit `third_party/`; `VENDOR_SHA` byte-intact; render the coverage gap as a finding (Pitfall 7) |
| dynamicLRP `params_to_interpret` | Resolved by tensor object identity in `make_graph_iter` — a clone/detach/`.to()` after `requires_grad_()` breaks it | `build_inputs` is the last touch of `pixel_values`; `requires_grad_()` is its final statement; identity-`is` asserted in `attribute()` (Pitfall 1) |
| transformers model class | `AutoModel` for the ViT classifier, or `Siglip2*` for CLIP/ViT → pooled/wrong-shape output | `CLIPModel`, `ViTForImageClassification`, `PaliGemmaForConditionalGeneration` exactly (STACK.md table) |
| Gemma tokenizer | `pip install sentencepiece` for PaliGemma | Unnecessary — 4.52.3's `PaliGemmaProcessor` uses `GemmaTokenizerFast` (tokenizers-backed); adding it perturbs the frozen resolve |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Co-resident VLMs in the singleton cache | `CUDA OOM` escalating across loop iterations; `nvidia-smi` shows 2 model footprints | Strictly sequential `load → … → del model → empty_cache()` in `finally` | At model 2 of 4 on the 24 GB L4 (each ~400M-class forward ≥4.3 GB + LRP activation retention) |
| Missing `empty_cache()` in `finally` (freed only in `except`) | Caught-but-OOMs model still starves the next model | Free in `finally:`, not `except:` | Immediately after the first failing model |
| LRP relevance-pass activation retention on top of forward peak | Forward fits (4.3 GB) but relevance pass OOMs | Accept it as a recorded datum on big models; sequential free; PaliGemma OOM is expected | PaliGemma-3B (~7.5× the so400m ceiling) — expected to exceed 24 GB; render "no heatmap (OOM)" |
| Re-mirroring already-mirrored weights every run | Slow cold-VM startup; redundant GCS egress | `model_loader` size-checked cache-first download (v1.0 pattern, per model_id); `mirror_model` idempotent skip-if-same-size | Cold VM with multi-GB PaliGemma re-downloaded each kernel |

## Security / Provenance Mistakes

| Mistake | Risk | Prevention |
|---------|------|------------|
| Committing `HF_TOKEN` / baking it into the GCS model dir | Credential leak; token in a redistributed bucket | Token is a mirror-VM env var only, never committed, never written into the uploaded model dir; runtime never needs it |
| Editing `third_party/dynamicLRP` (custom Promise) | Breaks `VENDOR_SHA` provenance (T-01-SC3); silently invalidates the comparison's "vendored engine" premise | Hard assertion `VENDOR_SHA == 405e7424…` + `git diff --quiet third_party/dynamicLRP` in pre-flight |
| Bumping torch/transformers to satisfy a model | Silent autograd-graph drift → subtly wrong relevance (worst case: looks plausible) | Pin-assertion pre-flight (`transformers==4.52.3`, `torch` 2.7.1, VENDOR_SHA) fails loudly |
| Gemma-licensed weights in a shared GCS bucket | License/redistribution exposure | Acceptable for the private single-researcher bucket; flagged in milestone notes; do not widen bucket access |

## UX Pitfalls (the researcher reading the side-by-side)

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| No per-panel target-semantics label | Reader misreads cosine-similarity vs token-logit vs class-logit as apples-to-apples | Each panel subtitled with `declare_semantics` (FEATURES.md differentiator — table stakes for honesty) |
| ViT panel unlabeled as class-conditioned | The text-less ViT's class-saliency is misread as query-conditioned like the others | Label "class-conditioned, not text-conditioned"; document the query→class mapping |
| Shared relevance scale across panels | Different head scales make weaker models visually vanish | Per-panel self-scaled symmetric `TwoSlopeNorm` + a one-line figure note "each panel scaled independently; magnitudes not cross-comparable" |
| Blank panel for a failed model | Looks like a render bug, not a result | Captioned "no heatmap (op-coverage gap), failed at `<node>`" tile — coverage visible in the grid (FEATURES.md differentiator) |
| Misaligned heatmap that still renders | Researcher draws false spatial conclusions | Known-answer geometry fixture per model (Pitfall 2); CLS token stripped before reshape |

## "Looks Done But Isn't" Checklist

- [ ] **Adapter contract:** Often missing the identity guarantee — verify `build_inputs` does device/dtype/reshape *before* `requires_grad_()`, and a `test_adapters.py` asserts `is`-identity through `attribute()`
- [ ] **Overlay parameterization:** Often still has a literal 27/14/378 somewhere — verify zero geometry literals in `overlay.py` and parametrized tests pass for 27×27, 14×14, 16×16
- [ ] **CLS token:** Often forgotten for CLIP/ViT — verify the adapter strips CLS before the `grid²` reshape (SigLIP-2 has none; CLIP/ViT do)
- [ ] **Per-model target:** Often a silent `logits_per_image` assumption — verify ViT uses a class logit, PaliGemma an answer-token logit, no `get_image_features` anywhere, each adapter `declare_semantics`
- [ ] **PaliGemma mirror:** Often a partial gated download — verify the GCS prefix is complete (size/manifest check) and the runtime loads it from GCS with no HF call
- [ ] **VRAM lifecycle:** Often frees in `except` not `finally` — verify `del model; empty_cache()` in `finally`, sequential load, registry order ends with PaliGemma
- [ ] **Pin/SHA integrity:** Often drifts to satisfy a model — verify `requirements.txt` unchanged, pin-assertion pre-flight green, `git diff --quiet third_party/dynamicLRP`
- [ ] **Notebook done-gate:** Often only run interactively — verify headless nbconvert exit 0, 0 cell errors, ≥4 tiles, failing models as captioned tiles (not tracebacks)
- [ ] **SigLIP-2 regression oracle:** Often skipped — verify the SigLIP-2 adapter reproduces the *identical* v1.0 recorded op-coverage finding (proves the seam preserved both invariants)

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Tensor-identity broken (1) | LOW–MEDIUM | Diff for `.clone()/.detach()/.to()/cast/reshape` after `requires_grad_()`; move them before it; add the `is`-identity assertion; re-run SigLIP-2 oracle |
| Geometry misalignment (2) | LOW | Add known-answer geometry fixture; derive patch_size/img_dim from the model's processor config; strip CLS; re-render |
| Wrong attribution target (3) | MEDIUM | Stop, resolve the per-model target as a documented decision (esp. PaliGemma answer-token); re-implement `select_target`/`declare_semantics`; query-swap to confirm conditioning |
| Gated-mirror failure (4) | LOW | Accept Gemma terms on the HF account; `export HF_TOKEN`; re-run the idempotent per-repo mirror; verify the GCS prefix size/manifest |
| VRAM OOM crash (5) | LOW | Move free to `finally`; make the except broad; treat PaliGemma OOM as a rendered "no heatmap (OOM)" datum (do NOT escalate hardware) |
| Pin/SHA violation (6/7) | MEDIUM–HIGH | Revert the bump / `third_party` edit; restore `VENDOR_SHA`; reinstall the frozen `requirements.txt`; re-run SigLIP-2 oracle to confirm the recorded finding is unchanged |
| Notebook non-zero exit (8) | LOW | Broaden to `except Exception`, free in `finally`, full-traceback caption; regenerate via `_build_02_multimodel.py`; re-verify headless |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. Tensor-identity invariant break | Phase 1 (contract) + Phase 3 (oracle); re-verified per adapter 4–6 | `is`-identity assertion in `attribute()`; SigLIP-2 reproduces the identical v1.0 finding |
| 2. Hardcoded geometry misalignment | Phase 2 (parameterize overlay); first exercised Phase 4 (ViT 14×14) | Parametrized `test_overlay_grid.py` (27/14/16); known-answer fixture; CLS stripped |
| 3. Wrong/ambiguous attribution target | Phase 1 (contract slots) + up-front decision before Phase 4; per model 3–6 | No `get_image_features`; query-swap changes the map; every panel has `declare_semantics` |
| 4. Gated weights / HF token | Phase 7 (`mirror_model` registry) | PaliGemma GCS prefix complete; runtime loads from GCS, no HF call; token uncommitted |
| 5. VRAM blowup / sequential-load | Phase 7 (notebook loop); informed by Phase 3 (4.326 GB) | Sequential load/`del`/`empty_cache` in `finally`; PaliGemma OOM → "no heatmap" tile, others render |
| 6. Frozen-pin violation | Phase 1 (pin-assertion pre-flight); continuous; canary at Phase 3 & 6 | `transformers==4.52.3`/`torch` 2.7.1/`VENDOR_SHA` assertion green; `requirements.txt` unchanged |
| 7. "Fixing" a coverage gap (scope) | Phases 3–6 (each adapter renders, never patches) | `git diff --quiet third_party/dynamicLRP`; `VENDOR_SHA` intact; no Promise/LXT/captum code |
| 8. Notebook not exit 0 on model failure | Phase 7 (notebook = done-gate); proven Phase 3 | Headless nbconvert exit 0, 0 cell errors, ≥4 tiles, failures as captioned tiles |

## Sources

- `.planning/archive/v1.0-milestone/.../01-03-SUMMARY.md` — SigLIP-2 `SplitWithSizesBackward0` op-coverage finding; Fallback Ladder explicitly declined; VENDOR_SHA 405e7424… intact; caught-finding→exit-0 pattern; 4.326 GB forward-only VRAM (relevance-pass peak unmeasurable); generator-authored-notebook discipline — HIGH (read directly)
- `.planning/research/ARCHITECTURE.md` — the four SigLIP hard-wire points, the two load-bearing invariants, `make_graph_iter` tensor-identity mechanism, the 7-step build order, Anti-Patterns 1–4 — HIGH
- `.planning/research/STACK.md` — per-model classes/targets, PaliGemma gating, ZERO dependency delta, `accelerate` conditional-only, no `sentencepiece`, no transformers/torch bump — HIGH
- `.planning/research/FEATURES.md` — the feature-defining per-architecture target question, anti-features (Fallback Ladder/table/metrics/sweep/tuning), per-panel semantics label, graceful-degradation done-criterion — HIGH
- `.planning/PROJECT.md` — v1.1 scope; Key Decisions (coverage gap = recorded result; Fallback Ladder declined; model+query only knobs; map locked); Out of Scope — HIGH
- `CLAUDE.md` "What NOT to Use" — torch/transformers hard pins, no `git+` install, no conda/poetry/uv, pooled-embed target forbidden, the MEDIUM-LOW SigLIP coverage risk flagged pre-v1.0 — HIGH

---
*Pitfalls research for: adding CLIP/PaliGemma/plain-ViT to the frozen-stack SigLIP-2 dynamic-LRP comparison harness (MapClass v1.1)*
*Researched: 2026-05-19*
