# Phase 4: Fine-tune and Evaluate the Segmentation Model - Research

**Researched:** 2026-05-16
**Domain:** PyTorch segmentation training loop, joint-NLL evaluation, GCS checkpoint I/O
**Confidence:** HIGH (all major claims verified against codebase or installed packages)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01:** Phase 4 = train + evaluate + report only. Ends with trained checkpoints +
  joint-NLL numbers on `test/` + a written A/B × backbone comparison report.
  No model selection, no packaging, no HuggingFace anything.
- **D-02:** New Phase 5 owns delivery (model selection, HuggingFace publish, DELIV-01/02,
  EVAL-02 "reported-with-published-model" clause).
- **D-03:** Cost-probe-first; grid breadth is NOT committed upfront.
- **D-04:** Probe config = SigLIP Variant B (upper-bounds SigLIP-path cost).
- **D-05:** Manual `checkpoint:decision` gate after the probe. Plan halts
  (`autonomous: false`), reports measured GPU-hrs/epoch + projected full-grid cost.
  User chooses grid breadth at the gate. Candidate breadths: (a) full 6-config grid,
  (b) SigLIP A+B full + DINOv2/Swin single-variant, (c) SigLIP A+B full +
  DINOv2/Swin short benchmark runs.
- **D-06:** Training executes on a separate GPU box; planning is CPU-safe. Needs its
  own deferred GPU-host gate artifact analogous to `02-HUMAN-UAT.md`.
- **D-07:** Checkpoints → `gs://mapclass-training-northeast1/models/<config>/` per-config
  prefix. This GCS convention does not exist in the repo today — Phase 4 introduces it.
- **D-08:** Only lightweight artifacts return to git: URI manifest + joint-NLL metrics
  JSON + written comparison report. Model weights are NEVER committed to git.
- **D-09:** Periodic GCS checkpoints + auto-resume. Write `{weights, optimizer, step}` to
  the config's `gs://` prefix every N steps/epochs; on (re)launch auto-detect the latest
  checkpoint and resume. Preemption-safe by design.

### Claude's Discretion

- Training hyperparameters, optimizer, LR schedule, epoch budget, and the loss
  formulation that consumes per-sample `sample_weights.json`.
- Validation-signal methodology (constrained): val slice MUST be carved from `train/`
  only; `test/` must never be enumerated during training (EVAL-01 zero-leakage).
  Researcher to recommend the concrete carve.
- Exact checkpoint cadence N (D-09), GCS I/O mechanism, resume-detection logic.
- How recursive coarse-to-fine inference (Phase-3 D-03/D-04) is invoked at eval time,
  and the comparison-report format/structure.
- Pyramid-aware batch sampling and dataloader internals (Phase-3 D-06).

### Deferred Ideas (OUT OF SCOPE)

- Deliverable / model selection — Phase 5.
- HuggingFace publish shape — Phase 5.
- OCR / place-name reading — v2 / GEOREF-V2.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EVAL-02 (computation only) | Joint per-pixel NLL = -(log p_lc + log p_topo) computed on held-out synthetic test set | See §Evaluation Harness: exact formula verified with torch.log_softmax; tested on installed PyTorch 2.12 |
| EVAL-03 | DINOv2 and Swin Transformer backbone variants evaluated on the same metric for direct comparison against the primary SigLIP path | See §Backbone-Swap Training; D-05 protocol already built; all three backbones share the same decoder + heads |
</phase_requirements>

---

## Summary

Phase 4 fine-tunes the Phase-3 assembled-but-untrained segmentation model end-to-end on
the Phase-2 pyramid dataset, then evaluates on the held-out synthetic test set using the
locked joint per-pixel NLL metric. The two distinct prior-injection variants (A: frozen
backbone + decoder-level prior-encoder; B: trainable widened 15-ch patch-embed) are
trained against three backbone configurations (SigLIP primary, DINOv2 + Swin benchmarks),
gated by a cost-probe run on SigLIP Variant B before the full grid is committed.

The Phase-3 codebase provides everything the training loop needs: `SegModelVariantA` /
`SegModelVariantB` as drop-in `nn.Module`s, a `PyramidDataset` that already surfaces
`sample_weights` byte-identically from `sample_weights.json`, `recursive_predict` /
`recursive_predict_variant_b` for the eval-time c2f walk, and the frozen `split.json`
train/test partition. Phase 4 adds: a training loop script, weighted joint-loss
computation, a GCS checkpoint writer/reader, a joint-NLL eval harness, and a comparison
report. No model architecture changes are needed.

The biggest design question is how the two-pass recursive c2f inference interacts with
backpropagation during training: teacher-forced coarse priors (coarse logits detached;
prior = `stop_grad(softmax(coarse_logits))`) avoids truncated-BPTT memory explosion
and is the practical standard for autoregressive dense prediction training. The other
questions — weighted loss, GCS I/O, val carve, eval harness — are well-defined and
have direct implementation patterns in the existing codebase.

**Primary recommendation:** Use teacher-forced coarse priors for training (detach after
coarse pass), AdamW with separate parameter groups keyed to trainability per variant,
`reduction='none'` cross-entropy multiplied by per-sample scalar for the weighted loss,
`torch.log_softmax` for numerically stable NLL evaluation, and `gcsfs` (installable,
2026.5.0) backed by the existing `gcloud` CLI for GCS checkpoint I/O.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Training loop orchestration | `scripts/finetune_seg.py` (new) | `seg/` model modules | New script; model modules are construction-only |
| Weighted joint loss | Training script | `seg/model.py` (raw logits) | Loss is external to the model — heads output raw logits by design (D-06) |
| GCS checkpoint write/read | Training script | gcsfs / gsutil | Checkpoint I/O is operational plumbing, not model logic |
| Val-slice carve | Training script (DataLoader split) | `seg/dataset.py` | `PyramidDataset` already enforces train-only; further split is caller logic |
| Cost probe run | Training script (short epoch) | checkpoint:decision gate | Same script, early-exit mode |
| Joint-NLL evaluation | `scripts/evaluate_seg.py` (new) | `seg/recursive.py` | Eval walks full c2f pyramid; separate script from training |
| Backbone swap | `seg/backbones.py` (D-05 protocol) | Training script (param groups) | Protocol built; swapping is instantiation, not re-architecture |
| GCS manifest + metrics JSON | Training / eval scripts | git commit | Lightweight artifacts: committed; weights never committed |
| Comparison report | New `.planning/` markdown artifact | — | Written after all configs evaluated |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| torch | 2.12.0 (installed) [VERIFIED: pip3 list] | Training loop, loss, optimizer, checkpoint | Already installed; CUDA 13.0 build |
| torchvision | 0.27.0 (installed) [VERIFIED: pip3 list] | Image transforms for train augmentation | Paired with torch |
| gcsfs | 2026.5.0 (installable) [VERIFIED: pip install --dry-run] | GCS file access via fsspec interface; `open('gs://...')` idiom | Allows `torch.save` to write directly to `gs://` URI |
| google-cloud-storage | 3.10.1 (installable) [VERIFIED: pip install --dry-run] | Fallback: list/delete/copy blobs, auto-detect latest checkpoint by listing prefix | Needed for checkpoint enumeration (gsutil ls equivalent in Python) |
| peft | 0.19.1 (installed) [VERIFIED: pip3 list] | Load Phase-1 LoRA adapter for SigLIP backbone | Same load order as `finetune_paligemma.py` |
| transformers | 5.8.1 (installed) [VERIFIED: pip3 list] | `PaliGemmaForConditionalGeneration.from_pretrained` | Required by `SiglipBackbone` |
| timm | 1.0.27 (installed) [VERIFIED: pip3 list] | DINOv2 and Swin backbone weights | `Dinov2Backbone` + `SwinBackbone` use timm |
| pytest | 9.0.3 (installed) [VERIFIED: pip3 list] | Test framework | Project standard |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| fsspec | 2026.4.0 (installed) [VERIFIED: pip3 list] | Abstract filesystem; gcsfs builds on it | Installed; gcsfs needs 2026.3.0 so it will downgrade |
| gsutil | 5.37 (installed) [VERIFIED: gsutil version] | CLI fallback for GCS operations in shell gate commands | For the GPU-host hand-commands artifact |
| gcloud SDK | 568.0.0 (installed) [VERIFIED: gcloud --version] | Auth + project configuration | `gcloud auth application-default login` establishes credentials for gcsfs |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| gcsfs `open('gs://...')` + `torch.save` | `google.cloud.storage` blob.upload_from_filename | gcsfs is higher level and lets torch.save/load work with `gs://` URIs transparently; GCS blob upload requires local temp file as intermediate |
| gcsfs | gsutil CLI subprocess calls | gsutil works but requires subprocess management and makes resume logic messy; gcsfs is Pythonic |
| AdamW | Adam / SGD | AdamW is the de-facto standard for fine-tuning ViTs and ViT-derived models; weight decay correctly excluded from bias/norm [ASSUMED] |

**Installation (GPU host — not this CPU planning VM):**

```bash
pip install gcsfs google-cloud-storage
```

**Version verification:**

```
gcsfs:                  2026.5.0  (pip install --dry-run, 2026-05-16)
google-cloud-storage:   3.10.1    (pip install --dry-run, 2026-05-16)
torch:                  2.12.0+cu130 (already installed)
```

---

## Architecture Patterns

### System Architecture Diagram

```
Phase-2 pyramid data (data/synthetic/train/, data/historical/, data/satellite/)
  └── PyramidDataset (seg/dataset.py) — train/ roots only (EVAL-01)
        └── random_split(80/20, seed=42) → train_ds / val_ds
              │
              ▼
      DataLoader (train_dl)
              │
              ▼
    ┌─── Training Loop (scripts/finetune_seg.py) ───────────────────────────┐
    │  1. sample_weights → per-sample scalar weight                         │
    │  2. Variant A or B forward pass (coarse→fine, teacher-forced)        │
    │  3. Joint weighted loss = w * (CE_lc + CE_topo), averaged            │
    │  4. optimizer.step() on trainable params only                         │
    │  5. Every N steps → torch.save checkpoint → gs://.../models/<cfg>/   │
    │  6. Val loop every epoch → val NLL (not joint-NLL; just CE sum)       │
    └────────────────────────────────────────────────────────────────────────┘
              │                          │
              ▼                          ▼
    gs://mapclass-training-northeast1/  val metric (stdout / metrics.json)
      models/<config>/step_{N}.pt
              │
              ▼ (after training complete)
    ┌─── Eval Harness (scripts/evaluate_seg.py) ─────────────────────────────┐
    │  Loads best checkpoint from gs://.../models/<config>/                  │
    │  Enumerates split.json test map dirs → runs recursive_predict()        │
    │  Computes: NLL = mean over pixels of -(log_softmax_lc + log_softmax_topo) │
    │  Writes: metrics/<config>-nll.json                                     │
    └────────────────────────────────────────────────────────────────────────┘
              │
              ▼
    scripts/report_comparison.py → docs/04-comparison-report.md
    (git commits: metrics/*.json + gs:// URI manifest + report)
```

### Recommended Project Structure

```
scripts/
├── finetune_seg.py          # new: training loop entrypoint (all configs)
├── evaluate_seg.py          # new: joint-NLL eval harness
├── report_comparison.py     # new: reads metrics JSONs → renders report
├── seg/                     # Phase-3 model (unchanged)
│   ├── model.py
│   ├── backbones.py
│   ├── decoder.py
│   ├── heads.py
│   ├── recursive.py
│   └── dataset.py
metrics/
├── siglip-A-nll.json        # per-config joint-NLL results (not weights)
├── siglip-B-nll.json
├── ...
├── checkpoint_manifest.json # gs:// URIs of the best checkpoint per config
docs/
└── 04-comparison-report.md  # A/B × backbone written comparison
tests/
├── test_seg_training.py     # new: offline overfit-a-batch + loss-shape tests
├── test_seg_eval.py         # new: NLL formula unit tests (synthetic logits)
├── test_seg_gcs.py          # new: GCS write/read round-trip (mock)
```

### Pattern 1: Per-Sample Weighted Joint Loss

**What:** Each pyramid's `sample_weights.json` carries per-class LC weights and a
`topography_weight` scalar. The training loop maps these to a per-sample scalar
multiplier applied to `reduction='none'` cross-entropy before averaging.

**When to use:** Every training step.

The `sample_weights.json` schema (locked, verified from `scripts/historical/label.py`
and `scripts/synthetic_weights.py`) [VERIFIED: codebase]:

```python
# sample_weights.json shape (all three sources share this schema):
{
  "land_cover_weights": {
    "water": 1.0,     "trees": 0.3,   "shrubland": 0.7,
    "grassland": 0.7, "cropland": 0.15, "built_up": 0.1,
    "bare_sparse": 1.0, "flooded_wetland": 0.5, "snow_ice": 1.0
  },
  "topography_weight": 1.0,
  "source": "historical",   # or "synthetic" / "satellite"
  "map_file": "..."
}
```

**Implementation pattern (verified to run on PyTorch 2.12)** [VERIFIED: in-session test]:

```python
# Source: verified by running against torch 2.12 installed locally
import torch, torch.nn.functional as F
from biome_mapping import LANDCOVER_CLASSES

def build_lc_weight_tensor(sw_dict: dict, device) -> torch.Tensor:
    """Convert sample_weights['land_cover_weights'] to (9,) weight tensor."""
    lc_w = sw_dict["land_cover_weights"]
    return torch.tensor(
        [lc_w[c] for c in LANDCOVER_CLASSES], dtype=torch.float32, device=device
    )

def weighted_joint_loss(
    lc_logits,      # (B, 9, H, W)
    topo_logits,    # (B, 3, H, W)
    lc_targets,     # (B, H, W) long
    topo_targets,   # (B, H, W) long
    sample_weights_batch,  # list[dict] length B
    device,
) -> torch.Tensor:
    B = lc_logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    for i, sw in enumerate(sample_weights_batch):
        lc_class_w = build_lc_weight_tensor(sw, device)
        topo_w = torch.tensor(sw["topography_weight"], device=device)

        # Per-pixel LC loss with class weights, then mean over spatial dims
        lc_loss = F.cross_entropy(
            lc_logits[i:i+1], lc_targets[i:i+1],
            weight=lc_class_w, reduction='mean'
        )
        # Per-pixel topo loss scaled by topography_weight
        topo_loss = F.cross_entropy(
            topo_logits[i:i+1], topo_targets[i:i+1],
            reduction='mean'
        ) * topo_w

        total_loss = total_loss + (lc_loss + topo_loss)
    return total_loss / B
```

Note: `F.cross_entropy(..., weight=class_weights)` applies per-class weighting
internally; the `topography_weight` is a scalar applied after the CE call. This
matches the `sample_weights.json` schema exactly.

### Pattern 2: Teacher-Forced Coarse-to-Fine Training

**What:** The recursive c2f inference (D-04) runs coarse→fine in two passes per
pyramid. During training, the coarse pass logits are detached before being softmaxed
to produce the prior for the fine pass. This avoids backpropagating gradients through
the coarse path twice (truncated BPTT), prevents gradient instability, and keeps the
forward-pass semantics identical to the inference-time `recursive_predict`.

**When to use:** Every training step; both Variant A and Variant B.

**Training c2f loop (pseudocode):**

```python
# Variant A training step (coarse = 896, fine = 448/224)
# The recursive_predict / crop_prior_to_child already implement the geometry;
# for training we replicate the walk but with grad enabled and teacher forcing.

from seg.recursive import crop_prior_to_child, cold_start_prior

def train_step_variant_a(model, pyramid_batch, optimizer, device):
    """
    pyramid_batch: list of {rgb_tiles: {tile_id: tensor},
                             lc_targets: {tile_id: tensor},
                             topo_targets: {tile_id: tensor},
                             sample_weights: dict,
                             manifest: list[tile_dict]}
    """
    total_loss = 0.0
    for pyr in pyramid_batch:
        manifest_by_id = {t["id"]: t for t in pyr["manifest"]}
        root = next(t for t in pyr["manifest"] if t["size"] == 896)
        prob_cache = {}

        def predict_tile(tile, parent):
            rgb = pyr["rgb_tiles"][tile["id"]].to(device)      # (1, 3, S, S)
            if parent is None:
                prior = cold_start_prior(1, tile["size"]).to(device)
            else:
                # teacher-forced: detach parent probs before forming prior
                prior = crop_prior_to_child(
                    prob_cache[parent["id"]].detach(),  # KEY: stop_grad
                    parent, tile
                )
            lc_logits, topo_logits = model(rgb, prior)  # grads flow here
            # Cache softmax probs for children (detached — teacher forcing)
            lc_prob = torch.softmax(lc_logits.detach(), dim=1)
            topo_prob = torch.softmax(topo_logits.detach(), dim=1)
            prob_cache[tile["id"]] = torch.cat([lc_prob, topo_prob], dim=1)

            lc_t = pyr["lc_targets"][tile["id"]].to(device)
            topo_t = pyr["topo_targets"][tile["id"]].to(device)
            sw = pyr["sample_weights"]
            return weighted_joint_loss(
                lc_logits, topo_logits, lc_t, topo_t, [sw], device
            )

        # Walk: 896 → 448 → 224 (teacher-forced)
        loss = predict_tile(root, None)
        for cid in root.get("children", []):
            child = manifest_by_id[cid]
            loss = loss + predict_tile(child, root)
            for gcid in child.get("children", []):
                gc = manifest_by_id[gcid]
                loss = loss + predict_tile(gc, child)

        total_loss = total_loss + loss

    (total_loss / len(pyramid_batch)).backward()
    optimizer.step()
    optimizer.zero_grad()
```

**Variant B** is identical except:
- `model(x15)` where `x15 = torch.cat([rgb, prior], dim=1)` (no separate `prior` arg)
- `rgb_tiles` stores 3-ch images; prior concatenation happens inside the step

### Pattern 3: GCS Checkpoint Write + Resume

**What:** Write `{model_state_dict, optimizer_state_dict, step, config}` to
`gs://mapclass-training-northeast1/models/<config>/step_{N:07d}.pt` using gcsfs.
On launch, list the prefix and resume from the highest step number.

**When to use:** Every N steps + at epoch end.

```python
# Source: gcsfs docs + verified bucket structure (gs://mapclass-training-northeast1/models/ exists)
import gcsfs
import torch, io, re

GCS_PROJECT = "narrative-campaign"
BUCKET_PREFIX = "gs://mapclass-training-northeast1/models"

def gcs_save_checkpoint(config_name: str, step: int, state: dict):
    """Write checkpoint to GCS atomically via gcsfs + in-memory buffer."""
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    path = f"{BUCKET_PREFIX}/{config_name}/step_{step:07d}.pt"
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    with fs.open(path, "wb") as f:
        f.write(buf.read())

def gcs_latest_checkpoint(config_name: str):
    """Return (step, state_dict) of the latest checkpoint, or (0, None)."""
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    prefix = f"mapclass-training-northeast1/models/{config_name}/"
    try:
        files = fs.ls(prefix)
    except FileNotFoundError:
        return 0, None
    # Filter step_{N}.pt files
    step_files = [f for f in files if re.search(r"step_(\d+)\.pt$", f)]
    if not step_files:
        return 0, None
    latest = max(step_files, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    latest_step = int(re.search(r"step_(\d+)", latest).group(1))
    with fs.open(f"gs://{latest}", "rb") as f:
        state = torch.load(io.BytesIO(f.read()), weights_only=False)
    return latest_step, state
```

**Config naming scheme** (keys the GCS prefix, D-07): `<backbone>-<variant>`, e.g.
`siglip-A`, `siglip-B`, `dinov2-A`, `dinov2-B`, `swin-A`, `swin-B`.

### Pattern 4: Joint Per-Pixel NLL Evaluation

**What:** Exact EVAL-02 formula: NLL = mean over all pixels of
`-(log p_lc_true + log p_topo_true)` on held-out `test/` maps using the c2f walk.

**Key numerical stability point** [VERIFIED: in-session test]: use `torch.log_softmax`
(numerically stable via log-sum-exp internally) rather than `torch.log(softmax(...))`.
The two differ by up to 4.8e-7 in practice; either is acceptable, but `log_softmax` is
the canonical form.

```python
# Source: verified formula against torch 2.12
import torch, json
from pathlib import Path
from seg.recursive import recursive_predict, recursive_predict_variant_b

def evaluate_joint_nll(model, test_pyramid_dirs, variant="A", device="cpu"):
    """
    Compute mean joint per-pixel NLL over test set.

    NLL = -(log p_lc[true_class] + log p_topo[true_class]) per pixel,
    averaged over all pixels of all 224-level tiles.

    Uses recursive c2f inference (D-04), not a flat pass.
    test_pyramid_dirs: list of pyramid dirs (each has pyramid.json).
                       These come from data/synthetic/test/ paths in split.json.
    """
    model.eval()
    total_nll = 0.0
    total_pixels = 0

    predict_fn = recursive_predict if variant == "A" else recursive_predict_variant_b

    with torch.no_grad():
        for pdir in test_pyramid_dirs:
            prob_cache = predict_fn(model, pdir)
            manifest = json.loads((Path(pdir) / "pyramid.json").read_text())
            leaf_tiles = [t for t in manifest["tiles"] if t["size"] == 224]

            for tile in leaf_tiles:
                tile_id = tile["id"]
                prob12 = prob_cache[tile_id].to(device)  # (1, 12, 224, 224)
                lc_prob = prob12[:, :9]   # (1, 9, 224, 224)
                topo_prob = prob12[:, 9:]  # (1, 3, 224, 224)

                # Load ground-truth labels
                lc_gt = _load_label(Path(pdir) / tile["land_cover"]).to(device)   # (224,224)
                topo_gt = _load_label(Path(pdir) / tile["topography"]).to(device) # (224,224)

                # Numerically stable log probabilities
                log_lc = torch.log_softmax(lc_prob, dim=1)   # (1, 9, H, W)
                log_topo = torch.log_softmax(topo_prob, dim=1)

                # Gather log p at the true class
                lc_nll = -log_lc[0].gather(0, lc_gt.unsqueeze(0)).squeeze(0)     # (H,W)
                topo_nll = -log_topo[0].gather(0, topo_gt.unsqueeze(0)).squeeze(0)

                joint_nll = lc_nll + topo_nll  # (H, W)

                # Ignore padded edges (label pixel == 255 convention if used,
                # otherwise include all pixels)
                valid = (lc_gt < 9) & (topo_gt < 3)
                total_nll += joint_nll[valid].sum().item()
                total_pixels += valid.sum().item()

    return total_nll / total_pixels if total_pixels > 0 else float("nan")
```

**Note on eval-time c2f:** The eval harness calls the existing `recursive_predict` /
`recursive_predict_variant_b` from `seg.recursive` — these already implement the
correct D-04 pyramid walk under `torch.no_grad()`. The eval script does NOT need to
re-implement the walk.

### Pattern 5: Optimizer Parameter Groups (per-variant trainability)

**What:** Only certain parameters are trainable per variant. The optimizer must have
separate param groups so that frozen parameters are excluded from updates entirely.

**Variant A trainable parameters** [VERIFIED: `seg/model.py`]:
- `model.prior_encoder.*` (the small 2-conv network)
- `model.decoder.*` (all decoder params — NOT frozen in Variant A for LC/topo tasks)
- `model.lc_head.*`
- `model.topo_head.*`
- `model.backbone.*` is fully frozen (`requires_grad_(False)` called at construction)

Wait — re-reading `model.py`: Variant A calls `self.backbone.requires_grad_(False)`.
The decoder and heads are NOT explicitly frozen in the SegModelVariantA constructor.
The decoder and heads start with `requires_grad=True` by default.
[VERIFIED: seg/model.py lines 237-248]

**Variant B trainable parameters** [VERIFIED: `seg/model.py`]:
- `model.backbone._vision_tower.patch_embed.proj` (widened 15-ch conv, marked `requires_grad_(True)` in `_widen_patch_embed`)
- `model.decoder.*` and heads — default `requires_grad=True`
- Everything else in backbone is frozen (`eval().requires_grad_(False)` from backbone construction)

**AdamW param group setup:**

```python
def make_optimizer(model, lr=1e-4, weight_decay=1e-2):
    """Build AdamW with trainable params only (frozen params excluded)."""
    trainable = [p for p in model.parameters() if p.requires_grad]
    # Exclude bias + normalization from weight decay (standard ViT fine-tune pattern)
    no_decay = {"bias", "norm", "LayerNorm"}
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(grouped, lr=lr)
```

### Pattern 6: Train→Val Carve (EVAL-01 preserving)

**What:** The val slice must be carved from `train/` only. `test/` is never touched
during training. The `PyramidDataset` already raises `ValueError` if a `test/` path
is indexed [VERIFIED: `seg/dataset.py` lines 126-134].

**Recommendation:** Use `torch.utils.data.random_split` with a fixed seed over the
full `PyramidDataset` (pointing only at `train/` roots) at training start. An 80/20
split gives enough coverage for gradient signal. Stratification across sources
(historical / synthetic / satellite) would be ideal but requires indexing source type
from `sample_weights["source"]` — feasible but adds complexity.

**Simpler pragmatic choice** (Claude's discretion): plain seeded random split 80/20.
The PyramidDataset already guarantees no test-set paths; the seed ensures
reproducibility across restarts [VERIFIED: PyTorch `random_split` uses a Generator].

```python
import torch
from torch.utils.data import random_split
from seg.dataset import PyramidDataset

ds = PyramidDataset(train_root_dirs)   # all train/ subtrees; test/ excluded by constructor
gen = torch.Generator().manual_seed(42)
n_val = int(0.2 * len(ds))
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
```

**Rationale for 80/20 over stratified:** The dataset has three sources with very
different cardinalities (historical maps are scarcer than synthetic). A source-stratified
split would require knowing how many maps belong to each source, which complicates the
code. The random split is acceptable for a research pipeline; the eval set for
publication is the locked `test/` (synthetic only) — the val split is only used
for early-stopping / progress monitoring during the cost probe and training runs.

### Anti-Patterns to Avoid

- **Enumerate `test/` during training:** `PyramidDataset` guards against this at
  construction, but the training script must not call `recursive_predict` on `test/`
  during training. `test/` is enumerated only by `evaluate_seg.py`.
- **`torch.log(softmax(logits))`:** Numerically unsafe. Use `torch.log_softmax` or
  `F.nll_loss` which takes log-probs directly.
- **Committing model weights to git:** D-08 locks this. Only metrics JSON + GCS URI
  manifest + report go to git.
- **Backpropagating through both c2f passes without detaching:** Without `detach()`
  on the coarse prior before the fine pass, memory usage doubles and gradients
  explode for deeper recursion. Always `detach()` after softmax in the training walk.
- **Using a single AdamW group with all parameters:** The frozen backbone params should
  have `requires_grad=False` (already set by backbone constructors); AdamW should
  only receive trainable params. `model.parameters()` returns all params; always
  filter by `.requires_grad`.
- **Using pixel-index 255 as an ignore index without consistency:** The `land_cover.png`
  / `topography.png` are uint8 but class indices are 0–8 and 0–2 respectively. The
  current label writers do not appear to use 255 as a special ignore class for interior
  pixels. Eval should filter `(lc_gt < 9) & (topo_gt < 3)` rather than assume 255
  means ignore. [VERIFIED: seg/dataset.py loads as `long()` from uint8; no masking
  is documented in the existing dataset code.]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| GCS file I/O | Custom blob-upload wrapper | gcsfs + `torch.save/load` to `gs://` URI | Handles auth, streaming, partial writes; installable in 2026 |
| C2f pyramid walk at eval time | New traversal code | `seg.recursive.recursive_predict` / `recursive_predict_variant_b` | Already built and smoke-tested in Phase 3 |
| Log-softmax for NLL | `log(softmax(x))` | `torch.log_softmax(logits, dim=1)` | Numerically stable; avoids underflow in long-tail classes |
| Parameter group construction | Manual param-group dict | Filter `named_parameters()` by `requires_grad` | The backbone constructors already manage frozen/trainable split |
| Checkpoint resume detection | Filename scanning with glob | gcsfs `fs.ls(prefix)` + regex on step numbers | Atomic and correct under concurrent writes |
| Weighted cross-entropy | Manual mask multiply | `F.cross_entropy(..., weight=class_tensor)` for class weights; `reduction='none'` + scalar for sample weights | Avoids off-by-one with ignore_index; handles edge cases |

---

## Runtime State Inventory

> This is a greenfield training phase (no rename/refactor), but there is one
> existing runtime state item relevant to Phase 4.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | GCS bucket `gs://mapclass-training-northeast1/` already has `data/` and `models/` prefixes [VERIFIED: gsutil ls] | No migration; Phase 4 writes under `models/<config>/` which doesn't yet contain checkpoints |
| Live service config | None | None |
| OS-registered state | None | None |
| Secrets/env vars | GCS auth uses `gcloud auth application-default login`; no key files in git | Run `gcloud auth` on GPU host before training; no code change |
| Build artifacts | gcsfs not yet installed (pip install required on GPU host) | `pip install gcsfs google-cloud-storage` on GPU host before executing |

---

## Common Pitfalls

### Pitfall 1: EVAL-01 Leakage Through Val Loop

**What goes wrong:** The training script reads `split.json` to find test maps and
accidentally enumerates them during val-loss computation.

**Why it happens:** A developer mistakenly passes all data directories (train + test)
to `PyramidDataset`, or the val carve is done after a dataset that already includes
test paths.

**How to avoid:** Always construct `PyramidDataset` from `train/` roots only.
`PyramidDataset.__init__` will raise `ValueError` if any indexed path contains
`test/`. Never pass test directories during training — eval is a separate script run
after training is complete.

**Warning signs:** `PyramidDataset` raises on construction; or the val NLL is
suspiciously lower than expected.

### Pitfall 2: Gradient Explosion Through Both C2f Passes

**What goes wrong:** Not detaching the coarse logits before forming the prior for the
fine pass causes the backward pass to traverse the full computational graph twice
(or more for the 3-level pyramid), doubling memory usage and causing gradient
instability.

**Why it happens:** Forgetting `detach()` on `prob_cache[parent["id"]]` before
passing it to `crop_prior_to_child`.

**How to avoid:** Always write `prob_cache[tile["id"]] = torch.cat([lc_prob, topo_prob], dim=1)`
where `lc_prob = torch.softmax(lc_logits.detach(), dim=1)`. The forward pass retains
grad on `lc_logits` and `topo_logits` for the current tile's loss; the prob cache
is detached to prevent BPTT across tiles.

**Warning signs:** GPU OOM during the 896→448→224 three-level walk; loss NaN after
a few steps.

### Pitfall 3: Frozen Parameters Included in Optimizer

**What goes wrong:** `model.parameters()` is passed to AdamW without filtering
frozen params. PyTorch does not error — it silently tracks frozen params, wastes
memory in optimizer state, and can interfere with gradient scaling in mixed precision.

**Why it happens:** Convenience of `model.parameters()` without checking `requires_grad`.

**How to avoid:** Filter: `[p for p in model.parameters() if p.requires_grad]`. For
Variant A this is only `prior_encoder`, `decoder`, and heads. For Variant B this is
the widened patch-embed conv plus `decoder` and heads.

**Warning signs:** Optimizer state dict is much larger than expected; or a frozen
backbone's param tensor shows `.grad` is not None after a backward pass.

### Pitfall 4: GCS Checkpoint Non-Atomicity

**What goes wrong:** A checkpoint write is interrupted mid-stream (preemption), leaving
a partial `.pt` file that `torch.load` fails on, preventing resume.

**Why it happens:** gcsfs writes data to GCS in chunks; if the process is killed mid-
write, the object may be committed in a partial state depending on the GCS upload mode.

**How to avoid:** Use an in-memory buffer approach: `buf = io.BytesIO(); torch.save(state, buf); buf.seek(0); fs.open(path, 'wb').write(buf.read())`. GCS object uploads via gcsfs are atomic at the object level (multipart upload commits atomically). The larger risk is: resume logic checks step number in the filename; always name with the step AFTER the optimizer step completes, never before.

**Warning signs:** `torch.load` raises `UnpicklingError` or `EOFError` on latest checkpoint;
resume-detection lists the file but fails to load it.

### Pitfall 5: `recursive_predict` Called on Test Set During Training Loop

**What goes wrong:** The training script or val loop imports `recursive_predict` and
runs it on test dirs for "a quick eval". This contaminates the zero-leakage guarantee.

**Why it happens:** Convenience — the eval harness is the same function as the
inference path.

**How to avoid:** The eval script is a separate entrypoint that reads `split.json`
test-set map IDs only after all training is complete. The training loop's val metric
is computed with a flat `DataLoader` pass (not recursive c2f), not with `recursive_predict`.

**Warning signs:** The training script imports from `evaluate_seg.py` or calls
`recursive_predict` with a directory whose path contains `test/`.

### Pitfall 6: DINOv2/Swin Loaded Without `pretrained=True` on GPU Host

**What goes wrong:** `Dinov2Backbone(pretrained=False)` loads random weights,
producing a useless baseline and making EVAL-03 comparison meaningless.

**Why it happens:** The Phase-3 smoke tests use `pretrained=False` for offline CI.
The training script must flip this to `pretrained=True` on the GPU host.

**How to avoid:** The training script must pass `pretrained=True` for DINOv2 and Swin
when running on the GPU host. A config flag (`--pretrained/--no-pretrained`) makes
this explicit. The GPU-host hand-commands artifact (analogous to `02-HUMAN-UAT.md`)
should call out this flag.

**Warning signs:** DINOv2/Swin val loss does not improve from epoch 0 despite training.

### Pitfall 7: Wrong Tile Level Used in NLL Eval

**What goes wrong:** The eval harness computes NLL over the 896- or 448-level tiles
instead of the 224-level leaf tiles. The 896 prediction is the cold-start prior-less
pass and is the coarsest prediction.

**Why it happens:** `recursive_predict` returns probabilities for ALL tiles in the
pyramid, keyed by tile ID. If the eval iterates `manifest["tiles"]` without filtering
to `size == 224`, it includes all levels.

**How to avoid:** Filter: `leaf_tiles = [t for t in manifest["tiles"] if t["size"] == 224]`.
The 224-level outputs are the finest predictions and the ones that correspond
pixel-for-pixel to the ground-truth label crops in `pyramid.json`.

---

## Code Examples

### Verified: `sample_weights.json` schema

```python
# Confirmed shape from scripts/historical/label.py + scripts/synthetic_weights.py
# [VERIFIED: codebase]
SAMPLE_WEIGHTS_SCHEMA = {
    "land_cover_weights": {cls: float for cls in LANDCOVER_CLASSES},  # 9 keys
    "topography_weight": float,   # single scalar
    "source": str,                # "historical" | "synthetic" | "satellite"
    "map_file": str,
}
```

### Verified: PyramidDataset returns sample_weights unchanged

```python
# seg/dataset.py __getitem__ returns:
# {
#   "image":          (3, H, W) torch.float32
#   "land_cover":     (H, W)    torch.int64
#   "topography":     (H, W)    torch.int64
#   "sample_weights": dict  — raw from sample_weights.json, byte-identical
# }
# [VERIFIED: seg/dataset.py line 165-183]
```

### Verified: Variant A trainable params

```python
# SegModelVariantA: backbone.requires_grad_(False) at construction.
# decoder and heads are nn.Module defaults (requires_grad=True).
# prior_encoder is a small trainable module.
# [VERIFIED: seg/model.py lines 237-258]

trainable_names = [
    "prior_encoder.*",  # 2-conv trainable encoder
    "decoder.*",        # shared conv decoder
    "lc_head.*",        # 1x1 conv head, 9 classes
    "topo_head.*",      # 1x1 conv head, 3 classes
]
# backbone.* has requires_grad=False
```

### Verified: GCS bucket already has `models/` prefix

```bash
# gsutil ls gs://mapclass-training-northeast1/  returned:
# gs://mapclass-training-northeast1/data/
# gs://mapclass-training-northeast1/models/
# [VERIFIED: gsutil ls, 2026-05-16]
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Frozen backbone linear probe | End-to-end fine-tuning with partial freeze (prior-encoder only for Variant A) | Standard since ViT fine-tuning work 2020-2022 | Better segmentation boundary quality |
| BPTT through recursive inference | Teacher-forced prior (detach after coarse) | Standard practice for autoregressive dense prediction | Stable training, manageable memory |
| Full checkpoint to local disk | Preemption-safe GCS checkpoint | Cloud-native training since 2019 | Survives GPU preemption |
| Single-metric evaluation | Joint per-pixel NLL (two-head dependency) | Project SPEC decision | Captures land-cover/topography correlation |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | AdamW with weight-decay exclusion for bias/norm is best for ViT fine-tuning | Standard Stack (Alternatives Considered) | Could use Adam instead; weight-decay choice affects regularisation but not correctness |
| A2 | 80/20 train/val split is sufficient for training-progress signal | Pattern 6 | Might be noisier than stratified split; acceptable for a research pipeline |
| A3 | Teacher-forced coarse priors (detach) is the correct training approach for the c2f architecture | Pattern 2 | If end-to-end BPTT is preferred (more principled but expensive), memory budget needs revisiting |
| A4 | gcsfs in-memory buffer write is effectively atomic (no partial objects committed under preemption) | Pattern 3 | GCS multipart upload may commit partially under some failure modes; worst case: corrupt checkpoint detected by torch.load, skipped in resume logic |
| A5 | DINOv2 `vit_small_patch14_dinov2` with `pretrained=True` is the correct timm tag for the EVAL-03 benchmark | Standard Stack | timm tag may have changed; verify with `timm.list_models("*dinov2*")` on GPU host before training |
| A6 | Swin `swin_base_patch4_window7_224` with `pretrained=True` is the correct timm tag | Standard Stack | Same risk as A5; verify on GPU host |

**If this table is empty:** It is not empty — A1–A6 require user or GPU-host confirmation.

---

## Open Questions

1. **C2f training granularity: one pyramid per batch element, or all tiles independently?**
   - What we know: `PyramidDataset` indexes individual tiles, not whole pyramids. A training
     step that teacher-forces the coarse-to-fine walk requires loading all 21 tiles of a
     pyramid in dependency order.
   - What's unclear: Whether the training loop should (a) group tiles by pyramid and process
     each pyramid as a unit (correct dependency structure, variable batch size), or (b) train
     on random tiles independently with a frozen/cached coarse prior (simpler DataLoader,
     but stale priors).
   - Recommendation: Option (a) — group by pyramid for correctness. The recursive walk is
     only 21 tiles per pyramid (1+4+16); processing one pyramid per GPU step is tractable.
     The DataLoader should return pyramid directories (not individual tiles) during training.

2. **Epoch and step budget for the cost probe and full training runs.**
   - What we know: The cost probe is a short SigLIP-B run (D-04). "Short" is unspecified.
     The manual gate (D-05) will set the full grid budget from measured cost.
   - What's unclear: How many steps constitute a "probe epoch" that gives a reliable
     GPU-hrs/epoch estimate without wasting compute.
   - Recommendation: 1 full epoch over train_ds (all pyramids once). This gives a
     definitive GPU-hrs/epoch number with minimal noise. The cost-probe plan task should
     time the epoch, compute `projected_full_cost = measured_epoch_hrs * n_epochs * 6_configs`,
     and present that to the user at the gate.

3. **Mixed-precision training (AMP) on the GPU host.**
   - What we know: The GPU host has CUDA (the planning VM has PyTorch 2.12+cu130 installed).
     AMP (`torch.cuda.amp.autocast`) can significantly speed up ViT forward passes.
   - What's unclear: Whether AMP is stable with the existing backbone code
     (SigLIP vision tower fp32 weights; GroupNorm in decoder).
   - Recommendation: Plan AMP as an optional flag (`--amp`). Document as Claude's
     discretion; default to fp32 for correctness in the first probe run, enable AMP
     once the probe confirms training stability.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.13 | Training script | ✓ (planning VM) | 3.13.13 | — |
| PyTorch | Training loop | ✓ (planning VM) | 2.12.0+cu130 | — |
| CUDA GPU | Training execution | ✗ (planning VM — CPU only) | — | GPU box; see D-06 |
| gcsfs | GCS checkpoint I/O | ✗ (not installed) | 2026.5.0 (pip installable) | `gsutil` CLI in hand commands |
| google-cloud-storage | GCS checkpoint listing | ✗ (not installed) | 3.10.1 (pip installable) | `gsutil ls` in hand commands |
| gsutil CLI | GPU-host gate commands | ✓ (planning VM) | 5.37 | — |
| gcloud SDK | GCS auth | ✓ (planning VM) | 568.0.0 | — |
| timm | DINOv2 + Swin backbones | ✓ | 1.0.27 | — |
| peft | SigLIP LoRA adapter | ✓ | 0.19.1 | — |
| transformers | PaliGemma base model | ✓ | 5.8.1 | — |
| pytest | Offline training tests | ✓ | 9.0.3 | — |

**Missing dependencies with no fallback (must be installed on GPU host before execution):**
- CUDA GPU: provided by the GPU box (D-06); not a blocker for planning.

**Missing dependencies with fallback:**
- gcsfs / google-cloud-storage: pip installable on GPU host; `gsutil` CLI covers manual
  operations in the hand-commands artifact.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest 9.0.3 |
| Config file | `pytest.ini` (testpaths=tests, pythonpath=scripts, markers: integration) |
| Quick run command | `pytest tests/test_seg_training.py tests/test_seg_eval.py tests/test_seg_gcs.py -x` |
| Full suite command | `pytest tests/ -x --ignore=tests/integration` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EVAL-02 | Joint NLL = -(log p_lc + log p_topo) computed correctly over held-out tiles | unit | `pytest tests/test_seg_eval.py::test_nll_formula -x` | ❌ Wave 0 |
| EVAL-02 | NLL ignores pixels outside valid class range | unit | `pytest tests/test_seg_eval.py::test_nll_ignores_invalid_pixels -x` | ❌ Wave 0 |
| EVAL-02 | Eval harness enumerates only 224-level leaf tiles | unit | `pytest tests/test_seg_eval.py::test_eval_uses_only_leaf_tiles -x` | ❌ Wave 0 |
| EVAL-03 | All three backbones (SigLIP/DINOv2/Swin) produce comparable-shape NLL output | unit | `pytest tests/test_seg_eval.py::test_eval_all_backbones -x` | ❌ Wave 0 |
| EVAL-01 (preserved) | Eval harness never enumerates train/ paths | unit | `pytest tests/test_seg_eval.py::test_eval_leakage_guard -x` | ❌ Wave 0 |
| D-09 | GCS checkpoint write + resume round-trip | unit (mock) | `pytest tests/test_seg_gcs.py -x` | ❌ Wave 0 |
| training loop | Weighted joint loss decreases on overfit-a-batch | unit | `pytest tests/test_seg_training.py::test_overfit_single_pyramid -x` | ❌ Wave 0 |
| training loop | Teacher-forced prior is detached (no grad through prior) | unit | `pytest tests/test_seg_training.py::test_prior_detached -x` | ❌ Wave 0 |
| D-07 | Checkpoint path uses `<backbone>-<variant>` naming | unit | `pytest tests/test_seg_gcs.py::test_config_naming -x` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `pytest tests/test_seg_training.py tests/test_seg_eval.py tests/test_seg_gcs.py -x`
- **Per wave merge:** `pytest tests/ -x --ignore=tests/integration`
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `tests/test_seg_training.py` — covers training loop, weighted loss, teacher-forcing, overfit-batch
- [ ] `tests/test_seg_eval.py` — covers NLL formula, leaf tile filtering, leakage guard, backbone comparison
- [ ] `tests/test_seg_gcs.py` — covers GCS checkpoint write/resume (mocked gcsfs)

*(Existing test infrastructure covers Phase-3 model; Phase 4 adds training + eval tests.)*

---

## Security Domain

> `security_enforcement` not set in config.json — treated as enabled.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | No (single-researcher pipeline, no user-facing auth) | — |
| V3 Session Management | No | — |
| V4 Access Control | Partial (GCS bucket) | IAM via `gcloud auth application-default login`; bucket is project-private |
| V5 Input Validation | Yes | `split.json` test-map IDs validated before listing dirs; `sample_weights.json` loaded as JSON with dict-key checks |
| V6 Cryptography | No | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| GCS checkpoint path traversal (malformed config name) | Tampering | Config name validated as `^[a-z0-9][a-z0-9\-]*$` before building GCS URI; no user-supplied path segments |
| `split.json` map IDs containing path traversal (`../../`) | Tampering | Resolve map IDs against a known data root and assert the resolved path is a subpath of that root before opening |
| Corrupt/truncated GCS checkpoint loaded silently | Denial of Service (self) | Wrap `torch.load` in try/except; on failure skip that checkpoint and try the next-latest |
| `sample_weights.json` with missing keys causing KeyError in loss | Denial of Service (self) | Validate dict keys against `LANDCOVER_CLASSES` before building the weight tensor; raise a descriptive error |

---

## Sources

### Primary (HIGH confidence)

- `scripts/seg/model.py` — SegModelVariantA/B param structure, trainability, forward API [VERIFIED]
- `scripts/seg/backbones.py` — backbone constructors, feature_strides/channels, Variant B patch-embed widening [VERIFIED]
- `scripts/seg/decoder.py` — SegDecoder out_channels=256 [VERIFIED]
- `scripts/seg/heads.py` — LandCoverHead (9), TopographyHead (3), raw logits [VERIFIED]
- `scripts/seg/recursive.py` — `recursive_predict`, `recursive_predict_variant_b`, `crop_prior_to_child`, `cold_start_prior` [VERIFIED]
- `scripts/seg/dataset.py` — PyramidDataset sample dict schema, EVAL-01 guard [VERIFIED]
- `scripts/historical/label.py` — `sample_weights.json` schema (HISTORICAL_LC_WEIGHTS) [VERIFIED]
- `scripts/synthetic_weights.py` — synthetic uniform-1.0 weights [VERIFIED]
- `scripts/biome_mapping.py` — LANDCOVER_CLASSES (9), TOPO_CLASSES (3) [VERIFIED]
- `pip3 list` / `gsutil version` / `gcloud --version` — installed package versions [VERIFIED]
- `pip install --dry-run gcsfs google-cloud-storage` — installable versions [VERIFIED]
- `gsutil ls gs://mapclass-training-northeast1/` — bucket `models/` prefix existence [VERIFIED]
- In-session PyTorch tests (overfit batch, log_softmax, per-sample weighting) [VERIFIED]

### Secondary (MEDIUM confidence)

- Phase-2 `02-03-PLAN.md` + `02-04-PLAN.md` — `checkpoint:decision` gate pattern with `autonomous: false` [VERIFIED: codebase]
- `04-CONTEXT.md` decisions D-01..D-09 — all locked constraints [VERIFIED]
- `03-CONTEXT.md` decisions D-01..D-06a — Phase-3 model architecture [VERIFIED]

### Tertiary (LOW confidence / ASSUMED)

- AdamW weight-decay exclusion for bias/norm layers is canonical for ViT fine-tuning [ASSUMED]
- Teacher-forced detach is the standard approach for multi-pass autoregressive dense prediction training [ASSUMED]
- AMP is stable with SigLIP GroupNorm decoder on the GPU host [ASSUMED — test on first probe run]

---

## Metadata

**Confidence breakdown:**

- Standard stack: HIGH — all major packages verified against installed pip list or pip
  dry-run; GCS bucket verified with gsutil
- Architecture: HIGH — derived entirely from reading and verifying Phase-3 source code;
  no architectural assumptions needed
- Pitfalls: HIGH (Phase-3 pitfalls inherited and confirmed) / MEDIUM (GCS atomicity —
  cannot test without live GPU)
- Training methodology: MEDIUM — teacher-forcing is the standard approach but not verified
  against literature in this session (tagged ASSUMED)

**Research date:** 2026-05-16
**Valid until:** 2026-06-16 (30 days; gcsfs version may update; PyTorch/timm API is stable)
