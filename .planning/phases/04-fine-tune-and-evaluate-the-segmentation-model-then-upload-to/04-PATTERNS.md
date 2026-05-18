# Phase 4: Fine-tune and Evaluate the Segmentation Model - Pattern Map

**Mapped:** 2026-05-16
**Files analyzed:** 7 (3 new scripts + 3 new test files + 1 new artifact dir)
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `scripts/finetune_seg.py` | script/training-loop | batch → backward → GCS checkpoint | `scripts/finetune_paligemma.py` | role-match (same training-loop shape; different model + GCS vs local) |
| `scripts/evaluate_seg.py` | script/eval-harness | c2f inference → NLL aggregation | `scripts/seg/recursive.py` + `finetune_paligemma.py` eval block | partial (eval harness; c2f walk pattern exact) |
| `scripts/report_comparison.py` | utility/report | JSON read → markdown render | `finetune_paligemma.py` CLI+argparse block | partial (CLI pattern; report rendering is novel) |
| `tests/test_seg_training.py` | test | offline overfit + grad-flow assertions | `tests/test_seg_smoke.py` + `tests/test_seg_dataset.py` | exact (mini_pyramid fixture + import-guard + class-per-concern pattern) |
| `tests/test_seg_eval.py` | test | NLL formula + eval guard assertions | `tests/test_seg_recursive.py` + `tests/test_seg_dataset.py` | exact (same fixture, same class-per-concern test layout) |
| `tests/test_seg_gcs.py` | test | mock GCS round-trip | `tests/test_seg_dataset.py` (mock filesystem pattern) | role-match (mock-object pattern; GCS is novel) |
| `metrics/` + `docs/04-comparison-report.md` | artifact/output | JSON write + markdown | `02-HUMAN-UAT.md` (artifact produced by gated wave) | partial (gated artifact pattern) |

---

## Pattern Assignments

### `scripts/finetune_seg.py` (script, training-loop → GCS checkpoint)

**Analog:** `scripts/finetune_paligemma.py`

**Imports pattern** (`finetune_paligemma.py` lines 33–56):
```python
import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from seg.model import SegModelVariantA, SegModelVariantB
from seg.dataset import PyramidDataset
from seg.recursive import cold_start_prior, crop_prior_to_child
from biome_mapping import LANDCOVER_CLASSES
```

**Device + dtype pattern** (`finetune_paligemma.py` lines 211–213):
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
print(f"Device: {device} | dtype: {dtype}")
```

**Trainable-param-only optimizer pattern** (`finetune_paligemma.py` lines 251–257):
```python
# Filter frozen params — backbone has requires_grad_(False) set at construction
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=args.lr,
    weight_decay=0.01,
)
total_steps = (len(loader) // args.grad_accum) * args.epochs
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
```

**Grad-clip + step pattern** (`finetune_paligemma.py` lines 279–291):
```python
torch.nn.utils.clip_grad_norm_(
    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
)
optimizer.step()
scheduler.step()
optimizer.zero_grad()
global_step += 1

if global_step % 20 == 0:
    avg = epoch_loss / step
    lr_now = scheduler.get_last_lr()[0]
    print(f"  Epoch {epoch} step {global_step} | loss {avg:.4f} | lr {lr_now:.2e}")
```

**DataLoader construction pattern** (`finetune_paligemma.py` lines 242–248):
```python
loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
```

**Val-carve pattern** (from RESEARCH.md Pattern 6 — no existing codebase analog; use this):
```python
from torch.utils.data import random_split
from seg.dataset import PyramidDataset

ds = PyramidDataset(train_root_dirs)   # only train/ subtrees; EVAL-01 guard fires at constructor
gen = torch.Generator().manual_seed(42)
n_val = int(0.2 * len(ds))
n_train = len(ds) - n_val
train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
```

**GCS checkpoint write/resume pattern** (from RESEARCH.md Pattern 3 — novel to this codebase):
```python
import gcsfs, io, re

GCS_PROJECT = "narrative-campaign"
BUCKET_PREFIX = "gs://mapclass-training-northeast1/models"

def gcs_save_checkpoint(config_name: str, step: int, state: dict):
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    path = f"{BUCKET_PREFIX}/{config_name}/step_{step:07d}.pt"
    buf = io.BytesIO()
    torch.save(state, buf)
    buf.seek(0)
    with fs.open(path, "wb") as f:
        f.write(buf.read())

def gcs_latest_checkpoint(config_name: str):
    fs = gcsfs.GCSFileSystem(project=GCS_PROJECT)
    prefix = f"mapclass-training-northeast1/models/{config_name}/"
    try:
        files = fs.ls(prefix)
    except FileNotFoundError:
        return 0, None
    step_files = [f for f in files if re.search(r"step_(\d+)\.pt$", f)]
    if not step_files:
        return 0, None
    latest = max(step_files, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    latest_step = int(re.search(r"step_(\d+)", latest).group(1))
    with fs.open(f"gs://{latest}", "rb") as f:
        state = torch.load(io.BytesIO(f.read()), weights_only=False)
    return latest_step, state
```

**Teacher-forced c2f training step pattern** (from RESEARCH.md Pattern 2 — novel to codebase; mirrors `recursive.py` walk structure):
```python
# Mirrors the depth-first walk in seg/recursive.py _predict() but with grads
# enabled and detach() after each tile's prob_cache entry.

from seg.recursive import crop_prior_to_child, cold_start_prior

def train_step_variant_a(model, pyramid_batch, optimizer, device):
    total_loss = 0.0
    for pyr in pyramid_batch:
        manifest_by_id = {t["id"]: t for t in pyr["manifest"]}
        root = next(t for t in pyr["manifest"] if t["size"] == 896)
        prob_cache = {}

        def predict_tile(tile, parent):
            rgb = pyr["rgb_tiles"][tile["id"]].to(device)
            if parent is None:
                prior = cold_start_prior(1, tile["size"]).to(device)
            else:
                prior = crop_prior_to_child(
                    prob_cache[parent["id"]].detach(),  # KEY: stop_grad
                    parent, tile
                )
            lc_logits, topo_logits = model(rgb, prior)
            lc_prob = torch.softmax(lc_logits.detach(), dim=1)
            topo_prob = torch.softmax(topo_logits.detach(), dim=1)
            prob_cache[tile["id"]] = torch.cat([lc_prob, topo_prob], dim=1)
            ...
```

**Weighted joint loss pattern** (from RESEARCH.md Pattern 1 — novel to codebase):
```python
import torch.nn.functional as F
from biome_mapping import LANDCOVER_CLASSES

def build_lc_weight_tensor(sw_dict: dict, device) -> torch.Tensor:
    lc_w = sw_dict["land_cover_weights"]
    return torch.tensor(
        [lc_w[c] for c in LANDCOVER_CLASSES], dtype=torch.float32, device=device
    )

def weighted_joint_loss(lc_logits, topo_logits, lc_targets, topo_targets,
                        sample_weights_batch, device):
    B = lc_logits.shape[0]
    total_loss = torch.tensor(0.0, device=device)
    for i, sw in enumerate(sample_weights_batch):
        lc_class_w = build_lc_weight_tensor(sw, device)
        topo_w = torch.tensor(sw["topography_weight"], device=device)
        lc_loss = F.cross_entropy(
            lc_logits[i:i+1], lc_targets[i:i+1],
            weight=lc_class_w, reduction='mean'
        )
        topo_loss = F.cross_entropy(
            topo_logits[i:i+1], topo_targets[i:i+1],
            reduction='mean'
        ) * topo_w
        total_loss = total_loss + (lc_loss + topo_loss)
    return total_loss / B
```

**AdamW param-group pattern** (RESEARCH.md Pattern 5 — novel; extends finetune_paligemma.py pattern):
```python
def make_optimizer(model, lr=1e-4, weight_decay=1e-2):
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

**CLI/argparse pattern** (`finetune_paligemma.py` lines 308–330):
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune seg model on pyramid dataset")
    parser.add_argument("--backbone",    choices=["siglip","dinov2","swin"], default="siglip")
    parser.add_argument("--variant",     choices=["A","B"], default="B",
                        help="Model variant (A=frozen-backbone, B=widened-patch-embed)")
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--ckpt-every",  type=int,   default=100,
                        help="Save GCS checkpoint every N steps")
    parser.add_argument("--probe",       action="store_true",
                        help="Cost-probe mode: run 1 epoch, print GPU-hrs, exit")
    parser.add_argument("--pretrained",  action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                        help="Use random weights (offline CI only)")
    parser.add_argument("--amp",         action="store_true",
                        help="Enable AMP (optional; default fp32 for first probe)")
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()
```

---

### `scripts/evaluate_seg.py` (script, eval-harness)

**Analog:** `scripts/seg/recursive.py` (c2f walk), `scripts/finetune_paligemma.py` (CLI scaffold)

**Imports pattern** (mirrors recursive.py lines 49–57 + finetune_paligemma.py lines 33–36):
```python
import argparse, json, sys
from pathlib import Path

import torch

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from seg.model import SegModelVariantA, SegModelVariantB
from seg.recursive import recursive_predict, recursive_predict_variant_b
```

**Joint-NLL eval pattern** (RESEARCH.md Pattern 4 — novel; uses existing `recursive_predict`):
```python
def evaluate_joint_nll(model, test_pyramid_dirs, variant="A", device="cpu"):
    model.eval()
    total_nll = 0.0
    total_pixels = 0
    predict_fn = recursive_predict if variant == "A" else recursive_predict_variant_b

    with torch.no_grad():
        for pdir in test_pyramid_dirs:
            prob_cache = predict_fn(model, pdir)
            manifest = json.loads((Path(pdir) / "pyramid.json").read_text())
            # Only leaf (224-level) tiles — see Pitfall 7
            leaf_tiles = [t for t in manifest["tiles"] if t["size"] == 224]

            for tile in leaf_tiles:
                prob12 = prob_cache[tile["id"]].to(device)
                lc_prob = prob12[:, :9]      # ALREADY softmaxed by recursive_predict
                topo_prob = prob12[:, 9:]    # ALREADY softmaxed by recursive_predict
                lc_gt = _load_label(Path(pdir) / tile["land_cover"]).to(device)
                topo_gt = _load_label(Path(pdir) / tile["topography"]).to(device)

                # recursive_predict returns a probability tensor (already softmaxed),
                # so take log directly with a floor — do NOT log_softmax again
                # (that would double-normalise). See plan 04-03 Task 1.
                log_lc = torch.log(lc_prob.clamp_min(1e-12))
                log_topo = torch.log(topo_prob.clamp_min(1e-12))

                lc_nll = -log_lc[0].gather(0, lc_gt.unsqueeze(0)).squeeze(0)
                topo_nll = -log_topo[0].gather(0, topo_gt.unsqueeze(0)).squeeze(0)
                joint_nll = lc_nll + topo_nll

                valid = (lc_gt < 9) & (topo_gt < 3)
                total_nll += joint_nll[valid].sum().item()
                total_pixels += valid.sum().item()

    return total_nll / total_pixels if total_pixels > 0 else float("nan")
```

**split.json test-set enumeration pattern** (mirrors PyramidDataset constructor's manifest-driven approach, `dataset.py` lines 94–115):
```python
def load_test_pyramid_dirs(split_json: Path, data_root: Path) -> list[Path]:
    """
    Enumerate held-out test pyramid directories from split.json.
    Only called from evaluate_seg.py — NEVER from training scripts (EVAL-01).
    """
    split = json.loads(split_json.read_text())
    test_ids = split["test"]
    dirs = []
    for map_id in test_ids:
        pyr_root = data_root / "test" / map_id
        pyramid_dirs = sorted(p for p in pyr_root.iterdir()
                              if p.is_dir() and (p / "pyramid.json").exists())
        dirs.extend(pyramid_dirs)
    return dirs
```

**Metrics JSON write pattern** (follows D-08 — lightweight artifact to git):
```python
import json
from pathlib import Path

def write_nll_metrics(config_name: str, nll: float, metrics_dir: Path) -> Path:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{config_name}-nll.json"
    out.write_text(json.dumps({
        "config": config_name,
        "joint_nll": nll,
        "eval_date": "2026-05-16",
    }, indent=2))
    return out
```

---

### `scripts/report_comparison.py` (utility, JSON → markdown report)

**Analog:** `scripts/finetune_paligemma.py` (CLI/argparse structure only)

**CLI pattern** (`finetune_paligemma.py` lines 308–327):
```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Render A/B × backbone comparison report")
    parser.add_argument("--metrics-dir", default="metrics/",
                        help="Directory containing <config>-nll.json files")
    parser.add_argument("--manifest",    default="metrics/checkpoint_manifest.json")
    parser.add_argument("--output",      default="docs/04-comparison-report.md")
    args = parser.parse_args()
    render_report(args)

if __name__ == "__main__":
    main()
```

**No closer analog exists for the report rendering logic itself** — see "No Analog Found" section.

---

### `tests/test_seg_training.py` (test, offline training assertions)

**Analog:** `tests/test_seg_smoke.py` (primary) + `tests/test_seg_dataset.py` (secondary)

**File-level import guard pattern** (`test_seg_smoke.py` lines 28–40 and `test_seg_dataset.py` lines 36–48):
```python
_import_error: Exception | None = None
try:
    from seg.model import SegModelVariantA, SegModelVariantB
    # Also import training utilities when they exist:
    # from finetune_seg import weighted_joint_loss, train_step_variant_a
except ImportError as _e:
    _import_error = _e


def _require_training():
    if _import_error is not None:
        pytest.fail(
            f"finetune_seg not yet implemented: {_import_error}",
            pytrace=False,
        )
```

**mini_pyramid + stub_vision_config fixture usage pattern** (`test_seg_smoke.py` lines 103–117):
```python
def test_overfit_single_pyramid(self, mini_pyramid, stub_vision_config):
    """Loss must decrease over N steps on a single pyramid (overfit-a-batch)."""
    _require_training()
    import torch
    model = SegModelVariantA(
        backbone_name="siglip",
        stub_config=stub_vision_config,
    )
    # ... training assertions using mini_pyramid as the data fixture
```

**Class-per-concern test layout** (`test_seg_smoke.py` lines 97–171, `test_seg_recursive.py` lines 65–168):
```python
class TestWeightedJointLoss:
    """Loss formula shape + value sanity."""

    def test_loss_scalar(self, ...): ...
    def test_loss_decreases_on_overfit(self, ...): ...


class TestTeacherForcedDetach:
    """Prior must be detached after coarse pass (gradient guard)."""

    def test_prior_detached(self, ...): ...
    def test_no_oom_three_level_walk(self, ...): ...


class TestValCarve:
    """random_split carve from train_ds only (EVAL-01 guard)."""

    def test_val_contains_no_test_paths(self, ...): ...
```

**torch.no_grad / grad assertion pattern** (`test_seg_smoke.py` lines 152–171):
```python
def test_prior_detached_no_grad(self, mini_pyramid, stub_vision_config):
    """
    After the coarse tile forward, prob_cache entry must have requires_grad=False
    (teacher-forced; coarse logits must be detached before softmax caching).
    """
    import torch
    # ... verify prob_cache[tile_id].requires_grad is False
    assert not prob_cache[root_id].requires_grad, (
        "prob_cache entry must be detached (teacher-forcing); "
        "grad through both passes causes OOM (Pitfall 2)"
    )
```

---

### `tests/test_seg_eval.py` (test, NLL formula + eval guard)

**Analog:** `tests/test_seg_recursive.py` (primary) + `tests/test_seg_dataset.py` (EVAL-01 leakage guard)

**Import guard + module-level skip pattern** (`test_seg_recursive.py` lines 34–50):
```python
_import_error: Exception | None = None
try:
    from evaluate_seg import evaluate_joint_nll, load_test_pyramid_dirs
except ImportError as _e:
    _import_error = _e


def _require_eval():
    if _import_error is not None:
        pytest.fail(
            f"evaluate_seg not yet implemented: {_import_error}",
            pytrace=False,
        )
```

**Synthetic-logits NLL formula test pattern** (`test_seg_recursive.py` lines 177–196, parametrized):
```python
class TestNLLFormula:
    """NLL = -(log p_lc[true] + log p_topo[true]) per pixel."""

    def test_nll_formula_correct(self):
        """Known logits → known NLL; verify against manual calculation."""
        _require_eval()
        import torch
        # Uniform logits → log(1/9) + log(1/3) = -log(9) - log(3)
        lc_logits = torch.zeros(1, 9, 4, 4)
        topo_logits = torch.zeros(1, 3, 4, 4)
        lc_gt = torch.zeros(4, 4, dtype=torch.long)
        topo_gt = torch.zeros(4, 4, dtype=torch.long)
        expected_nll = torch.log(torch.tensor(9.0)) + torch.log(torch.tensor(3.0))
        # ... assert computed NLL ≈ expected_nll

    def test_nll_uses_log_softmax_not_log_softmax_naive(self):
        """torch.log_softmax must be used, not log(softmax(...))."""
        ...

    def test_nll_ignores_invalid_pixels(self):
        """Pixels with lc_gt >= 9 or topo_gt >= 3 must be excluded."""
        ...

    def test_eval_uses_only_leaf_tiles(self, mini_pyramid, stub_vision_config):
        """Only size==224 tiles are included in NLL; 896 and 448 are excluded."""
        ...
```

**Leakage guard test pattern** (`test_seg_dataset.py` lines 140–194):
```python
class TestEvalLeakageGuard:
    """evaluate_seg.py must never enumerate train/ paths."""

    def test_load_test_dirs_only_test_paths(self, tmp_path):
        """
        load_test_pyramid_dirs must only return paths under test/.
        Mirrors test_seg_dataset.py::TestSplitSafety pattern.
        """
        _require_eval()
        # Build a split.json with known test IDs; assert returned dirs are all test/
        ...
        for d in dirs:
            assert "test" in Path(d).parts, (
                f"evaluate_seg must only enumerate test/ paths, got: {d}"
            )
```

---

### `tests/test_seg_gcs.py` (test, mock GCS checkpoint round-trip)

**Analog:** `tests/test_seg_dataset.py` (mock-filesystem + import-guard + class-per-concern)

**Import guard pattern** (`test_seg_dataset.py` lines 36–48):
```python
_import_error: Exception | None = None
try:
    from finetune_seg import gcs_save_checkpoint, gcs_latest_checkpoint
except ImportError as _e:
    _import_error = _e


def _require_gcs():
    if _import_error is not None:
        pytest.fail(
            f"finetune_seg GCS helpers not yet implemented: {_import_error}",
            pytrace=False,
        )
```

**mock-object filesystem pattern** (`tests/conftest.py` lines 235–262, _MockStacItem):
```python
import unittest.mock as mock

class _MockGCSFileSystem:
    """Offline stand-in for gcsfs.GCSFileSystem."""

    def __init__(self, project=None):
        self._store: dict[str, bytes] = {}

    def open(self, path: str, mode: str = "rb"):
        ...

    def ls(self, prefix: str) -> list[str]:
        return [k for k in self._store if k.startswith(prefix.lstrip("gs://"))]
```

**D-07 config-naming unit test pattern** (`test_seg_recursive.py` lines 191–196):
```python
class TestConfigNaming:
    """GCS checkpoint prefix uses <backbone>-<variant> naming (D-07)."""

    @pytest.mark.parametrize("backbone,variant,expected_prefix", [
        ("siglip", "A", "siglip-A"),
        ("siglip", "B", "siglip-B"),
        ("dinov2", "A", "dinov2-A"),
        ("swin",   "B", "swin-B"),
    ])
    def test_config_name_format(self, backbone, variant, expected_prefix):
        ...
```

**GCS round-trip test pattern** (novel — use tmp_path mock; mirrors conftest.py tmp_path pattern):
```python
class TestGCSCheckpointRoundTrip:
    """Checkpoint write + resume round-trip using mocked gcsfs."""

    def test_save_and_resume(self, tmp_path):
        _require_gcs()
        import torch
        with mock.patch("finetune_seg.gcsfs.GCSFileSystem", _MockGCSFileSystem):
            state = {"step": 42, "model_state_dict": {}, "optimizer_state_dict": {}}
            gcs_save_checkpoint("siglip-B", step=42, state=state)
            step, loaded = gcs_latest_checkpoint("siglip-B")
        assert step == 42
        assert loaded["step"] == 42

    def test_resume_returns_zero_on_empty_prefix(self):
        _require_gcs()
        with mock.patch("finetune_seg.gcsfs.GCSFileSystem", _MockGCSFileSystem):
            step, state = gcs_latest_checkpoint("nonexistent-config")
        assert step == 0 and state is None
```

---

## Shared Patterns

### Gemma Exclusion (SigLIP path only)
**Source:** `scripts/finetune_paligemma.py` lines 163–203 (`apply_lora`), `scripts/seg/model.py` lines 258–260 (`_assert_no_gemma`)
**Apply to:** `scripts/finetune_seg.py` (SigLIP variant instantiation)
```python
# seg/model.py _assert_no_gemma called at SegModelVariantA/B construction.
# finetune_seg.py must NOT load the full PaliGemmaForConditionalGeneration;
# load only the vision tower adapter via seg/backbones.py SiglipBackbone.
# Pattern from finetune_paligemma.py line 178:
model.language_model.requires_grad_(False)  # Gemma weights frozen at Phase-1
# Phase-4: Gemma is never instantiated; SiglipBackbone loads vision tower only.
```

### EVAL-01 Zero-Leakage Guard
**Source:** `scripts/seg/dataset.py` lines 125–134, `tests/test_seg_dataset.py` lines 140–194
**Apply to:** `scripts/finetune_seg.py` (dataset construction), `scripts/evaluate_seg.py` (split.json enumeration), `tests/test_seg_training.py` (val-carve test), `tests/test_seg_eval.py` (leakage guard test)
```python
# dataset.py lines 125-134: EVAL-01 hard guard at PyramidDataset construction
for entry in self.samples:
    parts = Path(entry["pdir"]).parts
    if "test" in parts:
        raise ValueError(
            f"EVAL-01 violation: dataset path contains a 'test/' component: "
            f"{entry['pdir']}\n"
            "PyramidDataset must never enumerate paths under test/ (Pitfall 6)."
        )
# finetune_seg.py must construct PyramidDataset from train/ roots ONLY.
# evaluate_seg.py must read split.json["test"] and enumerate data/synthetic/test/ ONLY.
```

### Module sys.path bootstrap
**Source:** `scripts/finetune_paligemma.py` lines 43–45
**Apply to:** `scripts/finetune_seg.py`, `scripts/evaluate_seg.py`, `scripts/report_comparison.py`
```python
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
```

### Import guard + graceful fail in tests
**Source:** `tests/test_seg_smoke.py` lines 28–40, `tests/test_seg_dataset.py` lines 36–48, `tests/test_seg_recursive.py` lines 34–50
**Apply to:** `tests/test_seg_training.py`, `tests/test_seg_eval.py`, `tests/test_seg_gcs.py`
```python
_import_error: Exception | None = None
try:
    from <module> import <symbols>
except ImportError as _e:
    _import_error = _e

def _require_<module>():
    if _import_error is not None:
        pytest.fail(
            f"<module> not yet implemented: {_import_error}",
            pytrace=False,
        )
```

### mini_pyramid fixture (offline Phase-2 pyramid)
**Source:** `tests/conftest.py` lines 306–340
**Apply to:** `tests/test_seg_training.py`, `tests/test_seg_eval.py`
```python
# All three new test files consume the same mini_pyramid fixture from conftest.py.
# It provides a real 21-tile 896→448→224 pyramid (1792x1792 source) with
# sample_weights.json, produced by tiling.tile() under tmp_path.
# Fully offline: no torch/transformers/timm at fixture construction time.
def test_something(self, mini_pyramid, stub_vision_config):
    man = json.loads((mini_pyramid / "pyramid.json").read_text())
    ...
```

### stub_vision_config fixture (offline SigLIP shape tests)
**Source:** `tests/conftest.py` lines 342–362
**Apply to:** `tests/test_seg_training.py`, `tests/test_seg_eval.py`
```python
# stub_vision_config provides hidden_size=1152, patch_size=14, num_hidden_layers=27
# for offline model instantiation without loading the 3B PaliGemma checkpoint.
# Pass as stub_config= to SegModelVariantA/B constructors.
model = SegModelVariantA(backbone_name="siglip", stub_config=stub_vision_config)
```

### pytest class-per-concern layout
**Source:** `tests/test_seg_smoke.py` lines 97–247, `tests/test_seg_recursive.py` lines 65–241
**Apply to:** all three new test files
```python
class TestWeightedJointLoss: ...       # test_seg_training.py
class TestTeacherForcedDetach: ...     # test_seg_training.py
class TestValCarve: ...                # test_seg_training.py
class TestNLLFormula: ...              # test_seg_eval.py
class TestEvalLeakageGuard: ...        # test_seg_eval.py
class TestGCSCheckpointRoundTrip: ...  # test_seg_gcs.py
class TestConfigNaming: ...            # test_seg_gcs.py
```

### Numerically-stable log of an already-softmaxed probability tensor
**Source:** RESEARCH.md Pattern 4 / Anti-Patterns section — no codebase analog yet
**Apply to:** `scripts/evaluate_seg.py` (NLL formula), `tests/test_seg_eval.py` (formula correctness test)

> NOTE: `recursive_predict` / `recursive_predict_variant_b` return a tensor that is
> ALREADY a probability (softmax already applied). The eval harness must therefore take
> `log` with a floor — NOT `log_softmax` (which would double-normalise the already-
> normalised probabilities). This matches plan 04-03 Task 1's correct action. The
> `log_softmax` form would only be correct if the eval consumed raw logits, which it
> does not.
```python
# Correct — recursive_predict output is already softmaxed, so log with a floor:
log_lc = torch.log(lc_prob.clamp_min(1e-12))
# WRONG (do not use) — double-normalises an already-softmaxed tensor:
# log_lc = torch.log_softmax(lc_prob, dim=1)
```

---

## No Analog Found

Files with no close match in the codebase (planner should use RESEARCH.md patterns instead):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `scripts/report_comparison.py` (rendering logic) | utility | JSON → markdown | No existing report-rendering script; CLI scaffold pattern from `finetune_paligemma.py` applies but the table/section rendering is wholly new |
| `metrics/*.json` + `metrics/checkpoint_manifest.json` | artifact | write-once | No existing metrics/ dir or JSON manifest convention; follows D-08 lightweight-artifact-to-git pattern but has no code analog |
| GCS checkpoint I/O (`gcs_save_checkpoint`, `gcs_latest_checkpoint`) | I/O utility | streaming write + resume | All prior checkpoints are local (`finetune_paligemma.py` saves to `Path / f"epoch{epoch:02d}"`); GCS via gcsfs is new to the codebase |
| `docs/04-comparison-report.md` | report artifact | produced by script | Artifact format novel; use RESEARCH.md §Architecture Patterns "Recommended Project Structure" as structural reference |

---

## Key Pattern Decisions for Planner

1. **Training loop structure:** One pyramid per training step (not individual tiles); the recursive c2f walk processes all 21 tiles depth-first with teacher-forced detach. Mirror `recursive.py`'s `_predict()` but with `requires_grad=True` on logits and `detach()` before caching probabilities.

2. **Optimizer:** AdamW with two param groups (weight-decay excluded from bias/norm). Filter `p.requires_grad` — backbone constructors already set frozen params to `requires_grad=False`. Source: `finetune_paligemma.py` lines 251–257 + RESEARCH.md Pattern 5.

3. **GCS I/O is wholly novel:** No existing `gcsfs` usage in the codebase. Pattern comes entirely from RESEARCH.md Pattern 3 (verified against installed packages). The `test_seg_gcs.py` tests mock `gcsfs.GCSFileSystem`.

4. **NLL eval uses existing `recursive_predict`:** `evaluate_seg.py` does NOT re-implement the c2f walk; it calls `seg.recursive.recursive_predict` / `recursive_predict_variant_b` directly. Only the NLL aggregation over the returned `prob_cache` is new.

5. **Test file structure is fully determined by existing Phase-3 test files:** import guard, mini_pyramid + stub_vision_config fixtures, class-per-concern layout, parametrize for backbone variants — all copy from `test_seg_smoke.py` and `test_seg_recursive.py`.

6. **Config naming for GCS prefixes:** `<backbone>-<variant>` lowercase, e.g. `siglip-A`, `siglip-B`, `dinov2-A`, `dinov2-B`, `swin-A`, `swin-B`. Validated with `^[a-z0-9][a-z0-9\-]*$` before building GCS URI (security, RESEARCH.md §Known Threat Patterns).

---

## Metadata

**Analog search scope:** `scripts/`, `scripts/seg/`, `tests/`, `.planning/phases/02-*/`, `.planning/phases/03-*/`
**Files scanned:** 10 source files read in full
**Pattern extraction date:** 2026-05-16
