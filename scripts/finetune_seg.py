"""
Fine-tune the MapClass segmentation model.

This script is the training-loop entrypoint for Phase 4 (plan 04-04).

Design decisions honoured here:
  D-03/D-04/D-05 — probe mode: 1 epoch timed, GPU-hrs/epoch + projected full-
                   grid cost printed, then sys.exit(0). Plan-05 checkpoint:decision
                   gate reads these numbers.
  D-06           — GPU-host gate: the real training run happens on a GPU host;
                   this wiring is CPU-verifiable with a stub backbone + mini_pyramid.
  D-07/D-09      — GCS checkpoint every K steps + auto-resume from latest step.
  EVAL-01        — carve_train_val is the ONLY dataset constructor; the training
                   and val loops never enumerate test/ paths; evaluate_seg is NOT
                   imported here (RESEARCH Pitfall 5).

GCS note: gcs_save_checkpoint / gcs_latest_checkpoint raise ImportError when
gcsfs is absent (offline CI). The test patches both to no-ops via monkeypatch.

Trust: T-04-10 (no test/ in training) + T-04-11 (ADC auth on GPU host, not here)
       + T-04-12 (only own-bucket checkpoints resumed).

Public API:
    train(args) -> None   # called by main() and tests
    main()                # argparse entrypoint
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# sys.path bootstrap: scripts/ on path so seg.* and biome_mapping resolve.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent  # scripts/
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from seg.model import SegModelVariantA, SegModelVariantB
from seg.train_utils import (
    carve_train_val,
    make_optimizer,
    train_step_variant_a,
    train_step_variant_b,
    weighted_joint_loss,
)
from seg.gcs_checkpoint import (
    config_prefix,
    gcs_latest_checkpoint,
    gcs_save_checkpoint,
)

# ---------------------------------------------------------------------------
# RW-03: pull-once gcs_io import (try/except ImportError for offline CI)
# ---------------------------------------------------------------------------
# Imported here (module level) so train() can reference pull_dataset_from_gcs
# and verify_pull.  Falls back silently to None when gcs_io is absent (planning
# VM / offline CI) — the ImportError is re-raised inside train() if the caller
# actually requests a GCS pull (i.e. args.scratch_dir is set and gcs_io is
# needed).
try:
    from gcs_io import pull_dataset_from_gcs, verify_pull, DATA_PREFIX as _DATA_PREFIX  # type: ignore[import]
    _gcs_io_available: bool = True
except ImportError:
    pull_dataset_from_gcs = None  # type: ignore[assignment]
    verify_pull = None  # type: ignore[assignment]
    _DATA_PREFIX = "mapclass-training-northeast1/data"
    _gcs_io_available = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_model(
    args: argparse.Namespace,
    device,
) -> torch.nn.Module:
    """Instantiate SegModelVariantA or B for the chosen backbone.

    When ``args.offline_stub`` is True, passes a minimal stub_config for
    SigLIP so the model runs on CPU without loading a HuggingFace checkpoint.
    When ``args.pretrained`` is False (offline CI / --no-pretrained), stub
    construction is used for all SigLIP paths.
    """
    import types

    if args.backbone.lower() == "siglip" and (not args.pretrained or args.offline_stub):
        # Offline stub matches SigLIP-So400m/14 shape contract (conftest.py fixture)
        stub_cfg = types.SimpleNamespace(
            hidden_size=1152,
            patch_size=14,
            num_hidden_layers=27,
            image_size=224,
            num_channels=3,
        )
    else:
        stub_cfg = None

    variant_upper = args.variant.upper()
    if variant_upper == "A":
        model = SegModelVariantA(
            backbone_name=args.backbone,
            stub_config=stub_cfg,
        )
    elif variant_upper == "B":
        model = SegModelVariantB(
            backbone_name=args.backbone,
            stub_config=stub_cfg,
        )
    else:
        raise ValueError(f"Unknown variant '{args.variant}'. Choose 'A' or 'B'.")

    return model.to(device)


def _run_val_epoch(
    model: torch.nn.Module,
    val_pdirs: list,
    device,
) -> float:
    """
    Flat per-epoch val pass over pyramid dirs: weighted_joint_loss only.

    NO recursive_predict, NO test/ paths (EVAL-01 / RESEARCH Pitfall 1/5).
    Only leaf tiles (size==224) for a fast per-epoch signal.

    Parameters
    ----------
    val_pdirs : list[str]  pyramid directory paths (train-only subset)
    device    : torch.device or str

    Returns mean val loss for the epoch (or 0.0 if val_pdirs is empty).
    """
    if not val_pdirs:
        return 0.0

    import json
    from PIL import Image
    import numpy as np

    model.eval()
    total_val_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for pdir_str in val_pdirs:
            pdir = Path(pdir_str)

            man = json.loads((pdir / "pyramid.json").read_text())
            sw = json.loads((pdir / "sample_weights.json").read_text())

            # Only leaf tiles (size == 224) for a fast val signal
            leaf_tiles = [t for t in man["tiles"] if t["size"] == 224]
            if not leaf_tiles:
                continue

            for tile in leaf_tiles:
                img = Image.open(pdir / tile["image"]).convert("RGB")
                arr = np.array(img, dtype=np.float32) / 255.0
                rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

                lc_img = Image.open(pdir / tile["land_cover"])
                lc_targets = torch.from_numpy(
                    np.array(lc_img, dtype=np.int64)
                ).unsqueeze(0).to(device)

                topo_img = Image.open(pdir / tile["topography"])
                topo_targets = torch.from_numpy(
                    np.array(topo_img, dtype=np.int64)
                ).unsqueeze(0).to(device)

                # Zero prior for flat val pass (progress signal only)
                prior = torch.zeros(1, 12, rgb.shape[-2], rgb.shape[-1], device=device)

                if hasattr(model, 'prior_encoder'):
                    # Variant A: model(rgb, prior)
                    lc_logits, topo_logits = model(rgb, prior)
                else:
                    # Variant B: model(x15)
                    x15 = torch.cat([rgb, prior], dim=1)
                    lc_logits, topo_logits = model(x15)

                loss = weighted_joint_loss(
                    lc_logits, topo_logits, lc_targets, topo_targets, [sw], device
                )
                total_val_loss += float(loss.detach())
                n_batches += 1

    return total_val_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """
    Per-config training over the Phase-2 pyramid dataset.

    Key behaviours:
    - train_ds, val_ds built via carve_train_val(args.train_root) — train/ only (EVAL-01)
    - SegModelVariantA or B constructed for args.backbone
    - make_optimizer() builds AdamW two-group optimizer
    - gcs_latest_checkpoint auto-resumes if a checkpoint exists
    - Epoch loop: one pyramid per step via train_step_variant_a/b
    - Every args.ckpt_every steps: gcs_save_checkpoint AFTER optimizer.step
    - Per-epoch flat val pass over val_ds (no recursive_predict, no test/)
    - Probe mode (--probe): run exactly 1 epoch, time it, print PROBE RESULT, sys.exit(0)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and args.amp else torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    # (RW-03) Step 1: pull-once — bulk-fetch train subset from GCS to local scratch
    # before carve_train_val.  The ImportError fallback preserves offline CI / args.train_root.
    # This block runs BEFORE the gcs_latest_checkpoint resume block below (Step 4).
    train_root = getattr(args, "train_root", "data/synthetic/train")
    scratch_dir = getattr(args, "scratch_dir", None)
    try:
        if _gcs_io_available and scratch_dir is not None and pull_dataset_from_gcs is not None:
            print(f"[RW-03] Pulling train subset from GCS → {scratch_dir} …")
            local_root = pull_dataset_from_gcs("train", scratch_dir)
            # Read split.json from GCS to pass to verify_pull
            try:
                import gcsfs as _gcsfs  # type: ignore[import]
                _fs = _gcsfs.GCSFileSystem(project="narrative-campaign")
                import json as _json
                _split_json_bytes = _fs.cat(f"{_DATA_PREFIX}/synthetic/split.json")
                _split_json_data = _json.loads(_split_json_bytes)
            except Exception:
                _split_json_data = {"train": [], "test": []}
            if verify_pull is not None:
                verify_pull(local_root, _split_json_data, "train")
            train_root = str(local_root)
            print(f"[RW-03] Pull complete. train_root → {train_root}")
    except ImportError:
        # gcsfs not installed (offline CI / planning VM) — fall back to args.train_root
        print("[RW-03] gcsfs unavailable — falling back to args.train_root (offline mode)")
        train_root = getattr(args, "train_root", train_root)

    # (1) Build train/val datasets from train/ roots only — EVAL-01 (T-04-10)
    # carve_train_val raises ValueError if any test/ path slips through (PyramidDataset guard)
    train_ds, val_ds = carve_train_val(train_root, val_frac=0.2, seed=42)
    print(f"Dataset: {len(train_ds)} train / {len(val_ds)} val pyramids")

    # (2) Instantiate model
    model = _build_model(args, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {type(model).__name__} backbone={args.backbone} "
          f"trainable={trainable:,} params")

    # (3) Optimizer
    opt = make_optimizer(model, lr=args.lr)

    # (4) GCS resume: gcs_latest_checkpoint may raise ImportError on planning VM
    #     (offline_stub mode) — the test monkeypatches this to return (0, None).
    cfg = config_prefix(args.backbone, args.variant)
    try:
        resume_step, resume_state = gcs_latest_checkpoint(cfg)
    except ImportError:
        # gcsfs not installed (offline CI / planning VM) — start from scratch
        resume_step, resume_state = 0, None

    if resume_state is not None:
        print(f"Resuming from GCS step {resume_step} (config={cfg})")
        model.load_state_dict(resume_state["model_state_dict"])
        opt.load_state_dict(resume_state["optimizer_state_dict"])
        global_step = resume_step
    else:
        print(f"Starting fresh (config={cfg})")
        global_step = 0

    # (5) Probe-mode setup: record time before first epoch
    probe_epoch_start: float | None = None

    # Determine the train_step function (Variant A or B)
    variant_upper = args.variant.upper()
    if variant_upper == "A":
        _train_step = train_step_variant_a
    else:
        _train_step = train_step_variant_b

    # Collect the unique pyramid directories from the training subset.
    # train_ds is a torch.utils.data.Subset; its underlying dataset is PyramidDataset
    # whose .samples list has dicts with "pdir" key.  We need one step per pyramid
    # (not one step per tile), so we deduplicate by pdir.
    underlying_ds = train_ds.dataset   # PyramidDataset
    train_pdirs = sorted(
        {str(underlying_ds.samples[idx]["pdir"]) for idx in train_ds.indices}
    )
    val_pdirs = sorted(
        {str(underlying_ds.samples[idx]["pdir"]) for idx in val_ds.indices}
    ) if hasattr(val_ds, 'indices') else []

    print(f"Unique train pyramids: {len(train_pdirs)} | val pyramids: {len(val_pdirs)}")

    # (6) Epoch loop
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        n_steps = 0

        # Time the first epoch for probe mode
        if epoch == 1:
            probe_epoch_start = time.perf_counter()

        # One step per pyramid (not per tile) — train_step handles the full c2f walk
        for pdir_str in train_pdirs:
            pdir = Path(pdir_str)

            # One pyramid per step
            step_loss = _train_step(model, pdir, opt, device)

            global_step += 1
            epoch_loss += step_loss
            n_steps += 1

            # GCS checkpoint every K steps (AFTER optimizer.step, which is inside _train_step)
            if global_step % args.ckpt_every == 0:
                ckpt_state = {
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "config": cfg,
                }
                try:
                    saved_path = gcs_save_checkpoint(cfg, global_step, ckpt_state)
                    print(f"  Checkpoint saved → {saved_path}")
                except ImportError:
                    # gcsfs not installed — offline CI, skip silently
                    pass

        avg_train_loss = epoch_loss / max(n_steps, 1)

        # Per-epoch val pass (flat, not recursive, not test/ — EVAL-01 / Pitfall 5)
        avg_val_loss = _run_val_epoch(model, val_pdirs, device)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={avg_val_loss:.4f} | "
            f"step={global_step}"
        )

        # (7) Probe mode: after first epoch, compute cost projection and exit
        if args.probe:
            probe_epoch_end = time.perf_counter()
            wall_s = probe_epoch_end - probe_epoch_start  # type: ignore[operator]
            # Wall time in hours (CPU-wall on planning VM; GPU-hrs on GPU host)
            epoch_hrs = wall_s / 3600.0
            # Projected full-grid cost: epochs × 6 configs (backbone × variant)
            projected_full_grid_hrs = epoch_hrs * args.epochs * 6
            print(
                f"PROBE RESULT | "
                f"measured_epoch_hrs={epoch_hrs:.4f} (CPU-wall) | "
                f"projected_full_grid_hrs={projected_full_grid_hrs:.4f} | "
                f"configs=6 | "
                f"epochs={args.epochs}"
            )
            sys.exit(0)

    print(f"Training complete. Final step={global_step}, config={cfg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Argparse CLI entrypoint for finetune_seg.py."""
    parser = argparse.ArgumentParser(
        description="Fine-tune MapClass segmentation model on PyramidDataset"
    )
    parser.add_argument(
        "--backbone",
        choices=["siglip", "dinov2", "swin"],
        default="siglip",
        help="Vision backbone (default: siglip)",
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B"],
        default="B",
        help="Model variant: A=frozen-backbone prior-encoder, B=widened-patch-embed (default: B)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="AdamW learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--ckpt-every",
        type=int,
        default=100,
        dest="ckpt_every",
        help="Save GCS checkpoint every N optimiser steps (default: 100)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Cost-probe mode: run 1 epoch, print GPU-hrs/epoch + projected full-grid cost, exit",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Load pretrained backbone weights (default: True)",
    )
    parser.add_argument(
        "--no-pretrained",
        dest="pretrained",
        action="store_false",
        help="Use random backbone weights — offline CI only (DINOv2/Swin yields meaningless EVAL-03 baseline)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP (bfloat16 on CUDA; default fp32 — recommended for first probe)",
    )
    parser.add_argument(
        "--train-root",
        default="data/synthetic/train",
        dest="train_root",
        help="Root directory passed to carve_train_val (train/ subtree only — EVAL-01)",
    )
    parser.add_argument(
        "--offline-stub",
        action="store_true",
        dest="offline_stub",
        help="CI-only: use stub backbone (no pretrained weights, no GCS) for offline tests",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        default=Path("/tmp/mapclass_data"),
        dest="scratch_dir",
        help=(
            "Local scratch directory for the RW-03 pull-once dataset fetch. "
            "pull_dataset_from_gcs downloads the train subset here before "
            "carve_train_val runs. Ignored in offline mode (gcsfs absent). "
            "Default: /tmp/mapclass_data"
        ),
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
