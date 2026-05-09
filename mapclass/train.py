"""
Train a GeoViLM model end-to-end (Phase 1: Mock backbone smoke run).

Usage:
    python -m mapclass.train --config mapclass/configs/v0.yaml [--seed 42] [--smoke]

Reads samples emitted by scripts/build_*.py, validates them via
mapclass.data.contract.assert_sample_valid, loads loss weights from
mapclass.data.loss_weights.LossWeights.load, and writes a safetensors checkpoint
with embedded metadata per Phase 1 success criterion 1:
    {model_version, dataset_manifest_sha, taxonomy_hash, training_seed,
     source_class_weights_hash, backbone, processor_identity}

Smoke-run mode (--smoke): 5-iteration mini-loop on the first 8 samples;
asserts loss decreases by >=0.5% (D-01 — proof the gradient path is real).
Phase 1 metadata invariants (PATTERNS.md cross-cutting gotcha #2): safetensors
metadata is dict[str, str]; every value is JSON-encoded.
"""

# Block 1: stdlib
import argparse
import hashlib
import json
import sys
from pathlib import Path

# Block 2: third-party
import torch
import yaml
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader

# Block 3: first-party
from mapclass.data.contract import assert_sample_valid
from mapclass.data.dataset import MapClassDataset
from mapclass.data.loss_weights import LossWeights
from mapclass.data.splits import build_splits, load_splits, write_splits
from mapclass.data.taxonomy import taxonomy_hash
from mapclass.model.geovilm import GeoViLM
from mapclass.model.mock_backbone import MockBackbone
from mapclass.model.seg_heads import LandCoverHead, TopographyHead
from mapclass.seeding import set_global_seed


# ---------------------------------------------------------------------------
# Config + seeding
# ---------------------------------------------------------------------------
def _load_config(path: Path) -> dict:
    """yaml.safe_load (PATTERNS.md gotcha — never yaml.load/unsafe_load)."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataset + splits
# ---------------------------------------------------------------------------
def _discover_sample_ids(root: Path, max_samples: int | None = None) -> list[str]:
    """Sorted list of sample subdirs under root. max_samples truncates."""
    ids = sorted(p.name for p in root.iterdir() if p.is_dir())
    if max_samples is not None:
        ids = ids[:max_samples]
    return ids


def _resolve_splits(splits_path: Path, sample_ids: list[str]) -> dict:
    """Load committed splits.json if it exists; otherwise generate + commit it."""
    if splits_path.is_file():
        return load_splits(splits_path)
    print(f"  splits.json not found at {splits_path} - generating from {len(sample_ids)} sample IDs")
    splits = build_splits(sample_ids)
    write_splits(splits, splits_path)
    return splits


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def _build_model(cfg: dict, seed: int) -> GeoViLM:
    """Phase 1: backbone='mock' is the only supported value."""
    if cfg["model"]["backbone"] != "mock":
        raise ValueError(
            f"Phase 1 only supports backbone=mock; got {cfg['model']['backbone']!r}. "
            "SmolVLM lands in Phase 2."
        )
    backbone = MockBackbone(training_seed=seed)
    n_feat = backbone.feature_channels["features"]
    return GeoViLM(backbone, LandCoverHead(in_channels=n_feat), TopographyHead(in_channels=n_feat))


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------
def _compute_loss(logits_lc: torch.Tensor, logits_topo: torch.Tensor,
                  labels_lc: torch.Tensor, labels_topo: torch.Tensor) -> torch.Tensor:
    """Per-pixel cross-entropy, averaged. Honors WATER_TOPO=255 sentinel via ignore_index=255.

    Phase 1 is unweighted (per-source loss-weight wiring is Phase 2 — CONTEXT.md
    Out-of-Scope item: "Per-source loss-weight WIRING through training — Phase 2").
    """
    ce_lc = nn.functional.cross_entropy(logits_lc, labels_lc, ignore_index=255)
    ce_topo = nn.functional.cross_entropy(logits_topo, labels_topo, ignore_index=255)
    return ce_lc + ce_topo


def _train_one_epoch(model, loader, optim, device, epoch_idx, n_epochs):
    """Returns mean loss for the epoch."""
    model.train()
    losses = []
    for step, batch in enumerate(loader):
        image = model.backbone.preprocess(batch["image"].to(device))
        out = model(image)
        loss = _compute_loss(out["land_cover_logits"], out["topography_logits"],
                             batch["land_cover"].to(device), batch["topography"].to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        if step % 5 == 0:
            print(f"  Epoch {epoch_idx}/{n_epochs} step {step}: loss={loss.item():.4f}")
    return sum(losses) / len(losses) if losses else float("nan")


def _smoke_run(model, dataset, optim, device) -> tuple[float, float]:
    """5 iterations on the first 8 samples. Returns (first_loss, final_loss)."""
    model.train()
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    losses = []
    iters = iter(loader)
    for it in range(5):
        try:
            batch = next(iters)
        except StopIteration:
            iters = iter(loader)
            batch = next(iters)
        image = model.backbone.preprocess(batch["image"].to(device))
        out = model(image)
        loss = _compute_loss(out["land_cover_logits"], out["topography_logits"],
                             batch["land_cover"].to(device), batch["topography"].to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        losses.append(loss.item())
        print(f"  smoke-iter {it+1}/5: loss={loss.item():.4f}")
    return losses[0], losses[-1]


# ---------------------------------------------------------------------------
# Checkpoint write
# ---------------------------------------------------------------------------
def _compute_dataset_manifest_sha(sample_ids: list[str]) -> str:
    """sha256 of newline-joined sorted sample IDs (Phase 1 dataset versioning anchor; CONCERNS.md 8b)."""
    payload = "\n".join(sorted(sample_ids)).encode()
    return hashlib.sha256(payload).hexdigest()


def _save_checkpoint(model: GeoViLM, out_path: Path, *, seed: int, dataset_manifest_sha: str,
                     source_class_weights_hash: str) -> None:
    """Embed metadata as dict[str, str]; every value JSON-encoded (PATTERNS.md gotcha #2)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_meta = {
        "model_version":             "0.1.0",
        "dataset_manifest_sha":      dataset_manifest_sha,
        "taxonomy_hash":             taxonomy_hash(),
        "training_seed":             seed,
        "source_class_weights_hash": source_class_weights_hash,
        "backbone":                  "mock",
        "processor_identity":        model.backbone.processor_identity,
    }
    metadata = {k: json.dumps(v) for k, v in raw_meta.items()}
    save_file(model.state_dict(), str(out_path), metadata=metadata)
    print(f"  checkpoint written: {out_path}")
    print(f"    metadata keys: {sorted(metadata.keys())}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a GeoViLM model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None,
                        help="Override config seed; default = config.seed (PITFALL 15)")
    parser.add_argument("--smoke", action="store_true",
                        help="5-iteration loss-decrease smoke run (D-01); writes checkpoint and exits.")
    args = parser.parse_args()

    print("=== Train: Phase 1 (Mock backbone) ===")
    cfg = _load_config(args.config)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    print(f"  Config: {args.config}")
    print(f"  Seed:   {seed}")
    print(f"  Smoke:  {args.smoke}")

    set_global_seed(seed)

    # Loss weights (PITFALL 3 — strict load at startup; hash embedded in checkpoint).
    lw = LossWeights.load(cfg["data"]["loss_weights_path"])
    print(f"  loss_weights hash: {lw.source_class_weights_hash[:12]}...")

    # Sample discovery + contract validation + splits.
    root = Path(cfg["data"]["root"])
    sample_ids = _discover_sample_ids(root, max_samples=cfg["data"].get("num_samples"))
    print(f"  discovered {len(sample_ids)} samples under {root}")
    splits = _resolve_splits(Path(cfg["data"]["splits_path"]), sample_ids)
    train_ids = splits["by_split"]["train"]
    print(f"  train split: {len(train_ids)} samples")

    dataset_manifest_sha = _compute_dataset_manifest_sha(sample_ids)
    train_ds = MapClassDataset(train_ids, root=root, validate=True)

    # Model + optimizer.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(cfg, seed=seed).to(device)
    # Auto-coerce lr to float: PyYAML 1.1 parses "1e-3" as a string (no dot in scientific
    # notation) — coerce here so the config can stay human-readable. (Rule 1 auto-fix.)
    lr = float(cfg["training"]["lr"])
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # SMOKE PATH: 5-iter loss-decrease assertion (D-01) -> write checkpoint -> exit.
    if args.smoke:
        print("=== Smoke run: 5 iterations on first 8 samples (D-01) ===")
        smoke_ds = MapClassDataset(train_ids[:8], root=root, validate=False)
        first_loss, final_loss = _smoke_run(model, smoke_ds, optim, device)
        print(f"  first_loss={first_loss:.4f}  final_loss={final_loss:.4f}")
        if not (final_loss < first_loss * 0.995):
            raise RuntimeError(
                f"D-01 violation: smoke-run loss did not decrease by >=0.5%; "
                f"first={first_loss:.4f}, final={final_loss:.4f}. "
                "Gradient path is broken - investigate before proceeding to Phase 2."
            )
        print("  D-01 ok: loss decreased >=0.5% over 5 iters")
        _save_checkpoint(model, Path(cfg["checkpoint"]["out_path"]),
                         seed=seed, dataset_manifest_sha=dataset_manifest_sha,
                         source_class_weights_hash=lw.source_class_weights_hash)
        return

    # FULL PATH: train_one_epoch x n_epochs -> checkpoint.
    n_epochs = cfg["training"]["n_epochs"]
    loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    print(f"=== Training: {n_epochs} epoch(s), batch_size={cfg['data']['batch_size']} ===")
    for ep in range(1, n_epochs + 1):
        mean = _train_one_epoch(model, loader, optim, device, epoch_idx=ep, n_epochs=n_epochs)
        print(f"  Epoch {ep}: mean_loss={mean:.4f}")

    _save_checkpoint(model, Path(cfg["checkpoint"]["out_path"]),
                     seed=seed, dataset_manifest_sha=dataset_manifest_sha,
                     source_class_weights_hash=lw.source_class_weights_hash)


if __name__ == "__main__":
    main()
