"""
Evaluate a GeoViLM checkpoint on a held-out split (Phase 1 success criterion 3).

Usage:
    python -m mapclass.eval --split heldout --checkpoint models/geovilm_phase1_mock.pt \\
        --config mapclass/configs/v0.yaml --report eval_report.json

Computes per-pixel NLL = -(log p_land_cover + log p_topography) on the test split,
broken down per source x per class. Writes eval_report.json with the full breakdown
(CONTEXT.md "Eval report format"). Phase 1 numbers will be near log(num_classes)
because the backbone is Mock — the *pipeline* is the deliverable, not the score.

Eval-time overlap assertion (PITFALL 5 prevention #3): refuses to run if the test
split sample-IDs intersect the train split (raises SplitsContaminationError).

The optional `calibration` field in the report is reserved as None for Phase 2+
(TS-6 reliability diagram + ECE land in Phase 2).
"""

# Block 1: stdlib
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

# Block 2: third-party
import torch
import yaml
from torch.utils.data import DataLoader

# Block 3: first-party
from mapclass.data.dataset import MapClassDataset
from mapclass.data.splits import SplitsContaminationError, load_splits
from mapclass.data.taxonomy import LANDCOVER_CLASSES, TOPO_CLASSES
from mapclass.infer import load_model


# ---------------------------------------------------------------------------
# Splits-contamination assertion (PITFALL 5 prevention #3)
# ---------------------------------------------------------------------------
def _assert_no_split_leakage(splits: dict) -> None:
    """Raise SplitsContaminationError if test ∩ train != ∅."""
    train = set(splits["by_split"]["train"])
    test = set(splits["by_split"]["test"])
    leak = train & test
    if leak:
        raise SplitsContaminationError(
            f"PITFALL 5 violation: {len(leak)} sample IDs appear in both train and test splits. "
            f"Examples: {sorted(leak)[:5]}"
        )


# ---------------------------------------------------------------------------
# Per-class NLL accumulator
# ---------------------------------------------------------------------------
def _per_class_into(accum: dict[int, list[float]], probs: torch.Tensor,
                    labels: torch.Tensor, n_cls: int, eps: float = 1e-6) -> None:
    """probs: (C, H, W) softmax. labels: (H, W) int64; ignore_index=255.

    Accumulates per-class (sum_nll, count) into accum[class_idx]. Skips pixels
    where label == 255 (WATER_TOPO/NODATA). eps clamps log(0) (T-02-03).
    """
    valid = (labels != 255)
    if not valid.any():
        return
    flat_labels = labels[valid].long()
    valid_probs = probs[:, valid]
    picked = valid_probs.gather(0, flat_labels.unsqueeze(0)).squeeze(0)
    nlls = -torch.log(picked.clamp_min(eps))
    for c in range(n_cls):
        mask_c = (flat_labels == c)
        n = int(mask_c.sum().item())
        if n > 0:
            accum[c][0] += float(nlls[mask_c].sum().item())
            accum[c][1] += n


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------
def _build_report(by_source: dict, metadata: dict, args, splits_path: Path) -> dict:
    """Assemble the eval_report.json structure per CONTEXT.md `## Claude's Discretion`."""
    sources_out = {}
    grand_total_nll = 0.0
    grand_total_n = 0
    for src, kinds in by_source.items():
        per_cls_lc = {LANDCOVER_CLASSES[c]: (kinds["land_cover"][c][0] / max(kinds["land_cover"][c][1], 1))
                      for c in kinds["land_cover"] if kinds["land_cover"][c][1] > 0}
        per_cls_topo = {TOPO_CLASSES[c]: (kinds["topography"][c][0] / max(kinds["topography"][c][1], 1))
                        for c in kinds["topography"] if kinds["topography"][c][1] > 0}
        src_total_nll = (sum(kinds["land_cover"][c][0] for c in kinds["land_cover"])
                         + sum(kinds["topography"][c][0] for c in kinds["topography"]))
        src_total_n = (sum(kinds["land_cover"][c][1] for c in kinds["land_cover"])
                       + sum(kinds["topography"][c][1] for c in kinds["topography"]))
        sources_out[src] = {
            "land_cover_nll_per_class": per_cls_lc,
            "topography_nll_per_class": per_cls_topo,
            "mean_nll": (src_total_nll / src_total_n) if src_total_n > 0 else float("nan"),
            "n_pixels": src_total_n,
        }
        grand_total_nll += src_total_nll
        grand_total_n += src_total_n

    return {
        "model_version": metadata.get("model_version"),
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "splits_json_path": str(splits_path),
        "by_source": sources_out,
        "overall_mean_nll": (grand_total_nll / grand_total_n) if grand_total_n > 0 else float("nan"),
        "calibration": None,                                       # Phase 2+ extension (TS-6)
        "notes": "Phase 1 / Mock backbone — numbers near log(num_classes) expected (D-01).",
    }


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a GeoViLM checkpoint on a held-out split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="Same YAML config used at training time (paths to splits.json, root).")
    parser.add_argument("--split", choices=["train", "val", "test", "heldout"], default="heldout",
                        help="heldout = test (PITFALL 5 - committed deterministic split).")
    parser.add_argument("--report", type=Path, default=Path("eval_report.json"))
    args = parser.parse_args()

    print("=== Eval: Phase 1 (Mock backbone) ===")
    cfg = yaml.safe_load(args.config.read_text())
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config:     {args.config}")
    print(f"  Split:      {args.split}")
    print(f"  Report out: {args.report}")

    # Load splits + assert no leakage (PITFALL 5 prevention #3).
    splits_path = Path(cfg["data"]["splits_path"])
    splits = load_splits(splits_path)
    _assert_no_split_leakage(splits)

    split_key = "test" if args.split == "heldout" else args.split
    eval_ids = splits["by_split"][split_key]
    print(f"  evaluating {len(eval_ids)} samples on split '{split_key}'")

    root = Path(cfg["data"]["root"])
    eval_ds = MapClassDataset(eval_ids, root=root, validate=True)
    loader = DataLoader(eval_ds, batch_size=1, shuffle=False)

    predictor = load_model(args.checkpoint, device="auto")
    model = predictor.model
    model.eval()
    metadata = predictor.metadata

    # Accumulators: by_source[<src>][("land_cover"|"topography")][cls_idx] -> [sum_nll, count]
    by_source: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(
        lambda: {"land_cover": defaultdict(lambda: [0.0, 0]),
                 "topography": defaultdict(lambda: [0.0, 0])}
    )

    print(f"=== Forward passes ({len(eval_ids)} samples) ===")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            src = batch["source_subtype"][0]                                # DataLoader batches str -> list
            image = model.backbone.preprocess(batch["image"].to(predictor.device))
            out = model(image)
            lc_probs = torch.softmax(out["land_cover_logits"][0], dim=0)    # (9, H, W)
            topo_probs = torch.softmax(out["topography_logits"][0], dim=0)  # (3, H, W)
            lc_lbl = batch["land_cover"][0].to(predictor.device)            # (H, W)
            topo_lbl = batch["topography"][0].to(predictor.device)          # (H, W)
            _per_class_into(by_source[src]["land_cover"], lc_probs, lc_lbl, n_cls=9)
            _per_class_into(by_source[src]["topography"], topo_probs, topo_lbl, n_cls=3)
            if step % 10 == 0:
                print(f"  step {step}: source={src}, lc_shape={tuple(lc_probs.shape)}")

    # Reduce to mean NLL per class + per source.
    report = _build_report(by_source, metadata, args, splits_path)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)                                     # Pattern H
    print(f"=== Eval complete ===")
    print(f"  overall_mean_nll: {report['overall_mean_nll']:.4f}")
    print(f"  report written:   {args.report}")


if __name__ == "__main__":
    main()
