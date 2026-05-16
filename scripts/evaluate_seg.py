"""
Joint per-pixel NLL evaluation harness for the segmentation model (EVAL-02, EVAL-03).

Reuses the Phase-3 recursive c2f walk (``recursive_predict`` /
``recursive_predict_variant_b``) unchanged — this module does NOT re-implement
the pyramid walk.

CLI usage:
    python scripts/evaluate_seg.py \\
        --backbone siglip \\
        --variant A \\
        --split-json data/synthetic/split.json \\
        --data-root data/synthetic/ \\
        --metrics-dir metrics/ \\
        --checkpoint path/to/checkpoint.pt

Security:
    T-04-07  ``load_test_pyramid_dirs`` resolves each path and asserts
             ``data_root`` is an ancestor — rejects ``../`` traversal in
             split.json map IDs.
    T-04-08  This module only constructs ``data_root / "test" / ...`` paths and
             reads ``split.json["test"]`` — never builds a ``train/`` path.
    T-04-09  ``evaluate_joint_nll`` clamps softmax probs to 1e-12 before log
             to avoid -inf NLL from long-tail underflow.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# sys.path bootstrap: add the scripts/ directory so that seg.* imports work
# whether the script is run from the repo root or from scripts/.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from seg.recursive import recursive_predict, recursive_predict_variant_b  # noqa: E402
from seg.model import SegModelVariantA, SegModelVariantB                  # noqa: E402


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pil_to_numpy(img: Image.Image) -> np.ndarray:
    """Convert PIL image to numpy array (mirrors seg/dataset.py idiom)."""
    return np.array(img)


def _load_label(label_path: Path) -> torch.Tensor:
    """
    Load a grayscale label PNG and return a (H, W) int64 tensor.

    Mirrors ``seg/dataset.py`` lines 171-176 — PIL L-mode → numpy → long.
    """
    img = Image.open(label_path)
    arr = _pil_to_numpy(img)
    return torch.from_numpy(arr).long()


# ---------------------------------------------------------------------------
# load_test_pyramid_dirs
# ---------------------------------------------------------------------------

def load_test_pyramid_dirs(split_json: Path, data_root: Path) -> List[Path]:
    """
    Enumerate held-out test pyramid directories from ``split.json``.

    Reads only ``split.json["test"]`` — never ``["train"]`` (EVAL-01 / T-04-08).
    For each map ID, resolves ``data_root / "test" / map_id`` and asserts the
    resolved path is a strict subpath of ``data_root`` before use (T-04-07
    path-traversal guard).

    Parameters
    ----------
    split_json:
        Path to the split.json file (must contain a "test" key).
    data_root:
        Root directory of the data tree.  All test pyramid dirs must live
        inside ``data_root / "test"``.

    Returns
    -------
    List[Path]
        Sorted list of pyramid directories (each containing ``pyramid.json``)
        under ``data_root / "test"``.  Only existing directories are returned.

    Raises
    ------
    ValueError
        If a map ID produces a resolved path outside ``data_root`` (path
        traversal attempt, e.g. map ID ``"../../etc"``.
    """
    split = json.loads(split_json.read_text())
    test_ids: List[str] = split["test"]
    data_root_resolved = data_root.resolve()
    test_root_resolved = (data_root / "test").resolve()

    dirs: List[Path] = []
    for map_id in test_ids:
        # Construct candidate path — always under test/, never train/
        candidate = (data_root / "test" / map_id).resolve()

        # T-04-07: assert the resolved path is strictly under data_root/test/.
        # Checking only data_root would allow map IDs like "../train/secret"
        # (which traverse from test/ into train/ but remain under data_root).
        # Checking test_root ensures no map ID can escape the test/ subtree.
        if test_root_resolved not in candidate.parents and candidate != test_root_resolved:
            raise ValueError(
                f"Map ID {map_id!r} resolves to {candidate} which is outside "
                f"data_root/test/ ({test_root_resolved}) — path traversal rejected (T-04-07)."
            )

        if not candidate.exists():
            continue

        # Collect all pyramid subdirectories (each containing pyramid.json).
        # The tiling producer creates pyramids at <map_id>/pyramids/<py_r*/c*>/,
        # so we recursively search for any subdirectory containing pyramid.json
        # rather than assuming a fixed nesting depth.
        if candidate.is_dir():
            for pjson in sorted(candidate.rglob("pyramid.json")):
                if not pjson.is_file():
                    continue
                pdir = pjson.parent.resolve()
                # Re-apply traversal guard for each discovered pyramid dir:
                # it must be under data_root/test/ (not merely data_root).
                if test_root_resolved in pdir.parents or pdir == test_root_resolved:
                    dirs.append(pdir)

    return dirs


# ---------------------------------------------------------------------------
# evaluate_joint_nll
# ---------------------------------------------------------------------------

def evaluate_joint_nll(
    model: torch.nn.Module,
    test_pyramid_dirs: List[Path],
    variant: str = "A",
    device: str = "cpu",
) -> float:
    """
    Compute mean joint per-pixel NLL over the test set (EVAL-02).

    Formula (per pixel):
        NLL = -(log p_lc[true_class] + log p_topo[true_class])

    Averaged over all valid pixels across all 224-level leaf tiles in all
    test pyramid directories.

    **Log probability computation note (T-04-09 / RESEARCH Pitfall):**
    ``recursive_predict`` returns probabilities that are ALREADY softmaxed
    (valid distributions in (0, 1]).  Applying ``torch.log_softmax`` to them
    would double-normalise and produce incorrect values.  Instead, we clamp
    to a small epsilon floor (1e-12) and apply ``torch.log`` directly.  This
    is numerically safe because the inputs are valid softmax outputs — the
    anti-pattern to avoid is ``log(softmax(raw_logits))`` on raw un-normalised
    logits, where ``softmax`` loses precision before the ``log``.  Here the
    probabilities are already stable softmax outputs, so ``log(clamp(prob))``
    is the correct and stable form.

    Parameters
    ----------
    model:
        A ``SegModelVariantA`` or ``SegModelVariantB`` in eval mode.
    test_pyramid_dirs:
        List of pyramid directories (each containing ``pyramid.json``).
        Must come from ``load_test_pyramid_dirs`` (test/ only, EVAL-01).
    variant:
        ``"A"`` → ``recursive_predict``; ``"B"`` → ``recursive_predict_variant_b``.
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).

    Returns
    -------
    float
        Mean joint NLL over all valid pixels.  Returns ``float("nan")`` if
        there are no valid pixels (empty test set).
    """
    model.eval()
    total_nll = 0.0
    total_pixels = 0

    predict_fn = recursive_predict if variant == "A" else recursive_predict_variant_b

    with torch.no_grad():
        for pdir in test_pyramid_dirs:
            pdir = Path(pdir)
            prob_cache = predict_fn(model, pdir)
            manifest = json.loads((pdir / "pyramid.json").read_text())

            # Filter to 224-level leaf tiles only (RESEARCH Pitfall 7 guard)
            leaf_tiles = [t for t in manifest["tiles"] if t["size"] == 224]

            for tile in leaf_tiles:
                tile_id = tile["id"]
                prob12 = prob_cache[tile_id].to(device)  # (1, 12, S, S)
                lc_prob = prob12[:, :9]    # (1, 9, S, S) — already softmaxed
                topo_prob = prob12[:, 9:]  # (1, 3, S, S) — already softmaxed

                # Load ground-truth label images
                lc_gt = _load_label(pdir / tile["land_cover"]).to(device)    # (S, S)
                topo_gt = _load_label(pdir / tile["topography"]).to(device)  # (S, S)

                # Numerically stable log probabilities.
                # clamping to 1e-12 prevents log(0) = -inf when a long-tail
                # class has underflowed to 0.  The inputs are valid softmax
                # outputs (already normalised), so clamp+log is correct here.
                log_lc = torch.log(lc_prob.clamp_min(1e-12))     # (1, 9, S, S)
                log_topo = torch.log(topo_prob.clamp_min(1e-12))  # (1, 3, S, S)

                # Valid mask: exclude out-of-range labels (T-04-09 intent)
                valid = (lc_gt < 9) & (topo_gt < 3)

                # Gather log p at the true class — result is (S, S).
                # Clamp gather indices to valid range BEFORE gather so that
                # out-of-range label values (e.g. 9 for LC, 3 for topo) do not
                # cause an index-out-of-bounds RuntimeError.  Their contribution
                # is masked out via the `valid` mask and never added to the sum.
                lc_gt_safe = lc_gt.clamp(0, 8)    # [0, num_lc_classes - 1]
                topo_gt_safe = topo_gt.clamp(0, 2)  # [0, num_topo_classes - 1]
                lc_nll = -log_lc[0].gather(0, lc_gt_safe.unsqueeze(0)).squeeze(0)
                topo_nll = -log_topo[0].gather(0, topo_gt_safe.unsqueeze(0)).squeeze(0)
                joint_nll = lc_nll + topo_nll  # (S, S)

                total_nll += joint_nll[valid].sum().item()
                total_pixels += valid.sum().item()

    return total_nll / total_pixels if total_pixels > 0 else float("nan")


# ---------------------------------------------------------------------------
# write_nll_metrics
# ---------------------------------------------------------------------------

def write_nll_metrics(
    config_name: str,
    nll: float,
    metrics_dir: Path,
    variant: str = "A",
    backbone: str = "siglip",
) -> Path:
    """
    Write per-config NLL metrics to a JSON file.

    Parameters
    ----------
    config_name:
        Identifier for the configuration (e.g. ``"siglip-A"``).
    nll:
        The computed joint NLL value.
    metrics_dir:
        Directory to write the JSON file into (created if absent).
    variant:
        Variant string (``"A"`` or ``"B"``).
    backbone:
        Backbone name (``"siglip"``, ``"dinov2"``, or ``"swin"``).

    Returns
    -------
    Path
        Path to the written metrics JSON file.
    """
    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out = metrics_dir / f"{config_name}-nll.json"
    out.write_text(
        json.dumps(
            {
                "config": config_name,
                "joint_nll": nll,
                "variant": variant,
                "backbone": backbone,
                "eval_date": datetime.date.today().isoformat(),
            },
            indent=2,
        )
    )
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for running the NLL evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Evaluate joint per-pixel NLL of a segmentation model on the test set.",
    )
    parser.add_argument(
        "--backbone",
        choices=["siglip", "dinov2", "swin"],
        default="siglip",
        help="Backbone name (default: siglip)",
    )
    parser.add_argument(
        "--variant",
        choices=["A", "B"],
        default="A",
        help="Model variant A or B (default: A)",
    )
    parser.add_argument(
        "--split-json",
        type=Path,
        required=True,
        help="Path to split.json (must have a 'test' key with map IDs)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root of the data tree; test pyramids are under <data-root>/test/",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("metrics"),
        help="Directory to write <config>-nll.json (default: metrics/)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to a local .pt checkpoint file, or a gs:// URI for a GCS checkpoint "
            "(gs:// URI support requires seg.gcs_checkpoint)"
        ),
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    if args.variant == "A":
        model = SegModelVariantA(backbone_name=args.backbone)
    else:
        model = SegModelVariantB(backbone_name=args.backbone)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    checkpoint_str = str(args.checkpoint)
    if checkpoint_str.startswith("gs://"):
        # GCS checkpoint: delegate to seg.gcs_checkpoint if available
        try:
            from seg import gcs_checkpoint as _gcs
            state_dict = _gcs.gcs_latest_checkpoint(checkpoint_str)
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "GCS checkpoint loading requires seg.gcs_checkpoint "
                f"(not yet available): {exc}"
            ) from exc
    else:
        state_dict = torch.load(checkpoint_str, map_location="cpu")

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Enumerate test pyramid dirs
    # ------------------------------------------------------------------
    test_dirs = load_test_pyramid_dirs(args.split_json, args.data_root)
    if not test_dirs:
        print("Warning: no test pyramid directories found — check --split-json and --data-root.")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    nll = evaluate_joint_nll(model, test_dirs, variant=args.variant)
    print(f"Joint per-pixel NLL ({args.backbone}-{args.variant}): {nll:.6f}")

    # ------------------------------------------------------------------
    # Write metrics
    # ------------------------------------------------------------------
    config_name = f"{args.backbone}-{args.variant}"
    out_path = write_nll_metrics(
        config_name=config_name,
        nll=nll,
        metrics_dir=args.metrics_dir,
        variant=args.variant,
        backbone=args.backbone,
    )
    print(f"Metrics written to: {out_path}")


if __name__ == "__main__":
    main()
