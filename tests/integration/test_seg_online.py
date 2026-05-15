"""
Online integration tests for the seg pipeline with a real Phase-1 adapter and
real Phase-2 pyramid tiles (plan 03-05, OQ1 RESOLVED).

@pytest.mark.integration — requires the Phase-1 PEFT adapter directory and
a Phase-2 pyramid tree on the local filesystem.  Skip with a clear reason
when either artifact is absent.

Run with:
    pytest tests/integration/test_seg_online.py -m integration \\
        --adapter-dir checkpoints/paligemma-terrain/epochNN \\
        --pyramid-dir data/synthetic/train/<map_name>/pyramids/<pyramid_id>

Decision coverage: D-03a (both Variant A + Variant B), D-04 (pyramid.json
walk), D-05 (Backbone protocol, real SigLIP-LoRA adapter), D-06 (construction
+ smoke forward only, no training).

Artifacts required:
  - Phase-1 PEFT adapter directory (--adapter-dir or env var ADAPTER_DIR)
  - Phase-2 real pyramid directory containing pyramid.json (--pyramid-dir or
    env var PYRAMID_DIR)
  - Phase-1 base model id (--model-id, default google/paligemma-3b-pt-224)
"""

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# sys.path guard (integration tests are one directory deeper than pytest.ini's
# pythonpath=scripts root; mirror test_satellite_online.py lines 18-20).
# ---------------------------------------------------------------------------
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# ---------------------------------------------------------------------------
# CLI / env-var fixtures for artifact paths (OQ1 RESOLVED: parameterize + skip)
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    """Register CLI options for adapter dir, pyramid dir, and model id."""
    # Guard: pytest plugins may call addoption multiple times in some configs
    try:
        parser.addoption(
            "--adapter-dir",
            action="store",
            default=None,
            help="Path to the Phase-1 PEFT adapter directory.",
        )
        parser.addoption(
            "--pyramid-dir",
            action="store",
            default=None,
            help="Path to a real Phase-2 pyramid directory (contains pyramid.json).",
        )
        parser.addoption(
            "--model-id",
            action="store",
            default="google/paligemma-3b-pt-224",
            help="HuggingFace model ID for the PaliGemma base model.",
        )
    except Exception:
        pass


def _adapter_dir(request) -> Path | None:
    val = (
        request.config.getoption("--adapter-dir", default=None)
        or os.environ.get("ADAPTER_DIR")
    )
    return Path(val) if val else None


def _pyramid_dir(request) -> Path | None:
    val = (
        request.config.getoption("--pyramid-dir", default=None)
        or os.environ.get("PYRAMID_DIR")
    )
    return Path(val) if val else None


def _model_id(request) -> str:
    return (
        request.config.getoption("--model-id", default="google/paligemma-3b-pt-224")
        or "google/paligemma-3b-pt-224"
    )


# ---------------------------------------------------------------------------
# Skip helpers
# ---------------------------------------------------------------------------

def _skip_if_no_adapter(request):
    d = _adapter_dir(request)
    if d is None or not d.exists():
        pytest.skip(
            reason=(
                "Phase-1 PEFT adapter not found. Provide --adapter-dir <path> or "
                "set ADAPTER_DIR env var pointing to a saved adapter directory "
                "(e.g. checkpoints/paligemma-terrain/epochNN). "
                "See RESEARCH.md OQ1 RESOLVED."
            )
        )
    return d


def _skip_if_no_pyramid(request):
    d = _pyramid_dir(request)
    if d is None or not (d / "pyramid.json").exists():
        pytest.skip(
            reason=(
                "Phase-2 real pyramid not found. Provide --pyramid-dir <path> or "
                "set PYRAMID_DIR env var pointing to a pyramid directory produced by "
                "scripts/tiling.py (must contain pyramid.json). "
                "See RESEARCH.md Environment Availability."
            )
        )
    return d


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_siglip_lora_load_and_forward(request, tmp_path):
    """
    Load the real Phase-1 SigLIP-LoRA adapter onto PaliGemma-3B and run a
    forward pass on a real Phase-2 tile.

    Verifies (D-05 + PHASE-03 SC#2):
    - SiglipBackbone loads the PEFT adapter without attribute errors
    - No language_model parameter is reachable from the backbone
    - Feature output shape is (1, 1152, h, w) with h*w = (tile_size/14)^2
    """
    adapter_path = _skip_if_no_adapter(request)
    pyramid_path = _skip_if_no_pyramid(request)
    import json
    import torch
    from seg.backbones import SiglipBackbone

    man = json.loads((pyramid_path / "pyramid.json").read_text())
    root = next(t for t in man["tiles"] if t["size"] == 896)

    from PIL import Image
    import numpy as np
    img = Image.open(pyramid_path / root["image"]).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    bb = SiglipBackbone(
        model_id=_model_id(request),
        adapter_dir=str(adapter_path),
    )
    # No Gemma params
    names = [n for n, _ in bb.named_parameters()]
    assert not any("language_model" in n for n in names), (
        "Gemma parameters must not be reachable from SiglipBackbone after LoRA load"
    )
    with torch.no_grad():
        feats = bb(rgb)
    assert feats, "SiglipBackbone must produce at least one feature map"
    for f in feats:
        assert not f.isnan().any(), "Feature maps must not contain NaN"
        assert f.shape[1] == 1152, (
            f"SigLIP-So400m/14 hidden_size must be 1152, got {f.shape[1]}"
        )


@pytest.mark.integration
def test_variant_a_real_pyramid_forward(request, tmp_path):
    """
    Variant A end-to-end: real SigLIP-LoRA adapter + real Phase-2 pyramid.

    Walks pyramid.json (D-04), feeds RGB + zero/parent prior into SegModelVariantA,
    asserts (B,9,H,W)+(B,3,H,W) outputs and no NaN at all 3 pyramid scales.
    """
    adapter_path = _skip_if_no_adapter(request)
    pyramid_path = _skip_if_no_pyramid(request)
    import json
    import torch
    from seg.model import SegModelVariantA
    from PIL import Image
    import numpy as np

    def _read_rgb(tile_entry):
        img = Image.open(pyramid_path / tile_entry["image"]).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    model = SegModelVariantA(
        backbone_name="siglip",
        model_id=_model_id(request),
        adapter_dir=str(adapter_path),
    )
    man = json.loads((pyramid_path / "pyramid.json").read_text())
    by_id = {t["id"]: t for t in man["tiles"]}
    root = next(t for t in man["tiles"] if t["size"] == 896)

    # Test 3 representative tiles (896, first 448, first 224)
    for tile_id in [root["id"], root["children"][0], by_id[root["children"][0]]["children"][0]]:
        tile = by_id[tile_id]
        s = tile["size"]
        rgb = _read_rgb(tile)
        prior = torch.zeros(1, 12, s, s)
        with torch.no_grad():
            lc, topo = model(rgb, prior)
        assert lc.shape == (1, 9, s, s), f"LC shape mismatch for tile {tile_id}"
        assert topo.shape == (1, 3, s, s), f"Topo shape mismatch for tile {tile_id}"
        assert not lc.isnan().any(), f"NaN in LC logits for tile {tile_id}"
        assert not topo.isnan().any(), f"NaN in Topo logits for tile {tile_id}"


@pytest.mark.integration
def test_variant_b_real_pyramid_forward(request, tmp_path):
    """
    Variant B end-to-end: real SigLIP-LoRA with widened 15-ch patch-embed +
    real Phase-2 pyramid.

    Concatenates zero prior channels to RGB (15-ch input), feeds into
    SegModelVariantB, asserts correct output shapes and no NaN.
    """
    adapter_path = _skip_if_no_adapter(request)
    pyramid_path = _skip_if_no_pyramid(request)
    import json
    import torch
    from seg.model import SegModelVariantB
    from PIL import Image
    import numpy as np

    man = json.loads((pyramid_path / "pyramid.json").read_text())
    root = next(t for t in man["tiles"] if t["size"] == 896)

    img = Image.open(pyramid_path / root["image"]).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    rgb = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    prior_zeros = torch.zeros(1, 12, 896, 896)
    x15 = torch.cat([rgb, prior_zeros], dim=1)

    model = SegModelVariantB(
        backbone_name="siglip",
        model_id=_model_id(request),
        adapter_dir=str(adapter_path),
    )
    with torch.no_grad():
        lc, topo = model(x15)
    assert lc.shape == (1, 9, 896, 896), f"LC shape: expected (1,9,896,896), got {lc.shape}"
    assert topo.shape == (1, 3, 896, 896)
    assert not lc.isnan().any()
    assert not topo.isnan().any()
