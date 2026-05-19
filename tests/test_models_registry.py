"""Unit checks for src/mapclass/models.py — the PLAIN per-model registry.

The registry is a plain ``dict[str, dict]`` (NO Protocol / dataclass /
conformance framework — explicitly Out of Scope per REQUIREMENTS.md). These
checks assert the structural contract and the ATTR-01 tensor-identity
invariant on a tiny synthetic stub (no real model / GPU / network):

  * exactly the 4 keys ``["siglip2","clip","vit_b16","paligemma"]``;
  * each value exposes callables ``load_fn``, ``build_inputs_fn``,
    ``target_fn`` and a ``patch_geom`` dict with keys
    ``patch_size``, ``img_dim``, ``has_cls``;
  * ``build_inputs_fn`` returns a dict whose ``pixel_values`` tensor has
    ``.requires_grad`` True and is the SAME Python object the model forward
    receives (identity-preservation invariant, asserted via an ``is`` check
    against a synthetic stub processor + model — NOT a Protocol framework).
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (
    _REPO_ROOT / "third_party" / "dynamicLRP" / "src",
    _REPO_ROOT / "src",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")

from mapclass import models  # noqa: E402

_MODELS_SRC = (_REPO_ROOT / "src" / "mapclass" / "models.py").read_text()

_EXPECTED_KEYS = {"siglip2", "clip", "vit_b16", "paligemma"}


def test_registry_has_exactly_the_four_keys():
    assert set(models.MODEL_REGISTRY.keys()) == _EXPECTED_KEYS


def test_each_entry_is_a_plain_dict_with_the_contract():
    for name, spec in models.MODEL_REGISTRY.items():
        assert isinstance(spec, dict), f"{name}: registry value must be a plain dict"
        for cb in ("load_fn", "build_inputs_fn", "target_fn"):
            assert callable(spec[cb]), f"{name}: {cb} must be callable"
        geom = spec["patch_geom"]
        assert isinstance(geom, dict), f"{name}: patch_geom must be a plain dict"
        assert set(geom.keys()) == {"patch_size", "img_dim", "has_cls"}, (
            f"{name}: patch_geom keys"
        )
        assert isinstance(geom["patch_size"], int)
        assert isinstance(geom["img_dim"], int)
        assert isinstance(geom["has_cls"], bool)


def test_patch_geom_values_match_research_table():
    g = {k: v["patch_geom"] for k, v in models.MODEL_REGISTRY.items()}
    assert g["siglip2"] == {"patch_size": 14, "img_dim": 384, "has_cls": False}
    assert g["clip"] == {"patch_size": 14, "img_dim": 224, "has_cls": True}
    assert g["vit_b16"] == {"patch_size": 16, "img_dim": 224, "has_cls": True}
    assert g["paligemma"] == {"patch_size": 14, "img_dim": 224, "has_cls": False}


# --------------------------------------------------------------------------
# ATTR-01 tensor-identity invariant via a synthetic stub (no real model).
# --------------------------------------------------------------------------
class _StubProcessor:
    """Returns a BatchEncoding-like dict with a pixel_values tensor."""

    def __init__(self, with_text):
        self._with_text = with_text

    def __call__(self, *args, **kwargs):
        class _BE(dict):
            def to(self, *_a, **_k):
                return self

        be = _BE()
        be["pixel_values"] = torch.rand(1, 3, 8, 8)
        if self._with_text:
            be["input_ids"] = torch.zeros(1, 4, dtype=torch.long)
            be["attention_mask"] = torch.ones(1, 4, dtype=torch.long)
        return be


def test_build_inputs_preserves_pixel_tensor_identity_and_requires_grad():
    """The SAME pixel tensor object must carry requires_grad and be reused.

    We call each registry build_inputs_fn with a stub processor and assert the
    returned forward-kwargs dict's pixel_values is requires_grad and is one
    object (identity is verified again end-to-end by attribution.attribute).
    """
    from PIL import Image

    pil = Image.new("RGB", (10, 12))
    for name, spec in models.MODEL_REGISTRY.items():
        with_text = name in ("siglip2", "clip", "paligemma")
        proc = _StubProcessor(with_text=with_text)
        forward_inputs, img_tensor = spec["build_inputs_fn"](
            None, proc, pil, "a river", "cpu"
        )
        assert img_tensor.requires_grad, f"{name}: pixel tensor must requires_grad"
        assert forward_inputs["pixel_values"] is img_tensor, (
            f"{name}: the requires_grad pixel tensor must be the SAME object "
            f"placed in the forward kwargs (ATTR-01 identity)"
        )


def test_no_protocol_or_dataclass_scaffold_in_source():
    """KEEP IT SIMPLE — no adapter Protocol / PatchGeometry dataclass."""
    assert "Protocol" not in _MODELS_SRC
    assert "@dataclass" not in _MODELS_SRC
    assert "class PatchGeometry" not in _MODELS_SRC


def test_no_forbidden_pooled_target_in_source():
    """CLAUDE.md: pooled embeddings are query-independent — forbidden target."""
    assert "get_image_features" not in _MODELS_SRC


def test_registry_contains_marker():
    # must_haves artifact contract: models.py contains "MODEL_REGISTRY".
    assert "MODEL_REGISTRY" in _MODELS_SRC
