"""Unit checks for src/mapclass/attribution.py.

A synthetic differentiable forward-graph fixture (no SigLIP-2, no GPU, no
network) exercises the contracts that are independently assertable:

  * ``attribute`` returns relevance shaped EXACTLY like the input pixel tensor
    (Assumption A3 — input-shaped relevance read by tensor identity).
  * The 0-dim-scalar vs ≥1-d target fallback path exists and is taken: a fake
    ``LRPEngine`` that rejects a 0-dim target forces the ``slice_2d`` fallback,
    and ``AttributionResult.target_form`` records which form worked
    (Empirical Risk 1 / Assumption A5).

Static source assertions (the live SigLIP-2 path is exercised by the
notebook + the human-verify gate, not here):

  * the target is literally ``output.logits_per_image[0, 0]``;
  * ``get_image_features`` / image-embedding-norm targets appear NOWHERE;
  * the SAME ``img_tensor`` object flows into the forward and
    ``params_to_interpret`` with no ``.clone()`` / ``.detach()`` / re-``.to()``
    between;
  * ``torch.cuda.empty_cache`` is called and ``max_memory_allocated`` is
    captured;
  * the Fallback Ladder is documented in the module docstring.
"""

import re
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

from mapclass import attribution  # noqa: E402

_ATTR_SRC = (_REPO_ROOT / "src" / "mapclass" / "attribution.py").read_text()


# --------------------------------------------------------------------------
# Synthetic forward-graph fixtures (no SigLIP-2 / GPU / network).
# --------------------------------------------------------------------------
class _FakeOutput:
    def __init__(self, logits_per_image):
        self.logits_per_image = logits_per_image


class _FakeModel:
    """A tiny differentiable image→similarity graph in the SAME scope."""

    def __call__(self, pixel_values, input_ids=None, attention_mask=None):
        # A real grad path from the input pixels to a (1,1) similarity tensor.
        sim = pixel_values.mean() * 3.0 + pixel_values.sum() * 1e-6
        return _FakeOutput(sim.reshape(1, 1))


def _make_engine_class(reject_0d: bool):
    """Build a fake LRPEngine; optionally reject a 0-dim target (forces
    the documented ≥1-d fallback)."""

    class _FakeEngine:
        def __init__(self, **kwargs):
            self.params_to_interpret = None

        def run(self, target):
            if reject_0d and target.dim() == 0:
                raise RuntimeError("fake engine rejects 0-dim scalar target")
            # Relevance is read by identity at the input → SAME shape.
            param = self.params_to_interpret[0]
            relevance = torch.ones_like(param) * float(target.reshape(-1)[0])
            return [], [relevance]

    return _FakeEngine


@pytest.fixture
def img_tensor():
    return torch.rand(1, 3, 384, 384, requires_grad=True)


# --------------------------------------------------------------------------
# Behavioral: input-shaped relevance + the target-form fallback path.
# --------------------------------------------------------------------------
def test_relevance_is_input_shaped(monkeypatch, img_tensor):
    monkeypatch.setattr(
        attribution, "LRPEngine", _make_engine_class(reject_0d=False)
    )
    ids = torch.zeros(1, 4, dtype=torch.long)
    res = attribution.attribute(_FakeModel(), img_tensor, ids, ids)

    assert tuple(res.relevance.shape) == tuple(img_tensor.shape) == (
        1,
        3,
        384,
        384,
    )
    # 0-dim scalar accepted first when the engine does not reject it.
    assert res.target_form == "scalar_0d"
    # peak_vram_bytes is None on CPU, an int (>=0) when CUDA is present
    # (this is a GPU VM — the synthetic fixture does not allocate so it may
    # legitimately be 0).
    if torch.cuda.is_available():
        assert isinstance(res.peak_vram_bytes, int)
        assert res.peak_vram_bytes >= 0
    else:
        assert res.peak_vram_bytes is None
    assert isinstance(res.similarity, float)


def test_target_form_fallback_when_0d_rejected(monkeypatch, img_tensor):
    """Empirical Risk 1 / A5: 0-dim rejected → falls back to the 2-D slice."""
    monkeypatch.setattr(
        attribution, "LRPEngine", _make_engine_class(reject_0d=True)
    )
    ids = torch.zeros(1, 4, dtype=torch.long)
    res = attribution.attribute(_FakeModel(), img_tensor, ids, ids)

    assert res.target_form == "slice_2d"
    assert tuple(res.relevance.shape) == tuple(img_tensor.shape)


# --------------------------------------------------------------------------
# Static source guarantees.
# --------------------------------------------------------------------------
def test_target_is_logits_per_image_00():
    assert "output.logits_per_image[0, 0]" in _ATTR_SRC


def test_no_forbidden_targets():
    assert "get_image_features" not in _ATTR_SRC
    # No image-embedding-norm target (e.g. image_embeds .norm()) as the target.
    assert not re.search(r"image_embeds\s*\.\s*norm", _ATTR_SRC)


def test_tensor_identity_preserved():
    """img_tensor flows unchanged into forward AND params_to_interpret.

    v1.1: the forward is now generic ``model(**forward_inputs)`` where
    ``forward_inputs["pixel_values"]`` IS the requires_grad ``img_tensor``
    object (built once in models.build_inputs_fn). The engine still resolves
    relevance by tensor identity via ``params_to_interpret = [img_tensor]``.
    """
    assert "params_to_interpret = [img_tensor]" in _ATTR_SRC
    assert "model(**forward_inputs)" in _ATTR_SRC
    # No clone/detach/re-.to() applied to img_tensor before the engine runs.
    assert "img_tensor.clone()" not in _ATTR_SRC
    assert "img_tensor.detach()" not in _ATTR_SRC
    assert "img_tensor.to(" not in _ATTR_SRC


def test_vram_capture_present():
    assert "empty_cache" in _ATTR_SRC
    assert "max_memory_allocated" in _ATTR_SRC


def test_fallback_ladder_documented():
    assert "Fallback Ladder" in attribution.__doc__
    assert "use_attn_lrp" in attribution.__doc__
    assert "captum" in attribution.__doc__.lower()
