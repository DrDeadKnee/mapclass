"""
Offline eval-harness test scaffold for Phase 4 (plan 04-01, filled in plan 04-03).

Guard imports from ``evaluate_seg`` which ships in plan 04-03.
This file fails LOUDLY (pytest.fail, not pytest.skip) if evaluate_seg is not
importable, honouring the Phase-4 Nyquist gate.

Body filled in plan 04-03; only the scaffold and loud-fail guard are set up here.
"""

import pytest

# ---------------------------------------------------------------------------
# Import guard — LOUD fail (never skip) so the Nyquist gate fires offline
# ---------------------------------------------------------------------------
_import_error: Exception | None = None
try:
    from evaluate_seg import evaluate_joint_nll, load_test_pyramid_dirs  # noqa: F401
except ImportError as _e:
    _import_error = _e


def _require_eval():
    if _import_error is not None:
        pytest.fail(
            f"evaluate_seg not yet implemented: {_import_error}",
            pytrace=False,
        )


# ---------------------------------------------------------------------------
# TestNLLFormula — NLL formula correctness (filled in plan 04-03)
# ---------------------------------------------------------------------------

class TestNLLFormula:
    """NLL = -(log p_lc[true] + log p_topo[true]) per pixel (plan 04-03)."""

    def test_nll_formula_placeholder(self):
        """Placeholder: calls _require_eval then skips (body filled in 04-03)."""
        _require_eval()
        pytest.skip("filled in 04-03")
