"""Unit tests for the ordered-index manifest reader (DATA-03, D-05, Pitfall A).

Self-contained: adds ``src/`` to sys.path so it runs whether or not Plan
01-01's pytest scaffold/conftest is present in this worktree.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mapclass.manifest import load_manifest, entry_by_index, count_down_from


def test_load_manifest_is_list_of_1544():
    m = load_manifest()
    assert isinstance(m, list)
    assert len(m) == 1544


def test_locked_slice_is_highest_list_index():
    # D-05: the slice is manifest[-1] (highest LIST index), positional.
    m = load_manifest()
    assert entry_by_index(m, -1)["id"] == "RUMSEY~8~1~344476~90112460"


def test_count_down_from_yields_descending_indices():
    m = load_manifest()
    pairs = list(count_down_from(m, 1543, 3))
    assert [i for i, _ in pairs] == [1543, 1542, 1541]
    assert pairs[0][1] is m[1543]
    assert pairs[2][1] is m[1541]


def test_count_down_from_clamps_at_zero():
    m = load_manifest()
    pairs = list(count_down_from(m, 1, 5))
    assert [i for i, _ in pairs] == [1, 0]


def test_module_documents_richness_non_monotonicity_and_no_sort():
    # Pitfall A: docstring must explicitly state richness_score is NOT
    # monotonic with index, and the module must not sort/filter on it.
    import mapclass.manifest as manifest_mod

    doc = manifest_mod.__doc__ or ""
    assert "richness_score" in doc
    assert "NOT monotonic" in doc

    src_path = manifest_mod.__file__
    with open(src_path, "r", encoding="utf-8") as fh:
        body_lines = fh.read().splitlines()
    # Strip the module docstring; assert no sort/filter on richness in code.
    code = "\n".join(body_lines)
    code_after_doc = code.split('"""', 2)[-1]
    assert ".sort(" not in code_after_doc
    assert "sorted(" not in code_after_doc
    assert "richness_score" not in code_after_doc
