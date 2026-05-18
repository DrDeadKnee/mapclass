"""Pytest path setup.

Prepends the vendored dynamicLRP source and the project ``src`` to ``sys.path``
so ``from lrp_engine import LRPEngine`` and ``import mapclass`` resolve in tests
WITHOUT installing either package (dynamicLRP has no packaging metadata; the
package is run from source per the package-of-modules architecture).
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_LRP_SRC = _REPO_ROOT / "third_party" / "dynamicLRP" / "src"
_PROJECT_SRC = _REPO_ROOT / "src"

for _p in (_VENDORED_LRP_SRC, _PROJECT_SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
