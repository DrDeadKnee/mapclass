#!/usr/bin/env python3
"""Thin runner: mirror the pinned SigLIP-2 weights HF snapshot -> GCS ``models/``.

Run-once and idempotent — re-running converges (2nd run uploads nothing; every
file reports ``skipped``). This is the companion to ``run_ingest.py``: both are
self-bootstrapping runners so the documented Phase-1 verification commands work
from a plain pinned venv without an editable install (the project runs
``mapclass`` from source via ``sys.path``, by design — no packaging metadata).

Usage (on the GCP VM, with the pinned venv from Plan 01-01):

    .venv/bin/python notebooks/scripts/run_mirror_model.py
"""

from __future__ import annotations

import os
import sys

# Make ``src/`` importable when run as a plain script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mapclass import config  # noqa: E402
from mapclass.mirror_model import mirror_model  # noqa: E402


def main() -> int:
    # mirror_model is generalized to (repo_id, gcs_dir, ...) in v1.1; this
    # SigLIP-2 runner pins the v1.0 repo/dir via config (non-gated, token=None).
    summary = mirror_model(
        config.MODEL_REPO_ID, config.MODEL_GCS_DIR, token=None
    )
    uploaded = sum(1 for v in summary.values() if v == "uploaded")
    skipped = sum(1 for v in summary.values() if v == "skipped")
    print(
        f"model mirror complete: {uploaded} uploaded, {skipped} skipped, "
        f"{len(summary)} files total"
    )
    # Idempotency signal: a converged re-run uploads nothing.
    if uploaded == 0 and len(summary) > 0:
        print("converged: nothing uploaded (idempotent re-run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
