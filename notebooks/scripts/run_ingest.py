#!/usr/bin/env python3
"""Thin runner: drive the full 1,544-entry Rumsey ingest and print a histogram.

This is the long-running, resumable step that gates Phase 1 (D-01/D-02). It is
idempotent — re-running it converges (all previously-ok ids report ``skipped``
and nothing new is uploaded).

Usage (on the GCP VM, with the pinned venv from Plan 01-01):

    .venv/bin/python notebooks/scripts/run_ingest.py
"""

from __future__ import annotations

import os
import sys

# Make ``src/`` importable when run as a plain script from the repo root.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from mapclass.manifest import load_manifest  # noqa: E402
from mapclass.ingest_images import run_full_ingest, status_histogram  # noqa: E402


def main() -> int:
    manifest = load_manifest()
    print(f"Loaded manifest: {len(manifest)} entries. Starting full ingest...")

    outcomes = run_full_ingest(manifest)

    hist = status_histogram(outcomes)
    print("\n=== Ingest status histogram ===")
    for status in sorted(hist):
        print(f"  {status:<18} {hist[status]}")
    covered = len(outcomes)
    print(f"  {'TOTAL ids covered':<18} {covered} / {len(manifest)}")

    if covered != len(manifest):
        print(
            "\nWARNING: outcome manifest does not cover every manifest id "
            "(expected for an interrupted run — re-run to converge)."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
