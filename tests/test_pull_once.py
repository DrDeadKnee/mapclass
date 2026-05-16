"""
Offline tests for the RW-03 pull-once wiring in finetune_seg.py and
evaluate_seg.py (plan 02-05, Task 2).

These tests are fully torch-free — they inspect the source of finetune_seg.py
and evaluate_seg.py as text to assert the structural contracts (arg presence,
import ordering, ImportError fallback) without requiring a GPU or the torch
package.  The GPU-host execution test is covered by the 02-HUMAN-UAT.md gate.

Coverage:
  - test_finetune_pull_once_before_resume
  - test_finetune_has_scratch_dir_arg
  - test_offline_importerror_fallback_finetune
  - test_evaluate_pull_once_before_load
  - test_evaluate_has_scratch_dir_arg
  - test_offline_importerror_fallback_evaluate
"""

from __future__ import annotations

import pathlib

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCRIPTS = pathlib.Path(__file__).parent.parent / "scripts"


def _src(filename: str) -> str:
    """Return the source text of a script under scripts/."""
    return (_SCRIPTS / filename).read_text()


# ---------------------------------------------------------------------------
# finetune_seg.py pull-once structural tests (RW-03)
# ---------------------------------------------------------------------------

class TestFinetunePullOnce:
    """
    finetune_seg.py must have --scratch-dir, pull_dataset_from_gcs + verify_pull
    before gcs_latest_checkpoint, and an ImportError fallback to args.train_root.
    """

    def test_finetune_pull_once_before_resume(self):
        """
        pull_dataset_from_gcs must appear BEFORE gcs_latest_checkpoint in
        finetune_seg.py (RW-03 — pull-once precedes checkpoint resume).
        """
        src = _src("finetune_seg.py")

        assert "pull_dataset_from_gcs" in src, (
            "finetune_seg.py must call pull_dataset_from_gcs (RW-03)"
        )
        assert "verify_pull" in src, (
            "finetune_seg.py must call verify_pull (RW-03)"
        )

        pull_pos = src.index("pull_dataset_from_gcs")
        resume_pos = src.index("gcs_latest_checkpoint")
        assert pull_pos < resume_pos, (
            f"pull_dataset_from_gcs (pos {pull_pos}) must appear before "
            f"gcs_latest_checkpoint (pos {resume_pos}) in finetune_seg.py (RW-03)"
        )

    def test_finetune_has_scratch_dir_arg(self):
        """finetune_seg.py must define a --scratch-dir argparse argument."""
        src = _src("finetune_seg.py")
        assert "--scratch-dir" in src, (
            "finetune_seg.py must define a --scratch-dir argparse argument (RW-03)"
        )

    def test_offline_importerror_fallback_finetune(self):
        """
        finetune_seg.py must wrap the gcs_io import in try/except ImportError
        with a fallback to args.train_root (offline CI guard).
        """
        src = _src("finetune_seg.py")
        assert "ImportError" in src, (
            "finetune_seg.py must catch ImportError from gcs_io import (offline fallback)"
        )
        assert "train_root" in src, (
            "finetune_seg.py must fall back to args.train_root when gcs_io unavailable"
        )


# ---------------------------------------------------------------------------
# evaluate_seg.py pull-once structural tests (RW-03)
# ---------------------------------------------------------------------------

class TestEvaluatePullOnce:
    """
    evaluate_seg.py must have --scratch-dir, pull_dataset_from_gcs + verify_pull
    before load_test_pyramid_dirs, and an ImportError fallback to local paths.
    """

    def test_evaluate_pull_once_before_load(self):
        """
        pull_dataset_from_gcs must appear BEFORE load_test_pyramid_dirs in
        evaluate_seg.py (RW-03 — pull-once precedes enumerate).
        """
        src = _src("evaluate_seg.py")

        assert "pull_dataset_from_gcs" in src, (
            "evaluate_seg.py must call pull_dataset_from_gcs (RW-03)"
        )
        assert "verify_pull" in src, (
            "evaluate_seg.py must call verify_pull (RW-03)"
        )

        pull_pos = src.index("pull_dataset_from_gcs")
        load_pos = src.index("load_test_pyramid_dirs")
        assert pull_pos < load_pos, (
            f"pull_dataset_from_gcs (pos {pull_pos}) must appear before "
            f"load_test_pyramid_dirs (pos {load_pos}) in evaluate_seg.py (RW-03)"
        )

    def test_evaluate_has_scratch_dir_arg(self):
        """evaluate_seg.py must define a --scratch-dir argparse argument."""
        src = _src("evaluate_seg.py")
        assert "--scratch-dir" in src, (
            "evaluate_seg.py must define a --scratch-dir argparse argument (RW-03)"
        )

    def test_offline_importerror_fallback_evaluate(self):
        """
        evaluate_seg.py must wrap the gcs_io import in try/except ImportError
        with a fallback to local split_json / data_root (offline CI guard).
        """
        src = _src("evaluate_seg.py")
        assert "ImportError" in src, (
            "evaluate_seg.py must catch ImportError from gcs_io import (offline fallback)"
        )
        # The fallback must reference at least one of: split_json or data_root
        assert ("split_json" in src or "data_root" in src), (
            "evaluate_seg.py must reference split_json or data_root in the fallback path"
        )
