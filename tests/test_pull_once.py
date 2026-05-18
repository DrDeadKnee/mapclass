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
        pull_dataset_from_gcs( call must appear BEFORE gcs_latest_checkpoint(
        in finetune_seg.py (RW-03 — pull-once precedes checkpoint resume).

        We look for the function call pattern (with opening parenthesis) to
        avoid matching imports or docstring mentions.
        """
        src = _src("finetune_seg.py")

        assert "pull_dataset_from_gcs" in src, (
            "finetune_seg.py must call pull_dataset_from_gcs (RW-03)"
        )
        assert "verify_pull" in src, (
            "finetune_seg.py must call verify_pull (RW-03)"
        )

        # Use the call-site pattern (with opening paren) to skip import lines
        pull_call = "pull_dataset_from_gcs("
        resume_call = "gcs_latest_checkpoint("
        assert pull_call in src, (
            "finetune_seg.py must have a pull_dataset_from_gcs(...) call site"
        )
        assert resume_call in src, (
            "finetune_seg.py must have a gcs_latest_checkpoint(...) call site"
        )
        pull_pos = src.index(pull_call)
        resume_pos = src.index(resume_call)
        assert pull_pos < resume_pos, (
            f"pull_dataset_from_gcs( (pos {pull_pos}) must appear before "
            f"gcs_latest_checkpoint( (pos {resume_pos}) in finetune_seg.py (RW-03)"
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
        pull_dataset_from_gcs( call must appear BEFORE load_test_pyramid_dirs(
        in evaluate_seg.py (RW-03 — pull-once precedes enumerate).
        """
        src = _src("evaluate_seg.py")

        assert "pull_dataset_from_gcs" in src, (
            "evaluate_seg.py must call pull_dataset_from_gcs (RW-03)"
        )
        assert "verify_pull" in src, (
            "evaluate_seg.py must call verify_pull (RW-03)"
        )

        # Use call-site patterns: look for the evaluate() body call signature
        # load_test_pyramid_dirs is defined as a function (line ~88) but called later.
        # We compare the last occurrence of pull_dataset_from_gcs( with
        # the last occurrence of load_test_pyramid_dirs( to find the call sites.
        pull_call = "pull_dataset_from_gcs("
        # load_test_pyramid_dirs is called as a function call inside evaluate().
        # The call-site search uses rfind to get the LAST occurrence (the call, not the def).
        assert pull_call in src, (
            "evaluate_seg.py must have a pull_dataset_from_gcs(...) call site"
        )
        assert "load_test_pyramid_dirs(" in src, (
            "evaluate_seg.py must have a load_test_pyramid_dirs(...) call site"
        )
        # rfind gives the LAST occurrence — the actual call in evaluate(), not the def
        pull_pos = src.rfind(pull_call)
        load_pos = src.rfind("load_test_pyramid_dirs(")
        assert pull_pos < load_pos, (
            f"pull_dataset_from_gcs( (pos {pull_pos}) must appear before "
            f"load_test_pyramid_dirs( (pos {load_pos}) in evaluate_seg.py (RW-03)"
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
