"""Run-once, idempotent SigLIP-2 weights mirror: HF snapshot → GCS ``models/``.

``huggingface_hub.snapshot_download`` (handles sharding/resume/revision) then a
recursive upload of every file into
``gs://{GCS_BUCKET}/{MODEL_GCS_DIR}/<relpath>``, skipping any blob that already
exists with a matching byte size. Re-runnable to convergence (2nd run uploads
nothing). One-time, trusted publisher, pinned repo id (threat boundary
huggingface.co → GCS models/).
"""

from __future__ import annotations

import os
from typing import Dict, List

from mapclass import config
from mapclass.ingest_images import get_bucket, verify_adc


def download_snapshot(local_dir: str | None = None) -> str:
    """Download the pinned SigLIP-2 checkpoint locally. Returns the path."""
    from huggingface_hub import snapshot_download

    target = local_dir or config.LOCAL_MODEL_CACHE_DIR
    os.makedirs(target, exist_ok=True)
    return snapshot_download(
        repo_id=config.MODEL_REPO_ID, local_dir=target
    )


def _iter_files(root: str):
    """Yield ``(absolute_path, relpath)`` for every file under ``root``."""
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            abspath = os.path.join(dirpath, name)
            relpath = os.path.relpath(abspath, root)
            yield abspath, relpath


def mirror_model(local_dir: str | None = None) -> Dict[str, str]:
    """Idempotently mirror the model snapshot to GCS.

    Returns a ``{relpath: status}`` map where status is ``uploaded`` or
    ``skipped`` (skipped == already present with the same size).
    """
    verify_adc()
    snapshot_dir = download_snapshot(local_dir)
    bucket = get_bucket()

    results: Dict[str, str] = {}
    for abspath, relpath in _iter_files(snapshot_dir):
        # Forward-slash key under the model dir (GCS keys are literal).
        key = f"{config.MODEL_GCS_DIR}/" + relpath.replace(os.sep, "/")
        blob = bucket.blob(key)
        local_size = os.path.getsize(abspath)

        if blob.exists():
            remote_size = blob.size
            if remote_size is None:
                blob.reload()
                remote_size = blob.size
            if remote_size == local_size:
                results[relpath] = "skipped"
                continue

        blob.upload_from_filename(abspath)
        results[relpath] = "uploaded"

    return results


def list_mirrored_keys() -> List[str]:
    """List the GCS object keys currently present under the model dir."""
    bucket = get_bucket()
    prefix = f"{config.MODEL_GCS_DIR}/"
    return [b.name for b in bucket.list_blobs(prefix=prefix)]


if __name__ == "__main__":  # pragma: no cover - manual runner entry point
    summary = mirror_model()
    uploaded = sum(1 for v in summary.values() if v == "uploaded")
    skipped = sum(1 for v in summary.values() if v == "skipped")
    print(
        f"model mirror complete: {uploaded} uploaded, {skipped} skipped, "
        f"{len(summary)} files total"
    )
