"""Run-once, idempotent model-weights mirror: HF snapshot → GCS ``models/``.

``huggingface_hub.snapshot_download`` (handles sharding/resume/revision) then a
recursive upload of every file into ``gs://{GCS_BUCKET}/{gcs_dir}/<relpath>``,
skipping any blob that already exists with a matching byte size. Re-runnable to
convergence (2nd run uploads nothing).

GENERALIZED (v1.1): ``download_snapshot`` / ``mirror_model`` take explicit
``(repo_id, gcs_dir, local_dir=None, token=None)`` instead of reading the
SigLIP-2-only ``config.MODEL_REPO_ID`` / ``config.MODEL_GCS_DIR``. The
idempotent skip-if-same-byte-size upload logic is unchanged.

The ``__main__`` runner mirrors the two NON-gated comparison repos required by
the v1.1 notebook (Task 3):

  * ``openai/clip-vit-large-patch14``  -> ``models/clip-vit-large-patch14``
  * ``google/vit-base-patch16-224``    -> ``models/vit-base-patch16-224``

Both pass ``token=None`` (public / non-gated). NO PaliGemma mirror step:
PaliGemma is ALREADY present at ``models/paligemma-3b-mix-224/`` and is NOT
gated here — there is deliberately NO HF-token / Gemma-license path anywhere
in this module. No package install, no pin bump, no vendor change.
"""

from __future__ import annotations

import os
from typing import Dict, List

from mapclass import config
from mapclass.ingest_images import get_bucket, verify_adc

# (repo_id, gcs_dir) pairs the v1.1 notebook needs mirrored. Both NON-gated.
_MIRROR_TARGETS = [
    ("openai/clip-vit-large-patch14", "models/clip-vit-large-patch14"),
    ("google/vit-base-patch16-224", "models/vit-base-patch16-224"),
]


def download_snapshot(
    repo_id: str, local_dir: str | None = None, token: str | None = None
) -> str:
    """Download a pinned checkpoint locally. Returns the snapshot path.

    ``token`` is forwarded to ``snapshot_download`` (None == anonymous; the
    v1.1 CLIP/ViT repos are non-gated so None is correct for them).
    """
    from huggingface_hub import snapshot_download

    target = local_dir or os.path.join(
        config.LOCAL_MODEL_CACHE_DIR, repo_id.replace("/", "__")
    )
    os.makedirs(target, exist_ok=True)
    return snapshot_download(
        repo_id=repo_id, local_dir=target, token=token
    )


def _iter_files(root: str):
    """Yield ``(absolute_path, relpath)`` for every file under ``root``."""
    for dirpath, _dirs, filenames in os.walk(root):
        for name in filenames:
            abspath = os.path.join(dirpath, name)
            relpath = os.path.relpath(abspath, root)
            yield abspath, relpath


def mirror_model(
    repo_id: str,
    gcs_dir: str,
    local_dir: str | None = None,
    token: str | None = None,
) -> Dict[str, str]:
    """Idempotently mirror a model snapshot to ``gs://.../{gcs_dir}/``.

    Returns a ``{relpath: status}`` map where status is ``uploaded`` or
    ``skipped`` (skipped == already present with the same byte size).
    """
    verify_adc()
    snapshot_dir = download_snapshot(repo_id, local_dir, token)
    bucket = get_bucket()

    results: Dict[str, str] = {}
    for abspath, relpath in _iter_files(snapshot_dir):
        # Forward-slash key under the model dir (GCS keys are literal).
        key = f"{gcs_dir}/" + relpath.replace(os.sep, "/")
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


def list_mirrored_keys(gcs_dir: str) -> List[str]:
    """List the GCS object keys currently present under ``gcs_dir``."""
    bucket = get_bucket()
    prefix = f"{gcs_dir}/"
    return [b.name for b in bucket.list_blobs(prefix=prefix)]


if __name__ == "__main__":  # pragma: no cover - manual runner entry point
    for _repo_id, _gcs_dir in _MIRROR_TARGETS:
        print(f"mirroring {_repo_id} -> gs://{config.GCS_BUCKET}/{_gcs_dir}/ ...")
        _summary = mirror_model(_repo_id, _gcs_dir, token=None)
        _up = sum(1 for v in _summary.values() if v == "uploaded")
        _sk = sum(1 for v in _summary.values() if v == "skipped")
        print(
            f"  {_repo_id}: {_up} uploaded, {_sk} skipped, "
            f"{len(_summary)} files total"
        )
    print("mirror complete (CLIP + ViT-b-16; no PaliGemma/HF-token step)")
