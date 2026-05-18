"""Run-once, idempotent, resumable Rumsey image mirror → GCS ``data/``.

Mirrors every manifest ``image_url`` into
``gs://{GCS_BUCKET}/{GCS_DATA_PREFIX}{id}.jpg`` with a per-id outcome manifest.

Security / threat mitigations (Security Domain V5; threat register
T-01-04..08):
  * ADC fail-fast at job start (T-01-07) — a clear error, not 1,000 items in.
  * Skip-if-exists-and-size>0 idempotency; re-runs converge to all "skipped"
    with zero new uploads (T-01-08 / Pitfall F).
  * Per-download validation before any GCS write (T-01-04 / T-01-05):
      - HTTP 200                       -> else ``dead-url``
      - Content-Type contains "image"  -> else ``bad-content-type``
      - byte cap (Content-Length + streamed size)  (decompression-bomb defense)
      - pixel-dimension cap on decode  (decompression-bomb defense)
      - ``PIL.Image.open(BytesIO).verify()``  -> else ``decode-fail``
  * The GCS key is ``data/<id>.jpg`` against a single fixed bucket via the
    blob API (no shell, no os.path.join) — ids are dataset-controlled RUMSEY~
    tokens, GCS treats the key literally (T-01-06).

The full 1,544 ingest is a long-running resumable step (D-01/D-02) — driven by
``notebooks/scripts/run_ingest.py``.
"""

from __future__ import annotations

import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from mapclass import config

# --- Tunables --------------------------------------------------------------
# Decompression-bomb defenses (Security Domain V5).
MAX_IMAGE_BYTES = 64 * 1024 * 1024          # 64 MiB hard cap on a single fetch
MAX_IMAGE_PIXELS = 178_956_970              # PIL's default bomb threshold
REQUEST_TIMEOUT_SECONDS = 60
MAX_CONCURRENCY = 8
MAX_RETRIES = 5                              # exponential backoff on 429/5xx

# Terminal statuses written into the per-id outcome manifest.
STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_DEAD_URL = "dead-url"
STATUS_BAD_CONTENT_TYPE = "bad-content-type"
STATUS_DECODE_FAIL = "decode-fail"
STATUS_TOO_LARGE = "too-large"
STATUS_ERROR = "error"

TERMINAL_STATUSES = frozenset(
    {
        STATUS_OK,
        STATUS_SKIPPED,
        STATUS_DEAD_URL,
        STATUS_BAD_CONTENT_TYPE,
        STATUS_DECODE_FAIL,
        STATUS_TOO_LARGE,
        STATUS_ERROR,
    }
)

OUTCOME_MANIFEST_BLOB = "data/_ingest_outcome.json"
LOCAL_OUTCOME_MANIFEST = os.path.join(
    config.LOCAL_IMAGE_CACHE_DIR, "_ingest_outcome.json"
)


# --- ADC fail-fast (T-01-07, Assumption A6) --------------------------------
def verify_adc() -> None:
    """Raise a clear error immediately if GCS ADC is absent/invalid.

    Called at job start so an auth problem fails fast — not 1,000 items in.
    """
    try:
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
    except Exception as exc:  # pragma: no cover - import-time only
        raise RuntimeError(
            "google-cloud-storage / google-auth is not importable. The pinned "
            "environment (Plan 01-01) must be installed before ingestion."
        ) from exc

    try:
        google.auth.default()
    except DefaultCredentialsError as exc:
        raise RuntimeError(
            "GCS Application Default Credentials are NOT available on this VM. "
            "Run `gcloud auth application-default login` (or confirm the VM "
            "service account) before ingestion. Failing fast at job start "
            "rather than mid-run (threat T-01-07)."
        ) from exc


def get_bucket():
    """Return the configured GCS bucket handle (ADC-authenticated)."""
    from google.cloud import storage

    client = storage.Client()
    return client.bucket(config.GCS_BUCKET)


# --- Single-id mirror (idempotent + validated) -----------------------------
def _validate_and_read(entry: Dict) -> Tuple[str, Optional[bytes]]:
    """Fetch + validate one image. Returns ``(status, data_or_None)``.

    Never writes to GCS. ``data`` is only non-None when status is OK.
    """
    import requests
    from PIL import Image

    url = entry["image_url"]
    try:
        resp = requests.get(
            url, timeout=REQUEST_TIMEOUT_SECONDS, stream=True
        )
    except requests.RequestException:
        return STATUS_DEAD_URL, None

    if resp.status_code != 200:
        return STATUS_DEAD_URL, None

    if "image" not in resp.headers.get("Content-Type", ""):
        return STATUS_BAD_CONTENT_TYPE, None

    # Content-Length pre-check before full materialization (T-01-05).
    declared = resp.headers.get("Content-Length")
    if declared is not None:
        try:
            if int(declared) > MAX_IMAGE_BYTES:
                return STATUS_TOO_LARGE, None
        except ValueError:
            pass  # malformed header — fall through to the streamed cap below

    # Streamed read with a hard byte cap (T-01-05).
    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 16):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            return STATUS_TOO_LARGE, None
        buf.write(chunk)
    data = buf.getvalue()
    if not data:
        return STATUS_DECODE_FAIL, None

    # Decode-safety: verify() + a pixel-dimension cap (T-01-04 / T-01-05).
    try:
        prev_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
        try:
            probe = Image.open(io.BytesIO(data))
            w, h = probe.size
            if w * h > MAX_IMAGE_PIXELS:
                return STATUS_TOO_LARGE, None
            Image.open(io.BytesIO(data)).verify()
        finally:
            Image.MAX_IMAGE_PIXELS = prev_limit
    except Exception:
        return STATUS_DECODE_FAIL, None

    return STATUS_OK, data


def mirror_one(bucket, entry: Dict) -> str:
    """Idempotently mirror one manifest entry. Returns its terminal status.

    Skip-if-exists-and-size>0 (idempotent / resumable convergence). On a fresh
    id, validate fully (Security Domain V5) before the single atomic
    ``upload_from_string``.
    """
    blob = bucket.blob(f"{config.GCS_DATA_PREFIX}{entry['id']}.jpg")

    if blob.exists():
        # ``size`` may require a reload for a freshly-listed blob handle.
        size = blob.size
        if size is None:
            blob.reload()
            size = blob.size
        if size and size > 0:
            return STATUS_SKIPPED

    last_status = STATUS_ERROR
    for attempt in range(MAX_RETRIES):
        status, data = _validate_and_read(entry)
        last_status = status
        # Retry only transient network failures (dead-url here may be a 429/5xx
        # transient); other terminal validation statuses are deterministic.
        if status == STATUS_OK and data is not None:
            blob.upload_from_string(data, content_type="image/jpeg")
            return STATUS_OK
        if status == STATUS_DEAD_URL and attempt < MAX_RETRIES - 1:
            backoff = (2 ** attempt) + random.random()
            time.sleep(backoff)
            continue
        return status
    return last_status


# --- Outcome manifest ------------------------------------------------------
def _write_outcome_manifest(bucket, outcomes: Dict[str, str]) -> None:
    """Persist the per-id outcome manifest to GCS and a local copy."""
    payload = json.dumps(outcomes, indent=2, sort_keys=True)
    os.makedirs(os.path.dirname(LOCAL_OUTCOME_MANIFEST), exist_ok=True)
    with open(LOCAL_OUTCOME_MANIFEST, "w", encoding="utf-8") as fh:
        fh.write(payload)
    bucket.blob(OUTCOME_MANIFEST_BLOB).upload_from_string(
        payload, content_type="application/json"
    )


def status_histogram(outcomes: Dict[str, str]) -> Dict[str, int]:
    """Return a ``{status: count}`` histogram over the outcome manifest."""
    hist: Dict[str, int] = {}
    for status in outcomes.values():
        hist[status] = hist.get(status, 0) + 1
    return hist


def run_full_ingest(
    manifest: List[Dict], max_workers: int = MAX_CONCURRENCY
) -> Dict[str, str]:
    """Mirror the full manifest with bounded concurrency. Resumable.

    ADC is verified fail-fast first. Every id ends with a terminal status in
    the returned outcome map (also written to GCS + a local copy). A 2nd run
    converges: already-present ids report ``skipped`` and upload nothing.
    """
    verify_adc()
    bucket = get_bucket()

    outcomes: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(mirror_one, bucket, entry): entry["id"]
            for entry in manifest
        }
        for fut in as_completed(futures):
            entry_id = futures[fut]
            try:
                outcomes[entry_id] = fut.result()
            except Exception:  # never let one id abort the whole run
                outcomes[entry_id] = STATUS_ERROR

    _write_outcome_manifest(bucket, outcomes)
    return outcomes
