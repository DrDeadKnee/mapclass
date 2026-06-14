"""
Fetch a IIIF image at a bounded max-edge size and scale Allmaps GCPs to the
fetched resolution.

The David Rumsey IIIF image surface is IIIF Image API 2.x; the size syntax
used here (``!w,h`` size-best-fit + ``/full/.../0/default.jpg``) is identical
between IIIF 2.x and 3.x, so versions are not special-cased.

``!w,h`` is *size-best-fit*: the server scales so neither dimension exceeds the
requested edge while preserving aspect ratio. The actually-served dimensions
are therefore NOT necessarily ``(max_edge, *)`` — they must be read back off
the saved JPEG and the GCP scale factor computed from the *actual* fetched
dimensions, not the requested ones (Pitfall 2).

HTTP retry mirrors ``rumsey._get_json``'s exponential-backoff structure; the
streaming-write idiom is salvaged from the deleted ``rumsey._download_wms_geotiff``.
"""

import time
from pathlib import Path

import requests
from PIL import Image

_IIIF_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}
_MAX_RETRIES = 4
_BACKOFF_BASE = 2.0  # seconds

# Abort before decoding if the server ignores `!w,h` and streams a huge body
# (threat T-02-04: oversized IIIF response exhausting memory/disk).
_MAX_CONTENT_BYTES = 200 * 1024 * 1024  # 200 MB


def build_iiif_url(image_service_id: str, max_edge: int = 4096) -> str:
    """
    Build the IIIF Image API request URL for a size-best-fit fetch.

    ``<image_service>/full/!{max_edge},{max_edge}/0/default.jpg`` — the ``!w,h``
    form preserves aspect ratio and guarantees neither dimension exceeds
    ``max_edge``. Any trailing ``/`` on the service id is stripped first.
    """
    base = image_service_id.rstrip("/")
    return f"{base}/full/!{max_edge},{max_edge}/0/default.jpg"


def fetch_iiif_image(
    image_service_id: str,
    output_path: Path,
    max_edge: int = 4096,
) -> tuple[Path, int, int]:
    """
    Fetch a IIIF image at a bounded max-edge size, stream it to ``output_path``,
    then reopen it to read the ACTUAL served ``(width, height)``.

    Returns ``(output_path, fetched_width, fetched_height)``. The caller passes
    the returned dimensions to :func:`scale_gcps` (the server may serve smaller
    than ``max_edge``; never assume the requested size).

    Aborts (raises ``RuntimeError``) if the server advertises a
    ``Content-Length`` greater than 200 MB (threat T-02-04).
    """
    output_path = Path(output_path)
    url = build_iiif_url(image_service_id, max_edge=max_edge)

    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(
                url, headers=_IIIF_HEADERS, timeout=120, stream=True
            )
            if resp.status_code in (429, 503):
                wait = _BACKOFF_BASE ** attempt
                print(f"  Rate limited ({resp.status_code}), waiting {wait:.0f}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()

            content_length = resp.headers.get("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) > _MAX_CONTENT_BYTES:
                        raise RuntimeError(
                            f"IIIF response too large "
                            f"({int(content_length)} bytes > "
                            f"{_MAX_CONTENT_BYTES}); aborting before decode"
                        )
                except ValueError:
                    pass  # unparseable header — fall through, decode guard below

            output_path.parent.mkdir(parents=True, exist_ok=True)
            written = 0
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    written += len(chunk)
                    if written > _MAX_CONTENT_BYTES:
                        raise RuntimeError(
                            f"IIIF stream exceeded {_MAX_CONTENT_BYTES} bytes; "
                            f"aborting"
                        )
                    f.write(chunk)

            with Image.open(output_path) as img:
                fetched_w, fetched_h = img.size
            return output_path, int(fetched_w), int(fetched_h)

        except requests.RequestException as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            time.sleep(_BACKOFF_BASE ** attempt)

    raise RuntimeError(f"exhausted retries fetching {url}")


def scale_gcps(
    gcps_orig: list,
    orig_w: int,
    orig_h: int,
    fetched_w: int,
    fetched_h: int,
) -> list:
    """
    Scale Allmaps GCPs (expressed in original-image pixel space) to the fetched
    resolution.

    ``gcps_orig`` is a list of ``((px, py), (lng, lat))`` tuples. The size-best-fit
    fetch preserves aspect ratio, so ``sx`` and ``sy`` are equal in the ideal
    case — but both are computed from the *actual* fetched dimensions and applied
    independently rather than assuming equality (Pitfall 2).
    """
    sx = fetched_w / orig_w
    sy = fetched_h / orig_h
    return [
        ((px * sx, py * sy), (lng, lat))
        for ((px, py), (lng, lat)) in gcps_orig
    ]
