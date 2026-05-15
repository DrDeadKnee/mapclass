"""
Cloud-filtered Sentinel-2 L2A scene search via the Element84 Earth Search
STAC API (https://earth-search.aws.element84.com/v1).

Mirrors ``historical/allmaps.py``: an endpoint/headers/retry-constant block, a
typed ``StacLookupError`` (analogous to ``AllmapsLookupError``), and the
canonical exponential-backoff retry loop. Transient failures are retried; an
empty result (no scene meeting the cloud filter) returns ``None`` so the caller
can drop-count it (D-13, same contract as Allmaps 404 → ``[]``).

RESEARCH Pattern 2: query the ``sentinel-2-l2a`` collection with the dict-style
``query={"eo:cloud_cover": {"lt": max_cloud}}`` filter and pick the
lowest-cloud item. The scene's ``visual`` asset is a pre-stacked B04/B03/B02
TCI 3-band uint8 COG (consumed by ``satellite/fetch.py``).

Pitfall 4 (datetime/cloud combinatorics): the caller supplies a per-region
``datetime_range`` (the coverage scan picks a latitude-appropriate season);
this module does not invent a global date window.

Anonymous public-S3 access for the downstream COG read is enabled module-top
(``AWS_NO_SIGN_REQUEST``), matching ``historical/worldcover.py:36``.
"""

import os
import time

# Allow anonymous access to public S3 buckets via GDAL VSI-CURL (sentinel-cogs).
os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")

ENDPOINT = "https://earth-search.aws.element84.com/v1"
_HEADERS = {"User-Agent": "mapclass-dataset-builder/0.1"}
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

_COLLECTION = "sentinel-2-l2a"


class StacLookupError(RuntimeError):
    """Raised for unexpected (non-empty-result) failures during STAC search."""


def find_lowest_cloud_scene(bbox, datetime_range, max_cloud: float = 10):
    """
    Return the lowest-cloud Sentinel-2 L2A STAC item for ``bbox`` within
    ``datetime_range`` whose ``eo:cloud_cover`` is below ``max_cloud``.

    Parameters
    ----------
    bbox : (west, south, east, north) in WGS84 degrees
    datetime_range : a STAC datetime string, e.g. "2023-05-01/2023-09-30"
                     (the caller / coverage scan picks a seasonal window)
    max_cloud : exclusive upper bound on eo:cloud_cover (percent)

    Returns
    -------
    A pystac ``Item`` with the minimum ``eo:cloud_cover``, or ``None`` when no
    scene qualifies (caller drop-counts as ``no_qualifying_scene``, D-13).

    Raises
    ------
    StacLookupError on unexpected/persistent failure after retries.
    """
    # Lazy import: keeps offline tests (which patch this symbol) free of a hard
    # pystac_client dependency at collection time.
    import pystac_client

    for attempt in range(_MAX_RETRIES):
        try:
            client = pystac_client.Client.open(ENDPOINT, headers=_HEADERS)
            search = client.search(
                collections=[_COLLECTION],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": max_cloud}},
            )
            items = list(search.items())
        except (TypeError, ValueError, AttributeError):
            # WR-04: deterministic programming / bad-input errors (bad bbox
            # shape, invalid datetime string, a pystac API change) are NOT
            # transient — retrying wastes ~6s+ per region and obscures the
            # real cause. Propagate immediately without retry/backoff.
            raise
        except Exception as exc:  # transient network / server failure
            if attempt == _MAX_RETRIES - 1:
                raise StacLookupError(
                    f"STAC search failed for bbox={bbox} "
                    f"datetime={datetime_range}: {exc}"
                ) from exc
            time.sleep(_BACKOFF_BASE ** attempt)
            continue

        if not items:
            return None  # no qualifying scene — caller drop-counts (D-13)

        return min(
            items,
            key=lambda it: it.properties.get("eo:cloud_cover", 100),
        )

    raise StacLookupError(
        f"exhausted retries for STAC search bbox={bbox} datetime={datetime_range}"
    )
