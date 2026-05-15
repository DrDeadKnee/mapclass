"""
Offline tests for ``historical.iiif`` — the IIIF best-fit URL builder and the
GCP scale-factor logic (plan 02-02, RESEARCH Pattern 3 / Pitfall 2).
"""

from historical import iiif


def test_build_iiif_url():
    """build_iiif_url emits <service>/full/!4096,4096/0/default.jpg, trailing / stripped."""
    url = iiif.build_iiif_url("https://iiif.davidrumsey.com/iiif/2/RUMSEY~8~1~123~456/")
    assert url.endswith("/full/!4096,4096/0/default.jpg")
    assert "//full" not in url  # trailing slash on the service id was stripped


def test_scale_factor_best_fit():
    """Scale from ACTUAL fetched dims, applied identically to x and y (Pitfall 2)."""
    orig_w, orig_h = 6000, 3000
    fetched_w, fetched_h = 4096, 2048  # size-best-fit preserves aspect ratio
    gcps = [((0.0, 0.0), (2.0, 51.0)), ((6000.0, 3000.0), (15.0, 41.0))]
    scaled = iiif.scale_gcps(gcps, orig_w, orig_h, fetched_w, fetched_h)

    sx = fetched_w / orig_w
    sy = fetched_h / orig_h
    assert abs(sx - sy) < 1e-9  # best-fit ⇒ identical ratio
    (p0, _), (p1, _) = scaled[0], scaled[1]
    assert abs(p1[0] - 6000.0 * sx) < 1e-6
    assert abs(p1[1] - 3000.0 * sy) < 1e-6
    assert p0 == (0.0, 0.0)
