"""
Offline assertions for ``historical.allmaps.lookup`` after the A6 / Pitfall 3
fix: lookup() returns a list of ALL parseable annotations per manifest
(a multi-canvas atlas yields one entry per canvas), [] for 404 / no-items /
all-malformed.

All network I/O is mocked — these tests never hit annotations.allmaps.org.
The online counterpart lives in tests/integration/test_allmaps_online.py.
"""

from historical import allmaps


class _FakeResponse:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = "" if payload is None else "fake"

    def json(self):
        return self._payload


def test_lookup_returns_all_annotations(mocker, sample_allmaps_multi):
    """lookup() returns every parseable annotation in items[], not just items[0]."""
    mocker.patch(
        "historical.allmaps.requests.get",
        return_value=_FakeResponse(200, sample_allmaps_multi),
    )
    result = allmaps.lookup("https://example.org/manifest")
    assert isinstance(result, list)
    assert len(result) == 3
    for ann in result:
        assert len(ann["gcps"]) >= 3


def test_lookup_empty_on_404(mocker):
    """A 404 (not in Allmaps) yields an empty list, never None."""
    mocker.patch(
        "historical.allmaps.requests.get",
        return_value=_FakeResponse(404),
    )
    assert allmaps.lookup("https://example.org/missing") == []


def test_lookup_empty_on_no_items(mocker):
    """A 200 with an empty items[] yields an empty list."""
    mocker.patch(
        "historical.allmaps.requests.get",
        return_value=_FakeResponse(200, {"items": []}),
    )
    assert allmaps.lookup("https://example.org/empty") == []


def test_lookup_skips_malformed(mocker, sample_allmaps_annotation):
    """A mix of one valid annotation and two <3-GCP ones returns only the valid one."""
    too_few_gcps = {
        "id": "https://annotations.allmaps.org/maps/bad1",
        "body": {
            "features": [
                {
                    "type": "Feature",
                    "properties": {"resourceCoords": [1.0, 2.0]},
                    "geometry": {"type": "Point", "coordinates": [3.0, 4.0]},
                }
            ]
        },
        "target": {"source": {"id": "x", "width": 10, "height": 10}},
    }
    no_features = {
        "id": "https://annotations.allmaps.org/maps/bad2",
        "body": {"features": []},
        "target": {"source": {"id": "y", "width": 10, "height": 10}},
    }
    payload = {"items": [sample_allmaps_annotation, too_few_gcps, no_features]}
    mocker.patch(
        "historical.allmaps.requests.get",
        return_value=_FakeResponse(200, payload),
    )
    result = allmaps.lookup("https://example.org/mixed")
    assert len(result) == 1
    assert result[0]["annotation_id"] == sample_allmaps_annotation["id"]
