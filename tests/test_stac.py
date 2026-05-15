"""
Offline tests for ``satellite.stac`` — the cloud-filtered Sentinel-2 L2A
STAC search wrapper (plan 02-04, D-13).

``pystac_client`` is patched so no network/STAC API is touched.
"""

import pytest

from satellite import stac


class _Item:
    def __init__(self, cloud):
        self.properties = {"eo:cloud_cover": cloud}


class _Search:
    def __init__(self, items):
        self._items = items

    def items(self):
        return iter(self._items)


class _Client:
    def __init__(self, items):
        self._items = items

    def search(self, **kwargs):
        # Record the query so the test can assert the cloud filter shape.
        _Client.last_kwargs = kwargs
        return _Search(self._items)


def _patch_client(mocker, items):
    fake_module = mocker.MagicMock()
    fake_module.Client.open.return_value = _Client(items)
    mocker.patch.dict("sys.modules", {"pystac_client": fake_module})
    return fake_module


def test_picks_lowest_cloud_scene(mocker):
    """find_lowest_cloud_scene returns the item with minimum eo:cloud_cover."""
    items = [_Item(42.0), _Item(3.2), _Item(17.5)]
    _patch_client(mocker, items)

    result = stac.find_lowest_cloud_scene(
        bbox=(4.0, 50.0, 5.0, 51.0),
        datetime_range="2023-05-01/2023-09-30",
        max_cloud=10,
    )
    assert result is items[1]
    assert result.properties["eo:cloud_cover"] == 3.2


def test_returns_none_for_empty_result(mocker):
    """No qualifying scene → None (caller drop-counts as no_qualifying_scene)."""
    _patch_client(mocker, [])
    result = stac.find_lowest_cloud_scene(
        bbox=(4.0, 50.0, 5.0, 51.0),
        datetime_range="2023-05-01/2023-09-30",
    )
    assert result is None


def test_query_uses_cloud_cover_lt_filter(mocker):
    """The STAC search is issued with the dict-style eo:cloud_cover < filter."""
    _patch_client(mocker, [_Item(1.0)])
    stac.find_lowest_cloud_scene(
        bbox=(0, 0, 1, 1),
        datetime_range="2023-01-01/2023-12-31",
        max_cloud=10,
    )
    q = _Client.last_kwargs["query"]
    assert q == {"eo:cloud_cover": {"lt": 10}}
    assert _Client.last_kwargs["collections"] == ["sentinel-2-l2a"]


def test_unexpected_failure_raises_stac_lookup_error(mocker):
    """A persistent client failure surfaces as the typed StacLookupError."""
    fake_module = mocker.MagicMock()
    fake_module.Client.open.side_effect = RuntimeError("boom")
    mocker.patch.dict("sys.modules", {"pystac_client": fake_module})
    mocker.patch("satellite.stac.time.sleep")  # no real backoff sleeps

    with pytest.raises(stac.StacLookupError):
        stac.find_lowest_cloud_scene(
            bbox=(0, 0, 1, 1),
            datetime_range="2023-01-01/2023-12-31",
        )
