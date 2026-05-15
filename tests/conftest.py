"""
Shared offline fixtures for the Phase 2 test suite.

Every fixture here is fully offline: no network, no S3, no live API. Fixtures
that touch the filesystem write only under pytest's ``tmp_path``.

Fixture inventory (consumed by downstream plans 02-02 … 02-05):
  - sample_luna_item          — a David Rumsey LUNA result dict
  - sample_allmaps_annotation — a single W3C Web-Annotation matching
                                ``allmaps._parse_annotation``'s expected shape
  - sample_allmaps_multi      — an ``items`` list of >=3 distinct annotations
                                (a multi-canvas atlas) for the Task 2 lookup fix
  - tiny_geotiff              — a 256x256 3-band uint8 EPSG:4326 GeoTIFF path
  - sample_azgaar_geojson     — a minimal Azgaar FeatureCollection path
  - mock_stac_item            — an offline stand-in for a pystac Item
"""

import json

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds


# ---------------------------------------------------------------------------
# LUNA (David Rumsey) search-result fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_luna_item():
    """
    A dict mirroring a single David Rumsey LUNA search result.

    Mirrors the fields ``rumsey.search_maps`` / ``build_historical_dataset``
    consume: a tilde-delimited ``id`` (RUMSEY~collection~mediabin~record~object)
    and an ``iiifManifest`` URL.
    """
    return {
        "id": "RUMSEY~8~1~123~456",
        "iiifManifest": "https://www.davidrumsey.com/luna/servlet/iiif/m/RUMSEY~8~1~123~456/manifest",
        "displayName": "A New Map of Europe, 1631",
        "thumbnailUrl": "https://www.davidrumsey.com/luna/servlet/iiif/RUMSEY~8~1~123~456/full/!256,256/0/default.jpg",
        "date": "1631",
    }


# ---------------------------------------------------------------------------
# Allmaps annotation fixtures
# ---------------------------------------------------------------------------

def _make_annotation(annotation_id, image_id, gcps, width=4096, height=3072):
    """
    Build a single W3C Web-Annotation in the shape ``_parse_annotation``
    expects: ``body.features[].properties.resourceCoords`` (pixel) and
    ``geometry.coordinates`` (lng, lat); ``target.source`` carries the
    IIIF image id + dimensions.

    ``gcps`` is a list of ((x_px, y_px), (lng, lat)) tuples.
    """
    features = [
        {
            "type": "Feature",
            "properties": {"resourceCoords": [float(px[0]), float(px[1])]},
            "geometry": {"type": "Point", "coordinates": [float(ll[0]), float(ll[1])]},
        }
        for px, ll in gcps
    ]
    return {
        "id": annotation_id,
        "type": "Annotation",
        "body": {
            "type": "GeoreferencedMap",
            "features": features,
            "transformation": {"type": "polynomial"},
            "_allmaps": {"id": annotation_id.rsplit("/", 1)[-1]},
        },
        "target": {
            "source": {"id": image_id, "width": width, "height": height},
            "selector": {
                "type": "SvgSelector",
                "value": f'<svg><polygon points="0,0 {width},0 {width},{height} 0,{height}" /></svg>',
            },
        },
    }


@pytest.fixture
def sample_allmaps_annotation():
    """
    A single well-formed Allmaps annotation with 4 GCPs (>= the 3-GCP guard
    in ``_parse_annotation``). Bounding box spans roughly central Europe.
    """
    gcps = [
        ((100.0, 120.0), (2.0, 51.0)),
        ((3900.0, 140.0), (15.0, 51.0)),
        ((3950.0, 2900.0), (15.0, 41.0)),
        ((90.0, 2950.0), (2.0, 41.0)),
    ]
    return _make_annotation(
        "https://annotations.allmaps.org/maps/abc123",
        "https://iiif.davidrumsey.com/iiif/2/RUMSEY~8~1~123~456",
        gcps,
    )


@pytest.fixture
def sample_allmaps_multi(sample_allmaps_annotation):
    """
    An ``items`` list with 3 distinct, valid annotations — simulating a
    multi-canvas Rumsey atlas where each canvas is independently
    georeferenced. Used by the Task 2 ``lookup()`` multi-annotation fix.
    """
    plate2 = _make_annotation(
        "https://annotations.allmaps.org/maps/def456",
        "https://iiif.davidrumsey.com/iiif/2/RUMSEY~8~1~123~457",
        [
            ((50.0, 60.0), (-5.0, 40.0)),
            ((2000.0, 55.0), (3.0, 40.0)),
            ((2010.0, 1500.0), (3.0, 35.0)),
            ((45.0, 1490.0), (-5.0, 35.0)),
        ],
        width=2048,
        height=1536,
    )
    plate3 = _make_annotation(
        "https://annotations.allmaps.org/maps/ghi789",
        "https://iiif.davidrumsey.com/iiif/2/RUMSEY~8~1~123~458",
        [
            ((10.0, 12.0), (10.0, 60.0)),
            ((3000.0, 15.0), (25.0, 60.0)),
            ((3010.0, 2000.0), (25.0, 50.0)),
            ((8.0, 1990.0), (10.0, 50.0)),
        ],
    )
    return {"items": [sample_allmaps_annotation, plate2, plate3]}


# ---------------------------------------------------------------------------
# Raster fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_geotiff(tmp_path):
    """
    Write a 256x256 3-band uint8 GeoTIFF with ``crs=EPSG:4326`` and a valid
    affine ``transform`` so ``historical.label.make_labels`` can consume
    ``ds.crs`` / ``ds.transform`` directly. Returns the path.

    The window spans a small ~1° box so any downstream WorldCover/DEM fetch
    is bounded (fetches are mocked in downstream tests, never live here).
    """
    path = tmp_path / "tiny.tif"
    width = height = 256
    # ~1 degree box over a land area; arbitrary but valid WGS84 bounds.
    west, south, east, north = 4.0, 50.0, 5.0, 51.0
    transform = from_bounds(west, south, east, north, width, height)
    data = (np.random.default_rng(42).integers(0, 256, size=(3, height, width))).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as ds:
        ds.write(data)
    return path


# ---------------------------------------------------------------------------
# Synthetic (Azgaar) fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_azgaar_geojson(tmp_path):
    """
    Write a minimal Azgaar-style FeatureCollection with 3 Polygon features,
    each carrying integer ``properties.biome`` and ``properties.height``
    (the fields ``scripts/label.py`` reads). Returns the path.
    """
    fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"biome": 6, "height": 35},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [100, 0], [100, 100], [0, 100], [0, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"biome": 1, "height": 80},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[100, 0], [200, 0], [200, 100], [100, 100], [100, 0]]],
                },
            },
            {
                "type": "Feature",
                "properties": {"biome": 12, "height": 5},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 100], [200, 100], [200, 200], [0, 200], [0, 100]]],
                },
            },
        ],
    }
    path = tmp_path / "azgaar_sample.geojson"
    path.write_text(json.dumps(fc))
    return path


# ---------------------------------------------------------------------------
# STAC fixture
# ---------------------------------------------------------------------------

class _MockAsset:
    def __init__(self, href):
        self.href = href


class _MockStacItem:
    """Offline stand-in for a pystac Item: ``.assets`` + ``.properties``."""

    def __init__(self):
        self.assets = {
            "visual": _MockAsset(
                "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
                "32/T/MT/2023/6/S2A_32TMT_20230615_0_L2A/TCI.tif"
            )
        }
        self.properties = {
            "eo:cloud_cover": 3.2,
            "datetime": "2023-06-15T10:20:00Z",
        }


@pytest.fixture
def mock_stac_item():
    """
    An offline object with ``.assets["visual"].href`` and
    ``.properties["eo:cloud_cover"]`` for STAC tests without network.
    """
    return _MockStacItem()
