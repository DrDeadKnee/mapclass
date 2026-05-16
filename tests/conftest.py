"""
Shared offline fixtures for the Phase 2 and Phase 3 test suites.

Every fixture here is fully offline: no network, no S3, no live API. Fixtures
that touch the filesystem write only under pytest's ``tmp_path``.

Fixture inventory (consumed by downstream plans 02-02 … 02-05 and 03-01 … 03-05):
  - sample_luna_item          — a David Rumsey LUNA result dict
  - sample_allmaps_annotation — a single W3C Web-Annotation matching
                                ``allmaps._parse_annotation``'s expected shape

  GCS I/O fixtures (02-01 Wave 0 — added for gcs_io module testing):
  - local_fs                  — fsspec.filesystem("file") for offline _GCSWriter tests
  - mock_gcs_module           — a module-level mock providing _MockGCSFileSystem
                                (in-memory _store: dict[str, bytes]) for patching
                                gcs_io.gcsfs in unit tests.  Mirrors the
                                _MockGCSModule pattern from tests/test_seg_gcs.py.
  - sample_allmaps_multi      — an ``items`` list of >=3 distinct annotations
                                (a multi-canvas atlas) for the Task 2 lookup fix
  - tiny_geotiff              — a 256x256 3-band uint8 EPSG:4326 GeoTIFF path
  - sample_azgaar_geojson     — a minimal Azgaar FeatureCollection path
  - mock_stac_item            — an offline stand-in for a pystac Item

  Phase 3 additions (03-01 Wave 0 — offline, Pillow + tiling only):
  - mini_pyramid              — a real Phase-2 21-tile pyramid produced by
                                tiling.tile(); returns the pyramid directory path
                                (the dir containing pyramid.json).  Fully offline:
                                Pillow writes PNGs; tiling.tile() emits the manifest.
                                No torch / transformers / timm imports.
  - stub_vision_config        — a minimal namespace object exposing hidden_size,
                                patch_size, num_hidden_layers for offline shape
                                tests where the real PaliGemma checkpoint is absent
                                (RESEARCH Assumption A1 / Environment Availability).
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


# ---------------------------------------------------------------------------
# Phase 3 seg fixtures (03-01 Wave 0)
# ---------------------------------------------------------------------------

import json as _json
from pathlib import Path as _Path
from PIL import Image as _Image
import types as _types


_SEG_WEIGHTS_BLOB = _json.dumps(
    {
        # All 9 LANDCOVER_CLASSES must be present (T-04-01 / build_lc_weight_tensor
        # validation).  Weights reflect the Phase-4 synthetic-source defaults:
        # satellite-only classes (cropland, built_up, flooded_wetland) set to 0.0
        # so the fixture never requires those labels; all others uniform at 1.0.
        "land_cover_weights": {
            "water":           1.0,
            "trees":           0.3,   # downweighted: historical stale
            "shrubland":       1.0,
            "grassland":       1.0,
            "cropland":        0.15,  # satellite-only
            "built_up":        0.15,  # satellite-only
            "bare_sparse":     1.0,
            "flooded_wetland": 0.15,  # satellite-only
            "snow_ice":        1.0,
        },
        "topography_weight": 1.0,
        "source": "synthetic",
        "map_file": "fixture",
    },
    indent=2,
)

# Source size chosen so that enumerate_pyramids emits at least one full
# 21-tile pyramid (896x896 fully on-map).  1792x1792 gives a 4x4 grid of
# pyramid origins on the stride-448 grid — 16 pyramids total; the first
# (origin 0,0) is 100% on-map.  Mirrors test_tiling.py::FULL_W/FULL_H.
_SEG_MAP_W = _SEG_MAP_H = 896 + 2 * 448  # 1792


def _make_seg_map_dir(base: _Path, width: int = _SEG_MAP_W, height: int = _SEG_MAP_H) -> _Path:
    """
    Write a complete per-map dir (image/lc/topo PNG + sample_weights.json).
    Mirrors tests/test_tiling.py::_make_map_dir exactly.
    No torch / transformers / timm dependencies.
    """
    base.mkdir(parents=True, exist_ok=True)
    _Image.new("RGB", (width, height), (10, 20, 30)).save(base / "image.png")
    _Image.new("L", (width, height), 1).save(base / "land_cover.png")
    _Image.new("L", (width, height), 2).save(base / "topography.png")
    (base / "sample_weights.json").write_text(_SEG_WEIGHTS_BLOB)
    return base


@pytest.fixture
def mini_pyramid(tmp_path):
    """
    A real Phase-2 21-tile pyramid produced by the live ``tiling.tile()``
    producer (D-04: never hand-write geometry — let the Phase-2 producer emit
    it).

    Synthesises a 1792x1792 map under ``tmp_path`` (RGB ``image.png``,
    L-mode ``land_cover.png`` + ``topography.png``, ``sample_weights.json``),
    then calls ``tiling.tile(map_dir)`` to produce the pyramid tree.

    Returns the first pyramid directory (the directory containing
    ``pyramid.json`` for the origin-0,0 pyramid), following the conftest
    ``tiny_geotiff`` return-path idiom.

    Fully offline: Pillow + tiling only; no torch / transformers / timm
    imports so the fixture works even when those packages are absent.

    Covered decisions: D-03, D-04, D-06a (construction-only smoke tests).
    """
    from tiling import tile

    map_dir = _make_seg_map_dir(tmp_path / "map")
    pyramid_root = tile(map_dir)

    # Pick the first (alphabetically lowest) pyramid dir.  The origin-0,0
    # pyramid is always present for a 1792x1792 source (fully on-map).
    pyramid_dirs = sorted(p for p in pyramid_root.iterdir() if p.is_dir())
    assert pyramid_dirs, "tiling.tile() produced no pyramid directories"
    # Sanity: the chosen dir must have a pyramid.json with 21 tiles.
    manifest = _json.loads((pyramid_dirs[0] / "pyramid.json").read_text())
    assert len(manifest["tiles"]) == 21, (
        f"Expected 21 tiles in mini_pyramid, got {len(manifest['tiles'])}"
    )
    return pyramid_dirs[0]


@pytest.fixture
def stub_vision_config():
    """
    A minimal namespace object exposing ``hidden_size``, ``patch_size``, and
    ``num_hidden_layers`` for offline shape tests where the real PaliGemma
    checkpoint is unavailable.

    Values match the SigLIP-So400m/14 config used by ``paligemma-3b-pt-224``
    (RESEARCH Assumption A1: hidden_size 1152, patch 14, 27 layers, no CLS).
    Tests that depend on this fixture skip gracefully when the real model is
    absent; the stub provides the numeric contracts so decoder/backbone shape
    assertions can be made without loading 3 B parameters.
    """
    cfg = _types.SimpleNamespace()
    cfg.hidden_size = 1152        # SigLIP-So400m/14 hidden dim
    cfg.patch_size = 14           # SigLIP-So400m/14 patch size
    cfg.num_hidden_layers = 27    # SigLIP-So400m/14 transformer depth
    cfg.image_size = 224          # native input resolution
    cfg.num_channels = 3          # RGB
    return cfg


# ---------------------------------------------------------------------------
# GCS I/O fixtures (02-01 Wave 0)
# Mirrors the _MockGCSModule pattern from tests/test_seg_gcs.py lines 76-121.
# ---------------------------------------------------------------------------

import io as _io
import fsspec as _fsspec
import types as _gcs_types


class _MockGCSFileSystem:
    """Offline in-memory stand-in for ``gcsfs.GCSFileSystem``.

    Stores blobs as ``dict[str, bytes]`` keyed by their bare path (no gs://).
    A single ``_store`` is shared across all instances within a test run.
    Reset between tests via ``_reset()`` if isolation is needed.

    Methods implemented:
      open(path, mode)   — BytesIO-backed context manager (read + write)
      ls(prefix)         — list keys starting with bare prefix
      pipe_file(path, data) — write bytes atomically to the store
      cat(path)          — read bytes from the store
      exists(path)       — check if key exists
      get(remote, local) — materialise store keys under a local directory
      mkdirs(path)       — no-op (GCS has no real directories)
    """

    _store: dict[str, bytes] = {}

    @classmethod
    def _reset(cls) -> None:
        cls._store.clear()

    def __init__(self, project: str | None = None) -> None:
        pass  # project arg accepted but ignored (offline)

    def open(self, path: str, mode: str = "rb"):
        """Return a BytesIO-backed context manager for read or write."""
        key = path.lstrip("gs://")
        store = self._store

        class _CM:
            def __init__(self_):
                self_._key = key
                self_._mode = mode

            def __enter__(self_):
                if "r" in self_._mode:
                    self_._buf = _io.BytesIO(store.get(self_._key, b""))
                else:
                    self_._buf = _io.BytesIO()
                return self_._buf

            def __exit__(self_, exc_type, exc_val, exc_tb):
                if "w" in self_._mode and exc_type is None:
                    store[self_._key] = self_._buf.getvalue()

        return _CM()

    def ls(self, prefix: str) -> list[str]:
        """Return stored keys whose bare path starts with *prefix*."""
        bare = prefix.lstrip("gs://")
        return [k for k in self._store if k.startswith(bare)]

    def pipe_file(self, path: str, data: bytes) -> None:
        """Write *data* to the store keyed by the bare path."""
        key = path.lstrip("gs://")
        self._store[key] = data

    def cat(self, path: str) -> bytes:
        """Read bytes from the store by bare path."""
        key = path.lstrip("gs://")
        return self._store.get(key, b"")

    def exists(self, path: str) -> bool:
        """Return True if the bare path is in the store."""
        key = path.lstrip("gs://")
        return key in self._store

    def get(self, remote: str, local: str, recursive: bool = False) -> None:
        """Materialise store keys whose prefix matches *remote* under *local*."""
        import os
        from pathlib import Path as _Path
        bare_remote = remote.lstrip("gs://").rstrip("/")
        local_root = _Path(local)
        local_root.mkdir(parents=True, exist_ok=True)
        for key, data in list(self._store.items()):
            if key.startswith(bare_remote + "/"):
                rel = key[len(bare_remote) + 1:]
                dest = local_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)

    def mkdirs(self, path: str, exist_ok: bool = True) -> None:
        """No-op for GCS (GCS has no real directories)."""
        pass


class _MockGCSModule:
    """Minimal stub for the ``gcsfs`` module (analogous to test_seg_gcs.py)."""
    GCSFileSystem = _MockGCSFileSystem


@pytest.fixture
def local_fs():
    """Return a local fsspec filesystem for offline _GCSWriter tests.

    Usage::

        def test_something(local_fs, tmp_path):
            writer = _GCSWriter(local_fs, str(tmp_path / "out"))
            writer.write_bytes(b"data")
    """
    return _fsspec.filesystem("file")


@pytest.fixture
def mock_gcs_module():
    """Return a mock gcsfs module exposing _MockGCSFileSystem.

    Resets the shared in-memory store at fixture setup so tests are isolated.

    Usage::

        def test_something(mock_gcs_module):
            with mock.patch("gcs_io.gcsfs", mock_gcs_module):
                pull_dataset_from_gcs("train", tmp_path)
    """
    _MockGCSFileSystem._reset()
    return _MockGCSModule
