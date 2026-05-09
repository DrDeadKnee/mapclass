"""
Per-source × per-class loss weights loader + strict schema validator.

PITFALL 3 prevention: schema is validated at training startup. Every source row
MUST list ALL 9 land-cover classes (explicit 0.0 for absent — no implicit defaults).

Hash of the YAML text content is embedded in safetensors checkpoint metadata as
source_class_weights_hash (TS-8 / Phase 1 success criterion 1).
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from mapclass.data.taxonomy import LANDCOVER_CLASSES, TOPO_CLASSES


class LossWeightsSchemaError(ValueError):
    """Raised by LossWeights.load on schema drift (PITFALL 3 prevention)."""


@dataclass(frozen=True)
class LossWeights:
    land_cover: dict[str, dict[str, float]]   # source_subtype -> class_name -> weight
    topography: dict[str, float]              # source_subtype -> weight
    source_class_weights_hash: str            # sha256 of the loaded YAML text

    @classmethod
    def load(cls, path: str | Path) -> "LossWeights":
        path = Path(path)
        text = path.read_text()
        data = yaml.safe_load(text)
        if not isinstance(data, dict) or "sources" not in data:
            raise LossWeightsSchemaError(f"{path}: missing top-level 'sources' key")

        valid_classes = set(LANDCOVER_CLASSES)
        land_cover: dict[str, dict[str, float]] = {}
        topography: dict[str, float] = {}

        for source_name, row in data["sources"].items():
            if not isinstance(row, dict):
                raise LossWeightsSchemaError(f"source '{source_name}': row must be a mapping")
            if "land_cover" not in row or "topography" not in row:
                raise LossWeightsSchemaError(
                    f"source '{source_name}': missing 'land_cover' or 'topography' key"
                )

            lc_row = row["land_cover"]
            for cls_name in lc_row:
                if cls_name not in valid_classes:
                    raise LossWeightsSchemaError(
                        f"source '{source_name}': unknown class '{cls_name}'; "
                        f"valid: {sorted(valid_classes)}"
                    )
            for required_cls in LANDCOVER_CLASSES:
                if required_cls not in lc_row:
                    raise LossWeightsSchemaError(
                        f"source '{source_name}': missing class '{required_cls}' — "
                        f"explicit 0.0 required, no implicit default (PITFALL 3 prevention)"
                    )
                if not isinstance(lc_row[required_cls], (int, float)):
                    raise LossWeightsSchemaError(
                        f"source '{source_name}', class '{required_cls}': weight must be a float"
                    )

            topo_w = row["topography"]
            if not isinstance(topo_w, (int, float)):
                raise LossWeightsSchemaError(
                    f"source '{source_name}': 'topography' must be a single float"
                )
            if not (0.0 <= float(topo_w) <= 1.0):
                raise LossWeightsSchemaError(
                    f"source '{source_name}': 'topography' weight {topo_w} out of [0.0, 1.0]"
                )

            land_cover[source_name] = {k: float(v) for k, v in lc_row.items()}
            topography[source_name] = float(topo_w)

        return cls(
            land_cover=land_cover,
            topography=topography,
            source_class_weights_hash=hashlib.sha256(text.encode()).hexdigest(),
        )

    def lookup(self, source_subtype: str, lc_class_name: str) -> tuple[float, float]:
        """Returns (lc_weight, topo_weight) for a given source × class."""
        if source_subtype not in self.land_cover:
            raise LossWeightsSchemaError(f"unknown source_subtype: {source_subtype}")
        return (self.land_cover[source_subtype][lc_class_name], self.topography[source_subtype])
