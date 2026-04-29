# Executive TODO

This file tracks tasks that require direct action by the project owner — typically
external coordination (contacting licensing holders, dataset hosts), strategic decisions,
or experiments that the owner wants to oversee personally. Implementation-side todos
that Claude can pick up in-session are not listed here.

## Dataset licensing verification

Before claiming a "fully open-source" combined dataset, confirm each source permits
redistribution of derivative works (and under what terms). Contact owners where the public
license is ambiguous.

- [ ] **Copernicus DEM GLO-30** — review the Copernicus DEM license (issued by ESA /
      Airbus). Confirm whether slope-derived topography rasters count as derivative
      products and whether redistribution requires attribution only or additional terms.
      Source: https://spacedata.copernicus.eu/

- [ ] **David Rumsey Map Collection** — most uncertain source. Originals (16th–17th c.)
      are public domain, but Rumsey's digitizations and georeferenced GeoTIFFs carry
      separate institutional terms (last known: personal / educational / non-commercial
      use). Contact David Rumsey Map Center at Stanford to clarify whether the
      derived label-pair dataset (image + land_cover.png + topography.png) can be
      redistributed openly, and under what attribution.

- [ ] **Synthetic illustrated maps (Azgaar's Fantasy Map Generator)** — confirm
      Azgaar's license (MIT for the generator code, but verify generated map outputs
      are unencumbered for redistribution as training data).

- [ ] Once licenses are confirmed, decide on a single dataset license that is
      compatible with the most restrictive upstream source, and update mockup.md to
      reflect the actual terms (drop "fully open-source" if any source forbids it).

## Zero-shot baselines for failure analysis

- [ ] Run zero-shot benchmark of CLIP, SigLIP (and at least one OpenCLIP variant) on the
      held-out stylized map test set: terrain classification on hex tiles or equivalent
      patches. Record per-class accuracy and confusion matrices. Use these numbers to
      replace the current hand-wavy "CLIP is useless" claim with measured evidence, and
      as the entry point for the dynamic-LRP failure localisation analysis.
