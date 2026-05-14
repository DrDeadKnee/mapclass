# Synthesis Summary

Entry point for `gsd-roadmapper` and other downstream consumers. This file summarises what was extracted from the classified inputs and points to the per-type intel files.

## Doc counts

- Total docs synthesized: 2
- SPEC: 1 (`/home/drdreadknee/mapclass/README.md`, locked, high confidence)
- DOC: 1 (`/home/drdreadknee/mapclass/notes/architectural_references.md`, non-locked, high confidence)
- ADR: 0
- PRD: 0
- UNKNOWN: 0

## Decisions

- ADR-derived decisions locked: 0 (no ADRs in the ingest set)
- See `decisions.md` for placeholder note.

## Requirements

- Requirements extracted: 0 (no PRDs in the ingest set)
- See `requirements.md` for placeholder note.

## Constraints (from SPEC)

- Total constraints: 10
- By type:
  - schema: 2 (CONSTRAINT-land-cover-taxonomy, CONSTRAINT-topography-taxonomy)
  - api-contract: 1 (CONSTRAINT-data-sources)
  - protocol: 6 (CONSTRAINT-paligemma-finetuning, CONSTRAINT-georeferencing-pipeline, CONSTRAINT-class-conditional-loss-weights, CONSTRAINT-segmentation-backbone, CONSTRAINT-evaluation-metric, CONSTRAINT-phasing)
  - nfr: 2 (CONSTRAINT-product-scope, CONSTRAINT-deliverable)
- All 10 constraints are sourced from `/home/drdreadknee/mapclass/README.md` and inherit its locked status.
- See `constraints.md` for full content.

## Context (from DOC)

- Context topics: 5
  - Why context-aware segmentation
  - Candidate backbone — Recursive coarse-to-fine
  - Candidate backbone — Large first-kernel ConvNet (stretch goal)
  - Planned experiment — Progressive backbone unfreezing for domain-gap analysis
  - Reference reading list
- All from `/home/drdreadknee/mapclass/notes/architectural_references.md`.
- See `context.md`.

## Conflicts

- Blockers: 0
- Competing variants: 0
- Auto-resolved: 0
- Info entries: 1 (provenance / no-conflict confirmation)
- See `/home/drdreadknee/mapclass/.planning/INGEST-CONFLICTS.md` for the full report.

## Cycle detection

- Cross-ref graph traversed; no cycles detected.
- README references the DOC (and several scripts and data paths); the DOC references only external arxiv URLs.
- Max traversal depth used: 1.

## Key locked commitments (from the SPEC, for downstream planners)

- Dense pixel-level prediction; regional scale 100–2000 km; two outputs (land cover, topography).
- Land cover: 9-class canonical taxonomy.
- Topography: 3-class slope-derived taxonomy from Copernicus DEM GLO-30.
- Backbone (primary): SigLIP from PaliGemma-3B, LoRA on attention layers (q_proj/k_proj/v_proj/out_proj) only, Gemma frozen, multi-modal projector trainable.
- Label sources: ESA WorldCover (s3://esa-worldcover), Copernicus DEM GLO-30 (s3://copernicus-dem-30m), David Rumsey Map Collection via LUNA API.
- Georeferencing: Allmaps/Georeferencer for registered maps; PaliGemma-driven semi-automatic registration for unregistered maps; MapWarper/QGIS manual fallback; output GeoTIFF in EPSG:4326.
- Evaluation: joint per-pixel NLL on held-out synthetic maps.
- Deliverable: HuggingFace upload; downstream polygonisation is out of scope.
- Phasing: 4 sequenced phases with PaliGemma threading through all of them.

## Pointers

- `/home/drdreadknee/mapclass/.planning/intel/decisions.md`
- `/home/drdreadknee/mapclass/.planning/intel/requirements.md`
- `/home/drdreadknee/mapclass/.planning/intel/constraints.md`
- `/home/drdreadknee/mapclass/.planning/intel/context.md`
- `/home/drdreadknee/mapclass/.planning/INGEST-CONFLICTS.md`
