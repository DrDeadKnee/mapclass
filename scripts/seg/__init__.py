"""
Dense semantic segmentation pipeline package (Phase 3).

Package marker so ``from seg.X import ...`` resolves correctly under
pytest.ini ``pythonpath = scripts``.

Modules (created in plans 03-02 through 03-05):
  seg.backbones  — Backbone protocol + SigLIP / DINOv2 / Swin implementations (D-05)
  seg.decoder    — UPerNet-style PPM+FPN conv decoder (D-01)
  seg.heads      — two thin 1×1-conv task heads (9-class LC, 3-class topo) (D-02)
  seg.model      — SegModel assembly: backbone + decoder + heads; Variant A/B (D-03a)
  seg.recursive  — coarse-to-fine inference orchestrator (D-03/D-04)
  seg.dataset    — PyramidDataset over Phase-2 pyramid tree, train-only (D-06)
"""

__all__: list[str] = []
