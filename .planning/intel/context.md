# Context

Running notes from DOC-classified sources. Context only — no locked decisions.

---

## Topic: Why context-aware segmentation

source: /home/drdreadknee/mapclass/notes/architectural_references.md

> Terrain classification cannot be done in isolation — the identity of a region depends as much on its surrounding context as on its own appearance. A mountain range reads differently next to a desert than next to a forest. Satellite imagery reinforces this: geophysical co-occurrence statistics (oceans border coasts, coasts border lowlands, etc.) are strong priors that a good model should implicitly learn.

This motivates both the dense (rather than tile-classification) approach and the coarse-to-fine recursive design.

---

## Topic: Candidate backbone — Recursive coarse-to-fine (preferred path in the DOC)

source: /home/drdreadknee/mapclass/notes/architectural_references.md

Divide the map into coarse tiles, obtain per-class probability distributions over each tile using a pre-trained backbone (DINOv2 or Swin Transformer), then use those distributions as priors when segmenting at the next finer scale. Repeat to target resolution.

Rationale:
- Captures broad spatial context naturally through the coarse pass.
- Leverages existing pre-trained representations — no need to learn low-level features from scratch.
- More compute-efficient than training a large-kernel model from scratch.
- DINOv2 and Swin Transformer are available frozen with strong dense-prediction transfer, putting most of the training budget on the task heads.

Note: the README SPEC locks the *primary* backbone as the Phase-1 fine-tuned SigLIP encoder. DINOv2 and Swin remain benchmark alternatives, consistent with this context note.

---

## Topic: Candidate backbone — Large first-kernel ConvNet (stretch goal)

source: /home/drdreadknee/mapclass/notes/architectural_references.md

A novel ConvNet with a large first convolution kernel (e.g. 31×31 or larger) to capture broad receptive field from layer one, providing spatial context without multi-scale passes.

Why benchmark it:
- Recent work (RepLKNet, SLaK, ConvNeXt) shows large-kernel CNNs close most of the performance gap with attention-based models on dense prediction tasks.
- Architecturally simpler — single forward pass, no multi-scale orchestration.
- May generalise better to the illustrated map domain where pre-trained satellite/natural-image features are less transferable.

Main concern: training from scratch needs substantially more data and compute. The README correctly classifies this as a stretch goal.

---

## Topic: Planned experiment — Progressive backbone unfreezing for domain-gap analysis

source: /home/drdreadknee/mapclass/notes/architectural_references.md

Initial model uses a fully frozen backbone. If illustrated-map performance plateaus:
1. Progressively unfreeze layers from the top of the backbone downward (last block first, then second-to-last, etc.) and track segmentation accuracy at each stage.
2. Run attribution analysis (GradCAM or attention rollout) at each unfreezing stage to identify which layers encode illustrated-map-relevant features vs. natural-image features.

Purpose: localise the domain gap inside the network rather than treating it as monolithic. Informs whether the gap is in low-level texture features, mid-level spatial composition, or high-level semantics.

---

## Topic: Reference reading list

source: /home/drdreadknee/mapclass/notes/architectural_references.md

Coarse-to-fine / hierarchical segmentation:
- Swin Transformer — https://arxiv.org/abs/2103.14030 — hierarchical ViT with linear complexity; dominant backbone for dense prediction.
- AerialFormer — https://arxiv.org/abs/2306.06842 — hierarchical coarse-to-fine on aerial/remote-sensing data; most directly relevant.
- DINOv2 — https://arxiv.org/abs/2304.07193 — self-supervised ViT; strong dense-prediction transfer with frozen backbone.

Large-kernel CNNs:
- Large Kernel Matters (Global Convolutional Network) — https://arxiv.org/abs/1703.02719 — case for large kernels over stacked small filters.
- RepLKNet — https://arxiv.org/abs/2203.06717 — 31×31 depthwise convolutions matching ViT performance.
- SLaK — https://arxiv.org/abs/2207.03620 — sparse factorised large kernels matching Swin Transformer.
- ConvNeXt — https://arxiv.org/abs/2201.03545 — modernised ResNet rivalling Swin on classification and segmentation.
