# Architectural References

## Context and Design Decisions

The core architectural challenge for MapClass is that **terrain classification cannot be done in isolation** — the identity of a region depends as much on its surrounding context as on its own appearance. A mountain range reads differently next to a desert than next to a forest. Satellite imagery reinforces this: geophysical co-occurrence statistics (oceans border coasts, coasts border lowlands, etc.) are strong priors that a good model should implicitly learn.

### Two candidate approaches

**1. Recursive coarse-to-fine segmentation (preferred)**

Divide the map into coarse tiles, obtain per-class probability distributions over each tile using a pre-trained backbone (e.g. DINOv2 or Swin Transformer), then use those distributions as priors when segmenting at the next finer scale. Repeat until the desired resolution is reached.

Why preferred:
- Captures broad spatial context naturally through the coarse pass
- Leverages existing pre-trained representations — no need to learn low-level features from scratch
- Far more compute-efficient than training a large-kernel model from scratch
- DINOv2 and Swin Transformer are available frozen with strong dense-prediction transfer, making the training budget go entirely toward the task heads

**2. Large first-kernel ConvNet (benchmark alternative)**

A novel convnet with a large first convolution kernel (e.g. 31×31 or larger) to capture broad receptive field from layer one, providing spatial context without needing multi-scale passes.

Why it's still worth benchmarking:
- Recent work (RepLKNet, SLaK, ConvNeXt) shows large-kernel CNNs close most of the performance gap with attention-based models on dense prediction tasks
- Architecturally simpler — single forward pass, no multi-scale orchestration
- May generalise better to the illustrated map domain where pre-trained satellite/natural-image features are less transferable

The main concern: training a large-kernel convnet from scratch requires substantially more data and compute. Given the limited compute budget, this is a stretch goal rather than the primary path.

---

## Planned Experiments

**Domain gap analysis — progressive backbone unfreezing**

The initial model uses a fully frozen backbone. If performance on illustrated maps plateaus, a planned follow-up experiment is to progressively unfreeze layers from the top of the backbone downward (i.e. unfreeze the last block first, then the second-to-last, etc.) and track segmentation accuracy at each stage. Alongside this, attribution analysis (e.g. GradCAM or attention rollout) at each unfreezing stage will show which layers encode features relevant to illustrated-map semantics vs. natural-image features. This localises the domain gap within the network rather than treating it as monolithic.

---

## Reference Reading

**Coarse-to-fine / hierarchical segmentation**
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — hierarchical ViT with linear complexity; the dominant backbone for dense prediction tasks
- [AerialFormer: Multi-resolution Transformer for Aerial Image Segmentation](https://arxiv.org/abs/2306.06842) — hierarchical coarse-to-fine approach benchmarked on aerial/remote sensing datasets; most directly relevant to this project
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — self-supervised ViT backbone; strong dense-prediction transfer with a frozen backbone, primary candidate for this project

**Large-kernel CNNs**
- [Large Kernel Matters: Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719) — argues the case for large kernels over stacked small filters in dense prediction
- [RepLKNet: Scaling Up Your Kernels to 31x31](https://arxiv.org/abs/2203.06717) — pure CNN with 31×31 depthwise convolutions matching ViT performance
- [SLaK: More ConvNets in the 2020s, Scaling Kernels Beyond 51×51](https://arxiv.org/abs/2207.03620) — sparse factorized large kernels matching Swin Transformer
- [ConvNeXt: A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545) — modernised ResNet rivalling Swin on classification and segmentation benchmarks
