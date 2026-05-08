# VLMs Can't Read Maps, but GeoViLMs do just fine: A system architecture for stylized map understanding

> **Scope.** This document is **milestone-2 paper framing**, opportunistic and deferred
> from v1. The locked v1 deliverable is a model handoff (small-VL-backbone checkpoint +
> inference module) for a downstream hex-grid app, with a PaliGemma-3B benchmark variant
> trained alongside as a bellwether. v1 scope, constraints, and out-of-scope items live
> in [.planning/PROJECT.md](./.planning/PROJECT.md) — that is the source of truth.
> Where this mockup and PROJECT.md disagree, PROJECT.md wins.

### Practical thoughts
I haven't had much success getting a VLM to read a map. Given the inherent variance in the 
map domain, I'm not sure that a massive model will be necessary. 

The practical outcome should be a model that takes a map and "understands" it in a few important ways. 
Most important is the ability to determine terrain types, followed by features. Ideally it can 
take as input a satelite image, road map, historical map, or stylized fantasy map, and infer 
whether any given pixel is best described as water, land, road, river, forest, field, city, etc.

Google has some relevant foundation models for this task (primarily trained on satellite
data), but to our knowledge none handle stylized maps well. Contrastive vision-language
models like CLIP and SigLIP appear to show poor zero-shot transfer to stylized cartography;
benchmarking this directly as part of the failure analysis is a **milestone-2 paper
item**, not part of v1.

I had some ideas on methods earlier, but getting started was a headache - need to try a different
approach. The order of training, the model type, etc. need to be clearly though through. Incorporation
of rotationally invariant OCR needs to take place also at an early stage. I'd like to use this project
to achieve a few outcomes. In order:

1. A cheap inference model to incorporate into an app
2. A way for me to learn about and play with multi-modal modelling
3. A way for me to play with [dynamic LRP](https://arxiv.org/pdf/2512.07010)

### Academic Motivation
For thousands of years, humans have used maps as a greatly compressed representation of the
2d surface of the world around them. Maps can contain a combination of text labels (which
may be rotated and which may appear at varying distances from the objects they describe), 
artistic or iconic representions of important places or areas, political boundaries, 
longitude and latitude lines, terrain type representations, and in particular they may emphasize
important connective routes such as roads and rivers. 

Of equal importance is the ability of humans to "fill in the gaps" in the visual
geophysically informed priors that are usually learned unconsiously. 

Due to the number of modalities involved in reading a map, and due to their lack
of relevant underlying priors, current vision models alone are ill-suited to completing
this task.

If milestone 2 happens, the candidate contributions narrow to two — the original
"fully open-source dataset" contribution is **dropped** because three of the four
upstream sources (David Rumsey, Copernicus DEM, Azgaar) have unverified
redistribution terms and v1 explicitly punts on dataset redistribution (ships
weights + code only). The dataset is reframed as "trained on these sources, here's
how to reproduce", not as a redistributable artifact.

**Contribution 1 — Mechanistic failure analysis.** Prior work has already documented
that VLMs struggle with cartographic question answering
([MAPWise](https://arxiv.org/abs/2409.00255),
[MapIQ](https://arxiv.org/html/2507.11625v1)), so the *observation* of failure is
not itself a contribution. The proposed contribution is to use
[dynamic LRP](https://arxiv.org/pdf/2512.07010) to localise where in the model
these failures originate, rather than only reporting end-to-end accuracy. Paired
with benchmarking on our test splits to confirm the failure modes generalise.

**Contribution 2 — GeoViLM system architecture.** A multi-component system that
combines a (small) vision-language backbone, a rotationally-invariant OCR module
for curved/rotated map text, and dense segmentation heads for land cover and
topography, trained with per-source class-conditional loss weighting and an
auto-georef bootstrap loop on the historical source. The contribution is the
system design and the empirical demonstration that its components are jointly
necessary on stylized maps — not a novel fusion primitive. The v1 ship target
runs on a single CPU host or 4–8 GB consumer GPU; a PaliGemma-3B variant of the
same stack is trained alongside as a benchmark/bellwether to bound the small
backbone's NLL on the held-out splits. Component ablations (backbone-only vs.
+OCR vs. +seg-heads vs. full system) are deferred to this milestone, not run in
v1. Suggested applications: optimal routing, game development, treasure hunting,
and travel guides.

### Target Images
I think we should have some target images. A good one would be a contrast bewtween a few popular
fantasy maps for the oriignal model (with attributions), and then a sort-of progression of the
activations over time.

I think we should target:
1. Middle-Earth (tolkein)
2. Westeros (George R.R. Martin)
3. [Circle of the World](https://www.reddit.com/r/TheFirstLaw/comments/pl6c7w/updated_first_law_map_should_now_feature_every/) (Joe Abercrombie)
4. Warhammer: The Old World (Games Workshop)
