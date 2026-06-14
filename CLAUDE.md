# MapClass

This repo is intended to house code to train and characterize
a "Map Model". This modelis  intended to view a map image and use it to create
a terrain map, where the terain characteristics are known
at each point. Important terrain types include cities, forests,
fields, lakes, and mountains.

The predominant map type ingested is likely to be hand-drawn,
sparsely-populated maps. These maps are characterized by
text as paramount for describing features and regions,
the presence of images (such as sea monsters) to lend
character to and assist in identifaction of regions, 
lack of explicit detail, leaving "fill in the gaps" for
forested, grassy, and hilly regions to the imagination,
and illustrations of single examples that are expexted to
be generalized (e.g. 3 pine trees = forest, 3 hills = 
30 km with 100 hills). They are also characterized by
rotated text, and text with different sizes, which 
may be a varing distance from the applicable boundaries.

The model needs to be able to classify arbitrary regions.
If a grid is drawn over the map, each cell of the grid
should be assigned a cover type and a topography type,
drawn from a somewhat limited set. The suggested method
is pixel-level labelling and voting for cells, possibly with
some post-processing.

Important considerations:
- Prior knowledge of topography is assumed to read a map. This can be learned using satelite data
- Historical data provides a rich supply of hand-drawn maps that could be aligned to known data for labelling
- Sets of hand-drawn tiles would also serve a source of pre-labelled data
- Text on the maps is very important
- Recognition of images as decorations rather than features is important
- Many map images will have borders, outside of which no classification should take place
- There is a lot more high-quality labelled satelite data than there is high-quality labelled drawn map data
- Would prefer a single end-to-end model than something like a bunch of bounding boxes individually labelled and stiched together
- Need to be able to train on modest architectures

Corolaries:
- Pre-training on satelite data is an attractive idea, but incorporating info from words seems tough
- The memory on the current device matters a lot for model candidates and/or batch sizes
- Presence and size of GPU is likely to be minimal


### Assets
Some relevant data has been assembled. It should be checked and
probably augmented. 

First is satelite data which is assembled 
from overlapping pyramids of EDA data. Pixel-level topography:
- gs://mapclass-training-northeast1/data/satellite

Second, a historical map colleciton has been fetched:
- gs://mapclass-training-northeast1/data/historical

Third, hex-images with labels do exist:
- gs://mapclass-training-northeast1/data/toons


### Prior work
This needs digging into. Some relevant vision models:
- Florence 2
- Capsule Networks
- DPText-DETR
- Oriented-DETR / OrientedFormer
- Grouning DINO


### Needs
- Method to easily inspect datasets
- Confirmation of datasets
- Surveil of architectures
- Training pipeline
- Comparison with existing models
