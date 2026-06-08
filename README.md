# MapClass

Trying to segment and classify hand-drawn and other maps. This is challenging because:
- Mix of images, text, and contours
- Inference relies on context from arbitrary distance
- Priors for topography propbability change are relevant
- Multiple artistic styles might be relevant

## Data
Data to start is in GCP. The primary dataset is taken from Ramsey's maps:
```gs://mapclass-training-northeast1/data/historical```

Some cartoon hex images are present as well:
```gs://mapclass-training-northeast1/data/toons```


And some satelite data:
```gs://mapclass-training-northeast1/data/satellite```


## Plan
Focus on historical map data to start. See how mapreader looks.
