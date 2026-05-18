# MapClass

This is an experimental modelling project intended to define which parts of a map image correspond to which
features. The goal is to use [dynamic LRP attribution](https://arxiv.org/pdf/2512.07010) to identify
which sections of an image correspond to which queries, effectively using pre-trained VLMs to create
pixel-level labels.

Artistic maps, such as Historical, Fantasy, and Game maps, specifically, are the domain to explore first.


### DataSet
The datasets are stored in a gcs bucket at gs://mapclass-training-northeast1/data/
Metadata is stored locally. Each data is simply a collection of images.

The plan for each dataset is simply to have both metadata (similar to the manifest for rumsey)
as well as an copy of assets in GCP (for stability). For now the datasets will be seperate, but
can be combined at a later date.

1. Historical Maps. This is David Rumsey's data. There is a URL to the public image for each image in the 
manifest. #TODO Each image should be copied to gcs for stability, and the link should be added there as well
2. Toons. These aren't maps individually, but are cartoon assets that can be used to create maps. 
3. Fantasy Maps. This is a smaller category that should include maps from Tolkein, 
Robert Jordan, Joe Abercrombie, George R.R. Martin, Warhammer, etc.
4. MapMaker Maps. This will be a few maps made using standard editors like Azgaar's and Inkarnate #Human TODO
5. Satelite Data. This isn't map data, but because of google's label availability it can help benchmark.
Aligment with Rumsey data might also be possible.


### Models
Main-task models. Models are stored in a gcs bucket at gs://mapclass-training-northeast1/models/
They are mostly copied from huggingface. They're stored again to reduce authentication overhead.

Models in-scope are smaller vision or VLM models. Dynamic LRP will probably perform worse as the model
size increases, not to mention additional overhead. Paligemma 3B is the biggest model we will use
for this purpose.

Candidate model list:
- Paligemma3B
- Florence-2
- SAM 3
- VGG16
- ViT-b-16
- SigLIP-2-So400m-base14-384

#### Human Testing (Reference)
This project is intended to take place on remove gcp VMs. Human will look at predictions in a jupyter notebook
for phase 1, and move on to an open source labeller for phase 2.

Once a VM is started and the repo is installed, these are the steps to follow:

1. Start Server
```
# If env not defined, install python and jupyterenv
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name mapclass --display-name "mapclass (.venv)"

# Launch Server
.venv/bin/jupyter lab --no-browser --port 8888 --ip 127.0.0.1

```

2. Establish SSH on local
```
gcloud compute ssh <vm_name> -- -N -L 8888:127.0.0.1:8888
```

3. Visit in local browser
```
http://localhost:8888
127.0.0.1:8888
```
