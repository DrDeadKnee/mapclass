# MapClass

Detecting and classifying terrain regions in map images, targeting grand strategy style campaigns at 100–2000 km scale. Multiple map image styles should be supported, from illustrated grand strategy aesthetics (EU4/CK3/HoI4 style) to satellite-derived imagery.

### Human Testing
A lot of the testing is intended to be done through the notebooks. If hosted in a google cloud VM:
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
