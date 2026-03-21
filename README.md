# MapClass

Taking map images and classifying all of the terrain types. Multiple map image styles should be incorporated,
from illustrated black-and-white to satelite images. 


## Project Plan
This project will progress in phases:
1. Build a dataset containing semantic content-pixel pairs. These will primarily come from:
    - Map illustrators, such as inkarnate
    - Google Satelite image classification
2. Clone Clip and create a training pipeline for fine-tuning
3. Tune a final model and upload to huggingface.
