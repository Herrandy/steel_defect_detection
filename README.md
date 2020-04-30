## Steel surface defect detection
This repository contains a neural network model for defect classification and segmentation.
The network is trained and evaluated using the [Severstal](https://www.kaggle.com/c/severstal-steel-defect-detection/overview) image dataset. 

### Dependencies
1) Linux (tested on Ubuntu 18.04)
2) Download the [image dataset.](https://www.kaggle.com/c/severstal-steel-defect-detection/data)
3) Install the following pip packages:
```bash
$ pip install --upgrade pip
$ pip install tensorflow
$ pip install keras
$ pip install segmentation-models
```
### Try it out!
```
from defect_detect import visualize_data
from defect_detect import train
from defect_detect import eval

# show example images with defects
visualize_data(<path_to_data>)

# train the net
train(<path_to_data>, <save_file>)

# evaluate model
eval(<path_to_data>, <saved_model_weights>)
```
### Example output of the network:
<img width="600" height="500" src="https://i.imgur.com/YLkayPI.png">




