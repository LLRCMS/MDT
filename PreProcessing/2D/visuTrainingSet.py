import os
import sys
import random
import math
import re
import time
import numpy as np
import hgcal2D as hg

# Root directory of the project\n",
ROOT_DIR = os.path.abspath("../../")
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn import utils

import matplotlib as plt

# Visualize the first images
nImagesToShow = 20

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


# GG The place ?
def get_ax(rows=1, cols=1, size=8):
     """Return a Matplotlib Axes array to be used in
     all visualizations in the notebook. Provide a
     central point to control graph sizes.
     
     Change the default size attribute to control the size
     of rendered images
     """
     _, ax = plt.pyplot.subplots(rows, cols, figsize=(size*cols, size*rows))
     return ax

config = hg.Hgcal2DConfig()
config.display()

# Training dataset
dataset_train = hg.Hgcal2DDataset()
dataset_train.load_hgcal2D('train.obj', nImagesToShow, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])

dataset_train.prepare()
    
# Validation dataset
dataset_val = hg.Hgcal2DDataset()
dataset_val.load_hgcal2D('train.obj', nImagesToShow, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])

dataset_val.prepare()

# Chose dataset
dataset = dataset_val

image_ids = dataset.image_ids[0:]
for image_id in image_ids:
  image = dataset.load_image(image_id)
  mask, class_ids = dataset.load_mask(image_id)
  print("classes ", class_ids, dataset.class_names)
  visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

