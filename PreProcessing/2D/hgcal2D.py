"""
Mask R-CNN
Configurations and data loading code for the HGCAL2D dataset.
This is base on the Shapes example

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import pickle

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

IMAGE = {'xSize':256, 'ySize':256 }

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


class Hgcal2DConfig(Config):
    """Configuration for training on the toy hgcal2D dataset.
    Derives from the base Config class and overrides values specific
    to the toy hgcal2D dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hgcal2D"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = IMAGE['xSize']
    IMAGE_MAX_DIM = IMAGE['ySize']

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels
    # Ratios of anchors at each cell (width/height)
    # (2) RPN_ANCHOR_RATIOS = [0.25, 0.5, 1 ]
    # (3) RPN_ANCHOR_RATIOS = [1, 2, 4 ]
    # (4) RPN_ANCHOR_RATIOS = [0.25, 0.5, 1 ]
    #     RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # (5) = (4)
    # BACKBONE_STRIDES = [2, 4, 8, 16, 64]
    # (6)
    # BACKBONE_STRIDES = [2, 4, 8, 16, 32]
    # RPN_ANCHOR_SCALES = (2, 4, 8, 16, 32)
    # (7)
    # BACKBONE_STRIDES = [2, 4, 8, 16, 64]
    # RPN_ANCHOR_SCALES = (2, 4, 8, 16, 64)
    # Training 1000 ev
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64)
    # GG
    # Don't work
    # - RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64, 128)
    # Its seemx that RPN_ANCHOR_RATIOS_min * RPN_ANCHOR_RATIOS_min >= 2

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 5000

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 50

    READ_EVENT_TRAIN = 5000
    READ_EVENT_VALIDATION = 50

class Hgcal2DDataset(utils.Dataset):
    """ Generate the images and mask from a pickle file
    images_   [:, :, 3], np.uint8
    labelIDs_ [[]] np.int32
    bboxes    [[( xmin,xmax,ymin,ymax)]] 
    labelIDs  [[]] np.array, np.int32
    """

    def load_hgcal2D(self, fname, count, height, width):
        """ Load all the images
        fname: pickle file name
        count: number of images to generate.
        height, width: the size of the generated images.
        Load the images which are generated from gray images (file)
        """
        # Add classes
        self.add_class("hgcal2D", 1, "EM")
        self.add_class("hgcal2D", 2, "Pion")

        # Read file
        file_ = open( fname, 'rb')
        # GG reading python 2 pickle file
        object_ = pickle.load(file_, encoding="latin1")
        # Several boxes per images and several labels per image
        # bboxes_[[]], labels_[[]], labelsID_[[]]
        # images_, bboxes_, labels_, labelIDs_, evIDs_, ellipses_,  hEnergies= object_
        (images_, bboxes_, labels_, labelIDs_, evIDs_, ellipses_, hEnergies_, evInfos_) = object_ 

        if ( len(images_) < count):
            print ("Error : # of images not sufficient ",  len(images_), "<", count)

        for i in range(count):
           # print ("Ellipses", ellipses_[i])
           bboxes = []
           for e in  ellipses_[i]:
             (bb, orig, axes, angles) = e
             bboxes.append(np.round(bb))
           
           self.add_image("hgcal2D", image_id=i, path=None,
                           width=width, height=height,
                           gray_image = images_[i],
                           bboxes = bboxes, labelIds=labelIDs_[i] )
           self.image_info[i]["ellipses"] = ellipses_[i]
        print ("<Hgcal2DDataset.load_image> done,", count, "images loaded")

    def load_image(self, image_id):
        """ 
        Generate an RGB image from a gray one
        """
        info = self.image_info[image_id]
        grayImg = info['gray_image']
        # grayImg = np.flip(grayImg, axis=0)
        image = np.empty([ info['height'], info['width'], 3 ], dtype=np.uint8)
        
        image[:,:,0] = grayImg
        image[:,:,1] = grayImg
        image[:,:,2] = grayImg
        return image

    def image_reference(self, image_id):
        """Return the hgcal2D data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "hgcal2D":
            return info["hgcal2D"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for bboxes of the image.
        """
        info = self.image_info[image_id]
        bboxes = info['bboxes']
        # Shoul be np.array, np.int32
        labelIds__ = info['labelIds']
        ellipses = info["ellipses"]
        count = len(bboxes)
        # print ("len(bboxes) ", bboxes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        """
        Generate mask
        """
        """
        for bb in range(len(bboxes)):
            # print "YYY ??? bb", bb
            (xmin, xmax, ymin, ymax) = bboxes[bb]
            # print "YYY ??? min/max", xmin, xmax+1, ymin, ymax+1
            mask[xmin:xmax+1, ymin:ymax+1, bb ] = 1
        """
        for k in range(count):
            mask_ = np.zeros((info['height'], info['width'], 3), dtype = "uint8")
            ( bboxes  , orig, axes, angles ) =   ellipses[k]
            bbox=  np.around( bboxes)
            orig=  np.around( orig )
            axes=  np.around( axes)
            # Transpose
            # bbox bbox = [ bbox[2], bbox[3], bbox[0], bbox[1] ]
            orig= [ orig[1], orig[0] ]
            axes = [  axes[1], axes[0] ]
            angles =  np.around( ( -  angles)  * 180  / np.pi )
           # cv2.ellipse(mask[:,:,k], (orig[0], orig[1]), (axes[0], axes[1]), angles[0] , 0, 360, 1, -1)
            cv2.ellipse(mask_, (orig[0], orig[1]), (axes[0], axes[1]), angles[0] , 0, 360, (255,255,255), -1)
            #cv2.ellipse(img, (50, 50), (80, 20), 5, 0, 360, (0,0,255), -1)

            # ind = np.where ( mask_[:,:] == 255, 1, 0)
            
            ind =( mask_[:,:,0] == 255)
            mask[ind, k] = 1
            """
            for i in range( info['height']):
              for j in range( info['width']):
                if (mask_[i,j, 0] == 255):
                  mask[i,j,k] = 1
            """
        ## ???
        ## Pb in data input file 
        # input type:
        #   [ np.array([1], dtype=np.int32), np.array([2], dtype=np.int32) ]
        # must be transformed in 
        ##   array([1, 2], dtype=int32)
        labelIds = np.array( labelIds__, dtype=np.int32) 
        labelIds = labelIds.ravel() 

        # GG Invalid ???
        # class_ids = np.array([self.class_names.index(s[0]) for s in hgcal2D])
        # print "GG mask :", mask.shape
        # print "GG label:", "XX", labelIds, labelIds[0]
        return mask, labelIds

