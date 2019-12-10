#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# GG TODO : refactoring of all functions in the BatchGenerator class

import numpy as np
import os
from collections import OrderedDict
import pandas as pd
import pickle
import time
import subprocess
import utils.dataloader_utils as dutils
# GG Needed to draw ellispes 
import cv2

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.transforms.spatial_transforms import MirrorTransform as Mirror
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading import SingleThreadedAugmenter
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform
# from batchgenerators.transforms.utility_transforms import ConvertSegToBoundingBoxCoordinates
from batchgenerators.transforms.utility_transforms import NullOperation


def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """

    all_data = load_dataset(cf, logger)

    batch_gen = {}

    batch_gen['train'] = create_data_gen_pipeline(
        all_data, cf=cf, nbrSamples =  cf.num_train_batches * cf.batch_size, do_aug=False)
    batch_gen['val_sampling'] = create_data_gen_pipeline( 
        all_data, cf=cf, nbrSamples = cf.num_val_batches, do_aug=False)

    batch_gen['n_val'] = cf.num_val_batches

    return batch_gen


def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    If cf.hold_out_test_set is True, gets the data from an external folder instead.
    """
    print("Warning: 'get_test_generator' not implemented")

    return 



def load_dataset(cf, logger, subset_ixs=None):
    """
    loads the dataset. if deployed in cloud also copies and unpacks the data to the working directory.
    :param subset_ixs: subset indices to be loaded from the dataset. used e.g. for testing to only load the test folds.
    :return: data: dictionary with one entry per patient (in this case per patient-breast, since they are treated as
    individual images for training) each entry is a dictionary containing respective meta-info as well as paths to the preprocessed
    numpy arrays to be loaded during batch-generation
    """
    if cf.debug_data_loader:
        print(">>> load_dataset")
        print ("  cf.pp_data_path, cf.input_df_name", 
                   cf.pp_data_path, cf.input_df_name )

    if cf.server_env:
        print("Error: mode not available")
        exit()

    fname = cf.pp_data_path
    file_ = open( fname, 'rb')
    # GG reading python2.x pickle file
    object_ = pickle.load(file_, encoding="latin1")
    (images_, bboxes_, labels_, labelIDs_, evIDs_, 
     ellipses_, hEnergies_, evInfos_) = object_
    
 
    if ( len(images_) < cf.batch_size * cf.num_train_batches ):
      logger.info("Warning: number of read events not suffisient {} read / {} required"
                  .format(len(images_),  cf.batch_size * cf.num_train_batches ) )
    
    if cf.server_env:
        print("Error: mode not available")
        exit()

    data = OrderedDict()
    data['images'] = images_
    data['labels'] = labelIDs_
    # Not used
    data['evID']   = evIDs_ 
    data['ellipses'] = ellipses_
    # Constant image size
    data['xSize'] = images_[0].shape[0]
    data['ySize'] = images_[0].shape[1]

    return data

def create_data_gen_pipeline(patient_data, cf, nbrSamples=0, do_aug=True):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(patient_data, batch_size=cf.batch_size, nbrSamples=nbrSamples, 
                              cf=cf)

    # add transformations to pipeline.
    my_transforms = []
    if do_aug:
        mirror_transform = Mirror(axes=np.arange(2, cf.dim+2, 1))
        my_transforms.append(mirror_transform)
        spatial_transform = SpatialTransform(patch_size=cf.patch_size[:cf.dim],
                                             patch_center_dist_from_border=cf.da_kwargs['rand_crop_dist'],
                                             do_elastic_deform=cf.da_kwargs['do_elastic_deform'],
                                             alpha=cf.da_kwargs['alpha'], sigma=cf.da_kwargs['sigma'],
                                             do_rotation=cf.da_kwargs['do_rotation'], angle_x=cf.da_kwargs['angle_x'],
                                             angle_y=cf.da_kwargs['angle_y'], angle_z=cf.da_kwargs['angle_z'],
                                             do_scale=cf.da_kwargs['do_scale'], scale=cf.da_kwargs['scale'],
                                             random_crop=cf.da_kwargs['random_crop'])

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(CenterCropTransform(crop_size=cf.patch_size[:cf.dim]))

    my_transforms.append(NullOperation(cf.dim))
    # GG TODO : test if "my_transforms = []" is sufficient
    my_transforms = []

    all_transforms = Compose(my_transforms)
   
    multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    # multithreaded_generator = MultiThreadedAugmenter(data_gen, all_transforms, num_processes=cf.n_workers, seeds=range(cf.n_workers))
    return multithreaded_generator

class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """
    def __init__(self, data, batch_size, nbrSamples, cf):
        # Nbr of event in the data-set
        super(BatchGenerator, self).__init__(data, batch_size)

        self.cf = cf
        # Nbr of batch already processed
        self.nbrOfBatchProcessed = 0
        self.shuffleEvIdx = np.zeros( nbrSamples, dtype=np.int32)
        self.nbrOfSamples = nbrSamples
        self.nbrOfBatch = int( nbrSamples  / batch_size )

    def generate_train_batch(self):
        """
        return values required by MRCNN model
        'data'       batch_images     np.shape(bs, c, xSize, ySize) 
        'roi_labels' batch_gt_classes [ bs, np.shape(nObjs) ],  int
        'bb_target'  batch_gt_bboxes  [ bs, np.shape(nObjs, 4) ]
        'roi_masks'  batch_gt_masks   [ bs, np.shape(nObjs, c, xSize, ySize) ]
        
        For post processing
        'pid'        batch_pid        [ bs ]
        
        With
        bs : batch size
        c : number of image channels
        xSize, ySize : image size
        nObjs : number of objects in the images
        """

        if self.cf.debug_generate_train_batch :
          print(">> generate_train_batch self._data", self.__dict__['_data'].keys() )

        # Shuffle at each epoch
        if ( self.nbrOfBatchProcessed == 0):
            self.shuffleEvIdx = np.array( 
                [i for i in range(self.nbrOfSamples) ], dtype=np.int32
            )  
            if not self.cf.debug_deactivate_shuffling:
              np.random.shuffle( self.shuffleEvIdx  )
              print("  Shuffling ... ",);

        images = self._data['images']
        ellipses = self._data['ellipses']
        labelsIDs = self._data['labels']

        if self.cf.debug_generate_train_batch :
          print("  len(images), len(ellipses)", len(images), len(ellipses) )

        bSize = self.batch_size
        c = self.cf.n_channels
        xSize = self._data['xSize']
        ySize = self._data['ySize']

        # Allocate/type return values
        batch_images     = np.zeros( (bSize, c, xSize, ySize) )
        # Image index (used in post-processing)
        batch_pid = []
        batch_gt_classes = []
        batch_gt_bboxes  = []
        batch_gt_masks   = []

        for b in range(self.batch_size) :
          Idx = self.batch_size * self.nbrOfBatchProcessed + b
          # Debug
          if self.cf.debug_generate_train_batch :
            print("  Idx, self.shuffleEvIdx[ b ]", Idx, self.shuffleEvIdx[ b ])
          evIdx = self.shuffleEvIdx[ Idx ]

          # Images
          # TODO : 
          #   - no permutation in mrcnn 
          #   - image with float
          batch_images[b, 0, :, :] = images[ evIdx ][:,:]
          batch_pid.append( evIdx )
          # GT classes
          nObjs = len( labelsIDs[ evIdx ] )
          if self.cf.debug_generate_train_batch :
            print("  nObjs ", nObjs )
          # Bboxes and Masks allocation
          gt_bboxes = np.zeros( (nObjs, 4), dtype=np.float )
          gt_masks  = np.zeros( (nObjs, c, xSize, ySize) )
          # GT Masks & Boxes
          for k in range(nObjs):
            ( bboxes, orig, axes, angles ) =   ellipses[ evIdx ][k]
            #
            # Boxes
            bb  = np.around( bboxes)
            # bboxes [[( xmin,xmax,ymin,ymax)]] ->  [ xmin,ymin,xmax,ymax)
            gt_bboxes[ k, :] = np.array( [ bb[0], bb[2], bb[1], bb [3] ])
            #
            # Masks
            #
            # Allocate mask for opencv
            mask_cv = np.zeros( (xSize, ySize, 3 ), dtype = "uint8")
            mask = np.zeros( (1, xSize, ySize ), dtype = "uint8")
            orig=  np.around( orig )
            axes=  np.around( axes)
            # Transpose
            orig= [ orig[1], orig[0] ]
            axes = [  axes[1], axes[0] ]
            angles =  np.around( ( -  angles)  * 180  / np.pi )           
            cv2.ellipse( mask_cv, (orig[0], orig[1]), (axes[0], axes[1]), angles[0] , 
                         0, 360, (255,255,255), -1)
            ind =( mask_cv[:,:,0] == 255)
            mask[0, ind] = 1
            gt_masks[ k, 0, :, :] = mask

          # Batch update
          batch_gt_classes.append( np.array( labelsIDs[ evIdx ], dtype=np.int) )
          batch_gt_masks.append( gt_masks )
          batch_gt_bboxes.append( gt_bboxes )

        # Update values which control the shuffling
        self.nbrOfBatchProcessed += self.batch_size
        if( self.nbrOfBatchProcessed >= self.nbrOfBatch): self.nbrOfBatchProcessed = 0

        return { 'data': batch_images, 'pid': batch_pid, 'roi_masks': batch_gt_masks, 
                 'roi_labels': batch_gt_classes, 'bb_target': batch_gt_bboxes }
