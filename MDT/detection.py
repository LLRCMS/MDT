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

import os
import numpy as np
import torch
from scipy.stats import norm
from collections import OrderedDict
from multiprocessing import Pool
import pickle
import pandas as pd
import matplotlib.pyplot as plt

import utils.model_utils as mutils
import utils.plotUtil as pltUtil
import MPUtils.utils as MPUtl

def plotWindow( gt_labels, gt_bboxes, gt_masks, pred_labels, pred_scores, pred_bboxes, pred_masks, img, evID):

    fig, axs = plt.subplots(nrows=1, ncols=1)

    # GT Image
    # pltUtil.plotImage(axs[0], img[:, :], title=str(evID), label="toto", withMasks= gt_masks )
    # pltUtil.plotImage(axs[0], img[:, :], title=str(evID), label="toto")
    # , withFrame= [140, 160, 0, 120]  )
    # pltUtil.plotBboxes( axs[0], gt_bboxes, boxTypes = gt_labels.reshape( gt_labels.shape[0]))
    # Prediction
    # pltUtil.plotImage(axs[1], np.log( img[ :, :] + 1.0), title=str(evID), label="toto", withMasks=pred_masks )
    pltUtil.plotImage(axs, img[ :, :], title=str(evID), label="toto", withFrame= [140, 170, 0, 120])
    boxLabels = [ str(i) + '/' + str(pred_scores[i]) for i in range(pred_scores.shape[0]) ]
    pltUtil.plotBboxes( axs, pred_bboxes, boxTypes= pred_labels, boxLabels=  boxLabels )
    plt.tight_layout()
    plt.show()

def plotTFPN( gt_labels, gt_bboxes, gt_masks, pred_labels, pred_scores, pred_bboxes, pred_masks, img, evID):

    fig, axs = plt.subplots(nrows=1, ncols=2)

    # GT Image
    # pltUtil.plotImage(axs[0], img[:, :], title=str(evID), label="toto", withMasks= gt_masks )
    pltUtil.plotImage(axs[0], img[:, :], title=str(evID), label="toto")
    # , withFrame= [140, 160, 0, 120]  )
    pltUtil.plotBboxes( axs[0], gt_bboxes, boxTypes = gt_labels.reshape( gt_labels.shape[0]))
    # Prediction
    # pltUtil.plotImage(axs[1], np.log( img[ :, :] + 1.0), title=str(evID), label="toto", withMasks=pred_masks )
    pltUtil.plotImage(axs[1], np.log( img[ :, :] + 1.0), title=str(evID), label="toto")
    # , withFrame= [140, 160, 0, 120])
    boxLabels = [ str(i) + '/' + str(pred_scores[i]) for i in range(pred_scores.shape[0]) ]
    pltUtil.plotBboxes( axs[1], pred_bboxes, boxTypes= pred_labels, boxLabels=  boxLabels )
    plt.tight_layout()
    plt.show()
        
def computeTFPN( gt_labels, gt_bboxes, pred_labels_, pred_scores_, pred_bboxes_, IoUCutOff = [0.5], scoreCutOff=0.0 ):
    # TF, ... selected with IoU as the alone criterion
    # TF defined with class, IoU, score

    np.set_printoptions(precision=2)
    print( "computeTFPN GT gt_labels, gt_bboxes :", gt_labels, gt_bboxes )
    print( "computeTFPN pred labels, scores, bboxes", pred_labels, pred_scores, pred_bboxes)
    nbrOfGT = gt_labels.shape[0]
    match

    # Score cut-off (noise !)
    ix = np.where(pred_scores >= ScoreCutOff)
    pred_scores = pred_scores_[ix]
    pred_labels = pred_labels_[ix]
    pred_bboxes = pred_bboxes_[ix,:]
    print( "computeTFPN/ after score cut-off pred labels, scores, bboxes", pred_labels, pred_scores, pred_bboxes)

    # Allocate returned values
    n_iou = len(  IoUCutOff )
    tp = np.zeros( ( n_iou) )
    fp = np.zeros( ( n_iou) )
    fn = np.zeros( ( n_iou) )
    nbrPred =  np.zeros( ( n_iou) )
    nbrGT = np.zeros( ( n_iou) )


    # TF, ....
    if pred_score.shape[0] == 0:
        print("GG no pred_score  :")
        fn = nbrGT
        return ( tp, fp, fn, nbrPred, nbrGT )
    if gt_bboxes.shape[0] == 0:
        prin("GG no gt_bboxes ")
        return ( tp, fp, fn, nbrPred, nbrGT )

    # Overlap matrix[ n-pred, n-GT ]
    overlaps = mutils.compute_overlaps(pred_bboxes, gt_bboxes)
    overlaps[ overlaps >   1.0 ] = 1.0

    # Assign a GT to different predictions with different classes
    gt_ix = np.count_nonzero( iou_overlap, axis = 0)


    for k, IoUCutOff in enumerate(IoU):
        iou_overlaps =  np.copy( overlaps )
        iou_overlaps[ iou_overlaps <  IoUCutOff ] = 0.0

        pred_sum = np.sum( iou_overlaps, axis=1 )
        ix = np.where ( pred_sum > 0.0 )
        iou_overlap = np.delete( iou_overlap, ix, axis=0)
        # remove predictions not satisfying IoU
        nbrPred = iou_overlap.shape[0]

        # Debug
        print( "computeTFPN iou_overlaps :")
        print( iou_overlaps )

        # TP & FP
        tp[k] += tffp.shape[0]
        tpfp = np.count_nonzero( iou_overlap, axis = 1)
        # Remove one TP (with the best score)
        tpfp = np.add(tffp, -1)
        fp[k] += np.sum( tpfp )

        # TP & FP

        # Prediction axis (TP & FP)
        idx = np.argmax(iou_overlaps, axis=1)
        idx_tp = [ idx for i, idx  in enumerate(idx) if iou_overlaps[i, idx] != 0.0 ]
        tp_ = len( idx_tp )
        fp_ = idx.shape[0] - tp

        # Remove selected in the matrix
        o = np.zeros( overlap.shape )
        for i, idx  in enumerate(idx):
          o[i, idx ] = iou_overlaps[i, idx]
        # Debug
        print( "computeTFPN idx   :", idx )
        print( "computeTFPN idx_tp:", idx_tp )
        print( "computeTFPN tp, fp:", tp_, fp_ )
        print( "computeTFPN ovelap after prediction selection")
        print( o )

        # GT axis axis (TP & FP)
        # Unique GT
        idx = np.argmax(o, axis=0)
        # Value at idx != 0
        uniq_gt_idx = [ idx for j, idx  in enumerate(idx) if o[idx,j] != 0.0 ]
        oneAndOnlyOneGT = len( uniq_gt_idx )
        fn_ = idx.shape[0] - oneAndOnlyOneGT
        # Debug
        print( "computeTFPN GT idx             :", idx )
        print( "computeTFPN uniq_gt_idx        :", uniq_gt_idx )
        print( "computeTFPN oneAndOnlyOneGT, fn:", oneAndOnlyOneGT, fn_ )
        tf[k] = tf_
        fn[k] = fn_
        fp[k] = fp_
        nbrPred[k] = overlap.shape[0] 
        nbrGT[k]   = overlap.shape[1]

    return ( tp, fp, fn, nbrPred, nbrGT )

class StatOnPredictions:
    """
    GG TODO: to rewrite description
    Prediction pipeline:
    - receives a patched patient image (n_patches, c, y, x, (z)) from patient data loader.
    - forwards patches through model in chunks of batch_size. (method: batch_tiling_forward)
    - unmolds predictions (boxes and segmentations) to original patient coordinates. (method: spatial_tiling_forward)

    Ensembling (mode == 'test'):
    - for inference, forwards 4 mirrored versions of image to through model and unmolds predictions afterwards
      accordingly (method: data_aug_forward)
    - for inference, loads multiple parameter-sets of the trained model corresponding to different epochs. for each
      parameter-set loops over entire test set, runs prediction pipeline for each patient. (method: predict_test_set)

    Consolidation of predictions:
    - consolidates a patient's predictions (boxes, segmentations) collected over patches, data_aug- and temporal ensembling,
      performs clustering and weighted averaging (external function: apply_wbc_to_patient) to obtain consistent outptus.
    - for 2D networks, consolidates box predictions to 3D cubes via clustering (adaption of non-maximum surpression).
      (external function: merge_2D_to_3D_preds_per_patient)

    Ground truth handling:
    - dissmisses any ground truth boxes returned by the model (happens in validation mode, patch-based groundtruth)
    - if provided by data loader, adds 3D ground truth to the final predictions to be passed to the evaluator.
    """
    def __init__(self, cf, net, logger):

        self.cf = cf
        self.logger = logger
        self.mode = 'test'

        # model instance. 
        self.net = net

        # GG ??? to remove
        # for correct weighting during consolidation.
        self.w_ix = '0'
        # number of ensembled models. used to calculate the number 
        # of expected predictions per position during consolidation 
        # of predictions. Default is 1 (no ensembling, e.g. in validation).
        self.n_ens = 1

        try:
            self.epoch_ranking = np.load(os.path.join(self.cf.fold_dir, 'epoch_ranking.npy'))[:cf.test_n_epochs]
        except:
            raise RuntimeError('no epoch ranking file in fold directory. '
                                   'seems like you are trying to run testing without prior training...')
        self.n_ens = cf.test_n_epochs
        if self.cf.test_aug:
            # GG for miroring 
            self.n_ens *= 4


    def runStats(self, batch_gen, plot=False):
        """
        Note : IoU is compouted with bboxes because MDT masks are
               accumulated in one image.
        """
        dict_of_results = OrderedDict()

        # Get paths of all parameter sets to be loaded for 
        # temporal ensembling. (or just one for no temp. ensembling).
        weight_paths = [os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(epoch), 
                                     'params.pth') for epoch in self.epoch_ranking]


        IoU = [0.1, 0.2 ]
        nbrIoU = len( IoU )
        cumul_tp = np.zeros( (nbrIoU) )
        cumul_fp = np.zeros( (nbrIoU) )
        cumul_fn = np.zeros( (nbrIoU) )
        cumul_nbrPred = np.zeros( (nbrIoU) )
        cumul_nbrGT = np.zeros( (nbrIoU) )
 
        for w_ix, weight_path in enumerate(weight_paths):
            self.logger.info(('tmp ensembling over w_ix:{} epoch:{}'.format(w_ix, weight_path)))
            self.net.load_state_dict(torch.load(weight_path))
            self.net.eval()
            self.w_ix = str(w_ix)  # get string of current rank for unique patch ids.
            
            with torch.no_grad():
                APs = []
                for itest in range(batch_gen['n_test']):
                    batch = next(batch_gen['test'])

                    # Test batch size
                    if len( batch['pid'] ) != 1 and batch['data'].shape[0] != 1:
                        print("Bad batch size", len( batch['pid'] ))
                        exit()

                    evID = batch['pid'][0]
                    img  = batch['data'][0,0,:,:]
                    gt_labels = batch['gt_labels'][0]
                    gt_bboxes = batch['gt_bboxes'][0]
                    gt_masks  = batch['gt_masks'][0]

                    # print ("GG ??? RunStats batch['pid'], evID", batch['pid'], evID )
                    # print ("GG ??? RunStats batch", batch )
                  
                    # Not Used
                    # store batch info in entry of results dict.
                    # if w_ix == 0:
                    #    dict_of_results[evID] = {}

                    results_dict = self.net.test_forward(batch, return_masks=True) # ??? return_masks=self.cf.return_masks_in_test)
                    # print("result_dict", results_dict ) 

                    # Test batch size
                    if len( results_dict['boxes'] ) != 1:
                        print("Bad batch size")
                        exit()

                    # 'boxes' sub-dictionnary: { 'box_coords', 'box_score', 'box_type': 'det', 'box_pred_class_id'}
                    pred_boxes  = results_dict['boxes'][0]
                    pred_bboxes = np.array( [ box['box_coords']        for box in pred_boxes ] )
                    pred_scores = np.array( [ box['box_score']         for box in pred_boxes ] )
                    pred_labels = np.array( [ box['box_pred_class_id'] for box in pred_boxes ] )

                    # Test batch size
                    if len( results_dict['seg_preds'] ) != 1:
                        print("Bad batch size")
                        exit()

                    # Mask accumulated in one image
                    pred_masks = results_dict['seg_preds'][0]
                    # print(" pred_masks.shape", pred_masks.shape )

                    AP, AR, precisions, recalls, overlaps = MPUtl.compute_ap( 
                                                           gt_bboxes, gt_labels, gt_masks,
                                                           pred_bboxes, pred_labels, pred_scores, pred_masks)
                    print( "AP, AR, precisions, recalls, overlaps", AP, AR, precisions, recalls )
                    # print( overlaps )
                    APs.append( AP )

                    """

                    # ( tp, fp, fn, nbrPred, nbrGT ) = computeMDT_TFPN( gt_labels, gt_bboxes, pred_labels, pred_scores, pred_bboxes, classes =range( 1, self.cf.head_classes), IoU = IoU )
                    print("tp,  cumul_tp", tp,  cumul_tp)
                    cumul_tp = np.add( cumul_tp, tp )
                    cumul_fp = np.add( cumul_fp, fp )
                    cumul_fn = np.add( cumul_fn, fn )
                    cumul_nbrPred = np.add( cumul_nbrPred, nbrPred )
                    cumul_nbrGT = np.add( cumul_nbrGT, nbrGT )
                    print("cumul_tp     ", cumul_tp)
                    print("cumul_fp     ", cumul_fp)
                    print("cumul_fn     ", cumul_fn)
                    print("cumul_nbrPred", cumul_nbrPred)
                    print("cumul_nbrGT  ", cumul_nbrGT)
                    """

                    if plot :
                      plotWindow( gt_labels, gt_bboxes, gt_masks, pred_labels, pred_scores, pred_bboxes,  pred_masks, img, evID)
                      # plotTFPN( gt_labels, gt_bboxes, gt_masks, pred_labels, pred_scores, pred_bboxes,  pred_masks, img, evID)

                print("mAP ", np.mean(APs) ) 
        #
        # GG Plots
        #          


    def runAndSave(self, batch_gen, fname="pred.obj"):
        """
        Note : Only run and return the predictions & gt
        """
        dict_of_results = OrderedDict()

        # Get paths of all parameter sets to be loaded for 
        # temporal ensembling. (or just one for no temp. ensembling).
        weight_paths = [os.path.join(self.cf.fold_dir, '{}_best_checkpoint'.format(epoch), 
                                     'params.pth') for epoch in self.epoch_ranking]

        all_images =[]; all_evIDs=[]; all_gt_labels=[]; all_gt_bboxes=[]; all_gt_masks=[]; 
        all_pred_labels=[]; all_pred_scores=[]; all_pred_bboxes=[]; all_pred_masks = [];

        print( "GG weight path", weight_paths )
        for w_ix, weight_path in enumerate(weight_paths[0:1]):
            self.logger.info(('tmp ensembling over w_ix:{} epoch:{}'.format(w_ix, weight_path)))
            self.net.load_state_dict(torch.load(weight_path))
            self.net.eval()
            # self.w_ix = str(w_ix)  # get string of current rank for unique patch ids.
            
            with torch.no_grad():
                for itest in range(batch_gen['n_test']):
                    batch = next(batch_gen['test'])

                    # Test batch size
                    if len( batch['pid'] ) != 1 and batch['data'].shape[0] != 1:
                        print("Bad batch size", len( batch['pid'] ))
                        exit()

                    evID = batch['pid'][0]
                    img  = batch['data'][0,0,:,:]
                    gt_labels = batch['gt_labels'][0]
                    gt_bboxes = batch['gt_bboxes'][0]
                    gt_masks  = batch['gt_masks'][0]

                    # print ("GG ??? RunStats batch['pid'], evID", batch['pid'], evID )
                    # print ("GG ??? RunStats batch", batch )
                  
                    # Not Used
                    # store batch info in entry of results dict.
                    # if w_ix == 0:
                    #    dict_of_results[evID] = {}

                    results_dict = self.net.test_forward(batch, return_masks=True) # ??? return_masks=self.cf.return_masks_in_test)
                    # print("result_dict", results_dict ) 

                    # Test batch size
                    if len( results_dict['boxes'] ) != 1:
                        print("Bad batch size")
                        exit()

                    # 'boxes' sub-dictionnary: { 'box_coords', 'box_score', 'box_type': 'det', 'box_pred_class_id'}
                    pred_boxes  = results_dict['boxes'][0]
                    pred_bboxes = np.array( [ box['box_coords']        for box in pred_boxes ] )
                    pred_scores = np.array( [ box['box_score']         for box in pred_boxes ] )
                    pred_labels = np.array( [ box['box_pred_class_id'] for box in pred_boxes ] )

                    # Test batch size
                    if len( results_dict['seg_preds'] ) != 1:
                        print("Bad batch size")
                        exit()

                    # Mask accumulated in one image
                    pred_masks = results_dict['seg_preds'][0]
                    # print(" pred_masks.shape", pred_masks.shape )
                    all_images.append( img )
                    all_evIDs.append( evID )
                    all_gt_labels.append( gt_labels )
                    all_gt_bboxes.append( gt_bboxes )
                    all_gt_masks.append( gt_masks )  

                    all_pred_labels.append( pred_labels )
                    all_pred_scores.append( pred_scores )
                    all_pred_bboxes.append( pred_bboxes )
                    all_pred_masks.append( pred_masks )  

        all_ = ( "Detection result file with the format: list[np.array()] ", 
                 all_images, all_evIDs, all_gt_labels, all_gt_bboxes, all_gt_masks, 
                 all_pred_labels, all_pred_scores, all_pred_bboxes, all_pred_masks)
        
        with open( fname, 'wb') as file_:
            # GG reading python2.x pickle file
            object_ = pickle.dump(all_, file_ )

        return

    def loadPrediction( self, fname="pred.obj" ):

        with open( fname, 'rb') as file_:
            object_ = pickle.load(file_, encoding="latin1")  
        return object_
