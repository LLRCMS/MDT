#! /usr/bin/python

__author__="grasseau"
__date__ ="$Apr 4, 2020 10:56:34 AM$"

import os
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

import utils.exp_utils as utils
import utils.model_utils as mutils
import utils.plotUtil as pltUtil
import MPUtils.utils as MPUtl

def displayStats( title, stat1, stat2, values, epoch1=[], epoch2=[], IoUs=[]):
  np.set_printoptions(precision=2)
  print("Stat of " + title)
  for v in values:
    print("-- " + v + " for set 0")
    print("-- stat1 ", stat1[0][v] )
    print("-- stat2 ", stat2[0][v])
    print("-- ")

def plotCompareAPStats( title, stat1, stat2, values, epoch1, epoch2, IoUs):
    ymax = 0.95
    xmax = 0.5
    nbrCols = len(stat1)
    fig, axs = plt.subplots(nrows=2, ncols=nbrCols, sharex="all", sharey="all")
    for k in range(nbrCols):
        axs[0][k].plot(  IoUs, stat1[k][values[0]], color='orange'  )
        # Draw max value
 
        # axs[0][k].plot(  IoUs, stat1[k]["mAR"], color='red' )
        axs[0][k].plot(  IoUs, stat2[k][values[0]], color='red'  )
        #axs[0][k].plot(  IoUs, stat2[k]["mAPwithMP"], color='red'  )
        # axs[0][k].set_xlabel( "IoU" )
        if k==0:
            axs[0][k].set_ylabel( values[0] )
            # Compute max value
            ymax0 = np.max( stat2[k][values[0]] )
            # idx = np.argmax( stat2[k][values[0]] )
            Idx = np.where( stat2[k][values[0]] == ymax0)
            idx = np.max( Idx )
            xmax0 = IoUs[idx]
            axs[0][k].plot(  [xmax0,xmax0], [0,ymax0], color='black', linestyle="dashed",  linewidth=0.5)
            axs[0][k].text( 0, ymax0, '{:1.2f}'.format(ymax0), fontsize=7, horizontalalignment="left", verticalalignment="bottom" )
            axs[0][k].text( xmax0, 0.2, '{:1.2f}'.format(xmax0), fontsize=7, horizontalalignment="left", verticalalignment="bottom" )
        # Draw max line
        axs[0][k].plot(  [0,1], [ymax0,ymax0], color='black', linestyle="dashed", linewidth=0.5 )
        #
        axs[0][k].set_xlim( 0.0, 1.0 )
        axs[0][k].set_ylim( 0.2, 1.05 )

        axs[0][k].set_title("Epoch " + epoch1[k]  + "/" +  epoch2[k], fontsize=9 )

        # axs[1][k].plot(  IoUs, stat2[k]["mAP"], color='orange' )
        axs[1][k].plot(  IoUs, stat1[k][values[1]], color='orange'  )
        axs[1][k].plot(  IoUs, stat2[k][values[1]], color='red'  )
        axs[1][k].set_xlabel( "IoU" )
        if k==0:
            axs[1][k].set_ylabel( values[1] )
            # Compute max value
            ymax1 = np.max( stat2[k][values[1]] )
            # idx = np.argmax( stat2[k][values[0]] )
            Idx = np.where( stat2[k][values[1]] == ymax1)
            idx = np.max( Idx )
            xmax1 = IoUs[idx]
            axs[1][k].plot(  [xmax1,xmax1], [0,ymax1], color='black', linestyle="dashed",  linewidth=0.5)
            axs[1][k].text( 0, ymax1, '{:1.2f}'.format(ymax1), fontsize=7, horizontalalignment="left", verticalalignment="bottom" )
            axs[1][k].text( xmax1, 0.2, '{:1.2f}'.format(xmax1), fontsize=7, horizontalalignment="left", verticalalignment="bottom" )
        # Draw max line
        axs[1][k].plot(  [0,1], [ymax1,ymax1], color='black', linestyle="dashed", linewidth=0.5 )
        axs[1][k].set_xlim( 0.0, 1.0 )
        axs[1][k].set_ylim( 0.2, 1.05 )
        # axs[1][k].set_title("Epoch " + epoch2[k] )
    #
    plt.suptitle(title)
    plt.show()

def plotOverlapScores(L4predOverlaps, L4minScores ,L5predOverlaps, L5minScores, L4Epochs, L5Epochs):
    
    nbrCols = len( L4predOverlaps)
    fig, axs = plt.subplots(nrows=2, ncols=nbrCols)
    for k in range(nbrCols):

        # Different class
        axs[0][k].plot( L4predOverlaps[k][1], L4minScores[k][1], 'bo', color='blue')
        # Same class
        axs[0][k].plot( L4predOverlaps[k][0], L4minScores[k][0], 'bo', color='red')
        axs[0][k].set_title("Epoch " + L4Epochs[k] + " (#: " + str( len(L4predOverlaps[k][0]) + len(L4predOverlaps[k][1])) + ")" )
        # L5
        # Different  class
        axs[1][k].plot( L5predOverlaps[k][1], L5minScores[k][1], 'bo', color='blue')
        # Same class 
        axs[1][k].plot( L5predOverlaps[k][0], L5minScores[k][0], 'bo', color='red')
        axs[1][k].set_title("Epoch " + L5Epochs[k] + " (#: " + str( len(L5predOverlaps[k][0]) + len(L5predOverlaps[k][1])) + ")" )

        """
        axs[1] = plt.title('Ovelaps (IoU) vs Min(scores)')
        axs[1] = plt.xlabel('Ovelaps (IoU)')
        axs[1] = plt.ylabel('Min(scores)')
        """
    plt.show()

def computeTFPN( gt_labels, gt_bboxes, pred_labels, pred_scores, pred_bboxes, IoUCutOff = [0.5], scoreCutOff=0.0, verbose=False ):
    # TF, ... selected with IoU as the alone criterion
    # TF defined with class, IoU, score

    np.set_printoptions(precision=2)
    # print( "computeTFPN GT gt_labels, gt_bboxes type :", type(gt_labels), type(gt_bboxes) )
    # print( "computeTFPN GT gt_labels, gt_bboxes :", gt_labels, gt_bboxes )
    # print( "computeTFPN pred labels, scores, bboxes", pred_labels, pred_scores, pred_bboxes)
    # print("type", type( gt_labels ))
    nbrOfGT = gt_labels.shape[0]

    # Score cut-off (noise !)
    """
    ix = np.where(pred_scores >= ScoreCutOff)
    pred_scores = pred_scores_[ix]
    pred_labels = pred_labels_[ix]
    pred_bboxes = pred_bboxes_[ix,:]
    print( "computeTFPN/ after score cut-off pred labels, scores, bboxes", pred_labels, pred_scores, pred_bboxes)
    """
    # Allocate returned values
    n_iou = IoUCutOff.shape[0]
    tp = np.zeros( ( n_iou) )
    fp = np.zeros( ( n_iou) )
    fn = np.zeros( ( n_iou) )
    nbrPred =  np.zeros( ( n_iou) )
    nbrGT = np.zeros( ( n_iou) )
    nbrGT = gt_bboxes.shape[0]


    # TF, ....
    if pred_scores.shape[0] == 0:
        print("Warning no pred_score  :")
        fn = nbrGT
        return ( tp, fp, fn, nbrPred, nbrGT )
    if gt_bboxes.shape[0] == 0:
        prin("Warning no gt_bboxes ")
        fp = nbrPred
        return ( tp, fp, fn, nbrPred, nbrGT )

    # Overlap matrix[ n-pred, n-GT ]
    overlaps = mutils.compute_overlaps(pred_bboxes, gt_bboxes)
    overlaps[ overlaps >   1.0 ] = 1.0

    # Assign a GT to different predictions with different classes
    # gt_ix = np.count_nonzero( iou_overlap, axis = 0)


    for k, cutOff in enumerate(IoUCutOff):
        iou_overlaps =  np.copy( overlaps )
        iou_overlaps[ iou_overlaps <  cutOff ] = 0.0
        """
        pred_sum = np.sum( iou_overlaps, axis=1 )
        ix = np.where ( pred_sum > 0.0 )
        iou_overlap = np.delete( iou_overlap, ix, axis=0)
        # remove predictions not satisfying IoU
        nbrPred = iou_overlap.shape[0]
        """
        nbrPred = iou_overlaps.shape[0]

        # Debug
        if verbose:
          print( "computeTFPN iou_overlaps :")
          print( iou_overlaps )

        # TP & FP
        tpfp = np.count_nonzero( iou_overlaps, axis = 1)
        # Count TP
        tp[k] += np.sum( tpfp >= 1 )
        # Counts FP
        # A prediction without a GT
        fp[k] += np.sum( tpfp == 0 )
        # One  prediction for 2 or more GT
        #   one is counted as a TP
        tpfp = np.add( tpfp, -1)
        tpfp[ np.where(tpfp < 0) ] = 0
        fp[k] += np.sum( tpfp > 0 )

        # FN (& bad TPs: 2 TPs or more for one GT)
        tpfn = np.count_nonzero( iou_overlaps, axis = 0)
        # Verify is there are not 2 for more pred for one GT
        # Tag then as negative
        xtp =  np.sum( tpfn >= 2 )
        if (xtp > 0.0):
          # tp[k] = np.nan
          fp[k] += xtp
        # FN
        fn[k] += np.sum( tpfn == 0 )


        # GG Following Invalid

        # Prediction axis (TP & FP)
        """
        idx = np.argmax(iou_overlaps, axis=1)
        idx_tp = [ idx for i, idx  in enumerate(idx) if iou_overlaps[i, idx] != 0.0 ]
        tp_ = len( idx_tp )
        fp_ = idx.shape[0] - tp
        """

        # Remove selected in the matrix
        """
        o = np.zeros( overlap.shape )
        for i, idx  in enumerate(idx):
          o[i, idx ] = iou_overlaps[i, idx]
        # Debug
        print( "computeTFPN idx   :", idx )
        print( "computeTFPN idx_tp:", idx_tp )
        print( "computeTFPN tp, fp:", tp_, fp_ )
        print( "computeTFPN ovelap after prediction selection")
        print( o )
        """
        # GT axis axis (TP & FP)
        # Unique GT
        """
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
        """
    return ( tp, fp, fn, nbrPred, nbrGT )

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

class Stats:
  def __init__(self):
      self.nbrEvents = 0

  def loadPredictions( self, fname="pred.obj" ):

    o = None
    self.nbrEvents = 0
    with open( fname, 'rb') as file_:
        o = pickle.load(file_, encoding="latin1")
        (self.title, self.images, self.evIDs, self.gt_labels, self.gt_bboxes, self.gt_masks,
                 self.pred_labels, self.pred_scores, self.pred_bboxes, self.pred_masks) = o
        self.nbrEvents = len( self.images )
        print( "Stats.loadPrediction: Read", self.nbrEvents, "events from file",fname)
    return o


  def analyzePredictions( self, IoUCutOff = [0.5], scoreCutOff=0.0, plot=False ):
    # TF, ... selected with IoU as the alone criterion
    # TF defined with class, IoU, score

    np.set_printoptions(precision=2)
    # GG Inv print( "computeTFPN pred labels, scores, bboxes", pred_labels, pred_scores, pred_bboxes)
    cumulOfPreds = 0
    overl = [ [],[] ]
    scores = [ [],[] ]
    labels = []
    for e in range(self.nbrEvents):
        if self.pred_scores[e].shape[0] == 0:
            print("GG no pred_score ")
            continue

        overlaps = mutils.compute_overlaps( self.pred_bboxes[e], self.pred_bboxes[e] )
        overlaps[ overlaps >   1.0 ] = 1.0
        nbrOfPreds = overlaps.shape[0]
        cumulOfPreds += nbrOfPreds
        # Remove diagonal
        for i in range(nbrOfPreds):
          for j in range(i+1):
            overlaps[i, j] = 0.0

        idx = np.where(overlaps != 0 )
        o = np.array( overlaps [ idx ] )

        # print("o ", o )
        # print("idx ", idx )
        # print("over ", overlaps )
        (i,j) = idx
        for k in range( len(i) ):
            # print( "overlaps", overlaps[ i[k], j[k] ], self.pred_scores[e][i[k]], self.pred_scores[e][j[k]], self.pred_labels[e][i[k]], self.pred_labels[e][j[k]])
            sameLabel = self.pred_labels[e][i[k]] == self.pred_labels[e][j[k]] 
            labels.append( sameLabel )
            o = overlaps[ i[k], j[k] ]
            s = min( self.pred_scores[e][i[k]], self.pred_scores[e][j[k]])
            if ( sameLabel ):
              overl[0].append( o )
              scores[0].append( s  )
            else:
                overl[1].append( o )
                scores[1].append( s  )
            if ( o > 0.1 and s > 0.7 ):
                    print( "Event ", e, ", objects (", i[k], "/",  j[k], "), overlap",o,
                        "class/scores", self.pred_labels[e][i[k]], "/", self.pred_scores[e][i[k]], self.pred_labels[e][j[k]], "/",self.pred_scores[e][j[k]] )

    print("cumulOfPreds", cumulOfPreds)
    print("Ambiguous cases", len(overl), len(scores), len(labels))
    print("Same Class:", labels.count(True), "different Classes:",labels.count(False))
    if plot:
        fig, axs = plt.subplots(nrows=1, ncols=2)
        axs[1]=plt.plot( overl[0], scores[0], 'bo', color='blue')
        axs[1]=plt.plot( overl[1], scores[1], 'bo', color='red')
        axs[1] = plt.title('Ovelaps (IoU) vs Min(scores)')
        axs[1] = plt.xlabel('Ovelaps (IoU)')
        axs[1] = plt.ylabel('Min(scores)')
        plt.show()
    return overl, scores

  def filterPredictions( self, IoUMin = 0.1, scoreMin=0.8, selectBestScore=False ):
    # Remove "ambigous" prediction i.e. high IoU values and low score prédictions
    # Seems to be a NMS like on predictions
    # if selectBestScore is true the worst score prediction is removed for all cases (IoU,score) cutoff
    # if selectBestScore only IoU < IoUMin and score < scoreMin are removed
    # In all the cases 1 prediction is removed at least
    np.set_printoptions(precision=2)
    # GG Inv print( "computeTFPN pred labels, scores, bboxes", pred_labels, pred_scores, pred_bboxes)
    cumulOfPreds = 0
    nbrOfRemovedObj  = 0
    for e in range(self.nbrEvents):
        if self.pred_scores[e].shape[0] == 0:
            print("GG no pred_score ")
            continue

        overlaps = mutils.compute_overlaps( self.pred_bboxes[e], self.pred_bboxes[e] )
        overlaps[ overlaps >   1.0 ] = 1.0
        nbrOfPreds = overlaps.shape[0]
        cumulOfPreds += nbrOfPreds
        # Remove upper triengular matrix
        for i in range(nbrOfPreds):
          for j in range(i+1):
            overlaps[i, j] = 0.0

        idx = np.where(overlaps != 0 )
        o = np.array( overlaps [ idx ] )

        # print("o ", o )
        # print("idx ", idx )
        # print("over ", overlaps )
        (i,j) = idx
        remove = []
        for k in range( len(i) ):
            # print( "overlaps", overlaps[ i[k], j[k] ], self.pred_scores[e][i[k]], self.pred_scores[e][j[k]], self.pred_labels[e][i[k]], self.pred_labels[e][j[k]])
            # sameLabel = self.pred_labels[e][i[k]] == self.pred_labels[e][j[k]]
            # labels.append( sameLabel )
            o = overlaps[ i[k], j[k] ]
            if self.pred_scores[e][i[k]] < self.pred_scores[e][j[k]]:
                s = self.pred_scores[e][i[k]]
                remove_idx = i[k]
            else:
                s = self.pred_scores[e][j[k]]
                remove_idx = j[k]

            if o > IoUMin :
                # Remove the lowest score
                remove.append( remove_idx )
                print( "Event ", e, "removed (IoU):",remove_idx, ", objects (", i[k], "/",  j[k], "), overlap",o,
                        "class/scores", self.pred_labels[e][i[k]], "/", self.pred_scores[e][i[k]], self.pred_labels[e][j[k]], "/",self.pred_scores[e][j[k]] )
            elif s < scoreMin:
                # Remove the lowest score
                remove.append( remove_idx )
                print( "Event ", e, "removed (score):",remove_idx, ", objects (", i[k], "/",  j[k], "), overlap",o,
                        "class/scores", self.pred_labels[e][i[k]], "/", self.pred_scores[e][i[k]], self.pred_labels[e][j[k]], "/",self.pred_scores[e][j[k]] )
            elif selectBestScore:
                # Remove the lowest score
                remove.append( remove_idx )
                print( "Event ", e, "removed (best s):",remove_idx, ", objects (", i[k], "/",  j[k], "), overlap",o,
                        "class/scores", self.pred_labels[e][i[k]], "/", self.pred_scores[e][i[k]], self.pred_labels[e][j[k]], "/",self.pred_scores[e][j[k]] )

        if len(remove) != 0:
            nbrOfRemovedObj += len(remove)
            remove.sort(reverse=True)
            # Unique values
            tmp = set(remove)
            remove = list(tmp)
            print( "Event ", e, ", remove objects:", remove)
            print( self.pred_bboxes[e].shape, self.pred_scores[e].shape,  self.pred_labels[e].shape, self.pred_masks[e].shape)
            self.pred_bboxes[e] = np.delete( self.pred_bboxes[e], remove, axis=0)
            self.pred_scores[e] = np.delete( self.pred_scores[e], remove, axis=0)
            self.pred_labels[e] = np.delete( self.pred_labels[e], remove, axis=0)
            # self.pred_masks[e] = np.delete( self.pred_masks[e], remove)
            # self.removed[e] = np.array( remove )
            print( self.pred_bboxes[e].shape, self.pred_scores[e].shape,  self.pred_labels[e].shape, self.pred_masks[e].shape)
            """ Inv
            for k in remove:
              print(  "objects ", k, ": ", self.pred_labels[e][k], self.pred_scores[e][k] )
              self.pred_masks[e].pop(k)
              self.pred_scores[e].pop(k)
              self.pred_labels[e].pop(k)
            """
    print("cumulOfPreds", cumulOfPreds)
    print("Removed predictions", nbrOfRemovedObj)

  

  def statOnTFPN( self, IoUs = [0.5],  verbose = False  ):
    TP = np.zeros( len(IoUs) )
    FP = np.zeros( len(IoUs) )
    FN = np.zeros( len(IoUs) )
    APs = np.zeros( len(IoUs) )
    NbrPred = np.zeros( len(IoUs) )
    NbrGT = np.zeros( len(IoUs) )
    # MatterPort computation
    aps = np.zeros( (self.nbrEvents, len(IoUs) ))
    for e in range(self.nbrEvents):
        ap_ = []
        ( tp, fp, fn, nbrPred, nbrGT ) = computeTFPN( self.gt_labels[e], self.gt_bboxes[e], self.pred_labels[e], self.pred_scores[e], self.pred_bboxes[e],
        IoUCutOff=IoUs, scoreCutOff=0.0 )
        if verbose :
            print("Event ", e, ": ",  tp, fp, fn, nbrPred, nbrGT )
        TP += tp; FP += fp; FN +=fn
        NbrPred += nbrPred; NbrGT += nbrGT
        x = tp + fp
        prec = tp / x
        x = tp + fn
        rec = tp / x
        if verbose:
          print ("prec", prec)
          print( "recalls", rec)
        # prec = np.divide( tp,  np.sum( tp, FP) )
        for i, iou in enumerate(IoUs):
          ap, ar, precisions, recalls, overlaps = MPUtl.compute_ap(
                                                           self.gt_bboxes[e], self.gt_labels[e], self.gt_masks[e],
                                                           self.pred_bboxes[e], self.pred_labels[e], self.pred_scores[e], self.pred_masks[e], iou_threshold=iou)
          if verbose:
            print( "i, iou, AP, AR, precisions, recalls, overlaps", i, iou, ap, ar, precisions, recalls )
            print("AP, AR", prec[i], rec[i])
          ap_.append( ap )
        aps[e,:] = ap_
    print("Over all TP, FP, FN, NbrPred, NbrGT",  ": ",  TP, FP, FN, NbrPred, NbrGT )
    APs = (TP)/(TP+FP)
    ARs = (TP)/(TP+FN)
    maps = np.mean( aps, axis=0)
    print("average precision, recalls", APs , ARs )
    print("mAP MP", maps  )
    TPRatio = TP.astype(np.float)/NbrGT
    FNRatio = FN.astype(np.float)/NbrGT
    all = TP + FP + FN
    Acc = TP.astype(np.float) / all
    retDict = {"TP":TP, "FP":FP, "FN": FN, "AP":APs, "AR":ARs, "APwithMP": maps, "nbrPred":NbrPred, "nbrGT": NbrGT, "TPRatio":TPRatio, "FNRatio":FNRatio, "Acc": Acc }
    return retDict

  def plot(self, n=1000):
    for e in range(min(self.nbrEvents,n)):
        plotTFPN( self.gt_labels[e], self.gt_bboxes[e], self.gt_masks[e],
            self.pred_labels[e], self.pred_scores[e], self.pred_bboxes[e],  self.pred_masks[e], self.images[e], self.evIDs[e])

if __name__ == "__main__":
    server_env = False
    # args.exp_source=""
    # args.exp_dir=""
    L4Files = ["pred_106.l4.100.obj", "pred_85.l4.100.obj", "pred_63.l4.100.obj", "pred_118.l4.100.obj", "pred_59.l4.100.obj"]
    L5Files = ["pred_72.l5.100.obj", "pred_112.l5.100.obj", "pred_74.l5.100.obj", "pred_101.l5.100.obj", "pred_89.l5.100.obj"]
    L4Epochs = ["106", "85", "63", "118", "59"]
    L5Epochs = ["72", "112", "74", "101", "89"]
    IoUs= np.arange(0.05, 1.0, 0.05)
    L4predOverlaps= []
    L4minScores= []
    L5predOverlaps= []
    L5minScores= []
    cf = utils.prep_exp("../experiments/HGCAL-2D", "../experiments/HGCAL-2D", server_env, is_training=False, use_stored_settings=True)
    L4stat = Stats()
    L5stat = Stats()
    rawStatL4 = []
    filterStatL4 = []
    rawStatL5 = []
    filterStatL5 = []
    for i in range(len(L4Files)):
        L4stat.loadPredictions(fname= L4Files[i])
        predOverlaps, minScores = L4stat.analyzePredictions(  )
        L4predOverlaps.append(predOverlaps)
        L4minScores.append(minScores)
        # Stat before filtering
        rawStatL4.append( L4stat.statOnTFPN(IoUs=IoUs) )
        #
        # L5
        L5stat.loadPredictions(fname= L5Files[i])
        predOverlaps, minScores = L5stat.analyzePredictions(  )
        L5predOverlaps.append(predOverlaps)
        L5minScores.append(minScores)
        rawStatL5.append( L5stat.statOnTFPN(IoUs=IoUs) )

        # Filtering prediction
        L4stat.filterPredictions( IoUMin = 0.1, scoreMin=0.8, selectBestScore=False )
        filterStatL4.append( L4stat.statOnTFPN(IoUs=IoUs) )
        L5stat.filterPredictions( IoUMin = 0.1, scoreMin=0.8, selectBestScore=False )
        filterStatL5.append( L5stat.statOnTFPN(IoUs=IoUs) )
    displayStats("Raw L4 model vs raw L5 model", rawStatL4, rawStatL5, ["FN", "nbrGT", "nbrPred"])
    displayStats("Filtered L4 model vs filtered L5 model", filterStatL4, filterStatL5, ["FN", "nbrGT", "nbrPred"])
    # plotCompareAPStats("Raw L4 model vs filtered L4 model", rawStatL4, filterStatL4, ['APwithMP', 'AR'], L4Epochs, L4Epochs, IoUs)
    # plotCompareAPStats("Raw L4 model vs filtered L5 model", rawStatL4, filterStatL5, ['AP', 'AR'], L4Epochs, L5Epochs, IoUs)
    # plotCompareAPStats("Raw L4 model vs raw L5 model", rawStatL4, rawStatL5, ['AP', 'AR'], L4Epochs, L5Epochs, IoUs)
    # plotCompareAPStats("Raw L4 model vs raw L5 model", rawStatL4, rawStatL5, ['Acc', 'AP'], L4Epochs, L5Epochs, IoUs)
    # plotCompareAPStats("filtered L4 model vs filtered L5 model", filterStatL4, filterStatL5, ['Acc', 'AP'], L4Epochs, L5Epochs, IoUs)
    # plotCompareAPStats("filtered L4 model vs filtered L5 model", filterStatL4, filterStatL5, ['Acc', 'AR'], L4Epochs, L5Epochs, IoUs)
    plotCompareAPStats("Raw L4 model vs filtered L5 model", rawStatL4, filterStatL5, ['Acc', 'AR'], L4Epochs, L5Epochs, IoUs)

    # plotOverlapScores(L4predOverlaps, L4minScores ,L5predOverlaps,L5minScores, L4Epochs, L5Epochs)
    # L4stat.statOnTFPN( IoUs= np.arange(0.05, 1.0, 0.05))
    # stat.plot(4)
    #stat.filterPredictions( IoUMin = 0.1, scoreMin=0.8, selectBestScore=False )
    #stat.statOnTFPN( IoUs=np.arange(0.05, 1.0, 0.05) )
    # stat.plot()
