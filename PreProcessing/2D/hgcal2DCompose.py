#! /usr/bin/python

__author__="grasseau"
__date__ ="$Oct 22, 2019 10:57:00 AM$"

import sys
import copy
import math
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import pickle
import cv2

import dataSet as ds
from Config import Config
import hgcal2DPlot as  hplt
import hgcal2Dhist as hgh
from hgcal2Dhist import histo2D
from hgcal2Dhist import Event

if __name__ == "__main__":

  np.random.RandomState(3)

  s = hgh.State()
  nTraining    = s.nTraining
  nValidation = s.nValidation
  """
  nTraining    = 10
  nValidation = 12
  """
  nObjectsPerEvent = s.minObjectsPerEvent

  print ("# ")
  print ("#   Training : load  the",  nTraining, "images with one object")
  print ("# ")

  h = hgh.load_v2(  "histo2D-layer", str(s.nHistoReadForTraining)  )

  """ ???
  h2 = load_v2("histo2D", "30-40")
  """

  print ("# ")
  print ("#  Training : Start Compouned objects")
  print ("# ")

  o = hgh.Histo2DCompound()
  # GGG Verue ???
  nTraining = nValidation
  o.assemble_v4( s, h, nRequestedHistos=nTraining, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  # o.save( s.prefixObjNameTraining, suffix=str(nTraining))
  o.save( 'eval-s.layer.12-20', suffix=str(nTraining))
  """
  
  print ("################################################# ")
  print ("# ")
  print ("#   Validation : load the",  nValidation, "images with one object")
  print ("# ")

  h.clear()

  h = hgh.load_v2(  "histo2D-layer", str(s.nHistoReadForValidation)  )

  print ("# ")
  print ("# Number of histos found :", len(h.h2D))
  print ("# ")
  print ("")

  print ("# ")
  print ("#  Validation : Start Compouned objects")
  print ("# ")

  o = hgh.Histo2DCompound()
  print ("Current event ID :", s.currentEvID)
  #o.assemble( s, h, nRequestedHistos=nValidation, nObjectsPerEvent=nObjectsPerEvent, maxCommonArea = 0.0)
  # o.assemble_v3( s, h, nRequestedHistos=nValidation, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  o.assemble_v4( s, h, nRequestedHistos=nValidation, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  # o.save( s.prefixObjNameEvaluation, suffix=str(nValidation) )
  o.save( 'eval-layer.12-20', suffix=str(nValidation) )
  print ("Current event ID :", s.currentEvID)

  print("Rejected Events : ")
  for r in s.evRejected :
      print (r)

  # obj = o.load( s.prefixObjNameEvaluation, suffix=str(nValidation) )
  # for i in range(len(obj[0])):
  #   hplt.plotAnalyseDataSet ( obj, i )
  """
