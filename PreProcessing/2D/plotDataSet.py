#! /usr/bin/python

__author__="grasseau"
__date__ ="$Jul 29, 2019 5:41:32 PM$"

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

DataSet = "/grid_mnt/data_cms_upgrade/grasseau/HGCAL-2D/DataSet/"

def load(fObjName):

    fileName =  fObjName
    print("Reading ", fileName)
    file_ = open( fileName, 'rb')
    obj = pickle.load( file_)
    return obj

class State(Config):

  genpart_branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  rechit_branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags',
                     'rechit_cluster2d', 'cluster2d_multicluster']
  branches  = genpart_branches
  branches += rechit_branches

  def __init__(self, rootFileName):
    super( State, self).__init__()
    self.currentEvID = 0
    # Distribution of the particle type (use for repporting)
    self.part = np.zeros( len(self.pidToIdx), dtype=int)
    # Event rejected
    self.evRejected = []
    #
    # Open tree
    self.fname = rootFileName
    self.tree = uproot.open(self.fname)["ana/hgc"]
    print ("Nbr of  entries in root file:", self.tree.numentries)
    return

  def setCurrentEvID(self, evID):
      self.currentEvID = evID
      return

if __name__ == "__main__":
    s = State(DataSet + "hgcalNtuple_electrons_photons_pions.root")
    # obj = load( "eval-50.10-20.obj")
    # obj = load( "train-1000.6-10.obj")
    # obj1 = load( "training-1000.10-20.obj")
    # obj1 = load( "train.obj")
    # obj1 = load( "eval-layer.12-20-100.obj")
    obj1 = load( "histo2D-50.no-layer.obj")

    print ('file size', len(obj1), len(obj1[0]))
    mean1 = hplt.computeDataSetMeans( s, obj1 )
    # obj2 = load( "eval-50.10-20.obj")
    # obj2 = load( "train-1000.6-10.obj")
    obj2 = load( "histo2D-50.no-layer.obj")
    mean2 = hplt.computeDataSetMeans( s, obj2 )
    hplt.compareHistos( s, mean1, mean2)

    for i in range(len(obj2[0])):
      hplt.plotAnalyseDataSet_v1 (s, obj2, i )
