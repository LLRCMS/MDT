#! /usr/bin/python
import numpy as np

class Config(object):

    # Detector in cm
  xMin = -266
  xMax =  266
  yMin =  273
  yMax = -273
  zMin =  320
  zMax =  520
  layMin = 1
  layMax = 52

  pidToName = dict([ (11,'e-'), (-11,'e+'), (22,'g'), (-211,'Pi-'), (211,'Pi+'), (130, 'K long'), (321, 'K+'), (-321,'K-'), (2112, 'n'), (-2112, 'an'),  (2212, 'p'), (-2212, 'ap') ])
  pidToLatex = dict([ (11,r'$e^{-}$'), (-11,r'$e^{+}$'), (22,r'$\gamma$'), (-211,r'$\pi^{-}$'), (211, r'$\pi^{+}$'), (130, r'$KL$'), (321, r'$K^+$'), (-321, r'$K^-$'), (2112, r'$n$'), (-2112, r'$an$'),  (2212, r'$p$'), (-2212, r'$ap$') ])
  pidToIdx = dict([ (11, 0), (-11, 1), (22, 2), (-211, 3), (211, 4), (130, 5), (321,6), (-321,7),(2112, 8), (-2112, 9),  (2212, 10), (-2212, 11) ])
  # For classification
  pidToClass = dict([ (11,'EM'), (-11,'EM'), (22,'EM'), (-211,'Pi'), (211,'Pi'), (130, 'Pi'), (321,'Pi'), (-321, 'Pi'), (2112, 'Pi'), (-2112, 'Pi'),  (2212, 'Pi'), (-2212, 'Pi') ])
  # Take care : Class 0 is background class
  pidToClassID = dict([ (11,1), (-11,1), (22,1), (-211, 2), (211, 2), (130, 2), (321,2), (-321, 2),(2112, 2), (-2112, 2), (2212, 2), (-2212, 2) ])

  def __init__(self):
    #fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"
    #fname = "/home/llr/cms/beaudette/hgcal/samples/hgcalNtuple_electrons_photons_pions.root"
    # self.fname = "/home/grasseau/Data/HGCALEvents/hgcalNtuple_electrons_photons_pions.root"
    self.fname = "partGun_PDGid11-22-211_Pt35_NTUP_1.root"
    #fname = "/home/llr/info/grasseau/HAhRD/Data/hgcalNtuple_electrons_photons_pions.root"

    self.TrainingName="train-ev.obj"
    # Particule Energy Cut (in GeV)
    self.pEnergyCut = 20.0
    #
    # Histogram Energy Cut (in MeV)
    # Applyed when the histogram is built
    self.histoEnergyCut = 0.5
    #
    self.histoWithLayers = True
    #
    self.plotHistogram =False
    # Pickle Output file name for training
    self.prefixObjNameTraining = "train"
    # Pickle Output file name for ecaluation
    self.prefixObjNameEvaluation = "eval"
    self.nHistoReadForTraining = 5000
    self.nHistoReadForValidation = 50
    #
    self.nTraining    = 20000
    self.nValidation = 100

    # Test
    """    
    self.nHistoReadForTraining = 20*4
    self.nHistoReadForValidation = 20*4
    self.nTraining    = 20
    self.nValidation = 20
    """
    self.minObjectsPerEvent = 12
    self.maxObjectsPerEvent = 20
    self.overlapBbox = 10
