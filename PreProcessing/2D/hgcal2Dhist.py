#!/usr/bin/env python
"""
Process
+ The events (forward & backward) are selected to make x-z and y-z histogram
   if:
   - the particles which reached the detector is > pEnergyCut
   - then, if the # particules is<=  nPartInDetect
   Once one histogram is done, a normalized (255) is built. The noise is
   removed by applying a filter to all histogram values > hEnergyCut (h for histogram)
"""
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
    
class State(Config):

  genpart_branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
  rechit_branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags',
                     'rechit_cluster2d', 'cluster2d_multicluster']
  branches  = genpart_branches
  branches += rechit_branches

  def __init__(self):
    super( State, self).__init__()
    self.currentEvID = 0
    # Distribution of the particle type (use for repporting)
    self.part = np.zeros( len(self.pidToIdx), dtype=int)
    # Event rejected
    self.evRejected = []
    #
    # Open tree
    self.tree = uproot.open(self.fname)["ana/hgc"]
    print ("Nbr of  entries in root file:", self.tree.numentries)
    return

  def setCurrentEvID(self, evID):
      self.currentEvID = evID
      return
  
""" 
Describe the event 
"""
class Event(object):
  def  __init__(self):
    """ 
    Hit Bounding Box in cm
    Defaut detector size 

    self.xMin = -266
    self.xMax =  266
    self.yMin =  273
    self.yMax = -273
    self.zMin =  320
    self.zMax =  520
    """
    #
    # Hit Min, Max
    """
    self.hXmin = + 1000
    self.hXmax = - 1000
    self.hYmin = + 1000
    self.hYmax = - 1000
    self.hZmin = + 1000
    self.hZmax = - 1000
    """
    self.hEmax =  -10
    self.ID = -1
    self.forward = True
    # One particle at the moment
    self.pid = []
    # Index of the particle in particle branches (genpat_
    self.pIndex = -1
    # Particle Energy
    self.pEnergy = []


def save_v2(obj, prefixName, suffix):
    #
  name = prefixName +"-"+suffix+".obj"
  print( "# Save histos : ", name )
  #
  with open( name, 'wb') as output:
      pickle.dump( obj, output, pickle.HIGHEST_PROTOCOL)

def load_v2(prefixName, suffix):
    #
    name = prefixName +"-"+suffix+".obj"
    print( "# Load histos : ", name )
    file_ = open( name, 'rb')
    obj = pickle.load( file_)
    return obj

"""
Collect the histo2D and event features to generate 
"images", mask, bb for the training part
"""
class histo2D(object):
  # Images
  xyBins = 256
  zBins  = 256
  zBinsLayers = 256

  def __init__(self):
    # Histogram with cuff off on enery
    self.h2D = []
    self.labels = []
    self.labelIDs = []
    #  Array of arrays  [xmin, xmax, yminx, ymax]
    self.bbox = []
    self.eMax = []
    # Nbr of histogram / 2
    self.ev = []
    self.ellipse = []
    self.sEnergyHits = []
    # Root Event location
    self.evIDs = []
    self.forwards = []
    # One particle at the moment
    self.genpartIdx = []
    # Operated a symmetry
    self.flip = []

  def clear(self):
     self.__init__()
     return

  def fillGTData(self, h, ev, sEnergyHits, flip):
     """
     h : histogram
     ev : event info
     """

     # One particle selected
     pID = ev.pid
     self.h2D.append(  h  )
     self.labels.append( State.pidToClass[ pID ] )
     self.labelIDs.append( np.array( [ State.pidToClassID[pID] ], dtype=np.int32) )

     # Compute Simple or Raw bbox
     rbbox = np.asarray( hplt.getbbox( h ), dtype=np.int32)
     self.bbox.append(  rbbox )
     # Elipse
     bbox, axes, angles, orig =  hplt.getellipse(  h, ratioECut=1.00, factorSigma = 2.0)
     # bbox = np.asarray( hplt.getbbox( h ), dtype=np.int32)

     angles = np.asarray( angles, dtype=np.float32)
     orig = np.asarray( orig, dtype=np.float32)
     axes = np.asarray( axes, dtype=np.float32)

     self.ellipse.append( (bbox, orig, axes, angles)  )
     # histo Max
     self.eMax.append( np.amax( h ))
     # Sum of energy hits
     self.sEnergyHits.append( sEnergyHits)

     # Root/Geant info
     self.evIDs.append( ev.ID )
     self.forwards.append( ev.forward)
     # One particle at the moment
     self.genpartIdx.append( ev.pIndex)
     self.flip.append( flip )

     # Check
     l = len(self.h2D)
     if ( l != len(self.evIDs) or l != len(self.forwards) or l != len(self.genpartIdx) ):
         print ("fill GTData ERROR: not the same size")
         sys.exit()
     if ( isinstance(ev.ID, list)  or isinstance(ev.forward, list) ):
         print(  "##", ev.ID, "##", ev.forward)
         print ("fill GTData ERROR: not unique element")
         sys.exit()
     # Debug
     """
     print ("fill GT Data -- raw bbox", rbbox )
     print ("fill GT Data -- bbox", bbox)
     print("fill GT Data -- Angles",  angles * 180.0 / np.pi )
     print("fill GT Data -- orig", orig)
     print("fill GT Data -- Sigmas, half axes", axes )
     """
     return

  def extractHistoFromEvent( self, state, ev, hx, hy, hz, he  ):
    """
    Build 2 2D histograms (xx//zz and yy/zz planes) from the hits
    Apply an Energy Cut on the resulting histo
    If one of the 2 histo is zero then the even must be skipped.
    Input arguments 
      pid - particle ID
      he = hit enargy
      x, y, z hit positions
    Modified arguments
      histos - histo/image list
      labels - label list
      labelIDs - label ID list
    """ 

    # Histos can be zeros because the energy cut applied to the histo
    nHistos = 0
    if ev.pid not in State.pidToClass:
          print ("  REJECTED Event - bad pid ", ev.pid)
          msg = "extractHistoFromEvent : bad pids :" + str(ev.pid)
          state.evRejected.append( (ev.ID, msg) )
          return 0

    # Forward / Backward Filter
    
    if ev.forward :
      ff =  hz > 0.0
    else :
      ff =  hz < 0.0
    #
    zz = np.absolute( hz[ ff ] )
    hee = he[ ff ]
    xx =  hx[ ff ]
    yy =  hy[ ff ]
    

    cbins = [histo2D.xyBins, histo2D.zBins]
    crange = [[Config.xMin, Config.xMax], [Config.zMin, Config.zMax]]
    if ( state.histoWithLayers):
     cbins = [histo2D.xyBins, histo2D.zBinsLayers]
     crange = [[Config.xMin, Config.xMax], [Config.layMin, Config.layMax]]
     if (np.amin(zz) < Config.layMin):
         print( "ERROR : extractHistoFromEvent layer min =", np.amin(zz))
     if (np.amax(zz) > Config.layMax):
         print( "ERROR : extractHistoFromEvent layer max =", np.amax(zz))
    #
    h1, xedges, yedges =  np.histogram2d( xx, zz, bins=cbins, range=crange, weights=hee )
    #
    # Padding
    #
    """ No Padding image 256x64
    if ( state.histoWithLayers):
      # print( "h1 shape", h1.shape)
      h=np.zeros( ( histo2D.xyBins, 64) )
      h[:,Config.layMin:Config.layMax+1] = h1[:,:]
      h[:,Config.layMax+1:] = 0.0
      #print( "h shape", h.shape)
      h1 = h
    """
    # print '  len equal ?', len(xx), len(zz), len( hee )
    # ??? To do in caller and remove
    h1 = np.where( h1 > state.histoEnergyCut, h1, 0.0)

    h2, xedges, yedges =  np.histogram2d( yy, zz, bins=cbins, range=crange, weights=hee )
    #
    # Padding
    #
    """ No padding
    if ( state.histoWithLayers):
      # print( "h2 shape", h2.shape)
      h=np.zeros( ( histo2D.xyBins, 64) )
      h[:,Config.layMin:Config.layMax+1] = h2[:,:]
      h[:,Config.layMax+1:] = 0.0
      # print( "h shape", h.shape)
      h2 = h
    """
    #
    h2 = np.where( h2 > state.histoEnergyCut, h2, 0.0)
    #
    if ( np.amax( h1) > 0.0  and  np.amax( h2) > 0.0 ):
      #
      # Abherent cases
      #
      badEvent = False
      bb1 = hplt.getbbox( h1 )
      s1 = (bb1[1]-bb1[0]) * (bb1[3]-bb1[2])
      bb2 = hplt.getbbox( h2 )
      s2 = (bb2[1]-bb2[0]) * (bb2[3]-bb2[2])
      cutOffEM = 1000
      cutOffPion = 1000
      if ( state.histoWithLayers):
          # Image 256x64
          """
           cutOffEM = 350
           cutOffPion = 200
          """
          cutOffEM = 350*4
          cutOffPion = 200*4
      #
      # e+/- and photon
      if ( abs( ev.pid ) ==11 or  ev.pid == 22) :
        if ( s1 > cutOffEM ) or (s2 > cutOffEM):
          badEvent = True
      if ( abs( ev.pid ) ==211 or  ev.pid == 130) :
        if ( s1 < cutOffPion ) or (s2 < cutOffPion) :
          badEvent = True
      if badEvent :
          print ("  REJECTED Event - bad areas ", s1, s2)
          msg = "extractHistoFromEvent : bad areas :" + str(ev.pid) + " " +str(s1) +" " + str(s2)
          state.evRejected.append( (ev.ID, msg) )
          return 0
      else :
        # Update stats on particles
        k = State.pidToIdx[ ev.pid ]
        state.part[ k ] += 1
        #
        sHits1 = np.sum( h1 )
        sHits2 = np.sum( h2 )
        self.fillGTData( h1, ev, sHits1, False  )
        self.fillGTData( h2, ev, sHits2, False  )
        self.fillGTData(  np.flip( h1, axis=0), ev, sHits1, True)
        self.fillGTData(  np.flip( h2, axis=0), ev, sHits2, True  )
        nHistos +=4

    return nHistos

  def processEvent( self, state, evID=0, eFilter=10.0,  nPartInDetect=1 ):

    nHistos = 0
    #
    # Hit Min, Max
    hXmin = + 1000
    hXmax = - 1000
    hYmin = + 1000
    hYmax = - 1000
    hZmin = + 1000
    hZmax = - 1000
    hEmax =  -10
    self.ID = -1

    print()
    print ("<processEvent> evID =", evID)
    pid = state.tree["genpart_pid"].array()[evID]
    print ('  pid',pid)
    r =  state.tree["genpart_reachedEE"].array()[evID]
    print ("  Reached EE :", r)
    print("  Energy :", state.tree["genpart_energy"].array()[evID])
    e =  state.tree["genpart_energy"].array()[evID]

    zz = state.tree["genpart_posz"].array()[evID]
    if ( len(e) != len(zz) or len(e) != len(pid) or len(e) != len(r) ) :
      print("ERROR #particle", len(e), len(zz), len(r), len(pid))
    # print(zz)
    z = []
    for i in range(len(zz)):
      if (len(zz[i]) == 0):
        # Particle not reached HGCAL
        z.append(0.0)
      else:
        z.append( zz[i][0] )
    #
    if ( len(e) != len(z) or len(e) != len(pid) or len(e) != len(r) ) :
      print("ERROR #particle", len(e), len(z), len(r), len(pid))
    #
    # -- Filter
    # -- Reached detector,  particle energy cut, forward
    #
    idx = []
    for i in range( len(r)):
      if (r[i] == 2)  and (e[i]>eFilter) and ( z[i] > 0.0):
          idx.append(i)
    idxFwd = idx
    idx = []
    for i in range( len(r)):
      if (r[i] == 2)  and (e[i]>eFilter) and ( z[i] < 0.0):
          idx.append(i)
    idxBwd = idx
    print ( "  Forward  : ", pid[idxFwd] )
    print ( "             ", e[idxFwd] )
    print ( "  Backward : ", pid[idxBwd] )
    print ( "             ", e[idxBwd] )

    z = np.array( z, dtype=np.float32  )

   # -- Hits
    u = state.tree["rechit_x"].array()[evID]
    x = np.array( u, dtype=np.float32  )
    xmin = np.amin( x )
    hXmin = min( hXmin, xmin)
    xmax = np.amax( x )
    hXmax = min( hXmax, xmax)
    #
    u = state.tree["rechit_y"].array()[evID]
    y = np.array( u, dtype=np.float32  )
    ymin = np.amin( y )
    hYmin = min( hYmin, ymin)
    ymax = np.amax( y )
    hYmax = max( hYmax, ymax)
    #
    u = state.tree["rechit_z"].array()[evID]
    z = np.array( u, dtype=np.float32  )
    za = np.absolute( z )
    zmin = np.amin( za )
    hZmin = min( hZmin, zmin)
    zmax = np.amax( za )
    hZmax = max( hZmax, zmax)
    #
    l = state.tree["rechit_layer"].array()[evID]
    l = np.array( l, dtype=np.float32  )
    layers = np.sign( z ) * l
    """
    # Debug
    print( "Layers min/max: ", np.amin(layers), ",", np.amax(layers), " / ", np.amin(z), ", ", np.amax(z)," / ", len(layers ), ",", len(z) )
    print( "> Layers ",  l[:10] )
    print( "> Layers ",  z[:10] )
    print( "> Layers ",  layers[:10] )
    """
    if (state.histoWithLayers):
        z=layers
    #
    u = state.tree["rechit_energy"].array()[evID]
    he = np.array( u, dtype=np.float32  )
    emax = np.amax( he )
    hEmax = max( hEmax, emax)

    # Extract histogram
    fwd = True
    for idx in [idxFwd, idxBwd]:
        if (len(idx) != 0) and ( len(idx)<= nPartInDetect):
          ev = Event()
          ev.pIndex = idx[0]
          ev.pid = pid[idx]
          ev.forward = fwd
          
          ev.hEmax = hEmax
          ev.ID = evID
          ev.pid = pid[idx][0]
          ev.pEnergy = e[idx][0]
          nh = self.extractHistoFromEvent( state, ev, x, y, z, he )
          if (nh > 0):
            self.ev.append(ev)
            nHistos += nh
          else:
            msg = "processEvent [forward=", ev.forward, "] : no histogram"
            print( "Event rejected, ", evID, msg)
            state.evRejected.append( (ev.ID, msg) )
        # Next : backward
        fwd = False
            
    return nHistos


  def processEventOld( self, state, evID=0, eFilter=10.0,  nPartInDetect=1 ):

    nHistos = 0
    #
    # Hit Min, Max
    hXmin = + 1000
    hXmax = - 1000
    hYmin = + 1000
    hYmax = - 1000
    hZmin = + 1000
    hZmax = - 1000
    hEmax =  -10
    self.ID = -1

    print()
    print ("<processEvent> evID =", evID)
    pid = state.tree["genpart_pid"].array()[evID]
    print ('  pid',pid)
    r =  state.tree["genpart_reachedEE"].array()[evID]
    print ("  Reached EE :", r)
    print("  Energy :", state.tree["genpart_energy"].array()[evID])

    #
    # -- Reached detector filte r
    #
    f = (r == 2)
    pid = pid[ f ]

    e =  state.tree["genpart_energy"].array()[evID]
    e = e[ f ]


    zz = state.tree["genpart_posz"].array()[evID]
    z = []
    for i in range(len(zz)):
     if ( f[i] ):
       z.append( zz[i][0] )
    #
    z = np.array( z, dtype=np.float32  )
    #
    print ("  PID : ", pid)
    print ("  Forward : ", z > 0)
    print ("  energy :", e)

    # -- Particle Energy filter
    #
    f = (e >= eFilter)
    # x = x[ f ]
    # y = y[ f ]
    z = z[ f ]
    e = e[ f ]
    pid = pid[ f ]

    #
    # -- Forward/Backward
    #
    f = ( z >= 0.0 )

    #  -- Forward
    ef = e[ f ]
    pidf = pid[ f ]
    print ("  Forward :", f)
    print ("    Energy, pid : ", ef, pidf )

    #  -- Backward
    f = ( z < 0.0)
    eb = e[ f ]
    pidb = pid[ f ]
    print ("  Backward:", f)
    print ("    Energy, pid : ", eb, pidb )
    
    # -- Hits
    u = state.tree["rechit_x"].array()[evID]
    x = np.array( u, dtype=np.float32  )
    xmin = np.amin( x )
    hXmin = min( hXmin, xmin)
    xmax = np.amax( x )
    hXmax = min( hXmax, xmax)
    #
    u = state.tree["rechit_y"].array()[evID]
    y = np.array( u, dtype=np.float32  )
    ymin = np.amin( y )
    hYmin = min( hYmin, ymin)
    ymax = np.amax( y )
    hYmax = max( hYmax, ymax)
    #
    u = state.tree["rechit_z"].array()[evID]
    z = np.array( u, dtype=np.float32  )
    za = np.absolute( z )
    zmin = np.amin( za )
    hZmin = min( hZmin, zmin)
    zmax = np.amax( za )
    hZmax = max( hZmax, zmax)
    #
    u = state.tree["rechit_energy"].array()[evID]
    he = np.array( u, dtype=np.float32  )
    emax = np.amax( he )
    hEmax = max( hEmax, emax)

    
    # Forward
    if (len(ef) != 0) and ( len(ef)<= nPartInDetect):
      ev = Event()
      ev.pid = pidf
      ev.forward = True
      ev.hXmin = hXmin
      ev.hXmax = hXmax
      ev.hYmin = hYmin
      ev.hYmax = hYmax
      ev.hZmin = hZmin
      ev.hZmax = hZmax
      ev.hEmax = hEmax
      ev.ID = evID
      ev.pid = pidf
      ev.pEnergy = ef[0]
      nh = self.extractHistoFromEvent( state, ev, x, y, z, he )
      if (nh > 0):
        self.ev.append(ev)
        nHistos += nh
      else:
        print (  "  Not selected: histogram energy cut")
        state.evRejected.append( ev.ID )

    # Backward
    if (len(eb) != 0) and ( len(eb)<= nPartInDetect):
      ev = Event()
      ev.pid = pidb
      ev.forward = False
      ev.hXmin = hXmin
      ev.hXmax = hXmax
      ev.hYmin = hYmin
      ev.hYmax = hYmax
      ev.hZmin = hZmin
      ev.hZmax = hZmax
      ev.hEmax = hEmax
      ev.ID = evID
      ev.pid = pidb
      ev.pEnergy = eb[0]
      nh = self.extractHistoFromEvent( state, ev, x, y, z, he  )
      if (nh > 0):
        self.ev.append(ev)
        nHistos += nh
      else:
        print ("  Not selected: histogram energy cut")
        state.evRejected.append( ev.ID )
    return nHistos

  """
  Build "nHistos" histograms, with associated labels
  """
  def get2DHistograms ( self, state, startEventID=0, nRequestedHistos=20 ):
    i = 0
    n = 0
    #
    # If startEventID negative then continue
    if (startEventID >= 0):
      state.currentEvID = startEventID
    #
    while i < nRequestedHistos:
       n = self.processEvent( state, evID=state.currentEvID, eFilter=state.pEnergyCut,  nPartInDetect=1)

       if (n != 0 and state.plotHistogram):
         hplt.plotAnalyseEvent ( self.ev[i//4:], self.h2D[i:], self.labels[i:], self.bbox[i:], self.ellipse[i:], sHits=self.sEnergyHits[i:])
       state.currentEvID += 1
       i += n  
       print ("<get2DHistogram> ", i, "/", nRequestedHistos)

    # plotImages( histos ) 
    for p in State.pidToName.keys():
      print (State.pidToName[p], " :",  state.part[ State.pidToIdx[p] ])
  

  def save(self, prefixName, suffix):
    #
    bboxes = []
    images = []
    IDs= []
    n = len( self.h2D)
    for i in range(n):
      h = self.h2D[i]
      # subi, bbox = extractSubImage( i, hEnergyCut )
      # for each histo, list of bboxes
      bboxes.append( [ self.bbox[i] ] )
      norm = 255.0 / np.amax( h )
      images.append( np.array( h * norm, dtype=np.uint8 ))
      # print "GGG ", labelIDs
      IDs.append( self.ev[i//4].ID)


    print ("images", len(images))
    print ("bboxes", len(bboxes))
    print ("labels", len(self.labels))
    print ("labelsIds", len(self.labelIDs))
    print ("ev", len(self.ev))

    file_ = open( prefixName +"-"+suffix+".obj", 'w')
    pickle.dump( (images, bboxes, self.labels, self.labelIDs, IDs), file_)



class  Histo2DCompound(object):
    
  def __init__(self):
      # h2D collection []
      self.h2D = []
      # List of list of array of bbox : [ [ [xmin, ..., ymax], [xmin, ..., ymax], ... ] ] type = int32
      self.bboxes = []
      # List of list of labels
      self.labels = []
      # List of list of labelIDs
      self.labelIDs = []    
      # List of list of evID
      # One per histo 2D h2D
      self.evIDs = []
      self.ellipses = []
      self.hEnergies = []
      self.evInfos = []

      # DataSet infos
      self.dataSetMax = 0.0


  def addHisto_v1(self, i, histObj, j, xShift=0 ):
      print("Add object", j, "->", i)
      print("Before xShift", xShift)
      print("       rBBox", histObj.bbox[j][0], histObj.bbox[j][1] )
      (ebbox, eorig, axes, angles ) = histObj.ellipse[j]
      print("       ebbox :", ebbox[0], ebbox[1])
      print("       eorig :", eorig[0], eorig[1])
      h = histObj.h2D[j]
      bbox = histObj.bbox[j]
      labels = histObj.labels[j]
      labelID = histObj.labelIDs[j]
      evID = histObj.ev[j//4].ID
      ellipse = histObj.ellipse[j]
      # Add the new histogram h in the Synthetic one self.h2D[i]
      # Old version
      # self.h2D[i] = np.add( self.h2D[i], h, xShift )
      # Test bounds
      # hd = self.h2D[i]
      xMin  = bbox[0].astype(int)
      xMax = bbox[1].astype(int)
      yMin  = bbox[2].astype(int)
      yMax = bbox[3].astype(int)
      xdMin  = bbox[0].astype(int) + xShift
      xdMax = bbox[1].astype(int) + xShift
      # hd[ xdMin:xdMax, : ] = hd[ xdMin:xdMax, : ] + h[ xMin:xMax, : ]
      # hd[ xdMin:xdMax, yMin:yMax ] =  hd[ xdMin:xdMax, yMin:yMax ]  + h[ xMin:xMax, yMin:yMax]+ np.full( (xMax-xMin, yMax-yMin), histObj.sEnergyHits[j]*0.5)
      # self.h2D[i] = hd
      # self.h2D[i][ xdMin:xdMax, yMin:yMax ] += ( h[ xMin:xMax, yMin:yMax]+ np.full( (xMax-xMin, yMax-yMin), histObj.sEnergyHits[j]*0.5, dtype=np.float32 ))
      # self.h2D[i][ xdMin:xdMax, yMin:yMax ] += (  np.full( (xMax-xMin, yMax-yMin), histObj.sEnergyHits[j]*0.5 , dtype=np.float32 )  )
      # self.h2D[i][ xdMin:xdMax, yMin:yMax ] = (   histObj.sEnergyHits[j]*0.5   )
      # self.h2D[i][ xdMin:xdMax, yMin:yMax ] = np.full(  (xMax-xMin, yMax-yMin),  histObj.sEnergyHits[j]*0.5 ,  dtype=np.float32 )
      np.add( self.h2D[i][ xdMin:xdMax, yMin:yMax ], np.full(  (xMax-xMin, yMax-yMin),  histObj.sEnergyHits[j]*0.5 ,  dtype=np.float32 ), out=self.h2D[i][ xdMin:xdMax, yMin:yMax ] )


      # Append the new bbox in the list
      # List of list of array of bbox : [ [ [xmin, ..., ymax], [xmin, ..., ymax], ... ] ]
      bbox[0] = xdMin; bbox[1] = xdMax;
      self.bboxes[i].append( bbox )
      # List of list of labels
      self.labels[i].append( labels )
      # List of list of labelIDs
      self.labelIDs[i].append( labelID )
      # List of list of evID
      # One per histo 2D h2D
      (eBBox,orig, axes, angles ) = ellipse
      eBBox[0] += xShift; eBBox[1] += xShift;
      orig[0] += xShift;
      self.ellipses[i].append( ( eBBox,orig, axes, angles )  )
      self.hEnergies[i].append( histObj.sEnergyHits[j] )
      print("Shift :", xShift )
      print("       rBBox",  self.bboxes[i][-1][0],  self.bboxes[i][-1][1])
      print("       min/max :", xMin, xMax, xdMin, xdMax)
      (ebbox, eorig, axes, angles ) = self.ellipses[i][-1]
      print("       ebbox :", ebbox[0], ebbox[1])
      print("       eorig :", eorig[0], eorig[1])
      # Event info
      ev = histObj.ev[ j//4]
      self.evIDs[i].append( ev.ID )
      self.evInfos[i].append( (ev.forward, histObj.flip[j], histObj.genpartIdx[j]) )

  def addHisto_v2(self, i, histObj, j, xShift=0, yShift=0 ):

      (ebbox, orig, axes, angles ) = copy.deepcopy( histObj.ellipse[j] )
      h = histObj.h2D[j]
      bbox = np.copy( histObj.bbox[j])
      labels = histObj.labels[j]
      labelID = histObj.labelIDs[j]
      evID = histObj.ev[j//4].ID
      """ Debug
      print("Add object", j, "->", i)
      print("Before xShift", xShift)
      print("       rBBox", histObj.bbox[j][0], histObj.bbox[j][1] )
      print("       ebbox :", ebbox[0], ebbox[1])
      print("       orig :", orig[0], orig[1])
      """
      # Add the new histogram h in the Synthetic one self.h2D[i]
      # Test bounds
      hd = self.h2D[i]
      xMin  = bbox[0].astype(int)
      xMax = bbox[1].astype(int)
      yMin  = bbox[2].astype(int)
      yMax = bbox[3].astype(int)
      xdMin  = bbox[0].astype(int) + xShift
      xdMax = bbox[1].astype(int) + xShift
      ydMin  = bbox[2].astype(int) + yShift
      ydMax = bbox[3].astype(int) + yShift
      hd[ xdMin:xdMax, ydMin:ydMax ] +=  h[ xMin:xMax, yMin:yMax]
      # Debug
      """
      hd[ xdMin:xdMax, yMin:yMax ] +=  h[ xMin:xMax, yMin:yMax] +  histObj.sEnergyHits[j]*0.05
      hd[ xMin:xMax, yMin:yMin+3 ] +=     histObj.sEnergyHits[j]*0.05
      """
      # Append the new bbox in the list
      # List of list of array of bbox : [ [ [xmin, ..., ymax], [xmin, ..., ymax], ... ] ]
      bbox[0] = xdMin; bbox[1] = xdMax; bbox[2] = ydMin; bbox[3] = ydMax;
      self.bboxes[i].append( bbox )
      # List of list of labels
      self.labels[i].append( labels )
      # List of list of labelIDs
      self.labelIDs[i].append( labelID )
      # List of list of evID
      # One per histo 2D h2D
      ebbox[0] += xShift; ebbox[1] += xShift;
      ebbox[2] += yShift; ebbox[3] += yShift;
      orig[0] += xShift; orig[1] += yShift;
      
      self.ellipses[i].append( ( ebbox,orig, axes, angles )  )
      self.hEnergies[i].append( histObj.sEnergyHits[j] )
      """ Debug
      print("Shift :", xShift )
      print("       rBBox",  self.bboxes[i][-1][0],  self.bboxes[i][-1][1])
      print("       min/max :", xMin, xMax, xdMin, xdMax)
      (ebbox, eorig, axes, angles ) = self.ellipses[i][-1]
      print("       ebbox :", ebbox[0], ebbox[1])
      print("       eorig :", eorig[0], eorig[1])
      """
      # Event info
      ev = histObj.ev[ j//4]
      self.evIDs[i].append( ev.ID )
      self.evInfos[i].append( (ev.forward, histObj.flip[j], histObj.genpartIdx[j]) )


  def compareBboxes ( self, bboxes, bbox):
      """ Compute the max ovelapping area of 'bbox' with
      other boxes 'bboxes' """
      overlapMax = -255*255

      # Debug
      # print ("  bbox set  :", bboxes)
      # print ("  bbox test :", bbox)
      
      for bb in bboxes:
         xmin = max( bb[0], bbox[0] )
         xmax = min( bb[1], bbox[1] )
         xdist = xmax - xmin
         ymin = max( bb[2], bbox[2])
         ymax = min( bb[3], bbox[3] )
         ydist = ymax - ymin
         #
         area = abs(xdist * ydist)
         """ If no overlap the 'area' is negative """
         Overlap = True
         if ( xdist<0) or (ydist <0):
           Overlap = False
         if  not Overlap:
           area = - area

         print ("  area, xDist, yDist",  area, xdist , ydist)
        
         if overlapMax <= 0:
            # No overlap before
            if Overlap:
                overlapMax = area
            else:
                # The overlapSum is kept negative
                overlapMax = max( overlapMax, area )
         else:
            # Overlap before, overlapMax > 0
            if Overlap:
                overlapMax = max( overlapMax, area)
            # else: do nothing, keep the overlapArea value

      return overlapMax
    
  def assemble(self, state, histObj, nRequestedHistos, nObjectsPerEvent = 2, maxCommonArea = 0.0):

      n = nRequestedHistos

      """
      Insert and init  the firsts objects
      """
      self.h2D = histObj.h2D[0:n]
      for i in range(n):
          # Note there are 2 histo per event (xz, yz)
          self.evIDs.append( [ histObj.ev[i//4].ID ] )
          self.bboxes.append( [ histObj.bbox[i] ] )
          self.labels.append( [ histObj.labels[i] ] )
          self.labelIDs.append( [ histObj.labelIDs[i] ] )
          self.ellipses.append( [ histObj.ellipse[i] ] )

      """
      Read orther events and try to assemble them
      """
      h = histo2D()

      # Read & append at most 'nObjectsPerEvent' objects per compoud ev.
      for i in range(n):
         addedObjs = 0
         # nbrNewObjs = np.random.randint(nObjectsPerEvent, size=1)[0]
         nbrNewObjs =( i % nObjectsPerEvent)
         # print "  rand / nObjectsPerEvent :", nbrNewObjs, "/", nObjectsPerEvent
         if (nbrNewObjs <1 ):
             print ("  histo (break)", i,  "# of objects ", addedObjs + 1, "/", nbrNewObjs + 1)
             continue
         h.clear()
         h.get2DHistograms ( state, startEventID= -1, nRequestedHistos=nbrNewObjs  )

         # for j in range( nbrObjs):
         for j in range( len(h.bbox) ):
             cArea = self.compareBboxes( self.bboxes[i], h.bbox[j] )
             print ("  cArea", cArea)
             if (cArea < maxCommonArea ):
                 addedObjs += 1
                 self.addHisto( i, h, j )
                 if (addedObjs >= nbrNewObjs):
                     break

         print ("  histo", i,  "# of objects ", addedObjs + 1, "/", nbrNewObjs + 1)
         print ()
    
  def assemble_v3(self, state, histObj, nRequestedHistos, minObjectsPerEvent = 2, maxObjectsPerEvent=4, maxCommonArea = 0.0):

    n = nRequestedHistos
    nSingleHistos =   len( histObj.h2D )

    """
    Insert and init  the first/initial objects
    """
    
    for k in range(n):
      i = np.random.randint( 0, nSingleHistos, size=1) [0]
      print ( "histo i selected :", i)
      # i = k
      # Note there are 4 histo per event (xz, yz)
      self.evIDs.append( [ histObj.ev[i//4].ID ] )
      self.h2D.append( histObj.h2D[i] )
      self.bboxes.append( [ histObj.bbox[i] ] )
      self.labels.append( [ histObj.labels[i] ] )
      self.labelIDs.append( [ histObj.labelIDs[i] ] )
      self.ellipses.append( [ histObj.ellipse[i] ] )
      self.hEnergies.append( [ histObj.sEnergyHits[i] ] )
      self.evInfos.append( [ ( histObj.ev[i//4].forward, histObj.flip[i], histObj.genpartIdx[i] ) ] )

    for i in range(n):
      print("# Compound ev ", i )
      # Number of objects per image
      nObj = np.random.randint( minObjectsPerEvent, maxObjectsPerEvent+1, size=1) [0]
      addedObjs = 0
      #
      (bb, _, _, _) = self.ellipses[i][0]
      bboxSet = [ bb]
      nPasses = 0
      nNewObjs = 1
      while ( (nNewObjs < nObj) and (nPasses < 2 * nObj)) :
        k = np.random.randint( 0, nSingleHistos, size=1) [0]
        (testBbox, _, _, _) = histObj.ellipse[k]

        maxCArea = self.compareBboxes ( bboxSet, testBbox )

        if (maxCArea < maxCommonArea ):
             nNewObjs += 1
             self.addHisto( i, histObj, k )
             bboxSet.append( testBbox )
        print (" maxCArea : ", maxCArea )
        print (" len(bboxSet): ", len(bboxSet))
        nPasses +=1

      print ("  histo", i,  ", # of req. objects ", nObj, ", # of objects / passes / max", nNewObjs, "/", nPasses, "/", 2*nObj )
      if ( nNewObjs != nObj) : print("### Nbre of objets not reached :", nNewObjs, "/", nObj)
      print ()

  def assemble_v4(self, state, histObj, nRequestedHistos, minObjectsPerEvent = 2, maxObjectsPerEvent=4, maxCommonArea = 0.0):

    n = nRequestedHistos
    nSingleHistos =   len( histObj.h2D )

    """
    Insert and init  the first/initial objects
    """

    for k in range(n):
      i = np.random.randint( 0, nSingleHistos, size=1) [0]
      # print ( "histo i selected :", i)
      # i = k
      # Note there are 4 histo per event (xz, yz)
      self.evIDs.append( [ histObj.ev[i//4].ID ] )
      self.h2D.append( np.copy( histObj.h2D[i] ) )
      self.bboxes.append( [ np.copy( histObj.bbox[i] ) ] )
      self.labels.append( [ histObj.labels[i] ] )
      self.labelIDs.append( [ histObj.labelIDs[i] ] )
      self.ellipses.append( [ copy.deepcopy ( histObj.ellipse[i] ) ] )
      self.hEnergies.append( [ histObj.sEnergyHits[i] ] )
      self.evInfos.append( [ ( histObj.ev[i//4].forward, histObj.flip[i], histObj.genpartIdx[i] ) ] )
      self.dataSetMax = max( self.dataSetMax,  histObj.eMax[i]  )

    for i in range(n):
      print("# Compound ev ", i )
      # Number of objects per image
      nObj = np.random.randint( minObjectsPerEvent, maxObjectsPerEvent+1, size=1) [0]
      addedObjs = 0
      #
      (bb, _, _, _) = self.ellipses[i][0]
      bboxSet = [ np.copy( bb ) ]
      nPasses = 0
      nNewObjs = 1
      while ( (nNewObjs < nObj) and (nPasses < 3 * nObj)) :
        k = np.random.randint( 0, nSingleHistos, size=1) [0]
        # x shift = [-1,0,+1]
        xShift = np.random.randint( -1, 1+1, size=1) [0]
        yShift =  0
        # Debug
        # yShift = 0
        (testBbox_, _, _, _) = histObj.ellipse[k]
        testBbox = np.copy( testBbox_ )
        testBbox[0] += xShift; testBbox[1] += xShift;
        testBbox[2] += yShift; testBbox[3] += yShift;
        # GG Not here ???
        area = abs ( (testBbox[1] - testBbox[0]) * (testBbox[3] - testBbox[2]) )
        if ( area > 10.0 ):
          maxCArea = self.compareBboxes ( bboxSet, testBbox )

          if (maxCArea < maxCommonArea ):
             nNewObjs += 1
             self.addHisto_v2( i, histObj, k, xShift=xShift, yShift=yShift )
             bboxSet.append( testBbox )
          print (" maxCArea, len(bboxSet) : ", maxCArea, len(bboxSet) )
        nPasses +=1
      #
      # Max of all the DataSet
      self.dataSetMax = max( self.dataSetMax,  np.amax( self.h2D[i] )  )

      print ("  histo", i,  ", # of req. objects ", nObj, ", # of objects / passes / max", nNewObjs, "/", nPasses, "/", 2*nObj )
      if ( nNewObjs != nObj) : print("### WARNING: Nbre of objets not reached :", nNewObjs, "/", nObj)
      print ()


  """
    Insert and init  the first/initial objects
  """
  """
  def transform(self, state, histObj, selectedHistos=[0,2] ) :

    n = nRequestedHistos
    nSingleHistos =   len( histObj.h2D )

  

    for i in selectedHistos:
      # i = k
      # Note there are 4 histo per event (xz, yz)
      self.evIDs.append( [ histObj.ev[i//4].ID ] )
      self.h2D.append( np.copy( histObj.h2D[i] ) )
      h = histObj.h2D[i]
      self.bboxes.append( [ np.copy( histObj.bbox[i] ) ] )
      self.labels.append( [ histObj.labels[i] ] )
      self.labelIDs.append( [ histObj.labelIDs[i] ] )
      self.ellipses.append( [ copy.deepcopy ( histObj.ellipse[i] ) ] )
      self.hEnergies.append( [ histObj.sEnergyHits[i] ] )
      self.evInfos.append( [ ( histObj.ev[i//4].forward, histObj.flip[i], histObj.genpartIdx[i] ) ] )
      self.dataSetMax = max( self.dataSetMax,  histObj.eMax[i]  )
      (ebb, _, _, _) = self.ellipses[i][0]

      xMiddle =  0.5 * (ebb[1] - ebb[0])
      yMiddle =  0.5 * (ebb[3] - ebb[2])
      yPos = ebb[0]

      if (yPos > 127):
        yShift = 1
        yMin = 128
        yMax = 256 * 0.5
      else
        yShift = -1
        yMin = 128 *0.5
        yMax =128

     for yy in range(yMin, yMax):
        h = np.zeros( 256, 256 )
        yShift = yy  - yPos
        bb = np.copy( histObj.bbox[i])

        ydMin = bb[2] + yShift; ydMax = bb[3] + yShift
        hd[ bb[0]:bb[1], ydMin:ydMax ] =  h[ bb[0]:bb[1], bb[2]:bb[3] ]
        bb[2] = ydMin; bb[3] = ydMax;


      ##################
    for i in range(n):
      print("# Compound ev ", i )
      # Number of objects per image
      nObj = np.random.randint( minObjectsPerEvent, maxObjectsPerEvent+1, size=1) [0]
      addedObjs = 0
      #
      (bb, _, _, _) = self.ellipses[i][0]
      bboxSet = [ np.copy( bb ) ]
      nPasses = 0
      nNewObjs = 1
      while ( (nNewObjs < nObj) and (nPasses < 2 * nObj)) :
        k = np.random.randint( 0, nSingleHistos, size=1) [0]
        # x shift = [-1,0,+1]
        xShift =  np.random.randint( -1, 1+1, size=1) [0]
        # Debug
        # xShift = 5
        (testBbox_, _, _, _) = histObj.ellipse[k]
        testBbox = np.copy( testBbox_ )
        testBbox[0] += xShift; testBbox[1] += xShift;
        maxCArea = self.compareBboxes ( bboxSet, testBbox )

        if (maxCArea < maxCommonArea ):
             nNewObjs += 1
             self.addHisto_v2( i, histObj, k, xShift )
             bboxSet.append( testBbox )
        print (" maxCArea, len(bboxSet) : ", maxCArea, len(bboxSet) )
        nPasses +=1
      #
      # Max of all the DataSet
      self.dataSetMax = max( self.dataSetMax,  np.amax( self.h2D[i] )  )

      print ("  histo", i,  ", # of req. objects ", nObj, ", # of objects / passes / max", nNewObjs, "/", nPasses, "/", 2*nObj )
      if ( nNewObjs != nObj) : print("### WARNING: Nbre of objets not reached :", nObj)
      print ()
  """

  def save(self, fObjName, suffix="", dataSetNorm=False):
    #
    print ("Compound.save() ")
    images = []
    n = len( self.h2D )
    for i in range(n):
      h = self.h2D[i]
      print ("save min/max histo", np.amin( h ), np.amax( h ), i)
      # subi, bbox = extractSubImage( i, hEnergyCut )
      # for each histo, list of bboxes
      # h = np.where( i > hEnergyCut, i, 0.0)
      if ( dataSetNorm ):
        norm = 255.0 / self.dataSetMax
      else:
        norm = 255.0 / np.amax( h )
      
      images.append( np.array( h * norm, dtype=np.uint8 ))

    print ("  images ", len(images))
    print ("  bboxes   :", [ len(self.bboxes[o]) for o  in range(len(self.bboxes))  ])
    print ("  labels   :",   [ len(self.labels[o]) for o  in range(len(self.labels))  ])
    print ("  labelIds :",  [ len(self.labelIDs[o]) for o  in range(len(self.labelIDs))  ])
    print ("  ev       :",    [ len(self.evIDs[o]) for o  in range(len(self.evIDs))  ])

    for e in range(len(images)):
        print ("Event :", e)
        #print (bbox for bbox in self.bboxes[e])
        print ("  bboxes:", self.bboxes[e])
        print ("  labels  :", self.labels[e])
        print ("  ev ID's :", self.evIDs[e])
        print ("")

    file_ = open( fObjName +"-"+suffix+".obj", 'wb')
    pickle.dump( (images, self.bboxes, self.labels, self.labelIDs, self.evIDs, self.ellipses, self.hEnergies, self.evInfos), file_)

  def load(self, fObjName, suffix=''):
    file_ = open( fObjName +"-"+suffix+".obj", 'rb')
    obj = pickle.load( file_)
    return obj

if __name__ == "__main__":

  np.random.RandomState(3)
  
  s = State()
  h = histo2D()
  nTraining    = s.nTraining
  nValidation = s.nValidation

  nObjectsPerEvent = s.minObjectsPerEvent

  print ("# ")
  print ("#   Training : get the",  nTraining, "images with one object")
  print ("# ")

  
  #  ??? inv
  # s.plotHistogram = True
  #h.get2DHistograms ( s, startEventID=181, nRequestedHistos=nTraining  )

  """ ???
  h.get2DHistograms ( s, startEventID=932, nRequestedHistos= 12  )
  h2 = load_v2("histo2D", "30-40")
  print (h2.bbox)
  """

  h.get2DHistograms ( s, startEventID=0, nRequestedHistos=s.nHistoReadForTraining  )
  # h.get2DHistograms ( s, startEventID=1265, nRequestedHistos=12  )
  save_v2(h, "histo2D", str(s.nHistoReadForTraining) )

  # h.save( s.prefixObjNameTraining, str(1) )

  print ("# ")
  print ("#  Training : Start Compouned objects")
  print ("# ")

  """"
  o = Histo2DCompound()
  # o.assemble( s, h, nRequestedHistos=nTraining, nObjectsPerEvent=nObjectsPerEvent, maxCommonArea = 0.0)
  # o.assemble_v3( s, h, nRequestedHistos=nTraining, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  o.assemble_v4( s, h, nRequestedHistos=nTraining, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  o.save( s.prefixObjNameTraining, suffix=str(nTraining))
  # Visu 
  # obj = o.load( s.prefixObjNameTraining, suffix=str(nTraining)  )
  # for i in range(len(obj[0])):
  #   hplt.plotAnalyseDataSet ( obj, i )
  """
  print ("################################################# ")
  print ("# ")
  print ("#   Validation : get the",  nValidation, "images with one object")
  print ("# ")
  print ("Current event ID :", s.currentEvID)

  h.clear()
  h.get2DHistograms ( s, startEventID=-1, nRequestedHistos=s.nHistoReadForValidation  )
  # h.get2DHistograms ( s, startEventID=-1, nRequestedHistos=12  )
  save_v2(h, "histo2D", str(s.nHistoReadForValidation) )

  # getHistos ( fname, startEventID=88, nHistos=5, fObjName="eval-x.obj" )

  print ("# ")
  print ("# Number of histos found :", len(h.h2D))
  print ("# ")
  print ("")

  """
  print ("# ")
  print ("#  Validation : Start Compouned objects")
  print ("# ")

  o = Histo2DCompound()
  print ("Current event ID :", s.currentEvID)
  #o.assemble( s, h, nRequestedHistos=nValidation, nObjectsPerEvent=nObjectsPerEvent, maxCommonArea = 0.0)
  # o.assemble_v3( s, h, nRequestedHistos=nValidation, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  o.assemble_v4( s, h, nRequestedHistos=nValidation, minObjectsPerEvent = s.minObjectsPerEvent, maxObjectsPerEvent=s.maxObjectsPerEvent, maxCommonArea = s.overlapBbox)
  o.save( s.prefixObjNameEvaluation, suffix=str(nValidation) )
  print ("Current event ID :", s.currentEvID)
  """

  print("Rejected Events : ")
  for r in s.evRejected :
      print (r)

  # obj = o.load( s.prefixObjNameEvaluation, suffix=str(nValidation) )
  # for i in range(len(obj[0])):
  #   hplt.plotAnalyseDataSet ( obj, i )
