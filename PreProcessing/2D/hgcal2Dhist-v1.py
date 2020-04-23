#!/usr/bin/env python
import uproot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

import cv2

#fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"
#fname = "/home/llr/cms/beaudette/hgcal/samples/hgcalNtuple_electrons_photons_pions.root"
#fname = "/mnt/c/Users/Grasseau/Downloads/hgcalNtuple_electrons_photons_pions.root"
fname = "/home/llr/info/grasseau/HAhRD/Data/hgcalNtuple_electrons_photons_pions.root"
#################
###  Physics  ###
#################

# Particule Energy Cut (in GeV)
pEnergyCut = 20.0
#
# Hit Energy Cut (in MeV)
hEnergyCut = 0.5

##################
###  Detector  ###
##################

# Detector in cm
xMin = -266
xMax =  266
yMin =  273
yMax = -273
zMin =  320
zMax =  520

##################
###   Others   ###
##################

# Root file (branc
pidToName = dict([ (11,'e-'), (-11,'e+'), (22,'g'), (-211,'Pi-'), (211,'Pi+') ])
pidToIdx = dict([ (11, 0), (-11, 1), (22, 2), (-211, 3), (211, 4) ])
pidToClass = dict([ (11,'EM'), (-11,'EM'), (22,'EM'), (-211,'Pi'), (211,'Pi') ])
# Take care : Class 0 is background class
pidToClassID = dict([ (11,1), (-11,1), (22,1), (-211, 2), (211, 2) ])

part = np.zeros( len(pidToIdx), dtype=int)     

genpart_branches = ["genpart_gen","genpart_reachedEE","genpart_energy","genpart_eta","genpart_phi", "genpart_pid",
               "genpart_posx","genpart_posy","genpart_posz"]
rechit_branches = ["rechit_x", "rechit_y", "rechit_z", "rechit_energy","rechit_layer", 'rechit_flags','rechit_cluster2d',
            'cluster2d_multicluster']
branches  = genpart_branches
branches += rechit_branches

# Images
xyBins = 256
zBins  = 256

# Others
hXmin = + 1000
hXmax = - 1000
hYmin = + 1000
hYmax = - 1000
hZmin = + 1000
hZmax = - 1000
hEmax =  -10

hENorm = 29.0
hNorm = 255.0 / hENorm 

curentEvID = 0


def extractSubImage( histo, eCut=0 ):

    histo = np.where(histo > eCut, histo, 0.0)
    print "MinMax", np.amin( histo), np.amax( histo)
    # Normalization
    # norm = 255.0 / np.amax( histo )
    # image = np.array( histo * norm, dtype=np.uint8 )
    image = histo
    
    # Extraction
    sumX = image.sum( axis=1 )
    sumY = image.sum( axis=0 )
    lX = len(sumX)
    lY = len(sumY)
    minX = lX +1; minY = lY +1
    maxX = -1;    maxY = -1;
    for i in range(lX):
      if sumX[i] != 0:
        minX = i
        break;
    for i in range(lX-1,-1,-1):
      if sumX[i] != 0:
        maxX = i
        break;
    for i in range(lY):
      if sumY[i] != 0:
        minY = i
        break;
    for i in range(lY-1,-1,-1):
      if sumY[i] != 0:
        maxY = i
        break;

    minX = max( minX-1, 0)
    maxX = min( maxX+1, lX)
    minY = max( minY-1, 0)
    maxY = min( maxY+1, lY)
    
    print "minmax: ", minX, maxX, minY, maxY
    print sumX[minX: maxX]
    subi = image[ minX:maxX+1, minY:maxY+1 ]
    return subi, (minX,  maxX, minY,  maxY)
  
def plotAnalyseEvent ( histos, labels ):
  global hEnergyCut
  from matplotlib.colors import ListedColormap
  from matplotlib import colors
  kernel3 =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  kernel5 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

  n =  len(histos) 
  print "plotHistos:", len(histos)
  print labels
  plt.set_cmap('nipy_spectral')
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['blue'])
  # print 'green', len(plt.get_cmap('nipy_spectral')._segmentdata['green'])
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['red'])
  """
  cmap = plt.get_cmap('nipy_spectral', 256)
  newcolors = cmap(np.linspace(0, 1, 256))
  white = np.array([1, 1, 1, 1])
  newcolors[0, :] = white
  newcmap = ListedColormap(newcolors)
  plt.set_cmap(newcmap)
  """
  viridis = plt.get_cmap('viridis', 256)
  newcolors = viridis(np.linspace(0, 1, 256))
  pink = np.array([1.0, 1.0, 1.0, 1])
  newcolors[0, :] = pink
  newcmap = ListedColormap(newcolors)
  plt.set_cmap('viridis')
  
  for i in range(n):
    k = i*4
    plt.subplot(n,4,k+1)
    # k = 4*i
    vmax = np.amax( histos[i] )
    print "vmax", vmax
    # image = np.array( histos[i] * norm, dtype=np.uint8) 
    
    image = histos[i]
    imgplot = plt.imshow(image, label=labels[i] )
    norm = colors.Normalize(vmin=0.0, vmax=vmax )
    imgplot.set_norm(norm)

    
    zoom, frame = extractSubImage( histos[i], eCut=hEnergyCut )

    if (frame[0] < frame[1]) and  (frame[2] < frame[3]):
      # Zoom
      plt.subplot(n,4,k+2)
      imgplot = plt.imshow(zoom, label=labels[i], cmap=newcmap)
      imgplot.set_norm(norm)

      # Mask
      plt.subplot(n,4,k+3)
      zoom = np.where( zoom > 0, vmax, 0)
      mask = cv2.morphologyEx(zoom, cv2.MORPH_CLOSE, kernel3)
      imgplot = plt.imshow(mask, label=labels[i], cmap=newcmap)
      imgplot.set_norm(norm)

      plt.subplot(n,4,k+4)
      mask = cv2.morphologyEx(zoom, cv2.MORPH_CLOSE, kernel5)
      imgplot = plt.imshow(mask, label=labels[i], cmap=newcmap)
      imgplot.set_norm(norm)
      
    plt.colorbar()

  plt.show()

def plotImages ( histos, labels ):
  global hEnergyCut
  from matplotlib.colors import ListedColormap
  
  kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  # n =  len(histos) / 4
  n =  len(histos) 
  print "plotHistos:", len(histos)
  print labels
  plt.set_cmap('nipy_spectral')
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['blue'])
  # print 'green', len(plt.get_cmap('nipy_spectral')._segmentdata['green'])
  # print 'blue', len(plt.get_cmap('nipy_spectral')._segmentdata['red'])
  """
  cmap = plt.get_cmap('nipy_spectral', 256)
  newcolors = cmap(np.linspace(0, 1, 256))
  white = np.array([1, 1, 1, 1])
  newcolors[0, :] = white
  newcmap = ListedColormap(newcolors)
  plt.set_cmap(newcmap)
  """
  viridis = plt.get_cmap('viridis', 256)
  newcolors = viridis(np.linspace(0, 1, 256))
  pink = np.array([1.0, 1.0, 1.0, 1])
  newcolors[0, :] = pink
  newcmap = ListedColormap(newcolors)
  plt.set_cmap('viridis')
  
  for i in range(n):
    plt.subplot(441)
    # k = 4*i
    k = i
    norm = 255.0 / np.amax( histos[k+0] )
    image = np.array( histos[k+0] * norm, dtype=np.uint8) 

    histos[k+0]
    imgplot = plt.imshow(image, label=labels[k+0])
    plt.colorbar()

    # Zoom
    plt.subplot(442)
    zoom, frame = extractSubImage( histos[k+0], eCut=hEnergyCut )
    imgplot = plt.imshow(zoom, label=labels[k], cmap=newcmap)
    plt.colorbar()

    # Mask
    plt.subplot(443)
    mask = cv2.morphologyEx(zoom, cv2.MORPH_OPEN, kernel)
    imgplot = plt.imshow(mask, label=labels[k], cmap=newcmap)
    plt.colorbar()

    plt.subplot(444)
    mask = cv2.morphologyEx(zoom, cv2.MORPH_CLOSE, kernel)
    imgplot = plt.imshow(mask, label=labels[k], cmap=newcmap)
    plt.colorbar()

    plt.show()

def processEvent( tree, evID=0, eFilter=10.0, histos=None, labels=None, labelIDs=None ):
#def processEvent( tree, evID, eFilter, histos, labels, labelIDs ):
  global fname, branches
  global part
  global hXmin, hXmax, hYmin, hYmax, hZmin, hZmax, hEmax

  nHistos = 0
  print
  print "<processEvent> evID =", evID  
  pid = tree["genpart_pid"].array()[evID]
  print '  pid',pid    
  r =  tree["genpart_reachedEE"].array()[evID]
  print "  Reached EE :", r

  #
  # -- Reached detector filter
  #
  f = (r == 2)
  pid = pid[ f ]


  e =  tree["genpart_energy"].array()[evID]
  e = e[ f ]
  print "  len(e)", len(e)
  print "  energy", e

  zz = tree["genpart_posz"].array()[evID]
  z = []
  for i in range(len(zz)):
   if ( f[i] ):
     z.append( zz[i][0] )
  #     
  z = np.array( z, dtype=np.float32  )
  #
  print " len(z)", len(z)


  # -- Energy filter
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
  print "  Forward:", f
  print "  Energy: ", e
  #  -- Forward
  ef = e[ f ]
  pidf = pid[ f ]
  #  -- Backward
  f = ( z < 0.0)
  eb = e[ f ]
  pidb = pid[ f ]

  # -- Hits
  u = tree["rechit_x"].array()[evID]
  x = np.array( u, dtype=np.float32  )
  xmin = np.amin( x )
  hXmin = min( hXmin, xmin)
  xmax = np.amax( x )
  hXmax = min( hXmax, xmax)
  #
  u = tree["rechit_y"].array()[evID]
  y = np.array( u, dtype=np.float32  )
  ymin = np.amin( y )
  hYmin = min( hYmin, ymin)
  ymax = np.amax( y )
  hYmax = max( hYmax, ymax)
  #
  u = tree["rechit_z"].array()[evID]
  z = np.array( u, dtype=np.float32  )
  za = np.absolute( z )
  zmin = np.amin( za )
  hZmin = min( hZmin, zmin)
  zmax = np.amax( za )
  hZmax = max( hZmax, zmax)
  #
  u = tree["rechit_energy"].array()[evID]
  he = np.array( u, dtype=np.float32  )
  emax = np.amax( he )
  hEmax = max( hEmax, emax)

  print "  len(ef), len(eb)", len(ef), len(eb)
  #
  if len(ef) == 1:
    k = pidToIdx[ pidf[0] ]
    part[ k ] += 1 
    print "OK forward one more", pidToName[pidf[0]] , part[ k ]
    ff =  z > 0.0
    zz = np.absolute( z[ ff ] )
    hee = he[ ff ]
    xx =  x[ ff ]
    yy =  y[ ff ]
    # print hee
    # print xx
    # print zz
    h , xedges, yedges =  np.histogram2d( xx, zz, bins=[xyBins, zBins], range=[[xMin, xMax], [zMin, zMax]], weights=hee )
    print '  len equal ?', len(xx), len(zz), len( hee )
    # norm = 255.0 / np.amax( h )
    histos.append(  h  )
    labels.append( pidToClass[ pidf[0] ] )
    labelIDs.append( np.array( [ pidToClassID[pidf[0]] ], dtype=np.int32) )
    nHistos +=1

    """
    plt.subplot(211)
    imgplot = plt.imshow(h)
    """
    h , xedges, yedges =  np.histogram2d( yy, zz, bins=[xyBins, zBins], range=[[xMin, xMax], [zMin, zMax]], weights=hee )
    # norm = 255.0 / np.amax( h )
    histos.append( h )
    labels.append( pidToClass[ pidf[0] ] )
    # labelIDs.append( pidToClassID[ pidf[0] ] )
    labelIDs.append( np.array( [ pidToClassID[pidf[0]] ], dtype=np.int32) )

    nHistos +=1
    """
    plt.subplot(212)
    imgplot = plt.imshow(h)
 
    plt.colorbar()
    plt.show()
    """

  if len(eb) == 1:
    k = pidToIdx[ pidb[0] ]
    part[ k ] += 1 
    print "OK backward one more", pidToName[pidb[0]] , part[ k ]
    ff = np.where ( z < 0.0 )
    print ff
    zz = np.absolute( z[ ff ] )
    print zz
    hee = he[ ff ]
    xx =  x[ ff ]
    yy =  y[ ff ]
    #
    h , xedges, yedges =  np.histogram2d( xx, zz, bins=[xyBins, zBins], range=[[xMin, xMax], [zMin, zMax]], weights=hee )
    print 'len equal ?', len(xx), len(zz), len( hee )
    #fig, ax = plt.subplots()
    # norm = 255.0 / np.amax( h )
    histos.append( h  )
    labels.append( pidToClass[ pidb[0] ] )
    # labelIDs.append( pidToClassID[ pidb[0] ] )
    labelIDs.append( np.array( [ pidToClassID[pidf[0]] ], dtype=np.int32) )

    nHistos +=1

    """
    plt.subplot(211)
    imgplot = plt.imshow(h)
    """

    h , xedges, yedges =  np.histogram2d( yy, zz, bins=[xyBins, zBins], range=[[xMin, xMax], [zMin, zMax]], weights=hee )
    # norm = 255.0 / np.amax( h )
    histos.append( h  )
    labels.append( pidToClass[ pidb[0] ] )
    # labelIDs.append( pidToClassID[ pidb[0] ] )
    labelIDs.append( np.array( [ pidToClassID[pidf[0]] ], dtype=np.int32) )

    nHistos +=1

    """
    plt.subplot(212)
    imgplot = plt.imshow(h)

    plt.colorbar()
    plt.show()
    """
  return nHistos 

def getHistos ( fname, startEventID=0, nHistos=20, fObjName="train.obj" ):
  global curentEvID
  tree = uproot.open(fname)["ana/hgc"]
  print "# entries :", tree.numentries
  #
  #    Open file and define the branches
  tree = uproot.open(fname)["ana/hgc"]
  print "# entries :", tree.numentries

  histos = []
  labels = []
  labelIDs = []
  evIDs = []

  i = 0
  n = 0
  # If startEventID negative continue
  if (startEventID >= 0):
    curentEvID = startEventID

  while i < nHistos:
     n = processEvent( tree, evID=curentEvID, eFilter=pEnergyCut, histos=histos, labels=labels, labelIDs=labelIDs )
     print curentEvID, n
     curentEvID += 1
     i += n  
     for k in range(n):
       evIDs.append( curentEvID ); 

  # plotImages( histos )

  print "# entries :", tree.numentries
  for p in pidToName.keys():
    print pidToName[p], " :",  part[ pidToIdx[p] ]

  print "hit x min/max", hXmin, hXmax
  print "hit y min/max", hYmin, hYmax
  print "hit z min/max", hZmin, hZmax
  print "hit energy max", hEmax

  # Postprocess & Save for training
  
  # 
  bboxes = []
  images = []
  for i in histos:
    subi, bbox = extractSubImage( i, hEnergyCut )
    # for each histo, list of bboxes
    bboxes.append( [ bbox] )
    # ??? redo extractimage ???
    h = np.where( i > hEnergyCut, i, 0.0)
    norm = 255.0 / np.amax( h )
    images.append( np.array( h * norm, dtype=np.uint8 ))
    print "GGG ", labelIDs

  # print "images", images
  # print "labels", labels[0]
  # print "labelsIds", labelIDs
  plotAnalyseEvent ( histos, labels )

  file_ = open(fObjName, 'w')
  pickle.dump( (images, bboxes, labels, labelIDs, evIDs), file_)

  return histos, labels, labelIDs

def scanEvents ( fname, nEvents ):
  global curentEvID, pEnegryCut
  tree = uproot.open(fname)["ana/hgc"]
  print "# entries :", tree.numentries
  #
  #    Open file and define the branches
  tree = uproot.open(fname)["ana/hgc"]
  print "# entries :", tree.numentries

  i = 0
  n = 0
  while i < nEvents:
    histos = []
    labels = []
    labelIDs = []
    n = processEvent( tree, evID=curentEvID, eFilter=pEnergyCut, histos=histos, labels=labels, labelIDs=labelIDs )
    if (n > 0 ):
      plotAnalyseEvent( histos, labels )
    print curentEvID, n
    curentEvID += 1
    i += 1  

  # plotImages( histos )

  print "# entries :", tree.numentries
  for p in pidToName.keys():
    print pidToName[p], " :",  part[ pidToIdx[p] ]

  print "hit x min/max", hXmin, hXmax
  print "hit y min/max", hYmin, hYmax
  print "hit z min/max", hZmin, hZmax
  print "hit energy max", hEmax

  file_ = open('hgcal.obj', 'w')
  pickle.dump( (histos, labels, labelIDs), file_)

  return histos, labels, labelIDs

def loadImages ( pfname, nStart, nHistos ):
  file_ = open( pfname, 'r')
  object_ = pickle.load(file_)
  histos, labels, labelIDs = object_
  return histos[nStart:nStart+nHistos], labels[nStart:nStart+nHistos], labelIDs[nStart:nStart+nHistos]

def checkObjFile ( pfname ):
  file_ = open( pfname, 'r')
  object_ = pickle.load(file_)
  images, bboxes, labels, labelIDs, evIDs = object_
  print "Check Obj file :", fname
  print "  # of image", len(images), len(bboxes), len(labels), len(labelIDs), len(evIDs)
  print evIDs
  #
  # Labels
  #
  lMax = 0; lMin = 255
  for i in range(len(labelIDs)):
    lMax = max( lMax, np.max( labelIDs[i] ))
    lMin = min( lMin, np.min( labelIDs[i] ))
  print "  Label Min/Max :", lMin, lMax
  for c in range(3):
   a = np.where( labelIDs == np.int64(c) )
   print a
   print "  Label stat : class ", c, np.count_nonzero(a)
  #
  # BBox
  #
  bMax = 0; bMin = 256*256
  for i in range(len(bboxes)):
    for b in range(len(bboxes[i])):
      xmin, xmax, ymin, ymax = bboxes[i][b]
      area = (xmax-xmin)*(ymax-ymin)
      bMax = max( bMax, area )
      bMin = min( bMin, area )
  print "  bbox Min/Max area :", bMin, bMax, float(bMin)/(256*256), float(bMax)/(256*256)

def trash():
 #
  #    Open file and define the branches
  tree = uproot.open(fname)["ana/hgc"]
  print "# entries :", tree.numentries


  histos = []
  labels = []
  for evID in range(10):
     processEvent( tree, evID=evID, eFilter=10.0, histos=histos, labels=labels )

  # print histos
  # print labels

  plotImages( histos )

  print "# entries :", tree.numentries
  for p in pidToName.keys():
    print pidToName[p], " :",  part[ pidToIdx[p] ]

  print "hit x min/max", hXmin, hXmax
  print "hit y min/max", hYmin, hYmax
  print "hit z min/max", hZmin, hZmax
  print "hit energy max", hEmax

if __name__ == "__main__":
  #
  #    Open file and define the branches

  #  histos, labels, labelIDs = scanEvents( fname, 20)

  # plotImages( histos, labels )

  # checkObjFile("train.obj")
  checkObjFile("eval-ev.obj")

  # getHistos ( fname, startEventID=0, nHistos=200, fObjName="train-ev.obj" )
  getHistos ( fname, startEventID=88, nHistos=5, fObjName="eval-x.obj" )

  tree = uproot.open(fname)["ana/hgc"]
  print "# entries :", tree.numentries

  print "hit x min/max", hXmin, hXmax
  print "hit y min/max", hYmin, hYmax
  print "hit z min/max", hZmin, hZmax
  print "hit energy max", hEmax

