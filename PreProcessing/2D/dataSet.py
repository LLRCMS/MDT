#!/usr/bin/env python
import pickle



def loadImages ( pfname, nStart, nHistos ):
  file_ = open( pfname, 'r')
  object_ = pickle.load(file_)
  histos, labels, labelIDs = object_
  return histos[nStart:nStart+nHistos], labels[nStart:nStart+nHistos], labelIDs[nStart:nStart+nHistos]

def checkObjFile ( pfname ):
  file_ = open( pfname, 'r')
  object_ = pickle.load(file_)
  images, bboxes, labels, labelIDs, evIDs = object_
  print ("Check Obj file :", fname)
  print ("  # of image", len(images), len(bboxes), len(labels), len(labelIDs), len(evIDs))
  print (evIDs)
  #
  # Labels
  #
  lMax = 0; lMin = 255
  for i in range(len(labelIDs)):
    lMax = max( lMax, np.max( labelIDs[i] ))
    lMin = min( lMin, np.min( labelIDs[i] ))
  print ("  Label Min/Max :", lMin, lMax)
  for c in range(3):
   a = np.where( labelIDs == np.int64(c) )
   print (a)
   print ("  Label stat : class ", c, np.count_nonzero(a))
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
  print ("  bbox Min/Max area :", bMin, bMax, float(bMin)/(256*256), float(bMax)/(256*256))

