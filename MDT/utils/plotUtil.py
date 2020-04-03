import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from matplotlib import colors
from scipy import stats
import cv2

class Config:
  def __init__(self):
    self.bboxColors = ['black', 'red', 'blue', 'green', 'purple', 'purple', 'purple', 'purple' ]
    return

cfg = Config()


def plotImage( ax, image, label=None, title='', vmin=0.0, vmax=None, withMasks=None, withFrame = None):
  """
  Remark:
  - mask shapes: [# occurences, rows, columns ] or [# occurences, 1, rows, columns ] 
  """
  frame = withFrame
  slice_ = np.s_[ 0:image.shape[0], 0:image.shape[1] ] 
  if withFrame is not None:
    slice_ = np.s_[ withFrame[0]:withFrame[1], withFrame[2]:withFrame[3] ]
  else:
     withFrame = [ 0, image.shape[0], 0, image.shape[1] ]; 
  img = np.copy(image[ slice_ ])

  if vmax==None:
    vmax = np.amax( img )  

  if withMasks is not None:
    # Remove the color dimension
    if withMasks.shape[1] == 1:
      withMasks = np.squeeze( withMasks, axis=1)
    #
    for i in range( withMasks.shape[0] ):
      img = np.add( img, vmax * 0.25 * withMasks[i][slice_])
    #
  vmax = np.amax( img )  

  norm = colors.Normalize(vmin=vmin, vmax=vmax )
  # GG: Take care "extent" invert the axes of the first dimension of the image 
  ax.imshow(img, label=label, vmin=0.0, vmax=vmax, origin='upper', extent=( withFrame[2]-0.5, withFrame[3] + 0.5, withFrame[1]-0.5, withFrame[0]+0.5 )) # , norm=norm)
  ax.set_title( title )


def plotBboxes( ax, bboxes, boxTypes = None, boxLabels = None, noLabel=False):

  if boxTypes is None:
    boxTypes = [ 1 for i in range(bboxes.shape[0]) ]
  if boxLabels is None:
    boxLabels = [ str(i) for i in range(bboxes.shape[0]) ]

  # Debug
  """
  print ("plotBboxes bboxes", bboxes)
  print ("plotBboxes boxTypes", boxTypes)
  print ("plotBboxes boxLabels", boxLabels )  
  """

  colors = [ cfg.bboxColors[t] for t in boxTypes ]
  # bboxes = bboxes.astype(np.int32) 

  for idx, bb in enumerate(bboxes):
    # Transpose
    if not noLabel:
      p = patches.Circle( (bb[1],  bb[0]), 5, edgecolor=colors[idx])
      ax.add_patch( p )
    p = patches.Rectangle( (bb[1], bb[0]), bb[3]-bb[1],  bb[2]-bb[0],
                           facecolor='none', linestyle='--',  edgecolor=colors[idx] )
    ax.add_patch( p )
    (xmin, ymin, xmax, ymax) = bb
    x = ymin-3
    y = xmin
    if not noLabel:
      ax.text( x, y, boxLabels[idx], fontsize=10, ha='left', va='center',color='white', alpha=0.8)
