import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
from scipy import stats
import cv2
from wpca import PCA, WPCA, EMPCA

from Config import Config

def roundBBoxAndClip( bbox, dtype='float32' ):
    """
    bbox : float
    get the external rectangle bbox
    """
    bb = np.floor( bbox)
    bb = np.maximum( bb, [0.0] )
    idx = np.array([1,3], dtype=np.int32)
    bb[idx] = np.ceil( bbox[ idx ])
    bb[[1,3]] = np.minimum( bb[ [1,3] ], [255.0])
    bb = bb.astype( dtype)
    return bb

def computeSumOfHits( img, bbox):
    """
    img and bbox are float
    """
    bb = roundBBoxAndClip( bbox, 'int32')
    sHits = np.sum( img[ bb[0]:(bb[1]+1), bb[2]:(bb[3]+1)])
    return sHits

def getbbox( histo, exclusive=False ):
    sumX = histo.sum( axis=1 )
    sumY = histo.sum( axis=0 )
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

    if exclusive:
      minX = max( minX-1, 0)
      maxX = min( maxX+1, lX)
      minY = max( minY-1, 0)
      maxY = min( maxY+1, lY)

    return [ int(minX), int(maxX), int(minY), int(maxY)]

def ellipse(axes=None, angles=None, orig=None, t=None):
  x = np.zeros( (t.size,2), dtype=np.float64)
  x[:,0] =  np.ravel( axes[0] * np.cos( t[:] ) )
  x[:,1] =  np.ravel( axes[1] * np.sin( t[:] )  )
  # Rotation
  s = np.sin( angles[0] )
  c = np.cos( angles[0] )
  R = np.array( [ [ c , -s],
                         [s,    c] ] )
  y = np.matmul( R, x.transpose())
  x = y.transpose() + orig
  # x -> x[:0], y -> x[:1]
  # Debug
  # print("Ellipse : theta, (x,y)=E(theta) : ", t, x)
  return x

def getellipse ( histo, ratioECut=0.95, factorSigma=2.0 ):
    """
    ratioECut : max energy ration to select
    factorSigma :  fraction of the gaussian integral
                      1 sigma ~ 63 %
                      2 sigma ~ 95 %
                      3 sigma ~ 99.7 %
    """
    # Warning 0 : Transpose to have i - > X, j -> Y
    # histo = histo.T

    # Tolal energy cut of the cluster
    ecut = np.sum( histo) * (1. - ratioECut)

    # Find pixel ecut 'pcut'
    ind = np.where( histo > 0. )
    a = histo[ind]
    a = np.sort( a )
    # Find pixel ecut 'pcut'
    s = 0.0; i= 0
    while ( s < ecut ):
        pcut = s
        s = s + a[i]
        i = i+1
    pcut = s
    # print ("getellipse ecut, pcut", ecut, pcut)

    # Remove pixel < pcut
    ind = np.where( histo > pcut )
    # ??? x = np.where( a >= pcut, a, 0 )
    x = np.array(ind, dtype=np.float32)
    # Debug
    # print ( x.T )
    w = np.sqrt( histo[ ind] )
    w = [ w, w ]
    w = np.transpose( w )
    # Debug
    # print (w)
    # Debug lin. regression
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x[0],x[1])
    print("slope", slope, intercept)
    xmin = np.min(x[0])
    xmax = np.max(x[0])
    xs = np.array([xmin, xmax])
    ys = xs*slope +intercept
    plt.scatter(x[0], x[1])
    plt.plot(xs, ys)
    plt.xlim(0, 256)
    plt.ylim(0, 256)
    plt.show()
    """
    # PCA
    kwds = {'weights': w}
    ncomp = 2
    # Warning 0 : transpose
    pca = WPCA(n_components=ncomp).fit( np.transpose(x), **kwds)
    # Debug : compute covariance
    """
    print("Shape x, : ", x.shape, w.shape )
    cov = np.cov( x  ) # , aweights=w[:,0] )
    print("cov: ", cov)
    eigVal, eigVec = np.linalg.eig( cov)
    print("eig: ", eigVal)
    print("eig. vect: ", eigVec)
    """
    orig_ =  pca.mean_
    axes_ =  factorSigma * np.sqrt(pca.explained_variance_)
    vectors_ = pca.components_[:ncomp]
    # ellipse rotation
    # Debug
    # print( "sin=", vectors_[0][1], "cos=", vectors_[0][0] )
    angles_ = np.array( [np.arctan2( vectors_[0][1], vectors_[0][0]) ]  )
    """ DEBUG
    print("PCA Components \n", vectors_)
    print("PCA Sigmas, half axes", axes_ )
    print("PCA Means/ origin", orig_ )
    print("PCA Angles, ellipse rotation",  angles_ * 180.0 / np.pi )
    """

    # BBox
    # vectors_[1] is the smallest
    if ( np.abs( vectors_[0][0]) <= 10e-7):
        xmin  = - axes_[0] + orig_[0]
        xmax = + axes_[0] + orig_[0]
        ymin  = - axes_[1] + orig_[1]
        ymax = + axes_[1] + orig_[1]
    else:
        tgR =  vectors_[0][1] / vectors_[0][0]
        #  Y
        # ---
        # Derivate dx/dt = 0
        theta = np.array( [np.arctan2( - axes_[1] * tgR, axes_[0] ) ] )
        xmin = ellipse(axes= axes_, angles= angles_, orig= orig_, t=theta)[0][0]
        xmax =  ellipse(axes= axes_, angles= angles_, orig= orig_, t=theta+np.pi)[0][0]
        if (xmin > xmax) : t = xmin; xmin = xmax; xmax = t
        # Debug
        #print ("PCA theta dX/dTheta, xmin, xmax = 0", theta* 180.0 / np.pi, xmin, xmax)
        #
        #  Y
        # ---
        # Derivate dy/dt = 0
        theta = np.array( [ np.arctan2( axes_[1] , axes_[0]*tgR ) ] )
        ymin = ellipse(axes= axes_, angles= angles_, orig= orig_, t=theta)[0][1]
        ymax =  ellipse(axes= axes_, angles= angles_, orig= orig_, t=theta+np.pi)[0][1]
        if (ymin > ymax) : t = ymin; ymin = ymax; ymax = t
        # Debug
        # print ("PCA theta dY/dTheta, ymin, ymax = 0", theta* 180.0 / np.pi, ymin, ymax)

    # Warning 0 : inverse transpose to have in pixel or  matrix indices
    xmin = max( 0, xmin  )
    ymin = max( 0, ymin  )
    xmax = min( 255, xmax  )
    ymax = min( 255, ymax  )
    bbox = np.array( [xmin, xmax, ymin, ymax], dtype=np.float32 )
    # angles_ = angles_ - np.pi/2
    axes_ = np.array( [ axes_[0], axes_[1] ], dtype=np.float32 )
    orig_ = np.array( [ orig_[0], orig_[1]  ], dtype=np.float32 )
    # print("Angle :", angles_*180/np.pi)
    return bbox, axes_, angles_, orig_

def extractSubImage( histo ):

    # Normalization
    # norm = 255.0 / np.amax( histo )
    # image = np.array( histo * norm, dtype=np.uint8 )
    image = histo
    
    (minX,  maxX, minY,  maxY) = getbbox( histo, exclusive=True)
    
    # print ("  SubImage minmax: ", minX, maxX, minY, maxY)

    subi = image[ minX:maxX+1, minY:maxY+1 ]
    return subi, (minX,  maxX, minY,  maxY)
  
def plotAnalyseEvent ( ev, histos, labels, rbbox = None, ellipse=None, sHits=None ):
  """
  WARNING 0 :
  Images are considered as matrix [i,j]
    i corresponds to Y axe
    j corresponds to X axe
  Images are transposed
  """
  global hEnergyCut
  from matplotlib.colors import ListedColormap
  from matplotlib import colors
  kernel3 =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  kernel5 =cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

  n =  len(histos)
  # Debug
  print ("plotHistos:", len(histos), len (ev))
  print (labels)
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
  # viridis = plt.get_cmap('viridis', 256)
  viridis = plt.get_cmap('nipy_spectral', 256)
  newcolors = viridis(np.linspace(0, 1, 256))
  pink = np.array([1.0, 1.0, 1.0, 1])
  newcolors[0, :] = pink
  newcmap = ListedColormap(newcolors)
  plt.set_cmap('nipy_spectral')

  # Title
  id = str( ev[0].ID )
  partName    =  Config.pidToLatex [ ev[0].pid ]

  titles =["x-z","y-z"]
  evID = len( histos) // 4
  
  for sym in range(2):
    if ev[0].forward:
      strfb = r"forward"
    else:
      strfb = r"backward"
    tmp = r"Event {}, ".format(id) + strfb + " " + partName +", E= {:5.1f} ?Gev".format( ev[0].pEnergy )
    plt.suptitle( tmp,fontsize=14)
    for  xy in range(2):
        k= 4*sym + 2 *xy
        # ??? + 2 ?
        i = xy + sym * 2

        ax = plt.subplot(2, 4, k+1)
        # k = 4*i
        vmax = np.amax( histos[i] )
        # print ("vmax", vmax)
        # image = np.array( histos[i] * norm, dtype=np.uint8)

        image = histos[i].T
        norm = colors.Normalize(vmin=0.0, vmax=vmax )
        """ Debug
        image[200][100] = vmax
        image[201][100] = vmax
        image[202][100] = vmax
        """
        imgplot = plt.imshow(image, label=labels[i],vmin=0.0, vmax=vmax, norm=norm)
        #p = patches.Rectangle( (1,1), 100, 40, color='red', edgecolor='red')
        #ax.add_patch( p )
        if rbbox is not None :
             bb = np.around( rbbox[i] )
             # Debug
             # print("plot rbbox :", bb)
             p = patches.Rectangle( (bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2], facecolor='none', linestyle='--',  edgecolor='blue')
             ax.add_patch( p )
        if ellipse is not None :
            ( bbox, orig, axes, angles) =  ellipse[i]
            """ Debug
            print("plot ellipse bbox:", bbox)
            print("plot ellipse orig:", orig)
            print("plot ellipse axes:", axes)
            """
            #
            # Transposition:
            angles =  angles /  np.pi * 180
            # Debug
            # print("plot ellipse angles:", angles)
            #
            p = patches.Ellipse( (orig[0], orig[1]) , 2*axes[0], 2*axes[1], angle=angles[0], facecolor='none',edgecolor='red')
            ax.add_patch( p )

        #imgplot.set_norm(norm)
        if rbbox is not None:
            s1 = np.sum(histos[i] [ bb[0]:bb[1]+1, bb[2]:bb[3]+1 ])
            if ( abs(s1-sHits[i]) > 1.0e-07):
                printf("WARNING : sHits != sum of bbox")
            sHitsRawBBox = "{:5.1f}".format( sHits[i] )
        plt.title( titles[xy] + ", sHits=" + sHitsRawBBox, fontsize=10  )  # str(k) )
  
        zoom, frame = extractSubImage( histos[i].T  )
        if (frame[0] < frame[1]) and  (frame[2] < frame[3]):
          # Zoom

          ax = plt.subplot(2,4,k+2)
          imgplot = plt.imshow(zoom, label=labels[i], extent=(frame[2], frame[3], frame[1], frame[0]), cmap=newcmap)
          imgplot.set_norm(norm)
          # print("plot frame :", frame)
          if rbbox is not None :
             bb = np.around (rbbox[i])
             # Debug
             # print("zoom plot rbbox :", bb)
             p = patches.Rectangle( (bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2], facecolor='none', linestyle='--',  edgecolor='blue')
             ax.add_patch( p )
          sEllipseHit = 0.0
          if ellipse is not None :
            ( bbox, orig, axes, angles) =  ellipse[i]
            """ Debug
            print("zoom plot ellipse bbox:", bbox)
            print("zoom plot ellipse or "{:5.2f}".format(ig:", orig)
            print("zoom plot ellipse axes:", axes)
            """
            # Transposition:
            angles =  angles /  np.pi * 180
            # Debug
            # print("zoom plot ellipse angles:", angles)
            p = patches.Ellipse( (orig[0], orig[1]) , 2*axes[0], 2*axes[1], angle=angles[0], facecolor='none',edgecolor='red')
            ax.add_patch( p )
            bb = roundBBoxAndClip( bbox)
            p = patches.Rectangle( (bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2], facecolor='none', linestyle='--',  edgecolor='red')
            ax.add_patch( p )
            sEllipseHits = computeSumOfHits( histos[i], bbox)
        tmp = ", sHits = {:5.1f}".format(sEllipseHits)
        plt.title( titles[xy] +  tmp, fontsize=10 )  # str(k) )

        plt.colorbar()

        """
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
        """

  plt.tight_layout()
  plt.show()

def plotImages ( histos, labels ):
  global hEnergyCut
  from matplotlib.colors import ListedColormap
  
  kernel =cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
  # n =  len(histos) / 4
  n =  len(histos) 
  print ("plotHistos:", len(histos))
  print (labels)
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
    zoom, frame = extractSubImage( histos[k+0] )
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

def plotAnalyseDataSet_v1( s, obj, i ):
  from matplotlib.colors import ListedColormap
  from matplotlib import colors

  ax = plt.subplot(1, 1,1)

  (images, bboxes_, labels_, labelIDs_, evIDs_, ellipses_, hEnergies_, evInfos_) = obj
  img = images[i]
  bboxes = bboxes_[i]
  labels = labels_[i]
  labelsIDs = labelIDs_[i]
  evIDs = evIDs_[i]
  ellipses = ellipses_[i]
  hEnergies = hEnergies_[i]
  evInfos = evInfos_[i]

  print ("labels", labels)
  print ("evIDs", evIDs)
  
  plt.set_cmap('nipy_spectral')
  """
  cmap = plt.get_cmap('nipy_spectral', 256)
  newcolors = cmap(np.linspace(0, 1, 256))
  white = np.array([1, 1, 1, 1])
  newcolors[0, :] = white
  newcmap = ListedColormap(newcolors)
  plt.set_cmap(newcmap)
  """
  # viridis = plt.get_cmap('viridis', 256)
  viridis = plt.get_cmap('nipy_spectral', 256)
  newcolors = viridis(np.linspace(0, 1, 256))
  pink = np.array([1.0, 1.0, 1.0, 1])
  newcolors[0, :] = pink
  newcmap = ListedColormap(newcolors)
  plt.set_cmap('nipy_spectral')

  vmax = np.amax( img )
  # imgplot = plt.imshow(img, label=labels, vmin=0.0, vmax=vmax, norm=norm)
  shape = img.shape
  mask = np.zeros((shape[0], shape[1], 3), dtype = "uint8")
  for k in range(len(bboxes)):
    (bbox, orig, axes, angles) = ellipses[k]
    bbox=  np.around( bbox)
    orig=  np.around( orig )
    axes=  np.around( axes)
    # Transpose
    bbox = [ bbox[2], bbox[3], bbox[0], bbox[1] ]
    orig= [ orig[1], orig[0] ]
    axes = [  axes[1], axes[0] ]
    angles =  np.around( ( -  angles)  * 180  / np.pi )
    cv2.ellipse(mask, (orig[0], orig[1]), (axes[0], axes[1]), angles[0] , 0, 360, (255,255,255), -1)
    #cv2.ellipse(img, (50, 50), (80, 20), 5, 0, 360, (0,0,255), -1)
    (bbox, orig, axes, angles) = ellipses[k]
    ( forward, flip, genpartIdx) = evInfos[k]
    print ("Object ev", evIDs[k])
    print("  raw bboxes", bboxes[k])
    print("  e bbox", bbox)
    print("  e axes", axes)
    print("  e orig :", orig)
    ind = (mask[:,:,0] == 255)
  # img = img.T
  img[ind] = img[ind] + vmax*0.5

  # Text

  print ("vmax", vmax)
  # image = np.array( histos[i] * norm, dtype=np.uint8)
  norm = colors.Normalize(vmin=0.0, vmax=vmax )  
  #
  imgplot = plt.imshow(img, label=labels, vmin=0.0, vmax=vmax, norm=norm)
  plt.suptitle( "Compound ev. " +  str(i),  fontsize=16 )
  plt.title( "Events : " + str(evIDs) )

  # Text
  bb =[0 for x in range(4)]
  for k in range(len(bboxes)):

    (bbox, orig, axes, angles) = ellipses[k]
    # In plot/image coord : Transpose
    bb[0] = bbox[2]; bb[1]=bbox[3]
    bb[2] = bbox[0]; bb[3]=bbox[1]
    p = patches.Rectangle( (bb[0], bb[2]), bb[1]-bb[0], bb[3]-bb[2], facecolor='none', linestyle='--',  edgecolor='red')
    ax.add_patch( p )
    #
    evID = evIDs[k]
    ( forward, flip, genpartIdx) = evInfos[k]
    pid = s.tree["genpart_pid"].array()[evID]
    e =  s.tree["genpart_energy"].array()[evID]
    pids_ =  s.tree["genpart_pid"].array()[evID]

    (xmin, xmax, ymin, ymax) = bb
    # In plot coord.
    x = xmax+2
    y = ymin-2
    f =  "f" if (forward) else "b"
    sym = "s" if (flip) else "n"
    evTxt = s.pidToLatex[ pid[genpartIdx] ] + "/" + str( evIDs[k] ) + " " + f + sym + " " + "{:5.1f}".format(hEnergies[k])
    # evTxt = labels[k]  + "/" + str( evIDs[k] ) + " " + f + sym
    # plt.text( x, y, evTxt, fontsize=10,bbox=dict(facecolor='white', alpha=0.5))
    plt.text( x, y, evTxt, fontsize=10, ha='left', va='top',color='white', alpha=0.8)

  plt.tight_layout()
  plt.show()

def computeDataSetMeans( s, obj ):
  from matplotlib.colors import ListedColormap
  from matplotlib import colors

  (images, bboxes_, labels_, labelIDs_, evIDs_, ellipses_, hEnergies_, evInfos_) = obj
  mean = np.zeros( images[0].shape, dtype = "float32" )
  for i in range(len(images)):
     mean = np.add( mean, images[i] )
  mean = mean / len(images)
  return mean

def compareHistos( s, img1, img2, ):
  from matplotlib.colors import ListedColormap
  from matplotlib import colors

  plt.set_cmap('nipy_spectral')

  ax = plt.subplot(1, 3,1)
  vmax = np.amax( img1 )
  norm = colors.Normalize(vmin=0.0, vmax=vmax )
  imgplot = ax.imshow(img1, label="toto", vmin=0.0, vmax=vmax, norm=norm)
  plt.suptitle( "1 ",  fontsize=16 )

  ax = plt.subplot(1, 3,2)
  vmax = np.amax( img2 )
  norm = colors.Normalize(vmin=0.0, vmax=vmax )
  imgplot = ax.imshow(img2, label="toto", vmin=0.0, vmax=vmax, norm=norm)
  plt.suptitle( "2 ",  fontsize=16 )

  ax = plt.subplot(1, 3,3)
  img = img2 - img1
  vmax = np.amax( img )
  norm = colors.Normalize(vmin=0.0, vmax=vmax )
  imgplot = ax.imshow(img, label="toto", vmin=0.0, vmax=vmax, norm=norm)
  plt.suptitle( "Diff ",  fontsize=16 )
  # plt.title( "Events : " + str(evIDs) )
  # Text

  plt.tight_layout()
  plt.show()

