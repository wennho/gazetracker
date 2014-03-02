import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys

if len(sys.argv) < 2:
    print 'Usage: python ' + __file__ + ' <image>'
    sys.exit()

imgFile = sys.argv[1]
origImg = cv2.imread(imgFile, cv2.CV_LOAD_IMAGE_COLOR)
numLines = origImg.shape[0]
startLine = int(numLines * 0.25)
endLine = int(numLines * 0.75)
origImg = origImg[startLine : endLine]
img = cv2.GaussianBlur(origImg, (0,0), 3)

# in BGR format
SKIN_COLOR = np.array([50, 58, 69], dtype=float)
EYE_COLOR = np.array([62, 63, 63], dtype=float)

# want to minimize response for skin, then maximize response for eye
unitSkin = SKIN_COLOR / np.linalg.norm(SKIN_COLOR)


unitEye = EYE_COLOR / np.linalg.norm(EYE_COLOR)
print 'unitEye:', unitEye

colorDiff = EYE_COLOR - unitSkin * EYE_COLOR.dot(unitSkin)

newBasis = colorDiff / np.linalg.norm(colorDiff)
print 'newBasis:', newBasis



M = np.vstack((newBasis.T, unitEye.T))
print M


# convert image to new basis
tmpImg = img.reshape((img.shape[0] * img.shape[1], img.shape[2]))
resultGray = np.dot(M, tmpImg.T)
resultGray = resultGray.T.reshape((img.shape[0], img.shape[1], 2))
# resultGray = np.absolute(resultGray)
resultGray = np.clip(resultGray,0,255)


origImg[:,:,(0,2)] = origImg[:,:,(2,0)]
plt.subplot(221), plt.imshow(origImg)
plt.subplot(222), plt.imshow(cv2.cvtColor(origImg, cv2.cv.CV_BGR2GRAY),cmap=cm.Greys_r)
plt.subplot(223), plt.imshow(resultGray[:,:,0], cmap=cm.Greys_r), plt.title('newBasis')
plt.subplot(224), plt.imshow(resultGray[:,:,1], cmap=cm.Greys_r), plt.title('unitEye')
plt.show()
