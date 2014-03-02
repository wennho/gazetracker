import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys
from util import rotate2d
import pdb
import math

if len(sys.argv) < 2:
    print 'Usage: python ' + __file__ + ' <image>'
    sys.exit()


def nothing(x):
    pass


imgFile = sys.argv[1]
origImg = cv2.imread(imgFile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
numLines = origImg.shape[0]

startLine = int(numLines * 0.25)
endLine = int(numLines * 0.75)
origImg = origImg[startLine: endLine]
img = cv2.GaussianBlur(origImg, (0, 0), 1)

newImg = np.zeros(img.shape)
newImg[img < 48] = 1

index = np.vstack(newImg.nonzero()) # 2xM matrix

index[(0,1),:] = index[(1,0),:]

index = index[:,index[1].argsort()[::-1]]
XmaxIdx = np.argmax(index[0])
XminIdx = np.argmin(index[0])


diff = index[:, XmaxIdx] - index[:, XminIdx]

angle = math.atan2(diff[0], diff[1])
rotIdx = rotate2d(angle).dot(index)
YmaxIdx = np.argmax(rotIdx[1])
YminIdx = np.argmin(rotIdx[1])
Xavg = (index[0,XmaxIdx] + index[0,XminIdx]) / 2

index = index.T

keypoints = [
    tuple(index[XminIdx]),
    tuple(index[XmaxIdx]),
    (Xavg, index[YmaxIdx, 1]),
    (Xavg, index[YminIdx, 1]),
]

print keypoints

for x in keypoints:
    cv2.circle(img, x, 2, (0, 255, 0))



plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(newImg)
# plt.subplot(223)

plt.show()