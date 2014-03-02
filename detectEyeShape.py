from imports import *
from util import rotate2d, rotZ
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

imgCenter = (origImg.shape[0] / 2, origImg.shape[1] / 2)

img = cv2.GaussianBlur(origImg, (0, 0), 1)

# convert origImg back to color
origImg = cv2.cvtColor(origImg, cv2.COLOR_GRAY2RGB)

newImg = np.zeros(img.shape)
newImg[img < 44] = 1

index = np.vstack(newImg.nonzero()) # 2xM matrix

index[(0, 1), :] = index[(1, 0), :]

index = index[:, index[1].argsort()[::-1]]
index = np.concatenate((index,np.ones((1,index.shape[1]),dtype=int)), axis=0)

XmaxIdx = np.argmax(index[0])
XminIdx = np.argmin(index[0])

diff = index[:, XmaxIdx] - index[:, XminIdx]

angle = math.degrees( math.atan2(diff[1], diff[0] ))

print angle
rotation = cv2.getRotationMatrix2D(imgCenter, angle, 1)
rotIdx = rotation.dot(index)

rotImg = cv2.warpAffine(img, rotation, img.shape[::-1])
yMax = np.max(rotIdx[1])
yMin = np.min(rotIdx[1])

rotXavg = (rotIdx[0, XmaxIdx] + rotIdx[0, XminIdx]) / 2.
yOrigPoints = np.array([
    [rotXavg, rotXavg],
    [yMin, yMax],
    [1,1],
])
yPoints = cv2.getRotationMatrix2D(imgCenter, -angle, 1).dot(yOrigPoints).astype(int)

eyeBase = yPoints[:,1]  # use yMax since y-axis increases downwards

yOrigPoints = yOrigPoints[:2].astype(int)
cv2.circle(rotImg, tuple(yOrigPoints[:,0]), 2, (0, 255, 0))
cv2.circle(rotImg, tuple(yOrigPoints[:,1]), 2, (0, 255, 0))

keypoints = [
    tuple(index[:2,XminIdx]),
    tuple(index[:2,XmaxIdx]),
    tuple(yPoints[:, 0]),
    tuple(yPoints[:, 1]),
]

# Hough circle detection

# set a low threshold (param2) so that we are guaranteed at least 1. Set a high minimum distance (1000)
# so that we have at most 1
circles = cv2.HoughCircles(img, cv2.cv.CV_HOUGH_GRADIENT, 2, 1000, param1=30, param2=10, minRadius=10,
                                   maxRadius=15)

# virtually guaranteed one circle. raise an error otherwise
circle = circles[0,0]
circleCenter = circle[0:2]

circle = np.around(circle)   # round for display only
cv2.circle(origImg, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)
cv2.circle(origImg, (circle[0], circle[1]), 1, (0, 255, 0), 1)

# calculate vector relative to centered base of eye
pupilVector = circleCenter - eyeBase
print 'pupilVector = ', pupilVector

print keypoints
for x in keypoints:
    cv2.circle(origImg, x, 1, (255, 0, 0))




plt.subplot(221), plt.imshow(img, cmap=cm.Greys_r)
plt.subplot(222), plt.imshow(newImg)
plt.subplot(223), plt.imshow(rotImg)
plt.subplot(224), plt.imshow(origImg)

plt.show()

