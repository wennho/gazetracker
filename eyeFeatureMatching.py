import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
import pdb

MIN_MATCH_COUNT = 2

inputImg = cv2.imread('testeye2_0.png', 0)          # queryImage
trainImg = cv2.imread('testeye2_7.png', 0) # trainImage

# Initiate SIFT detector
surf = cv2.SURF()

# find the keypoints and descriptors with SURF
kp1, des1 = surf.detectAndCompute(inputImg, None)
kp2, des2 = surf.detectAndCompute(trainImg, None)

pdb.set_trace()

inputImg = cv2.drawKeypoints(inputImg, kp1, None, (255, 0, 0), 4)
trainImg = cv2.drawKeypoints(trainImg, kp2, None, (255, 0, 0), 4)

plt.subplot(121), plt.imshow(inputImg)
plt.subplot(122), plt.imshow(trainImg)
plt.show()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des1, des2, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    h, w = inputImg.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    trainImg = cv2.polylines(trainImg, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT)
    matchesMask = None

draw_params = dict(matchColor=(0, 255, 0), # draw matches in green color
                   singlePointColor=None,
                   matchesMask=matchesMask, # draw only inliers
                   flags=2)


# draw matches
# img3 = drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

h1, w1 = inputImg.shape[:2]
h2, w2 = trainImg.shape[:2]
nWidth = w1 + w2
nHeight = max(h1, h2)
newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

inputImg = cv2.cvtColor(inputImg, cv2.COLOR_GRAY2RGB)
trainImg = cv2.cvtColor(trainImg, cv2.COLOR_GRAY2RGB)

# inputImg = cv2.drawKeypoints(inputImg, kp1, None, (255, 0, 0), 4)
# trainImg = cv2.drawKeypoints(trainImg, kp2, None, (255, 0, 0), 4)

newimg[:h2, :w2] = trainImg
newimg[:h1, w2:w1 + w2] = inputImg

plt.imshow(newimg, 'gray')
plt.title('Left=Train, Right=Input')
plt.show()