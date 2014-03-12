from imports import *
from random import randint

# We need 3 points to compute the homography

img = cv2.imread('leftOutline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

#y, then x
origin = np.array([36, 78])

#x, then y
left = np.array([12, 37])
bot = np.array([45, 38])
top = np.array([47, 13])

# right,left,top
src = np.array([
                   [78, 36],
                   [12, 37],
                   [47, 13],
               ], dtype=np.float32)

testImg = cv2.imread('testeyeoutline1_1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
testIdx = np.vstack(testImg.nonzero()).astype(np.float32)[::-1] # 2xM matrix , x co-ords in 2nd row
testIdx = testIdx.T

testIdxXsort = testIdx[testIdx[:, 0].argsort(), :]
testIdxYsort = testIdx[testIdx[:, 1].argsort(), :]


bestWarp = None
bestResult = 0
for i in xrange(100000):

    right = randint(testIdxXsort.shape[0] * 3 / 4, testIdxXsort.shape[0] - 1)
    # print testIdxXsort[right]
    dst = np.vstack((
        testIdxXsort[right],
        testIdxXsort[randint(0, testIdxXsort.shape[0] / 4)],
        testIdxYsort[randint(0, testIdxYsort.shape[0] / 4)],
    ))

    # print dst
    # print src
    # pdb.set_trace()
    affine = cv2.getAffineTransform(src, dst)
    warpedImg = cv2.warpAffine(img, affine, (testImg.shape[1], testImg.shape[0]))
    result = np.sum(np.sum(warpedImg * testImg))
    if result > bestResult:
        # print dst
        # print src
        bestWarp = warpedImg
        bestResult = result
        print result


# testImg = cv2.cvtColor(testImg, cv2.COLOR_GRAY2RGB)
# cv2.circle(testImg, tuple(offset + origin[::-1]), 2, (0, 255, 0))
# cv2.circle(testImg, tuple(offset + top), 2, (0, 255, 0))
# cv2.circle(testImg, tuple(offset + bot), 2, (0, 255, 0))
# cv2.circle(testImg, tuple(offset + left), 2, (0, 255, 0))

plt.subplot(221), plt.imshow(img, cmap=cm.Greys_r)
plt.subplot(222), plt.imshow(testImg)
plt.subplot(223), plt.imshow(warpedImg, cmap=cm.Greys_r)

plt.show()