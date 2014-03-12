from imports import *

img = cv2.imread('leftOutline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)

#y, then x
origin = np.array([36, 78])

#x, then y
left = np.array([12, 37])
bot = np.array([45, 38])
top = np.array([47, 13])

index = np.vstack(img.nonzero()) # 2xM matrix , x co-ords in 2nd row

tileTup = (index.shape[1], 1)
directions = np.tile(origin, tileTup) - index.T
print directions

testImg = cv2.imread('testeyeoutline1_1.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
testIdx = np.vstack(testImg.nonzero())


houghImg = np.zeros((testImg.shape[0],testImg.shape[1]))

for idx in testIdx.T:
    colorIdx = directions + np.tile(idx, tileTup)
    colorIdx = colorIdx[np.logical_and(colorIdx[:, 0] >= 0, colorIdx[:, 0] < houghImg.shape[0])]
    colorIdx = colorIdx[np.logical_and(colorIdx[:, 1] >= 0, colorIdx[:, 1] < houghImg.shape[1])]

    houghImg[colorIdx[:, 0], colorIdx[:, 1]] += 1

correctIdx = np.unravel_index(np.argmax(houghImg), houghImg.shape)
offset = (correctIdx - origin)[::-1]
print offset


testImg = cv2.cvtColor(testImg, cv2.COLOR_GRAY2RGB)
cv2.circle(testImg, tuple(offset + origin[::-1]), 2, (0, 255, 0))
cv2.circle(testImg, tuple(offset + top), 2, (0, 255, 0))
cv2.circle(testImg, tuple(offset + bot), 2, (0, 255, 0))
cv2.circle(testImg, tuple(offset + left), 2, (0, 255, 0))



plt.subplot(221), plt.imshow(img, cmap=cm.Greys_r)
plt.subplot(222), plt.imshow(testImg)
plt.subplot(223), plt.imshow(houghImg, cmap=cm.Greys_r)

plt.show()