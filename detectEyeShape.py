from imports import *
import math
from calculateScreenFromPupil import calculateXprime


def nothing(x):
    pass


def getEyeFeatures(origImg, verbose):
    result = {}
    numLines = origImg.shape[0]
    startLine = int(numLines * 0.25)
    endLine = int(numLines * 0.75)
    origImg = origImg[startLine: endLine]

    imgCenter = (origImg.shape[0] / 2, origImg.shape[1] / 2)
    img = cv2.GaussianBlur(origImg, (0, 0), 1)

    # apply thresholding to get eye shape
    thresholdImg = np.zeros(img.shape)
    thresholdImg[img < 55] = 150

    cannyBlurImg = cv2.GaussianBlur(origImg, (0, 0), 2)
    highThreshold = 70
    cannyEdges = cv2.Canny(cannyBlurImg, highThreshold * 0.5, highThreshold)
    combined = cannyEdges * 0.5 + thresholdImg * 0.5

    index = np.vstack(combined.nonzero()) # 2xM matrix
    index[(0, 1), :] = index[(1, 0), :]
    index = index[:, index[1].argsort()[::-1]]
    index = np.concatenate((index, np.ones((1, index.shape[1]), dtype=int)), axis=0)

    XmaxIdx = np.argmax(index[0])
    XminIdx = np.argmin(index[0])

    result['left'] = index[:2, XminIdx]
    result['right'] = index[:2, XmaxIdx]

    diff = index[:, XmaxIdx] - index[:, XminIdx]
    angle = math.degrees(math.atan2(diff[1], diff[0]))

    # get top and bottom eye limits from rotated view
    rotation = cv2.getRotationMatrix2D(imgCenter, angle, 1)
    rotIdx = rotation.dot(index)

    rotImg = cv2.warpAffine(img, rotation, img.shape[::-1])
    yMax = np.max(rotIdx[1])
    yMin = np.min(rotIdx[1])

    rotXavg = (rotIdx[0, XmaxIdx] + rotIdx[0, XminIdx]) / 2.
    yOrigPoints = np.array([
        [rotXavg, rotXavg],
        [yMin, yMax],
        [1, 1],
    ])

    yPoints = cv2.getRotationMatrix2D(imgCenter, -angle, 1).dot(yOrigPoints)
    result['bottom'] = yPoints[:, 1]  # use yMax since y-axis increases downwards
    result['top'] = yPoints[:, 0]

    # Hough circle detection

    # set a low threshold (param2) so that we are guaranteed at least 1. Set a high minimum distance (1000)
    # so that we have at most 1
    circles = cv2.HoughCircles(origImg, cv2.cv.CV_HOUGH_GRADIENT, 1, 1000, param1=50, param2=1, minRadius=10,
                               maxRadius=15)

    # virtually guaranteed one circle. raise an error otherwise
    circle = circles[0, 0]
    result['pupil'] = result['bottom'] - circle[0:2]

    if verbose:

        # convert origImg back to color
        origImg = cv2.cvtColor(origImg, cv2.COLOR_GRAY2RGB)

        print 'Eye angle:', angle

        yPoints = yPoints.astype(int)

        keypoints = [
            tuple(index[:2, XminIdx]),
            tuple(index[:2, XmaxIdx]),
            tuple(yPoints[:, 0]),
            tuple(yPoints[:, 1]),
        ]

        # draw keypoint in rotated image
        yOrigPoints = yOrigPoints[:2].astype(int)
        cv2.circle(rotImg, tuple(yOrigPoints[:, 0]), 2, (0, 255, 0))
        cv2.circle(rotImg, tuple(yOrigPoints[:, 1]), 2, (0, 255, 0))

        # draw pupil center
        circle = np.around(circle)   # round for display only
        cv2.circle(origImg, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)
        cv2.circle(origImg, (circle[0], circle[1]), 1, (0, 255, 0), 1)

        # draw keypoints
        for x in keypoints:
            cv2.circle(origImg, x, 1, (255, 0, 0))
        print keypoints

        plt.subplot(321), plt.imshow(img, cmap=cm.Greys_r)
        plt.subplot(322), plt.imshow(combined)
        plt.subplot(323), plt.imshow(cannyEdges)
        plt.subplot(324), plt.imshow(cannyBlurImg, cmap=cm.Greys_r)
        plt.subplot(325), plt.imshow(origImg)
        plt.subplot(326), plt.hist(img.flatten(), bins=30)
        plt.show()

    return result


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'Usage: python ' + __file__ + ' <image>'
        sys.exit()

    imgFile = sys.argv[1]
    origImg = cv2.imread(imgFile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    result = getEyeFeatures(origImg, True)
    print result