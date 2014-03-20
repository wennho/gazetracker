from imports import *
import time


def getEyeCircle(grayImg, verbose=False):
    highThreshold = 60
    cannyEdges = cv2.Canny(grayImg, highThreshold * 0.5, highThreshold)

    # radius ranges 10-15

    houghAccum = np.zeros(grayImg.shape + (6,))
    edgeIdx = np.vstack(cannyEdges.nonzero())

    xx, yy = np.mgrid[:35, :35]
    circle = np.around(np.sqrt((xx - 17) ** 2 + (yy - 17) ** 2))

    radii = range(11, 15)

    for i, radius in enumerate(radii):
        # generate circle

        circIdx = np.vstack((circle == radius).nonzero())
        circIdx = circIdx.T - 17  # zero the offset

        for idx in edgeIdx.T:
            voteIdx = circIdx + np.tile(idx, (circIdx.shape[0], 1))

            # discard those outside edges
            voteIdx = voteIdx[(voteIdx[:, 0] >= 0)
                              & (voteIdx[:, 0] < houghAccum.shape[0])
                              & (voteIdx[:, 1] >= 0)
                              & (voteIdx[:, 1] < houghAccum.shape[1])]

            houghAccum[voteIdx[:, 0], voteIdx[:, 1], i] += 1
            # plt.subplot(121), plt.imshow(cannyEdges)
            # plt.subplot(122), plt.imshow(houghAccum[:,:,i])
            # plt.show()

    # find max idx
    y, x, i = np.unravel_index(np.argmax(houghAccum), houghAccum.shape)
    radius = radii[i]

    # draw result
    if verbose:
        colorImg = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2RGB)
        cv2.circle(colorImg, (x, y), radius, (0, 255, 0), 1)
        cv2.circle(colorImg, (x, y), 1, (0, 255, 0), 1)
        plt.subplot(121), plt.imshow(cannyEdges)
        plt.subplot(122), plt.imshow(colorImg)
        plt.show()

    return (x, y, radius)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'Usage: python ' + __file__ + ' <image>'
        sys.exit()

    testImg = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_GRAYSCALE)
    start = time.clock()
    result = getEyeCircle(testImg)
    end = time.clock()
    print 'time elapsed:', end - start
    print result