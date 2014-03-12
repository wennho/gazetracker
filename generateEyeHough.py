from imports import *
import time


def getLeftEyeTemplate():
    img = cv2.imread('leftOutline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    index = np.vstack(img.nonzero()) # 2xM matrix , x co-ords in 2nd row

    origin = np.array([36, 78]) #y, then x
    template = {
        'origin': origin,
        'left': np.array([12, 37]), #x, then y
        'bot': np.array([45, 38]),
        'top': np.array([47, 13]),
        'tileTup': (index.shape[1], 1),
        'directions': np.tile(origin, (index.shape[1], 1)) - index.T,
    }
    return template


def getRightEyeTemplate():
    img = cv2.imread('rightOutline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    index = np.vstack(img.nonzero()) # 2xM matrix , x co-ords in 2nd row

    origin = np.array([34, 82]) #y, then x
    template = {
        'origin': origin,
        'left': np.array([7, 39]), #x, then y
        'bot': np.array([40, 38]),
        'top': np.array([41, 12]),
        'tileTup': (index.shape[1], 1),
        'directions': np.tile(origin, (index.shape[1], 1)) - index.T,
    }
    return template


def eyeHough(grayImg, template, verbose=False):
    highThreshold = 60
    cannyEdges = cv2.Canny(grayImg, highThreshold * 0.5, highThreshold)
    testIdx = np.vstack(cannyEdges.nonzero())
    houghImg = np.zeros((cannyEdges.shape[0], cannyEdges.shape[1]))

    for idx in testIdx.T:
        colorIdx = template['directions'] + np.tile(idx, template['tileTup'])

        # limit indexes to those within image
        colorIdx = colorIdx[np.logical_and(colorIdx[:, 0] >= 0, colorIdx[:, 0] < houghImg.shape[0])]
        colorIdx = colorIdx[np.logical_and(colorIdx[:, 1] >= 0, colorIdx[:, 1] < houghImg.shape[1])]

        houghImg[colorIdx[:, 0], colorIdx[:, 1]] += 1

    correctIdx = np.unravel_index(np.argmax(houghImg), houghImg.shape)
    offset = (correctIdx - template['origin'])[::-1]

    if verbose:
        print offset
        grayImg = cv2.cvtColor(grayImg, cv2.COLOR_GRAY2RGB)
        cv2.circle(grayImg, tuple(offset + template['origin'][::-1]), 2, (0, 255, 0))
        cv2.circle(grayImg, tuple(offset + template['top']), 2, (0, 255, 0))
        cv2.circle(grayImg, tuple(offset + template['bot']), 2, (0, 255, 0))
        cv2.circle(grayImg, tuple(offset + template['left']), 2, (0, 255, 0))
        # plt.subplot(221), plt.imshow(img, cmap=cm.Greys_r)
        plt.subplot(221), plt.imshow(grayImg)
        plt.subplot(222), plt.imshow(cannyEdges)
        plt.subplot(223), plt.imshow(houghImg, cmap=cm.Greys_r)

        plt.show()

    return offset


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Usage: python ' + __file__ + ' <image> <isLeft>'
        sys.exit()

    if int(sys.argv[2]) > 0:
        template = getLeftEyeTemplate()
    else:
        template = getRightEyeTemplate()
    testImg = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_COLOR)
    grayImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

    start = time.clock()
    offset = eyeHough(grayImg, template, True)
    end = time.clock()
    print 'time elapsed:', end - start