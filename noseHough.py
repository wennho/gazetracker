from imports import *
import time
from eyeHough import detectHough

def getNoseTemplate():
    img = cv2.imread('noseOutline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    index = np.vstack(img.nonzero()) # 2xM matrix , x co-ords in 2nd row

    origin = np.array([21, 70]) #y, then x
    template = {
        'right': origin,
        'left': np.array([2, 21]),
        'bot': np.array([37, 21]),
        'top': np.array([36, 5]),
        'tileTup': (index.shape[1], 1),
        'directions': np.tile(origin, (index.shape[1], 1)) - index.T,
    }
    return template


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Usage: python ' + __file__ + ' <image> <isLeft>'
        sys.exit()


    template = getNoseTemplate()
    testImg = cv2.imread(sys.argv[1], cv2.CV_LOAD_IMAGE_COLOR)
    grayImg = cv2.cvtColor(testImg, cv2.COLOR_BGR2GRAY)

    start = time.clock()
    offset = detectHough(grayImg, template, True)
    end = time.clock()
    print 'time elapsed:', end - start