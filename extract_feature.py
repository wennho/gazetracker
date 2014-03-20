from imports import *
from helpers import getAndDrawHough, getEyeFacePos

def extractFeatures(image, settings, imageNum=None):
    result = getEyeFacePos(image, settings['face_cascade'], settings['eye_cascade'])

    if imageNum:
        # save eye images
        (xMin, yMin), (xMax, yMax) = result['eyeLeft']
        eye = image[yMin:yMax, xMin:xMax]
        cv2.imwrite('learnEyeLeft_' + str(imageNum) + '.png', eye)

        (xMin, yMin), (xMax, yMax) = result['eyeRight']
        eye = image[yMin:yMax, xMin:xMax]
        cv2.imwrite('learnEyeRight_' + str(imageNum) + '.png', eye)

    houghLeft = getAndDrawHough(image, settings['templateLeft'], result['eyeLeft'])
    houghRight = getAndDrawHough(image, settings['templateRight'], result['eyeRight'])
    data = np.vstack((houghLeft['features'], houghRight['features']))

    featX = data[:, 0]
    featY = data[:, 1]

    return featX, featY


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Usage: python ' + __file__ + ' <image> <imageNum>'
        sys.exit()

    imgFile = sys.argv[1]
    image = cv2.imread(imgFile)
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    featX, featY = extractFeatures(image, sys.argv[2], True, True)
    print 'x features:', featX
    print 'y features:', featY