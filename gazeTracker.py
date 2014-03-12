from imports import *
from extract_feature import extractFeatures, getEyeFacePos
from generateEyeHough import *
import pickle
from collect_calibrate import isKey
from detectEyeShape import getHoughCircle

WINDOW_NAME = "GazeTracker"


def getAndDrawHoughEye(image, template, posTuple):
    (xMin, yMin), (xMax, yMax) = posTuple
    grayEye = cv2.cvtColor(image[yMin:yMax, xMin:xMax], cv2.COLOR_BGR2GRAY)
    offset = eyeHough(grayEye, template)
    offset += np.array([xMin, yMin])
    botRight = offset + template['origin'][::-1] + np.array([5, 5])
    posTuple = (
        (
            offset[0] + template['left'][0] - 5,
            offset[1] + template['top'][1] - 10,
        ),
        tuple(botRight)
    )

    circle = getHoughCircle(grayEye)
    circle = np.around(circle + np.array([xMin, yMin, 0])).astype(int)   # round for display only

    cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)
    cv2.circle(image, (circle[0], circle[1]), 1, (0, 255, 0), 1)

    cv2.circle(image, tuple(offset + template['origin'][::-1]), 2, (0, 255, 0))
    cv2.circle(image, tuple(offset + template['top']), 2, (0, 255, 0))
    cv2.circle(image, tuple(offset + template['bot']), 2, (0, 255, 0))
    cv2.circle(image, tuple(offset + template['left']), 2, (0, 255, 0))

    return {
        'offset': offset,
        'posTuple': posTuple,
        'circle': circle,
    }


def getEyeTrackTemplate(cap, templateLeft, templateRight):
    haveEyeFacePos = False
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    while not haveEyeFacePos:

        for i in range(5):
            ret, image = cap.read()

        result = getEyeFacePos(image, face_cascade, eye_cascade)
        if not result:
            continue

        cv2.putText(image, 'ok with eye boundaries?', (100, 500), font, 1, (255, 255, 255), 2, cv2.CV_AA)


        # process eyes
        houghResult = getAndDrawHoughEye(image, templateLeft, result['eyeLeft'])
        result['eyeLeft'] = houghResult['posTuple']
        houghResult = getAndDrawHoughEye(image, templateRight, result['eyeRight'])
        result['eyeRight'] = houghResult['posTuple']

        # draw rectangles around eye
        cv2.rectangle(image, result['eyeLeft'][0], result['eyeLeft'][1], (0, 255, 0), 1)
        cv2.rectangle(image, result['eyeRight'][0], result['eyeRight'][1], (0, 255, 0), 1)

        cv2.imshow(WINDOW_NAME, image)

        while not haveEyeFacePos:
            key = cv2.waitKey(100)
            if isKey(key, 'enter'):
                haveEyeFacePos = True
            elif isKey(key, 'space'):
                break
            elif isKey(key, 'esc'):
                cap.release()
                cv2.destroyAllWindows()
                sys.exit()

    return result


if __name__ == "__main__":

    # load all needed stuff
    font = cv2.FONT_HERSHEY_COMPLEX
    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)
    learnResult = pickle.load(open('learnResult.pickle', 'rb'))
    xSVM = learnResult['xSVM']
    ySVM = learnResult['ySVM']

    templateLeft = getLeftEyeTemplate()
    templateRight = getRightEyeTemplate()

    result = getEyeTrackTemplate(cap, templateLeft, templateRight)

    framesSkipped = 0
    while True:


        ret, image = cap.read()

        if framesSkipped < 5:
            framesSkipped += 1
            continue

        # do all processing here


        # featX, featY = extractFeatures(image, -1, False)
        # features = np.hstack((featX, featY))
        # xOrig = xSVM.predict(features);
        # x = np.clip(xOrig, 0, 1920)
        # yOrig = ySVM.predict(features)
        # y = np.clip(yOrig, 0, 1080)
        # predict = (x, y)
        # print (xOrig[0], yOrig[0])
        # cv2.circle(image, predict, 10, (0, 255, 0))

        # featX, featY = extractFeatures(image, face_cascade, eye_cascade, False)


        # draw rectangles around eyes
        houghResult = getAndDrawHoughEye(image, templateLeft, result['eyeLeft'])
        result['eyeLeft'] = houghResult['posTuple']
        houghResult = getAndDrawHoughEye(image, templateRight, result['eyeRight'])
        result['eyeRight'] = houghResult['posTuple']

        cv2.imshow(WINDOW_NAME, image)

        key = cv2.waitKey(10)

        if isKey(key, 'enter') or isKey(key, 'esc'):
            break
        elif isKey(key, 'space'):
            result = getEyeTrackTemplate(cap, templateLeft, templateRight)

    cv2.destroyAllWindows()
