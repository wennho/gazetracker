from imports import *
from extract_feature import extractFeatures, getEyeFacePos
from eyeHough import *
import pickle
from collect_calibrate import isKey, circleLoc
from detectEyeShape import getHoughCircle

WINDOW_NAME = "GazeTracker"

font = cv2.FONT_HERSHEY_COMPLEX


def getAndDrawHough(image, template, posTuple):
    (xMin, yMin), (xMax, yMax) = posTuple
    grayEye = cv2.cvtColor(image[yMin:yMax, xMin:xMax], cv2.COLOR_BGR2GRAY)
    offset = detectHough(grayEye, template)
    offset += np.array([xMin, yMin])
    botRight = offset + template['right'][::-1] + np.array([5, 10])
    posTuple = (
        (
            offset[0] + template['left'][0] - 5,
            offset[1] + template['top'][1] - 5,
        ),
        tuple(botRight)
    )

    diff = None
    if 'isEye' in template:
        circleOrig = getHoughCircle(grayEye)
        circle = np.around(circleOrig + np.array([xMin, yMin, 0])).astype(int)   # round for display only
        cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 1)
        cv2.circle(image, (circle[0], circle[1]), 1, (0, 255, 0), 1)
        diff = offset + template['bot']
        diff[0] = circle[0] - diff[0]
        diff[1] = circle[1] - diff[1]

    cv2.circle(image, tuple(offset + template['right'][::-1]), 2, (0, 255, 0))
    # cv2.circle(image, tuple(offset + template['top']), 2, (0, 255, 0))
    # cv2.circle(image, tuple(offset + template['bot']), 2, (0, 255, 0))
    cv2.circle(image, tuple(offset + template['left']), 2, (0, 255, 0))



    return {
        'posTuple': posTuple,
        'features': diff,
    }


def getEyeTrackTemplate(cap, templateLeft, templateRight, templateNose):
    haveEyeFacePos = False
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    while not haveEyeFacePos:

        for i in range(5):
            ret, image = cap.read()

        image = cv2.flip(image, 1)
        result = getEyeFacePos(image, face_cascade, eye_cascade)
        if not result:
            continue

        cv2.putText(image, 'ok with eye boundaries?', (100, 500), font, 1, (255, 255, 255), 2, cv2.CV_AA)


        # process eyes
        houghResult = getAndDrawHough(image, templateLeft, result['eyeLeft'])
        result['eyeLeft'] = houghResult['posTuple']
        houghResult = getAndDrawHough(image, templateRight, result['eyeRight'])
        result['eyeRight'] = houghResult['posTuple']
        houghNose = getAndDrawHough(image, templateNose, result['nose'])
        result['nose'] = houghNose['posTuple']

        # draw rectangles around eyes, nose
        cv2.rectangle(image, result['eyeLeft'][0], result['eyeLeft'][1], (0, 255, 0), 1)
        cv2.rectangle(image, result['eyeRight'][0], result['eyeRight'][1], (0, 255, 0), 1)
        cv2.rectangle(image, result['nose'][0], result['nose'][1], (0, 255, 0), 1)

        cv2.imshow(WINDOW_NAME, image)

        result['initPos'] = (
            result['eyeLeft'][0][0] + result['eyeRight'][0][0],
            result['eyeLeft'][0][1] + result['eyeRight'][0][1],
        )

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
