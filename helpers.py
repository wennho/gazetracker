from imports import *
from eyeHough import *
import pickle
from collect_calibrate import isKey, circleLoc
from detectEyeShape import getHoughCircle
from eyeCircleHough import getEyeCircle

WINDOW_NAME = "GazeTracker"

font = cv2.FONT_HERSHEY_COMPLEX

def getEyeFacePos(image, face_cascade, eye_cascade, imageSaveName=None):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    (x, y, w, h) = faces[0]

    result = {'face': (x, y, w, h)}

    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_color = image[y:y + h, x:x + w]
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    if len(eyes) < 2 or isinstance(eyes[0], (int, long, float)):
        return None

    indices = eyes[:, 1].argsort()[:2]
    eyes = eyes[indices]  # take top 2 occurrences only
    eyes = eyes[eyes[:, 0].argsort()]  # sort by x-axis

    eyeNum = 0

    for (ex, ey, ew, eh) in eyes:

        eyeNum += 1
        if eyeNum == 2:
            ew += 15

        if imageSaveName:
            eyeColorImg = roi_color[ey:ey + eh, ex:ex + ew]
            cv2.imwrite(imageSaveName + '.png', eyeColorImg)

        # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        eyeName = 'eyeLeft' if eyeNum == 1 else 'eyeRight'
        result[eyeName] = ((ex + x, int(ey + y + eh * 0.15)), (ex + x + ew, int(ey + y + eh * .85)))

    # guesstimate nose position
    x = (result['eyeLeft'][0][0] + result['eyeRight'][1][0]) / 2
    y = (result['eyeLeft'][0][1] + result['eyeRight'][1][1]) / 2 + 80
    result['nose'] = ((x - 50, y - 30), (x + 50, y + 30))

    return result



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
        # circleOrig = getHoughCircle(grayEye)
        circleOrig = np.array(getEyeCircle(grayEye))
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


        # process eyes, refine bounds
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
