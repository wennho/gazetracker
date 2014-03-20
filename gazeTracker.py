from helpers import *
from random import randint
from learnAll import learn
from noseHough import getNoseTemplate

if __name__ == "__main__":

    # load all needed stuff

    record = True

    if record:
        out = cv2.VideoWriter('output.avi', -1, 20, (640, 480), True)

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    templateLeft = getLeftEyeTemplate()
    templateRight = getRightEyeTemplate()
    templateNose = getNoseTemplate()

    result = getEyeTrackTemplate(cap, templateLeft, templateRight, templateNose)

    calibrating = True
    isInitialCalibrate = True
    displayCapture = False
    calibrateState = 0
    framesSkipped = 0

    # calibrateData = np.zeros((9, 2))
    calibrateData = np.zeros((9, 4))
    svmResult = None
    captureRight = None
    captureLeft = None
    labels = None
    calibrateTarget = circleLoc[calibrateState]

    x = 960
    y = 540
    while True:


        ret, image = cap.read()

        if framesSkipped < 10:
            framesSkipped += 1
            continue

        image = cv2.flip(image, 1)

        # draw rectangles around eyes
        houghLeft = getAndDrawHough(image, templateLeft, result['eyeLeft'])
        result['eyeLeft'] = houghLeft['posTuple']
        houghRight = getAndDrawHough(image, templateRight, result['eyeRight'])
        result['eyeRight'] = houghRight['posTuple']
        data = np.hstack((houghLeft['features'], houghRight['features']))

        houghNose = getAndDrawHough(image, templateNose, result['nose'])
        result['nose'] = houghNose['posTuple']

        # data = np.array([
        #     data[0] + result['eyeLeft'][0][0],
        #     data[2] + result['eyeRight'][0][0],
        #     data[1] + result['eyeLeft'][0][1],
        #     data[3] + result['eyeRight'][0][1],
        # ])
        data = data[[0, 2, 1, 3]]

        if calibrating:
            cv2.circle(image, calibrateTarget, 10, (0, 255, 255), 3)
            if displayCapture:
                image[50:50 + captureLeft.shape[0], 1500:1500 + captureLeft.shape[1]] = captureLeft
                image[200:200 + captureRight.shape[0], 1500:1500 + captureRight.shape[1]] = captureRight

        elif svmResult:
            # make prediction
            # x = svmResult['xSVM'].predict(data)
            # y = svmResult['ySVM'].predict(data)
            newX = svmResult['xSVM'].predict(data[0:2])
            newY = svmResult['ySVM'].predict(data[2:4])

            x = int(x * 0.7 + newX * 0.3)
            y = int(y * 0.7 + newY * 0.3)

            cv2.putText(image, str((int(x), int(y))), (100, 900), font, 1, (255, 255, 255), 2, cv2.CV_AA)
            cv2.circle(image, (x, y), 10, (255, 0, 0), 3)

        cv2.imshow(WINDOW_NAME, image)

        if record:
            out.write(image)

        key = cv2.waitKey(10)

        if isKey(key, 'enter'):
            print calibrateState
            if calibrating and displayCapture:
                displayCapture = False

                # add results to calibration, set next target
                if isInitialCalibrate:
                    calibrateData[calibrateState] = data
                    calibrateState += 1
                    if calibrateState <= 8:
                        calibrateTarget = circleLoc[calibrateState]
                else:
                    calibrateData = np.vstack((calibrateData, data))
                    calibrateState += 1
                    labels = np.vstack((labels, np.array(calibrateTarget)))
                    calibrateTarget = randint(20, 1900), randint(20, 1060)

                if isInitialCalibrate and calibrateState >= 9:  # finished calibration
                    calibrating = False
                    isInitialCalibrate = False
                    labels = np.array(circleLoc)
                    # svmResult = learn(calibrateData, calibrateData, labels)
                    svmResult = learn(calibrateData[:, 0:2], calibrateData[:, 2:4], labels)
                elif not isInitialCalibrate and calibrateState >= 5:
                    calibrating = False
                    print calibrateData
                    print labels
                    svmResult = learn(calibrateData[:, 0:2], calibrateData[:, 2:4], labels)
            elif calibrating:
                displayCapture = True
                (xMin, yMin), (xMax, yMax) = result['eyeLeft']
                captureLeft = image[yMin:yMax, xMin:xMax]
                (xMin, yMin), (xMax, yMax) = result['eyeRight']
                captureRight = image[yMin:yMax, xMin:xMax]
            else:
                calibrating = True
                calibrateTarget = randint(20, 1900), randint(20, 1060)
                calibrateState = 0
        elif isKey(key, 'esc'):
            break
        elif isKey(key, 'space'):
            if calibrating and displayCapture:
                displayCapture = False
            else:
                result = getEyeTrackTemplate(cap, templateLeft, templateRight, templateNose)

    cv2.destroyAllWindows()
    if record:
        out.release()

