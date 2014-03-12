from gazeTrackerHelper import *
from learnAll import learn

if __name__ == "__main__":

    # load all needed stuff

    cap = cv2.VideoCapture(0)
    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    templateLeft = getLeftEyeTemplate()
    templateRight = getRightEyeTemplate()

    result = getEyeTrackTemplate(cap, templateLeft, templateRight)

    calibrating = True
    displayCapture = False
    calibrateState = 0
    framesSkipped = 0

    calibrateData = np.zeros((9, 4))
    svmResult = None
    captureRight = None
    captureLeft = None
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
        houghLeft = getAndDrawHoughEye(image, templateLeft, result['eyeLeft'])
        result['eyeLeft'] = houghLeft['posTuple']
        houghRight = getAndDrawHoughEye(image, templateRight, result['eyeRight'])
        result['eyeRight'] = houghRight['posTuple']
        data = np.hstack((houghLeft['features'], houghRight['features']))

        if calibrating:
            cv2.circle(image, circleLoc[calibrateState], 10, (0, 255, 0))
            if displayCapture:
                image[50:50 + captureLeft.shape[0], 1500:1500 + captureLeft.shape[1]] = captureLeft
                image[200:200 + captureRight.shape[0], 1500:1500 + captureRight.shape[1]] = captureRight


        elif svmResult:
            # make prediction
            x = svmResult['xSVM'].predict(data)
            y = svmResult['ySVM'].predict(data)
            print (x, y)
            cv2.circle(image, (x, y), 10, (255, 0, 0))

        cv2.imshow(WINDOW_NAME, image)

        key = cv2.waitKey(10)

        if isKey(key, 'enter'):
            if calibrating and displayCapture:
                displayCapture = False

                # add results to calibration
                calibrateData[calibrateState] = data
                calibrateState += 1

                if calibrateState >= 9:  # finished calibration
                    calibrating = False
                    svmResult = learn(calibrateData, calibrateData)
            elif calibrating:
                displayCapture = True
                (xMin, yMin), (xMax, yMax) = result['eyeLeft']
                captureLeft = image[yMin:yMax, xMin:xMax]
                (xMin, yMin), (xMax, yMax) = result['eyeRight']
                captureRight = image[yMin:yMax, xMin:xMax]
        elif isKey(key, 'esc'):
            break
        elif isKey(key, 'space'):
            if calibrating and displayCapture:
                displayCapture = False
            else:
                result = getEyeTrackTemplate(cap, templateLeft, templateRight)

    cv2.destroyAllWindows()
