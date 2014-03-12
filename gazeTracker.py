from imports import *
from extract_feature import extractFeatures
import pickle
from collect_calibrate import isKey

WINDOW_NAME = "GazeTracker"

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    learnResult = pickle.load(open('learnResult.pickle', 'rb'))
    xSVM = learnResult['xSVM']
    ySVM = learnResult['ySVM']

    while True:

        ret, image = cap.read()
        featX, featY = extractFeatures(image, -1, False)
        features = np.hstack((featX, featY))
        xOrig = xSVM.predict(features);
        x = np.clip(xOrig, 0, 1920)
        yOrig = ySVM.predict(features)
        y = np.clip(yOrig, 0, 1080)
        predict = (x, y)
        print (xOrig[0], yOrig[0])
        cv2.circle(image, predict, 10, (0, 255, 0))

        cv2.imshow(WINDOW_NAME, image)

        key = cv2.waitKey(10)
        toExit = False
        while not isKey(key, 'enter'):
            key = cv2.waitKey(10)

            if isKey(key, 'esc'):
                toExit = True
                break
        if toExit:
            break

    cv2.destroyAllWindows()
