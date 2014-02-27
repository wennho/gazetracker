import numpy as np
import cv2


CAMERA_INDEX = 0
SCREEN_SIZE = (1920, 1080)


def isKey(num, keyString):
    if keyString == 'enter':
        return num == 1113997 or num == 1048586
    elif keyString == 'esc':
        return num == 1048603

    return True


WINDOW_NAME = "GazeTracker"

if __name__ == "__main__":

    capture = cv2.VideoCapture(CAMERA_INDEX)
    faces = []

    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    bgImg = np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0], 3))

    pd = 50  # padding

    circleLoc = [
        (pd, pd),
        (SCREEN_SIZE[0] - pd, pd),
        (SCREEN_SIZE[0] - pd, SCREEN_SIZE[1] - pd),
        (pd, SCREEN_SIZE[1] - pd),
    ]

    canvas = np.copy(bgImg)
    cv2.circle(canvas, circleLoc[0], 10, (0, 255, 0))

    state = 0

    while True:

        key = cv2.waitKey(10)
        cv2.imshow(WINDOW_NAME, canvas)

        if isKey(key, 'esc'):
            break
        elif not isKey(key, 'enter'):
            continue

        state += 1

        if state > len(circleLoc) - 1:
            break

        canvas = np.copy(bgImg)
        cv2.circle(canvas, circleLoc[state], 10, (0, 255, 0))

        flag, image = capture.read()

    cv2.destroyAllWindows()
