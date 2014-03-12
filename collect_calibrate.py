import numpy as np
import cv2


SCREEN_SIZE = (1920, 1080)

pd = 50  # padding
circleLoc = [
    # first row
    (pd, pd),
    (SCREEN_SIZE[0] / 2, pd),
    (SCREEN_SIZE[0] - pd, pd),

    # middle row
    (pd, SCREEN_SIZE[1] / 2),
    (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2),
    (SCREEN_SIZE[0] - pd, SCREEN_SIZE[1] / 2),

    # bottom row
    (pd, SCREEN_SIZE[1] - pd),
    (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] - pd),
    (SCREEN_SIZE[0] - pd, SCREEN_SIZE[1] - pd),
]


def isKey(num, keyString):
    if keyString == 'enter':
        return num == 1113997 or num == 1048586
    elif keyString == 'esc':
        return num == 1048603
    elif keyString == 'space':
        return num == 1048608

    return True


WINDOW_NAME = "GazeTracker"

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    state = 0

    while True:

        ret, image = cap.read()
        cv2.circle(image, circleLoc[state], 10, (0, 255, 0))

        key = cv2.waitKey(10)
        cv2.imshow(WINDOW_NAME, image)

        if isKey(key, 'esc'):
            break
        elif not isKey(key, 'enter'):
            continue

        ret, image = cap.read()
        cv2.imwrite('calibrate_' + str(state) + '.png', image)

        state += 1
        if state > len(circleLoc) - 1:
            break

    cv2.destroyAllWindows()
