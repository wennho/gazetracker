import numpy as np
import cv2
import pygame
import pygame.camera


CAMERA_INDEX = 0
SCREEN_SIZE = (1920, 1080)

pd = 50  # padding
circleLoc = [
    # first row
    (pd, pd),
    (SCREEN_SIZE[0] / 2, pd),
    (SCREEN_SIZE[0] - pd, pd),

    # middle row
    (SCREEN_SIZE[0] - pd, SCREEN_SIZE[1] / 2),
    (SCREEN_SIZE[0] / 2, SCREEN_SIZE[1] / 2),
    (pd, SCREEN_SIZE[1] / 2),

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

    return True


WINDOW_NAME = "GazeTracker"

if __name__ == "__main__":

    pygame.init()
    pygame.camera.init()

    cam = pygame.camera.Camera("/dev/video0", (1920, 1080))
    cam.start()

    cv2.namedWindow(WINDOW_NAME, 0)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    bgImg = np.zeros((SCREEN_SIZE[1], SCREEN_SIZE[0], 3))

    canvas = np.copy(bgImg)
    cv2.circle(canvas, circleLoc[0], 10, (0, 255, 0))

    state = 0

    for loc in circleLoc:

        key = cv2.waitKey(10)
        cv2.imshow(WINDOW_NAME, canvas)

        if isKey(key, 'esc'):
            break
        elif not isKey(key, 'enter'):
            continue

        for i in range(1, 4):
            while not cam.query_image():
                pygame.time.wait(100)
            img = cam.get_image()  # flush this

        image = cam.get_image()

        pygame.image.save(image, 'calibrate_' + str(state) + '.png')

        state += 1

        canvas = np.copy(bgImg)
        cv2.circle(canvas, loc, 10, (0, 255, 0))

    cv2.destroyAllWindows()
