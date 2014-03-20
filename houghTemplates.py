from imports import *


def getMouthTemplate():
    img = cv2.imread('mouthOutline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
    index = np.vstack(img.nonzero()) # 2xM matrix , x co-ords in 2nd row

    origin = np.array([45, 125]) #y, then x
    template = {
        'right': origin,
        'left': np.array([18, 48]), #x, then y
        # 'bot': np.array([26, 60]),
        # 'top': np.array([112, 31]),
        'tileTup': (index.shape[1], 1),
        'directions': np.tile(origin, (index.shape[1], 1)) - index.T,
    }
    return template