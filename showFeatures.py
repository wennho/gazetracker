import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt
import pdb


def nothing(x):
    pass


img = cv2.imread('testeye2_4.png')

cv2.namedWindow('features')
cv2.setWindowProperty('features', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

cv2.createTrackbar('HighThresh', 'features', 0, 255, nothing)
cv2.createTrackbar('LowThresh', 'features', 0, 255, nothing)
cv2.createTrackbar('SURFThresh', 'features', 0, 4000, nothing)

cv2.setTrackbarPos('HighThresh', 'features', 90)
cv2.setTrackbarPos('LowThresh', 'features', 20)
cv2.setTrackbarPos('SURFThresh', 'features', 3000)

# img = cv2.GaussianBlur(img, (9,9), 3)
img = cv2.medianBlur(img, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

while True:

    highThresh = cv2.getTrackbarPos('HighThresh', 'features')
    lowThresh = cv2.getTrackbarPos('LowThresh', 'features')
    surfThresh = cv2.getTrackbarPos('SURFThresh', 'features')
    edges = cv2.Canny(gray, lowThresh, highThresh)


    # fast = cv2.FastFeatureDetector(10)
    # kp = fast.detect(img,None)
    # img2 = cv2.drawKeypoints(img, kp, color=(255,0,0))

    surf = cv2.SURF(surfThresh)
    kp, des = surf.detectAndCompute(edges, None)
    edges = cv2.drawKeypoints(edges, kp, None, (0, 0, 255), 4)

    if len(kp) > 100:
        print 'too many features, have', len(kp)
        sys.exit()

    # combine images
    h, w, d = img.shape
    h2, w2, d2 = edges.shape

    comb = np.zeros((max(h, h2), w + w2, 3))
    comb[:h, :w] = img
    comb[:h2, w: (w + w2)] = edges

    # pdb.set_trace()
    cv2.imshow('features', comb)
    # pdb.set_trace()

    # plt.subplot(121),plt.imshow(img,cmap = 'gray')
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.show()

    key = cv2.waitKey(100)
    if key != -1:
        break

cv2.destroyAllWindows()