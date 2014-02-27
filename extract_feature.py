import numpy as np
import cv2
import sys


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print 'Usage: python ' + __file__ + ' <image>'
        sys.exit()

    imgFile = sys.argv[1]
    image = cv2.imread(imgFile)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # only process one face
    if len(faces) == 0:
        sys.exit()

    (x, y, w, h) = faces[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)

    # take top 2 occurrences only
    if len(eyes) > 2:
        eyes = sorted(eyes, key=lambda e: e[1])[:2]

    # do pre-processing for better detection
    # alpha = np.array([2.0])
    # beta = np.array([-50.0])
    # # cv2.add(roi_gray, beta, roi_gray)
    # cv2.multiply(roi_gray, alpha, roi_gray)

    # blurImg = roi_gray
    blurImg = cv2.medianBlur(roi_gray, 5)
    # blurImg = cv2.GaussianBlur(roi_gray, (5,5), 4)

    eyeNum = 0
    eyeImage = np.zeros(( max(eyes[0][1], eyes[1][1]), eyes[0][0] + eyes[1][0], 3))
    for (ex, ey, eh, ew) in eyes:
        eyeNum += 1
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # detect pupils
        eye_gray = blurImg[ey:ey + eh, ex:ex + ew]
        cv2.imwrite('testeye' + str(eyeNum) + '.png', eye_gray)

        eye_color = roi_color[ey:ey + eh, ex:ex + ew]

        # set a low threshold (param2) so that we are guaranteed at least 1. Set a high minimum distance (1000)
        # so that we have at most 1
        circles = cv2.HoughCircles(eye_gray, cv2.cv.CV_HOUGH_GRADIENT, 2, 1000, param1=30, param2=10, minRadius=10,
                                   maxRadius=15)

        print circles
        if circles is not None:
            circles = np.around(circles)
            for i in circles[0, :]:
                cv2.circle(eye_color, (i[0], i[1]), i[2], (0, 255, 0), 1)
                cv2.circle(eye_color, (i[0], i[1]), 2, (0, 0, 255), 1)



    # newImg = cv2.resize(image, (960, 540))
    cv2.imshow('image (press any key to quit)', roi_color)
    # cv2.imshow('gray (press any key to quit)', roi_gray)

    key = -1
    while key == -1:
        key = cv2.waitKey(100)

    cv2.destroyAllWindows()
