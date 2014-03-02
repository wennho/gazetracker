import numpy as np
import cv2
import sys
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm


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
    print "Found faces"

    # only process one face
    if len(faces) == 0:
        sys.exit()

    (x, y, w, h) = faces[0]
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = image[y:y + h, x:x + w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    print "Found eyes"

    # take top 2 occurrences only
    if len(eyes) > 2:
        eyes = sorted(eyes, key=lambda e: e[1])[:2]

    # blurImg = roi_gray
    blurImg = cv2.medianBlur(roi_gray, 3)
    # blurImg = cv2.GaussianBlur(roi_gray, (5,5), 4)

    eyeNum = 0
    eyeImage = np.zeros(( max(eyes[0][1], eyes[1][1]), eyes[0][0] + eyes[1][0], 3))
    for (ex, ey, ew, eh) in eyes:
        ex += 3  # offset some
        eyeNum += 1


        # detect pupils
        eye_gray = blurImg[ey:ey + eh, ex:ex + ew]
        eye_notblur = roi_color[ey:ey + eh, ex:ex + ew]

        if len(sys.argv) > 2:
            cv2.imwrite('testeye' + str(eyeNum) + '_' + sys.argv[2] + '.png', eye_notblur)

        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        eye_color = roi_color[ey:ey + eh, ex:ex + ew]

        print np.min(eye_gray)
        print np.max(eye_gray)

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
    # surf = cv2.SURF(400)
    # surf.extended = True
    # surf.upright = True
    # kp, des = surf.detectAndCompute(blurImg, None)
    # roi_color = cv2.drawKeypoints(roi_color, kp, None, (255, 0, 0), 4)

    # edges = cv2.Canny(roi_gray, 30,50)
    # cv2.imshow('image (press any key to quit)', roi_color)
    # cv2.imshow('edges (press any key to quit)', edges)

    roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

    # fig = plt.figure()
    # a = fig.add_subplot(2, 2, 1)
    # plt.imshow(roi_color)
    #
    # a = fig.add_subplot(2, 2, 2)
    #
    # a = fig.add_subplot(2, 2, 3)
    #
    # a = fig.add_subplot(2, 2, 4)
    # plt.imshow(roi_gray, cmap=cm.Greys_r)
    plt.imshow(roi_color)
    plt.show()

    # key = -1
    # while key == -1:
    #     key = cv2.waitKey(100)

    # cv2.destroyAllWindows()
