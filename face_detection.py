import numpy as np
import cv2


CAMERA_INDEX = 0

if __name__ == "__main__":
    capture = cv2.VideoCapture(CAMERA_INDEX)
    faces = []

    count = 0
    key = -1

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cv2.namedWindow("GazeTracker", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("GazeTracker", cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

    while key == -1:
        flag, image = capture.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Only run the Detection algorithm every 5 frames to improve performance
        if count % 5 == 0:
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if faces is not None:
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = image[y:y + h, x:x + w]
                if count % 5 == 0:
                    eyes = eye_cascade.detectMultiScale(roi_gray)

                # take top 2 occurrences only
                if len(eyes) > 2:
                    eyes = sorted(eyes,key=lambda e: e[1])[:2]

                for (ex, ey, eh, ew) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)


        cv2.imshow('image (press ESC to quit)', image)
        key = cv2.waitKey(10)
        count += 1

    cv2.destroyAllWindows()
