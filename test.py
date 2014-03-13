import numpy as np
import cv2
from subprocess import check_call
import pdb

cap = cv2.VideoCapture(0)
# print cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
# print cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

# check_call('v4l2-ctl --set-fmt-video=width=1920,height=1080,pixelformat=1', shell=True)

print cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
print cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

out = cv2.VideoWriter('output.avi', cv2.cv.FOURCC('M','J','P','G'), 20, (1920, 1080), True)
out.open('output.avi', cv2.cv.FOURCC('M','J','P','G'), 20, (1920, 1080), True)
# pdb.set_trace()
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', gray)
    out.write(frame)
    #
    # key = cv2.waitKey(10)
    # if key != -1:
    #     print key

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
out.release()