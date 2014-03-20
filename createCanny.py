from imports import *

for i in range(9):
    imgFile = 'calibrate_' + str(i) + '.png'
    image = cv2.imread(imgFile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    highThreshold = 60
    cannyEdges = cv2.Canny(image, highThreshold * 0.5, highThreshold)
    cv2.imwrite('calibrate_' + str(i) + '.png', cannyEdges)
    print 'finished', i