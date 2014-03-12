from subprocess import check_call
from detectEyeShape import getEyeFeatures
import cv2

for i in xrange(9):
    for eyeNum in xrange(1, 3):
        imgFile = 'testeye' + str(eyeNum) + '_' + str(i) + '.png'
        origImg = cv2.imread(imgFile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        getEyeFeatures(origImg, False, (str(eyeNum), str(i)))


    
