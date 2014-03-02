import numpy as np
import cv2
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import sys


for i in xrange(0,9):


    newImg = np.zeros(img.shape)
    newImg[img < 48] = 1
    plt.subplot(921+i), plt.imshow(img)
    plt.subplot(222), plt.imshow(newImg)
    plt.subplot(223)

plt.show()