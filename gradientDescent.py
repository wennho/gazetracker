import numpy as np
import cv2
from matplotlib import pyplot as plt
from util import *
import scipy.optimize as opt
import matplotlib.cm as cm
import pdb
import sys

"""
Given x = [I 0]q and y = [R t]q, we want to find
y = (R - tv^T)x
   = Hx
where v is the normal to the plane that the eye lies in
"""

DEPTH = 0
ROT_X = 1
ROT_Y = 2
ROT_Z = 3
T_X = 4
T_Y = 5
T_Z = 6

N = 7

# Establish bounds for values
ROT_X_MAX = 15
ROT_X_MIN = -40
ROT_Y_MAX = 30
ROT_Y_MIN = -30
ROT_Z_MAX = 25
ROT_Z_MIN = -25
T_X_MAX = 45
T_Y_MAX = 30
T_Z_MAX = 40
T_Z_MIN = -40
D_MIN = 500
D_MAX = 2000

bounds = range(N)
bounds[ROT_X] = (ROT_X_MIN, ROT_X_MAX)
bounds[ROT_Y] = (ROT_Y_MIN, ROT_Y_MAX)
bounds[ROT_Z] = (ROT_Z_MIN, ROT_Z_MAX)
bounds[T_X] = (-T_X_MAX, T_X_MAX)
bounds[T_Y] = (-T_Y_MAX, T_Y_MAX)
bounds[T_Z] = (T_Z_MIN, T_Z_MAX)
bounds[DEPTH] = (D_MIN, D_MAX)

V_norm = np.array([0., 0., -0.001])

tmpl = cv2.imread('eye2_outline.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
origImg = cv2.imread('testeye2_2.png', cv2.CV_LOAD_IMAGE_GRAYSCALE)
img = origImg.astype(float)

i, j = tmpl.nonzero()
x_idx = np.vstack((i, j, np.ones((i.size,))))

# pre-processing to turn img into a convex map
img = cv2.GaussianBlur(img, (0, 0), 5)

max_i = img.shape[0]
max_j = img.shape[1]


preH = np.array([
                    [1, 0, -tmpl.shape[1] * 0.5],
                    [0, 1, -tmpl.shape[0] * 0.5],
                    [0, 0, 1],
                ], dtype=float)

postH = np.array([
                     [1, 0, tmpl.shape[1] * 0.5],
                     [0, 1, tmpl.shape[0] * 0.5],
                     [0, 0, 1],
                 ], dtype=float)


def computeH(rot_x, rot_y, rot_z, d, t):
    H = getProjection(rot_x, rot_y, rot_z, d, t)
    H = postH.dot(H).dot(preH)
    return H

def getHfromx(x):
    # x *= 25.0
    t = x[T_X: T_Z + 1]
    # t[0] = -T_X_MAX + 2 * T_X_MAX * t[0]
    # t[1] = -T_Y_MAX + 2 * T_Y_MAX * t[1]
    # t[2] = -T_Z_MAX + 2 * T_Z_MAX * t[2]

    # rot_x = ROT_X_MIN + (ROT_X_MAX - ROT_X_MIN ) * x[3]
    # rot_y = ROT_Y_MIN + (ROT_Y_MAX - ROT_Y_MIN ) * x[4]
    # rot_z = ROT_Z_MIN + (ROT_Z_MAX - ROT_Z_MIN ) * x[5]

    d = x[DEPTH]
    rot_x = x[ROT_X]
    rot_y = x[ROT_Y]
    rot_z = x[ROT_Z]
    return computeH(rot_x, rot_y, rot_z, d, t)


def objectiveFn(x):
    H = getHfromx(x)

    idx = np.dot(H, x_idx)
    idx = idx / idx[2, :]
    idx = idx.astype(int)

    y_i = idx[0, :]
    y_j = idx[1, :]

    toUse = np.fromiter((k for k in xrange(1, y_i.size)
                         if 0 <= y_i[k] < max_i and 0 <= y_j[k] < max_j),
                        dtype=int)

    if toUse.size == 0:
        return 10000

    y_i = y_i[toUse]
    y_j = y_j[toUse]
    result = img[y_i, y_j]
    result = np.dot(result, result)
    error = result / float(toUse.size)
    print error
    return error


x0 = np.zeros((len(bounds),), dtype=float)
x0[DEPTH] = 100

def gradientDescent(x):
    count = 0
    score = objectiveFn(x)
    iter = 0
    while count < 100:
        count += 1
        iter += 1
        perturb = (np.random.sample((N,)) - 0.5)
        newX = x + perturb
        newScore = objectiveFn(newX)
        if newScore < score:
            x = newX
            score = newScore
            count = 0
    return x, score, iter


x, f, d = opt.fmin_l_bfgs_b(objectiveFn, x0, approx_grad=True, bounds=bounds, iprint=1)

if d['warnflag'] != 0:
    print 'Did not achieve convergence'
    sys.exit()



# x, score, iter = gradientDescent(x0)
#
# print 'x:', x
# print 'iterations:', iter

H = getHfromx(x)

# H = getHfromx(x0)

newTmpl = cv2.warpPerspective(tmpl, H, (66, 66))

plt.subplot(221), plt.imshow(tmpl)
plt.subplot(222), plt.imshow(newTmpl)
plt.subplot(223), plt.imshow(origImg)
plt.subplot(224), plt.imshow(img)

# plt.subplot(241), plt.imshow(tmpl)
# plt.subplot(242), plt.imshow(newTmpl)
# for i in xrange(1, 7):
#     H = computeH(i * 10, 0., 0., x0[DEPTH], np.array([0, 0, 0], dtype=float))
#     print H
#     newTmpl = cv2.warpPerspective(tmpl, H, (66, 66))
#     plt.subplot(242 + i), plt.imshow(newTmpl)

plt.show()
