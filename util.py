import numpy as np
import math


def rotX(ang):
    angRad = math.radians(ang)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
                        [1, 0, 0, 0],
                        [0, c0, -s0, 0],
                        [0, s0, c0, 0],
                        [0, 0, 0, 1],
                    ], dtype=float)


def rotY(ang):
    angRad = math.radians(ang)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
                        [c0, 0, s0, 0],
                        [0, 1, 0, 0],
                        [-s0, 0, c0, 0],
                        [0, 0, 0, 1],
                    ], dtype=float)


def rotZ(ang):
    angRad = math.radians(ang)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
                        [c0, -s0, 0, 0],
                        [s0, c0, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                    ], dtype=float)


def rotate2d(angle):
    angRad = math.radians(angle)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
        [c0, s0],
        [-s0, c0],
    ])


def trans(t):
    T = np.eye(4, dtype=float)
    T[0:3, 3] = t
    return T


def prePerspective(d):
    return np.array([
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 1],
                    ], dtype=float)


def postPerspective(d):
    return np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1. / d, 1],
                    ], dtype=float)


def getProjection(rot_x, rot_y, rot_z, d, t):
    # t[2] += 10
    return postPerspective(d).dot(trans(t)).dot(rotZ(rot_z)).dot(rotY(rot_y)).dot(rotX(rot_x)).dot(prePerspective(d))


def scale(data):
    s = np.max(data)
    return data / float(s), s


def scaleMatrix(data):
    data = data.astype(float)
    for i in range(data.shape[1]):
        data[:, i], s = scale(data[:, i])
    return data