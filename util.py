import numpy as np
import math

def rotX(ang):
    angRad = math.radians(ang)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
        [1,  0,   0, 0],
        [0, c0, -s0, 0],
        [0, s0,  c0, 0],
        [0,  0,   0, 1],
    ], dtype=float)

def rotY(ang):
    angRad = math.radians(ang)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
        [ c0, 0, s0, 0],
        [  0, 1,  0, 0],
        [-s0, 0, c0, 0],
        [  0, 0,  0, 1],
    ], dtype=float)

def rotZ(ang):
    angRad = math.radians(ang)
    c0 = math.cos(angRad)
    s0 = math.sin(angRad)
    return np.array([
        [c0, -s0, 0, 0],
        [s0,  c0, 0, 0],
        [ 0,   0, 1, 0],
        [ 0,   0, 0, 1],
    ], dtype=float)

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
        [0, 0, 1./d, 1],
    ], dtype=float)


def getProjection(rot_x, rot_y, rot_z, d, t):
    # t[2] += 10
    return postPerspective(d).dot(trans(t)).dot(rotZ(rot_z)).dot(rotY(rot_y)).dot(rotX(rot_x)).dot(prePerspective(d))