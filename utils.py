import math


def point3DDistance(x1, y1, z1, x2, y2, z2):
    """Simple 3D distance calculation, given the coordinates (x1,y1,z1) and (x2,y2,z2) of the two points"""
    diff_x = x1 - x2
    diff_y = y1 - y2
    diff_h = z1 - z2
    return math.sqrt(pow(diff_x, 2) + pow(diff_y, 2) + pow(diff_h, 2))


def newMod(a, b):
    """Implements the non-euclidean version of modulo"""
    res = a % b
    return res if not res else res - b if a < 0 else res
