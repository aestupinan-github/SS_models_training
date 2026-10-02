"""Cube-wall reference solution.

Implements the exact analytical cube support function for overlap computation.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

CUBE_VERTICES = np.array([
    [-0.5, -0.5, -0.5],
    [-0.5, -0.5,  0.5],
    [-0.5,  0.5, -0.5],
    [-0.5,  0.5,  0.5],
    [ 0.5, -0.5, -0.5],
    [ 0.5, -0.5,  0.5],
    [ 0.5,  0.5, -0.5],
    [ 0.5,  0.5,  0.5],
], dtype=np.float64)


def canonicalize_quaternion(q):
    q = np.array(q, dtype=np.float64)
    q = q / np.linalg.norm(q)
    if q[0] < 0.0:
        q = q * -1.0
    return q.tolist()


def _rotate_vertices(q):
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    return rot.apply(CUBE_VERTICES)


def z_touch(q):
    rotated = _rotate_vertices(q)
    return -np.min(rotated[:, 2])


def compute_overlap(q, z_center):
    rotated = _rotate_vertices(q)
    world_z = rotated[:, 2] + z_center
    return max(0.0, -np.min(world_z))
