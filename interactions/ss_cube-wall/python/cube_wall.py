"""Cube-wall reference solution.

Implements the exact analytical cube support function for signed-distance
computation, following the LIGGGHTS sign convention:
positive = gap (no contact), negative = penetration.
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
    """Height of the cube centre at which the lowest vertex touches the wall."""
    rotated = _rotate_vertices(q)
    return -np.min(rotated[:, 2])


def compute_signed_distance(q, z_center):
    """Signed distance from the lowest vertex to the wall plane at z = 0.

    Positive = gap (no contact), negative = penetration.
    This is the LIGGGHTS `deltan` convention.
    """
    rotated = _rotate_vertices(q)
    world_z = rotated[:, 2] + z_center
    return float(np.min(world_z))