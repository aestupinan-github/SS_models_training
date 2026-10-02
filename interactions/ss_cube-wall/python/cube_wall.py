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


# Range of the signed distance covered by the surrogate (canonical units, L = 1).
SD_GAP_MAX = 0.1
SD_PENETRATION_MAX = 0.1
# Shape constants (spec 01 R01.13, rev 3), derived from the point set.
# Circumscribed-sphere radius about the centre.
R_CIRCUMSCRIBED = float(np.max(np.linalg.norm(CUBE_VERTICES, axis=1)))
# Minimum of the support height h(q) over all orientations: distance from the
# centre to the nearest hull face plane. Cube: face-down, exactly 0.5.
H_MIN = 0.5
# Above this canonical height the pre-filter declares "no contact".
Z_PREFILTER = R_CIRCUMSCRIBED + SD_GAP_MAX
# Largest sd of any query that reaches the model.
SD_COVER_MAX = Z_PREFILTER - H_MIN


def canonicalize_quaternion(q):
    """Normalize and canonicalize a quaternion (qw, qx, qy, qz), body to world.

    Resolves the q / -q duality: qw >= 0; if qw == 0, the first nonzero
    vector component is positive. Raises ValueError for zero norm or
    non-finite components.
    """
    q = np.array(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"quaternion must have 4 components, got shape {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("quaternion has non-finite components")
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("quaternion has zero norm")
    q = q / norm
    if q[0] < 0.0:
        q = -q
    elif q[0] == 0.0:
        for c in q[1:]:
            if c != 0.0:
                if c < 0.0:
                    q = -q
                break
    return (q + 0.0).tolist()  # "+ 0.0" turns -0.0 into 0.0


def _check_size(L_actual):
    if not np.isfinite(L_actual) or L_actual <= 0.0:
        raise ValueError(f"L_actual must be finite and positive, got {L_actual}")


def to_canonical(position_z, L_actual):
    """Actual height -> canonical height (L = 1)."""
    _check_size(L_actual)
    return position_z / L_actual


def from_canonical(sd_canonical, L_actual):
    """Canonical signed distance -> actual signed distance."""
    _check_size(L_actual)
    return sd_canonical * L_actual


def classify_sd(sd):
    """'far' if sd > 0.1, 'too_deep' if sd < -0.1, else 'in_range' (canonical units)."""
    if sd > SD_GAP_MAX:
        return "far"
    if sd < -SD_PENETRATION_MAX:
        return "too_deep"
    return "in_range"


def is_far(position_z_canonical):
    """Shape-general pre-filter: beyond the circumscribed radius plus the gap
    range, no orientation can be within range, so there is no contact."""
    return position_z_canonical > R_CIRCUMSCRIBED + SD_GAP_MAX


def classify_zone(sd):
    """Zone of a query by its exact sd (R01.13, rev 3)."""
    if sd < -SD_PENETRATION_MAX:
        return "too_deep"
    if sd <= SD_GAP_MAX:
        return "accuracy"
    if sd <= SD_COVER_MAX:
        return "gap_safety"
    return "beyond"


def support_height(points, q):
    """Shape-general support height h(q) = max_v v . u, u = -R(q)^T e_z.

    `points` is an (N, 3) body-frame point set (any shape, convex or not;
    only hull vertices matter). sd = position_z - h(q). R01.16.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"points must have shape (N, 3) with N > 0, got {pts.shape}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points have non-finite components")
    qc = canonicalize_quaternion(q)
    rot = R.from_quat([qc[1], qc[2], qc[3], qc[0]])
    u = -rot.inv().apply([0.0, 0.0, 1.0])
    return float(np.max(pts @ u))


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