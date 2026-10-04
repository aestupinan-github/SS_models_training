"""Dataset generator for the ss_cube-wall interaction (spec 02 rev 2).

Implemented so far:
- T02.1, configuration and validation (R02.11, R02.13).
- T02.2, shape description from a point set (R02.1, R02.3).
- T02.3, orientation sampler: uniform and special, with spin, yaw and
  canonicalization (R02.3).
- T02.4, orientation sampler: near_kink by bisection on a tilt angle, and
  exclusion by body-frame direction u (R02.3, D02.3).
- T02.5, sd sampler and row assembly with recomputed labels (R02.1, R02.2,
  R02.4, R02.6; D02.4).
- T02.6, coverage table and acceptance checks (R02.5, R02.12).
- T02.7, writer, versioning, provenance, seeding (R02.7-R02.10; D02.6).
Later task T02.8 adds the CLI and summary.

All quantities are dimensionless canonical units (L = 1), see spec 01.
"""

import dataclasses
import datetime
import hashlib
import json
import math
import numbers
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

# Largest dataset that may be generated without explicit approval (R02.13).
SMOKE_MAX_ROWS = 10_000

# Approved default fractions (spec 02 R02.2, R02.3; user approval 2026-10-03).
DEFAULT_SD_FRACTIONS = {
    "penetration": 0.40,
    "gap": 0.30,
    "near_contact": 0.05,
    "exact_contact": 0.01,
    "gap_safety": 0.24,
}
DEFAULT_ORIENT_FRACTIONS = {
    "uniform": 0.70,
    "near_kink": 0.29,
    "special": 0.01,
}

# Upper bound of the accuracy zone (spec 01 R01.11): |sd| <= 0.1.
SD_ACCURACY_MAX = 0.1
# Tolerance for "fractions sum to 1".
FRACTION_SUM_TOL = 1e-9


class ConfigError(ValueError):
    """Invalid generator configuration (R02.11). `param` names the parameter."""

    def __init__(self, param, message):
        super().__init__(f"invalid configuration parameter '{param}': {message}")
        self.param = param


@dataclass(frozen=True)
class GenConfig:
    """Generator configuration (spec 02 design, Interfaces).

    `seed` has no default on purpose: a missing seed is a configuration
    error (R02.11), so it defaults to None and is rejected by validation.
    """

    n_orient: int
    k_per_orient: int
    seed: int = None
    sd_fractions: dict = field(default_factory=lambda: dict(DEFAULT_SD_FRACTIONS))
    orient_fractions: dict = field(default_factory=lambda: dict(DEFAULT_ORIENT_FRACTIONS))
    sd_log_min: float = 1e-5
    delta_range: tuple = (1e-6, 1e-1)
    exclude_quats: object = None
    exclude_angle_deg: float = 0.0
    m_min: int = 1
    allow_full_scale: bool = False

    @property
    def n_rows(self):
        return self.n_orient * self.k_per_orient


def _is_int(x):
    return isinstance(x, numbers.Integral) and not isinstance(x, (bool, np.bool_))


def _is_real(x):
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def _check_positive_int(name, value):
    if not _is_int(value):
        raise ConfigError(name, f"must be an integer, got {value!r}")
    if value <= 0:
        raise ConfigError(name, f"must be positive, got {value}")


def _check_fractions(name, fractions, expected_keys):
    if not isinstance(fractions, dict):
        raise ConfigError(name, f"must be a dict, got {type(fractions).__name__}")
    keys = set(fractions)
    if keys != set(expected_keys):
        missing = sorted(set(expected_keys) - keys)
        unknown = sorted(keys - set(expected_keys))
        raise ConfigError(name, f"keys must be {sorted(expected_keys)}; missing {missing}, unknown {unknown}")
    for key, value in fractions.items():
        if not _is_real(value) or not math.isfinite(value):
            raise ConfigError(name, f"fraction '{key}' must be a finite number, got {value!r}")
        if value < 0.0:
            raise ConfigError(name, f"fraction '{key}' must not be negative, got {value}")
    total = math.fsum(fractions.values())
    if abs(total - 1.0) > FRACTION_SUM_TOL:
        raise ConfigError(name, f"fractions must sum to 1, got {total!r}")


def _check_exclusion(cfg):
    angle = cfg.exclude_angle_deg
    if not _is_real(angle) or not math.isfinite(angle):
        raise ConfigError("exclude_angle_deg", f"must be a finite number, got {angle!r}")
    if not 0.0 <= angle <= 180.0:
        raise ConfigError("exclude_angle_deg", f"must be in [0, 180] degrees, got {angle}")
    if cfg.exclude_quats is None:
        return
    try:
        q = np.asarray(cfg.exclude_quats, dtype=np.float64)
    except (TypeError, ValueError):
        raise ConfigError("exclude_quats", "must be an (M, 4) array of quaternions") from None
    if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] == 0:
        raise ConfigError("exclude_quats", f"must have shape (M, 4) with M > 0, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ConfigError("exclude_quats", "has non-finite components")
    if np.any(np.linalg.norm(q, axis=1) == 0.0):
        raise ConfigError("exclude_quats", "has a zero-norm quaternion")
    if angle == 0.0:
        raise ConfigError("exclude_angle_deg", "must be positive when exclude_quats is given "
                                               "(an angle of 0 would exclude nothing)")


def validate_config(cfg):
    """Validate a GenConfig. Raises ConfigError naming the parameter (R02.11).

    Pure check: never writes or creates anything. Also enforces the smoke
    size limit unless `allow_full_scale` is set (R02.13).
    """
    if not isinstance(cfg, GenConfig):
        raise ConfigError("cfg", f"must be a GenConfig, got {type(cfg).__name__}")

    _check_positive_int("n_orient", cfg.n_orient)
    _check_positive_int("k_per_orient", cfg.k_per_orient)
    _check_positive_int("m_min", cfg.m_min)

    if cfg.seed is None:
        raise ConfigError("seed", "is missing; a fixed, recorded seed is required (R02.9)")
    if not _is_int(cfg.seed) or cfg.seed < 0:
        raise ConfigError("seed", f"must be a non-negative integer, got {cfg.seed!r}")

    _check_fractions("sd_fractions", cfg.sd_fractions, DEFAULT_SD_FRACTIONS)
    _check_fractions("orient_fractions", cfg.orient_fractions, DEFAULT_ORIENT_FRACTIONS)

    lo = cfg.sd_log_min
    if not _is_real(lo) or not math.isfinite(lo) or not 0.0 < lo < SD_ACCURACY_MAX:
        raise ConfigError("sd_log_min", f"must satisfy 0 < sd_log_min < {SD_ACCURACY_MAX}, got {lo!r}")

    dr = cfg.delta_range
    if not isinstance(dr, (tuple, list)) or len(dr) != 2:
        raise ConfigError("delta_range", f"must be a pair (low, high), got {dr!r}")
    d_lo, d_hi = dr
    if not all(_is_real(v) and math.isfinite(v) for v in dr):
        raise ConfigError("delta_range", f"bounds must be finite numbers, got {dr!r}")
    if not 0.0 < d_lo < d_hi:
        raise ConfigError("delta_range", f"must satisfy 0 < low < high, got {dr!r}")

    _check_exclusion(cfg)

    if not isinstance(cfg.allow_full_scale, (bool, np.bool_)):
        raise ConfigError("allow_full_scale", f"must be a bool, got {cfg.allow_full_scale!r}")
    if cfg.n_rows > SMOKE_MAX_ROWS and not cfg.allow_full_scale:
        raise ConfigError("allow_full_scale",
                          f"{cfg.n_rows} rows exceed the smoke limit of {SMOKE_MAX_ROWS}; "
                          "full-scale generation needs user approval and allow_full_scale=True (R02.13)")


# ---------------------------------------------------------------------------
# T02.2 Shape description from a point set (shape-general, spec 01 R01.16)
# ---------------------------------------------------------------------------

# Two hull simplices belong to the same facet if their plane equations
# (unit normal and offset) agree within this tolerance (design: 1e-12).
FACET_MERGE_TOL = 1e-12


class ShapeError(ValueError):
    """Invalid point set for a shape description."""


def _read_only(a):
    a = np.array(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class Shape:
    """Convex-hull description of a body-frame point set (canonical units).

    vertices        (V, 3) hull vertices, in input order
    facet_normals   (F, 3) unit outward normals of the merged (coplanar) facets
    facet_offsets   (F,)   distance from the origin (body centre) to each facet plane
    facet_vertices  tuple of F frozensets of indices into `vertices`
    edge_facets     tuple of E pairs (i, j), i < j, of facets sharing an edge
    edge_vertices   tuple of E pairs (a, b), a < b, the two vertices of each edge
    r_circ          circumscribed-sphere radius about the origin, max |v|
    h_min           min over facets of facet_offsets = min over orientations of h(q)
    """

    vertices: np.ndarray
    facet_normals: np.ndarray
    facet_offsets: np.ndarray
    facet_vertices: tuple
    edge_facets: tuple
    edge_vertices: tuple
    r_circ: float
    h_min: float


def shape_from_points(points):
    """Build the convex-hull Shape of a body-frame point set (R02.1, R02.3).

    Works for any point set, convex or not: only hull vertices matter for a
    wall interaction (spec 01 R01.16). The origin (body centre) must lie
    strictly inside the hull, since heights and `h_min` are measured from it.
    Raises ShapeError for invalid input.
    """
    try:
        pts = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError):
        raise ShapeError("points must be an (N, 3) numeric array") from None
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ShapeError(f"points must have shape (N, 3), got {pts.shape}")
    if pts.shape[0] < 4:
        raise ShapeError(f"at least 4 points are needed for a 3-D hull, got {pts.shape[0]}")
    if not np.all(np.isfinite(pts)):
        raise ShapeError("points have non-finite components")
    try:
        hull = ConvexHull(pts)
    except QhullError as e:
        raise ShapeError(f"points do not span a 3-D volume (degenerate hull): {e}".splitlines()[0]) from None

    # Qhull equations: n . x + c <= 0 inside, n unit outward, so offset = -c.
    normals_s = hull.equations[:, :3]
    offsets_s = -hull.equations[:, 3]
    if offsets_s.min() <= FACET_MERGE_TOL:
        raise ShapeError("the origin (body centre) must lie strictly inside the hull; "
                         f"smallest facet distance is {offsets_s.min():.3e}")

    # Keep hull vertices only, re-indexed in input order.
    hv = np.sort(hull.vertices)
    new_index = {int(old): i for i, old in enumerate(hv)}

    # Merge coplanar simplices into facets.
    facet_normals, facet_offsets, facet_sets = [], [], []
    for n, c, simplex in zip(normals_s, offsets_s, hull.simplices):
        idx = {new_index[int(v)] for v in simplex}
        for f, (fn, fc) in enumerate(zip(facet_normals, facet_offsets)):
            if np.abs(fn - n).max() <= FACET_MERGE_TOL and abs(fc - c) <= FACET_MERGE_TOL:
                facet_sets[f] |= idx
                break
        else:
            facet_normals.append(n.copy())
            facet_offsets.append(c)
            facet_sets.append(set(idx))

    # Coplanar hull vertices that Qhull did not list in any simplex of a facet
    # (possible for facets with more than 3 vertices): add them by distance.
    verts = pts[hv]
    for f, (fn, fc) in enumerate(zip(facet_normals, facet_offsets)):
        on_plane = np.where(np.abs(verts @ fn - fc) <= 1e-9 * max(1.0, fc))[0]
        facet_sets[f] |= {int(i) for i in on_plane}

    # Edges: pairs of facets sharing exactly two hull vertices.
    edge_facets, edge_vertices = [], []
    for i in range(len(facet_sets)):
        for j in range(i + 1, len(facet_sets)):
            shared = facet_sets[i] & facet_sets[j]
            if len(shared) == 2:
                a, b = sorted(shared)
                edge_facets.append((i, j))
                edge_vertices.append((a, b))

    facet_offsets = np.array(facet_offsets)
    return Shape(
        vertices=_read_only(verts),
        facet_normals=_read_only(np.array(facet_normals)),
        facet_offsets=_read_only(facet_offsets),
        facet_vertices=tuple(frozenset(s) for s in facet_sets),
        edge_facets=tuple(edge_facets),
        edge_vertices=tuple(edge_vertices),
        r_circ=float(np.max(np.linalg.norm(verts, axis=1))),
        h_min=float(facet_offsets.min()),
    )


# ---------------------------------------------------------------------------
# T02.3 Orientation sampler: uniform and special (R02.3)
# ---------------------------------------------------------------------------

# Integer codes of the `component` column (design: Data and units).
COMPONENT_CODES = {"uniform": 0, "near_kink": 1, "special": 2}


def _check_count(n):
    if not _is_int(n) or n <= 0:
        raise ValueError(f"count must be a positive integer, got {n!r}")


def canonicalize_quaternions(q):
    """Vectorized spec 01 R01.4 canonicalization of (n, 4) quaternions (w, x, y, z).

    Normalize; if qw < 0 negate; if qw == 0 negate so that the first nonzero
    of (qx, qy, qz) is positive. Bitwise equal to cube_wall.canonicalize_quaternion.
    Raises ValueError for zero-norm or non-finite rows.
    """
    q = np.array(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quaternions must have shape (n, 4), got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError("quaternions have non-finite components")
    norm = np.linalg.norm(q, axis=1)
    if np.any(norm == 0.0):
        raise ValueError("quaternion has zero norm")
    q = q / norm[:, None]
    flip = q[:, 0] < 0.0
    zero_w = q[:, 0] == 0.0
    if np.any(zero_w):
        v = q[zero_w, 1:]
        first = np.argmax(v != 0.0, axis=1)        # index of first nonzero component
        flip[zero_w] = v[np.arange(len(v)), first] < 0.0
    q[flip] = -q[flip]
    return q + 0.0                                   # -0.0 -> 0.0, as in cube_wall


def _to_scipy(q):
    return q[:, [1, 2, 3, 0]]


def _from_scipy(rot):
    return rot.as_quat()[:, [3, 0, 1, 2]]


def u_from_quats(q):
    """Body-frame direction towards the wall, u = -R(q)^T e_z (spec 01 R01.16)."""
    rot = Rotation.from_quat(_to_scipy(np.asarray(q, dtype=np.float64)))
    return -rot.inv().apply([0.0, 0.0, 1.0])


def support_gap(shape, u):
    """Support gap Delta(u): largest minus second-largest of v . u over hull vertices.

    Equals the second-smallest minus the smallest world height (spec 02
    Definitions). Delta = 0 where the lowest feature is an edge or a face.
    """
    p = np.sort(np.asarray(u, dtype=np.float64) @ shape.vertices.T, axis=1)
    return p[:, -1] - p[:, -2]


def rotation_from_u(u):
    """Minimal rotation per row mapping body direction u to world -e_z."""
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u, axis=1, keepdims=True)
    target = np.array([0.0, 0.0, -1.0])
    axis = np.cross(u, target)
    s = np.linalg.norm(axis, axis=1)
    c = u @ target
    angle = np.arctan2(s, c)
    rotvec = np.zeros_like(u)
    ok = s > 1e-300
    rotvec[ok] = axis[ok] / s[ok, None] * angle[ok, None]
    # antiparallel (u = +e_z): rotate by pi about body x
    anti = (~ok) & (c < 0.0)
    rotvec[anti] = [np.pi, 0.0, 0.0]
    return Rotation.from_rotvec(rotvec)


def _spin_and_yaw(rot0, u, rng):
    """rot = Rz(psi) * rot0 * R_u(phi): random spin about body u, random yaw about world z."""
    n = len(u)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    spin = Rotation.from_rotvec(u * phi[:, None])
    yaw = Rotation.from_rotvec(np.outer(psi, [0.0, 0.0, 1.0]))
    return yaw * rot0 * spin


def _uniform_batch(n, rng):
    rot = Rotation.random(n, random_state=rng)
    psi = rng.uniform(0.0, 2.0 * np.pi, n)
    rot = Rotation.from_rotvec(np.outer(psi, [0.0, 0.0, 1.0])) * rot
    return canonicalize_quaternions(_from_scipy(rot))


def sample_uniform(n, rng, exclude_quats=None, exclude_angle_deg=0.0):
    """`uniform` component: n canonical quaternions, Haar measure on SO(3), then yaw.

    The yaw does not change the distribution (Haar is invariant) but is applied
    for uniformity with the other components (design). Excluded directions
    (D02.3) are redrawn.
    """
    _check_count(n)
    return _draw_with_exclusion(lambda m: (_uniform_batch(m, rng),), n, exclude_quats, exclude_angle_deg)[0]


def special_directions(shape):
    """Shape-defined special body directions u (R02.3): face, edge, vertex down.

    face:   unit outward facet normals
    edge:   normalized sum of the two facet normals of each edge
    vertex: normalized hull vertex directions
    Returns (dirs (m, 3), kind (m,) array of 'face'/'edge'/'vertex').
    """
    faces = np.array(shape.facet_normals)
    edges = np.array([faces[i] + faces[j] for i, j in shape.edge_facets])
    edges /= np.linalg.norm(edges, axis=1, keepdims=True)
    verts = np.array(shape.vertices) / np.linalg.norm(shape.vertices, axis=1, keepdims=True)
    dirs = np.vstack([faces, edges, verts])
    kind = np.array(["face"] * len(faces) + ["edge"] * len(edges) + ["vertex"] * len(verts))
    return dirs, kind


def sample_special(shape, n, rng, exclude_quats=None, exclude_angle_deg=0.0):
    """`special` component: n orientations whose u is a special direction.

    Directions are drawn uniformly from the finite set of directions that are
    not excluded (D02.3), then random spin about u and yaw about world z are
    applied (design). Returns (quats (n, 4) canonical, which (n,) index into
    special_directions(shape)). Raises ValueError if every special direction
    is excluded.
    """
    _check_count(n)
    dirs, _ = special_directions(shape)
    allowed = np.arange(len(dirs))
    if exclude_quats is not None:
        allowed = allowed[~_u_excluded(dirs, exclude_quats, exclude_angle_deg)]
        if len(allowed) == 0:
            raise ValueError("every special direction is excluded; cannot sample the special component")
    which = allowed[rng.integers(0, len(allowed), n)]
    u = dirs[which]
    rot = _spin_and_yaw(rotation_from_u(u), u, rng)
    return canonicalize_quaternions(_from_scipy(rot)), which


# ---------------------------------------------------------------------------
# T02.4 near_kink sampler and exclusion by u (R02.3, D02.3)
# ---------------------------------------------------------------------------

# Tilt-angle bracket and bisection steps for near_kink (design).
NEAR_KINK_THETA_MAX = 0.5
NEAR_KINK_BISECTION_STEPS = 60
# Safety bound on redraw rounds (exclusion, unreachable targets).
MAX_REDRAW_ROUNDS = 1000


def _u_excluded(u, exclude_quats, exclude_angle_deg):
    """True where body direction u is within exclude_angle_deg of an excluded u (D02.3)."""
    u = np.asarray(u, dtype=np.float64)
    if exclude_quats is None:
        return np.zeros(len(u), dtype=bool)
    u_ex = u_from_quats(canonicalize_quaternions(exclude_quats))
    ang = np.degrees(np.arccos(np.clip(u @ u_ex.T, -1.0, 1.0)))
    return np.any(ang < exclude_angle_deg, axis=1)


def exclusion_mask(q, exclude_quats, exclude_angle_deg):
    """True for each quaternion whose u is within the exclusion angle of an excluded u (D02.3).

    Exclusion is by body-frame direction u, not by full orientation: sd depends
    only on u, and spin and yaw would otherwise leave the same sd in training.
    """
    q = np.asarray(q, dtype=np.float64)
    if exclude_quats is None:
        return np.zeros(len(q), dtype=bool)
    return _u_excluded(u_from_quats(q), exclude_quats, exclude_angle_deg)


def _draw_with_exclusion(draw, n, exclude_quats, exclude_angle_deg):
    """Call draw(m) -> tuple of arrays (first is quats) until n non-excluded rows are kept."""
    parts = None
    have = 0
    need = n
    for _ in range(MAX_REDRAW_ROUNDS):
        out = draw(need)
        keep = ~exclusion_mask(out[0], exclude_quats, exclude_angle_deg)
        out = tuple(a[keep] for a in out)
        parts = out if parts is None else tuple(np.concatenate([p, a]) for p, a in zip(parts, out))
        have = len(parts[0])
        if have >= n:
            return tuple(a[:n] for a in parts)
        need = n - have
    raise ValueError(f"could not draw {n} non-excluded orientations in {MAX_REDRAW_ROUNDS} rounds "
                     f"(kept {have}); exclusion too large?")


def _near_kink_starts(shape, n, rng):
    """Start directions u0 on the kink set and unit tilt directions w (design).

    Half the rows (alternating) use face mode: u0 = a facet normal, w random
    and perpendicular to u0. The others use edge mode: u0 uniform in arc angle
    on the great arc between the two facet normals of an edge, w = +/- (n_i x
    n_j) normalized (perpendicular to the arc).
    """
    normals = np.asarray(shape.facet_normals)
    u0 = np.empty((n, 3))
    w = np.empty((n, 3))
    face = (np.arange(n) % 2) == 0
    nf = int(face.sum())
    ne = n - nf

    f = rng.integers(0, len(normals), nf)
    u0f = normals[f]
    wf = rng.normal(size=(nf, 3))
    wf -= np.sum(wf * u0f, axis=1, keepdims=True) * u0f
    u0[face] = u0f
    w[face] = wf

    if ne:
        e = rng.integers(0, len(shape.edge_facets), ne)
        pairs = np.asarray(shape.edge_facets)[e]
        n1, n2 = normals[pairs[:, 0]], normals[pairs[:, 1]]
        a = np.arccos(np.clip(np.sum(n1 * n2, axis=1), -1.0, 1.0))
        x = rng.uniform(0.0, 1.0, ne) * a
        u0e = (np.sin(a - x)[:, None] * n1 + np.sin(x)[:, None] * n2) / np.sin(a)[:, None]
        we = np.cross(n1, n2) * rng.choice([-1.0, 1.0], ne)[:, None]
        u0[~face] = u0e
        w[~face] = we

    u0 /= np.linalg.norm(u0, axis=1, keepdims=True)
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return u0, w


def _near_kink_batch(shape, n, rng, delta_range):
    """One batch: targets, bisection on theta; drops rows whose bracket misses the target."""
    lo_d, hi_d = delta_range
    target = 10.0 ** rng.uniform(np.log10(lo_d), np.log10(hi_d), n)
    u0, w = _near_kink_starts(shape, n, rng)

    def u_at(theta):
        return np.cos(theta)[:, None] * u0 + np.sin(theta)[:, None] * w

    lo = np.zeros(n)
    hi = np.full(n, NEAR_KINK_THETA_MAX)
    reach = support_gap(shape, u_at(hi)) >= target
    for _ in range(NEAR_KINK_BISECTION_STEPS):
        mid = 0.5 * (lo + hi)
        below = support_gap(shape, u_at(mid)) < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)

    u = u_at(hi)[reach]
    target = target[reach]
    rot = _spin_and_yaw(rotation_from_u(u), u, rng)
    return canonicalize_quaternions(_from_scipy(rot)), target


def sample_near_kink(shape, n, rng, delta_range=(1e-6, 1e-1), exclude_quats=None, exclude_angle_deg=0.0):
    """`near_kink` component: n orientations with support gap log-uniform in delta_range.

    Starts on the non-smooth set of h (facet normals and edge arcs, shape
    defined), tilts by theta in [0, 0.5] and bisects theta (60 steps) so that
    Delta(u) equals the target. Rows whose bracket cannot reach the target,
    and excluded rows (D02.3), are redrawn. Returns (quats (n, 4) canonical,
    target Delta (n,)).
    """
    _check_count(n)
    lo_d, hi_d = delta_range
    if not (math.isfinite(lo_d) and math.isfinite(hi_d) and 0.0 < lo_d < hi_d):
        raise ValueError(f"delta_range must satisfy 0 < low < high, got {delta_range!r}")
    return _draw_with_exclusion(lambda m: _near_kink_batch(shape, m, rng, delta_range),
                                n, exclude_quats, exclude_angle_deg)


# ---------------------------------------------------------------------------
# T02.5 sd sampler and row assembly (R02.1, R02.2, R02.4, R02.6; D02.4)
# ---------------------------------------------------------------------------

# Integer codes of the `sd_part` column (design: Data and units).
SD_PART_CODES = {"penetration": 0, "gap": 1, "near_contact": 2, "exact_contact": 3, "gap_safety": 4}

# Stored columns, in order: R02.6 columns, then D02.1 extra columns.
ROW_COLUMNS = ("position_z", "qw", "qx", "qy", "qz", "sd", "h",
               "orientation_id", "component", "sd_part", "delta")


class GenerationError(RuntimeError):
    """Failure while generating data (R02.11: non-finite values, reference errors)."""


def allocate_counts(fractions, k):
    """Stratified counts per part for k rows (design; D02.4).

    Round half up (not Python's banker's rounding), then give any difference
    from k to the part with the largest fraction. Sum is always k.
    """
    names = list(fractions)
    counts = {n: int(math.floor(fractions[n] * k + 0.5)) for n in names}
    largest = max(names, key=lambda n: fractions[n])
    counts[largest] += k - sum(counts.values())
    if counts[largest] < 0:                     # cannot happen for valid fractions; guard anyway
        raise ValueError(f"allocation failed for k={k}: {counts}")
    return counts


def sample_orientations(cfg, shape, rng):
    """Mixture of the orientation components by fractions (R02.3).

    Counts per component use allocate_counts. Rows are ordered uniform, then
    near_kink, then special (deterministic, no shuffling: R02.7). Returns
    (quats (n_orient, 4) canonical, component (n_orient,) int8 codes).
    """
    validate_config(cfg)
    counts = allocate_counts(cfg.orient_fractions, cfg.n_orient)
    ex = dict(exclude_quats=cfg.exclude_quats, exclude_angle_deg=cfg.exclude_angle_deg)
    quats, comps = [], []
    for name in ("uniform", "near_kink", "special"):
        n = counts[name]
        if n == 0:
            continue
        if name == "uniform":
            q = sample_uniform(n, rng, **ex)
        elif name == "near_kink":
            q, _ = sample_near_kink(shape, n, rng, delta_range=tuple(cfg.delta_range), **ex)
        else:
            q, _ = sample_special(shape, n, rng, **ex)
        quats.append(q)
        comps.append(np.full(n, COMPONENT_CODES[name], dtype=np.int8))
    return np.concatenate(quats), np.concatenate(comps)


def _part_assignment(cfg, n_orient, rng):
    """sd part code for each of the n_orient * k rows (D02.4)."""
    k = cfg.k_per_orient
    names = list(SD_PART_CODES)
    counts = allocate_counts(cfg.sd_fractions, k)
    starved = any(cfg.sd_fractions[n] > 0.0 and counts[n] == 0 for n in names)
    if not starved:
        block = np.concatenate([np.full(counts[n], SD_PART_CODES[n], dtype=np.int8) for n in names])
        return np.tile(block, n_orient)
    p = np.array([cfg.sd_fractions[n] for n in names])
    return rng.choice(np.array([SD_PART_CODES[n] for n in names], dtype=np.int8),
                      size=n_orient * k, p=p / p.sum())


def sample_sd(cfg, shape, h, rng):
    """Draw sd for every row (R02.2, R02.4). h: (n_orient,) support heights.

    Rows are grouped per orientation (k consecutive rows). Parts:
      penetration   -10^U(log10 sd_log_min, log10 0.1)
      gap           +10^U(log10 sd_log_min, log10 0.1)
      near_contact  U(-sd_log_min, sd_log_min)
      exact_contact 0
      gap_safety    U(0.1, Z_PREFILTER - h); if that interval is empty
                    (Z_PREFILTER - h <= 0.1, e.g. corner-down) the row is drawn as gap.
    Returns (sd (n_rows,), part (n_rows,) int8 codes).
    """
    validate_config(cfg)
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (cfg.n_orient,):
        raise ValueError(f"h must have shape ({cfg.n_orient},), got {h.shape}")
    k = cfg.k_per_orient
    part = _part_assignment(cfg, cfg.n_orient, rng)
    hr = np.repeat(h, k)
    z_pre = shape.r_circ + SD_ACCURACY_MAX
    hi_safety = z_pre - hr
    P = SD_PART_CODES
    part = np.where((part == P["gap_safety"]) & (hi_safety <= SD_ACCURACY_MAX), np.int8(P["gap"]), part)

    lo_log, hi_log = math.log10(cfg.sd_log_min), math.log10(SD_ACCURACY_MAX)
    n = len(part)
    # draw every stream for all rows so the result does not depend on part order
    log_mag = 10.0 ** rng.uniform(lo_log, hi_log, n)
    near = rng.uniform(-cfg.sd_log_min, cfg.sd_log_min, n)
    safety = SD_ACCURACY_MAX + rng.uniform(0.0, 1.0, n) * (hi_safety - SD_ACCURACY_MAX)

    sd = np.zeros(n)
    sd = np.where(part == P["penetration"], -log_mag, sd)
    sd = np.where(part == P["gap"], log_mag, sd)
    sd = np.where(part == P["near_contact"], near, sd)
    sd = np.where(part == P["gap_safety"], safety, sd)
    # gap safety: keep strictly above 0.1 (uniform draw of exactly 0 is possible in principle)
    gs = part == P["gap_safety"]
    sd[gs] = np.maximum(sd[gs], np.nextafter(SD_ACCURACY_MAX, np.inf))
    return sd, part


def build_rows(cfg, shape, quats, component, sd, sd_part):
    """Assemble the stored columns (R02.1, R02.4, R02.6).

    position_z = h + sd (target); then h is recomputed from the stored
    quaternion and the stored label is sd = position_z - h, so every row is
    self-consistent with the spec 01 reference (R02.1). Raises
    GenerationError for non-finite values (R02.11).
    """
    k = cfg.k_per_orient
    quats = np.asarray(quats, dtype=np.float64)
    if quats.shape != (cfg.n_orient, 4) or len(component) != cfg.n_orient:
        raise ValueError(f"expected {cfg.n_orient} orientations, got quats {quats.shape}, "
                         f"component {np.shape(component)}")
    if np.shape(sd) != (cfg.n_rows,) or np.shape(sd_part) != (cfg.n_rows,):
        raise ValueError(f"expected {cfg.n_rows} rows, got sd {np.shape(sd)}, sd_part {np.shape(sd_part)}")
    sd = np.asarray(sd, dtype=np.float64)
    if not (np.all(np.isfinite(quats)) and np.all(np.isfinite(sd))):
        raise GenerationError("non-finite value in orientations or sd targets")

    u = u_from_quats(quats)
    h_orient = (u @ shape.vertices.T).max(axis=1)
    delta_orient = support_gap(shape, u)

    q_rows = np.repeat(quats, k, axis=0)
    h = np.repeat(h_orient, k)
    position_z = h + sd
    # exact contact rows: position_z = h exactly, so sd recomputes to exactly 0
    label = position_z - h

    rows = {
        "position_z": position_z,
        "qw": q_rows[:, 0].copy(),
        "qx": q_rows[:, 1].copy(),
        "qy": q_rows[:, 2].copy(),
        "qz": q_rows[:, 3].copy(),
        "sd": label,
        "h": h,
        "orientation_id": np.repeat(np.arange(cfg.n_orient, dtype=np.int64), k),
        "component": np.repeat(np.asarray(component, dtype=np.int8), k),
        "sd_part": np.asarray(sd_part, dtype=np.int8).copy(),
        "delta": np.repeat(delta_orient, k),
    }
    for name in ("position_z", "sd", "h", "delta"):
        if not np.all(np.isfinite(rows[name])):
            raise GenerationError(f"non-finite value in column '{name}'")
    return {name: rows[name] for name in ROW_COLUMNS}


# ---------------------------------------------------------------------------
# T02.6 Coverage table and acceptance checks (R02.5, R02.12)
# ---------------------------------------------------------------------------

# Decade ranges of the coverage table, fixed by R02.5.
COVERAGE_SD_RANGE = (1e-5, 1e-1)
COVERAGE_DELTA_RANGE = (1e-6, 1e-1)

# Tolerances of R02.12.
LABEL_TOL = 1e-12
UNIT_NORM_TOL = 1e-12

# Order of the acceptance checks (R02.12).
ACCEPTANCE_CHECKS = ("label", "h_consistent", "ranges", "prefilter", "finite",
                     "unit_norm", "canonical", "coverage", "exclusion")


def _check_rows(rows):
    missing = [c for c in ROW_COLUMNS if c not in rows]
    if missing:
        raise ValueError(f"rows are missing columns {missing}")
    n = len(rows[ROW_COLUMNS[0]])
    bad = [c for c in ROW_COLUMNS if np.shape(rows[c]) != (n,)]
    if bad:
        raise ValueError(f"columns {bad} do not have shape ({n},)")
    return n


def _decades(lo, hi):
    """Decade cells [10^k, 10^(k+1)) covering [lo, hi]; lo and hi are powers of 10."""
    k0, k1 = int(round(math.log10(lo))), int(round(math.log10(hi)))
    return [(10.0 ** k, 10.0 ** (k + 1)) for k in range(k0, k1)]


def _decade_counts(values, lo, hi):
    """Counts per decade; lower bound inclusive, upper exclusive, last decade includes hi."""
    v = np.asarray(values, dtype=np.float64)
    cells = []
    edges = _decades(lo, hi)
    for i, (a, b) in enumerate(edges):
        top = (v <= b) if i == len(edges) - 1 else (v < b)
        cells.append({"lo": a, "hi": b, "count": int(np.count_nonzero((v >= a) & top))})
    return cells


def coverage_table(rows):
    """Coverage table of R02.5 (JSON-serializable dict).

    sd_negative / sd_positive: rows per decade of |sd| over [1e-5, 1e-1] for
    each sign; delta: rows per decade of the support gap over [1e-6, 1e-1];
    components / sd_parts: rows per component and per sd part.
    """
    n = _check_rows(rows)
    sd = np.asarray(rows["sd"], dtype=np.float64)
    comp = np.asarray(rows["component"])
    part = np.asarray(rows["sd_part"])
    return {
        "n_rows": int(n),
        "sd_negative": _decade_counts(np.where(sd < 0.0, -sd, np.nan), *COVERAGE_SD_RANGE),
        "sd_positive": _decade_counts(np.where(sd > 0.0, sd, np.nan), *COVERAGE_SD_RANGE),
        "delta": _decade_counts(rows["delta"], *COVERAGE_DELTA_RANGE),
        "components": {name: int(np.count_nonzero(comp == code)) for name, code in COMPONENT_CODES.items()},
        "sd_parts": {name: int(np.count_nonzero(part == code)) for name, code in SD_PART_CODES.items()},
    }


def _result(n_fail, worst, tol):
    worst = float(worst)
    return {"passed": bool(n_fail == 0), "n_fail": int(n_fail),
            "worst": worst if math.isfinite(worst) else None, "tol": tol}


def _nan_max(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else 0.0


def acceptance_checks(rows, cfg, shape):
    """Acceptance checks of R02.12. Returns {name: {passed, n_fail, worst, tol}}.

    Pure function: does not modify `rows` and writes nothing. `h_ref` is
    recomputed independently from the stored quaternions, so a row whose h
    and sd are wrong together is still caught.
    """
    n = _check_rows(rows)
    P = SD_PART_CODES
    f64 = {c: np.asarray(rows[c], dtype=np.float64)
           for c in ("position_z", "qw", "qx", "qy", "qz", "sd", "h", "delta")}
    q = np.stack([f64["qw"], f64["qx"], f64["qy"], f64["qz"]], axis=1)
    sd, z, part = f64["sd"], f64["position_z"], np.asarray(rows["sd_part"])
    res = {}

    # finite (first: other checks skip non-finite rows rather than crash)
    finite_rows = np.all(np.isfinite(np.stack(list(f64.values()), axis=1)), axis=1)
    res["finite"] = _result(np.count_nonzero(~finite_rows), 0.0, 0.0)
    qn = np.linalg.norm(q, axis=1)
    usable = finite_rows & (qn > 0.0)

    # label: |sd - (position_z - h_ref)| with h_ref from the stored quaternion
    h_ref = np.full(n, np.nan)
    if np.any(usable):
        qu = q[usable] / qn[usable, None]
        h_ref[usable] = (u_from_quats(qu) @ shape.vertices.T).max(axis=1)
    err = np.abs(sd - (z - h_ref))
    res["label"] = _result(np.count_nonzero(~(err <= LABEL_TOL)), _nan_max(err), LABEL_TOL)
    err_h = np.abs(f64["h"] - h_ref)
    res["h_consistent"] = _result(np.count_nonzero(~(err_h <= LABEL_TOL)), _nan_max(err_h), LABEL_TOL)

    # ranges per sd part (R02.2)
    lo = cfg.sd_log_min
    z_pre = shape.r_circ + SD_ACCURACY_MAX
    ok = np.zeros(n, dtype=bool)
    m = part == P["penetration"]
    ok[m] = (sd[m] >= -SD_ACCURACY_MAX) & (sd[m] <= -lo)
    m = part == P["gap"]
    ok[m] = (sd[m] >= lo) & (sd[m] <= SD_ACCURACY_MAX)
    m = part == P["near_contact"]
    ok[m] = (sd[m] > -lo) & (sd[m] < lo)
    m = part == P["exact_contact"]
    ok[m] = sd[m] == 0.0
    m = part == P["gap_safety"]
    ok[m] = (sd[m] > SD_ACCURACY_MAX) & (z[m] <= z_pre)
    # global range: nothing deeper than -0.1 (R02.2)
    ok &= sd >= -SD_ACCURACY_MAX
    out = np.maximum(-SD_ACCURACY_MAX - sd, 0.0)
    res["ranges"] = _result(np.count_nonzero(~ok), _nan_max(out), 0.0)

    # pre-filter: no row above Z_PREFILTER (R02.2)
    above = z - z_pre
    res["prefilter"] = _result(np.count_nonzero(~(above <= 0.0)), max(_nan_max(above), 0.0), 0.0)

    # unit norm and canonical form (R01.4)
    nerr = np.abs(qn - 1.0)
    res["unit_norm"] = _result(np.count_nonzero(~(nerr <= UNIT_NORM_TOL)), _nan_max(nerr), UNIT_NORM_TOL)
    canon_bad = np.ones(n, dtype=bool)
    if np.any(usable):
        c = canonicalize_quaternions(q[usable])
        canon_bad[usable] = np.abs(c - q[usable] / qn[usable, None]).max(axis=1) > UNIT_NORM_TOL
    res["canonical"] = _result(np.count_nonzero(canon_bad), 0.0, UNIT_NORM_TOL)

    # coverage: every decade cell >= m_min (R02.5)
    table = coverage_table(rows)
    cells = table["sd_negative"] + table["sd_positive"] + table["delta"]
    short = [cell for cell in cells if cell["count"] < cfg.m_min]
    res["coverage"] = _result(len(short), min(cell["count"] for cell in cells), cfg.m_min)

    # exclusion by u (D02.3)
    if cfg.exclude_quats is None:
        res["exclusion"] = _result(0, 0.0, cfg.exclude_angle_deg)
    else:
        excl = np.zeros(n, dtype=bool)
        excl[usable] = exclusion_mask(q[usable] / qn[usable, None], cfg.exclude_quats, cfg.exclude_angle_deg)
        res["exclusion"] = _result(np.count_nonzero(excl), 0.0, cfg.exclude_angle_deg)

    return {name: res[name] for name in ACCEPTANCE_CHECKS}


def require_acceptance(results):
    """Raise GenerationError naming every failed check (R02.11: failures abort the write)."""
    failed = [f"{name} ({r['n_fail']} failing, worst {r['worst']})"
              for name, r in results.items() if not r["passed"]]
    if failed:
        raise GenerationError("acceptance checks failed: " + "; ".join(failed))


# ---------------------------------------------------------------------------
# T02.7 Writer, versioning, provenance, seeding (R02.7-R02.10; D02.6)
# ---------------------------------------------------------------------------

SPEC_REVISIONS = {"01": "rev 3", "02": "rev 2"}
GENERATOR_NAME = "generate_dataset.py"
# Random streams derived from the master seed (D02.6), in spawn order.
SEED_STREAMS = ("orientations", "sd")
DATASET_PREFIX = "dataset_"
_DATASET_RE = re.compile(r"^dataset_(\d{3,})$")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _cube_points():
    import cube_wall
    return cube_wall.CUBE_VERTICES


def generate_rows(cfg, shape):
    """Sample and assemble all rows for `cfg` (R02.9).

    One master seed; independent streams are spawned with numpy SeedSequence
    (SEED_STREAMS order). Returns (rows, seed_info) where seed_info records
    the master seed and each stream's entropy and spawn key, so every stream
    can be reconstructed.
    """
    validate_config(cfg)
    ss = np.random.SeedSequence(cfg.seed)
    children = ss.spawn(len(SEED_STREAMS))
    rngs = {name: np.random.default_rng(child) for name, child in zip(SEED_STREAMS, children)}
    quats, comp = sample_orientations(cfg, shape, rngs["orientations"])
    h = (u_from_quats(quats) @ shape.vertices.T).max(axis=1)
    sd, part = sample_sd(cfg, shape, h, rngs["sd"])
    rows = build_rows(cfg, shape, quats, comp, sd, part)
    seed_info = {
        "master_seed": int(cfg.seed),
        "streams": {name: {"entropy": int(child.entropy), "spawn_key": [int(k) for k in child.spawn_key]}
                    for name, child in zip(SEED_STREAMS, children)},
    }
    return rows, seed_info


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_state():
    """(commit, dirty) of the repository holding this script; (None, None) if unknown."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_HERE, capture_output=True,
                                text=True, timeout=10)
        if commit.returncode != 0:
            return None, None
        status = subprocess.run(["git", "status", "--porcelain"], cwd=_HERE, capture_output=True,
                                text=True, timeout=10)
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit.stdout.strip(), dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def _config_record(cfg):
    """All GenConfig fields, JSON-serializable; the exclusion list by count and sha256."""
    rec = {}
    for f in dataclasses.fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "exclude_quats":
            if v is None:
                rec[f.name] = None
            else:
                a = np.ascontiguousarray(np.asarray(v, dtype=np.float64))
                rec[f.name] = {"count": int(a.shape[0]), "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
        elif isinstance(v, dict):
            rec[f.name] = {k: float(x) for k, x in v.items()}
        elif isinstance(v, (tuple, list)):
            rec[f.name] = [float(x) for x in v]
        elif isinstance(v, (bool, np.bool_)):
            rec[f.name] = bool(v)
        elif _is_int(v):
            rec[f.name] = int(v)
        else:
            rec[f.name] = float(v)
    return rec


def _shape_record(shape, source):
    """Shape used for the labels: source name, hull vertices and constants."""
    return {
        "source": source,
        "points": np.asarray(shape.vertices).tolist(),
        "n_hull_vertices": int(len(shape.vertices)),
        "n_facets": int(len(shape.facet_normals)),
        "n_edges": int(len(shape.edge_facets)),
        "r_circ": float(shape.r_circ),
        "h_min": float(shape.h_min),
        "z_prefilter": float(shape.r_circ + SD_ACCURACY_MAX),
    }


def _next_index(root):
    """1 + largest existing dataset_XXX number under root (0 if none)."""
    if not os.path.isdir(root):
        return 1
    nums = [int(m.group(1)) for m in (_DATASET_RE.match(n) for n in os.listdir(root)) if m]
    return (max(nums) if nums else 0) + 1


def _write_csv(rows, path):
    """CSV with header, comma delimiter, floats with 17 significant digits (R02.7).

    `component` and `sd_part` are written as their names (design: Data and units).
    """
    inv_comp = {v: k for k, v in COMPONENT_CODES.items()}
    inv_part = {v: k for k, v in SD_PART_CODES.items()}
    n = len(rows[ROW_COLUMNS[0]])
    cols = []
    for name in ROW_COLUMNS:
        a = rows[name]
        if name == "component":
            cols.append([inv_comp[int(v)] for v in a])
        elif name == "sd_part":
            cols.append([inv_part[int(v)] for v in a])
        elif name == "orientation_id":
            cols.append([str(int(v)) for v in a])
        else:
            cols.append(["%.17g" % v for v in a])
    with open(path, "w", newline="") as f:
        f.write(",".join(ROW_COLUMNS) + "\n")
        for i in range(n):
            f.write(",".join(c[i] for c in cols) + "\n")


def _publish(tmp, final):
    """Rename tmp to final without ever replacing an existing path (D02.6).

    os.rename silently replaces an existing EMPTY directory on POSIX, so the
    target is first claimed with os.mkdir (fails if it exists), then the
    claim is removed and the rename done; a race in that gap makes rename
    fail on a non-empty target or, at worst, the second writer fails at mkdir.
    """
    try:
        os.mkdir(final)
    except FileExistsError:
        raise GenerationError(f"target {final} already exists; refusing to overwrite. "
                              f"Data left in {tmp}") from None
    os.rmdir(final)
    try:
        os.rename(tmp, final)
    except OSError as e:
        raise GenerationError(f"could not publish {tmp} as {final}: {e}") from None


def write_dataset(rows, cfg, shape, out_root, seed_info, csv=False, shape_source="cube_wall.CUBE_VERTICES"):
    """Check rows, write a new dataset folder, return its path (R02.7, R02.8, R02.10).

    Steps: acceptance checks (abort before writing on failure, R02.11); build
    in out_root/.tmp_dataset_XXX_<pid>; write data.npz (uncompressed,
    byte-reproducible), optional data.csv, then provenance.json last;
    publish atomically as dataset_XXX. Existing datasets are never
    overwritten. On a write failure the temp folder is left for inspection
    and no complete-looking dataset folder exists.
    """
    validate_config(cfg)
    if csv and cfg.n_rows > SMOKE_MAX_ROWS:
        raise ConfigError("csv", f"CSV export is limited to {SMOKE_MAX_ROWS} rows (R02.7), got {cfg.n_rows}")
    _check_rows(rows)
    acceptance = acceptance_checks(rows, cfg, shape)
    require_acceptance(acceptance)
    coverage = coverage_table(rows)

    os.makedirs(out_root, exist_ok=True)
    idx = _next_index(out_root)
    name = f"{DATASET_PREFIX}{idx:03d}"
    final = os.path.join(out_root, name)
    tmp = os.path.join(out_root, f".tmp_{name}_{os.getpid()}")
    try:
        os.mkdir(tmp)
    except FileExistsError:
        raise GenerationError(f"temporary folder {tmp} already exists; remove it and retry") from None

    try:
        npz_path = os.path.join(tmp, "data.npz")
        np.savez(npz_path, **{k: rows[k] for k in ROW_COLUMNS})
        sha = {"data.npz": _sha256(npz_path)}
        if csv:
            csv_path = os.path.join(tmp, "data.csv")
            _write_csv(rows, csv_path)
            sha["data.csv"] = _sha256(csv_path)
        commit, dirty = _git_state()
        import scipy
        provenance = {
            "spec": dict(SPEC_REVISIONS),
            "generator": GENERATOR_NAME,
            "git_commit": commit,
            "git_dirty": dirty,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__},
            "config": _config_record(cfg),
            "seed": seed_info,
            "shape": _shape_record(shape, shape_source),
            "n_rows": int(cfg.n_rows),
            "columns": list(ROW_COLUMNS),
            "codes": {"component": dict(COMPONENT_CODES), "sd_part": dict(SD_PART_CODES)},
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "sha256": sha,
            "coverage": coverage,
            "acceptance": acceptance,
        }
        with open(os.path.join(tmp, "provenance.json"), "w") as f:
            json.dump(provenance, f, indent=2, sort_keys=False)
            f.write("\n")
    except GenerationError:
        raise
    except Exception as e:
        raise GenerationError(f"writing dataset failed ({type(e).__name__}: {e}); "
                              f"partial data left in {tmp}") from None

    _publish(tmp, final)
    return final


def generate(cfg, out_root, csv=False, points=None):
    """Whole pipeline: validate, sample, assemble, check, write. Returns the dataset path.

    `points` is the body-frame point set (default: the canonical cube).
    Validation errors are raised before anything is created (R02.11).
    """
    validate_config(cfg)
    if csv and cfg.n_rows > SMOKE_MAX_ROWS:
        raise ConfigError("csv", f"CSV export is limited to {SMOKE_MAX_ROWS} rows (R02.7), got {cfg.n_rows}")
    source = "cube_wall.CUBE_VERTICES" if points is None else "user point set"
    shape = shape_from_points(_cube_points() if points is None else points)
    rows, seed_info = generate_rows(cfg, shape)
    return write_dataset(rows, cfg, shape, out_root, seed_info, csv=csv, shape_source=source)
