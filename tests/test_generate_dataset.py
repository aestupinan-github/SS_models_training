"""Tests for the ss_cube-wall dataset generator (spec 02).

T02.1: configuration and validation (R02.11, R02.13).
T02.2: shape description from a point set (R02.1, R02.3).
T02.3: orientation sampler, uniform and special, with spin, yaw and
       canonicalization (R02.3).
T02.4: orientation sampler, near_kink by bisection, and exclusion by u
       (R02.3, D02.3).
T02.5: sd sampler and row assembly with recomputed labels
       (R02.1, R02.2, R02.4, R02.6).
T02.6: coverage table and acceptance checks (R02.5, R02.12).
T02.7: writer, versioning, provenance, seeding (R02.7, R02.8, R02.9, R02.10).
T02.8: CLI and summary (R02.14, R02.11, R02.13).
"""

import csv
import dataclasses
import hashlib
import json
import math
import shutil
import subprocess
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactions", "ss_cube-wall", "python"))

import cube_wall as cw  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402
from generate_dataset import (  # noqa: E402
    ConfigError,
    ACCEPTANCE_CHECKS,
    format_summary,
    main,
    SEED_STREAMS,
    generate,
    generate_rows,
    write_dataset,
    COMPONENT_CODES,
    COVERAGE_DELTA_RANGE,
    COVERAGE_SD_RANGE,
    acceptance_checks,
    coverage_table,
    require_acceptance,
    GenerationError,
    ROW_COLUMNS,
    SD_PART_CODES,
    allocate_counts,
    build_rows,
    sample_orientations,
    sample_sd,
    Shape,
    ShapeError,
    canonicalize_quaternions,
    exclusion_mask,
    rotation_from_u,
    sample_near_kink,
    sample_special,
    sample_uniform,
    shape_from_points,
    special_directions,
    support_gap,
    u_from_quats,
    DEFAULT_ORIENT_FRACTIONS,
    DEFAULT_SD_FRACTIONS,
    GenConfig,
    SMOKE_MAX_ROWS,
    validate_config,
)

SEED = 20261003


def cfg(**overrides):
    """Valid smoke-size config with approved default fractions, then overrides."""
    base = dict(n_orient=1250, k_per_orient=8, seed=SEED)
    base.update(overrides)
    return GenConfig(**base)


class TestDefaults(unittest.TestCase):
    """Approved defaults (spec 02 R02.2, R02.3, R02.13)."""

    def test_valid_default_config_passes(self):
        self.assertIsNone(validate_config(cfg()))

    def test_default_fractions_are_the_approved_values(self):
        self.assertEqual(DEFAULT_SD_FRACTIONS, {"penetration": 0.40, "gap": 0.30, "near_contact": 0.05,
                                                "exact_contact": 0.01, "gap_safety": 0.24})
        self.assertEqual(DEFAULT_ORIENT_FRACTIONS, {"uniform": 0.70, "near_kink": 0.29, "special": 0.01})
        c = cfg()
        self.assertEqual(c.sd_fractions, DEFAULT_SD_FRACTIONS)
        self.assertEqual(c.orient_fractions, DEFAULT_ORIENT_FRACTIONS)
        self.assertEqual(c.sd_log_min, 1e-5)
        self.assertEqual(c.delta_range, (1e-6, 1e-1))
        self.assertFalse(c.allow_full_scale)

    def test_default_dicts_are_not_shared(self):
        a, b = cfg(), cfg()
        self.assertIsNot(a.sd_fractions, b.sd_fractions)
        self.assertIsNot(a.sd_fractions, DEFAULT_SD_FRACTIONS)

    def test_config_is_frozen(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            cfg().seed = 1

    def test_config_error_is_value_error_and_names_parameter(self):
        with self.assertRaises(ConfigError) as ctx:
            validate_config(cfg(n_orient=0))
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertEqual(ctx.exception.param, "n_orient")
        self.assertIn("n_orient", str(ctx.exception))

    def test_rejects_non_config_object(self):
        with self.assertRaises(ConfigError):
            validate_config({"n_orient": 1})


class TestInvalidCases(unittest.TestCase):
    """Each invalid case of R02.11 raises ConfigError naming the parameter."""

    def assertRejects(self, param, **overrides):
        with self.assertRaises(ConfigError, msg=f"{overrides}") as ctx:
            validate_config(cfg(**overrides))
        self.assertEqual(ctx.exception.param, param, msg=f"{overrides}: {ctx.exception}")

    # non-positive counts
    def test_non_positive_counts(self):
        for name in ("n_orient", "k_per_orient", "m_min"):
            for bad in (0, -1):
                self.assertRejects(name, **{name: bad})

    def test_non_integer_counts(self):
        for name in ("n_orient", "k_per_orient", "m_min"):
            for bad in (1.5, 2.0, True, "8", None):
                self.assertRejects(name, **{name: bad})

    def test_numpy_integer_counts_accepted(self):
        self.assertIsNone(validate_config(cfg(n_orient=np.int64(1250), k_per_orient=np.int32(8))))

    # missing seed
    def test_missing_seed(self):
        with self.assertRaises(ConfigError) as ctx:
            validate_config(GenConfig(n_orient=1250, k_per_orient=8))
        self.assertEqual(ctx.exception.param, "seed")

    def test_invalid_seed(self):
        for bad in (None, -1, 1.0, True, "1"):
            self.assertRejects("seed", seed=bad)

    def test_seed_zero_is_valid(self):
        self.assertIsNone(validate_config(cfg(seed=0)))

    # fractions negative or not summing to 1
    def test_negative_fraction(self):
        sd = dict(DEFAULT_SD_FRACTIONS, penetration=-0.1, gap=0.80)
        self.assertRejects("sd_fractions", sd_fractions=sd)
        orient = dict(DEFAULT_ORIENT_FRACTIONS, uniform=-0.29, near_kink=1.28)
        self.assertRejects("orient_fractions", orient_fractions=orient)

    def test_fractions_not_summing_to_one(self):
        self.assertRejects("sd_fractions", sd_fractions=dict(DEFAULT_SD_FRACTIONS, gap=0.31))
        self.assertRejects("orient_fractions", orient_fractions=dict(DEFAULT_ORIENT_FRACTIONS, special=0.0))

    def test_non_finite_fraction(self):
        self.assertRejects("sd_fractions", sd_fractions=dict(DEFAULT_SD_FRACTIONS, gap=float("nan")))
        self.assertRejects("orient_fractions", orient_fractions=dict(DEFAULT_ORIENT_FRACTIONS, uniform=float("inf")))

    def test_missing_or_unknown_fraction_keys(self):
        missing = {k: v for k, v in DEFAULT_SD_FRACTIONS.items() if k != "gap_safety"}
        missing["penetration"] += 0.24
        self.assertRejects("sd_fractions", sd_fractions=missing)
        unknown = dict(DEFAULT_ORIENT_FRACTIONS, special=0.0, edge=0.01)
        self.assertRejects("orient_fractions", orient_fractions=unknown)

    def test_fractions_wrong_type(self):
        self.assertRejects("sd_fractions", sd_fractions=[0.4, 0.3, 0.05, 0.01, 0.24])
        self.assertRejects("orient_fractions", orient_fractions=dict(DEFAULT_ORIENT_FRACTIONS, uniform="0.70"))

    def test_zero_fraction_allowed(self):
        # disabling a component is allowed if the rest sums to 1
        orient = {"uniform": 0.71, "near_kink": 0.29, "special": 0.0}
        self.assertIsNone(validate_config(cfg(orient_fractions=orient)))

    # ranges outside spec 01; lower bound not below upper bound
    def test_sd_log_min_range(self):
        # must satisfy 0 < sd_log_min < 0.1 (spec 01 accuracy zone upper bound)
        for bad in (0.0, -1e-5, 0.1, 0.2, float("nan"), float("inf"), "1e-5", None):
            self.assertRejects("sd_log_min", sd_log_min=bad)
        self.assertIsNone(validate_config(cfg(sd_log_min=1e-8)))

    def test_delta_range(self):
        for bad in ((0.0, 1e-1), (-1e-6, 1e-1), (1e-1, 1e-6), (1e-3, 1e-3),
                    (1e-6, float("nan")), (1e-6, float("inf")), (1e-6,), (1e-6, 1e-3, 1e-1), None, "ab"):
            self.assertRejects("delta_range", delta_range=bad)
        self.assertIsNone(validate_config(cfg(delta_range=[1e-5, 1e-2])))

    def test_exclude_angle(self):
        for bad in (-1.0, 180.1, float("nan"), None, "5"):
            self.assertRejects("exclude_angle_deg", exclude_angle_deg=bad)
        self.assertIsNone(validate_config(cfg(exclude_angle_deg=180.0)))

    def test_exclude_quats(self):
        for bad in (np.zeros(4) + 1.0, np.ones((2, 3)), np.array([[1.0, np.nan, 0, 0]]),
                    np.array([[1.0, 0, 0, 0], [0.0, 0, 0, 0]]), "q"):
            self.assertRejects("exclude_quats", exclude_quats=bad, exclude_angle_deg=1.0)
        ok = np.array([[1.0, 0, 0, 0], [0.0, 2.0, 0, 0]])  # non-unit allowed, normalized later (R01.4)
        self.assertIsNone(validate_config(cfg(exclude_quats=ok, exclude_angle_deg=1.0)))

    def test_exclude_quats_without_angle(self):
        # a list without a positive angle would exclude nothing: reported, not ignored
        self.assertRejects("exclude_angle_deg", exclude_quats=np.array([[1.0, 0, 0, 0]]), exclude_angle_deg=0.0)

    def test_allow_full_scale_must_be_bool(self):
        for bad in (1, "yes", None):
            self.assertRejects("allow_full_scale", allow_full_scale=bad)


class TestSizeLimit(unittest.TestCase):
    """R02.13 / R02.11: above the smoke limit only with explicit override."""

    def test_smoke_limit_value(self):
        self.assertEqual(SMOKE_MAX_ROWS, 10_000)

    def test_at_limit_passes(self):
        self.assertIsNone(validate_config(cfg(n_orient=1250, k_per_orient=8)))

    def test_above_limit_refused(self):
        with self.assertRaises(ConfigError) as ctx:
            validate_config(cfg(n_orient=1251, k_per_orient=8))
        self.assertEqual(ctx.exception.param, "allow_full_scale")
        self.assertIn("10008", str(ctx.exception))

    def test_above_limit_with_override_passes(self):
        self.assertIsNone(validate_config(cfg(n_orient=10**6, k_per_orient=8, allow_full_scale=True)))


class TestNoSideEffects(unittest.TestCase):
    """R02.11: validation stops before writing anything."""

    def test_validation_writes_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            old = os.getcwd()
            os.chdir(d)
            try:
                validate_config(cfg())
                with self.assertRaises(ConfigError):
                    validate_config(cfg(n_orient=0))
            finally:
                os.chdir(old)
            self.assertEqual(os.listdir(d), [])


TETRA = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)


class TestShapeCube(unittest.TestCase):
    """T02.2 on the cube: 6 facets, 12 edges, constants equal to cube_wall."""

    @classmethod
    def setUpClass(cls):
        cls.s = shape_from_points(cw.CUBE_VERTICES)

    def test_type(self):
        self.assertIsInstance(self.s, Shape)

    def test_counts(self):
        self.assertEqual(len(self.s.facet_normals), 6)
        self.assertEqual(len(self.s.edge_facets), 12)
        self.assertEqual(len(self.s.vertices), 8)

    def test_constants_equal_cube_wall(self):
        self.assertEqual(self.s.r_circ, cw.R_CIRCUMSCRIBED)
        self.assertEqual(self.s.h_min, cw.H_MIN)

    def test_facet_normals_are_the_six_axis_directions(self):
        n = np.round(self.s.facet_normals, 12)
        expected = {tuple(v) for v in np.vstack([np.eye(3), -np.eye(3)])}
        self.assertEqual({tuple(v) for v in n}, expected)
        np.testing.assert_allclose(np.linalg.norm(self.s.facet_normals, axis=1), 1.0, atol=1e-15)

    def test_facet_offsets_and_vertex_sets(self):
        np.testing.assert_allclose(self.s.facet_offsets, 0.5, atol=1e-15)
        for f, verts in enumerate(self.s.facet_vertices):
            self.assertEqual(len(verts), 4)
            # all facet vertices lie on the facet plane
            d = self.s.vertices[list(verts)] @ self.s.facet_normals[f]
            np.testing.assert_allclose(d, self.s.facet_offsets[f], atol=1e-12)

    def test_outward_normals(self):
        # every hull vertex is on the inner side of every facet plane
        d = self.s.vertices @ self.s.facet_normals.T - self.s.facet_offsets
        self.assertLessEqual(d.max(), 1e-12)

    def test_edges(self):
        seen = set()
        for (i, j), (a, b) in zip(self.s.edge_facets, self.s.edge_vertices):
            self.assertLess(i, j)
            self.assertEqual(set(self.s.facet_vertices[i]) & set(self.s.facet_vertices[j]), {a, b})
            # cube edges join perpendicular faces and have length 1
            self.assertAlmostEqual(self.s.facet_normals[i] @ self.s.facet_normals[j], 0.0, places=14)
            self.assertAlmostEqual(np.linalg.norm(self.s.vertices[a] - self.s.vertices[b]), 1.0, places=14)
            seen.add(frozenset((a, b)))
        self.assertEqual(len(seen), 12)

    def test_arrays_are_read_only(self):
        with self.assertRaises(ValueError):
            self.s.facet_normals[0, 0] = 2.0
        with self.assertRaises(ValueError):
            self.s.vertices[0, 0] = 2.0

    def test_hull_vertices_give_same_support_height(self):
        for r in R.random(200, random_state=SEED):
            x, y, z, w = r.as_quat()
            q = [w, x, y, z]
            self.assertEqual(cw.support_height(self.s.vertices, q), cw.support_height(cw.CUBE_VERTICES, q))

    def test_h_min_is_min_of_support_height(self):
        hs = [cw.support_height(self.s.vertices, [w, x, y, z])
              for x, y, z, w in R.random(5000, random_state=SEED).as_quat()]
        self.assertGreaterEqual(min(hs), self.s.h_min - 1e-12)


class TestShapeGeneral(unittest.TestCase):
    """Shape-general behaviour (R01.16): other point sets."""

    def test_rotated_cube_same_topology_and_constants(self):
        pts = R.random(random_state=7).apply(cw.CUBE_VERTICES)
        s = shape_from_points(pts)
        self.assertEqual((len(s.facet_normals), len(s.edge_facets), len(s.vertices)), (6, 12, 8))
        self.assertAlmostEqual(s.r_circ, cw.R_CIRCUMSCRIBED, places=14)
        self.assertAlmostEqual(s.h_min, cw.H_MIN, places=14)

    def test_scaled_cube_scales_constants(self):
        s = shape_from_points(3.0 * cw.CUBE_VERTICES)
        self.assertAlmostEqual(s.r_circ, 3.0 * cw.R_CIRCUMSCRIBED, places=14)
        self.assertAlmostEqual(s.h_min, 1.5, places=14)

    def test_interior_points_ignored(self):
        rng = np.random.default_rng(SEED)
        pts = np.vstack([cw.CUBE_VERTICES, rng.uniform(-0.45, 0.45, (100, 3))])
        s = shape_from_points(pts)
        self.assertEqual((len(s.facet_normals), len(s.edge_facets), len(s.vertices)), (6, 12, 8))
        self.assertEqual(s.h_min, 0.5)

    def test_tetrahedron(self):
        s = shape_from_points(TETRA)
        self.assertEqual((len(s.facet_normals), len(s.edge_facets), len(s.vertices)), (4, 6, 4))
        self.assertAlmostEqual(s.r_circ, math.sqrt(3), places=14)
        self.assertAlmostEqual(s.h_min, 1 / math.sqrt(3), places=14)
        for verts in s.facet_vertices:
            self.assertEqual(len(verts), 3)

    def test_non_convex_point_set_uses_hull(self):
        # L-shaped prism (non-convex); the notch corner is not a hull vertex
        base = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float) - 0.8
        pts = np.vstack([np.c_[base, np.full(6, -0.5)], np.c_[base, np.full(6, 0.5)]])
        s = shape_from_points(pts)
        self.assertEqual(len(s.vertices), 10)          # notch corners (2) dropped
        self.assertEqual(len(s.facet_normals), 7)      # 5 sides (incl. the cut) + top + bottom
        self.assertEqual(len(s.edge_facets), 15)       # Euler: V - E + F = 2
        self.assertGreater(s.h_min, 0.0)

    def test_euler_characteristic(self):
        for pts in (cw.CUBE_VERTICES, TETRA, R.random(random_state=3).apply(cw.CUBE_VERTICES)):
            s = shape_from_points(pts)
            self.assertEqual(len(s.vertices) - len(s.edge_facets) + len(s.facet_normals), 2)


class TestShapeInvalid(unittest.TestCase):
    """Invalid point sets raise ShapeError (a ValueError)."""

    def assertShapeError(self, pts):
        with self.assertRaises(ShapeError) as ctx:
            shape_from_points(pts)
        self.assertIsInstance(ctx.exception, ValueError)
        return ctx.exception

    def test_wrong_shape(self):
        for bad in (np.zeros((8, 2)), np.zeros(24), np.zeros((0, 3)), "points"):
            self.assertShapeError(bad)

    def test_too_few_points(self):
        self.assertShapeError(TETRA[:3])

    def test_non_finite(self):
        pts = cw.CUBE_VERTICES.copy()
        pts[0, 0] = np.nan
        self.assertShapeError(pts)

    def test_degenerate_flat_set(self):
        pts = cw.CUBE_VERTICES.copy()
        pts[:, 2] = 0.0
        self.assertShapeError(pts)

    def test_origin_outside_hull(self):
        e = self.assertShapeError(cw.CUBE_VERTICES + [0.6, 0.0, 0.0])
        self.assertIn("origin", str(e))

    def test_origin_on_hull_boundary(self):
        self.assertShapeError(cw.CUBE_VERTICES + [0.5, 0.0, 0.0])


def _cube_shape():
    return shape_from_points(cw.CUBE_VERTICES)


class TestVectorHelpers(unittest.TestCase):
    """T02.3 helpers: vectorized canonicalization, u, support gap, rotation from u."""

    def test_canonicalize_matches_spec01_reference_bitwise(self):
        q = R.random(5000, random_state=SEED).as_quat()[:, [3, 0, 1, 2]]
        special = np.array([[0, 1, 0, 0], [0, -1, 0, 0], [0, 0, -0.6, 0.8], [0, 0, 0, -1],
                            [-1, 0, 0, 0], [-2, 0, 0, 0], [0, 0, 3, 0]], float)
        q = np.vstack([q, -q, special])
        got = canonicalize_quaternions(q)
        ref = np.array([cw.canonicalize_quaternion(x) for x in q])
        self.assertTrue(np.array_equal(got, ref))

    def test_canonicalize_rejects_invalid(self):
        for bad in (np.zeros((1, 4)), np.array([[np.nan, 0, 0, 1]]), np.ones((2, 3))):
            with self.assertRaises(ValueError):
                canonicalize_quaternions(bad)

    def test_u_from_quats_matches_reference_support(self):
        s = _cube_shape()
        q = canonicalize_quaternions(R.random(500, random_state=SEED).as_quat()[:, [3, 0, 1, 2]])
        u = u_from_quats(q)
        np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1.0, atol=1e-14)
        h = (u @ s.vertices.T).max(axis=1)
        ref = np.array([cw.support_height(cw.CUBE_VERTICES, x) for x in q])
        np.testing.assert_allclose(h, ref, rtol=0, atol=1e-15)

    def test_support_gap(self):
        s = _cube_shape()
        u = np.array([[0, 0, 1.0], [1, 1, 0], [1, 1, 1], [0.1, 0.2, 1.0]])
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        d = support_gap(s, u)
        self.assertEqual(d[0], 0.0)
        self.assertLess(d[1], 1e-15)
        self.assertAlmostEqual(d[2], 1 / math.sqrt(3), places=14)
        self.assertGreater(d[3], 0.0)

    def test_rotation_from_u_maps_u_to_wall(self):
        rng = np.random.default_rng(SEED)
        u = rng.normal(size=(2000, 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        u = np.vstack([u, [[0, 0, 1.0], [0, 0, -1.0], [1, 0, 0]]])   # incl. antiparallel case
        rot = rotation_from_u(u)
        np.testing.assert_allclose(rot.apply(u), np.tile([0, 0, -1.0], (len(u), 1)), atol=1e-14)


class TestSampleUniform(unittest.TestCase):
    """`uniform` component: Haar SO(3), yaw, canonical quaternions."""

    @classmethod
    def setUpClass(cls):
        cls.q = sample_uniform(20000, np.random.default_rng(SEED))

    def test_shape_and_canonical(self):
        self.assertEqual(self.q.shape, (20000, 4))
        self.assertEqual(self.q.dtype, np.float64)
        np.testing.assert_allclose(np.linalg.norm(self.q, axis=1), 1.0, atol=1e-15)
        # idempotent up to the last bit (re-normalization), as in spec 01 tests
        np.testing.assert_allclose(self.q, canonicalize_quaternions(self.q), rtol=0, atol=1e-15)
        self.assertTrue(np.all(self.q[:, 0] >= 0.0))

    def test_haar_statistics(self):
        # Haar measure: body direction u uniform on S^2 -> mean ~0, E[u_i^2] = 1/3;
        # rotation angle density (1 - cos a)/pi -> E[cos a] = -1/2... for |q|: E[qw^2] = 1/4 per comp
        u = u_from_quats(self.q)
        n = len(u)
        self.assertLess(np.abs(u.mean(axis=0)).max(), 5 / math.sqrt(n))
        np.testing.assert_allclose((u ** 2).mean(axis=0), 1 / 3, atol=5 * math.sqrt(4 / 45 / n))
        np.testing.assert_allclose((self.q ** 2).mean(axis=0), 0.25, atol=0.01)

    def test_yaw_statistics(self):
        # world-frame yaw of the body x axis is uniform: mean of cos and sin near 0
        x_world = R.from_quat(self.q[:, [1, 2, 3, 0]]).apply([1.0, 0, 0])
        psi = np.arctan2(x_world[:, 1], x_world[:, 0])
        tol = 5 / math.sqrt(2 * len(psi))
        self.assertLess(abs(np.cos(psi).mean()), tol)
        self.assertLess(abs(np.sin(psi).mean()), tol)

    def test_reproducible(self):
        a = sample_uniform(100, np.random.default_rng(1))
        b = sample_uniform(100, np.random.default_rng(1))
        c = sample_uniform(100, np.random.default_rng(2))
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_invalid_count(self):
        for bad in (0, -1, 1.5):
            with self.assertRaises(ValueError):
                sample_uniform(bad, np.random.default_rng(1))


class TestSpecial(unittest.TestCase):
    """`special` component: face, edge and vertex directions, exact h and Delta."""

    @classmethod
    def setUpClass(cls):
        cls.s = _cube_shape()
        cls.dirs, cls.kind = special_directions(cls.s)

    def test_special_set(self):
        self.assertEqual(len(self.dirs), 6 + 12 + 8)
        self.assertEqual([int((self.kind == k).sum()) for k in ("face", "edge", "vertex")], [6, 12, 8])
        np.testing.assert_allclose(np.linalg.norm(self.dirs, axis=1), 1.0, atol=1e-15)

    def test_exact_h_and_delta_per_kind(self):
        h = (self.dirs @ self.s.vertices.T).max(axis=1)
        d = support_gap(self.s, self.dirs)
        face, edge, vert = (self.kind == "face"), (self.kind == "edge"), (self.kind == "vertex")
        np.testing.assert_allclose(h[face], 0.5, atol=1e-15)
        np.testing.assert_allclose(h[edge], math.sqrt(2) / 2, atol=1e-15)
        np.testing.assert_allclose(h[vert], math.sqrt(3) / 2, atol=1e-15)
        self.assertLessEqual(d[face].max(), 1e-15)
        self.assertLessEqual(d[edge].max(), 1e-15)
        np.testing.assert_allclose(d[vert], 1 / math.sqrt(3), atol=1e-15)

    def test_sampled_special_quaternions(self):
        rng = np.random.default_rng(SEED)
        q, which = sample_special(self.s, 3000, rng)
        self.assertEqual(q.shape, (3000, 4))
        np.testing.assert_allclose(q, canonicalize_quaternions(q), rtol=0, atol=1e-15)
        self.assertTrue(np.all(q[:, 0] >= 0.0))
        u = u_from_quats(q)
        # the sampled u is exactly (to round-off) the chosen special direction
        np.testing.assert_allclose(u, self.dirs[which], atol=1e-14)
        h = np.array([cw.support_height(cw.CUBE_VERTICES, x) for x in q[:300]])
        ref = (self.dirs[which[:300]] @ self.s.vertices.T).max(axis=1)
        np.testing.assert_allclose(h, ref, atol=1e-14)
        # every special direction is drawn
        self.assertEqual(len(np.unique(which)), len(self.dirs))

    def test_special_spin_and_yaw_vary(self):
        # same u, different full orientations (spin about u and yaw about z)
        q, which = sample_special(self.s, 3000, np.random.default_rng(SEED))
        sel = q[which == which[0]]
        self.assertGreater(len(sel), 10)
        self.assertGreater(np.ptp(sel, axis=0).max(), 0.5)

    def test_special_on_tetrahedron(self):
        s = shape_from_points(TETRA)
        dirs, kind = special_directions(s)
        self.assertEqual(len(dirs), 4 + 6 + 4)
        d = support_gap(s, dirs)
        self.assertLessEqual(d[kind != "vertex"].max(), 1e-14)
        self.assertGreater(d[kind == "vertex"].min(), 0.0)

    def test_component_codes(self):
        self.assertEqual(COMPONENT_CODES, {"uniform": 0, "near_kink": 1, "special": 2})


class TestNearKink(unittest.TestCase):
    """`near_kink` component: target support gap by bisection on a tilt angle."""

    N = 20000

    @classmethod
    def setUpClass(cls):
        cls.s = _cube_shape()
        cls.q, cls.target = sample_near_kink(cls.s, cls.N, np.random.default_rng(SEED))
        cls.delta = support_gap(cls.s, u_from_quats(cls.q))

    def test_shapes_and_canonical(self):
        self.assertEqual(self.q.shape, (self.N, 4))
        self.assertEqual(self.target.shape, (self.N,))
        np.testing.assert_allclose(self.q, canonicalize_quaternions(self.q), rtol=0, atol=1e-15)
        self.assertTrue(np.all(self.q[:, 0] >= 0.0))

    def test_achieved_delta_matches_target(self):
        # Done-when: within 1e-9 relative of target (measured worst 6.5e-10 incl. quaternion chain)
        rel = np.abs(self.delta - self.target) / self.target
        self.assertLess(rel.max(), 1e-9)

    def test_targets_in_range_and_log_uniform(self):
        lo, hi = 1e-6, 1e-1
        self.assertGreaterEqual(self.target.min(), lo)
        self.assertLessEqual(self.target.max(), hi)
        # log-uniform: each decade holds about 1/5 of the targets
        counts = [int(((self.target >= a) & (self.target < 10 * a)).sum()) for a in 10.0 ** np.arange(-6, -1)]
        for c in counts:
            self.assertGreater(c, 0.8 * self.N / 5)
            self.assertLess(c, 1.2 * self.N / 5)

    def test_every_decade_populated(self):
        for a in 10.0 ** np.arange(-6, -1):
            self.assertGreater(int(((self.delta >= a) & (self.delta < 10 * a)).sum()), 0, msg=f"decade {a}")

    def test_near_faces_and_edges(self):
        # Cube kink set: the great circles u_i = 0 (edges), crossing at the faces.
        # Edge mode starts anywhere on an edge arc (uniform in arc angle), so
        # "near an edge" means near a circle, away from the faces.
        # Measured: all rows near a circle (uniform SO(3): 28%), 55% near a face
        # (uniform: 3%), 45% on a circle away from faces.
        u = np.sort(np.abs(u_from_quats(self.q)), axis=1)
        near_circle = u[:, 0] < 0.1
        near_face = u[:, 2] > 0.99
        self.assertGreater(near_circle.mean(), 0.99)
        self.assertGreater(near_face.mean(), 0.4)
        self.assertGreater((near_circle & ~near_face).mean(), 0.3)

    def test_custom_delta_range(self):
        q, t = sample_near_kink(self.s, 500, np.random.default_rng(1), delta_range=(1e-4, 1e-3))
        self.assertTrue(np.all((t >= 1e-4) & (t <= 1e-3)))
        d = support_gap(self.s, u_from_quats(q))
        self.assertLess((np.abs(d - t) / t).max(), 1e-9)

    def test_reproducible(self):
        a = sample_near_kink(self.s, 200, np.random.default_rng(3))
        b = sample_near_kink(self.s, 200, np.random.default_rng(3))
        c = sample_near_kink(self.s, 200, np.random.default_rng(4))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))
        self.assertFalse(np.array_equal(a[0], c[0]))

    def test_tetrahedron(self):
        # Shape-general check. The 1e-9 relative criterion is for the cube; on the
        # tetrahedron (r_circ = 1.73) round-off of the quaternion chain gives an
        # absolute error of about 2e-15 (measured worst relative 1.5e-9 at 1.2e-6).
        s = shape_from_points(TETRA)
        q, t = sample_near_kink(s, 2000, np.random.default_rng(SEED))
        d = support_gap(s, u_from_quats(q))
        self.assertLess(np.abs(d - t).max(), 1e-14)
        self.assertLess((np.abs(d - t) / t).max(), 1e-8)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            sample_near_kink(self.s, 0, np.random.default_rng(1))
        for bad in ((0.0, 1e-1), (1e-3, 1e-4)):
            with self.assertRaises(ValueError):
                sample_near_kink(self.s, 10, np.random.default_rng(1), delta_range=bad)


class TestExclusion(unittest.TestCase):
    """Exclusion by body-frame direction u (D02.3)."""

    def test_mask_by_u_angle(self):
        excl = np.array([[1.0, 0, 0, 0]])                  # u = (0, 0, 1)... of identity
        u_ex = u_from_quats(excl)[0]
        rng = np.random.default_rng(SEED)
        q = sample_uniform(20000, rng)
        u = u_from_quats(q)
        ang = np.degrees(np.arccos(np.clip(u @ u_ex, -1, 1)))
        m = exclusion_mask(q, excl, 10.0)
        self.assertTrue(np.array_equal(m, ang < 10.0))
        self.assertGreater(m.sum(), 0)

    def test_mask_ignores_spin_and_yaw(self):
        # an excluded orientation excludes every orientation with the same u
        s = _cube_shape()
        excl, _ = sample_special(s, 1, np.random.default_rng(1))
        u = u_from_quats(excl)
        same_u = canonicalize_quaternions(_from_rot(rotation_from_u(np.repeat(u, 50, axis=0)), u, 50))
        self.assertTrue(np.all(exclusion_mask(same_u, excl, 0.01)))

    def test_mask_none(self):
        q = sample_uniform(10, np.random.default_rng(1))
        self.assertFalse(np.any(exclusion_mask(q, None, 0.0)))

    def test_samplers_respect_exclusion(self):
        s = _cube_shape()
        rng = np.random.default_rng(SEED)
        excl = np.vstack([sample_special(s, 3, rng)[0], sample_uniform(3, rng)])
        u_ex = u_from_quats(excl)
        angle = 15.0
        for name, q in (
            ("uniform", sample_uniform(5000, np.random.default_rng(1), exclude_quats=excl, exclude_angle_deg=angle)),
            ("special", sample_special(s, 2000, np.random.default_rng(2), exclude_quats=excl, exclude_angle_deg=angle)[0]),
            ("near_kink", sample_near_kink(s, 2000, np.random.default_rng(3), exclude_quats=excl, exclude_angle_deg=angle)[0]),
        ):
            u = u_from_quats(q)
            ang = np.degrees(np.arccos(np.clip(u @ u_ex.T, -1, 1))).min(axis=1)
            self.assertEqual(len(q), {"uniform": 5000, "special": 2000, "near_kink": 2000}[name])
            self.assertGreaterEqual(ang.min(), angle, msg=name)

    def test_all_excluded_special_raises(self):
        # every special direction excluded -> cannot sample: error, not an endless loop
        s = _cube_shape()
        excl = sample_special(s, 400, np.random.default_rng(5))[0]
        with self.assertRaises(ValueError):
            sample_special(s, 10, np.random.default_rng(6), exclude_quats=excl, exclude_angle_deg=1.0)


def _from_rot(rot0, u, n):
    """Helper: n orientations with the same u, different spin and yaw."""
    rng = np.random.default_rng(99)
    phi = rng.uniform(0, 2 * np.pi, n)
    psi = rng.uniform(0, 2 * np.pi, n)
    rot = R.from_rotvec(np.outer(psi, [0, 0, 1.0])) * rot0 * R.from_rotvec(np.repeat(u, n, axis=0) * phi[:, None])
    return rot.as_quat()[:, [3, 0, 1, 2]]


class TestAllocateCounts(unittest.TestCase):
    """Stratified counts (design; D02.4: round half up, remainder to the largest part)."""

    def test_exact_default_at_k100(self):
        c = allocate_counts(DEFAULT_SD_FRACTIONS, 100)
        self.assertEqual(c, {"penetration": 40, "gap": 30, "near_contact": 5, "exact_contact": 1, "gap_safety": 24})

    def test_sum_always_k(self):
        for k in range(1, 400):
            self.assertEqual(sum(allocate_counts(DEFAULT_SD_FRACTIONS, k).values()), k)
            self.assertEqual(sum(allocate_counts(DEFAULT_ORIENT_FRACTIONS, k).values()), k)

    def test_round_half_up_not_bankers(self):
        # exact_contact 0.01 * 50 = 0.5 -> 1 (Python round(0.5) would give 0)
        self.assertEqual(allocate_counts(DEFAULT_SD_FRACTIONS, 50)["exact_contact"], 1)

    def test_zero_fraction_gets_zero(self):
        c = allocate_counts({"uniform": 0.71, "near_kink": 0.29, "special": 0.0}, 1000)
        self.assertEqual(c["special"], 0)
        self.assertEqual(sum(c.values()), 1000)

    def test_never_negative(self):
        for k in range(1, 400):
            self.assertTrue(all(v >= 0 for v in allocate_counts(DEFAULT_SD_FRACTIONS, k).values()))


def _orients(n=1250, **kw):
    c = cfg(n_orient=n, **kw)
    s = _cube_shape()
    q, comp = sample_orientations(c, s, np.random.default_rng(SEED))
    return c, s, q, comp


class TestSampleOrientations(unittest.TestCase):
    """Mixture of components by fractions (R02.3), deterministic order (R02.7)."""

    @classmethod
    def setUpClass(cls):
        cls.c, cls.s, cls.q, cls.comp = _orients()

    def test_counts_per_component(self):
        want = allocate_counts(DEFAULT_ORIENT_FRACTIONS, 1250)
        for name, code in COMPONENT_CODES.items():
            self.assertEqual(int((self.comp == code).sum()), want[name])
        self.assertEqual(self.q.shape, (1250, 4))
        self.assertEqual(self.comp.dtype, np.int8)

    def test_order_is_by_component(self):
        # no shuffling at generation: uniform, then near_kink, then special
        self.assertTrue(np.all(np.diff(self.comp) >= 0))

    def test_canonical(self):
        np.testing.assert_allclose(self.q, canonicalize_quaternions(self.q), rtol=0, atol=1e-15)

    def test_special_rows_have_special_u(self):
        dirs, _ = special_directions(self.s)
        u = u_from_quats(self.q[self.comp == COMPONENT_CODES["special"]])
        self.assertLess(np.abs(u @ dirs.T).max(axis=1).min(), 1.0 + 1e-14)
        self.assertGreater(np.abs(u @ dirs.T).max(axis=1).min(), 1.0 - 1e-14)

    def test_reproducible(self):
        a = sample_orientations(self.c, self.s, np.random.default_rng(1))
        b = sample_orientations(self.c, self.s, np.random.default_rng(1))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_exclusion_passed_through(self):
        excl = sample_uniform(4, np.random.default_rng(9))
        c, s, q, comp = _orients(exclude_quats=excl, exclude_angle_deg=20.0)
        self.assertFalse(np.any(exclusion_mask(q, excl, 20.0)))

    def test_validates_config(self):
        with self.assertRaises(ConfigError):
            sample_orientations(cfg(n_orient=0), self.s, np.random.default_rng(1))


class TestSampleSd(unittest.TestCase):
    """sd parts and ranges (R02.2, R02.4)."""

    @classmethod
    def setUpClass(cls):
        cls.s = _cube_shape()
        cls.c = cfg(n_orient=100, k_per_orient=100)          # stratified (K = 100)
        rng = np.random.default_rng(SEED)
        cls.h = support_gap(cls.s, u_from_quats(sample_uniform(100, rng))) * 0 + \
            (u_from_quats(sample_uniform(100, np.random.default_rng(2))) @ cls.s.vertices.T).max(axis=1)
        cls.sd, cls.part = sample_sd(cls.c, cls.s, cls.h, np.random.default_rng(SEED))
        cls.hr = np.repeat(cls.h, 100)

    def test_lengths_and_codes(self):
        self.assertEqual(self.sd.shape, (10000,))
        self.assertEqual(self.part.dtype, np.int8)
        self.assertEqual(SD_PART_CODES, {"penetration": 0, "gap": 1, "near_contact": 2,
                                         "exact_contact": 3, "gap_safety": 4})

    def test_ranges_per_part(self):
        P = SD_PART_CODES
        sd, part = self.sd, self.part
        pen, gap, near = sd[part == P["penetration"]], sd[part == P["gap"]], sd[part == P["near_contact"]]
        self.assertTrue(np.all((pen >= -0.1) & (pen <= -1e-5)))
        self.assertTrue(np.all((gap >= 1e-5) & (gap <= 0.1)))
        self.assertTrue(np.all((near > -1e-5) & (near < 1e-5)))
        self.assertTrue(np.all(sd[part == P["exact_contact"]] == 0.0))
        gs = part == P["gap_safety"]
        self.assertTrue(np.all(sd[gs] > 0.1))
        self.assertTrue(np.all(sd[gs] <= cw.Z_PREFILTER - self.hr[gs]))

    def test_stratified_per_orientation(self):
        want = allocate_counts(DEFAULT_SD_FRACTIONS, 100)
        parts = self.part.reshape(100, 100)
        for code_name, code in SD_PART_CODES.items():
            per = (parts == code).sum(axis=1)
            if code_name in ("gap", "gap_safety"):
                # gap safety may fall back to gap (empty interval); the total is preserved
                continue
            self.assertTrue(np.all(per == want[code_name]), msg=code_name)
        both = ((parts == SD_PART_CODES["gap"]) | (parts == SD_PART_CODES["gap_safety"])).sum(axis=1)
        self.assertTrue(np.all(both == want["gap"] + want["gap_safety"]))

    def test_log_uniform_penetration(self):
        pen = -self.sd[self.part == SD_PART_CODES["penetration"]]
        counts = [int(((pen >= a) & (pen < 10 * a)).sum()) for a in 10.0 ** np.arange(-5, -1)]
        for cnt in counts:
            self.assertGreater(cnt, 0.8 * len(pen) / 4)
            self.assertLess(cnt, 1.2 * len(pen) / 4)

    def test_gap_safety_fallback_at_corner_down(self):
        # corner-down: Z_PREFILTER - h = 0.1 exactly -> empty interval -> drawn as gap
        h = np.full(20, math.sqrt(3) / 2)
        sd, part = sample_sd(cfg(n_orient=20, k_per_orient=100), self.s, h, np.random.default_rng(1))
        self.assertEqual(int((part == SD_PART_CODES["gap_safety"]).sum()), 0)
        self.assertTrue(np.all(sd <= 0.1))

    def test_random_parts_for_small_k(self):
        # K = 8: stratified would drop near_contact and exact_contact (D02.4) -> random draw
        c = cfg(n_orient=1250, k_per_orient=8)
        h = np.full(1250, 0.6)
        sd, part = sample_sd(c, self.s, h, np.random.default_rng(SEED))
        frac = {k: (part == v).mean() for k, v in SD_PART_CODES.items()}
        for k, f in DEFAULT_SD_FRACTIONS.items():
            self.assertAlmostEqual(frac[k], f, delta=4 * math.sqrt(f * (1 - f) / 10000) + 1e-3, msg=k)
        self.assertGreater(int((part == SD_PART_CODES["exact_contact"]).sum()), 0)

    def test_custom_sd_log_min(self):
        sd, part = sample_sd(cfg(n_orient=100, k_per_orient=100, sd_log_min=1e-8), self.s, self.h,
                             np.random.default_rng(1))
        pen = -sd[part == SD_PART_CODES["penetration"]]
        self.assertGreaterEqual(pen.min(), 1e-8)
        self.assertLess(pen.min(), 1e-6)

    def test_reproducible(self):
        a = sample_sd(self.c, self.s, self.h, np.random.default_rng(5))
        b = sample_sd(self.c, self.s, self.h, np.random.default_rng(5))
        self.assertTrue(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]))

    def test_h_length_must_match(self):
        with self.assertRaises(ValueError):
            sample_sd(self.c, self.s, self.h[:10], np.random.default_rng(1))


class TestBuildRows(unittest.TestCase):
    """Row assembly with recomputed labels (R02.1, R02.4, R02.6)."""

    @classmethod
    def setUpClass(cls):
        cls.c, cls.s, cls.q, cls.comp = _orients()
        h = (u_from_quats(cls.q) @ cls.s.vertices.T).max(axis=1)
        cls.sd, cls.part = sample_sd(cls.c, cls.s, h, np.random.default_rng(SEED))
        cls.rows = build_rows(cls.c, cls.s, cls.q, cls.comp, cls.sd, cls.part)

    def test_columns_order_and_dtypes(self):
        self.assertEqual(ROW_COLUMNS, ("position_z", "qw", "qx", "qy", "qz", "sd", "h",
                                       "orientation_id", "component", "sd_part", "delta"))
        self.assertEqual(tuple(self.rows), ROW_COLUMNS)
        for name in ("position_z", "qw", "qx", "qy", "qz", "sd", "h", "delta"):
            self.assertEqual(self.rows[name].dtype, np.float64, msg=name)
        self.assertEqual(self.rows["orientation_id"].dtype, np.int64)
        self.assertEqual(self.rows["component"].dtype, np.int8)
        self.assertEqual(self.rows["sd_part"].dtype, np.int8)
        for a in self.rows.values():
            self.assertEqual(a.shape, (self.c.n_rows,))

    def test_orientation_id_and_quaternion_blocks(self):
        oid = self.rows["orientation_id"]
        np.testing.assert_array_equal(oid, np.repeat(np.arange(self.c.n_orient), self.c.k_per_orient))
        qs = np.c_[self.rows["qw"], self.rows["qx"], self.rows["qy"], self.rows["qz"]]
        np.testing.assert_array_equal(qs, np.repeat(self.q, self.c.k_per_orient, axis=0))
        np.testing.assert_array_equal(self.rows["component"], np.repeat(self.comp, self.c.k_per_orient))

    def test_labels_match_reference(self):
        r = self.rows
        idx = np.random.default_rng(1).choice(self.c.n_rows, 2000, replace=False)
        for i in idx:
            q = [r["qw"][i], r["qx"][i], r["qy"][i], r["qz"][i]]
            ref = cw.compute_signed_distance(q, r["position_z"][i])
            self.assertLessEqual(abs(r["sd"][i] - ref), 1e-12)
            self.assertLessEqual(abs(r["h"][i] - cw.z_touch(q)), 1e-12)
        np.testing.assert_array_equal(r["sd"], r["position_z"] - r["h"])

    def test_labels_close_to_targets(self):
        np.testing.assert_allclose(self.rows["sd"], self.sd, rtol=0, atol=1e-15)
        self.assertTrue(np.all(self.rows["sd"][self.part == SD_PART_CODES["exact_contact"]] == 0.0))

    def test_ranges_and_prefilter(self):
        r = self.rows
        self.assertGreaterEqual(r["sd"].min(), -0.1 - 1e-15)
        self.assertLessEqual(r["position_z"].max(), cw.Z_PREFILTER)
        self.assertGreater(r["position_z"].min(), 0.0)

    def test_delta_column(self):
        qs = np.c_[self.rows["qw"], self.rows["qx"], self.rows["qy"], self.rows["qz"]]
        np.testing.assert_array_equal(self.rows["delta"], support_gap(self.s, u_from_quats(qs)))

    def test_all_parts_and_components_present(self):
        self.assertEqual(set(np.unique(self.rows["component"])), set(COMPONENT_CODES.values()))
        self.assertEqual(set(np.unique(self.rows["sd_part"])), set(SD_PART_CODES.values()))

    def test_non_finite_raises_generation_error(self):
        bad = self.sd.copy()
        bad[3] = np.nan
        with self.assertRaises(GenerationError):
            build_rows(self.c, self.s, self.q, self.comp, bad, self.part)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            build_rows(self.c, self.s, self.q[:-1], self.comp[:-1], self.sd, self.part)


def _smoke_rows(**kw):
    """Default smoke config (1250 x 8 = 10000 rows), assembled rows."""
    c = cfg(**kw)
    s = _cube_shape()
    rng = np.random.default_rng(c.seed)
    q, comp = sample_orientations(c, s, rng)
    h = (u_from_quats(q) @ s.vertices.T).max(axis=1)
    sd, part = sample_sd(c, s, h, rng)
    return c, s, build_rows(c, s, q, comp, sd, part)


def _copy(rows):
    return {k: v.copy() for k, v in rows.items()}


def _row(rows, part):
    """Index of the first row of a given sd part."""
    return int(np.flatnonzero(rows["sd_part"] == SD_PART_CODES[part])[0])


class TestCoverageTable(unittest.TestCase):
    """R02.5: rows per decade of |sd| (each sign) and of Delta, per component and per part."""

    @classmethod
    def setUpClass(cls):
        cls.c, cls.s, cls.rows = _smoke_rows()
        cls.t = coverage_table(cls.rows)

    def test_fixed_ranges_from_requirements(self):
        self.assertEqual(COVERAGE_SD_RANGE, (1e-5, 1e-1))
        self.assertEqual(COVERAGE_DELTA_RANGE, (1e-6, 1e-1))

    def test_structure(self):
        self.assertEqual(set(self.t), {"sd_negative", "sd_positive", "delta", "components", "sd_parts", "n_rows"})
        self.assertEqual(len(self.t["sd_negative"]), 4)
        self.assertEqual(len(self.t["sd_positive"]), 4)
        self.assertEqual(len(self.t["delta"]), 5)
        self.assertEqual([c["lo"] for c in self.t["sd_positive"]], [1e-5, 1e-4, 1e-3, 1e-2])
        self.assertEqual([c["hi"] for c in self.t["delta"]], [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])

    def test_counts_match_direct_count(self):
        sd, d = self.rows["sd"], self.rows["delta"]
        for cell in self.t["sd_negative"]:
            lo, hi = cell["lo"], cell["hi"]
            top = (-sd <= hi) if hi == 1e-1 else (-sd < hi)
            self.assertEqual(cell["count"], int(((-sd >= lo) & top & (sd < 0)).sum()))
        for cell in self.t["sd_positive"]:
            lo, hi = cell["lo"], cell["hi"]
            top = (sd <= hi) if hi == 1e-1 else (sd < hi)
            self.assertEqual(cell["count"], int(((sd >= lo) & top).sum()))
        for cell in self.t["delta"]:
            lo, hi = cell["lo"], cell["hi"]
            top = (d <= hi) if hi == 1e-1 else (d < hi)
            self.assertEqual(cell["count"], int(((d >= lo) & top).sum()))

    def test_decade_boundaries(self):
        # lower bound inclusive, upper exclusive, except the last decade which includes 0.1
        n = 8
        r = {k: v[:n].copy() for k, v in self.rows.items()}
        r["sd"][:] = [1e-5, 1e-4, 0.1, -0.1, -1e-5, 5e-6, 0.2, 0.0]
        r["delta"][:] = [1e-6, 1e-5, 0.1, 0.2, 5e-7, 0.0, 1e-6, 1e-6]
        t = coverage_table(r)
        self.assertEqual([c["count"] for c in t["sd_positive"]], [1, 1, 0, 1])      # 1e-5, 1e-4, 0.1
        self.assertEqual([c["count"] for c in t["sd_negative"]], [1, 0, 0, 1])      # -1e-5, -0.1
        self.assertEqual([c["count"] for c in t["delta"]], [3, 1, 0, 0, 1])         # 1e-6 x3, 1e-5, 0.1

    def test_components_and_parts_sum_to_n_rows(self):
        self.assertEqual(sum(self.t["components"].values()), self.c.n_rows)
        self.assertEqual(sum(self.t["sd_parts"].values()), self.c.n_rows)
        self.assertEqual(set(self.t["components"]), set(COMPONENT_CODES))
        self.assertEqual(set(self.t["sd_parts"]), set(SD_PART_CODES))
        self.assertEqual(self.t["n_rows"], self.c.n_rows)

    def test_every_cell_populated_at_smoke_size(self):
        for key in ("sd_negative", "sd_positive", "delta"):
            for cell in self.t[key]:
                self.assertGreater(cell["count"], 0, msg=f"{key} {cell}")

    def test_json_serializable(self):
        json.dumps(self.t)

    def test_does_not_modify_rows(self):
        before = _copy(self.rows)
        coverage_table(self.rows)
        for k in self.rows:
            np.testing.assert_array_equal(self.rows[k], before[k])


class TestAcceptanceChecks(unittest.TestCase):
    """R02.12: each check passes on generated rows and fails on a corrupted row."""

    @classmethod
    def setUpClass(cls):
        cls.c, cls.s, cls.rows = _smoke_rows()
        cls.res = acceptance_checks(cls.rows, cls.c, cls.s)

    def test_names(self):
        self.assertEqual(ACCEPTANCE_CHECKS, ("label", "h_consistent", "ranges", "prefilter", "finite",
                                             "unit_norm", "canonical", "coverage", "exclusion"))
        self.assertEqual(tuple(self.res), ACCEPTANCE_CHECKS)

    def test_all_pass_on_generated_rows(self):
        for name, r in self.res.items():
            self.assertTrue(r["passed"], msg=f"{name}: {r}")
            self.assertEqual(r["n_fail"], 0, msg=name)
        self.assertIsNone(require_acceptance(self.res))

    def test_result_fields_json_serializable(self):
        for r in self.res.values():
            self.assertEqual(set(r), {"passed", "n_fail", "worst", "tol"})
            self.assertIsInstance(r["passed"], bool)
            self.assertIsInstance(r["n_fail"], int)
        json.dumps(self.res)

    def test_label_worst_is_round_off(self):
        self.assertLessEqual(self.res["label"]["worst"], 1e-15)

    def corrupt(self, mutate, cfg_=None):
        r = _copy(self.rows)
        mutate(r)
        return acceptance_checks(r, cfg_ or self.c, self.s)

    def assertOnlyFails(self, res, names, n_fail=1):
        for name, r in res.items():
            if name in names:
                self.assertFalse(r["passed"], msg=name)
                if n_fail is not None:
                    self.assertEqual(r["n_fail"], n_fail, msg=name)
            else:
                self.assertTrue(r["passed"], msg=f"{name} should pass: {r}")

    def test_label_corrupted(self):
        def m(r):
            r["sd"][17] += 1e-10
        res = self.corrupt(m)
        self.assertOnlyFails(res, {"label"})
        self.assertAlmostEqual(res["label"]["worst"], 1e-10, delta=1e-15)

    def test_consistently_wrong_h_is_caught(self):
        # h and sd shifted together (position_z - h still equals sd): the check
        # must recompute h independently from the quaternion
        def m(r):
            r["h"][5] += 1e-9
            r["sd"][5] -= 1e-9
        self.assertOnlyFails(self.corrupt(m), {"label", "h_consistent"})

    def test_range_corrupted(self):
        def m(r):
            i = _row(r, "penetration")
            r["sd"][i] = -0.2
            r["position_z"][i] = r["h"][i] - 0.2
        self.assertOnlyFails(self.corrupt(m), {"ranges"})

    def test_part_range_corrupted(self):
        # a gap row with a value in the near-contact band violates its part range
        def m(r):
            i = _row(r, "gap")
            r["sd"][i] = 1e-7
            r["position_z"][i] = r["h"][i] + 1e-7
        self.assertOnlyFails(self.corrupt(m), {"ranges"})

    def test_exact_contact_must_be_zero(self):
        def m(r):
            i = _row(r, "exact_contact")
            r["sd"][i] = 1e-16
            r["position_z"][i] = r["h"][i] + 1e-16
        res = self.corrupt(m)
        self.assertFalse(res["ranges"]["passed"])

    def test_above_prefilter_corrupted(self):
        def m(r):
            i = _row(r, "gap_safety")
            r["position_z"][i] = cw.Z_PREFILTER + 1e-3
            r["sd"][i] = r["position_z"][i] - r["h"][i]
        res = self.corrupt(m)
        self.assertFalse(res["prefilter"]["passed"])
        self.assertEqual(res["prefilter"]["n_fail"], 1)
        self.assertTrue(res["label"]["passed"])

    def test_non_finite_corrupted(self):
        def m(r):
            r["delta"][3] = np.nan
        res = self.corrupt(m)
        self.assertFalse(res["finite"]["passed"])
        self.assertEqual(res["finite"]["n_fail"], 1)

    def test_unit_norm_corrupted(self):
        def m(r):
            for k in ("qw", "qx", "qy", "qz"):
                r[k][9] *= 1.0 + 1e-9
        res = self.corrupt(m)
        self.assertFalse(res["unit_norm"]["passed"])
        self.assertEqual(res["unit_norm"]["n_fail"], 1)
        self.assertTrue(res["label"]["passed"])   # same rotation

    def test_canonical_corrupted(self):
        def m(r):
            for k in ("qw", "qx", "qy", "qz"):
                r[k][11] = -r[k][11]               # -q: same rotation, not canonical
        self.assertOnlyFails(self.corrupt(m), {"canonical"})

    def test_coverage_below_m_min(self):
        res = acceptance_checks(self.rows, cfg(m_min=10**6), self.s)
        self.assertFalse(res["coverage"]["passed"])
        self.assertEqual(res["coverage"]["n_fail"], 13)       # 4 + 4 + 5 decade cells
        self.assertTrue(all(r["passed"] for n, r in res.items() if n != "coverage"))

    def test_exclusion_violated(self):
        i = 40
        q = np.array([[self.rows[k][i] for k in ("qw", "qx", "qy", "qz")]])
        res = acceptance_checks(self.rows, cfg(exclude_quats=q, exclude_angle_deg=1.0), self.s)
        self.assertFalse(res["exclusion"]["passed"])
        self.assertGreaterEqual(res["exclusion"]["n_fail"], self.c.k_per_orient)

    def test_does_not_modify_rows(self):
        before = _copy(self.rows)
        acceptance_checks(self.rows, self.c, self.s)
        for k in self.rows:
            np.testing.assert_array_equal(self.rows[k], before[k])

    def test_require_acceptance_raises_with_names(self):
        def m(r):
            r["sd"][17] += 1e-10
            for k in ("qw", "qx", "qy", "qz"):
                r[k][11] = -r[k][11]
        with self.assertRaises(GenerationError) as ctx:
            require_acceptance(self.corrupt(m))
        self.assertIn("label", str(ctx.exception))
        self.assertIn("canonical", str(ctx.exception))

    def test_malformed_rows(self):
        r = _copy(self.rows)
        del r["delta"]
        with self.assertRaises(ValueError):
            acceptance_checks(r, self.c, self.s)
        r = _copy(self.rows)
        r["sd"] = r["sd"][:-1]
        with self.assertRaises(ValueError):
            acceptance_checks(r, self.c, self.s)


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _small(**kw):
    """Small valid config for writer tests (500 rows)."""
    base = dict(n_orient=125, k_per_orient=4, seed=SEED)
    base.update(kw)
    return cfg(**base)


class _TmpRoot(unittest.TestCase):
    def setUp(self):
        self._d = tempfile.TemporaryDirectory()
        self.root = os.path.join(self._d.name, "data")

    def tearDown(self):
        self._d.cleanup()

    def datasets(self):
        if not os.path.isdir(self.root):
            return []
        return sorted(n for n in os.listdir(self.root) if n.startswith("dataset_"))


class TestSeeding(unittest.TestCase):
    """R02.9: one master seed, derived streams, bitwise reproducible arrays."""

    def test_stream_names(self):
        self.assertEqual(SEED_STREAMS, ("orientations", "sd"))

    def test_same_seed_identical_arrays(self):
        c = _small()
        r1, info1 = generate_rows(c, _cube_shape())
        r2, info2 = generate_rows(c, _cube_shape())
        for k in ROW_COLUMNS:
            self.assertTrue(np.array_equal(r1[k], r2[k]), msg=k)
        self.assertEqual(info1, info2)

    def test_different_seed_different_arrays(self):
        r1, _ = generate_rows(_small(seed=1), _cube_shape())
        r2, _ = generate_rows(_small(seed=2), _cube_shape())
        self.assertFalse(np.array_equal(r1["qw"], r2["qw"]))

    def test_seed_info_recorded(self):
        _, info = generate_rows(_small(seed=7), _cube_shape())
        self.assertEqual(info["master_seed"], 7)
        self.assertEqual(set(info["streams"]), set(SEED_STREAMS))
        keys = [tuple(v["spawn_key"]) for v in info["streams"].values()]
        self.assertEqual(len(set(keys)), len(keys))
        # the recorded entropy and spawn key reproduce the stream
        v = info["streams"]["sd"]
        a = np.random.default_rng(np.random.SeedSequence(v["entropy"], spawn_key=v["spawn_key"])).random(3)
        b = np.random.default_rng(np.random.SeedSequence(7).spawn(len(SEED_STREAMS))[SEED_STREAMS.index("sd")]).random(3)
        self.assertTrue(np.array_equal(a, b))

    def test_rows_pass_acceptance(self):
        c = _small()
        rows, _ = generate_rows(c, _cube_shape())
        self.assertIsNone(require_acceptance(acceptance_checks(rows, c, _cube_shape())))


class TestWriter(_TmpRoot):
    """R02.7, R02.8, R02.10: files, versioned folders, provenance."""

    def test_first_dataset_folder_and_files(self):
        path = generate(_small(), self.root)
        self.assertEqual(os.path.basename(path), "dataset_001")
        self.assertEqual(sorted(os.listdir(path)), ["data.npz", "provenance.json"])
        self.assertEqual(self.datasets(), ["dataset_001"])
        self.assertEqual([n for n in os.listdir(self.root) if n.startswith(".tmp")], [])

    def test_npz_content(self):
        c = _small()
        path = generate(c, self.root)
        rows, _ = generate_rows(c, _cube_shape())
        with np.load(os.path.join(path, "data.npz")) as z:
            self.assertEqual(tuple(z.files), ROW_COLUMNS)
            for k in ROW_COLUMNS:
                self.assertTrue(np.array_equal(z[k], rows[k]), msg=k)
                self.assertEqual(z[k].dtype, rows[k].dtype, msg=k)

    def test_never_overwrites_and_increments(self):
        p1 = generate(_small(), self.root)
        before = _sha(os.path.join(p1, "data.npz"))
        p2 = generate(_small(seed=1), self.root)
        p3 = generate(_small(seed=2), self.root)
        self.assertEqual([os.path.basename(p) for p in (p1, p2, p3)], ["dataset_001", "dataset_002", "dataset_003"])
        self.assertEqual(_sha(os.path.join(p1, "data.npz")), before)

    def test_counter_follows_existing_max(self):
        os.makedirs(os.path.join(self.root, "dataset_007"))
        open(os.path.join(self.root, "dataset_007", "keep.txt"), "w").close()
        os.makedirs(os.path.join(self.root, "notes"))
        p = generate(_small(), self.root)
        self.assertEqual(os.path.basename(p), "dataset_008")
        self.assertEqual(os.listdir(os.path.join(self.root, "dataset_007")), ["keep.txt"])

    def test_existing_target_is_never_replaced(self):
        # D02.6: an existing (even empty) target folder must not be replaced
        import generate_dataset as gd
        os.makedirs(os.path.join(self.root, "dataset_001"))
        marker = os.path.join(self.root, "dataset_001", "precious.txt")
        open(marker, "w").close()
        orig = gd._next_index
        gd._next_index = lambda root: 1
        try:
            with self.assertRaises(GenerationError):
                generate(_small(), self.root)
        finally:
            gd._next_index = orig
        self.assertEqual(os.listdir(os.path.join(self.root, "dataset_001")), ["precious.txt"])

    def test_existing_empty_target_is_never_replaced(self):
        import generate_dataset as gd
        os.makedirs(os.path.join(self.root, "dataset_001"))
        orig = gd._next_index
        gd._next_index = lambda root: 1
        try:
            with self.assertRaises(GenerationError):
                generate(_small(), self.root)
        finally:
            gd._next_index = orig
        self.assertEqual(os.listdir(os.path.join(self.root, "dataset_001")), [])

    def test_same_seed_identical_file_checksum(self):
        a = generate(_small(), self.root)
        b = generate(_small(), self.root)
        self.assertEqual(_sha(os.path.join(a, "data.npz")), _sha(os.path.join(b, "data.npz")))

    def test_provenance_fields(self):
        c = _small(exclude_quats=np.array([[1.0, 0, 0, 0]]), exclude_angle_deg=2.0)
        path = generate(c, self.root, csv=True)
        with open(os.path.join(path, "provenance.json")) as f:
            p = json.load(f)
        for key in ("spec", "generator", "git_commit", "git_dirty", "versions", "config", "seed",
                    "shape", "n_rows", "columns", "codes", "timestamp_utc", "sha256", "coverage", "acceptance"):
            self.assertIn(key, p)
        self.assertEqual(p["spec"], {"01": "rev 3", "02": "rev 2"})
        self.assertEqual(p["generator"], "generate_dataset.py")
        self.assertEqual(set(p["versions"]), {"python", "numpy", "scipy"})
        self.assertEqual(p["versions"]["numpy"], np.__version__)
        self.assertEqual(p["seed"]["master_seed"], SEED)
        self.assertEqual(set(p["seed"]["streams"]), set(SEED_STREAMS))
        self.assertEqual(p["n_rows"], 500)
        self.assertEqual(p["columns"], list(ROW_COLUMNS))
        self.assertEqual(p["codes"], {"component": COMPONENT_CODES, "sd_part": SD_PART_CODES})
        self.assertRegex(p["timestamp_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?Z$")
        self.assertEqual(p["sha256"]["data.npz"], _sha(os.path.join(path, "data.npz")))
        self.assertEqual(p["sha256"]["data.csv"], _sha(os.path.join(path, "data.csv")))
        self.assertTrue(all(r["passed"] for r in p["acceptance"].values()))
        self.assertEqual(p["coverage"]["n_rows"], 500)
        # config: every GenConfig field; exclusion list by checksum and count
        self.assertEqual(set(p["config"]), {f.name for f in dataclasses.fields(GenConfig)})
        ex = p["config"]["exclude_quats"]
        self.assertEqual(ex["count"], 1)
        self.assertEqual(ex["sha256"], hashlib.sha256(np.ascontiguousarray([[1.0, 0, 0, 0]], dtype=np.float64).tobytes()).hexdigest())
        self.assertEqual(p["config"]["exclude_angle_deg"], 2.0)
        self.assertEqual(p["config"]["sd_fractions"], DEFAULT_SD_FRACTIONS)
        # shape: points and constants
        self.assertEqual(p["shape"]["n_hull_vertices"], 8)
        self.assertEqual(p["shape"]["r_circ"], cw.R_CIRCUMSCRIBED)
        self.assertEqual(p["shape"]["h_min"], cw.H_MIN)
        self.assertEqual(np.array(p["shape"]["points"]).shape, (8, 3))

    def test_git_fields(self):
        path = generate(_small(), self.root)
        with open(os.path.join(path, "provenance.json")) as f:
            p = json.load(f)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
        if head.returncode == 0:
            self.assertEqual(p["git_commit"], head.stdout.strip())
            self.assertIsInstance(p["git_dirty"], bool)

    def test_csv_round_trip(self):
        c = _small()
        path = generate(c, self.root, csv=True)
        with np.load(os.path.join(path, "data.npz")) as z:
            arrays = {k: z[k] for k in z.files}
        with open(os.path.join(path, "data.csv"), newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            body = list(reader)
        self.assertEqual(header, list(ROW_COLUMNS))
        self.assertEqual(len(body), 500)
        inv_comp = {v: k for k, v in COMPONENT_CODES.items()}
        inv_part = {v: k for k, v in SD_PART_CODES.items()}
        for j, name in enumerate(ROW_COLUMNS):
            col = [row[j] for row in body]
            if name == "component":
                self.assertEqual(col, [inv_comp[int(v)] for v in arrays[name]])
            elif name == "sd_part":
                self.assertEqual(col, [inv_part[int(v)] for v in arrays[name]])
            elif name == "orientation_id":
                self.assertEqual([int(v) for v in col], arrays[name].tolist())
            else:
                # 17 significant digits: exact float64 round trip
                self.assertTrue(np.array_equal(np.array([float(v) for v in col]), arrays[name]), msg=name)

    def test_csv_limited_to_smoke_size(self):
        c = cfg(n_orient=1300, k_per_orient=8, allow_full_scale=True)
        with self.assertRaises(ConfigError):
            generate(c, self.root, csv=True)
        self.assertEqual(self.datasets(), [])

    def test_invalid_config_writes_nothing(self):
        with self.assertRaises(ConfigError):
            generate(_small(n_orient=0), self.root)
        self.assertFalse(os.path.exists(self.root) and os.listdir(self.root))

    def test_failed_acceptance_writes_nothing(self):
        c = _small()
        rows, info = generate_rows(c, _cube_shape())
        rows["sd"][0] += 1e-9
        with self.assertRaises(GenerationError):
            write_dataset(rows, c, _cube_shape(), self.root, seed_info=info)
        self.assertEqual(self.datasets(), [])

    def test_write_failure_leaves_no_complete_folder(self):
        import generate_dataset as gd
        orig = gd.np.savez

        def boom(*a, **k):
            raise OSError("disk full (simulated)")
        gd.np.savez = boom
        try:
            with self.assertRaises(GenerationError) as ctx:
                generate(_small(), self.root)
        finally:
            gd.np.savez = orig
        self.assertEqual(self.datasets(), [])
        tmp = [n for n in os.listdir(self.root) if n.startswith(".tmp_dataset_")]
        self.assertEqual(len(tmp), 1)
        self.assertIn(tmp[0], str(ctx.exception))      # temp folder named for inspection

    def test_provenance_written_last(self):
        # a folder without provenance.json is never published
        path = generate(_small(), self.root)
        self.assertTrue(os.path.isfile(os.path.join(path, "provenance.json")))


class TestGitignore(unittest.TestCase):
    """D02.2: generated dataset folders are ignored by git."""

    def test_dataset_folders_ignored(self):
        if shutil.which("git") is None:
            self.skipTest("git not available")
        for rel in ("interactions/ss_cube-wall/data/dataset_001/data.npz",
                    "interactions/ss_cube-wall/data/dataset_123/provenance.json",
                    "interactions/ss_cube-wall/data/.tmp_dataset_004_99/data.npz"):
            r = subprocess.run(["git", "check-ignore", "-q", rel], cwd=REPO_ROOT)
            self.assertEqual(r.returncode, 0, msg=rel)


import contextlib  # noqa: E402
import io  # noqa: E402

SCRIPT = os.path.join(REPO_ROOT, "interactions", "ss_cube-wall", "python", "generate_dataset.py")
PYTHON = sys.executable


def _run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


class TestCli(_TmpRoot):
    """T02.8: command line (design CLI), exit codes, summary (R02.14)."""

    def args(self, *extra):
        return ["--n-orient", "125", "--k", "4", "--seed", str(SEED), "--out", self.root, *extra]

    def test_writes_dataset_and_prints_summary(self):
        code, out, err = _run_main(self.args())
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(self.datasets(), ["dataset_001"])
        path = os.path.join(self.root, "dataset_001")
        for needle in ("SMOKE", "rows", "500", "uniform", "near_kink", "special",
                       "penetration", "gap_safety", "coverage", "acceptance", "label", "PASS", path):
            self.assertIn(needle, out, msg=needle)

    def test_csv_flag(self):
        code, _, err = _run_main(self.args("--csv"))
        self.assertEqual(code, 0, msg=err)
        self.assertEqual(sorted(os.listdir(os.path.join(self.root, "dataset_001"))),
                         ["data.csv", "data.npz", "provenance.json"])

    def test_same_cli_same_data(self):
        _run_main(self.args())
        _run_main(self.args())
        a, b = (os.path.join(self.root, d, "data.npz") for d in ("dataset_001", "dataset_002"))
        self.assertEqual(_sha(a), _sha(b))

    def test_missing_seed_is_an_error(self):
        code, out, err = _run_main(["--n-orient", "125", "--k", "4", "--out", self.root])
        self.assertNotEqual(code, 0)
        self.assertEqual(self.datasets(), [])

    def test_invalid_config_exit_code_and_message(self):
        code, out, err = _run_main(["--n-orient", "0", "--k", "4", "--seed", "1", "--out", self.root])
        self.assertEqual(code, 2)
        self.assertIn("n_orient", err)
        self.assertFalse(os.path.exists(self.root) and os.listdir(self.root))

    def test_full_scale_refused_without_flag(self):
        code, out, err = _run_main(["--n-orient", "1251", "--k", "8", "--seed", "1", "--out", self.root])
        self.assertEqual(code, 2)
        self.assertIn("allow_full_scale", err)
        self.assertEqual(self.datasets(), [])

    def test_exclude_file(self):
        excl = os.path.join(self._d.name, "excl.csv")
        np.savetxt(excl, np.array([[1.0, 0, 0, 0], [0.0, 1.0, 0, 0]]), delimiter=",")
        code, _, err = _run_main(self.args("--exclude", excl, "--exclude-angle", "5"))
        self.assertEqual(code, 0, msg=err)
        with open(os.path.join(self.root, "dataset_001", "provenance.json")) as f:
            p = json.load(f)
        self.assertEqual(p["config"]["exclude_quats"]["count"], 2)
        self.assertEqual(p["config"]["exclude_angle_deg"], 5.0)
        self.assertTrue(p["acceptance"]["exclusion"]["passed"])

    def test_bad_exclude_file(self):
        code, _, err = _run_main(self.args("--exclude", os.path.join(self._d.name, "missing.csv"),
                                           "--exclude-angle", "5"))
        self.assertEqual(code, 2)
        self.assertEqual(self.datasets(), [])

    def test_generation_error_exit_code(self):
        import generate_dataset as gd
        orig = gd.np.savez

        def boom(*a, **k):
            raise OSError("disk full (simulated)")
        gd.np.savez = boom
        try:
            code, out, err = _run_main(self.args())
        finally:
            gd.np.savez = orig
        self.assertEqual(code, 1)
        self.assertIn("disk full", err)
        self.assertEqual(self.datasets(), [])

    def test_default_out_is_interaction_data_folder(self):
        import generate_dataset as gd
        self.assertEqual(os.path.normpath(gd.DEFAULT_OUT_ROOT),
                         os.path.normpath(os.path.join(REPO_ROOT, "interactions", "ss_cube-wall", "data")))

    def test_script_runs_as_program(self):
        # the real entry point, in a subprocess
        r = subprocess.run([PYTHON, "-B", SCRIPT, *self.args()], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, msg=r.stderr)
        self.assertIn("dataset_001", r.stdout)
        r = subprocess.run([PYTHON, "-B", SCRIPT, "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        for opt in ("--n-orient", "--k", "--seed", "--csv", "--allow-full-scale", "--exclude", "--exclude-angle", "--out"):
            self.assertIn(opt, r.stdout)


class TestSummary(unittest.TestCase):
    """R02.14 summary content."""

    def test_summary_lists_everything(self):
        with tempfile.TemporaryDirectory() as d:
            c = _small()
            path = generate(c, d)
            with open(os.path.join(path, "provenance.json")) as f:
                p = json.load(f)
        text = format_summary(p, path)
        self.assertIn("SMOKE", text)
        self.assertIn(str(c.n_rows), text)
        for name in list(COMPONENT_CODES) + list(SD_PART_CODES) + list(ACCEPTANCE_CHECKS):
            self.assertIn(name, text, msg=name)
        self.assertIn(path, text)
        self.assertIn("not a scientific", text)


if __name__ == "__main__":
    unittest.main()
