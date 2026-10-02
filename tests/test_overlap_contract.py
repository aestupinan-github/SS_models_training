import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactions", "ss_cube-wall", "python"))

import itertools
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from cube_wall import (compute_signed_distance, canonicalize_quaternion, z_touch, CUBE_VERTICES,
                       to_canonical, from_canonical, classify_sd, is_far)

SEED = 20261003


def _wxyz(rot):
    x, y, z, w = rot.as_quat()
    return [w, x, y, z]


class TestQuaternionCanonicalization(unittest.TestCase):
    def test_positive_qw_unchanged(self):
        q = [1.0, 0.0, 0.0, 0.0]
        result = canonicalize_quaternion(q)
        self.assertEqual(result, [1.0, 0.0, 0.0, 0.0])

    def test_negative_qw_flipped(self):
        q = [-1.0, 0.0, 0.0, 0.0]
        result = canonicalize_quaternion(q)
        self.assertEqual(result, [1.0, 0.0, 0.0, 0.0])

    def test_negative_qw_nonzero_vector(self):
        q = [-0.5, 0.5, 0.5, 0.5]
        result = canonicalize_quaternion(q)
        self.assertAlmostEqual(result[0], 0.5)
        self.assertAlmostEqual(result[1], -0.5)
        self.assertAlmostEqual(result[2], -0.5)
        self.assertAlmostEqual(result[3], -0.5)


class TestCubeVertices(unittest.TestCase):
    def test_eight_vertices(self):
        self.assertEqual(len(CUBE_VERTICES), 8)

    def test_half_size(self):
        for v in CUBE_VERTICES:
            for coord in v:
                self.assertAlmostEqual(abs(coord), 0.5)


class TestSignedDistance(unittest.TestCase):
    def test_flat_face_contact_is_zero(self):
        q = [1.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_signed_distance(q, 0.5), 0.0)

    def test_flat_face_penetration_is_negative(self):
        q = [1.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_signed_distance(q, 0.3), -0.2)

    def test_no_contact_is_positive(self):
        q = [1.0, 0.0, 0.0, 0.0]
        self.assertAlmostEqual(compute_signed_distance(q, 1.0), 0.5)

    def test_corner_contact_is_zero(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q = [qw, qx, qy, qz]
        self.assertAlmostEqual(compute_signed_distance(q, z_touch(q)), 0.0, places=5)

    def test_corner_penetration_is_negative(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q = [qw, qx, qy, qz]
        self.assertAlmostEqual(compute_signed_distance(q, z_touch(q) - 0.1), -0.1, places=5)

    def test_corner_gap_is_positive(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q = [qw, qx, qy, qz]
        self.assertAlmostEqual(compute_signed_distance(q, z_touch(q) + 0.1), 0.1, places=5)


class TestSlopeIsUnity(unittest.TestCase):
    def test_slope_with_z_is_plus_one(self):
        q = [1.0, 0.0, 0.0, 0.0]
        dz = 1e-7
        a = compute_signed_distance(q, 0.3)
        b = compute_signed_distance(q, 0.3 + dz)
        self.assertAlmostEqual((b - a) / dz, 1.0, places=6)

    def test_slope_with_z_is_plus_one_corner(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q = [qw, qx, qy, qz]
        zt = z_touch(q)
        dz = 1e-7
        a = compute_signed_distance(q, zt - 0.05)
        b = compute_signed_distance(q, zt - 0.05 + dz)
        self.assertAlmostEqual((b - a) / dz, 1.0, places=6)


class TestZTouch(unittest.TestCase):
    def test_flat_face_z_touch(self):
        self.assertAlmostEqual(z_touch([1.0, 0.0, 0.0, 0.0]), 0.5)

    def test_z_touch_is_orientation_dependent(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        self.assertNotAlmostEqual(z_touch([1.0, 0.0, 0.0, 0.0]), z_touch([qw, qx, qy, qz]))

    def test_z_touch_equals_negative_signed_distance_at_zero(self):
        q = [0.5, 0.5, 0.5, 0.5]
        self.assertAlmostEqual(z_touch(q), -compute_signed_distance(q, 0.0))


class TestQuaternionRules(unittest.TestCase):
    """R01.4"""

    def test_qw_zero_pair_maps_to_one_output(self):
        self.assertEqual(canonicalize_quaternion([0.0, 1.0, 0.0, 0.0]),
                         canonicalize_quaternion([0.0, -1.0, 0.0, 0.0]))
        self.assertEqual(canonicalize_quaternion([0.0, 0.0, -1.0, 0.0]),
                         [0.0, 0.0, 1.0, 0.0])

    def test_qw_zero_general_pair(self):
        a = canonicalize_quaternion([0.0, 0.0, 0.6, -0.8])
        b = canonicalize_quaternion([0.0, 0.0, -0.6, 0.8])
        self.assertEqual(a, b)
        self.assertGreater(a[2], 0.0)

    def test_non_unit_is_normalized(self):
        q = canonicalize_quaternion([2.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(q), 1.0)
        self.assertEqual(q, [1.0, 0.0, 0.0, 0.0])

    def test_zero_norm_rejected(self):
        with self.assertRaises(ValueError):
            canonicalize_quaternion([0.0, 0.0, 0.0, 0.0])

    def test_non_finite_rejected(self):
        for bad in ([np.nan, 0, 0, 1], [1, np.inf, 0, 0], [1, 0, -np.inf, 0]):
            with self.assertRaises(ValueError):
                canonicalize_quaternion(bad)

    def test_canonical_is_idempotent_and_sign_invariant(self):
        for r in R.random(200, random_state=SEED):
            q = np.array(_wxyz(r))
            c1 = canonicalize_quaternion(q)
            self.assertEqual(c1, canonicalize_quaternion(-q))
            # re-normalization may change the last bit; the orientation is unchanged
            np.testing.assert_allclose(c1, canonicalize_quaternion(c1), rtol=0, atol=1e-15)
            self.assertGreaterEqual(c1[0], 0.0)


class TestRescale(unittest.TestCase):
    """R01.7"""

    def test_identity_for_canonical_size(self):
        self.assertEqual(to_canonical(0.8, 1.0), 0.8)
        self.assertEqual(from_canonical(-0.03, 1.0), -0.03)

    def test_scales_with_edge_length(self):
        self.assertAlmostEqual(to_canonical(0.8, 2.0), 0.4)
        self.assertAlmostEqual(from_canonical(-0.03, 2.0), -0.06)

    def test_round_trip_matches_direct_solution(self):
        # sd of a cube with edge L at height z equals L * sd_canonical(z / L)
        for r in R.random(100, random_state=SEED):
            q = _wxyz(r)
            for L in (0.25, 1.0, 3.0):
                z = 0.9 * L
                direct = L * compute_signed_distance(q, z / L)
                self.assertAlmostEqual(from_canonical(compute_signed_distance(q, to_canonical(z, L)), L), direct)

    def test_invalid_size_rejected(self):
        for bad in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                to_canonical(0.5, bad)
            with self.assertRaises(ValueError):
                from_canonical(0.1, bad)


class TestRangeClassification(unittest.TestCase):
    """R01.13"""

    def test_boundaries(self):
        self.assertEqual(classify_sd(0.1000001), "far")
        self.assertEqual(classify_sd(0.1), "in_range")
        self.assertEqual(classify_sd(0.0), "in_range")
        self.assertEqual(classify_sd(-0.1), "in_range")
        self.assertEqual(classify_sd(-0.1000001), "too_deep")

    def test_prefilter_never_contradicts_exact_solution(self):
        rng = np.random.default_rng(SEED)
        for r in R.random(2000, random_state=SEED):
            z = rng.uniform(0.0, 1.5)
            if is_far(z):
                self.assertEqual(classify_sd(compute_signed_distance(_wxyz(r), z)), "far")

    def test_prefilter_threshold(self):
        self.assertTrue(is_far(math.sqrt(3) / 2 + 0.1 + 1e-9))
        self.assertFalse(is_far(math.sqrt(3) / 2 + 0.1 - 1e-9))


class TestInvariances(unittest.TestCase):
    """R01.14, random poses with fixed seed"""

    @classmethod
    def setUpClass(cls):
        cls.rots = R.random(300, random_state=SEED)
        cls.sym = []
        for p in itertools.permutations(range(3)):
            for sg in itertools.product([1, -1], repeat=3):
                m = np.zeros((3, 3))
                for i in range(3):
                    m[i, p[i]] = sg[i]
                if np.isclose(np.linalg.det(m), 1.0):
                    cls.sym.append(R.from_matrix(m))

    def test_24_symmetries_found(self):
        self.assertEqual(len(self.sym), 24)

    def test_symmetry_invariance(self):
        for r in self.rots:
            g0 = z_touch(_wxyz(r))
            for s in self.sym:
                self.assertAlmostEqual(z_touch(_wxyz(r * s)), g0, places=12)

    def test_yaw_invariance(self):
        for r in self.rots:
            g0 = z_touch(_wxyz(r))
            for a in (0.3, 1.7, 4.0):
                self.assertAlmostEqual(z_touch(_wxyz(R.from_euler("z", a) * r)), g0, places=12)

    def test_quaternion_sign_invariance(self):
        for r in self.rots:
            q = np.array(_wxyz(r))
            self.assertEqual(z_touch(q), z_touch(-q))
            self.assertAlmostEqual(compute_signed_distance(q, 0.7), compute_signed_distance(-q, 0.7), places=12)

    def test_slope_on_random_poses(self):
        for r in self.rots:
            q = _wxyz(r)
            self.assertAlmostEqual(compute_signed_distance(q, 1.0) - compute_signed_distance(q, 0.0), 1.0, places=12)

    def test_g_range(self):
        for r in self.rots:
            g = z_touch(_wxyz(r))
            self.assertGreaterEqual(g, 0.5 - 1e-12)
            self.assertLessEqual(g, math.sqrt(3) / 2 + 1e-12)

    def test_sd_equals_z_minus_g(self):
        for r in self.rots:
            q = _wxyz(r)
            self.assertAlmostEqual(compute_signed_distance(q, 0.77), 0.77 - z_touch(q), places=12)


if __name__ == "__main__":
    unittest.main()