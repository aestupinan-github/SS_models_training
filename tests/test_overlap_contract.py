import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactions", "ss_cube-wall", "python"))

from cube_wall import compute_signed_distance, canonicalize_quaternion, z_touch, CUBE_VERTICES


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


if __name__ == "__main__":
    unittest.main()