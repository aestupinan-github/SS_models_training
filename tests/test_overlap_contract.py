import unittest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "interactions", "ss_cube-wall", "python"))

from cube_wall import compute_overlap, canonicalize_quaternion, z_touch, CUBE_VERTICES


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


class TestOverlapComputation(unittest.TestCase):
    def test_flat_face_contact(self):
        q = [1.0, 0.0, 0.0, 0.0]
        z = 0.5
        overlap = compute_overlap(q, z)
        self.assertAlmostEqual(overlap, 0.0)

    def test_flat_face_penetration(self):
        q = [1.0, 0.0, 0.0, 0.0]
        z = 0.3
        overlap = compute_overlap(q, z)
        self.assertAlmostEqual(overlap, 0.2)

    def test_no_contact(self):
        q = [1.0, 0.0, 0.0, 0.0]
        z = 1.0
        overlap = compute_overlap(q, z)
        self.assertAlmostEqual(overlap, 0.0)

    def test_corner_contact(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q = [qw, qx, qy, qz]
        zt = z_touch(q)
        overlap = compute_overlap(q, zt)
        self.assertAlmostEqual(overlap, 0.0, places=5)

    def test_corner_penetration(self):
        import math
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q = [qw, qx, qy, qz]
        zt = z_touch(q)
        overlap = compute_overlap(q, zt - 0.1)
        self.assertAlmostEqual(overlap, 0.1, places=5)

    def test_overlap_never_negative(self):
        q = [1.0, 0.0, 0.0, 0.0]
        for z in [0.0, 0.1, 0.5, 1.0, 2.0]:
            overlap = compute_overlap(q, z)
            self.assertGreaterEqual(overlap, 0.0)


class TestZTouch(unittest.TestCase):
    def test_flat_face_z_touch(self):
        q = [1.0, 0.0, 0.0, 0.0]
        zt = z_touch(q)
        self.assertAlmostEqual(zt, 0.5)

    def test_z_touch_is_orientation_dependent(self):
        import math
        q_flat = [1.0, 0.0, 0.0, 0.0]
        angle = math.radians(54.7356)
        axis = [1.0 / math.sqrt(3.0)] * 3
        qw = math.cos(angle / 2.0)
        qx, qy, qz = [math.sin(angle / 2.0) * a for a in axis]
        q_corner = [qw, qx, qy, qz]
        self.assertNotAlmostEqual(z_touch(q_flat), z_touch(q_corner))


if __name__ == "__main__":
    unittest.main()
