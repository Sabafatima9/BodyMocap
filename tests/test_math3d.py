"""Unit tests for core math3d (pure Python)."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bodymocap.core.math3d import (
    X_AXIS,
    Y_AXIS,
    Z_AXIS,
    basis_to_quat,
    cumulative_weights,
    delta_between_frames,
    euler_xyz_to_quat,
    fabrik,
    frame_aim_up,
    frame_left_up,
    polyline_point_at,
    quat_angle_deg,
    quat_average,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_two_vectors,
    quat_identity,
    quat_normalize,
    quat_pow,
    quat_slerp,
    quat_swing_twist,
    quat_to_basis,
    quat_twist_angle,
    signed_angle_about,
    vec_roll_to_quat,
    wrap_angle,
)
from bodymocap.core.types import Quat, Vec3


def rand_vec(rng):
    return Vec3(rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1))


def rand_quat(rng):
    return quat_normalize(Quat(rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)))


class TestQuaternions(unittest.TestCase):
    def test_identity_mul(self):
        q = quat_from_axis_angle(Z_AXIS, 0.7)
        r = quat_identity() @ q
        self.assertLess(quat_angle_deg(r, q), 1e-9)

    def test_rotate_matches_axis_angle(self):
        q = quat_from_axis_angle(Z_AXIS, math.pi / 2)
        v = q.rotate(X_AXIS)
        self.assertAlmostEqual(v.x, 0.0, places=9)
        self.assertAlmostEqual(v.y, 1.0, places=9)

    def test_from_two_vectors(self):
        rng = random.Random(1)
        for _ in range(500):
            a, b = rand_vec(rng), rand_vec(rng)
            q = quat_from_two_vectors(a, b)
            self.assertLess((q.rotate(a.normalized()) - b.normalized()).length(), 1e-9)
        q = quat_from_two_vectors(X_AXIS, -X_AXIS)  # antiparallel
        self.assertLess((q.rotate(X_AXIS) + X_AXIS).length(), 1e-9)

    def test_slerp_endpoints_and_midpoint(self):
        a = quat_identity()
        b = quat_from_axis_angle(Y_AXIS, 1.2)
        self.assertLess(quat_angle_deg(quat_slerp(a, b, 0.0), a), 1e-6)
        self.assertLess(quat_angle_deg(quat_slerp(a, b, 1.0), b), 1e-6)
        m = quat_slerp(a, b, 0.5)
        self.assertAlmostEqual(quat_angle_deg(a, m), math.degrees(0.6), places=6)

    def test_pow_scales_angle(self):
        q = quat_from_axis_angle(Vec3(1, 2, 3), 1.5)
        h = quat_pow(q, 1.0 / 3.0)
        self.assertLess(quat_angle_deg(h @ h @ h, q), 1e-6)

    def test_swing_twist_roundtrip(self):
        rng = random.Random(2)
        for _ in range(300):
            q = rand_quat(rng)
            axis = rand_vec(rng).normalized()
            swing, twist = quat_swing_twist(q, axis)
            self.assertLess(quat_angle_deg(swing @ twist, q), 1e-4)
            # swing leaves the axis orthogonal component, twist is about axis
            tw_axis = Vec3(twist.x, twist.y, twist.z)
            if tw_axis.length() > 1e-6:
                self.assertAlmostEqual(abs(tw_axis.normalized().dot(axis)), 1.0, places=6)

    def test_twist_angle(self):
        q = quat_from_axis_angle(Y_AXIS, 0.8) @ quat_from_axis_angle(X_AXIS, 0.3)
        self.assertAlmostEqual(quat_twist_angle(quat_from_axis_angle(Y_AXIS, 0.8), Y_AXIS), 0.8, places=9)
        self.assertLess(abs(quat_twist_angle(quat_from_axis_angle(X_AXIS, 0.4), Y_AXIS)), 1e-9)
        self.assertIsInstance(quat_twist_angle(q, Y_AXIS), float)

    def test_average(self):
        a = quat_from_axis_angle(Z_AXIS, 0.2)
        b = quat_from_axis_angle(Z_AXIS, 0.4)
        self.assertLess(quat_angle_deg(quat_average([a, b]), quat_from_axis_angle(Z_AXIS, 0.3)), 1e-6)

    def test_euler_xyz_order(self):
        # Blender XYZ euler: X applied first, then Y, then Z
        q = euler_xyz_to_quat(0.3, 0.2, 0.1)
        ref = quat_from_axis_angle(Z_AXIS, 0.1) @ quat_from_axis_angle(Y_AXIS, 0.2) @ \
            quat_from_axis_angle(X_AXIS, 0.3)
        self.assertLess(quat_angle_deg(q, ref), 1e-9)


class TestFrames(unittest.TestCase):
    def test_basis_roundtrip(self):
        rng = random.Random(3)
        for _ in range(500):
            q = rand_quat(rng)
            bx, by, bz = quat_to_basis(q)
            self.assertLess(quat_angle_deg(basis_to_quat(bx, by, bz), q), 1e-4)

    def test_frame_aim_up(self):
        rng = random.Random(4)
        for _ in range(500):
            aim, up = rand_vec(rng), rand_vec(rng)
            f = frame_aim_up(aim, up)
            x, y, z = quat_to_basis(f)
            self.assertLess((y - aim.normalized()).length(), 1e-9)
            self.assertGreaterEqual(z.dot(up), -1e-9)
            self.assertLess(abs(x.dot(y)) + abs(y.dot(z)) + abs(x.dot(z)), 1e-9)

    def test_frame_left_up_canonical(self):
        f = frame_left_up(X_AXIS, Z_AXIS)
        self.assertLess(quat_angle_deg(f, quat_identity()), 1e-9)

    def test_delta_between_frames(self):
        rng = random.Random(5)
        a, b = rand_quat(rng), rand_quat(rng)
        d = delta_between_frames(b, a)
        self.assertLess(quat_angle_deg(d @ a, b), 1e-6)

    def test_signed_angle_and_wrap(self):
        self.assertAlmostEqual(signed_angle_about(X_AXIS, Y_AXIS, Z_AXIS), math.pi / 2, places=9)
        self.assertAlmostEqual(signed_angle_about(Y_AXIS, X_AXIS, Z_AXIS), -math.pi / 2, places=9)
        self.assertAlmostEqual(wrap_angle(3 * math.pi), math.pi, places=9)
        self.assertAlmostEqual(wrap_angle(-0.5), -0.5, places=12)

    def test_vec_roll_to_quat_axes(self):
        # Bone pointing +X with roll 0: Blender's rest matrix is a -90deg Z rotation
        q = vec_roll_to_quat(Vec3(1, 0, 0), 0.0)
        x, y, z = quat_to_basis(q)
        self.assertLess((y - X_AXIS).length(), 1e-9)
        self.assertLess((z - Z_AXIS).length(), 1e-9)
        # roll rotates about the bone axis
        q2 = vec_roll_to_quat(Vec3(1, 0, 0), math.pi / 2)
        _, y2, z2 = quat_to_basis(q2)
        self.assertLess((y2 - X_AXIS).length(), 1e-9)
        self.assertLess((z2 - (-Y_AXIS)).length(), 1e-9)
        # pointing straight down (-Y singularity handled)
        q3 = vec_roll_to_quat(Vec3(0, -1, 0), 0.0)
        _, y3, _ = quat_to_basis(q3)
        self.assertLess((y3 - (-Y_AXIS)).length(), 1e-9)


class TestChains(unittest.TestCase):
    def test_cumulative_weights(self):
        c = cumulative_weights([1.0, 1.0, 2.0])
        self.assertEqual(len(c), 3)
        self.assertAlmostEqual(c[0], 0.25)
        self.assertAlmostEqual(c[1], 0.5)
        self.assertEqual(c[2], 1.0)

    def test_polyline_point_at(self):
        pts = [Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, 1, 0)]
        p = polyline_point_at(pts, 0.75)
        self.assertLess((p - Vec3(1, 0.5, 0)).length(), 1e-12)

    def test_fabrik_enforces_lengths_even_if_end_on_target(self):
        # regression: an initial guess whose end sits on the target must still
        # be relaxed until every segment has its rest length
        base, target = Vec3(0, 0, 0), Vec3(0, -0.3, -0.35)
        joints = [base, Vec3(0, -0.29, -0.02), Vec3(0, -0.37, -0.16), target.copy()]
        lengths = [0.29, 0.29, 0.29]
        err = fabrik(joints, lengths, target)
        self.assertLess(err, 1e-5)
        for i, L in enumerate(lengths):
            self.assertAlmostEqual((joints[i + 1] - joints[i]).length(), L, places=6)
        self.assertLess((joints[0] - base).length(), 1e-12)

    def test_fabrik_unreachable_stretches_straight(self):
        joints = [Vec3(0, 0, 0), Vec3(0, 0, 1), Vec3(0, 0, 2)]
        err = fabrik(joints, [1.0, 1.0], Vec3(5, 0, 0))
        self.assertAlmostEqual(err, 3.0, places=9)
        self.assertLess((joints[2] - Vec3(2, 0, 0)).length(), 1e-9)


if __name__ == "__main__":
    unittest.main()
