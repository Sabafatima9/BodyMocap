"""Length-proportional rotation distribution (theta_i = theta * L_i / sum L)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bodymocap.core.math3d import (
    X_AXIS,
    Z_AXIS,
    cumulative_quat_product,
    euler_xyz_to_quat,
    quat_angle_deg,
    quat_from_axis_angle,
    quat_identity,
)
from bodymocap.core.types import Vec3
from bodymocap.retarget.proportional import (
    chain_parameter_breaks,
    chain_product,
    cumulative_rotations,
    distribute_rotation,
    remap_chain,
)


def _rx(deg):
    return euler_xyz_to_quat(math.radians(deg), 0.0, 0.0)


class TestProportional(unittest.TestCase):
    def test_breaks(self):
        self.assertEqual(chain_parameter_breaks([1, 1, 2]), [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(chain_parameter_breaks([]), [0.0, 1.0])

    def test_increments_proportional_to_length(self):
        total = quat_from_axis_angle(Vec3(0.2, 1.0, 0.3), math.radians(60.0))
        lengths = [0.1, 0.3, 0.2]
        inc = distribute_rotation(total, lengths)
        angles = [quat_angle_deg(q, quat_identity()) for q in inc]
        for a, L in zip(angles, lengths):
            self.assertAlmostEqual(a, 60.0 * L / sum(lengths), places=6)
        self.assertLess(quat_angle_deg(chain_product(inc), total), 1e-6)

    def test_cumulative(self):
        total = quat_from_axis_angle(Z_AXIS, math.radians(90.0))
        cum = cumulative_rotations(total, [1.0, 1.0, 2.0])
        self.assertAlmostEqual(quat_angle_deg(cum[0], quat_identity()), 22.5, places=6)
        self.assertAlmostEqual(quat_angle_deg(cum[1], quat_identity()), 45.0, places=6)
        self.assertLess(quat_angle_deg(cum[2], total), 1e-6)

    def test_remap_2_to_4_preserves_chain_rotation(self):
        src = [_rx(20), _rx(40)]
        out = remap_chain(src, [2, 2], [1, 1, 1, 1])
        self.assertEqual(len(out), 4)
        self.assertLess(quat_angle_deg(cumulative_quat_product(out), cumulative_quat_product(src)), 1e-6)
        # equal lengths -> equal increments
        a = [quat_angle_deg(q, quat_identity()) for q in out]
        self.assertAlmostEqual(max(a), min(a), places=6)

    def test_remap_4_to_2(self):
        src = [_rx(10), _rx(10), _rx(15), _rx(15)]
        out = remap_chain(src, [1, 1, 1, 1], [3, 1])
        self.assertLess(quat_angle_deg(cumulative_quat_product(out), cumulative_quat_product(src)), 1e-6)
        self.assertAlmostEqual(quat_angle_deg(out[0], quat_identity()), 37.5, places=6)

    def test_equal_topology_passthrough(self):
        src = [_rx(12), _rx(8), _rx(4)]
        out = remap_chain(src, [1, 1, 1], [1, 1, 1])
        for a, b in zip(src, out):
            self.assertLess(quat_angle_deg(a, b), 1e-9)


if __name__ == "__main__":
    unittest.main()
