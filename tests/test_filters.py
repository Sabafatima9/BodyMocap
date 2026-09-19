"""One Euro filter behaviour: jitter suppression at low speed, low lag at high speed."""

from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bodymocap.core.filters import OneEuroFilter, OneEuroParams, OneEuroVec3, PointSetFilter
from bodymocap.core.types import Vec3


def _rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs))


class TestOneEuro(unittest.TestCase):
    def test_static_jitter_reduced(self):
        rng = random.Random(0)
        f = OneEuroFilter(OneEuroParams())
        raw, out = [], []
        for i in range(300):
            t = i / 30.0
            x = rng.gauss(0.0, 0.01)
            raw.append(x)
            out.append(f(x, t))
        self.assertLess(_rms(out[30:]), 0.6 * _rms(raw[30:]))

    def test_fast_motion_low_lag(self):
        # 2 Hz, 0.2 m amplitude gesture: amplitude should be mostly preserved
        f = OneEuroVec3(OneEuroParams())
        peak = 0.0
        for i in range(300):
            t = i / 30.0
            x = 0.2 * math.sin(2 * math.pi * 2.0 * t)
            y = f(Vec3(x, 0, 0), t)
            if t > 2.0:
                peak = max(peak, abs(y.x))
        self.assertGreater(peak, 0.8 * 0.2)

    def test_reset_on_gap(self):
        f = OneEuroVec3(OneEuroParams(min_cutoff=0.5, beta=0.0, reset_gap=0.5))
        f(Vec3(0, 0, 0), 0.0)
        f(Vec3(0, 0, 0), 0.033)
        y = f(Vec3(1, 0, 0), 2.0)  # long gap -> restart at the new value
        self.assertAlmostEqual(y.x, 1.0)

    def test_non_increasing_time_returns_state(self):
        f = OneEuroFilter()
        f(1.0, 1.0)
        self.assertEqual(f(5.0, 1.0), 1.0)

    def test_disabled_passthrough(self):
        pf = PointSetFilter(OneEuroParams(min_cutoff=0.0))
        v = pf.filter("a", Vec3(1, 2, 3), 0.0)
        self.assertEqual(v.as_tuple(), (1.0, 2.0, 3.0))


if __name__ == "__main__":
    unittest.main()
