"""Speed-adaptive temporal filtering (One Euro filter, Casiez et al. 2012).

Low cutoff at low speed removes jitter from slow/subtle motion; the cutoff
rises with speed so fast gestures are not smeared.  Pure Python.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from .types import Vec3


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


@dataclass
class OneEuroParams:
    # Tuned for landmarks in metres at 15-60 fps: halves static jitter while a
    # 2.5 Hz / 15 cm wave keeps ~98 % amplitude with ~20 ms lag.
    min_cutoff: float = 1.0   # Hz
    beta: float = 5.0         # cutoff gain per (m/s) of speed
    d_cutoff: float = 3.0     # Hz, derivative smoothing
    reset_gap: float = 0.5    # seconds without samples -> restart filter

    @property
    def enabled(self) -> bool:
        return self.min_cutoff > 0.0


class OneEuroFilter:
    """Scalar One Euro filter."""

    def __init__(self, params: Optional[OneEuroParams] = None):
        self.p = params or OneEuroParams()
        self._x: Optional[float] = None
        self._dx = 0.0
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._x = None
        self._dx = 0.0
        self._t = None

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or self._t is None or t - self._t > self.p.reset_gap:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        if t <= self._t:
            return self._x
        dt = t - self._t
        dx = (x - self._x) / dt
        a_d = _alpha(self.p.d_cutoff, dt)
        self._dx = self._dx + a_d * (dx - self._dx)
        cutoff = self.p.min_cutoff + self.p.beta * abs(self._dx)
        a = _alpha(cutoff, dt)
        self._x = self._x + a * (x - self._x)
        self._t = t
        return self._x


class OneEuroVec3:
    """Isotropic One Euro filter for 3D points (speed = |velocity|)."""

    def __init__(self, params: Optional[OneEuroParams] = None):
        self.p = params or OneEuroParams()
        self._x: Optional[Vec3] = None
        self._dx = Vec3()
        self._t: Optional[float] = None

    def reset(self) -> None:
        self._x = None
        self._dx = Vec3()
        self._t = None

    @property
    def value(self) -> Optional[Vec3]:
        return self._x

    def __call__(self, x: Vec3, t: float) -> Vec3:
        if self._x is None or self._t is None or t - self._t > self.p.reset_gap:
            self._x, self._dx, self._t = x.copy(), Vec3(), t
            return x.copy()
        if t <= self._t:
            return self._x.copy()
        dt = t - self._t
        v = (x - self._x) / dt
        a_d = _alpha(self.p.d_cutoff, dt)
        self._dx = self._dx + (v - self._dx) * a_d
        cutoff = self.p.min_cutoff + self.p.beta * self._dx.length()
        a = _alpha(cutoff, dt)
        self._x = self._x + (x - self._x) * a
        self._t = t
        return self._x.copy()


class PointSetFilter:
    """Keeps one OneEuroVec3 per named point."""

    def __init__(self, params: Optional[OneEuroParams] = None):
        self.params = params or OneEuroParams()
        self._filters: Dict[str, OneEuroVec3] = {}

    def reset(self) -> None:
        self._filters.clear()

    def filter(self, name: str, x: Vec3, t: float) -> Vec3:
        if not self.params.enabled:
            return x.copy()
        f = self._filters.get(name)
        if f is None:
            f = OneEuroVec3(self.params)
            self._filters[name] = f
        return f(x, t)
