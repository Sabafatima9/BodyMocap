"""Metric root translation from world + image landmarks (pure Python).

MediaPipe returns hip-centred metric *world* landmarks and normalised *image*
landmarks.  Given the tracking camera's field of view, the camera-space hip
position T solves, for every visible landmark i (MediaPipe camera axes: x
right, y down, z forward, normalised image coords u, v):

    u_i = cx + fx (X_i + Tx) / (Z_i + Tz),   v_i = cy + fy (Y_i + Ty) / (Z_i + Tz)

which is linear in T:  fx Tx - (u_i - cx) Tz = (u_i - cx) Z_i - fx X_i  (same
for y).  A visibility-weighted least-squares solve gives T; the result is
returned in capture axes (x right, +Y away from camera, +Z up).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

from .landmarks import capture_to_mp_world
from .types import Landmark, Vec3


def intrinsics(hfov: float, width: int, height: int) -> Tuple[float, float, float, float]:
    """Normalised pinhole intrinsics (fx, fy, cx, cy) for image coords in 0..1."""
    fx = 0.5 / math.tan(hfov * 0.5)
    fy = fx * width / max(height, 1)
    return fx, fy, 0.5, 0.5


def _solve3(a, b) -> Optional[Tuple[float, float, float]]:
    (a00, a01, a02), (a10, a11, a12), (a20, a21, a22) = a
    det = (a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20)
           + a02 * (a10 * a21 - a11 * a20))
    if abs(det) < 1e-12:
        return None
    inv = (
        ((a11 * a22 - a12 * a21) / det, (a02 * a21 - a01 * a22) / det, (a01 * a12 - a02 * a11) / det),
        ((a12 * a20 - a10 * a22) / det, (a00 * a22 - a02 * a20) / det, (a02 * a10 - a00 * a12) / det),
        ((a10 * a21 - a11 * a20) / det, (a01 * a20 - a00 * a21) / det, (a00 * a11 - a01 * a10) / det),
    )
    return tuple(sum(inv[r][c] * b[c] for c in range(3)) for r in range(3))  # type: ignore


def solve_root(
    landmarks: Dict[str, Landmark],
    hfov: float,
    width: int,
    height: int,
    min_conf: float = 0.5,
    names: Optional[Iterable[str]] = None,
) -> Tuple[Optional[Vec3], float]:
    """Return (hip-centre position relative to camera in capture axes, confidence)."""
    fx, fy, cx, cy = intrinsics(hfov, width, height)
    A = [[0.0] * 3 for _ in range(3)]
    b = [0.0] * 3
    wsum = 0.0
    n = 0
    for name, lm in landmarks.items():
        if names is not None and name not in names:
            continue
        if lm.image_xy is None or lm.confidence < min_conf or not lm.valid:
            continue
        X, Y, Z = capture_to_mp_world(lm.position)
        u, v = lm.image_xy
        w = lm.confidence
        # row for x: [fx, 0, -(u-cx)] . T = (u-cx) Z - fx X
        du, dv = u - cx, v - cy
        rows = (((fx, 0.0, -du), du * Z - fx * X), ((0.0, fy, -dv), dv * Z - fy * Y))
        for coef, rhs in rows:
            for i in range(3):
                b[i] += w * coef[i] * rhs
                for j in range(3):
                    A[i][j] += w * coef[i] * coef[j]
        wsum += w
        n += 1
    if n < 4:
        return None, 0.0
    T = _solve3(A, b)
    if T is None or T[2] <= 0.1:
        return None, 0.0
    tx, ty, tz = T
    conf = min(1.0, wsum / max(n, 1))
    return Vec3(tx, tz, -ty), conf
