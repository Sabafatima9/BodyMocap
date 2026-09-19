"""Pure 3D math helpers (no bpy).

Conventions match Blender/mathutils: quaternions are (w, x, y, z), matrices are
column-major lists of basis vectors, rotations are right-handed.  A *frame* is a
rotation (Quat) whose columns are the frame's axes expressed in the parent space.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .types import Quat, Vec3

EPS = 1e-12

X_AXIS = Vec3(1.0, 0.0, 0.0)
Y_AXIS = Vec3(0.0, 1.0, 0.0)
Z_AXIS = Vec3(0.0, 0.0, 1.0)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge1 <= edge0:
        return 1.0 if x >= edge1 else 0.0
    t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def vec_lerp(a: Vec3, b: Vec3, t: float) -> Vec3:
    return a.lerp(b, t)


def angle_between(a: Vec3, b: Vec3) -> float:
    """Unsigned angle in radians between two vectors (0 if degenerate)."""
    la = a.length()
    lb = b.length()
    if la < EPS or lb < EPS:
        return 0.0
    return math.acos(clamp(a.dot(b) / (la * lb), -1.0, 1.0))


def reject(v: Vec3, axis_unit: Vec3) -> Vec3:
    """Component of v perpendicular to a unit axis."""
    return v - axis_unit * v.dot(axis_unit)


def any_perpendicular(v: Vec3) -> Vec3:
    """A unit vector perpendicular to v (deterministic)."""
    n = v.normalized()
    ref = X_AXIS if abs(n.x) < 0.9 else Y_AXIS
    return n.cross(ref).normalized()


def signed_angle_about(a: Vec3, b: Vec3, axis_unit: Vec3) -> float:
    """Signed angle (radians) rotating a onto b around axis (right-hand rule).

    Both vectors are projected onto the plane perpendicular to the axis first.
    """
    pa = reject(a, axis_unit)
    pb = reject(b, axis_unit)
    if pa.length_squared() < EPS or pb.length_squared() < EPS:
        return 0.0
    return math.atan2(axis_unit.dot(pa.cross(pb)), pa.dot(pb))


def wrap_angle(a: float) -> float:
    """Wrap to (-pi, pi]."""
    a = math.fmod(a + math.pi, 2.0 * math.pi)
    if a <= 0.0:
        a += 2.0 * math.pi
    return a - math.pi


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------

def quat_identity() -> Quat:
    return Quat(1.0, 0.0, 0.0, 0.0)


def quat_normalize(q: Quat) -> Quat:
    n = math.sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z)
    if n < EPS:
        return quat_identity()
    return Quat(q.w / n, q.x / n, q.y / n, q.z / n)


def quat_conjugate(q: Quat) -> Quat:
    return Quat(q.w, -q.x, -q.y, -q.z)


def quat_mul(a: Quat, b: Quat) -> Quat:
    return a @ b


def quat_dot(a: Quat, b: Quat) -> float:
    return a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z


def quat_neg(q: Quat) -> Quat:
    return Quat(-q.w, -q.x, -q.y, -q.z)


def quat_align_hemisphere(q: Quat, ref: Quat) -> Quat:
    """Return q or -q, whichever is closer to ref (same rotation)."""
    return quat_neg(q) if quat_dot(q, ref) < 0.0 else q


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    return q.rotate(v)


def quat_from_axis_angle(axis: Vec3, angle: float) -> Quat:
    n = axis.normalized()
    if n.length_squared() < EPS:
        return quat_identity()
    s = math.sin(angle * 0.5)
    return Quat(math.cos(angle * 0.5), n.x * s, n.y * s, n.z * s)


def quat_to_axis_angle(q: Quat) -> Tuple[Vec3, float]:
    """Shortest-arc axis/angle (angle in [0, pi])."""
    q = quat_normalize(q)
    if q.w < 0.0:
        q = quat_neg(q)
    s = math.sqrt(max(0.0, 1.0 - q.w * q.w))
    angle = 2.0 * math.atan2(s, q.w)
    if s < 1e-9:
        return Vec3(1.0, 0.0, 0.0), 0.0
    return Vec3(q.x / s, q.y / s, q.z / s), angle


def quat_pow(q: Quat, t: float) -> Quat:
    """Scale the rotation angle of q by t (shortest arc)."""
    axis, angle = quat_to_axis_angle(q)
    return quat_from_axis_angle(axis, angle * t)


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = quat_dot(a, b)
    if dot < 0.0:
        b = quat_neg(b)
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(
            Quat(lerp(a.w, b.w, t), lerp(a.x, b.x, t), lerp(a.y, b.y, t), lerp(a.z, b.z, t))
        )
    theta_0 = math.acos(clamp(dot, -1.0, 1.0))
    sin_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.cos(theta) - dot * math.sin(theta) / sin_0
    s1 = math.sin(theta) / sin_0
    return Quat(
        s0 * a.w + s1 * b.w,
        s0 * a.x + s1 * b.x,
        s0 * a.y + s1 * b.y,
        s0 * a.z + s1 * b.z,
    )


def quat_from_two_vectors(from_v: Vec3, to_v: Vec3) -> Quat:
    """Smallest rotation taking from_v to to_v (swing)."""
    a = from_v.normalized()
    b = to_v.normalized()
    if a.length_squared() < EPS or b.length_squared() < EPS:
        return quat_identity()
    d = a.dot(b)
    if d > 0.999999:
        return quat_identity()
    if d < -0.999999:
        axis = any_perpendicular(a)
        return Quat(0.0, axis.x, axis.y, axis.z)
    c = a.cross(b)
    return quat_normalize(Quat(1.0 + d, c.x, c.y, c.z))


def quat_swing_twist(q: Quat, axis_unit: Vec3) -> Tuple[Quat, Quat]:
    """Decompose q = swing @ twist, twist being a rotation about axis_unit."""
    p = axis_unit.dot(Vec3(q.x, q.y, q.z))
    twist = Quat(q.w, axis_unit.x * p, axis_unit.y * p, axis_unit.z * p)
    n = math.sqrt(twist.w * twist.w + twist.x * twist.x + twist.y * twist.y + twist.z * twist.z)
    if n < 1e-9:
        twist = quat_identity()
    else:
        twist = Quat(twist.w / n, twist.x / n, twist.y / n, twist.z / n)
    swing = q @ quat_conjugate(twist)
    return swing, twist


def quat_twist_angle(q: Quat, axis_unit: Vec3) -> float:
    """Signed twist angle (radians, wrapped) of q about axis_unit."""
    _, tw = quat_swing_twist(q, axis_unit)
    p = axis_unit.dot(Vec3(tw.x, tw.y, tw.z))
    return wrap_angle(2.0 * math.atan2(p, tw.w))


def quat_average(quats: Sequence[Quat]) -> Quat:
    """Sign-aligned normalised sum (good approximation for nearby rotations)."""
    if not quats:
        return quat_identity()
    ref = quat_normalize(quats[0])
    acc = Quat(0.0, 0.0, 0.0, 0.0)
    for q in quats:
        q = quat_align_hemisphere(quat_normalize(q), ref)
        acc = Quat(acc.w + q.w, acc.x + q.x, acc.y + q.y, acc.z + q.z)
    return quat_normalize(acc)


def cumulative_quat_product(quats: Sequence[Quat]) -> Quat:
    r = quat_identity()
    for q in quats:
        r = r @ q
    return r


def quat_angle(a: Quat, b: Quat) -> float:
    """Angular difference in radians between two rotations."""
    d = abs(quat_dot(quat_normalize(a), quat_normalize(b)))
    return 2.0 * math.acos(clamp(d, 0.0, 1.0))


def quat_angle_deg(a: Quat, b: Quat) -> float:
    return math.degrees(quat_angle(a, b))


def euler_xyz_to_quat(rx: float, ry: float, rz: float) -> Quat:
    """Euler XYZ (Blender 'XYZ' order: X applied first) radians -> quaternion."""
    qx = quat_from_axis_angle(X_AXIS, rx)
    qy = quat_from_axis_angle(Y_AXIS, ry)
    qz = quat_from_axis_angle(Z_AXIS, rz)
    return quat_normalize(qz @ qy @ qx)


# ---------------------------------------------------------------------------
# Matrices / frames
# ---------------------------------------------------------------------------

Basis = Tuple[Vec3, Vec3, Vec3]  # (x_axis, y_axis, z_axis) columns


def basis_to_quat(bx: Vec3, by: Vec3, bz: Vec3) -> Quat:
    """Rotation matrix given by columns -> unit quaternion."""
    m00, m01, m02 = bx.x, by.x, bz.x
    m10, m11, m12 = bx.y, by.y, bz.y
    m20, m21, m22 = bx.z, by.z, bz.z
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        q = Quat(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s)
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = Quat((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = Quat((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = Quat((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
    q = quat_normalize(q)
    if q.w < 0.0:
        q = quat_neg(q)
    return q


def quat_to_basis(q: Quat) -> Basis:
    return (q.rotate(X_AXIS), q.rotate(Y_AXIS), q.rotate(Z_AXIS))


def frame_aim_up(aim: Vec3, up: Vec3) -> Optional[Quat]:
    """Bone-style frame: Y along aim, Z towards up (orthogonalised), X = Y x Z.

    Returns None when aim is degenerate.  When up is (nearly) parallel to aim a
    deterministic perpendicular is used.
    """
    y = aim.normalized()
    if y.length_squared() < 0.5:
        return None
    z = reject(up, y)
    if z.length_squared() < 1e-10:
        z = any_perpendicular(y)
    else:
        z = z.normalized()
    x = y.cross(z)
    return basis_to_quat(x, y, z)


def frame_left_up(left: Vec3, up: Vec3) -> Optional[Quat]:
    """Body frame with columns (left, back, up) = (X, Y, Z) of a subject that
    faces -Y.  ``up`` is orthogonalised against ``left``.
    """
    l = left.normalized()
    if l.length_squared() < 0.5:
        return None
    u = reject(up, l)
    if u.length_squared() < 1e-10:
        return None
    u = u.normalized()
    b = u.cross(l)
    return basis_to_quat(l, b, u)


def delta_between_frames(current: Quat, rest: Quat) -> Quat:
    """Rotation D with D @ rest == current."""
    return quat_normalize(current @ quat_conjugate(rest))


# ---------------------------------------------------------------------------
# Chains / arc length
# ---------------------------------------------------------------------------

def chain_rest_lengths(positions: Sequence[Vec3]) -> List[float]:
    """Segment lengths between consecutive joint positions."""
    return [(positions[i + 1] - positions[i]).length() for i in range(len(positions) - 1)]


def normalize_weights(lengths: Sequence[float]) -> List[float]:
    total = float(sum(lengths))
    n = len(lengths)
    if n == 0:
        return []
    if total < EPS:
        return [1.0 / n] * n
    return [L / total for L in lengths]


def cumulative_weights(lengths: Sequence[float]) -> List[float]:
    """Arc-length parameter at the *end* of each segment (last value is 1)."""
    out: List[float] = []
    acc = 0.0
    for w in normalize_weights(lengths):
        acc += w
        out.append(acc)
    if out:
        out[-1] = 1.0
    return out


def polyline_point_at(points: Sequence[Vec3], u: float) -> Vec3:
    """Point at normalised arc length u in [0, 1] along a polyline."""
    if not points:
        return Vec3()
    if len(points) == 1:
        return points[0].copy()
    lengths = chain_rest_lengths(points)
    total = sum(lengths)
    if total < EPS:
        return points[0].copy()
    target = clamp(u, 0.0, 1.0) * total
    acc = 0.0
    for i, seg in enumerate(lengths):
        if acc + seg >= target or i == len(lengths) - 1:
            t = 0.0 if seg < EPS else (target - acc) / seg
            return points[i].lerp(points[i + 1], clamp(t, 0.0, 1.0))
        acc += seg
    return points[-1].copy()


def direction_from_landmarks(parent: Vec3, child: Vec3) -> Vec3:
    return (child - parent).normalized()


def fabrik(
    joints: List[Vec3],
    lengths: Sequence[float],
    target: Vec3,
    iterations: int = 24,
    tolerance: float = 1e-6,
) -> float:
    """In-place FABRIK solve with a fixed base. Returns the final end error.

    joints has len(lengths) + 1 entries; joints[0] is the fixed base.
    """
    n = len(lengths)
    if n == 0:
        return (target - joints[0]).length()
    base = joints[0].copy()
    total = float(sum(lengths))
    to_target = target - base
    dist = to_target.length()
    if dist >= total - 1e-12:
        d = to_target.normalized() if dist > EPS else (joints[-1] - base).normalized()
        if d.length_squared() < 0.5:
            d = Y_AXIS
        acc = base
        joints[0] = base
        for i in range(n):
            acc = acc + d * lengths[i]
            joints[i + 1] = acc
        return (joints[-1] - target).length()

    def _dir(a: Vec3, b: Vec3, fallback: Vec3) -> Vec3:
        v = a - b
        L = v.length()
        if L < 1e-12:
            return fallback
        return v / L

    # Always run at least one backward/forward pass: an initial guess whose end
    # already sits on the target may still violate the segment lengths.
    err = float("inf")
    for _ in range(max(1, iterations)):
        joints[n] = target.copy()
        for i in range(n - 1, -1, -1):
            joints[i] = joints[i + 1] + _dir(joints[i], joints[i + 1], -Y_AXIS) * lengths[i]
        joints[0] = base.copy()
        for i in range(n):
            joints[i + 1] = joints[i] + _dir(joints[i + 1], joints[i], Y_AXIS) * lengths[i]
        err = (joints[-1] - target).length()
        if err <= tolerance:
            break
    return err


def vec_roll_to_quat(direction: Vec3, roll: float) -> Quat:
    """Blender's bone rest orientation from a (head->tail) vector and roll.

    Port of ``vec_roll_to_mat3_normalized`` from Blender's armature code so
    pure-Python rigs get byte-for-byte the same rest frames as edit bones.
    """
    nor = direction.normalized()
    x, y, z = nor.x, nor.y, nor.z
    SAFE_THRESHOLD = 6.1e-3
    CRITICAL_THRESHOLD = 2.5e-4
    theta = 1.0 + y
    theta_alt = x * x + z * z
    if theta > SAFE_THRESHOLD or theta_alt > CRITICAL_THRESHOLD * CRITICAL_THRESHOLD:
        if theta <= SAFE_THRESHOLD:
            theta = theta_alt * 0.5 + theta_alt * theta_alt * 0.125
        bx = Vec3(1.0 - x * x / theta, -x, -x * z / theta)
        by = Vec3(x, y, z)
        bz = Vec3(-x * z / theta, -z, 1.0 - z * z / theta)
    else:
        bx = Vec3(-1.0, 0.0, 0.0)
        by = Vec3(0.0, -1.0, 0.0)
        bz = Vec3(0.0, 0.0, 1.0)
    b = basis_to_quat(bx, by, bz)
    r = quat_from_axis_angle(nor, roll)
    return quat_normalize(r @ b)
