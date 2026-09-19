"""Shared types for BodyMocap (pure Python, no bpy).

Vec3 / Quat are small slotted value types so the math core can run (and be
unit-tested) outside Blender.  Quaternions are scalar-first (w, x, y, z), the
same convention as ``mathutils.Quaternion``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


class TrackingState(Enum):
    """Pose tracking quality (FR-023)."""

    OK = auto()
    DEGRADED = auto()
    LOST = auto()


class RestPoseStyle(Enum):
    T_POSE = "T_POSE"
    A_POSE = "A_POSE"


class Vec3:
    __slots__ = ("x", "y", "z")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def from_seq(cls, seq: Sequence[float]) -> "Vec3":
        return cls(seq[0], seq[1], seq[2])

    def __repr__(self) -> str:
        return f"Vec3({self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y
        yield self.z

    def __getitem__(self, i: int) -> float:
        return (self.x, self.y, self.z)[i]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vec3):
            return NotImplemented
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self) -> int:
        return hash((self.x, self.y, self.z))

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def copy(self) -> "Vec3":
        return Vec3(self.x, self.y, self.z)

    def __add__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o: "Vec3") -> "Vec3":
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __rmul__(self, s: float) -> "Vec3":
        return Vec3(self.x * s, self.y * s, self.z * s)

    def __truediv__(self, s: float) -> "Vec3":
        return Vec3(self.x / s, self.y / s, self.z / s)

    def __neg__(self) -> "Vec3":
        return Vec3(-self.x, -self.y, -self.z)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def length_squared(self) -> float:
        return self.x * self.x + self.y * self.y + self.z * self.z

    def normalized(self) -> "Vec3":
        n = self.length()
        if n < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / n, self.y / n, self.z / n)

    def dot(self, o: "Vec3") -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, o: "Vec3") -> "Vec3":
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def lerp(self, o: "Vec3", t: float) -> "Vec3":
        return Vec3(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )


class Quat:
    """Quaternion (w, x, y, z), scalar-first."""

    __slots__ = ("w", "x", "y", "z")

    def __init__(self, w: float = 1.0, x: float = 0.0, y: float = 0.0, z: float = 0.0):
        self.w = float(w)
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    @classmethod
    def from_seq(cls, seq: Sequence[float]) -> "Quat":
        return cls(seq[0], seq[1], seq[2], seq[3])

    def __repr__(self) -> str:
        return f"Quat({self.w:.6g}, {self.x:.6g}, {self.y:.6g}, {self.z:.6g})"

    def __iter__(self) -> Iterator[float]:
        yield self.w
        yield self.x
        yield self.y
        yield self.z

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Quat):
            return NotImplemented
        return (self.w, self.x, self.y, self.z) == (other.w, other.x, other.y, other.z)

    def __hash__(self) -> int:
        return hash((self.w, self.x, self.y, self.z))

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.w, self.x, self.y, self.z)

    def as_xyzw(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.w)

    def copy(self) -> "Quat":
        return Quat(self.w, self.x, self.y, self.z)

    def __matmul__(self, b: "Quat") -> "Quat":
        a = self
        return Quat(
            a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        )

    def conjugated(self) -> "Quat":
        return Quat(self.w, -self.x, -self.y, -self.z)

    def rotate(self, v: Vec3) -> Vec3:
        """Rotate vector v by this (unit) quaternion."""
        # t = 2 * cross(q.xyz, v); v' = v + w * t + cross(q.xyz, t)
        qx, qy, qz, qw = self.x, self.y, self.z, self.w
        tx = 2.0 * (qy * v.z - qz * v.y)
        ty = 2.0 * (qz * v.x - qx * v.z)
        tz = 2.0 * (qx * v.y - qy * v.x)
        return Vec3(
            v.x + qw * tx + (qy * tz - qz * ty),
            v.y + qw * ty + (qz * tx - qx * tz),
            v.z + qw * tz + (qx * ty - qy * tx),
        )


@dataclass
class Landmark:
    """One detected landmark.

    ``position`` is in *capture space* (metres, Blender axes: +X = subject's
    left when facing the camera, -Y = towards the camera, +Z = up), centred
    on the hips.  ``image_xy`` holds normalised image coordinates (0..1, y
    down) when the backend provides them.
    """

    name: str
    position: Vec3
    confidence: float = 1.0
    valid: bool = True
    image_xy: Optional[Tuple[float, float]] = None


@dataclass
class PoseFrame:
    """One frame of pose estimation output."""

    landmarks: Dict[str, Landmark] = field(default_factory=dict)
    tracking_state: TrackingState = TrackingState.LOST
    timestamp: float = 0.0
    frame_index: int = 0
    image_size: Optional[Tuple[int, int]] = None  # (width, height) of the analysed frame


@dataclass
class RecordingFrame:
    frame_index: int
    bone_rotations: Dict[str, Quat] = field(default_factory=dict)
    tracking_state: TrackingState = TrackingState.OK
    timestamp: float = 0.0


@dataclass
class ChainDefinition:
    name: str
    bone_names: List[str]
    side: str = ""  # L, R, or ""


@dataclass
class CalibrationData:
    """Per-subject neutral pose (SourceSkeleton) + derived quantities."""

    rest_style: RestPoseStyle = RestPoseStyle.T_POSE
    scale: float = 0.0               # subject leg length (hip->knee->ankle), metres
    neutral: Optional[object] = None  # averaged SourceSkeleton of the calibration window
    root_origin: Optional[Vec3] = None
    samples: int = 0
    valid: bool = False
