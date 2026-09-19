"""Canonical source skeleton built from pose landmarks (pure Python).

The retargeting solver never sees raw backend output: every backend (MediaPipe,
synthetic, recorded takes) is converted into a :class:`SourceSkeleton` whose
joints live in *capture space* (metres, Blender axes, subject facing -Y,
+X = subject's left, +Z = up, origin at the hip centre).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .filters import OneEuroParams, PointSetFilter
from .types import PoseFrame, Vec3

SIDES = ("L", "R")

# MediaPipe landmark name -> canonical joint name
JOINT_FROM_MP: Dict[str, str] = {
    "nose": "nose",
    "left_eye": "eye_L",
    "right_eye": "eye_R",
    "left_ear": "ear_L",
    "right_ear": "ear_R",
    "left_shoulder": "shoulder_L",
    "right_shoulder": "shoulder_R",
    "left_elbow": "elbow_L",
    "right_elbow": "elbow_R",
    "left_wrist": "wrist_L",
    "right_wrist": "wrist_R",
    "left_index": "index_L",
    "right_index": "index_R",
    "left_pinky": "pinky_L",
    "right_pinky": "pinky_R",
    "left_thumb": "thumb_L",
    "right_thumb": "thumb_R",
    "left_hip": "hip_L",
    "right_hip": "hip_R",
    "left_knee": "knee_L",
    "right_knee": "knee_R",
    "left_ankle": "ankle_L",
    "right_ankle": "ankle_R",
    "left_heel": "heel_L",
    "right_heel": "heel_R",
    "left_foot_index": "toe_L",
    "right_foot_index": "toe_R",
}

MP_FROM_JOINT: Dict[str, str] = {v: k for k, v in JOINT_FROM_MP.items()}

# Derived joints: name -> (a, b) midpoint
DERIVED_MIDPOINTS: Dict[str, Tuple[str, str]] = {
    "hips_mid": ("hip_L", "hip_R"),
    "shoulders_mid": ("shoulder_L", "shoulder_R"),
    "head": ("ear_L", "ear_R"),
    "hand_L": ("index_L", "pinky_L"),
    "hand_R": ("index_R", "pinky_R"),
}

PRIMARY_JOINTS: Tuple[str, ...] = tuple(JOINT_FROM_MP.values())


def mirror_name(name: str) -> str:
    if name.endswith("_L"):
        return name[:-2] + "_R"
    if name.endswith("_R"):
        return name[:-2] + "_L"
    return name


@dataclass
class SourceSkeleton:
    joints: Dict[str, Vec3] = field(default_factory=dict)
    conf: Dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    frame_index: int = 0
    # Hip-centre position relative to the tracking camera, capture axes (metres).
    root: Optional[Vec3] = None
    root_conf: float = 0.0

    # -- queries -------------------------------------------------------------
    def get(self, name: str, min_conf: float = 0.0) -> Optional[Vec3]:
        p = self.joints.get(name)
        if p is None:
            return None
        if self.conf.get(name, 0.0) < min_conf:
            return None
        return p

    def has(self, names: Iterable[str], min_conf: float = 0.0) -> bool:
        return all(self.get(n, min_conf) is not None for n in names)

    def confidence(self, name: str) -> float:
        return self.conf.get(name, 0.0)

    def copy(self) -> "SourceSkeleton":
        return SourceSkeleton(
            joints={k: v.copy() for k, v in self.joints.items()},
            conf=dict(self.conf),
            timestamp=self.timestamp,
            frame_index=self.frame_index,
            root=self.root.copy() if self.root is not None else None,
            root_conf=self.root_conf,
        )

    # -- construction --------------------------------------------------------
    def derive(self) -> "SourceSkeleton":
        """(Re)compute midpoint joints from their primaries, in place."""
        for name, (a, b) in DERIVED_MIDPOINTS.items():
            pa = self.joints.get(a)
            pb = self.joints.get(b)
            if pa is None or pb is None:
                continue
            self.joints[name] = pa.lerp(pb, 0.5)
            self.conf[name] = min(self.conf.get(a, 0.0), self.conf.get(b, 0.0))
        return self

    @classmethod
    def from_pose_frame(cls, pf: PoseFrame) -> "SourceSkeleton":
        sk = cls(timestamp=pf.timestamp, frame_index=pf.frame_index)
        for mp_name, lm in pf.landmarks.items():
            j = JOINT_FROM_MP.get(mp_name)
            if j is None:
                continue
            sk.joints[j] = lm.position.copy()
            sk.conf[j] = float(lm.confidence) if lm.valid else 0.0
        return sk.derive()

    def mirrored(self) -> "SourceSkeleton":
        """Mirror across the sagittal plane (swap sides, negate X)."""
        out = SourceSkeleton(timestamp=self.timestamp, frame_index=self.frame_index,
                             root_conf=self.root_conf)
        for k, v in self.joints.items():
            out.joints[mirror_name(k)] = Vec3(-v.x, v.y, v.z)
        for k, c in self.conf.items():
            out.conf[mirror_name(k)] = c
        if self.root is not None:
            out.root = Vec3(-self.root.x, self.root.y, self.root.z)
        return out

    # -- serialisation -------------------------------------------------------
    def to_dict(self, precision: int = 5) -> dict:
        d = {
            "t": round(self.timestamp, 6),
            "i": self.frame_index,
            "j": {k: [round(v.x, precision), round(v.y, precision), round(v.z, precision),
                      round(self.conf.get(k, 0.0), 3)]
                  for k, v in self.joints.items() if k in PRIMARY_JOINTS},
        }
        if self.root is not None:
            d["root"] = [round(self.root.x, precision), round(self.root.y, precision),
                         round(self.root.z, precision), round(self.root_conf, 3)]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SourceSkeleton":
        sk = cls(timestamp=float(d.get("t", 0.0)), frame_index=int(d.get("i", 0)))
        for k, v in d.get("j", {}).items():
            sk.joints[k] = Vec3(v[0], v[1], v[2])
            sk.conf[k] = float(v[3]) if len(v) > 3 else 1.0
        r = d.get("root")
        if r:
            sk.root = Vec3(r[0], r[1], r[2])
            sk.root_conf = float(r[3]) if len(r) > 3 else 1.0
        return sk.derive()


class SkeletonFilter:
    """Live temporal conditioning: One Euro smoothing + short gap hold.

    Invalid joints (confidence below ``min_conf``) are held at their last valid
    position *relative to the hip centre* for up to ``max_hold`` seconds, so a
    briefly occluded hand keeps following the body instead of freezing in space.
    """

    def __init__(
        self,
        params: Optional[OneEuroParams] = None,
        min_conf: float = 0.5,
        max_hold: float = 0.35,
    ):
        self.params = params or OneEuroParams()
        self.min_conf = min_conf
        self.max_hold = max_hold
        self._filter = PointSetFilter(self.params)
        self._last_rel: Dict[str, Vec3] = {}
        self._last_time: Dict[str, float] = {}
        self._root_filter = PointSetFilter(OneEuroParams(
            min_cutoff=max(self.params.min_cutoff * 0.5, 0.05) if self.params.enabled else 0.0,
            beta=self.params.beta, d_cutoff=self.params.d_cutoff,
            reset_gap=self.params.reset_gap))

    def reset(self) -> None:
        self._filter.reset()
        self._root_filter.reset()
        self._last_rel.clear()
        self._last_time.clear()

    def process(self, sk: SourceSkeleton) -> SourceSkeleton:
        t = sk.timestamp
        out = SourceSkeleton(timestamp=t, frame_index=sk.frame_index)
        hips = sk.get("hips_mid", self.min_conf)
        for name in PRIMARY_JOINTS:
            p = sk.joints.get(name)
            c = sk.conf.get(name, 0.0)
            if p is not None and c >= self.min_conf:
                fp = self._filter.filter(name, p, t)
                out.joints[name] = fp
                out.conf[name] = c
                if hips is not None:
                    self._last_rel[name] = fp - hips
                    self._last_time[name] = t
            else:
                last = self._last_rel.get(name)
                lt = self._last_time.get(name)
                if last is not None and lt is not None and hips is not None and t - lt <= self.max_hold:
                    out.joints[name] = hips + last
                    # Decay confidence while holding so downstream can tell.
                    out.conf[name] = self.min_conf
                elif p is not None:
                    out.joints[name] = p.copy()
                    out.conf[name] = c
        if sk.root is not None and sk.root_conf > 0.0:
            out.root = self._root_filter.filter("root", sk.root, t)
            out.root_conf = sk.root_conf
        return out.derive()


def interpolate_gaps(
    frames: Sequence[SourceSkeleton],
    min_conf: float = 0.5,
    max_gap: float = 0.5,
) -> List[SourceSkeleton]:
    """Offline gap filling: linearly interpolate low-confidence joints between
    the nearest valid neighbours (relative to the hip centre) when the gap is
    at most ``max_gap`` seconds.  Returns new skeletons.
    """
    out = [f.copy() for f in frames]
    n = len(out)
    if n == 0:
        return out
    for name in PRIMARY_JOINTS:
        valid_idx = [i for i, f in enumerate(frames)
                     if f.get(name, min_conf) is not None and f.get("hips_mid", min_conf) is not None]
        if not valid_idx:
            continue
        for a, b in zip(valid_idx, valid_idx[1:]):
            if b - a <= 1:
                continue
            ta, tb = frames[a].timestamp, frames[b].timestamp
            if tb - ta > max_gap:
                continue
            ra = frames[a].joints[name] - frames[a].joints["hips_mid"]
            rb = frames[b].joints[name] - frames[b].joints["hips_mid"]
            ca = frames[a].conf.get(name, 0.0)
            cb = frames[b].conf.get(name, 0.0)
            for i in range(a + 1, b):
                hips = frames[i].get("hips_mid", min_conf)
                if hips is None:
                    continue
                s = (frames[i].timestamp - ta) / max(tb - ta, 1e-9)
                out[i].joints[name] = hips + ra.lerp(rb, s)
                out[i].conf[name] = max(min_conf, min(ca, cb) * 0.9)
    for f in out:
        f.derive()
    return out


def filter_sequence(
    frames: Sequence[SourceSkeleton],
    params: Optional[OneEuroParams] = None,
    min_conf: float = 0.5,
    max_gap: float = 0.5,
) -> List[SourceSkeleton]:
    """Offline conditioning used by bake: gap interpolation then One Euro."""
    filled = interpolate_gaps(frames, min_conf=min_conf, max_gap=max_gap)
    flt = SkeletonFilter(params, min_conf=min_conf, max_hold=max_gap)
    return [flt.process(f) for f in filled]


def average_skeletons(frames: Sequence[SourceSkeleton], min_conf: float = 0.5) -> Optional[SourceSkeleton]:
    """Hip-relative average of every joint over the frames where it is valid."""
    sums: Dict[str, Vec3] = {}
    counts: Dict[str, int] = {}
    confs: Dict[str, float] = {}
    roots: List[Vec3] = []
    for f in frames:
        hips = f.get("hips_mid", min_conf)
        if hips is None:
            continue
        for name in PRIMARY_JOINTS:
            p = f.get(name, min_conf)
            if p is None:
                continue
            sums[name] = sums.get(name, Vec3()) + (p - hips)
            counts[name] = counts.get(name, 0) + 1
            confs[name] = confs.get(name, 0.0) + f.conf.get(name, 0.0)
        if f.root is not None and f.root_conf >= min_conf:
            roots.append(f.root)
    if not counts:
        return None
    out = SourceSkeleton(timestamp=0.0)
    for name, c in counts.items():
        out.joints[name] = sums[name] / c
        out.conf[name] = confs[name] / c
    if roots:
        acc = Vec3()
        for r in roots:
            acc = acc + r
        out.root = acc / len(roots)
        out.root_conf = 1.0
    return out.derive()


def build_calibration(frames: Sequence[SourceSkeleton], min_conf: float = 0.5, rest_style=None):
    """CalibrationData from a window of frames of the subject holding a neutral pose."""
    from .types import CalibrationData, RestPoseStyle

    neutral = average_skeletons(frames, min_conf)
    cal = CalibrationData(rest_style=rest_style or RestPoseStyle.T_POSE)
    if neutral is None:
        return cal
    legs = []
    for sd in SIDES:
        a, b, c = (neutral.joints.get(f"hip_{sd}"), neutral.joints.get(f"knee_{sd}"),
                   neutral.joints.get(f"ankle_{sd}"))
        if a and b and c:
            legs.append((b - a).length() + (c - b).length())
    cal.scale = sum(legs) / len(legs) if legs else 0.0
    cal.neutral = neutral
    cal.root_origin = neutral.root.copy() if neutral.root is not None else None
    cal.samples = len(frames)
    cal.valid = True
    return cal
