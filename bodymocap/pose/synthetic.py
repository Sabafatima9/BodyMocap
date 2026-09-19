"""Synthetic performer: ground-truth motion clips -> MediaPipe-like landmark feeds.

The actor is the procedural RIG_A skeleton posed from anatomical joint angles.
Landmarks follow MediaPipe's 33-point naming; world coordinates are expressed
in capture space (hip-centred), normalised image coordinates are produced by a
pinhole model of the tracking camera.  Condition models simulate how lighting
and contrast degrade a real tracker (noise, depth error, visibility loss,
dropouts, side-dependent shadowing) so the full pipeline can be stress-tested
without a webcam.  Pure Python, deterministic for a given seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from ..assets.rig_specs import rig_a_spec
from ..core.landmarks import MEDIAPIPE_POSE_NAMES
from ..core.math3d import X_AXIS, Y_AXIS, Z_AXIS, quat_from_axis_angle, quat_identity, quat_pow
from ..core.types import Landmark, PoseFrame, Quat, TrackingState, Vec3
from ..retarget.rig import PoseState, RigModel

D2R = math.pi / 180.0


def _rx(a: float) -> Quat:
    return quat_from_axis_angle(X_AXIS, a)


def _ry(a: float) -> Quat:
    return quat_from_axis_angle(Y_AXIS, a)


def _rz(a: float) -> Quat:
    return quat_from_axis_angle(Z_AXIS, a)


@dataclass
class ArmAngles:
    elev: float = 0.0     # lowering from T-pose (+) / raising (-), radians
    flex: float = 0.0     # forward swing about vertical
    twist: float = 0.0    # humeral rotation about the arm axis
    elbow: float = 0.0    # elbow flexion
    pron: float = 0.0     # forearm pronation
    wrist: float = 0.0    # wrist flexion


@dataclass
class LegAngles:
    flex: float = 0.0     # hip flexion (thigh forward)
    abd: float = 0.0      # hip abduction
    twist: float = 0.0
    knee: float = 0.0     # knee flexion
    ankle: float = 0.0    # plantar flexion (+)


@dataclass
class ActorPose:
    root: Vec3 = field(default_factory=Vec3)
    pelvis_yaw: float = 0.0
    pelvis_pitch: float = 0.0
    pelvis_roll: float = 0.0
    spine_bend: float = 0.0
    spine_side: float = 0.0
    spine_twist: float = 0.0
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    head_roll: float = 0.0
    arm_L: ArmAngles = field(default_factory=ArmAngles)
    arm_R: ArmAngles = field(default_factory=ArmAngles)
    leg_L: LegAngles = field(default_factory=LegAngles)
    leg_R: LegAngles = field(default_factory=LegAngles)


# Rest-space landmark points rigidly attached to actor bones (RIG_A layout).
_ATTACHED: Dict[str, Tuple[str, Tuple[float, float, float]]] = {
    # nose tip ~15 deg below the ear line, as MediaPipe reports for a level head
    "nose": ("Head", (0.0, -0.10, 1.655 - 0.1 * math.tan(15.0 * D2R))),
    "left_eye_inner": ("Head", (0.018, -0.085, 1.668)),
    "left_eye": ("Head", (0.033, -0.083, 1.668)),
    "left_eye_outer": ("Head", (0.048, -0.077, 1.668)),
    "right_eye_inner": ("Head", (-0.018, -0.085, 1.668)),
    "right_eye": ("Head", (-0.033, -0.083, 1.668)),
    "right_eye_outer": ("Head", (-0.048, -0.077, 1.668)),
    "left_ear": ("Head", (0.075, 0.0, 1.655)),
    "right_ear": ("Head", (-0.075, 0.0, 1.655)),
    "mouth_left": ("Head", (0.025, -0.09, 1.595)),
    "mouth_right": ("Head", (-0.025, -0.09, 1.595)),
    # index / pinky knuckles symmetric about the hand axis (y = 0.01)
    "left_index": ("Hand.L", (0.815, -0.0225, 1.42)),
    "left_pinky": ("Hand.L", (0.805, 0.0425, 1.42)),
    "left_thumb": ("Hand.L", (0.78, -0.05, 1.41)),
    "right_index": ("Hand.R", (-0.815, -0.0225, 1.42)),
    "right_pinky": ("Hand.R", (-0.805, 0.0425, 1.42)),
    "right_thumb": ("Hand.R", (-0.78, -0.05, 1.41)),
    "left_heel": ("Foot.L", (0.10, 0.07, 0.01)),
    "right_heel": ("Foot.R", (-0.10, 0.07, 0.01)),
}
_HEADS: Dict[str, str] = {
    "left_shoulder": "UpperArm.L", "right_shoulder": "UpperArm.R",
    "left_elbow": "Forearm.L", "right_elbow": "Forearm.R",
    "left_wrist": "Hand.L", "right_wrist": "Hand.R",
    "left_hip": "Thigh.L", "right_hip": "Thigh.R",
    "left_knee": "Shin.L", "right_knee": "Shin.R",
    "left_ankle": "Foot.L", "right_ankle": "Foot.R",
}
_TAILS: Dict[str, str] = {"left_foot_index": "Toe.L", "right_foot_index": "Toe.R"}


class SyntheticActor:
    """RIG_A-based performer posed from :class:`ActorPose`."""

    def __init__(self):
        self.rig = RigModel.from_spec(rig_a_spec(), name="SyntheticActor")
        sp = ["Spine", "Chest"]
        lens = [self.rig.bones[n].length for n in sp]
        tot = sum(lens)
        self._spine_cum = [lens[0] / tot, 1.0]

    def deltas(self, ap: ActorPose) -> Dict[str, Quat]:
        D: Dict[str, Quat] = {}
        pel = _rz(ap.pelvis_yaw) @ _rx(ap.pelvis_pitch) @ _ry(ap.pelvis_roll)
        D["Hips"] = pel
        sp_rel = _rz(ap.spine_twist) @ _rx(ap.spine_bend) @ _ry(ap.spine_side)
        D["Spine"] = pel @ quat_pow(sp_rel, self._spine_cum[0])
        D["Chest"] = pel @ sp_rel
        head_rel = _rz(ap.head_yaw) @ _rx(ap.head_pitch) @ _ry(ap.head_roll)
        D["Neck"] = D["Chest"] @ quat_pow(head_rel, 0.4)
        D["Head"] = D["Chest"] @ head_rel
        for side, sgn in (("L", 1.0), ("R", -1.0)):
            a: ArmAngles = getattr(ap, f"arm_{side}")
            D[f"Shoulder.{side}"] = D["Chest"]
            sh = _rz(-sgn * a.flex) @ _ry(sgn * a.elev) @ _rx(a.twist)
            D[f"UpperArm.{side}"] = D["Chest"] @ sh
            D[f"Forearm.{side}"] = D[f"UpperArm.{side}"] @ _rz(-sgn * a.elbow) @ _rx(a.pron)
            D[f"Hand.{side}"] = D[f"Forearm.{side}"] @ _ry(sgn * a.wrist)
            g: LegAngles = getattr(ap, f"leg_{side}")
            hip = _rx(-g.flex) @ _ry(-sgn * g.abd) @ _rz(sgn * g.twist)
            D[f"Thigh.{side}"] = pel @ hip
            D[f"Shin.{side}"] = D[f"Thigh.{side}"] @ _rx(g.knee)
            D[f"Foot.{side}"] = D[f"Shin.{side}"] @ _rx(g.ankle)
            D[f"Toe.{side}"] = D[f"Foot.{side}"]
        return D

    def pose(self, ap: ActorPose) -> Tuple[Dict[str, Quat], Dict[str, Vec3], PoseState]:
        D = self.deltas(ap)
        rig = self.rig
        local: Dict[str, Quat] = {}
        for n in rig.order:
            b = rig.bones[n]
            pd = D.get(b.parent) if b.parent else None
            local[n] = rig.local_from_delta(n, D.get(n, pd or quat_identity()), pd)
        loc = {"Hips": rig.bones["Hips"].rest.conjugated().rotate(ap.root)}
        st = rig.fk(local, loc)
        return local, loc, st

    def world_landmarks(self, ap: ActorPose) -> Dict[str, Vec3]:
        """Landmark positions in actor armature space (== world, actor at origin)."""
        _, _, st = self.pose(ap)
        out: Dict[str, Vec3] = {}
        for lm, bone in _HEADS.items():
            out[lm] = st.head[bone].copy()
        for lm, bone in _TAILS.items():
            out[lm] = st.tail(bone)
        for lm, (bone, p) in _ATTACHED.items():
            out[lm] = st.point(bone, Vec3(*p))
        return out


# ---------------------------------------------------------------------------
# Motion clips (t in seconds)
# ---------------------------------------------------------------------------

def _s(t: float, hz: float, phase: float = 0.0) -> float:
    return math.sin(2.0 * math.pi * hz * t + phase)


def _ease(t: float, period: float) -> float:
    """0 -> 1 -> 0 smooth cycle."""
    return 0.5 - 0.5 * math.cos(2.0 * math.pi * t / period)


def clip_tpose(t: float) -> ActorPose:
    return ActorPose()


def clip_arm_raise(t: float) -> ActorPose:
    k = _ease(t, 4.0)
    ap = ActorPose()
    for side in ("L", "R"):
        a = getattr(ap, f"arm_{side}")
        a.elev = (75.0 - 150.0 * k) * D2R
        a.elbow = 12.0 * D2R
        a.flex = 10.0 * D2R
    return ap


def clip_wave(t: float) -> ActorPose:
    ap = ActorPose()
    ap.arm_L.elev = 75.0 * D2R
    ap.arm_L.elbow = 10.0 * D2R
    r = ap.arm_R
    r.elev = 5.0 * D2R
    r.flex = 20.0 * D2R
    r.twist = 90.0 * D2R
    r.elbow = (75.0 + 35.0 * _s(t, 2.5)) * D2R
    r.pron = 20.0 * D2R
    ap.head_yaw = -10.0 * D2R
    return ap


def clip_punch_fast(t: float) -> ActorPose:
    ap = ActorPose()
    for side, ph in (("L", 0.0), ("R", math.pi)):
        a = getattr(ap, f"arm_{side}")
        k = 0.5 + 0.5 * _s(t, 3.0, ph)
        a.elev = 20.0 * D2R
        a.flex = (80.0 - 10.0 * k) * D2R
        a.twist = 60.0 * D2R
        a.elbow = (8.0 + 110.0 * k) * D2R
        a.pron = 60.0 * (1.0 - k) * D2R
    ap.spine_twist = 12.0 * _s(t, 3.0) * D2R
    return ap


def clip_squat(t: float) -> ActorPose:
    k = _ease(t, 3.0)
    ap = ActorPose()
    ap.root = Vec3(0.0, 0.0, -0.32 * k)
    ap.pelvis_pitch = 25.0 * k * D2R
    ap.spine_bend = 12.0 * k * D2R
    for side in ("L", "R"):
        g = getattr(ap, f"leg_{side}")
        g.flex = 85.0 * k * D2R
        g.knee = 115.0 * k * D2R
        g.ankle = -28.0 * k * D2R
        g.abd = 8.0 * D2R
        a = getattr(ap, f"arm_{side}")
        a.elev = (75.0 - 70.0 * k) * D2R
        a.flex = 85.0 * k * D2R
        a.elbow = 10.0 * D2R
    return ap


def clip_reach_cross(t: float) -> ActorPose:
    k = _ease(t, 3.0)
    ap = ActorPose()
    ap.spine_twist = 20.0 * k * D2R
    ap.arm_L.elev = 70.0 * D2R
    ap.arm_L.elbow = 30.0 * D2R
    r = ap.arm_R
    r.elev = (60.0 - 50.0 * k) * D2R
    r.flex = (20.0 + 110.0 * k) * D2R
    r.elbow = (50.0 - 40.0 * k) * D2R
    r.twist = 30.0 * D2R
    ap.head_yaw = 25.0 * k * D2R
    return ap


def clip_subtle_idle(t: float) -> ActorPose:
    ap = ActorPose()
    ap.spine_bend = 2.5 * _s(t, 0.25) * D2R
    ap.spine_side = 1.5 * _s(t, 0.17, 1.0) * D2R
    ap.head_yaw = 6.0 * _s(t, 0.2) * D2R
    ap.head_pitch = 3.0 * _s(t, 0.15, 0.5) * D2R
    ap.pelvis_roll = 1.5 * _s(t, 0.12) * D2R
    for side, ph in (("L", 0.0), ("R", 1.3)):
        a = getattr(ap, f"arm_{side}")
        a.elev = (72.0 + 3.0 * _s(t, 0.3, ph)) * D2R
        a.elbow = (12.0 + 4.0 * _s(t, 0.25, ph)) * D2R
        a.flex = 5.0 * D2R
        g = getattr(ap, f"leg_{side}")
        g.knee = (4.0 + 2.0 * _s(t, 0.2, ph)) * D2R
        g.flex = 2.0 * D2R
    return ap


def clip_torso_twist(t: float) -> ActorPose:
    ap = ActorPose()
    ap.spine_twist = 35.0 * _s(t, 0.5) * D2R
    ap.spine_side = 15.0 * _s(t, 0.5, math.pi / 2) * D2R
    ap.spine_bend = (10.0 + 10.0 * _s(t, 0.25)) * D2R
    ap.head_yaw = 30.0 * _s(t, 0.5, 0.4) * D2R
    for side in ("L", "R"):
        a = getattr(ap, f"arm_{side}")
        a.elev = 45.0 * D2R
        a.elbow = 20.0 * D2R
    return ap


def clip_march(t: float) -> ActorPose:
    ap = ActorPose()
    for side, ph in (("L", 0.0), ("R", math.pi)):
        k = max(0.0, _s(t, 1.0, ph))
        g = getattr(ap, f"leg_{side}")
        g.flex = 70.0 * k * D2R
        g.knee = 90.0 * k * D2R
        g.ankle = 10.0 * k * D2R
        a = getattr(ap, f"arm_{side}")
        a.elev = 75.0 * D2R
        a.flex = -30.0 * _s(t, 1.0, ph) * D2R
        a.elbow = (20.0 + 30.0 * k) * D2R
    ap.pelvis_roll = 4.0 * _s(t, 1.0) * D2R
    return ap


def clip_straight_arm_circles(t: float) -> ActorPose:
    """Arms nearly straight tracing circles: stresses pole-vector stability."""
    ap = ActorPose()
    for side, ph in (("L", 0.0), ("R", math.pi / 3)):
        a = getattr(ap, f"arm_{side}")
        a.elev = (10.0 + 50.0 * _s(t, 0.5, ph)) * D2R
        a.flex = (45.0 + 45.0 * _s(t, 0.5, ph + math.pi / 2)) * D2R
        a.elbow = (3.0 + 2.0 * _s(t, 1.7, ph)) * D2R
    return ap


def clip_side_kick(t: float) -> ActorPose:
    k = _ease(t, 2.0)
    ap = ActorPose()
    ap.pelvis_roll = -10.0 * k * D2R
    ap.spine_side = 10.0 * k * D2R
    g = ap.leg_R
    g.abd = 50.0 * k * D2R
    g.flex = 15.0 * k * D2R
    g.knee = (40.0 - 30.0 * k) * D2R
    ap.leg_L.knee = 10.0 * k * D2R
    for side in ("L", "R"):
        a = getattr(ap, f"arm_{side}")
        a.elev = (60.0 - 40.0 * k) * D2R
        a.elbow = 60.0 * D2R
        a.flex = 30.0 * D2R
    return ap


CLIPS: Dict[str, Callable[[float], ActorPose]] = {
    "tpose": clip_tpose,
    "arm_raise": clip_arm_raise,
    "wave": clip_wave,
    "punch_fast": clip_punch_fast,
    "squat": clip_squat,
    "reach_cross": clip_reach_cross,
    "subtle_idle": clip_subtle_idle,
    "torso_twist": clip_torso_twist,
    "march": clip_march,
    "straight_arm_circles": clip_straight_arm_circles,
    "side_kick": clip_side_kick,
}


# ---------------------------------------------------------------------------
# Tracking camera + condition models
# ---------------------------------------------------------------------------

@dataclass
class CameraModel:
    """Pinhole model of the tracking camera (capture/world axes)."""

    position: Vec3 = field(default_factory=lambda: Vec3(0.0, -3.2, 1.05))
    hfov: float = 60.0 * D2R
    width: int = 640
    height: int = 480

    @property
    def fx(self) -> float:  # normalised (image width = 1)
        return 0.5 / math.tan(self.hfov * 0.5)

    def project(self, p: Vec3) -> Tuple[float, float, float]:
        """World point -> (u, v, depth) with u, v normalised (0..1, v down)."""
        rel = p - self.position
        depth = max(rel.y, 1e-6)
        fy = self.fx * self.width / self.height
        return 0.5 + self.fx * rel.x / depth, 0.5 - fy * rel.z / depth, depth


@dataclass
class Condition:
    name: str
    noise_xz: float = 0.004     # metres (image-plane axes)
    noise_depth: float = 0.012  # metres (camera depth axis)
    depth_walk: float = 0.0     # random-walk depth bias step (m/frame)
    vis: Tuple[float, float] = (0.9, 0.99)
    dropout: float = 0.0        # probability a distal landmark is lost per frame
    shadow_side: str = ""       # "L"/"R": side facing away from the key light
    shadow_noise: float = 1.0   # noise multiplier on the shadow side
    shadow_vis: float = 1.0     # visibility multiplier on the shadow side
    miss_frame: float = 0.0     # probability the whole frame is lost


CONDITIONS: Dict[str, Condition] = {
    "clean": Condition("clean", noise_xz=0.0, noise_depth=0.0, vis=(0.99, 0.99)),
    "normal": Condition("normal"),
    "low_contrast": Condition("low_contrast", noise_xz=0.010, noise_depth=0.03, depth_walk=0.002,
                              vis=(0.55, 0.85), dropout=0.03, miss_frame=0.02),
    "backlight": Condition("backlight", noise_xz=0.014, noise_depth=0.04, depth_walk=0.003,
                           vis=(0.5, 0.8), dropout=0.06, miss_frame=0.04),
    "extreme_key": Condition("extreme_key", noise_xz=0.006, noise_depth=0.02, vis=(0.75, 0.95),
                             dropout=0.04, shadow_side="R", shadow_noise=3.0, shadow_vis=0.7,
                             miss_frame=0.01),
    "low_light": Condition("low_light", noise_xz=0.016, noise_depth=0.05, depth_walk=0.004,
                           vis=(0.5, 0.75), dropout=0.08, miss_frame=0.05),
}

_DISTAL = ("wrist", "index", "pinky", "thumb", "ankle", "heel", "foot_index")


@dataclass
class SyntheticFeed:
    frames: List[PoseFrame]
    truth: List[Dict[str, Vec3]]            # clean world landmarks per frame
    actor_local: List[Dict[str, Quat]]      # ground-truth actor local rotations
    actor_root: List[Dict[str, Vec3]]
    camera: CameraModel
    clip: str
    condition: str


class SyntheticStream:
    """Frame-by-frame landmark generator (used by the live mock backend)."""

    def __init__(
        self,
        clip: str = "wave",
        condition: str = "normal",
        speed: float = 1.0,
        seed: int = 0,
        camera: Optional[CameraModel] = None,
        actor: Optional[SyntheticActor] = None,
    ):
        self.clip = clip
        self.condition = condition
        self.speed = speed
        self.rng = random.Random(seed)
        self.cam = camera or CameraModel()
        self.cond = CONDITIONS[condition]
        self.fn = CLIPS[clip]
        self.actor = actor or SyntheticActor()
        self._bias = Vec3()

    def frame(self, index: int, t: float):
        """Return (PoseFrame, clean world landmarks, actor local rotations, actor root)."""
        rng, cam, cond = self.rng, self.cam, self.cond
        ap = self.fn(t * self.speed)
        local, loc, st = self.actor.pose(ap)
        world = self.actor.world_landmarks(ap)
        hips_mid = world["left_hip"].lerp(world["right_hip"], 0.5)
        if cond.depth_walk > 0.0:
            self._bias = self._bias * 0.97 + Vec3(0.0, rng.gauss(0.0, cond.depth_walk), 0.0)
        lost = rng.random() < cond.miss_frame
        pf = PoseFrame(timestamp=t, frame_index=index, image_size=(cam.width, cam.height),
                       tracking_state=TrackingState.LOST if lost else TrackingState.OK)
        if not lost:
            for idx in range(33):
                name = MEDIAPIPE_POSE_NAMES[idx]
                p = world.get(name)
                if p is None:
                    continue
                side = "L" if name.startswith("left") else ("R" if name.startswith("right") else "")
                shadow = bool(cond.shadow_side) and side == cond.shadow_side
                k = cond.shadow_noise if shadow else 1.0
                noisy = Vec3(
                    p.x + rng.gauss(0.0, cond.noise_xz * k),
                    p.y + rng.gauss(0.0, cond.noise_depth * k) + self._bias.y,
                    p.z + rng.gauss(0.0, cond.noise_xz * k),
                )
                vis = rng.uniform(*cond.vis) * (cond.shadow_vis if shadow else 1.0)
                if any(d in name for d in _DISTAL) and rng.random() < cond.dropout:
                    vis = rng.uniform(0.05, 0.3)
                u, v, _ = cam.project(p)
                if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
                    vis *= 0.3
                pf.landmarks[name] = Landmark(
                    name=name, position=noisy - hips_mid, confidence=vis, valid=True,
                    image_xy=(u + rng.gauss(0.0, 0.002 * k), v + rng.gauss(0.0, 0.002 * k)),
                )
        return pf, world, local, loc


def generate_feed(
    clip: str,
    duration: float = 4.0,
    fps: float = 30.0,
    condition: str = "normal",
    speed: float = 1.0,
    seed: int = 0,
    camera: Optional[CameraModel] = None,
    timing_jitter: float = 0.0,
    actor: Optional[SyntheticActor] = None,
) -> SyntheticFeed:
    stream = SyntheticStream(clip, condition, speed, seed, camera, actor)
    rng = stream.rng
    n = max(1, int(round(duration * fps)))
    frames: List[PoseFrame] = []
    truth: List[Dict[str, Vec3]] = []
    locals_: List[Dict[str, Quat]] = []
    roots: List[Dict[str, Vec3]] = []
    t = 0.0
    for i in range(n):
        if i > 0:
            dt = 1.0 / fps
            if timing_jitter > 0.0:
                dt *= 1.0 + rng.uniform(-timing_jitter, timing_jitter)
            t += dt
        pf, world, local, loc = stream.frame(i, t)
        frames.append(pf)
        truth.append(world)
        locals_.append(local)
        roots.append(loc)
    return SyntheticFeed(frames, truth, locals_, roots, stream.cam, clip, condition)
