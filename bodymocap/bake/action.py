"""Bake a take (source skeleton frames) into a Blender Action (FR-060-063).

Pipeline per target armature:
  take --(gap fill + One Euro)--> resampled to scene frames --> RetargetSolver
  --> hemisphere-continuous local rotations (+ root location) --> bulk F-Curves.
The solving half is pure Python (``solve_take``) so it can be unit-tested; only
``write_action`` touches bpy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.filters import OneEuroParams
from ..core.math3d import quat_align_hemisphere, quat_angle
from ..core.skeleton import SourceSkeleton, filter_sequence
from ..core.types import CalibrationData, Quat, Vec3
from ..retarget.rig import RigModel
from ..retarget.solver import RetargetSolver, SolveResult, SolverSettings
from ..retarget.topology import TopologyProfile


@dataclass
class BakeSettings:
    action_name: str = "BodyMocapAction"
    start_frame: int = 1
    fps: float = 24.0
    overwrite: bool = True
    rotation_mode: str = "QUATERNION"   # QUATERNION | EULER
    interpolation: str = "LINEAR"       # LINEAR | BEZIER
    smoothing: OneEuroParams = field(default_factory=OneEuroParams)
    gap_fill: float = 0.5               # seconds
    min_conf: float = 0.5


@dataclass
class BakeStats:
    frames: int = 0
    bones: int = 0
    keys: int = 0
    channels: int = 0
    duration: float = 0.0
    key_density: float = 0.0            # keys per channel per frame
    mean_step_deg: float = 0.0          # mean frame-to-frame rotation change
    mean_accel_deg: float = 0.0         # mean second difference (jitter proxy)
    solver: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = dict(self.__dict__)
        d.update({f"solver_{k}": v for k, v in self.solver.items()})
        d.pop("solver", None)
        return d


def resample_frames(frames: Sequence[SourceSkeleton], fps: float) -> List[SourceSkeleton]:
    """Linearly resample a take onto a uniform time grid (t = k / fps)."""
    if not frames:
        return []
    t0 = frames[0].timestamp
    t_end = frames[-1].timestamp
    n = int(math.floor((t_end - t0) * fps + 1e-6)) + 1
    out: List[SourceSkeleton] = []
    j = 0
    for k in range(n):
        t = t0 + k / fps
        while j + 1 < len(frames) - 1 and frames[j + 1].timestamp < t:
            j += 1
        a = frames[j]
        b = frames[min(j + 1, len(frames) - 1)]
        span = b.timestamp - a.timestamp
        s = 0.0 if span <= 1e-9 else min(max((t - a.timestamp) / span, 0.0), 1.0)
        sk = SourceSkeleton(timestamp=k / fps, frame_index=k)
        for name, pa in a.joints.items():
            pb = b.joints.get(name)
            ca, cb = a.conf.get(name, 0.0), b.conf.get(name, 0.0) if pb is not None else 0.0
            if pb is None:
                sk.joints[name] = pa.copy()
                sk.conf[name] = ca
            else:
                sk.joints[name] = pa.lerp(pb, s)
                sk.conf[name] = min(ca, cb) if 0.0 < s < 1.0 else (ca if s < 0.5 else cb)
        if a.root is not None and b.root is not None:
            sk.root = a.root.lerp(b.root, s)
            sk.root_conf = min(a.root_conf, b.root_conf)
        elif a.root is not None:
            sk.root, sk.root_conf = a.root.copy(), a.root_conf
        out.append(sk.derive())
    return out


def solve_take(
    frames: Sequence[SourceSkeleton],
    rig: RigModel,
    profile: TopologyProfile,
    solver_settings: Optional[SolverSettings] = None,
    bake: Optional[BakeSettings] = None,
    calibration: Optional[CalibrationData] = None,
) -> Tuple[List[SolveResult], List[SourceSkeleton], Dict[str, float]]:
    bake = bake or BakeSettings()
    conditioned = filter_sequence(frames, bake.smoothing, min_conf=bake.min_conf, max_gap=bake.gap_fill)
    grid = resample_frames(conditioned, bake.fps)
    solver = RetargetSolver(rig, profile, solver_settings, calibration)
    results = [solver.solve(sk, compute_pose=False) for sk in grid]
    return results, grid, dict(solver.stats)


def motion_metrics(results: Sequence[SolveResult]) -> Tuple[float, float]:
    """Mean per-frame rotation step and mean second difference (degrees)."""
    if len(results) < 3:
        return 0.0, 0.0
    bones = list(results[0].local_rot.keys())
    steps, accs = [], []
    for b in bones:
        qs = [r.local_rot.get(b) for r in results]
        if any(q is None for q in qs):
            continue
        ang = [quat_angle(qs[i], qs[i + 1]) for i in range(len(qs) - 1)]
        steps.extend(ang)
        accs.extend(abs(ang[i + 1] - ang[i]) for i in range(len(ang) - 1))
    if not steps:
        return 0.0, 0.0
    return (math.degrees(sum(steps) / len(steps)),
            math.degrees(sum(accs) / max(len(accs), 1)))


def write_action(arm_obj, results: Sequence[SolveResult], settings: BakeSettings):
    """Write solve results as keyframes into an Action assigned to ``arm_obj``."""
    import bpy
    from mathutils import Euler, Quaternion

    from ..utils.anim_compat import assign_action, ensure_fcurve, fill_fcurve, new_action

    if not results:
        return None, 0, 0
    action = new_action(settings.action_name, overwrite=settings.overwrite)
    assign_action(arm_obj, action)
    frames = [settings.start_frame + i for i in range(len(results))]
    bones = [b for b in results[0].local_rot.keys() if b in arm_obj.pose.bones]
    use_euler = settings.rotation_mode == "EULER"
    keys = channels = 0
    for bname in bones:
        pb = arm_obj.pose.bones[bname]
        qs = [r.local_rot.get(bname, Quat()) for r in results]
        path = f'pose.bones["{bname}"]'
        if use_euler:
            pb.rotation_mode = "XYZ"
            prev = None
            eul = []
            for q in qs:
                e = Quaternion((q.w, q.x, q.y, q.z)).to_euler("XYZ", prev) if prev else \
                    Quaternion((q.w, q.x, q.y, q.z)).to_euler("XYZ")
                eul.append(e)
                prev = e
            for i in range(3):
                fc = ensure_fcurve(action, arm_obj, f"{path}.rotation_euler", i, bname)
                fill_fcurve(fc, frames, [e[i] for e in eul], settings.interpolation)
                keys += len(frames)
                channels += 1
        else:
            pb.rotation_mode = "QUATERNION"
            for i, comp in enumerate(("w", "x", "y", "z")):
                fc = ensure_fcurve(action, arm_obj, f"{path}.rotation_quaternion", i, bname)
                fill_fcurve(fc, frames, [getattr(q, comp) for q in qs], settings.interpolation)
                keys += len(frames)
                channels += 1
    root_bones = {b for r in results for b in r.root_loc.keys()}
    for bname in root_bones:
        if bname not in arm_obj.pose.bones:
            continue
        path = f'pose.bones["{bname}"].location'
        locs = [r.root_loc.get(bname, Vec3()) for r in results]
        for i, comp in enumerate(("x", "y", "z")):
            fc = ensure_fcurve(action, arm_obj, path, i, bname)
            fill_fcurve(fc, frames, [getattr(v, comp) for v in locs], settings.interpolation)
            keys += len(frames)
            channels += 1
    try:
        action.frame_range = (frames[0], frames[-1])
        action.use_frame_range = False
    except Exception:
        pass
    return action, keys, channels


def bake_take(
    arm_obj,
    frames: Sequence[SourceSkeleton],
    profile: TopologyProfile,
    settings: BakeSettings,
    solver_settings: Optional[SolverSettings] = None,
    calibration: Optional[CalibrationData] = None,
    rig: Optional[RigModel] = None,
) -> Tuple[bool, str, object, BakeStats]:
    """Solve + write. Returns (ok, message, action, stats)."""
    stats = BakeStats()
    if not frames:
        return False, "Take has no frames", None, stats
    rig = rig or RigModel.from_blender(arm_obj)
    errs = profile.validate(rig)
    if errs:
        return False, "Topology profile invalid: " + "; ".join(errs[:3]), None, stats
    results, grid, solver_stats = solve_take(frames, rig, profile, solver_settings, settings, calibration)
    action, keys, channels = write_action(arm_obj, results, settings)
    stats.frames = len(results)
    stats.bones = len(results[0].local_rot) if results else 0
    stats.keys = keys
    stats.channels = channels
    stats.duration = grid[-1].timestamp if grid else 0.0
    stats.key_density = keys / max(channels * max(stats.frames, 1), 1)
    stats.mean_step_deg, stats.mean_accel_deg = motion_metrics(results)
    stats.solver = {k: float(v) for k, v in solver_stats.items()}
    return True, (f"Baked {stats.frames} frames x {stats.bones} bones into '{action.name}' "
                  f"({keys} keys)"), action, stats
