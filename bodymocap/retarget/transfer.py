"""Armature -> armature retargeting via pseudo-landmarks (FR-070-076).

An existing Action on a *source* armature is converted, frame by frame, into
the same canonical :class:`SourceSkeleton` a camera would produce (joint
positions + hand/foot/head reference points rigidly attached to the source
bones).  The target is then solved with the regular topology-agnostic solver,
so any source topology maps onto any target topology.  The target solver is
calibrated on the source's rest pose so rest-to-rest maps to identity.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.math3d import quat_conjugate
from ..core.skeleton import SourceSkeleton
from ..core.types import CalibrationData, Quat, Vec3
from .rig import PoseState, RigModel
from .solver import RetargetSolver, SolverSettings
from .topology import TopologyProfile, detect_topology


class LandmarkRig:
    """Derives capture-space pseudo-landmarks from a posed rig."""

    def __init__(self, rig: RigModel, profile: TopologyProfile):
        self.rig = rig
        self.profile = profile
        self.ref = RetargetSolver(rig, profile)  # reuse rest analysis (frames, M)
        self.Minv = quat_conjugate(self.ref.M)
        self.points: Dict[str, Tuple[str, Vec3]] = {}
        self._build_attachments()

    def _attach(self, name: str, bone: Optional[str], rest_point: Vec3) -> None:
        if bone:
            self.points[name] = (bone, rest_point)

    def _build_attachments(self) -> None:
        s = self.ref
        prof = self.profile
        scale = max(s.chord0.length() / 0.47, 1e-3)  # relative to the reference body
        for key, lc in prof.limbs.items():
            r = s.limb_rest[key]
            sd = lc.side
            first = (lc.upper or lc.lower)[0]
            if lc.kind == "arm":
                self._attach(f"shoulder_{sd}", first, r.S0)
                elbow_bone = lc.lower[0] if lc.lower else first
                self._attach(f"elbow_{sd}", elbow_bone, r.E0 if lc.lower else r.E0)
                self._attach(f"wrist_{sd}", lc.end or lc.bones[-1], r.W0)
                if lc.end and r.end_frame0 is not None:
                    aim = r.end_frame0.rotate(Vec3(0, 1, 0))
                    nrm = r.end_frame0.rotate(Vec3(0, 0, 1))
                    side_ax = aim.cross(nrm)  # X axis of the hand frame
                    sgn = 1.0 if sd == "L" else -1.0
                    L = 0.085 * scale
                    self._attach(f"index_{sd}", lc.end, r.W0 + aim * L - side_ax * (0.0325 * scale * sgn))
                    self._attach(f"pinky_{sd}", lc.end, r.W0 + aim * (L * 0.95) + side_ax * (0.0325 * scale * sgn))
                    self._attach(f"thumb_{sd}", lc.end, r.W0 + aim * (L * 0.6) - side_ax * (0.06 * scale * sgn))
            else:
                self._attach(f"hip_{sd}", first, r.S0)
                self._attach(f"knee_{sd}", lc.lower[0] if lc.lower else first, r.E0)
                self._attach(f"ankle_{sd}", lc.end or lc.bones[-1], r.W0)
                if lc.end and r.end_aim0 is not None:
                    toe_bone = lc.extra[0] if lc.extra else lc.end
                    self._attach(f"toe_{sd}", toe_bone, r.W0 + r.end_aim0 * (0.19 * scale))
                    self._attach(f"heel_{sd}", lc.end, r.W0 - r.end_aim0 * (0.05 * scale) - s.up0 * (0.07 * scale))
        head = prof.head
        if head:
            hb = self.rig.bones[head]
            center = hb.head + s.up0 * (0.095 * scale)
            left, back, up = s.left0, s.back0, s.up0
            import math
            drop = 0.1 * math.tan(s.s.head_pitch_offset)
            self._attach("ear_L", head, center + left * (0.075 * scale))
            self._attach("ear_R", head, center - left * (0.075 * scale))
            self._attach("nose", head, center - back * (0.10 * scale) - up * (drop * scale))
            self._attach("eye_L", head, center + left * (0.033 * scale) - back * (0.083 * scale) + up * (0.013 * scale))
            self._attach("eye_R", head, center - left * (0.033 * scale) - back * (0.083 * scale) + up * (0.013 * scale))

    def skeleton(self, pose: PoseState, timestamp: float = 0.0, index: int = 0,
                 with_root: bool = True) -> SourceSkeleton:
        sk = SourceSkeleton(timestamp=timestamp, frame_index=index)
        pts = {name: pose.point(bone, p) for name, (bone, p) in self.points.items()}
        if "hip_L" not in pts or "hip_R" not in pts:
            return sk
        hm = pts["hip_L"].lerp(pts["hip_R"], 0.5)
        for name, p in pts.items():
            sk.joints[name] = self.Minv.rotate(p - hm)
            sk.conf[name] = 1.0
        if with_root:
            sk.root = self.Minv.rotate(hm)
            sk.root_conf = 1.0
        return sk.derive()

    def rest_skeleton(self) -> SourceSkeleton:
        return self.skeleton(self.rig.rest_pose())


def transfer_frames(
    source_rig: RigModel,
    source_profile: TopologyProfile,
    source_poses: List[PoseState],
    target_rig: RigModel,
    target_profile: TopologyProfile,
    settings: Optional[SolverSettings] = None,
    fps: float = 24.0,
):
    """Retarget a list of source poses; returns (results, source skeletons)."""
    lr = LandmarkRig(source_rig, source_profile)
    rest = lr.rest_skeleton()
    cal = CalibrationData(valid=True, neutral=rest, root_origin=rest.root.copy() if rest.root else None)
    legs = [lr.ref.limb_rest[k].leg_length for k in ("leg_L", "leg_R") if k in lr.ref.limb_rest]
    cal.scale = max(legs) if legs else 0.0
    solver = RetargetSolver(target_rig, target_profile, settings, cal)
    skels = [lr.skeleton(p, timestamp=i / fps, index=i) for i, p in enumerate(source_poses)]
    return [solver.solve(sk) for sk in skels], skels


def transfer_in_blender(
    source_obj,
    target_obj,
    action_name: str,
    new_action_name: str,
    start_frame: int = 1,
    source_profile: Optional[TopologyProfile] = None,
    target_profile: Optional[TopologyProfile] = None,
    settings: Optional[SolverSettings] = None,
    rotation_mode: str = "QUATERNION",
):
    """Blender-side: sample the source Action and write a new Action on target."""
    import bpy

    from ..bake.action import BakeSettings, write_action

    action = bpy.data.actions.get(action_name) if action_name else None
    if action is None and source_obj.animation_data:
        action = source_obj.animation_data.action
    if action is None:
        return False, f"No action to retarget on {source_obj.name}", None
    src_rig = RigModel.from_blender(source_obj)
    tgt_rig = RigModel.from_blender(target_obj)
    src_prof = source_profile or detect_topology(src_rig)
    tgt_prof = target_profile or detect_topology(tgt_rig)
    errs = src_prof.validate(src_rig) + tgt_prof.validate(tgt_rig)
    if errs:
        return False, "Topology problem: " + "; ".join(errs[:3]), None

    if source_obj.animation_data is None:
        source_obj.animation_data_create()
    prev = source_obj.animation_data.action
    from ..utils.anim_compat import assign_action
    assign_action(source_obj, action)
    scene = bpy.context.scene
    f0, f1 = (int(round(v)) for v in action.frame_range)
    keep = scene.frame_current
    poses: List[PoseState] = []
    for f in range(f0, f1 + 1):
        scene.frame_set(f)
        local: Dict[str, Quat] = {}
        loc: Dict[str, Vec3] = {}
        for pb in source_obj.pose.bones:
            q = pb.matrix_basis.to_quaternion()
            local[pb.name] = Quat(q.w, q.x, q.y, q.z)
            t = pb.matrix_basis.translation
            if t.length > 1e-9:
                loc[pb.name] = Vec3(t.x, t.y, t.z)
        poses.append(src_rig.fk(local, loc))
    scene.frame_set(keep)
    if prev is not action:
        source_obj.animation_data.action = prev
    fps = scene.render.fps / scene.render.fps_base
    results, _ = transfer_frames(src_rig, src_prof, poses, tgt_rig, tgt_prof, settings, fps)
    bs = BakeSettings(action_name=new_action_name, start_frame=start_frame, overwrite=True,
                      rotation_mode=rotation_mode, fps=fps)
    act, keys, _ = write_action(target_obj, results, bs)
    return True, f"Retargeted {len(results)} frames '{action.name}' -> '{act.name}' ({keys} keys)", act
