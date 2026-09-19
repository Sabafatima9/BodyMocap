"""Topology-agnostic retargeting solver: SourceSkeleton -> bone rotations.

Every driven bone gets an *armature-space delta* ``D`` (posed = D @ rest);
local pose rotations are derived afterwards, so the result is independent of
bone rolls, rest-pose style (T/A) and how many bones make up each chain.

Chains
------
* **Pelvis / chest / head** receive absolute frames built from landmarks.
* **Spine & neck** distribute the relative rotation between their end frames
  along the chain proportionally to bone length (theta_i = theta * L_i / sum L),
  then an optional FABRIK pass moves the chest so the shoulder centre lands on
  the topology-invariant target (hips + rest chord along the source chord).
* **Limbs** in SEGMENTED mode rotate each anatomical group rigidly onto its
  source segment using an aim + pole-vector frame (hinge-consistent elbows and
  knees); forearm/shin twist towards the hand/foot frame is distributed along
  the lower group by arc length.  CONTINUOUS limbs (no anatomical joint) are
  fitted to the arc-length parameterised source polyline with FABRIK.
* **Anti-flip**: pole vectors blend towards a transported fallback when a limb
  is nearly straight, sudden pole reversals are rejected, and knees that
  appear to bend backwards (monocular depth ambiguity) are mirrored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.math3d import (
    Y_AXIS,
    Z_AXIS,
    X_AXIS,
    angle_between,
    clamp,
    cumulative_weights,
    delta_between_frames,
    fabrik,
    frame_aim_up,
    frame_left_up,
    polyline_point_at,
    quat_align_hemisphere,
    quat_angle,
    quat_conjugate,
    quat_from_axis_angle,
    quat_from_two_vectors,
    quat_identity,
    quat_normalize,
    quat_pow,
    quat_twist_angle,
    reject,
    signed_angle_about,
    smoothstep,
)
from ..core.skeleton import SourceSkeleton
from ..core.types import CalibrationData, Quat, Vec3
from .proportional import cumulative_rotations
from .rig import PoseState, RigModel
from .topology import CONTINUOUS, SEGMENTED, LimbChain, TopologyProfile

_SOURCE_JOINTS = {
    "arm": ("shoulder", "elbow", "wrist"),
    "leg": ("hip", "knee", "ankle"),
}


@dataclass
class SolverSettings:
    min_conf: float = 0.5
    pole_blend_lo: float = math.radians(6.0)
    pole_blend_hi: float = math.radians(25.0)
    flip_guard: float = math.radians(35.0)
    pole_rate_straight: float = math.radians(240.0)  # rad/s allowed when the limb is straight
    pole_rate_bent: float = math.radians(720.0)      # rad/s allowed when clearly bent
    knee_forward: bool = True
    twist_share: float = 1.0
    twist_limit: float = math.radians(110.0)
    drive_hands: bool = True
    drive_feet: bool = True
    drive_head: bool = True
    spine_position_match: bool = True
    pelvis_tilt_share: float = 0.5
    head_pitch_offset: float = math.radians(15.0)
    root_motion: bool = False
    root_scale: float = 0.0          # 0 = automatic (rig leg length / subject leg length)
    root_vertical: bool = True
    hold_frames: int = 12            # frames to hold a chain when its joints are lost


@dataclass
class SolveResult:
    local_rot: Dict[str, Quat] = field(default_factory=dict)
    root_loc: Dict[str, Vec3] = field(default_factory=dict)
    pose: Optional[PoseState] = None
    diagnostics: Dict[str, object] = field(default_factory=dict)


def _snap_axis(v: Vec3, tol: float = math.radians(20.0)) -> Vec3:
    """Snap a direction to the nearest principal axis when within ``tol``."""
    n = v.normalized()
    if n.length_squared() < 0.5:
        return Z_AXIS
    best = None
    best_dot = -2.0
    for ax in (X_AXIS, -X_AXIS, Y_AXIS, -Y_AXIS, Z_AXIS, -Z_AXIS):
        d = n.dot(ax)
        if d > best_dot:
            best, best_dot = ax, d
    if best is not None and math.acos(clamp(best_dot, -1.0, 1.0)) < tol:
        return best
    return n


def palm_normal(side: str, wrist: Vec3, index: Vec3, pinky: Vec3) -> Vec3:
    """Direction the palm faces (same convention for source and rig)."""
    a = pinky - wrist
    b = index - wrist
    n = a.cross(b) if side == "L" else b.cross(a)
    return n.normalized()


class _LimbRest:
    def __init__(self):
        self.S0 = Vec3()
        self.E0 = Vec3()
        self.W0 = Vec3()
        self.pole0 = Vec3()
        self.frame_upper0: Optional[Quat] = None
        self.frame_lower0: Optional[Quat] = None
        self.lower_cum: List[float] = []
        self.upper_cum: List[float] = []
        self.end_frame0: Optional[Quat] = None
        self.end_aim0: Optional[Vec3] = None
        # continuous
        self.chain: List[str] = []
        self.joints0: List[Vec3] = []
        self.lengths: List[float] = []
        self.cum: List[float] = []
        self.aims0: List[Vec3] = []
        self.path_bones: List[str] = []
        self.leg_length = 0.0


class RetargetSolver:
    def __init__(
        self,
        rig: RigModel,
        profile: TopologyProfile,
        settings: Optional[SolverSettings] = None,
        calibration: Optional[CalibrationData] = None,
    ):
        self.rig = rig
        self.profile = profile
        self.s = settings or SolverSettings()
        self.cal = calibration
        self._desc = {n: set(rig.descendants(n)) for n in rig.order}
        self._end_X: Dict[str, Quat] = {}
        self._head_X: Optional[Quat] = None
        self._limb_base: Dict[str, Quat] = {}
        # local rotations that realise the calibrated neutral pose (same world
        # pose on every topology, unlike the raw rest pose of each rig)
        self._neutral_local: Dict[str, Quat] = {}
        self._prepare_rest()
        self.reset()
        if calibration is not None and calibration.valid and calibration.neutral is not None:
            self.calibrate_from(calibration.neutral)

    # ------------------------------------------------------------------
    # Rest analysis
    # ------------------------------------------------------------------
    def _h0(self, n: str) -> Vec3:
        return self.rig.bones[n].head

    def _next_joint0(self, bones: Sequence[str], i: int, end: Optional[str]) -> Vec3:
        """Rest position of the joint after bones[i] (head of the next bone)."""
        if i + 1 < len(bones):
            return self._h0(bones[i + 1])
        if end is not None:
            return self._h0(end)
        return self.rig.bones[bones[i]].tail

    def _prepare_rest(self) -> None:
        rig, prof = self.rig, self.profile
        self.valid = prof.hips is not None and prof.hips in rig.bones
        limbs = prof.limbs
        arm_l, arm_r = limbs.get("arm_L"), limbs.get("arm_R")
        leg_l, leg_r = limbs.get("leg_L"), limbs.get("leg_R")

        def first(lc: Optional[LimbChain]) -> Optional[Vec3]:
            if lc is None:
                return None
            b = (lc.upper or lc.lower or [None])[0]
            return self._h0(b) if b else None

        hip_l, hip_r = first(leg_l), first(leg_r)
        sh_l, sh_r = first(arm_l), first(arm_r)
        hips_head = self._h0(prof.hips) if self.valid else Vec3()
        self.hm0 = hip_l.lerp(hip_r, 0.5) if (hip_l and hip_r) else hips_head
        chest = prof.chest
        if sh_l and sh_r:
            self.sm0 = sh_l.lerp(sh_r, 0.5)
        elif chest:
            self.sm0 = self.rig.bones[chest].tail
        else:
            self.sm0 = hips_head + Vec3(0, 0, 0.5)

        if hip_l and hip_r:
            left0 = hip_l - hip_r
        elif sh_l and sh_r:
            left0 = sh_l - sh_r
        else:
            left0 = X_AXIS
        self.chord0 = self.sm0 - self.hm0
        if self.chord0.length() < 1e-6:
            self.chord0 = Z_AXIS * 0.5
        # Vertical: feet -> head when available (rest poses stand upright),
        # snapped to the armature's principal axes like the lateral axis.
        up_raw = self.chord0
        ankles = [self._h0(lc.end) for lc in (leg_l, leg_r) if lc is not None and lc.end]
        if ankles and prof.head and prof.head in rig.bones:
            feet = ankles[0] if len(ankles) == 1 else ankles[0].lerp(ankles[1], 0.5)
            up_raw = self._h0(prof.head) - feet
        up_axis = _snap_axis(up_raw)
        left_axis = _snap_axis(reject(left0, up_axis))
        body = frame_left_up(left_axis, up_axis) or frame_left_up(left0, up_raw) or quat_identity()
        # Capture (canonical) -> rig armature space.
        self.M = body
        self.left0 = body.rotate(X_AXIS)
        self.back0 = body.rotate(Y_AXIS)
        self.up0 = body.rotate(Z_AXIS)
        w = clamp(self.s.pelvis_tilt_share, 0.0, 1.0)
        self.pelvis0 = frame_left_up(left0, self.up0 * (1.0 - w) + self.chord0.normalized() * w) or body
        chest_left = (sh_l - sh_r) if (sh_l and sh_r) else left0
        self.chest0 = frame_left_up(chest_left, self.chord0) or body
        self.head0 = body

        # spine lengths (anatomical: head to next head, chest to shoulder centre)
        sp = prof.spine
        self.spine_len = []
        for i, n in enumerate(sp):
            nxt = self._h0(sp[i + 1]) if i + 1 < len(sp) else self.sm0
            L = (nxt - self._h0(n)).length()
            self.spine_len.append(max(L, 1e-4))
        self.spine_cum = cumulative_weights(self.spine_len) if sp else []
        nk = list(prof.neck) + ([prof.head] if prof.head else [])
        self.neck_len = []
        for i, n in enumerate(nk):
            nxt = self._h0(nk[i + 1]) if i + 1 < len(nk) else self.rig.bones[n].tail
            self.neck_len.append(max((nxt - self._h0(n)).length(), 1e-4))
        self.neck_cum = cumulative_weights(self.neck_len) if nk else []

        self.limb_rest: Dict[str, _LimbRest] = {}
        for key, lc in limbs.items():
            self.limb_rest[key] = self._limb_rest(lc)

        # bones we always write (identity if not driven) so undriven helpers on
        # the chain paths can't corrupt world-space results.
        self.written: List[str] = []
        if self.valid:
            self.written.append(prof.hips)
            self.written += list(prof.spine) + list(prof.neck)
            if prof.head:
                self.written.append(prof.head)
            for lc in limbs.values():
                self.written += lc.root + lc.upper + lc.lower
                if lc.end:
                    self.written.append(lc.end)
                self.written += lc.extra
            chest_top = prof.chest
            if prof.head and chest_top and rig.is_ancestor(chest_top, prof.head):
                for n in rig.path(chest_top, prof.head) or []:
                    if n not in self.written:
                        self.written.append(n)
        seen = set()
        self.written = [n for n in self.written if n in rig.bones and not (n in seen or seen.add(n))]

    def _limb_rest(self, lc: LimbChain) -> _LimbRest:
        rig = self.rig
        r = _LimbRest()
        chain = list(lc.upper) + list(lc.lower)
        r.chain = chain
        end = lc.end
        r.S0 = self._h0(chain[0])
        if lc.mode == SEGMENTED and lc.lower:
            r.E0 = self._h0(lc.lower[0])
        r.W0 = self._h0(end) if end else rig.bones[chain[-1]].tail
        if lc.mode != SEGMENTED or not lc.lower:
            # continuous: elbow estimate at the arc-length midpoint
            joints = [self._h0(n) for n in chain] + [r.W0]
            r.E0 = polyline_point_at(joints, 0.5)
        axis = (r.W0 - r.S0).normalized()
        bend_vec = reject(r.E0 - r.S0, axis)
        default = self.back0 if lc.kind == "arm" else -self.back0
        if bend_vec.length() > 0.02 * max((r.W0 - r.S0).length(), 1e-6) and \
                angle_between(r.E0 - r.S0, r.W0 - r.E0) > math.radians(3.0):
            pole = bend_vec.normalized()
        else:
            pole = reject(default, axis).normalized()
            if pole.length_squared() < 0.5:
                pole = reject(self.up0 if lc.kind == "arm" else -self.back0, axis).normalized()
        r.pole0 = pole
        r.frame_upper0 = frame_aim_up(r.E0 - r.S0, pole)
        r.frame_lower0 = frame_aim_up(r.W0 - r.E0, pole)
        lens_u = [(self._next_joint0(chain, chain.index(n), end) - self._h0(n)).length() for n in lc.upper]
        lens_l = [(self._next_joint0(chain, chain.index(n), end) - self._h0(n)).length() for n in lc.lower]
        r.upper_cum = cumulative_weights(lens_u) if lens_u else []
        r.lower_cum = cumulative_weights(lens_l) if lens_l else []
        r.joints0 = [self._h0(n) for n in chain] + [r.W0]
        r.lengths = [max((r.joints0[i + 1] - r.joints0[i]).length(), 1e-5) for i in range(len(chain))]
        r.cum = cumulative_weights(r.lengths)
        r.aims0 = [r.joints0[i + 1] - r.joints0[i] for i in range(len(chain))]
        r.leg_length = sum(r.lengths)

        # end frame (hand / foot)
        if end:
            if lc.kind == "arm":
                r.end_aim0, r.end_frame0 = self._hand_rest(lc, r)
            else:
                r.end_aim0, r.end_frame0 = self._foot_rest(lc, r)
        return r

    def _hand_rest(self, lc: LimbChain, r: _LimbRest) -> Tuple[Vec3, Optional[Quat]]:
        rig = self.rig
        hand = rig.bones[lc.end]
        fore = r.W0 - (r.E0 if lc.lower else r.S0)
        yax = hand.y_axis
        aim = yax if angle_between(yax, fore) < math.radians(45.0) else fore.normalized()
        # finger bases for a palm normal
        idx = pky = None
        from .topology import name_tags, split_side
        for d in rig.descendants(lc.end):
            base, _ = split_side(d)
            b = base.lower()
            if idx is None and "index" in b:
                idx = d
            if pky is None and ("pinky" in b or "little" in b):
                pky = d
        normal = None
        if idx and pky:
            wrist = hand.head
            # use the knuckle end of the first finger/palm bone
            pi = rig.bones[idx].tail if "palm" in idx.lower() else rig.bones[idx].head
            pp = rig.bones[pky].tail if "palm" in pky.lower() else rig.bones[pky].head
            n = palm_normal(lc.side, wrist, pi, pp)
            if n.length_squared() > 0.5:
                normal = n
                mid = pi.lerp(pp, 0.5)
                if (mid - wrist).length() > 1e-4:
                    aim = (mid - wrist).normalized()
        if normal is None:
            normal = reject(-self.up0, aim.normalized())
            if normal.length_squared() < 1e-6:
                normal = reject(-self.back0, aim.normalized())
            normal = normal.normalized()
        return aim, frame_aim_up(aim, normal)

    def _foot_rest(self, lc: LimbChain, r: _LimbRest) -> Tuple[Vec3, Optional[Quat]]:
        rig = self.rig
        foot = rig.bones[lc.end]
        ankle = foot.head
        fwd = -self.back0
        toe_pt = None
        if lc.extra:
            t = rig.bones[lc.extra[0]]
            toe_pt = t.head
            if angle_between(t.y_axis, fwd) < math.radians(60.0):
                toe_pt = t.tail
        elif angle_between(foot.y_axis, fwd) < math.radians(75.0):
            toe_pt = foot.tail
        aim = (toe_pt - ankle) if toe_pt is not None else fwd
        if aim.length() < 1e-6:
            aim = fwd
        knee = r.E0
        up = reject(knee - ankle, aim.normalized())
        if up.length_squared() < 1e-8:
            up = self.up0
        return aim.normalized(), frame_aim_up(aim, up)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._prev_pole: Dict[str, Vec3] = {}
        self._prev_local: Dict[str, Quat] = {}
        self._hold: Dict[str, int] = {}
        self._prev_t: Optional[float] = None
        self._dt = 0.0
        self._root_origin: Optional[Vec3] = (
            self.cal.root_origin.copy() if (self.cal and self.cal.root_origin is not None) else None)
        self.stats = {"frames": 0, "pole_flips_prevented": 0, "knee_inversions_fixed": 0,
                      "chains_held": 0, "spine_residual_max": 0.0}

    # ------------------------------------------------------------------
    # Lazy FK over assigned deltas
    # ------------------------------------------------------------------
    def _begin(self, root_t: Optional[Vec3]) -> None:
        self._assigned: Dict[str, Quat] = {}
        self._cD: Dict[str, Quat] = {}
        self._cH: Dict[str, Vec3] = {}
        self._root_t = root_t

    def _set(self, n: str, d: Quat) -> None:
        self._assigned[n] = quat_normalize(d)
        for k in (n, *self._desc[n]):
            self._cD.pop(k, None)
            self._cH.pop(k, None)

    def _D(self, n: str) -> Quat:
        d = self._assigned.get(n)
        if d is not None:
            return d
        d = self._cD.get(n)
        if d is not None:
            return d
        b = self.rig.bones[n]
        if b.parent is None or not b.inherit_rotation:
            d = quat_identity()
        else:
            d = self._D(b.parent)
        self._cD[n] = d
        return d

    def _H(self, n: str) -> Vec3:
        h = self._cH.get(n)
        if h is not None:
            return h
        b = self.rig.bones[n]
        if b.parent is None:
            h = b.head.copy()
        else:
            pb = self.rig.bones[b.parent]
            h = self._H(b.parent) + self._D(b.parent).rotate(b.head - pb.head)
        if n == self.profile.hips and self._root_t is not None:
            h = h + self._root_t
        self._cH[n] = h
        return h

    def _point(self, n: str, rest_point: Vec3) -> Vec3:
        return self._H(n) + self._D(n).rotate(rest_point - self.rig.bones[n].head)

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    def solve(self, sk: SourceSkeleton, compute_pose: bool = True) -> SolveResult:
        res = SolveResult()
        if not self.valid:
            return res
        s = self.s
        self.stats["frames"] += 1
        if self._prev_t is not None and sk.timestamp > self._prev_t:
            self._dt = min(sk.timestamp - self._prev_t, 0.5)
        else:
            self._dt = 0.0
        self._prev_t = sk.timestamp
        M = self.M
        mc = s.min_conf

        J = self._map_joints(sk)

        # ---------- root translation (hip centre) ----------
        root_delta: Optional[Vec3] = None
        if s.root_motion and sk.root is not None and sk.root_conf >= mc:
            if self._root_origin is None:
                self._root_origin = sk.root.copy()
            scale = s.root_scale if s.root_scale > 0.0 else self._auto_root_scale(J)
            d = M.rotate(sk.root - self._root_origin) * scale
            if not s.root_vertical:
                d = d - self.up0 * d.dot(self.up0)
            root_delta = d
        self._begin(None)

        prof = self.profile
        hips = prof.hips

        # ---------- pelvis ----------
        D_pelvis = self._D(hips)
        have_torso = all(k in J for k in ("hip_L", "hip_R", "shoulder_L", "shoulder_R"))
        if have_torso:
            hm, sm = J["hips_mid"], J["shoulders_mid"]
            torso_up = sm - hm
            world_up = self.up0  # capture +Z mapped to rig space
            w = clamp(s.pelvis_tilt_share, 0.0, 1.0)
            pel_up = world_up * (1.0 - w) + torso_up.normalized() * w
            Fp = frame_left_up(J["hip_L"] - J["hip_R"], pel_up)
            Fc = frame_left_up(J["shoulder_L"] - J["shoulder_R"], torso_up)
            if Fp is not None and Fc is not None:
                D_pelvis = delta_between_frames(Fp, self.pelvis0)
                D_chest = delta_between_frames(Fc, self.chest0)
                self._set(hips, D_pelvis)
                self._solve_spine(D_pelvis, D_chest, hm, sm)
            else:
                have_torso = False
        if not have_torso:
            self._hold_chain("torso", [hips] + list(prof.spine), res)

        if root_delta is not None:
            # translate so the rig's hip centre moves by root_delta
            hm_now = self._point(hips, self.hm0)
            target = self.hm0 + root_delta
            self._root_t = target - hm_now
            for k in (hips, *self._desc[hips]):
                self._cH.pop(k, None)

        chest = prof.chest or hips
        D_chest_now = self._D(chest)

        # ---------- neck / head ----------
        self._solve_head(J, D_chest_now, res)

        # ---------- limbs ----------
        for key, lc in prof.limbs.items():
            self._solve_limb(key, lc, J, res)

        # ---------- outputs ----------
        for n in self.written:
            b = self.rig.bones[n]
            pd = self._D(b.parent) if b.parent is not None else None
            q = self.rig.local_from_delta(n, self._D(n), pd)
            prev = self._prev_local.get(n)
            if prev is not None:
                q = quat_align_hemisphere(q, prev)
            res.local_rot[n] = q
            self._prev_local[n] = q
        if self._root_t is not None:
            b = self.rig.bones[hips]
            ref = b.rest
            if b.parent is not None and b.inherit_rotation:
                ref = self._D(b.parent) @ b.rest
            res.root_loc[hips] = quat_conjugate(ref).rotate(self._root_t)
        if compute_pose:
            res.pose = self.rig.fk(res.local_rot, res.root_loc)
        res.diagnostics = dict(self.stats)
        return res

    def solve_sequence(self, frames: Sequence[SourceSkeleton]) -> List[SolveResult]:
        return [self.solve(f) for f in frames]

    def _map_joints(self, sk: SourceSkeleton) -> Dict[str, Vec3]:
        mc = self.s.min_conf
        return {k: self.M.rotate(v) for k, v in sk.joints.items() if sk.conf.get(k, 0.0) >= mc}

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate_from(self, neutral: SourceSkeleton) -> Dict[str, float]:
        """Per-subject offsets from a neutral pose (subject in the rig's rest-like
        pose: palms/feet/head neutral; arms and legs may be T or A since limbs are
        solved absolutely).  Hands, feet and head then sit at their rest relation
        to their parent segment when the subject is neutral, whatever the
        landmark geometry.  Returns the correction angles in degrees."""
        self._end_X = {}
        self._head_X = None
        self.reset()
        self.solve(neutral, compute_pose=False)
        self._neutral_local = dict(self._prev_local)
        J = self._map_joints(neutral)
        report: Dict[str, float] = {}
        for key, lc in self.profile.limbs.items():
            r = self.limb_rest[key]
            base = self._limb_base.get(key)
            F = self._end_frame(lc, J)
            if base is not None and F is not None and r.end_frame0 is not None:
                X = quat_normalize(quat_conjugate(F) @ base @ r.end_frame0)
                self._end_X[key] = X
                report[key] = math.degrees(quat_angle(X, quat_identity()))
        F_head = self._head_frame_raw(J)
        chest = self.profile.chest or self.profile.hips
        if F_head is not None and chest:
            X = quat_normalize(quat_conjugate(F_head) @ self._D(chest) @ self.head0)
            self._head_X = X
            report["head"] = math.degrees(quat_angle(X, quat_identity()))
        self.reset()
        return report

    # ------------------------------------------------------------------
    def _auto_root_scale(self, J: Dict[str, Vec3]) -> float:
        rig_leg = 0.0
        subj = 0.0
        for side in ("L", "R"):
            r = self.limb_rest.get(f"leg_{side}")
            if r is None:
                continue
            rig_leg = max(rig_leg, r.leg_length)
            a, b, c = J.get(f"hip_{side}"), J.get(f"knee_{side}"), J.get(f"ankle_{side}")
            if a and b and c:
                subj = max(subj, (b - a).length() + (c - b).length())
        if self.cal and self.cal.scale > 0.0 and self.cal.valid:
            subj = self.cal.scale
        if rig_leg <= 0.0 or subj <= 1e-6:
            return 1.0
        return rig_leg / subj

    def _hold_chain(self, key: str, bones: Sequence[str], res: SolveResult) -> None:
        """Keep the previous local rotations of a chain whose joints are lost.

        Before the chain's first valid frame (take starts with an occluded
        limb) the calibrated neutral pose is used, so rigs with different rest
        poses / bone counts still agree in world space; only without a
        calibration does the chain fall back to each rig's own rest pose."""
        n_held = self._hold.get(key, 0) + 1
        self._hold[key] = n_held
        if n_held > self.s.hold_frames:
            return  # give up: bones fall back to rest (identity local)
        self.stats["chains_held"] += 1
        for n in bones:
            q = self._prev_local.get(n)
            if q is None:
                q = self._neutral_local.get(n)
            if q is None:
                continue
            b = self.rig.bones[n]
            pd = self._D(b.parent) if b.parent is not None else None
            self._set(n, self.rig.delta_from_local(n, q, pd))

    # ------------------------------------------------------------------
    # Spine
    # ------------------------------------------------------------------
    def _solve_spine(self, D_pelvis: Quat, D_chest: Quat, hm: Vec3, sm: Vec3) -> None:
        sp = self.profile.spine
        self._hold.pop("torso", None)
        if not sp:
            return
        # theta_i = theta * L_i / sum(L): bend + twist spread along the spine
        dq = quat_normalize(D_chest @ quat_conjugate(D_pelvis))
        for n, R in zip(sp, cumulative_rotations(dq, self.spine_len)):
            self._set(n, R @ D_pelvis)
        self._set(sp[-1], D_chest)
        if not self.s.spine_position_match or len(sp) < 2:
            return
        hips = self.profile.hips
        hm_posed = self._point(hips, self.hm0)
        chord_dir = (sm - hm).normalized()
        if chord_dir.length_squared() < 0.5:
            return
        T_sm = hm_posed + chord_dir * self.chord0.length()
        chest = sp[-1]
        target_head = T_sm - D_chest.rotate(self.sm0 - self._h0(chest))
        lower = sp[:-1]
        joints = [self._H(n) for n in lower] + [self._H(chest)]
        lengths = [(self._h0(sp[i + 1]) - self._h0(sp[i])).length() for i in range(len(lower))]
        if len(lower) == 1:
            # single bone: aim it at the target (distance can't change)
            new = [joints[0], joints[0] + (target_head - joints[0]).normalized() * lengths[0]]
        else:
            new = [j.copy() for j in joints]
            fabrik(new, lengths, target_head)
        for i, n in enumerate(lower):
            cur = self._D(n).rotate(self._h0(sp[i + 1]) - self._h0(n))
            want = new[i + 1] - new[i]
            if want.length_squared() < 1e-12:
                continue
            self._set(n, quat_from_two_vectors(cur, want) @ self._D(n))
        self._set(chest, D_chest)
        resid = (self._point(chest, self.sm0) - T_sm).length()
        self.stats["spine_residual_max"] = max(self.stats["spine_residual_max"], resid)

    # ------------------------------------------------------------------
    # Head
    # ------------------------------------------------------------------
    def _solve_head(self, J: Dict[str, Vec3], D_chest: Quat, res: SolveResult) -> None:
        prof = self.profile
        chain = list(prof.neck) + ([prof.head] if prof.head else [])
        if not chain:
            return
        if not self.s.drive_head or not all(k in J for k in ("ear_L", "ear_R", "nose")):
            for n in chain:
                self._set(n, D_chest)
            return
        F = self._head_frame_raw(J)
        if F is None:
            for n in chain:
                self._set(n, D_chest)
            return
        X = self._head_X
        if X is None:
            X = quat_from_axis_angle(X_AXIS, -self.s.head_pitch_offset)
        D_head = quat_normalize(F @ X @ quat_conjugate(self.head0))
        dq = quat_normalize(D_head @ quat_conjugate(D_chest))
        for n, R in zip(chain, cumulative_rotations(dq, self.neck_len)):
            self._set(n, R @ D_chest)
        self._set(chain[-1], D_head)

    def _head_frame_raw(self, J: Dict[str, Vec3]) -> Optional[Quat]:
        if not all(k in J for k in ("ear_L", "ear_R", "nose", "head")):
            return None
        left = J["ear_L"] - J["ear_R"]
        fwd = J["nose"] - J["head"]
        return frame_left_up(left, fwd.cross(left.normalized()))

    # ------------------------------------------------------------------
    # Limbs
    # ------------------------------------------------------------------
    def _pole(
        self,
        key: str,
        kind: str,
        s: Vec3,
        e: Vec3,
        w: Vec3,
        r: _LimbRest,
        D_parent: Quat,
    ) -> Tuple[Vec3, Vec3, float]:
        """Return (pole, possibly corrected elbow/knee, bend angle)."""
        st = self.s
        u = (w - s).normalized()
        if u.length_squared() < 0.5:
            u = (e - s).normalized()
        beta = angle_between(e - s, w - e)
        p_src = reject(e - s, u)
        p_src = p_src.normalized() if p_src.length() > 1e-5 else None

        if kind == "leg" and st.knee_forward and p_src is not None and beta > math.radians(15.0):
            fwd = -D_parent.rotate(self.back0)
            if p_src.dot(fwd) < -0.25:
                # knee seems to bend backwards: mirror across the hip-ankle line
                v = e - s
                along = u * v.dot(u)
                e = s + along * 2.0 - v
                p_src = -p_src
                self.stats["knee_inversions_fixed"] += 1
                beta = angle_between(e - s, w - e)

        prev = self._prev_pole.get(key)
        fb = None
        if prev is not None:
            t = reject(prev, u)
            if t.length() > 1e-4:
                fb = t.normalized()
        if fb is None:
            base = D_parent.rotate(r.pole0)
            rest_axis = D_parent.rotate(r.W0 - r.S0)
            base = quat_from_two_vectors(rest_axis, w - s).rotate(base)
            fb = reject(base, u)
            fb = fb.normalized() if fb.length() > 1e-6 else reject(Z_AXIS, u).normalized()

        if p_src is None:
            pole = fb
        else:
            wgt = smoothstep(st.pole_blend_lo, st.pole_blend_hi, beta)
            phi = signed_angle_about(fb, p_src, u)
            pole = quat_from_axis_angle(u, phi * wgt).rotate(fb)

        if prev is not None and fb is not None:
            # Rate-limit pole (humeral / femoral twist) motion.  Near-straight
            # limbs have an ill-conditioned bend plane, so they get the tight
            # limit; bent limbs may twist quickly but never flip in one frame.
            dt = self._dt if self._dt > 0.0 else 1.0 / 30.0
            k = smoothstep(0.3 * st.flip_guard, 2.0 * st.flip_guard, beta)
            lim_rate = st.pole_rate_straight + (st.pole_rate_bent - st.pole_rate_straight) * k
            limit = min(math.pi, lim_rate * dt)
            delta = signed_angle_about(fb, pole, u)
            if abs(delta) > limit:
                if abs(delta) > math.radians(90.0):
                    self.stats["pole_flips_prevented"] += 1
                pole = quat_from_axis_angle(u, math.copysign(limit, delta)).rotate(fb)
        self._prev_pole[key] = pole
        return pole, e, beta

    def _end_frame(self, lc: LimbChain, J: Dict[str, Vec3]) -> Optional[Quat]:
        sd = lc.side
        if lc.kind == "arm":
            if not self.s.drive_hands:
                return None
            wr, ix, pk = J.get(f"wrist_{sd}"), J.get(f"index_{sd}"), J.get(f"pinky_{sd}")
            if wr is None or ix is None or pk is None:
                return None
            aim = ix.lerp(pk, 0.5) - wr
            n = palm_normal(sd, wr, ix, pk)
            if aim.length() < 1e-5 or n.length_squared() < 0.5:
                return None
            return frame_aim_up(aim, n)
        if not self.s.drive_feet:
            return None
        an, toe, kn = J.get(f"ankle_{sd}"), J.get(f"toe_{sd}"), J.get(f"knee_{sd}")
        if an is None or toe is None or kn is None:
            return None
        aim = toe - an
        if aim.length() < 1e-5:
            return None
        up = reject(kn - an, aim.normalized())
        if up.length_squared() < 1e-10:
            return None
        return frame_aim_up(aim, up)

    def _solve_limb(self, key: str, lc: LimbChain, J: Dict[str, Vec3], res: SolveResult) -> None:
        r = self.limb_rest[key]
        names = _SOURCE_JOINTS[lc.kind]
        sd = lc.side
        s, e, w = (J.get(f"{names[0]}_{sd}"), J.get(f"{names[1]}_{sd}"), J.get(f"{names[2]}_{sd}"))
        chain_bones = list(lc.upper) + list(lc.lower) + ([lc.end] if lc.end else [])
        if s is None or e is None or w is None:
            self._hold_chain(key, chain_bones, res)
            return
        self._hold.pop(key, None)
        parent_of_limb = self.rig.bones[r.chain[0]].parent
        D_parent = self._D(parent_of_limb) if parent_of_limb else quat_identity()

        pole, e, beta = self._pole(key, lc.kind, s, e, w, r, D_parent)
        d1 = (e - s).normalized()
        d2 = (w - e).normalized()
        if d1.length_squared() < 0.5 or d2.length_squared() < 0.5:
            self._hold_chain(key, chain_bones, res)
            return

        F_end = self._end_frame(lc, J)
        D_end = None
        if F_end is not None and r.end_frame0 is not None:
            X = self._end_X.get(key)
            D_end = quat_normalize((F_end @ X if X is not None else F_end) @ quat_conjugate(r.end_frame0))

        if lc.mode == SEGMENTED and lc.lower:
            Fu = frame_aim_up(d1, pole)
            Fl = frame_aim_up(d2, pole)
            D_up = delta_between_frames(Fu, r.frame_upper0)
            D_lo = delta_between_frames(Fl, r.frame_lower0)
            self._limb_base[key] = D_lo
            for n in lc.upper:
                self._set(n, D_up)
            tau = 0.0
            if D_end is not None:
                rel = D_end @ quat_conjugate(D_lo)
                tau = clamp(quat_twist_angle(rel, d2), -self.s.twist_limit, self.s.twist_limit)
                tau *= self.s.twist_share
            for i, (n, c) in enumerate(zip(lc.lower, r.lower_cum)):
                base = D_lo
                axis = base.rotate(r.aims0[len(lc.upper) + i]).normalized()
                self._set(n, quat_from_axis_angle(axis, tau * c) @ base)
            last_D = self._D(lc.lower[-1])
        else:
            last_D = self._solve_continuous(key, lc, r, s, e, w, pole, D_end)

        if lc.end:
            if D_end is not None:
                self._set(lc.end, D_end)
            else:
                self._set(lc.end, last_D)

    def _solve_continuous(self, key, lc, r, s, e, w, pole, D_end) -> Quat:
        chain = r.chain
        base = self._H(chain[0])
        total = sum(r.lengths)
        a = (e - s).length()
        b = (w - e).length()
        ratio = a / (a + b) if (a + b) > 1e-6 else 0.5
        P0 = base
        P1 = P0 + (e - s).normalized() * (total * ratio)
        P2 = P1 + (w - e).normalized() * (total * (1.0 - ratio))
        joints = [P0] + [polyline_point_at([P0, P1, P2], c) for c in r.cum]
        fabrik(joints, r.lengths, P2)
        D_last = quat_identity()
        for i, name in enumerate(chain):
            F = frame_aim_up(joints[i + 1] - joints[i], pole)
            F0 = frame_aim_up(r.aims0[i], r.pole0)
            if F is None or F0 is None:
                continue
            D = delta_between_frames(F, F0)
            self._set(name, D)
            D_last = D
        self._limb_base[key] = D_last
        if D_end is not None:
            rel = D_end @ quat_conjugate(D_last)
            ax = (joints[-1] - joints[-2]).normalized()
            tau = clamp(quat_twist_angle(rel, ax), -self.s.twist_limit, self.s.twist_limit)
            tau *= self.s.twist_share
            for i, name in enumerate(chain):
                D = self._assigned.get(name)
                if D is None:
                    continue
                ax_i = (joints[i + 1] - joints[i]).normalized()
                self._set(name, quat_from_axis_angle(ax_i, tau * r.cum[i]) @ D)
                D_last = self._assigned[name]
        return D_last
