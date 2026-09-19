"""Retargeting solver: topology/rest/roll invariance, accuracy, anti-flip, holds."""

from __future__ import annotations

import math
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from bodymocap.assets.rig_specs import RIG_SPECS, rig_a_spec, rig_b_spec
from bodymocap.core.math3d import X_AXIS, Y_AXIS, quat_angle_deg, quat_dot, quat_from_axis_angle, quat_identity
from bodymocap.core.skeleton import SourceSkeleton
from bodymocap.core.types import Vec3
from bodymocap.pose.synthetic import CLIPS, SyntheticActor, generate_feed
from bodymocap.retarget.rig import RigModel
from bodymocap.retarget.solver import RetargetSolver, SolverSettings
from bodymocap.retarget.topology import detect_topology
from rig_fixtures import cesium_man_spec, rigify_metarig_spec

ARM_LEN = 0.29 + 0.26 + 0.09
LEG_LEN = 0.44 + 0.43

EFFECTORS = {
    "RigA_Standard": {"hand_L": ("tail", "Hand.L"), "hand_R": ("tail", "Hand.R"),
                      "wrist_L": ("head", "Hand.L"), "elbow_L": ("head", "Forearm.L"),
                      "ankle_L": ("head", "Foot.L"), "ankle_R": ("head", "Foot.R"),
                      "knee_R": ("head", "Shin.R"), "toe_L": ("tail", "Toe.L")},
    "RigB_Segmented": {"hand_L": ("tail", "Hand.L"), "hand_R": ("tail", "Hand.R"),
                       "wrist_L": ("head", "Hand.L"), "elbow_L": ("head", "Forearm.01.L"),
                       "ankle_L": ("head", "Foot.L"), "ankle_R": ("head", "Foot.R"),
                       "knee_R": ("head", "Shin.01.R"), "toe_L": ("tail", "Toe.L")},
    "RigC_Continuous": {"hand_L": ("tail", "Hand.L"), "hand_R": ("tail", "Hand.R"),
                        "wrist_L": ("head", "Hand.L"), "ankle_L": ("head", "Foot.L"),
                        "ankle_R": ("head", "Foot.R"), "toe_L": ("tail", "Toe.L")},
}


def effectors(key, pose):
    out = {}
    for name, (kind, bone) in EFFECTORS[key].items():
        out[name] = pose.head[bone] if kind == "head" else pose.tail(bone)
    return out


def build(key, spec=None):
    rig = RigModel.from_spec(spec or RIG_SPECS[key](), name=key)
    return rig, detect_topology(rig)


def run(solver, feed):
    out = []
    for pf in feed.frames:
        out.append(solver.solve(SourceSkeleton.from_pose_frame(pf)))
    return out


class TestSolverBasics(unittest.TestCase):
    def test_tpose_is_identity(self):
        rig, prof = build("RigA_Standard")
        feed = generate_feed("tpose", duration=0.2, fps=30, condition="clean")
        res = run(RetargetSolver(rig, prof), feed)[-1]
        for bone, q in res.local_rot.items():
            self.assertLess(quat_angle_deg(q, quat_identity()), 0.1, bone)

    def test_output_quaternion_continuity(self):
        rig, prof = build("RigB_Segmented")
        feed = generate_feed("punch_fast", duration=2.0, fps=30, condition="backlight", seed=4)
        prev = None
        for r in run(RetargetSolver(rig, prof), feed):
            if prev:
                for b, q in r.local_rot.items():
                    self.assertGreaterEqual(quat_dot(q, prev[b]), 0.0, b)
            prev = r.local_rot

    def test_performance(self):
        rig, prof = build("RigB_Segmented")
        feed = generate_feed("wave", duration=2.0, fps=30, condition="normal")
        sks = [SourceSkeleton.from_pose_frame(pf) for pf in feed.frames]
        s = RetargetSolver(rig, prof)
        t0 = time.perf_counter()
        for sk in sks:
            s.solve(sk, compute_pose=False)
        ms = (time.perf_counter() - t0) / len(sks) * 1000.0
        self.assertLess(ms, 20.0, f"{ms:.2f} ms per solve")


class TestTopologyInvariance(unittest.TestCase):
    """Same capture stream on rigs with different bone counts per chain."""

    def _compare(self, condition, tol_frac, seed=1):
        keys = ("RigA_Standard", "RigB_Segmented")
        worst = 0.0
        for clip in CLIPS:
            feed = generate_feed(clip, duration=2.0, fps=30, condition=condition, seed=seed)
            solvers = {k: RetargetSolver(*build(k)) for k in keys}
            for pf in feed.frames:
                sk = SourceSkeleton.from_pose_frame(pf)
                ea = effectors(keys[0], solvers[keys[0]].solve(sk).pose)
                eb = effectors(keys[1], solvers[keys[1]].solve(sk).pose)
                for name in ea:
                    L = ARM_LEN if name.startswith(("hand", "wrist", "elbow")) else LEG_LEN
                    worst = max(worst, (ea[name] - eb[name]).length() / L)
        self.assertLess(worst, tol_frac, f"{condition}: worst normalised error {worst:.4%}")
        return worst

    def test_clean(self):
        self._compare("clean", 0.005)

    def test_noisy_conditions(self):
        for cond in ("normal", "low_contrast", "backlight", "extreme_key", "low_light"):
            self._compare(cond, 0.05)

    def test_continuous_chain_reaches_end(self):
        feed = generate_feed("squat", duration=3.0, fps=30, condition="clean")
        sa, sc = RetargetSolver(*build("RigA_Standard")), RetargetSolver(*build("RigC_Continuous"))
        for pf in feed.frames:
            sk = SourceSkeleton.from_pose_frame(pf)
            ea = effectors("RigA_Standard", sa.solve(sk).pose)
            ec = effectors("RigC_Continuous", sc.solve(sk).pose)
            for name in ("wrist_L", "ankle_L", "ankle_R"):
                self.assertLess((ea[name] - ec[name]).length(), 0.004, name)


class TestRestInvariance(unittest.TestCase):
    def _positions(self, spec, clip="reach_cross"):
        rig, prof = build("x", spec)
        feed = generate_feed(clip, duration=1.5, fps=20, condition="clean")
        res = run(RetargetSolver(rig, prof), feed)
        return rig, [r.pose for r in res]

    def test_bone_roll_invariance(self):
        rng = random.Random(7)
        spec = rig_a_spec()
        rolled = [dict(b, roll=rng.uniform(-math.pi, math.pi)) for b in spec]
        _, pa = self._positions(spec)
        _, pb = self._positions(rolled)
        for a, b in zip(pa, pb):
            for n in ("Hand.L", "Hand.R", "Foot.L", "Head", "Forearm.R"):
                self.assertLess((a.head[n] - b.head[n]).length(), 1e-6, n)
                self.assertLess((a.tail(n) - b.tail(n)).length(), 1e-6, n)

    def test_a_pose_vs_t_pose_rest(self):
        # RIG_B is authored in A-pose; build a T-pose copy with identical joints
        spec_a = rig_b_spec()
        _, pa = self._positions(spec_a)
        t_spec = []
        import bodymocap.assets.rig_specs as rs
        orig = rs._arm_dir
        try:
            rs._arm_dir = lambda pose: (1.0, 0.0, 0.0)
            t_spec = rs.rig_b_spec()
        finally:
            rs._arm_dir = orig
        _, pt = self._positions(t_spec)
        for a, b in zip(pa, pt):
            for n in ("Hand.L", "Forearm.01.R", "Foot.R"):
                self.assertLess((a.head[n] - b.head[n]).length(), 1e-6, n)

    def test_rotated_armature_space(self):
        # A Y-up armature (e.g. an un-applied glTF import) must give the same
        # world result once its object transform is taken into account.
        rot = quat_from_axis_angle(X_AXIS, -math.pi / 2)  # Z-up world -> Y-up armature
        spec = rig_a_spec()
        rspec = []
        for b in spec:
            nb = dict(b)
            nb["head"] = rot.rotate(Vec3(*b["head"])).as_tuple()
            nb["tail"] = rot.rotate(Vec3(*b["tail"])).as_tuple()
            rspec.append(nb)
        _, pa = self._positions(spec)
        _, pr = self._positions(rspec)
        inv = rot.conjugated()
        for a, r in zip(pa, pr):
            for n in ("Hand.L", "Foot.R", "Head"):
                self.assertLess((a.head[n] - inv.rotate(r.head[n])).length(), 1e-6, n)


class TestAccuracy(unittest.TestCase):
    def test_ground_truth_recovery_clean(self):
        actor = SyntheticActor()
        rig, prof = build("RigA_Standard")
        for clip, fn in CLIPS.items():
            feed = generate_feed(clip, duration=2.0, fps=20, condition="clean")
            s = RetargetSolver(rig, prof)
            worst = 0.0
            for pf in feed.frames:
                pose = s.solve(SourceSkeleton.from_pose_frame(pf)).pose
                _, _, gt = actor.pose(fn(pf.timestamp))
                ea, eg = effectors("RigA_Standard", pose), effectors("RigA_Standard", gt)
                ha, hg = pose.head["Hips"], gt.head["Hips"]
                for name in ea:
                    worst = max(worst, ((ea[name] - ha) - (eg[name] - hg)).length())
            # limbs are exact; only the unobservable pelvis-tilt / spine split adds error
            self.assertLess(worst, 0.012, f"{clip}: {worst * 1000:.1f} mm")

    def test_rigify_and_gltf_rigs_follow_source_directions(self):
        feed = generate_feed("arm_raise", duration=2.0, fps=15, condition="clean")
        for spec_fn, arm in ((rigify_metarig_spec, ("upper_arm.L", "forearm.L", "hand.L")),
                             (cesium_man_spec, ("Skeleton_arm_joint_L__4_", "Skeleton_arm_joint_L__3_",
                                                "Skeleton_arm_joint_L__2_"))):
            rig = RigModel.from_spec(spec_fn(), "r")
            s = RetargetSolver(rig, detect_topology(rig))
            for pf in feed.frames:
                sk = SourceSkeleton.from_pose_frame(pf)
                pose = s.solve(sk).pose
                up = (pose.head[arm[1]] - pose.head[arm[0]]).normalized()
                lo = (pose.head[arm[2]] - pose.head[arm[1]]).normalized()
                src_up = (sk.joints["elbow_L"] - sk.joints["shoulder_L"]).normalized()
                src_lo = (sk.joints["wrist_L"] - sk.joints["elbow_L"]).normalized()
                self.assertGreater(up.dot(src_up), math.cos(math.radians(1.0)))
                self.assertGreater(lo.dot(src_lo), math.cos(math.radians(1.0)))


class TestRobustness(unittest.TestCase):
    def test_knee_inversion_is_corrected(self):
        rig, prof = build("RigA_Standard")
        s = RetargetSolver(rig, prof)
        feed = generate_feed("squat", duration=1.5, fps=30, condition="clean")
        for i, pf in enumerate(feed.frames):
            sk = SourceSkeleton.from_pose_frame(pf)
            if 15 <= i <= 30:
                # mirror the left knee behind the hip-ankle line (depth ambiguity)
                h, k, a = sk.joints["hip_L"], sk.joints["knee_L"], sk.joints["ankle_L"]
                u = (a - h).normalized()
                v = k - h
                sk.joints["knee_L"] = h + u * (2.0 * v.dot(u)) - v
            pose = s.solve(sk).pose
            if 15 <= i <= 30:
                knee = pose.head["Shin.L"]
                hip, ank = pose.head["Thigh.L"], pose.head["Foot.L"]
                u = (ank - hip).normalized()
                off = (knee - hip) - u * (knee - hip).dot(u)
                if off.length() > 0.03:
                    self.assertLess(off.y, 0.0, f"frame {i}: knee bends backwards")
        self.assertGreater(s.stats["knee_inversions_fixed"], 0)

    def test_straight_arm_pole_stability(self):
        rig, prof = build("RigB_Segmented")
        s = RetargetSolver(rig, prof)
        feed = generate_feed("straight_arm_circles", duration=4.0, fps=30, condition="low_light", seed=9)
        prev = None
        worst = 0.0
        for r in run(s, feed):
            q = r.pose.orientation("UpperArm.01.L")
            y = q.rotate(Y_AXIS)
            z = q.rotate(Vec3(0, 0, 1))
            if prev is not None:
                py, pz = prev
                # compare roll after removing the swing between frames
                from bodymocap.core.math3d import quat_from_two_vectors
                pz_t = quat_from_two_vectors(py, y).rotate(pz)
                worst = max(worst, math.degrees(math.acos(max(-1.0, min(1.0, pz_t.dot(z))))))
            prev = (y, z)
        self.assertLess(worst, 60.0, f"upper-arm roll jumped {worst:.1f} deg in one frame")

    def test_lost_joint_holds_chain(self):
        rig, prof = build("RigA_Standard")
        s = RetargetSolver(rig, prof)
        feed = generate_feed("wave", duration=1.0, fps=30, condition="clean")
        last = None
        for i, pf in enumerate(feed.frames):
            sk = SourceSkeleton.from_pose_frame(pf)
            if i >= 20:
                sk.conf["elbow_R"] = 0.0
            r = s.solve(sk)
            if i == 19:
                last = r.local_rot["Forearm.R"]
            if 20 <= i < 25:
                self.assertLess(quat_angle_deg(r.local_rot["Forearm.R"], last), 1e-6)
        self.assertGreater(s.stats["chains_held"], 0)

    def test_root_motion(self):
        rig, prof = build("RigA_Standard")
        s = RetargetSolver(rig, prof, SolverSettings(root_motion=True))
        feed = generate_feed("squat", duration=1.5, fps=30, condition="clean")
        zs = []
        for pf in feed.frames:
            sk = SourceSkeleton.from_pose_frame(pf)
            gt_hips = feed.truth[pf.frame_index]["left_hip"].lerp(feed.truth[pf.frame_index]["right_hip"], 0.5)
            sk.root = gt_hips - feed.camera.position
            sk.root_conf = 1.0
            zs.append(s.solve(sk).pose.head["Hips"].z)
        self.assertAlmostEqual(min(zs) - zs[0], -0.32, delta=0.01)

    def test_first_frame_loss_uses_calibrated_neutral(self):
        # A take whose first frames have an occluded wrist must not snap each rig
        # to its own rest pose (T-pose vs A-pose arm): with a calibration, both
        # rigs reproduce the same world pose from frame 0 (topology parity).
        from bodymocap.core.skeleton import build_calibration
        cal_feed = generate_feed("tpose", duration=0.7, fps=30, condition="clean")
        cal = build_calibration([SourceSkeleton.from_pose_frame(pf) for pf in cal_feed.frames])
        feed = generate_feed("wave", duration=0.5, fps=30, condition="clean", seed=2)
        skels = [SourceSkeleton.from_pose_frame(pf) for pf in feed.frames]
        skels[0].conf["wrist_L"] = 0.0
        ra, pa = build("RigA_Standard")
        rb, pb = build("RigB_Segmented")
        ea = RetargetSolver(ra, pa, calibration=cal).solve(skels[0]).pose
        eb = RetargetSolver(rb, pb, calibration=cal).solve(skels[0]).pose
        err = (effectors("RigA_Standard", ea)["wrist_L"]
               - effectors("RigB_Segmented", eb)["wrist_L"]).length() / ARM_LEN
        self.assertLess(err, 0.05, f"first-frame loss parity {err:.1%}")

    def test_proportional_spine_distribution(self):
        rig, prof = build("RigB_Segmented")
        s = RetargetSolver(rig, prof, SolverSettings(spine_position_match=False, pelvis_tilt_share=0.0))
        feed = generate_feed("torso_twist", duration=1.0, fps=10, condition="clean")
        r = run(s, feed)[-1]
        angles = [quat_angle_deg(r.local_rot[n], quat_identity()) for n in prof.spine]
        lengths = s.spine_len
        ratios = [a / L for a, L in zip(angles, lengths)]
        self.assertGreater(min(angles), 0.5)
        self.assertLess(max(ratios) / min(ratios), 1.02, f"angles {angles} lengths {lengths}")


if __name__ == "__main__":
    unittest.main()
