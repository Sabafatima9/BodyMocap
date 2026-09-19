"""SourceSkeleton construction, mirroring, serialisation and gap handling."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bodymocap.core.filters import OneEuroParams
from bodymocap.core.landmarks import capture_to_mp_world, mp_world_to_capture
from bodymocap.core.skeleton import SkeletonFilter, SourceSkeleton, filter_sequence, interpolate_gaps
from bodymocap.core.types import Landmark, PoseFrame, Vec3
from bodymocap.pose.synthetic import generate_feed


class TestSkeleton(unittest.TestCase):
    def test_capture_axes_roundtrip(self):
        v = mp_world_to_capture(0.1, -0.2, 0.3)
        self.assertEqual(v.as_tuple(), (0.1, 0.3, 0.2))
        self.assertEqual(capture_to_mp_world(v), (0.1, -0.2, 0.3))

    def test_from_pose_frame_and_derived(self):
        feed = generate_feed("tpose", duration=0.1, fps=30, condition="clean")
        sk = SourceSkeleton.from_pose_frame(feed.frames[0])
        for j in ("hips_mid", "shoulders_mid", "head", "hand_L", "hand_R"):
            self.assertIn(j, sk.joints)
        self.assertLess(sk.joints["hips_mid"].length(), 1e-9)  # hip-centred
        self.assertGreater(sk.joints["shoulder_L"].x, 0.1)       # subject's left = +X

    def test_invalid_landmark_zero_conf(self):
        pf = PoseFrame(landmarks={"left_wrist": Landmark("left_wrist", Vec3(1, 0, 0), 0.9, False)})
        sk = SourceSkeleton.from_pose_frame(pf)
        self.assertEqual(sk.conf["wrist_L"], 0.0)

    def test_mirror(self):
        feed = generate_feed("wave", duration=0.5, fps=30, condition="clean")
        sk = SourceSkeleton.from_pose_frame(feed.frames[5])
        m = sk.mirrored()
        self.assertAlmostEqual(m.joints["wrist_L"].x, -sk.joints["wrist_R"].x)
        self.assertAlmostEqual(m.joints["wrist_L"].z, sk.joints["wrist_R"].z)

    def test_dict_roundtrip(self):
        feed = generate_feed("squat", duration=0.5, fps=30, condition="normal", seed=3)
        sk = SourceSkeleton.from_pose_frame(feed.frames[7])
        sk.root = Vec3(0.1, 3.0, -0.2)
        sk.root_conf = 0.8
        back = SourceSkeleton.from_dict(sk.to_dict())
        for k in ("elbow_L", "ankle_R", "nose"):
            self.assertLess((back.joints[k] - sk.joints[k]).length(), 1e-4)
        self.assertLess((back.root - sk.root).length(), 1e-4)

    def test_filter_holds_short_dropout(self):
        feed = generate_feed("subtle_idle", duration=1.0, fps=30, condition="clean")
        flt = SkeletonFilter(OneEuroParams(min_cutoff=0.0), min_conf=0.5, max_hold=0.3)
        out = []
        for i, pf in enumerate(feed.frames):
            sk = SourceSkeleton.from_pose_frame(pf)
            if 10 <= i < 14:
                sk.conf["wrist_L"] = 0.1
            out.append(flt.process(sk))
        self.assertIn("wrist_L", out[12].joints)
        self.assertGreaterEqual(out[12].conf["wrist_L"], 0.5)

    def test_interpolate_gaps(self):
        feed = generate_feed("arm_raise", duration=1.0, fps=30, condition="clean")
        frames = [SourceSkeleton.from_pose_frame(pf) for pf in feed.frames]
        truth = frames[15].joints["wrist_L"].copy()
        for i in range(13, 18):
            frames[i].conf["wrist_L"] = 0.0
            frames[i].joints["wrist_L"] = Vec3(9, 9, 9)
        filled = interpolate_gaps(frames, min_conf=0.5, max_gap=0.5)
        self.assertLess((filled[15].joints["wrist_L"] - truth).length(), 0.03)
        seq = filter_sequence(frames, OneEuroParams(min_cutoff=1.0, beta=0.3))
        self.assertEqual(len(seq), len(frames))


if __name__ == "__main__":
    unittest.main()
