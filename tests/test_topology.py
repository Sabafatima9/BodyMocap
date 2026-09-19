"""Topology detection across naming conventions and layouts, profile JSON I/O."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from bodymocap.assets.rig_specs import RIG_SPECS
from bodymocap.retarget.rig import RigModel
from bodymocap.retarget.topology import (
    CONTINUOUS,
    SEGMENTED,
    TopologyProfile,
    detect_topology,
    name_tags,
    split_side,
)
from rig_fixtures import cesium_man_spec, rigify_metarig_spec


def _rig(name):
    return RigModel.from_spec(RIG_SPECS[name](), name=name)


class TestNames(unittest.TestCase):
    def test_split_side(self):
        cases = {
            "upper_arm.L": ("upper_arm", "L"),
            "UpperArm.01.R": ("UpperArm.01", "R"),
            "mixamorig:LeftForeArm": ("ForeArm", "L"),
            "hand_r": ("hand", "R"),
            "Skeleton_arm_joint_L__4_": ("arm_joint4_", "L"),
            "leg_joint_R_1": ("leg_joint_1", "R"),
            "lShldrBend": ("ShldrBend", "L"),
            "J_Bip_L_UpperArm": ("UpperArm", "L"),
            "spine.003": ("spine.003", ""),
            "Hips": ("Hips", ""),
        }
        for name, (base, side) in cases.items():
            self.assertEqual(split_side(name), (base, side), name)

    def test_tags(self):
        self.assertIn("forearm", name_tags("mixamorig:LeftForeArm"))
        self.assertIn("hips", name_tags("pelvis"))
        self.assertNotIn("ignore", name_tags("pelvis"))          # 'vis' must not trigger
        self.assertIn("ignore", name_tags("MCH-thigh_ik.L"))
        self.assertIn("ignore", name_tags("ik_foot_l"))
        self.assertIn("finger", name_tags("f_index.01.L"))
        self.assertNotIn("face", name_tags("forearm.L"))          # 'ear' inside forearm
        self.assertIn("face", name_tags("ear.L"))


class TestDetection(unittest.TestCase):
    def test_rig_a(self):
        p = detect_topology(_rig("RigA_Standard"))
        self.assertEqual(p.hips, "Hips")
        self.assertEqual(p.spine, ["Spine", "Chest"])
        self.assertEqual(p.neck, ["Neck"])
        self.assertEqual(p.head, "Head")
        a = p.limbs["arm_L"]
        self.assertEqual((a.root, a.upper, a.lower, a.end), (["Shoulder.L"], ["UpperArm.L"], ["Forearm.L"], "Hand.L"))
        l = p.limbs["leg_R"]
        self.assertEqual((l.upper, l.lower, l.end, l.extra), (["Thigh.R"], ["Shin.R"], "Foot.R", ["Toe.R"]))
        self.assertEqual(p.validate(_rig("RigA_Standard")), [])

    def test_rig_b_four_segment(self):
        p = detect_topology(_rig("RigB_Segmented"))
        self.assertEqual(p.spine, ["Spine.01", "Spine.02", "Spine.03", "Spine.04"])
        a = p.limbs["arm_R"]
        self.assertEqual(a.upper, ["UpperArm.01.R", "UpperArm.02.R"])
        self.assertEqual(a.lower, ["Forearm.01.R", "Forearm.02.R"])
        self.assertEqual(a.mode, SEGMENTED)
        l = p.limbs["leg_L"]
        self.assertEqual(l.upper, ["Thigh.01.L", "Thigh.02.L"])
        self.assertEqual(l.lower, ["Shin.01.L", "Shin.02.L"])

    def test_rig_c_continuous(self):
        p = detect_topology(_rig("RigC_Continuous"))
        for k in ("arm_L", "arm_R", "leg_L", "leg_R"):
            self.assertEqual(p.limbs[k].mode, CONTINUOUS, k)
            self.assertEqual(len(p.limbs[k].upper), 3)
        self.assertEqual(p.hips, "Pelvis")

    def test_mixamo_and_unreal(self):
        p = detect_topology(_rig("Rig_MixamoNames"))
        self.assertEqual(p.hips, "mixamorig:Hips")
        self.assertEqual(p.head, "mixamorig:Head")
        self.assertEqual(p.limbs["arm_L"].root, ["mixamorig:LeftShoulder"])
        self.assertEqual(p.limbs["leg_L"].lower, ["mixamorig:LeftLeg"])
        u = detect_topology(_rig("Rig_UnrealNames"))
        self.assertEqual(u.hips, "pelvis")                       # not 'root'
        self.assertEqual(u.limbs["arm_R"].upper, ["upperarm_r"])  # twist children excluded
        self.assertEqual(u.limbs["leg_L"].end, "foot_l")          # ik_foot_l ignored

    def test_rigify_metarig(self):
        rig = RigModel.from_spec(rigify_metarig_spec(), "metarig")
        p = detect_topology(rig)
        self.assertEqual(p.hips, "spine")
        self.assertEqual(p.spine, ["spine.001", "spine.002", "spine.003"])
        self.assertEqual(p.neck, ["spine.004", "spine.005"])
        self.assertEqual(p.head, "spine.006")                    # found without a 'head' name
        a = p.limbs["arm_L"]
        self.assertEqual((a.root, a.upper, a.lower, a.end),
                         (["shoulder.L"], ["upper_arm.L"], ["forearm.L"], "hand.L"))
        self.assertEqual(p.limbs["leg_L"].extra, ["toe.L"])       # heel excluded
        self.assertEqual(p.validate(rig), [])

    def test_cesium_man_gltf(self):
        rig = RigModel.from_spec(cesium_man_spec(), "CesiumMan")
        p = detect_topology(rig)
        self.assertEqual(p.hips, "Skeleton_torso_joint_1")
        self.assertEqual(p.spine, ["Skeleton_torso_joint_2", "torso_joint_3"])
        self.assertEqual(p.head, "Skeleton_neck_joint_2")
        a = p.limbs["arm_L"]
        self.assertEqual(a.upper, ["Skeleton_arm_joint_L__4_"])
        self.assertEqual(a.lower, ["Skeleton_arm_joint_L__3_"])
        self.assertEqual(a.end, "Skeleton_arm_joint_L__2_")
        l = p.limbs["leg_R"]
        self.assertEqual((l.upper, l.lower, l.end, l.extra),
                         (["leg_joint_R_1"], ["leg_joint_R_2"], "leg_joint_R_3", ["leg_joint_R_5"]))


class TestProfileIO(unittest.TestCase):
    def test_roundtrip(self):
        p = detect_topology(_rig("RigB_Segmented"))
        d = p.to_dict()
        q = TopologyProfile.from_dict(json.loads(json.dumps(d)))
        self.assertEqual(q.to_dict(), d)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "profile.json"
            p.save(str(path))
            r = TopologyProfile.load(str(path))
            self.assertEqual(r.to_dict(), d)

    def test_validate_catches_errors(self):
        rig = _rig("RigA_Standard")
        p = detect_topology(rig)
        p.limbs["arm_L"].upper = ["Forearm.L"]
        p.limbs["arm_L"].lower = ["UpperArm.L"]
        errs = p.validate(rig)
        self.assertTrue(any("ancestor" in e for e in errs))
        p.limbs["arm_R"].upper = ["NoSuchBone"]
        self.assertTrue(any("not in armature" in e for e in p.validate(rig)))

    def test_bad_type_rejected(self):
        with self.assertRaises(ValueError):
            TopologyProfile.from_dict({"type": "something_else"})


if __name__ == "__main__":
    unittest.main()
