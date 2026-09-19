"""Real-world rig layouts captured from Blender 5.2 for pure-Python tests.

* ``rigify_metarig_spec``: Rigify "Human" metarig (face reduced to a few bones),
  rest positions as created by ``object.armature_human_metarig_add``.
* ``cesium_man_spec``: Khronos CesiumMan (CC-BY 4.0) as imported by the glTF
  importer: joint heads are anatomical but every bone tail points along +X
  (glTF has no bone tails), exercising head-based retargeting.
"""

from __future__ import annotations

from typing import List


def _b(name, head, tail, parent=None, connect=False, roll=0.0):
    return {"name": name, "head": head, "tail": tail, "parent": parent,
            "connect": connect, "roll": roll}


def _mirror(bones: List[dict]) -> List[dict]:
    out = []
    for b in bones:
        if not b["name"].endswith(".L"):
            continue
        nb = dict(b)
        nb["name"] = b["name"][:-2] + ".R"
        nb["head"] = (-b["head"][0], b["head"][1], b["head"][2])
        nb["tail"] = (-b["tail"][0], b["tail"][1], b["tail"][2])
        if b["parent"] and b["parent"].endswith(".L"):
            nb["parent"] = b["parent"][:-2] + ".R"
        out.append(nb)
    return out


def rigify_metarig_spec() -> List[dict]:
    core = [
        _b("spine", (0.0, 0.055, 1.010), (0.0, 0.017, 1.157)),
        _b("spine.001", (0.0, 0.017, 1.157), (0.0, 0.0, 1.293), "spine", True),
        _b("spine.002", (0.0, 0.0, 1.293), (0.0, 0.006, 1.466), "spine.001", True),
        _b("spine.003", (0.0, 0.006, 1.466), (0.0, 0.011, 1.658), "spine.002", True),
        _b("spine.004", (0.0, 0.011, 1.658), (0.0, -0.013, 1.720), "spine.003"),
        _b("spine.005", (0.0, -0.013, 1.720), (0.0, -0.025, 1.781), "spine.004", True),
        _b("spine.006", (0.0, -0.025, 1.781), (0.0, -0.025, 1.980), "spine.005", True),
        _b("face", (0.0, -0.025, 1.781), (0.0, -0.025, 1.849), "spine.006"),
        _b("nose", (0.0, -0.13, 1.84), (0.0, -0.16, 1.80), "face"),
        _b("jaw", (0.0, -0.05, 1.78), (0.0, -0.12, 1.72), "face"),
        _b("eye.L", (0.03, -0.09, 1.86), (0.03, -0.11, 1.86), "face"),
        _b("ear.L", (0.07, -0.02, 1.85), (0.08, -0.01, 1.88), "face"),
    ]
    left = [
        _b("shoulder.L", (0.018, -0.068, 1.605), (0.169, 0.021, 1.605), "spine.003"),
        _b("upper_arm.L", (0.195, 0.027, 1.585), (0.442, 0.089, 1.449), "shoulder.L"),
        _b("forearm.L", (0.442, 0.089, 1.449), (0.659, 0.049, 1.306), "upper_arm.L", True),
        _b("hand.L", (0.659, 0.049, 1.306), (0.723, 0.041, 1.258), "forearm.L", True),
        _b("palm.01.L", (0.692, 0.022, 1.288), (0.746, 0.005, 1.248), "hand.L"),
        _b("f_index.01.L", (0.746, 0.005, 1.248), (0.772, 0.001, 1.211), "palm.01.L"),
        _b("thumb.01.L", (0.670, 0.021, 1.274), (0.686, 0.002, 1.240), "palm.01.L"),
        _b("palm.02.L", (0.697, 0.039, 1.288), (0.752, 0.028, 1.249), "hand.L"),
        _b("f_middle.01.L", (0.752, 0.028, 1.249), (0.776, 0.023, 1.206), "palm.02.L"),
        _b("palm.03.L", (0.696, 0.054, 1.287), (0.754, 0.052, 1.248), "hand.L"),
        _b("f_ring.01.L", (0.754, 0.052, 1.248), (0.771, 0.050, 1.207), "palm.03.L"),
        _b("palm.04.L", (0.693, 0.070, 1.287), (0.753, 0.076, 1.243), "hand.L"),
        _b("f_pinky.01.L", (0.753, 0.076, 1.243), (0.759, 0.076, 1.216), "palm.04.L"),
        _b("breast.L", (0.118, 0.049, 1.460), (0.118, -0.091, 1.460), "spine.003"),
        _b("pelvis.L", (0.0, 0.055, 1.010), (0.111, -0.045, 1.153), "spine"),
        _b("thigh.L", (0.098, 0.012, 1.072), (0.098, -0.029, 0.537), "spine"),
        _b("shin.L", (0.098, -0.029, 0.537), (0.098, 0.016, 0.085), "thigh.L", True),
        _b("foot.L", (0.098, 0.016, 0.085), (0.098, -0.093, 0.017), "shin.L", True),
        _b("toe.L", (0.098, -0.093, 0.017), (0.098, -0.161, 0.017), "foot.L", True),
        _b("heel.02.L", (0.060, 0.046, 0.0), (0.140, 0.046, 0.0), "foot.L"),
    ]
    return core + left + _mirror(left)


def cesium_man_spec() -> List[dict]:
    X = 0.1  # all glTF-imported tails point along +X
    def b(name, head, parent=None, length=0.1):
        return _b(name, head, (head[0] + length, head[1], head[2]), parent)
    return [
        b("Skeleton_torso_joint_1", (0.005, 0.000, 0.679), None, 0.097),
        b("Skeleton_torso_joint_2", (0.005, -0.011, 0.824), "Skeleton_torso_joint_1", 0.251),
        b("torso_joint_3", (0.005, 0.004, 1.074), "Skeleton_torso_joint_2", 0.065),
        b("Skeleton_neck_joint_1", (0.005, -0.006, 1.138), "torso_joint_3", 0.052),
        b("Skeleton_neck_joint_2", (0.005, -0.008, 1.190), "Skeleton_neck_joint_1", 0.052),
        b("Skeleton_arm_joint_L__4_", (0.096, 0.004, 1.074), "torso_joint_3", 0.242),
        b("Skeleton_arm_joint_L__3_", (0.312, 0.016, 0.965), "Skeleton_arm_joint_L__4_", 0.188),
        b("Skeleton_arm_joint_L__2_", (0.455, -0.066, 0.875), "Skeleton_arm_joint_L__3_", 0.188),
        b("Skeleton_arm_joint_R", (-0.086, 0.004, 1.074), "torso_joint_3", 0.242),
        b("Skeleton_arm_joint_R__2_", (-0.301, 0.016, 0.965), "Skeleton_arm_joint_R", 0.188),
        b("Skeleton_arm_joint_R__3_", (-0.445, -0.067, 0.875), "Skeleton_arm_joint_R__2_", 0.188),
        b("leg_joint_L_1", (0.073, -0.024, 0.614), "Skeleton_torso_joint_1", 0.266),
        b("leg_joint_L_2", (0.082, -0.068, 0.352), "leg_joint_L_1", 0.276),
        b("leg_joint_L_3", (0.083, 0.005, 0.086), "leg_joint_L_2", 0.072),
        b("leg_joint_L_5", (0.085, -0.027, 0.021), "leg_joint_L_3", 0.072),
        b("leg_joint_R_1", (-0.063, -0.024, 0.614), "Skeleton_torso_joint_1", 0.266),
        b("leg_joint_R_2", (-0.072, -0.068, 0.352), "leg_joint_R_1", 0.276),
        b("leg_joint_R_3", (-0.073, 0.005, 0.086), "leg_joint_R_2", 0.072),
        b("leg_joint_R_5", (-0.075, -0.027, 0.021), "leg_joint_R_3", 0.072),
    ]
