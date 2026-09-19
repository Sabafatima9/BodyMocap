"""Procedural humanoid rig specifications (pure data, no bpy).

Each spec is a list of bone dicts (name, parent, head, tail, roll, connect) in
Blender edit-bone convention (metres, +Z up, character facing -Y).  The same
specs build real armatures (``bodymocap.assets.build``) and pure-Python
:class:`~bodymocap.retarget.rig.RigModel` instances for unit tests.

* ``RIG_A``  standard topology, T-pose: UpperArm -> Forearm -> Hand.
* ``RIG_B``  4-segment limbs (UpperArm.01/.02, Forearm.01/.02, Thigh.01/.02,
  Shin.01/.02) and a 4-bone spine, A-pose, different bone rolls.  Its joint
  layout matches RIG_A exactly (only segmentation / rest pose / rolls differ).
* ``RIG_C``  generic 3-bone limbs with no anatomical names or joint at mid
  length (forces CONTINUOUS arc-length mapping) and a 3-bone spine.
* ``RIG_MIXAMO`` / ``RIG_UNREAL``  naming-convention variants for detection.
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

V = Tuple[float, float, float]

# --- shared body layout (metres) -------------------------------------------
HIPS_HEAD: V = (0.0, 0.0, 0.98)
SPINE_PTS: List[V] = [(0.0, 0.0, 1.08), (0.0, 0.01, 1.27), (0.0, 0.0, 1.45)]
NECK_HEAD: V = (0.0, 0.0, 1.45)
HEAD_HEAD: V = (0.0, -0.01, 1.56)
HEAD_TAIL: V = (0.0, -0.01, 1.76)
CLAV_HEAD_X = 0.02
SHOULDER: V = (0.18, 0.01, 1.42)
UPPER_ARM_LEN = 0.29
FOREARM_LEN = 0.26
HAND_LEN = 0.09
HIP_JOINT: V = (0.10, 0.0, 0.95)
KNEE: V = (0.10, -0.01, 0.51)
ANKLE: V = (0.10, 0.02, 0.08)
BALL: V = (0.10, -0.10, 0.02)
TOE_TIP: V = (0.10, -0.17, 0.02)


def _add(a: V, b: V) -> V:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: V, s: float) -> V:
    return (a[0] * s, a[1] * s, a[2] * s)


def _lerp(a: V, b: V, t: float) -> V:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


def _mirror(p: V) -> V:
    return (-p[0], p[1], p[2])


def _bone(name, head, tail, parent=None, roll=0.0, connect=False) -> dict:
    return {"name": name, "head": tuple(head), "tail": tuple(tail), "parent": parent,
            "roll": roll, "connect": connect}


def _polyline_split(pts: Sequence[V], n: int) -> List[V]:
    """n+1 points at equal arc length along a polyline."""
    lens = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    total = sum(lens)
    out = []
    for k in range(n + 1):
        target = total * k / n
        acc = 0.0
        for i, L in enumerate(lens):
            if acc + L >= target - 1e-12 or i == len(lens) - 1:
                t = 0.0 if L < 1e-12 else (target - acc) / L
                out.append(_lerp(pts[i], pts[i + 1], min(max(t, 0.0), 1.0)))
                break
            acc += L
    return out


def _arm_dir(pose: str) -> V:
    if pose == "A":
        a = math.radians(45.0)
        return (math.cos(a), 0.0, -math.sin(a))
    return (1.0, 0.0, 0.0)


def _arm_points(pose: str) -> Tuple[V, V, V, V]:
    d = _arm_dir(pose)
    s = SHOULDER
    e = _add(s, _scale(d, UPPER_ARM_LEN))
    w = _add(e, _scale(d, FOREARM_LEN))
    h = _add(w, _scale(d, HAND_LEN))
    return s, e, w, h


def _mirror_side(bones: List[dict]) -> List[dict]:
    """Create .R bones from .L bones (names ending with .L)."""
    out = []
    for b in bones:
        if not b["name"].endswith(".L"):
            continue
        nb = dict(b)
        nb["name"] = b["name"][:-2] + ".R"
        nb["head"] = _mirror(b["head"])
        nb["tail"] = _mirror(b["tail"])
        if b["parent"] and b["parent"].endswith(".L"):
            nb["parent"] = b["parent"][:-2] + ".R"
        nb["roll"] = -b["roll"]
        out.append(nb)
    return out


def rig_a_spec() -> List[dict]:
    """Standard 2-segment limbs, T-pose."""
    s, e, w, h = _arm_points("T")
    bones = [
        _bone("Hips", HIPS_HEAD, SPINE_PTS[0]),
        _bone("Spine", SPINE_PTS[0], SPINE_PTS[1], "Hips", connect=True),
        _bone("Chest", SPINE_PTS[1], SPINE_PTS[2], "Spine", connect=True),
        _bone("Neck", NECK_HEAD, HEAD_HEAD, "Chest", connect=True),
        _bone("Head", HEAD_HEAD, HEAD_TAIL, "Neck", connect=True),
    ]
    left = [
        _bone("Shoulder.L", (CLAV_HEAD_X, 0.0, s[2]), (s[0] - 0.01, s[1], s[2]), "Chest"),
        _bone("UpperArm.L", s, e, "Shoulder.L"),
        _bone("Forearm.L", e, w, "UpperArm.L", connect=True),
        _bone("Hand.L", w, h, "Forearm.L", connect=True),
        _bone("Thigh.L", HIP_JOINT, KNEE, "Hips"),
        _bone("Shin.L", KNEE, ANKLE, "Thigh.L", connect=True),
        _bone("Foot.L", ANKLE, BALL, "Shin.L", connect=True),
        _bone("Toe.L", BALL, TOE_TIP, "Foot.L", connect=True),
    ]
    return bones + left + _mirror_side(left)


def rig_b_spec() -> List[dict]:
    """4-segment limbs + 4-bone spine, A-pose, non-zero rolls."""
    s, e, w, h = _arm_points("A")
    r_limb = math.radians(90.0)
    r_spine = math.radians(30.0)
    sp = _polyline_split(SPINE_PTS, 4)
    bones = [_bone("Hips", HIPS_HEAD, SPINE_PTS[0], roll=r_spine)]
    parent = "Hips"
    for i in range(4):
        name = f"Spine.{i + 1:02d}"
        bones.append(_bone(name, sp[i], sp[i + 1], parent, roll=r_spine, connect=True))
        parent = name
    bones += [
        _bone("Neck", NECK_HEAD, HEAD_HEAD, "Spine.04", roll=r_spine, connect=True),
        _bone("Head", HEAD_HEAD, HEAD_TAIL, "Neck", roll=r_spine, connect=True),
    ]
    ua_mid = _lerp(s, e, 0.5)
    fa_mid = _lerp(e, w, 0.5)
    th_mid = _lerp(HIP_JOINT, KNEE, 0.5)
    sh_mid = _lerp(KNEE, ANKLE, 0.5)
    left = [
        _bone("Shoulder.L", (CLAV_HEAD_X, 0.0, s[2]), (s[0] - 0.01, s[1], s[2]), "Spine.04", roll=r_limb),
        _bone("UpperArm.01.L", s, ua_mid, "Shoulder.L", roll=r_limb),
        _bone("UpperArm.02.L", ua_mid, e, "UpperArm.01.L", roll=r_limb, connect=True),
        _bone("Forearm.01.L", e, fa_mid, "UpperArm.02.L", roll=r_limb, connect=True),
        _bone("Forearm.02.L", fa_mid, w, "Forearm.01.L", roll=r_limb, connect=True),
        _bone("Hand.L", w, h, "Forearm.02.L", roll=r_limb, connect=True),
        _bone("Thigh.01.L", HIP_JOINT, th_mid, "Hips", roll=r_limb),
        _bone("Thigh.02.L", th_mid, KNEE, "Thigh.01.L", roll=r_limb, connect=True),
        _bone("Shin.01.L", KNEE, sh_mid, "Thigh.02.L", roll=r_limb, connect=True),
        _bone("Shin.02.L", sh_mid, ANKLE, "Shin.01.L", roll=r_limb, connect=True),
        _bone("Foot.L", ANKLE, BALL, "Shin.02.L", roll=r_limb, connect=True),
        _bone("Toe.L", BALL, TOE_TIP, "Foot.L", roll=r_limb, connect=True),
    ]
    return bones + left + _mirror_side(left)


def rig_c_spec() -> List[dict]:
    """Generic 3-bone limbs (no anatomical joint), 3-bone spine, T-pose."""
    s, e, w, h = _arm_points("T")
    sp = _polyline_split(SPINE_PTS, 3)
    bones = [_bone("Pelvis", HIPS_HEAD, SPINE_PTS[0])]
    parent = "Pelvis"
    for i in range(3):
        name = f"Torso.{i + 1}"
        bones.append(_bone(name, sp[i], sp[i + 1], parent, connect=True))
        parent = name
    bones += [
        _bone("Neck", NECK_HEAD, HEAD_HEAD, "Torso.3", connect=True),
        _bone("Head", HEAD_HEAD, HEAD_TAIL, "Neck", connect=True),
    ]
    arm = _polyline_split([s, e, w], 3)
    leg = _polyline_split([HIP_JOINT, KNEE, ANKLE], 3)
    left = [
        _bone("Arm.1.L", arm[0], arm[1], "Torso.3"),
        _bone("Arm.2.L", arm[1], arm[2], "Arm.1.L", connect=True),
        _bone("Arm.3.L", arm[2], arm[3], "Arm.2.L", connect=True),
        _bone("Hand.L", w, h, "Arm.3.L", connect=True),
        _bone("Leg.1.L", leg[0], leg[1], "Pelvis"),
        _bone("Leg.2.L", leg[1], leg[2], "Leg.1.L", connect=True),
        _bone("Leg.3.L", leg[2], leg[3], "Leg.2.L", connect=True),
        _bone("Foot.L", ANKLE, BALL, "Leg.3.L", connect=True),
        _bone("Toe.L", BALL, TOE_TIP, "Foot.L", connect=True),
    ]
    return bones + left + _mirror_side(left)


def _rename(spec: List[dict], mapping: Dict[str, str]) -> List[dict]:
    out = []
    for b in spec:
        nb = dict(b)
        nb["name"] = mapping.get(b["name"], b["name"])
        if b["parent"]:
            nb["parent"] = mapping.get(b["parent"], b["parent"])
        out.append(nb)
    return out


def rig_mixamo_spec() -> List[dict]:
    """RIG_A layout with Mixamo names (+ a 3-bone Mixamo spine)."""
    base = rig_a_spec()
    m = {"Hips": "mixamorig:Hips", "Spine": "mixamorig:Spine", "Chest": "mixamorig:Spine1",
         "Neck": "mixamorig:Neck", "Head": "mixamorig:Head"}
    for side, word in (("L", "Left"), ("R", "Right")):
        m.update({
            f"Shoulder.{side}": f"mixamorig:{word}Shoulder",
            f"UpperArm.{side}": f"mixamorig:{word}Arm",
            f"Forearm.{side}": f"mixamorig:{word}ForeArm",
            f"Hand.{side}": f"mixamorig:{word}Hand",
            f"Thigh.{side}": f"mixamorig:{word}UpLeg",
            f"Shin.{side}": f"mixamorig:{word}Leg",
            f"Foot.{side}": f"mixamorig:{word}Foot",
            f"Toe.{side}": f"mixamorig:{word}ToeBase",
        })
    spec = _rename(base, m)
    spec.append(_bone("mixamorig:HeadTop_End", HEAD_TAIL, _add(HEAD_TAIL, (0, 0, 0.05)),
                      "mixamorig:Head", connect=True))
    return spec


def rig_unreal_spec() -> List[dict]:
    """RIG_A layout with UE mannequin names, a root bone, twist children and IK bones."""
    base = rig_a_spec()
    m = {"Hips": "pelvis", "Spine": "spine_01", "Chest": "spine_02", "Neck": "neck_01",
         "Head": "head"}
    for side, sfx in (("L", "l"), ("R", "r")):
        m.update({
            f"Shoulder.{side}": f"clavicle_{sfx}", f"UpperArm.{side}": f"upperarm_{sfx}",
            f"Forearm.{side}": f"lowerarm_{sfx}", f"Hand.{side}": f"hand_{sfx}",
            f"Thigh.{side}": f"thigh_{sfx}", f"Shin.{side}": f"calf_{sfx}",
            f"Foot.{side}": f"foot_{sfx}", f"Toe.{side}": f"ball_{sfx}",
        })
    spec = _rename(base, m)
    spec.insert(0, _bone("root", (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)))
    spec[1]["parent"] = "root"
    extra = []
    by = {b["name"]: b for b in spec}
    for sfx in ("l", "r"):
        ua, la = by[f"upperarm_{sfx}"], by[f"lowerarm_{sfx}"]
        extra.append(_bone(f"upperarm_twist_01_{sfx}", _lerp(ua["head"], ua["tail"], 0.5),
                           ua["tail"], f"upperarm_{sfx}"))
        extra.append(_bone(f"lowerarm_twist_01_{sfx}", _lerp(la["head"], la["tail"], 0.5),
                           la["tail"], f"lowerarm_{sfx}"))
        extra.append(_bone(f"ik_foot_{sfx}", by[f"foot_{sfx}"]["head"],
                           _add(by[f"foot_{sfx}"]["head"], (0, 0, 0.1)), "ik_foot_root"))
    extra.insert(0, _bone("ik_foot_root", (0, 0, 0), (0, 0, 0.1), "root"))
    return spec + extra


RIG_SPECS = {
    "RigA_Standard": rig_a_spec,
    "RigB_Segmented": rig_b_spec,
    "RigC_Continuous": rig_c_spec,
    "Rig_MixamoNames": rig_mixamo_spec,
    "Rig_UnrealNames": rig_unreal_spec,
}
