"""Build armatures from rig specs and a generic mannequin mesh (bpy).

The mannequin is made of rigidly-skinned ellipsoids placed from the rig's
topology profile (so it works for any bone count), with simple facial features
and clothing colours so pose detectors see a plausible human figure.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import bmesh
import bpy
from mathutils import Matrix, Vector

from ..core.types import Vec3
from ..retarget.rig import RigModel
from ..retarget.topology import TopologyProfile, detect_topology


def build_armature(spec: Sequence[dict], name: str, location=(0.0, 0.0, 0.0),
                   collection=None, display: str = "OCTAHEDRAL"):
    """Create an armature object from a bone spec list."""
    coll = collection or bpy.context.scene.collection
    data = bpy.data.armatures.new(name)
    data.display_type = display
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    coll.objects.link(obj)
    view_layer = bpy.context.view_layer
    prev_active = view_layer.objects.active
    for o in view_layer.objects:
        if o.select_get():
            o.select_set(False)
    view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    ebs = data.edit_bones
    for b in spec:
        eb = ebs.new(b["name"])
        eb.head = Vector(b["head"])
        eb.tail = Vector(b["tail"])
        eb.roll = float(b.get("roll", 0.0))
        eb.use_deform = bool(b.get("deform", True))
    for b in spec:
        if b.get("parent"):
            eb = ebs[b["name"]]
            eb.parent = ebs[b["parent"]]
            eb.use_connect = bool(b.get("connect", False))
    bpy.ops.object.mode_set(mode="OBJECT")
    for pb in obj.pose.bones:
        pb.rotation_mode = "QUATERNION"
    obj.select_set(False)
    if prev_active is not None:
        view_layer.objects.active = prev_active
    return obj


# ---------------------------------------------------------------------------
# Mannequin
# ---------------------------------------------------------------------------

def _material(name: str, color, roughness: float = 0.6):
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = (*color, 1.0)
            bsdf.inputs["Roughness"].default_value = roughness
        mat.diffuse_color = (*color, 1.0)
    return mat


PALETTE = {
    "skin": (0.80, 0.55, 0.43),
    "shirt": (0.18, 0.32, 0.55),
    "pants": (0.12, 0.13, 0.20),
    "shoes": (0.06, 0.05, 0.05),
    "eyes": (0.02, 0.02, 0.02),
    "hair": (0.10, 0.06, 0.04),
}


def _frame(axis: Vector, up_hint: Vector) -> Matrix:
    z = axis.normalized()
    x = up_hint - z * up_hint.dot(z)
    if x.length < 1e-6:
        x = Vector((1, 0, 0)) - z * z.x
        if x.length < 1e-6:
            x = Vector((0, 1, 0))
    x.normalize()
    y = z.cross(x)
    m = Matrix((x, y, z)).transposed()
    return m.to_4x4()


class _MeshBuilder:
    def __init__(self, rig_obj):
        self.rig_obj = rig_obj
        self.bm = bmesh.new()
        self.deform = self.bm.verts.layers.deform.verify()
        self.groups: List[str] = []
        self.materials: List[str] = []

    def _group(self, name: str) -> int:
        if name not in self.groups:
            self.groups.append(name)
        return self.groups.index(name)

    def _mat(self, key: str) -> int:
        if key not in self.materials:
            self.materials.append(key)
        return self.materials.index(key)

    def ellipsoid(self, bone: str, center: Vector, axis: Vector, half_len: float, r_a: float,
                  r_b: float, up_hint: Vector, mat: str, segs: int = 18, rings: int = 10):
        rot = _frame(axis, up_hint)
        m = Matrix.Translation(center) @ rot @ Matrix.Diagonal((r_a, r_b, half_len, 1.0))
        ret = bmesh.ops.create_uvsphere(self.bm, u_segments=segs, v_segments=rings, radius=1.0,
                                        matrix=m)
        gi = self._group(bone)
        mi = self._mat(mat)
        for v in ret["verts"]:
            v[self.deform][gi] = 1.0
        faces = {f for v in ret["verts"] for f in v.link_faces}
        for f in faces:
            f.material_index = mi
            f.smooth = True

    def segment(self, bone: str, a: Vector, b: Vector, r: float, mat: str, up_hint: Vector,
                flatten: float = 1.0, overlap: float = 1.08):
        axis = b - a
        L = axis.length
        if L < 1e-5:
            return
        self.ellipsoid(bone, (a + b) * 0.5, axis, L * 0.5 * overlap, r, r * flatten, up_hint, mat)

    def build(self, name: str):
        me = bpy.data.meshes.new(name)
        self.bm.to_mesh(me)
        self.bm.free()
        obj = bpy.data.objects.new(name, me)
        for key in self.materials:
            me.materials.append(_material(f"BodyMocap_{key}", PALETTE[key],
                                          0.35 if key == "eyes" else 0.65))
        for g in self.groups:
            obj.vertex_groups.new(name=g)
        return obj


def build_mannequin(rig_obj, profile: Optional[TopologyProfile] = None, name: str = "",
                    collection=None):
    """Create a skinned mannequin mesh for ``rig_obj`` from its topology."""
    rig = RigModel.from_blender(rig_obj)
    prof = profile or detect_topology(rig)
    mb = _MeshBuilder(rig_obj)
    B = rig.bones
    H = lambda n: Vector(B[n].head.as_tuple())  # noqa: E731
    T = lambda n: Vector(B[n].tail.as_tuple())  # noqa: E731

    from ..retarget.solver import RetargetSolver
    solver = RetargetSolver(rig, prof)
    up = Vector(solver.up0.as_tuple())
    left = Vector(solver.left0.as_tuple())
    back = Vector(solver.back0.as_tuple())
    hm = Vector(solver.hm0.as_tuple())
    sm = Vector(solver.sm0.as_tuple())
    torso_h = (sm - hm).length
    s = torso_h / 0.47  # scale relative to the reference body

    # torso: pelvis, abdomen, chest ellipsoids distributed along the spine
    hips = prof.hips
    spine = list(prof.spine)
    if hips:
        mb.ellipsoid(hips, hm + up * 0.02 * s, up, 0.11 * s, 0.17 * s, 0.11 * s, left, "pants")
    if spine:
        seg_pts = [H(n) for n in spine] + [sm]
        for i, n in enumerate(spine):
            a, b = seg_pts[i], seg_pts[i + 1]
            frac = (i + 0.5) / len(spine)
            width = (0.14 + 0.05 * frac) * s
            mb.ellipsoid(n, (a + b) * 0.5, up, max((b - a).length * 0.62, 0.05 * s), width,
                         (0.10 + 0.015 * frac) * s, left, "shirt")
        mb.ellipsoid(spine[-1], sm - up * 0.05 * s, left, 0.2 * s, 0.07 * s, 0.1 * s, up, "shirt")
    # neck + head
    neck = list(prof.neck)
    head = prof.head
    if head:
        base = H(neck[0]) if neck else H(head)
        head_center = H(head) + up * 0.1 * s
        mb.segment(neck[0] if neck else head, base, H(head) + up * 0.03 * s, 0.05 * s, "skin", back)
        mb.ellipsoid(head, head_center, up, 0.125 * s, 0.095 * s, 0.105 * s, left, "skin")
        mb.ellipsoid(head, head_center + up * 0.04 * s + back * 0.02 * s, up, 0.1 * s, 0.1 * s,
                     0.1 * s, left, "hair")
        fwd = -back
        for sd in (1.0, -1.0):
            eye = head_center + fwd * 0.093 * s + left * (0.035 * sd * s) + up * 0.02 * s
            mb.ellipsoid(head, eye, fwd, 0.012 * s, 0.016 * s, 0.012 * s, up, "eyes", 10, 6)
            ear = head_center + left * (0.097 * sd * s)
            mb.ellipsoid(head, ear, left, 0.015 * s, 0.03 * s, 0.02 * s, up, "skin", 10, 6)
        nose = head_center + fwd * 0.105 * s - up * 0.015 * s
        mb.ellipsoid(head, nose, fwd, 0.025 * s, 0.015 * s, 0.02 * s, up, "skin", 10, 6)
        mouth = head_center + fwd * 0.09 * s - up * 0.055 * s
        mb.ellipsoid(head, mouth, left, 0.025 * s, 0.006 * s, 0.01 * s, fwd, "eyes", 10, 6)
    # limbs
    for key, lc in prof.limbs.items():
        arm = lc.kind == "arm"
        radii_u = (0.05, 0.042) if arm else (0.085, 0.058)
        radii_l = (0.04, 0.032) if arm else (0.055, 0.04)
        mat_u = "skin" if arm else "pants"
        mat_l = "skin" if arm else "pants"
        chain = list(lc.upper) + list(lc.lower)
        joints = [H(n) for n in chain] + [H(lc.end) if lc.end else T(chain[-1])]
        n_up = len(lc.upper)
        for i, n in enumerate(chain):
            ru = radii_u if i < n_up else radii_l
            t0 = (i if i < n_up else i - n_up) / max(n_up if i < n_up else len(lc.lower), 1)
            r = (ru[0] + (ru[1] - ru[0]) * t0) * s
            mb.segment(n, joints[i], joints[i + 1], r, mat_u if i < n_up else mat_l, back)
        # joint balls (shoulder / elbow / hip / knee)
        mb.ellipsoid(chain[0], joints[0], up, radii_u[0] * 1.15 * s, radii_u[0] * 1.15 * s,
                     radii_u[0] * 1.15 * s, left, "shirt" if arm else "pants")
        if lc.lower:
            j = joints[n_up]
            mb.ellipsoid(lc.lower[0], j, up, radii_l[0] * 1.1 * s, radii_l[0] * 1.1 * s,
                         radii_l[0] * 1.1 * s, left, mat_l)
        if lc.end:
            r = solver.limb_rest[key]
            end_aim = Vector(r.end_aim0.as_tuple()) if r.end_aim0 is not None else \
                (joints[-1] - joints[-2]).normalized()
            w = joints[-1]
            if arm:
                palm_n = Vector(r.end_frame0.rotate(Vec3(0.0, 0.0, 1.0)).as_tuple()) \
                    if r.end_frame0 is not None else -up
                mb.ellipsoid(lc.end, w + end_aim * 0.055 * s, end_aim, 0.065 * s, 0.022 * s,
                             0.045 * s, palm_n, "skin")
                side_ax = end_aim.cross(palm_n).normalized()
                thumb_dir = (end_aim * 0.6 - side_ax * (1.0 if lc.side == "L" else -1.0)).normalized()
                mb.ellipsoid(lc.end, w + end_aim * 0.03 * s + thumb_dir * 0.035 * s, thumb_dir,
                             0.03 * s, 0.013 * s, 0.013 * s, palm_n, "skin", 10, 6)
            else:
                toe = lc.extra[0] if lc.extra else lc.end
                mb.ellipsoid(lc.end, w + end_aim * 0.07 * s - up * 0.04 * s, end_aim, 0.12 * s,
                             0.035 * s, 0.05 * s, up, "shoes")
                if lc.extra:
                    mb.segment(toe, H(toe), T(toe), 0.035 * s, "shoes", up, flatten=1.3)

    obj = mb.build(name or f"{rig_obj.name}_Mannequin")
    (collection or rig_obj.users_collection[0]).objects.link(obj)
    obj.parent = rig_obj
    obj.matrix_parent_inverse = Matrix.Identity(4)
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig_obj
    return obj


def build_test_rigs(with_mesh: bool = True, spacing: float = 1.4, names: Sequence[str] = (
        "RigA_Standard", "RigB_Segmented")) -> Dict[str, object]:
    """Create the standard test rigs side by side (demo / validation)."""
    from .rig_specs import RIG_SPECS
    out = {}
    for i, key in enumerate(names):
        spec = RIG_SPECS[key]()
        obj = bpy.data.objects.get(key)
        if obj is None:
            obj = build_armature(spec, key, location=(i * spacing, 0.0, 0.0))
        out[key] = obj
        if with_mesh and not any(c.type == "MESH" for c in obj.children):
            build_mannequin(obj)
    return out
