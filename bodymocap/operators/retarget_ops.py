"""Topology mapping profiles, extra targets and armature->armature retargeting."""

from __future__ import annotations

import json

import bpy
from bpy.props import BoolProperty, StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..retarget.rig import RigModel
from ..retarget.topology import LimbChain, TopologyProfile, detect_topology
from ..runtime import get_profile, get_runtime, solver_settings, store_profile, target_objects

SEGMENT_OF = {"root": "ROOT", "upper": "UPPER", "lower": "LOWER", "end": "END", "extra": "EXTRA"}


def profile_to_entries(prof: TopologyProfile, s) -> None:
    s.profile_entries.clear()

    def add(chain, seg, bone):
        e = s.profile_entries.add()
        e.chain, e.segment, e.bone = chain, seg, bone

    if prof.hips:
        add("hips", "CORE", prof.hips)
    for b in prof.spine:
        add("spine", "CORE", b)
    for b in prof.neck:
        add("neck", "CORE", b)
    if prof.head:
        add("head", "CORE", prof.head)
    for key in ("arm_L", "arm_R", "leg_L", "leg_R"):
        lc = prof.limbs.get(key)
        if not lc:
            continue
        for b in lc.root:
            add(key, "ROOT", b)
        for b in lc.upper:
            add(key, "UPPER", b)
        for b in lc.lower:
            add(key, "LOWER", b)
        if lc.end:
            add(key, "END", lc.end)
        for b in lc.extra:
            add(key, "EXTRA", b)
    s.profile_summary = prof.summary()
    s.profile_warnings = " | ".join(prof.warnings[:4])


def entries_to_profile(s, name: str = "") -> TopologyProfile:
    prof = TopologyProfile(name=name)
    limbs = {}
    for e in s.profile_entries:
        if not e.bone:
            continue
        if e.chain == "hips":
            prof.hips = e.bone
        elif e.chain == "spine":
            prof.spine.append(e.bone)
        elif e.chain == "neck":
            prof.neck.append(e.bone)
        elif e.chain == "head":
            prof.head = e.bone
        else:
            lc = limbs.setdefault(e.chain, LimbChain(
                name=e.chain, kind="arm" if e.chain.startswith("arm") else "leg",
                side=e.chain[-1], method="manual"))
            if e.segment == "END":
                lc.end = e.bone
            elif e.segment == "ROOT":
                lc.root.append(e.bone)
            elif e.segment == "UPPER":
                lc.upper.append(e.bone)
            elif e.segment == "LOWER":
                lc.lower.append(e.bone)
            elif e.segment == "EXTRA":
                lc.extra.append(e.bone)
    for lc in limbs.values():
        lc.mode = "SEGMENTED" if lc.lower else "CONTINUOUS"
    prof.limbs = limbs
    return prof


class POSE_OT_detect_topology(Operator):
    """Auto-detect bone chains (spine, neck, arms, legs, segment split) on the
    target armature(s) and store them as a topology mapping profile"""

    bl_idname = "pose.detect_topology"
    bl_label = "Detect Topology"
    bl_options = {"REGISTER", "UNDO"}

    all_targets: BoolProperty(name="All Targets", default=True)

    def execute(self, context):
        s = context.scene.bodymocap
        objs = target_objects(s, include_extra=self.all_targets)
        if not objs and context.active_object and context.active_object.type == "ARMATURE":
            objs = [context.active_object]
            s.target = context.active_object
        if not objs:
            self.report({"ERROR"}, "Select or set a target armature")
            return {"CANCELLED"}
        for i, obj in enumerate(objs):
            prof = get_profile(obj, redetect=True)
            if i == 0:
                profile_to_entries(prof, s)
            level = {"WARNING"} if prof.warnings else {"INFO"}
            self.report(level, f"{obj.name}: {prof.summary()}" +
                        (f" ({'; '.join(prof.warnings[:2])})" if prof.warnings else ""))
        get_runtime().solvers.clear()
        return {"FINISHED"}


class POSE_OT_profile_apply_edits(Operator):
    """Validate the edited mapping list and store it on the target armature"""

    bl_idname = "pose.profile_apply_edits"
    bl_label = "Apply Mapping Edits"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        s = context.scene.bodymocap
        obj = s.target
        if obj is None:
            self.report({"ERROR"}, "Set a target armature")
            return {"CANCELLED"}
        prof = entries_to_profile(s, obj.name)
        errs = prof.validate(RigModel.from_blender(obj))
        if errs:
            self.report({"ERROR"}, "; ".join(errs[:3]))
            return {"CANCELLED"}
        store_profile(obj, prof)
        s.profile_summary = prof.summary()
        get_runtime().solvers.clear()
        self.report({"INFO"}, f"Profile stored on {obj.name}: {prof.summary()}")
        return {"FINISHED"}


class POSE_OT_profile_save(Operator, ExportHelper):
    bl_idname = "pose.profile_save"
    bl_label = "Save Mapping Profile"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        s = context.scene.bodymocap
        if s.target is None:
            self.report({"ERROR"}, "Set a target armature")
            return {"CANCELLED"}
        prof = get_profile(s.target)
        path = prof.save(bpy.path.abspath(self.filepath))
        self.report({"INFO"}, f"Saved profile to {path}")
        return {"FINISHED"}


class POSE_OT_profile_load(Operator, ImportHelper):
    bl_idname = "pose.profile_load"
    bl_label = "Load Mapping Profile"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        s = context.scene.bodymocap
        if s.target is None:
            self.report({"ERROR"}, "Set a target armature")
            return {"CANCELLED"}
        try:
            prof = TopologyProfile.load(bpy.path.abspath(self.filepath))
        except Exception as exc:
            self.report({"ERROR"}, f"Could not load profile: {exc}")
            return {"CANCELLED"}
        errs = prof.validate(RigModel.from_blender(s.target))
        if errs:
            self.report({"ERROR"}, "Profile does not fit this armature: " + "; ".join(errs[:3]))
            return {"CANCELLED"}
        store_profile(s.target, prof)
        profile_to_entries(prof, s)
        get_runtime().solvers.clear()
        self.report({"INFO"}, f"Loaded profile: {prof.summary()}")
        return {"FINISHED"}


class POSE_OT_mocap_target_add(Operator):
    """Add the selected armatures as additional targets (driven/baked together)"""

    bl_idname = "pose.mocap_target_add"
    bl_label = "Add Selected Armatures"

    def execute(self, context):
        s = context.scene.bodymocap
        existing = {t.obj for t in s.extra_targets}
        n = 0
        for obj in context.selected_objects:
            if obj.type == "ARMATURE" and obj != s.target and obj not in existing:
                t = s.extra_targets.add()
                t.obj = obj
                n += 1
        get_runtime().solvers.clear()
        self.report({"INFO"}, f"Added {n} target(s)")
        return {"FINISHED"}


class POSE_OT_mocap_target_remove(Operator):
    bl_idname = "pose.mocap_target_remove"
    bl_label = "Remove Target"

    def execute(self, context):
        s = context.scene.bodymocap
        if 0 <= s.extra_index < len(s.extra_targets):
            s.extra_targets.remove(s.extra_index)
            s.extra_index = max(0, s.extra_index - 1)
        get_runtime().solvers.clear()
        return {"FINISHED"}


class POSE_OT_retarget_action(Operator):
    """Retarget an Action from a source armature onto the target armature,
    whatever the bone counts of either rig"""

    bl_idname = "pose.retarget_action"
    bl_label = "Retarget Action"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        from ..retarget.transfer import transfer_in_blender
        from .bake_ops import apply_action

        s = context.scene.bodymocap
        src, tgt = s.retarget_source, s.target
        if src is None or tgt is None or src == tgt:
            self.report({"ERROR"}, "Set a source armature and a (different) target armature")
            return {"CANCELLED"}
        settings = solver_settings(s)
        settings.root_motion = True
        ok, msg, act = transfer_in_blender(
            src, tgt, s.retarget_action, s.retarget_new_action, s.start_frame,
            get_profile(src), get_profile(tgt), settings, s.rotation_mode)
        if not ok:
            self.report({"ERROR"}, msg)
            return {"CANCELLED"}
        apply_action(tgt, act, s.apply_mode, s.start_frame)
        self.report({"INFO"}, msg)
        return {"FINISHED"}


CLASSES = (
    POSE_OT_detect_topology,
    POSE_OT_profile_apply_edits,
    POSE_OT_profile_save,
    POSE_OT_profile_load,
    POSE_OT_mocap_target_add,
    POSE_OT_mocap_target_remove,
    POSE_OT_retarget_action,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
