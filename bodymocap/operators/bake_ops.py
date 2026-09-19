"""POSE_OT_bake_animation and take / action management operators."""

from __future__ import annotations

import bpy
from bpy.props import StringProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper, ImportHelper

from ..runtime import calibration_from_take, get_profile, get_runtime, smoothing_params, solver_settings, target_objects


def apply_action(obj, action, mode: str = "ACTION", start_frame: int = 1) -> str:
    from ..utils.anim_compat import assign_action
    if obj.animation_data is None:
        obj.animation_data_create()
    ad = obj.animation_data
    if mode == "NLA":
        track = ad.nla_tracks.get("BodyMocap") or ad.nla_tracks.new()
        track.name = "BodyMocap"
        for strip in list(track.strips):
            if strip.action == action:
                track.strips.remove(strip)
        strip = track.strips.new(action.name, int(start_frame), action)
        ad.action = None
        return f"NLA strip '{strip.name}' on {obj.name}"
    assign_action(obj, action)
    return f"Action '{action.name}' assigned to {obj.name}"


class POSE_OT_bake_animation(Operator):
    """Bake the last recorded take into Actions on the target rig(s) and apply them"""

    bl_idname = "pose.bake_animation"
    bl_label = "Bake Animation"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return get_runtime().last_take is not None

    def execute(self, context):
        from ..bake.action import BakeSettings, bake_take

        s = context.scene.bodymocap
        rt = get_runtime()
        take = rt.last_take
        if take is None or take.frame_count() < 2:
            self.report({"ERROR"}, "No recorded take to bake")
            return {"CANCELLED"}
        targets = target_objects(s, include_extra=s.bake_all_targets)
        if not targets:
            self.report({"ERROR"}, "Set a target armature first")
            return {"CANCELLED"}
        cal = rt.calibration
        if cal is None and s.calibrate_from_take:
            cal = calibration_from_take(take, s.calibration_seconds, s.min_visibility)
        scene = context.scene
        fps = scene.render.fps / scene.render.fps_base
        rt.last_bake = {}
        last_end = s.start_frame
        for i, obj in enumerate(targets):
            name = s.action_name if i == 0 else f"{s.action_name}_{obj.name}"
            bs = BakeSettings(action_name=name, start_frame=s.start_frame, fps=fps,
                              overwrite=s.overwrite_action, rotation_mode=s.rotation_mode,
                              interpolation=s.interpolation, smoothing=smoothing_params(s),
                              min_conf=s.min_visibility)
            profile = get_profile(obj)
            ok, msg, action, stats = bake_take(obj, take.frames, profile, bs, solver_settings(s), cal)
            if not ok:
                self.report({"ERROR"}, f"{obj.name}: {msg}")
                continue
            apply_msg = apply_action(obj, action, s.apply_mode, s.start_frame)
            rt.last_bake[obj.name] = {"action": action.name, "stats": stats}
            last_end = max(last_end, s.start_frame + stats.frames - 1)
            self.report({"INFO"}, f"{msg}; {apply_msg}")
        if not rt.last_bake:
            return {"CANCELLED"}
        if s.set_scene_range:
            scene.frame_start = s.start_frame
            scene.frame_end = last_end
            scene.frame_set(s.start_frame)
        return {"FINISHED"}


class POSE_OT_apply_mocap_action(Operator):
    """Assign an existing Action (or NLA strip) to the target armature"""

    bl_idname = "pose.apply_mocap_action"
    bl_label = "Apply Action"
    bl_options = {"REGISTER", "UNDO"}

    action: StringProperty(name="Action")

    def execute(self, context):
        s = context.scene.bodymocap
        obj = s.target
        if obj is None:
            self.report({"ERROR"}, "Set a target armature first")
            return {"CANCELLED"}
        act = bpy.data.actions.get(self.action or s.action_name)
        if act is None:
            self.report({"ERROR"}, f"Action '{self.action or s.action_name}' not found")
            return {"CANCELLED"}
        self.report({"INFO"}, apply_action(obj, act, s.apply_mode, s.start_frame))
        return {"FINISHED"}


class POSE_OT_save_take(Operator, ExportHelper):
    """Save the last recorded take (landmarks, not video) to a JSON file"""

    bl_idname = "pose.save_take"
    bl_label = "Save Take"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    @classmethod
    def poll(cls, context):
        return get_runtime().last_take is not None

    def execute(self, context):
        take = get_runtime().last_take
        path = take.save(bpy.path.abspath(self.filepath))
        self.report({"INFO"}, f"Saved {take.frame_count()} frames to {path}")
        return {"FINISHED"}


class POSE_OT_load_take(Operator, ImportHelper):
    """Load a take file so it can be baked onto any rig"""

    bl_idname = "pose.load_take"
    bl_label = "Load Take"
    filename_ext = ".json"
    filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

    def execute(self, context):
        from ..recording.session import Take
        try:
            take = Take.load(bpy.path.abspath(self.filepath))
        except Exception as exc:
            self.report({"ERROR"}, f"Could not load take: {exc}")
            return {"CANCELLED"}
        get_runtime().last_take = take
        context.scene.bodymocap.take_frame_count = take.frame_count()
        self.report({"INFO"}, f"Loaded take '{take.name}' ({take.frame_count()} frames)")
        return {"FINISHED"}


CLASSES = (POSE_OT_bake_animation, POSE_OT_apply_mocap_action, POSE_OT_save_take, POSE_OT_load_take)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
