"""Sidebar UI: View3D > Sidebar (N) > Mocap."""

from __future__ import annotations

import os

import bpy
from bpy.types import Panel

CATEGORY = "Mocap"


def _rt():
    from ..runtime import get_runtime
    return get_runtime()


class _Base:
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = CATEGORY


class VIEW3D_PT_pose_retargeter(_Base, Panel):
    bl_label = "Pose Capture & Retarget"
    bl_idname = "VIEW3D_PT_pose_retargeter"

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        rt = _rt()

        layout.prop(s, "target", icon="ARMATURE_DATA")

        box = layout.box()
        col = box.column(align=True)
        state = s.tracking_status if rt.session is not None else "Idle"
        icon = {"OK": "CHECKMARK", "DEGRADED": "ERROR", "LOST": "CANCEL"}.get(state, "PAUSE")
        row = col.row()
        row.label(text=f"Tracking: {state}", icon=icon)
        if rt.session is not None:
            row.label(text=f"{s.capture_fps_live:.0f} fps")
        col.label(text=f"Lighting: {s.lighting_status}", icon="LIGHT_SUN")
        col.label(text=s.calibration_status, icon="CHECKMARK" if s.is_calibrated else "INFO")

        col = layout.column(align=True)
        row = col.row(align=True)
        row.scale_y = 1.6
        if rt.recording.is_recording:
            row.operator("pose.record_toggle", text="Stop & Bake", icon="SNAP_FACE")
        else:
            row.operator("pose.record_toggle", text="Record", icon="REC")
        row = col.row(align=True)
        row.scale_y = 1.2
        if rt.session is not None:
            row.operator("pose.stop_capture", text="Stop Tracking", icon="PAUSE")
        else:
            row.operator("pose.start_capture", text="Start Tracking", icon="PLAY")
        row = col.row(align=True)
        row.operator("pose.spawn_camera", icon="OUTLINER_OB_CAMERA")
        row.operator("pose.calibrate_capture", icon="ARMATURE_DATA")
        if s.is_recording:
            layout.label(text=f"Recording... {s.record_frame_count} frames", icon="REC")
        elif rt.last_take is not None:
            layout.label(text=f"Last take: {rt.last_take.frame_count()} frames "
                              f"({rt.last_take.duration:.1f}s)", icon="SEQUENCE")
        if s.status_text:
            layout.label(text=s.status_text)


class VIEW3D_PT_pose_retargeter_source(_Base, Panel):
    bl_label = "Camera & Source"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        layout.prop(s, "source")
        col = layout.column(align=True)
        if s.source == "WEBCAM":
            col.prop(s, "device_index")
            row = col.row(align=True)
            row.prop(s, "capture_width")
            row.prop(s, "capture_height")
            col.prop(s, "capture_fps")
        elif s.source == "VIDEO":
            col.prop(s, "video_path")
            col.prop(s, "loop_source")
        elif s.source == "IMAGES":
            col.prop(s, "image_dir")
            col.prop(s, "capture_fps")
        elif s.source == "SYNTHETIC":
            col.prop(s, "synth_clip")
            col.prop(s, "synth_condition")
            col.prop(s, "synth_speed")
            col.prop(s, "synth_duration")
        elif s.source == "TAKE":
            col.prop(s, "take_path")
        layout.separator()
        col = layout.column(align=True)
        col.prop(s, "tracking_fov")
        col.prop(s, "camera_distance")
        row = layout.row(align=True)
        row.prop(s, "mirror_preview", toggle=True)
        row.prop(s, "mirror_motion", toggle=True)
        row.prop(s, "show_overlay", toggle=True)
        layout.operator("pose.spawn_camera", icon="OUTLINER_OB_CAMERA")


class VIEW3D_PT_pose_retargeter_tracking(_Base, Panel):
    bl_label = "Tracking & Calibration"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        from ..runtime import model_file
        layout = self.layout
        s = context.scene.bodymocap
        row = layout.row(align=True)
        row.prop(s, "model_variant")
        if s.source not in ("SYNTHETIC", "TAKE") and not os.path.isfile(model_file(s, context)):
            row.operator("pose.download_pose_model", text="", icon="IMPORT")
        col = layout.column(align=True)
        col.label(text="Landmark Sensitivity")
        col.prop(s, "min_visibility")
        col.prop(s, "min_detection")
        col.prop(s, "min_tracking")
        col.prop(s, "min_presence")
        col = layout.column(align=True)
        col.prop(s, "smoothing_enabled")
        sub = col.column(align=True)
        sub.active = s.smoothing_enabled
        sub.prop(s, "smooth_min_cutoff")
        sub.prop(s, "smooth_beta")
        col = layout.column(align=True)
        col.prop(s, "live_apply")
        col.prop(s, "live_apply_all_targets")
        box = layout.box()
        box.label(text="Calibration", icon="ARMATURE_DATA")
        box.prop(s, "rest_pose_style")
        box.prop(s, "calibration_seconds")
        row = box.row(align=True)
        row.operator("pose.calibrate_capture")
        row.operator("pose.clear_mocap_calibration", text="", icon="X")


class VIEW3D_PT_pose_retargeter_lighting(_Base, Panel):
    bl_label = "Lighting Compensation"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        layout.prop(s, "exposure_mode", expand=True)
        col = layout.column(align=True)
        if s.exposure_mode == "MANUAL":
            col.prop(s, "exposure_ev")
            col.prop(s, "exposure_gamma")
            col.prop(s, "exposure_contrast")
        if s.exposure_mode != "OFF":
            col.prop(s, "exposure_clahe")
            sub = col.row()
            sub.active = s.exposure_clahe
            sub.prop(s, "exposure_clahe_clip")
        layout.label(text=f"Camera sees: {s.lighting_status}", icon="LIGHT_SUN")


class VIEW3D_PT_pose_retargeter_mapping(_Base, Panel):
    bl_label = "Topology Mapping"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        row = layout.row(align=True)
        row.operator("pose.detect_topology", icon="VIEWZOOM")
        row.operator("pose.profile_apply_edits", text="", icon="CHECKMARK")
        row.operator("pose.profile_save", text="", icon="EXPORT")
        row.operator("pose.profile_load", text="", icon="IMPORT")
        if s.profile_summary:
            layout.label(text=s.profile_summary)
        if s.profile_warnings:
            layout.label(text=s.profile_warnings, icon="ERROR")
        layout.template_list("BODYMOCAP_UL_profile", "", s, "profile_entries", s, "profile_index", rows=6)

        box = layout.box()
        box.label(text="Additional Targets", icon="OUTLINER_OB_ARMATURE")
        row = box.row()
        row.template_list("BODYMOCAP_UL_targets", "", s, "extra_targets", s, "extra_index", rows=2)
        col = row.column(align=True)
        col.operator("pose.mocap_target_add", text="", icon="ADD")
        col.operator("pose.mocap_target_remove", text="", icon="REMOVE")

        box = layout.box()
        box.label(text="Chain Solver", icon="CON_SPLINEIK")
        col = box.column(align=True)
        col.prop(s, "anti_knee_inversion")
        col.prop(s, "pole_blend_lo")
        col.prop(s, "pole_blend_hi")
        col.prop(s, "twist_share")
        col.prop(s, "pelvis_tilt_share")
        col.prop(s, "spine_position_match")
        row = box.row(align=True)
        row.prop(s, "drive_hands", toggle=True)
        row.prop(s, "drive_feet", toggle=True)
        row.prop(s, "drive_head", toggle=True)
        row = box.row(align=True)
        row.prop(s, "root_motion")
        sub = row.row()
        sub.active = s.root_motion
        sub.prop(s, "root_scale")


class VIEW3D_PT_pose_retargeter_bake(_Base, Panel):
    bl_label = "Record & Bake"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        col = layout.column(align=True)
        col.prop(s, "action_name")
        col.prop(s, "start_frame")
        row = col.row(align=True)
        row.prop(s, "rotation_mode", text="")
        row.prop(s, "interpolation", text="")
        col.prop(s, "apply_mode")
        col = layout.column(align=True)
        col.prop(s, "auto_bake")
        col.prop(s, "bake_all_targets")
        col.prop(s, "calibrate_from_take")
        col.prop(s, "set_scene_range")
        col.prop(s, "degraded_warn_fraction")
        layout.operator("pose.bake_animation", icon="ACTION")
        row = layout.row(align=True)
        row.operator("pose.save_take", icon="EXPORT")
        row.operator("pose.load_take", icon="IMPORT")
        row.operator("pose.discard_take", text="", icon="TRASH")
        layout.operator("pose.apply_mocap_action", icon="PLAY")


class VIEW3D_PT_pose_retargeter_retarget(_Base, Panel):
    bl_label = "Retarget Action"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        col = layout.column(align=True)
        col.prop(s, "retarget_source")
        col.prop_search(s, "retarget_action", bpy.data, "actions")
        col.prop(s, "retarget_new_action")
        col.label(text=f"Target: {s.target.name if s.target else '-'}")
        layout.operator("pose.retarget_action", icon="ARROW_LEFTRIGHT")


class VIEW3D_PT_pose_retargeter_deps(_Base, Panel):
    bl_label = "Setup"
    bl_parent_id = "VIEW3D_PT_pose_retargeter"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        s = context.scene.bodymocap
        box = layout.box()
        for part in (s.deps_status or "Press Refresh").split(" | "):
            box.label(text=part, icon="CHECKMARK" if "OK" in part else "ERROR")
        row = layout.row(align=True)
        row.operator("pose.install_mocap_dependencies", icon="IMPORT")
        row.operator("pose.refresh_mocap_dependencies", text="", icon="FILE_REFRESH")
        layout.operator("pose.download_pose_model", icon="URL")
        layout.separator()
        layout.operator("pose.create_test_rigs", icon="OUTLINER_OB_ARMATURE")


CLASSES = (
    VIEW3D_PT_pose_retargeter,
    VIEW3D_PT_pose_retargeter_source,
    VIEW3D_PT_pose_retargeter_tracking,
    VIEW3D_PT_pose_retargeter_lighting,
    VIEW3D_PT_pose_retargeter_mapping,
    VIEW3D_PT_pose_retargeter_bake,
    VIEW3D_PT_pose_retargeter_retarget,
    VIEW3D_PT_pose_retargeter_deps,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
