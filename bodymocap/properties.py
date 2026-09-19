"""Scene settings and UI collections for BodyMocap."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from .pose.synthetic import CLIPS, CONDITIONS


def _is_armature(self, obj):
    return obj is not None and obj.type == "ARMATURE"


def _target_changed(self, context):
    from .runtime import get_runtime
    get_runtime().solvers.clear()


def _solver_changed(self, context):
    from .runtime import get_runtime
    get_runtime().solvers.clear()


class BODYMOCAP_PG_target(PropertyGroup):
    obj: PointerProperty(name="Armature", type=bpy.types.Object, poll=_is_armature,
                         update=_target_changed)
    enabled: BoolProperty(name="Enabled", default=True)


CHAIN_ITEMS = [
    ("hips", "Hips", "Root / pelvis bone"),
    ("spine", "Spine", "Hips (excl.) to chest (incl.)"),
    ("neck", "Neck", "Neck bones"),
    ("head", "Head", "Head bone"),
    ("arm_L", "Arm L", ""), ("arm_R", "Arm R", ""),
    ("leg_L", "Leg L", ""), ("leg_R", "Leg R", ""),
]
SEGMENT_ITEMS = [
    ("CORE", "Core", "Hips / spine / neck / head bone"),
    ("ROOT", "Root", "Clavicle / hip-side bone (kept at rest)"),
    ("UPPER", "Upper", "Upper arm / thigh segment"),
    ("LOWER", "Lower", "Forearm / shin segment"),
    ("END", "End", "Hand / foot"),
    ("EXTRA", "Extra", "Toes etc. (kept at rest)"),
]


class BODYMOCAP_PG_chain_entry(PropertyGroup):
    chain: EnumProperty(name="Chain", items=CHAIN_ITEMS)
    segment: EnumProperty(name="Segment", items=SEGMENT_ITEMS)
    bone: StringProperty(name="Bone")


class BODYMOCAP_PG_settings(PropertyGroup):
    # ------------------------------------------------------------------ target
    target: PointerProperty(name="Target Armature", type=bpy.types.Object, poll=_is_armature,
                            update=_target_changed,
                            description="Armature driven by the capture (rig selection)")
    extra_targets: CollectionProperty(type=BODYMOCAP_PG_target)
    extra_index: IntProperty(default=0)
    live_apply: BoolProperty(name="Live Preview on Rig", default=True,
                             description="Pose the target armature(s) while tracking")
    live_apply_all_targets: BoolProperty(name="Drive All Targets Live", default=True)

    # ------------------------------------------------------------------ source
    source: EnumProperty(
        name="Source",
        items=[
            ("WEBCAM", "Webcam", "Live camera via OpenCV", "OUTLINER_OB_CAMERA", 0),
            ("VIDEO", "Video File", "Track a recorded video", "FILE_MOVIE", 1),
            ("IMAGES", "Image Sequence", "Track a folder of frames", "FILE_IMAGE", 2),
            ("SYNTHETIC", "Synthetic Performer", "Procedural test performer (no camera needed)",
             "ARMATURE_DATA", 3),
            ("TAKE", "Recorded Take", "Replay a saved take file", "FILE_BLANK", 4),
        ],
        default="WEBCAM",
    )
    device_index: IntProperty(name="Camera Index", default=0, min=0, max=16)
    video_path: StringProperty(name="Video", subtype="FILE_PATH")
    image_dir: StringProperty(name="Images", subtype="DIR_PATH")
    take_path: StringProperty(name="Take File", subtype="FILE_PATH")
    loop_source: BoolProperty(name="Loop", default=False)
    capture_width: IntProperty(name="Width", default=640, min=160, max=3840)
    capture_height: IntProperty(name="Height", default=480, min=120, max=2160)
    capture_fps: FloatProperty(name="FPS", default=30.0, min=1.0, max=120.0)
    tracking_fov: FloatProperty(name="Camera FOV", default=60.0, min=10.0, max=150.0,
                                description="Horizontal field of view of the physical camera "
                                            "(used by the spawned camera and root tracking)")
    camera_distance: FloatProperty(name="Subject Distance", default=0.0, min=0.0, max=30.0,
                                   description="Camera distance for Spawn Camera (0 = fit rig)")
    mirror_preview: BoolProperty(name="Mirror Preview", default=True,
                                 description="Show the camera image mirrored (selfie view)")
    mirror_motion: BoolProperty(name="Mirror Motion", default=False,
                                description="Rig moves like a mirror image of the performer")
    show_overlay: BoolProperty(name="Skeleton Overlay", default=True)
    save_debug_video: BoolProperty(
        name="Keep Preview Frames", default=False,
        description="Never write raw video to disk unless enabled (privacy, FR-092)")

    # synthetic performer
    synth_clip: EnumProperty(name="Motion", items=[(k, k.replace("_", " ").title(), "") for k in CLIPS],
                             default="wave")
    synth_condition: EnumProperty(name="Lighting Sim",
                                  items=[(k, k.replace("_", " ").title(), "") for k in CONDITIONS],
                                  default="normal")
    synth_speed: FloatProperty(name="Speed", default=1.0, min=0.1, max=4.0)
    synth_duration: FloatProperty(name="Duration (offline)", default=4.0, min=0.1, max=600.0,
                                  description="Length of the synthetic performance when "
                                              "processed offline / in background mode")

    # ------------------------------------------------------------ tracking
    model_variant: EnumProperty(
        name="Model",
        items=[("LITE", "Lite", "Fastest (~5 MB)"), ("FULL", "Full", "Balanced (~9 MB)"),
               ("HEAVY", "Heavy", "Most accurate, slow on CPU (~29 MB)")],
        default="FULL",
    )
    min_detection: FloatProperty(name="Detection", default=0.5, min=0.05, max=0.99,
                                 description="Minimum person-detection confidence")
    min_presence: FloatProperty(name="Presence", default=0.5, min=0.05, max=0.99)
    min_tracking: FloatProperty(name="Tracking", default=0.5, min=0.05, max=0.99,
                                description="Minimum confidence to keep tracking between frames")
    min_visibility: FloatProperty(name="Landmark Sensitivity", default=0.5, min=0.05, max=0.99,
                                  description="Landmarks below this visibility are treated as "
                                              "lost (held briefly, then chains hold pose)",
                                  update=_solver_changed)
    smoothing_enabled: BoolProperty(name="Smoothing", default=True)
    smooth_min_cutoff: FloatProperty(name="Min Cutoff (Hz)", default=1.0, min=0.05, max=10.0,
                                     description="Lower = smoother slow motion")
    smooth_beta: FloatProperty(name="Speed Response", default=5.0, min=0.0, max=50.0,
                               description="Higher = less lag on fast gestures")

    # ------------------------------------------------------------ lighting
    exposure_mode: EnumProperty(
        name="Exposure",
        items=[("OFF", "Off", "Feed frames unchanged"),
               ("AUTO", "Auto", "Auto contrast / gamma, CLAHE for back-light"),
               ("MANUAL", "Manual", "Fixed gain / gamma / contrast")],
        default="AUTO",
    )
    exposure_ev: FloatProperty(name="Gain (EV)", default=0.0, min=-4.0, max=4.0)
    exposure_gamma: FloatProperty(name="Gamma", default=1.0, min=0.2, max=3.0)
    exposure_contrast: FloatProperty(name="Contrast", default=1.0, min=0.2, max=3.0)
    exposure_clahe: BoolProperty(name="Local Contrast (CLAHE)", default=True)
    exposure_clahe_clip: FloatProperty(name="CLAHE Clip", default=2.0, min=0.5, max=8.0)

    # ------------------------------------------------------------- solver
    root_motion: BoolProperty(name="Root Motion", default=False, update=_solver_changed,
                              description="Translate the hips from the camera-space position")
    root_scale: FloatProperty(name="Root Scale", default=0.0, min=0.0, max=100.0,
                              update=_solver_changed,
                              description="0 = automatic (rig leg length / performer leg length)")
    anti_knee_inversion: BoolProperty(name="Prevent Knee Inversion", default=True,
                                      update=_solver_changed)
    pole_blend_lo: FloatProperty(name="Straight Limb Below", default=6.0, min=0.0, max=45.0,
                                 update=_solver_changed,
                                 description="Bend angle (deg) under which the pole vector falls "
                                             "back to its stable estimate")
    pole_blend_hi: FloatProperty(name="Trusted Bend Above", default=25.0, min=1.0, max=90.0,
                                 update=_solver_changed)
    twist_share: FloatProperty(name="Forearm Twist", default=1.0, min=0.0, max=1.0,
                               update=_solver_changed,
                               description="Share of hand twist distributed along the forearm")
    pelvis_tilt_share: FloatProperty(name="Pelvis Tilt Share", default=0.5, min=0.0, max=1.0,
                                     update=_solver_changed)
    spine_position_match: BoolProperty(name="Match Shoulder Position", default=True,
                                       update=_solver_changed)
    drive_hands: BoolProperty(name="Hands", default=True, update=_solver_changed)
    drive_feet: BoolProperty(name="Feet", default=True, update=_solver_changed)
    drive_head: BoolProperty(name="Head", default=True, update=_solver_changed)

    # ------------------------------------------------------------- calibration
    rest_pose_style: EnumProperty(name="Rest Pose", items=[("T_POSE", "T-Pose", ""), ("A_POSE", "A-Pose", "")],
                                  default="T_POSE")
    calibration_seconds: FloatProperty(name="Duration", default=2.0, min=0.5, max=10.0)
    is_calibrated: BoolProperty(default=False)
    calibration_status: StringProperty(default="Not calibrated")
    calibrate_from_take: BoolProperty(name="Calibrate From Take Start", default=False,
                                      description="Use the first seconds of the take as the neutral pose")

    # ------------------------------------------------------------- record / bake
    action_name: StringProperty(name="Action", default="MocapTake")
    start_frame: IntProperty(name="Start Frame", default=1)
    overwrite_action: BoolProperty(name="Overwrite", default=True)
    rotation_mode: EnumProperty(name="Rotation", items=[("QUATERNION", "Quaternion", ""),
                                                        ("EULER", "Euler XYZ", "")], default="QUATERNION")
    interpolation: EnumProperty(name="Interpolation", items=[("LINEAR", "Linear", ""),
                                                             ("BEZIER", "Bezier", "")], default="LINEAR")
    apply_mode: EnumProperty(name="Apply As", items=[("ACTION", "Active Action", ""),
                                                     ("NLA", "NLA Strip", "")], default="ACTION")
    auto_bake: BoolProperty(name="Bake on Stop", default=True,
                            description="One-click: stopping the recording bakes and applies the Action")
    bake_all_targets: BoolProperty(name="Bake All Targets", default=True)
    set_scene_range: BoolProperty(name="Set Scene Range", default=True)
    degraded_warn_fraction: FloatProperty(name="Warn If Degraded >", default=0.15, min=0.0, max=1.0,
                                          subtype="FACTOR")

    # ------------------------------------------------------------- retarget
    retarget_source: PointerProperty(name="Source Armature", type=bpy.types.Object, poll=_is_armature)
    retarget_action: StringProperty(name="Source Action")
    retarget_new_action: StringProperty(name="New Action", default="RetargetedAction")

    # ------------------------------------------------------------- topology editing
    profile_entries: CollectionProperty(type=BODYMOCAP_PG_chain_entry)
    profile_index: IntProperty(default=0)
    profile_summary: StringProperty(default="")
    profile_warnings: StringProperty(default="")

    # ------------------------------------------------------------- status
    tracking_status: StringProperty(default="Idle")
    lighting_status: StringProperty(default="-")
    capture_fps_live: FloatProperty(default=0.0)
    is_capturing: BoolProperty(default=False)
    is_recording: BoolProperty(default=False)
    record_frame_count: IntProperty(default=0)
    take_frame_count: IntProperty(default=0)
    status_text: StringProperty(default="")
    deps_status: StringProperty(default="")


CLASSES = (BODYMOCAP_PG_target, BODYMOCAP_PG_chain_entry, BODYMOCAP_PG_settings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.bodymocap = PointerProperty(type=BODYMOCAP_PG_settings)


def unregister():
    if hasattr(bpy.types.Scene, "bodymocap"):
        del bpy.types.Scene.bodymocap
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
