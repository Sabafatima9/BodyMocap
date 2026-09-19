"""POSE_OT_spawn_camera: a scene camera that matches the physical tracking camera."""

from __future__ import annotations

import math

import bpy
from bpy.props import BoolProperty
from bpy.types import Operator
from mathutils import Vector

CAMERA_NAME = "BodyMocap_TrackingCam"


def armature_bounds(obj):
    """World-space (min, max) over bone heads/tails of the current pose."""
    mw = obj.matrix_world
    pts = []
    for pb in obj.pose.bones:
        pts.append(mw @ pb.head)
        pts.append(mw @ pb.tail)
    if not pts:
        c = mw.translation
        return c - Vector((0.5, 0.5, 1.0)), c + Vector((0.5, 0.5, 1.0))
    mn = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    mx = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return mn, mx


def rig_forward_up(obj):
    """World-space facing direction and up axis of a humanoid armature."""
    try:
        from ..retarget.rig import RigModel
        from ..retarget.solver import RetargetSolver
        from ..runtime import get_profile
        rig = RigModel.from_blender(obj)
        solver = RetargetSolver(rig, get_profile(obj, rig))
        m3 = obj.matrix_world.to_3x3()
        fwd = (m3 @ Vector(tuple(-solver.back0))).normalized()
        up = (m3 @ Vector(tuple(solver.up0))).normalized()
        return fwd, up
    except Exception:
        m3 = obj.matrix_world.to_3x3()
        return (m3 @ Vector((0, -1, 0))).normalized(), (m3 @ Vector((0, 0, 1))).normalized()


def spawn_tracking_camera(context, target=None, fov_deg=60.0, width=640, height=480,
                          distance=0.0, make_active=True, set_resolution=True):
    scene = context.scene
    cam_obj = bpy.data.objects.get(CAMERA_NAME)
    if cam_obj is None or cam_obj.type != "CAMERA":
        cam_data = bpy.data.cameras.new(CAMERA_NAME)
        cam_obj = bpy.data.objects.new(CAMERA_NAME, cam_data)
    if cam_obj.name not in scene.collection.all_objects:
        scene.collection.objects.link(cam_obj)
    cam = cam_obj.data
    cam.type = "PERSP"
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 36.0
    cam.angle = math.radians(fov_deg)
    cam.clip_start = 0.05
    cam.clip_end = 200.0

    if target is not None:
        mn, mx = armature_bounds(target)
        fwd, up = rig_forward_up(target)
    else:
        mn, mx = Vector((-0.5, -0.3, 0.0)), Vector((0.5, 0.3, 1.8))
        fwd, up = Vector((0, -1, 0)), Vector((0, 0, 1))
    center = (mn + mx) * 0.5
    height_m = max((mx - mn).dot(up), 0.5)
    vfov = 2.0 * math.atan(math.tan(math.radians(fov_deg) * 0.5) * height / max(width, 1))
    dist = distance if distance > 0.0 else (height_m * 0.6) / math.tan(vfov * 0.5)
    cam_obj.location = center + fwd * dist
    direction = (center - cam_obj.location).normalized()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam_obj["bodymocap_fov"] = float(fov_deg)
    cam_obj["bodymocap_target"] = target.name if target else ""

    if set_resolution:
        scene.render.resolution_x = int(width)
        scene.render.resolution_y = int(height)
        scene.render.resolution_percentage = 100
        scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    if make_active:
        scene.camera = cam_obj

    # live feed as camera background (visible in camera view)
    from ..camera.preview import ensure_preview_image
    img = ensure_preview_image(width, height)
    cam.show_background_images = True
    bg = None
    for b in cam.background_images:
        if b.image == img:
            bg = b
            break
    if bg is None:
        bg = cam.background_images.new()
        bg.image = img
    bg.alpha = 0.85
    bg.display_depth = "BACK"
    bg.frame_method = "FIT"
    return cam_obj


class POSE_OT_spawn_camera(Operator):
    """Create (or update) a camera matching the tracking camera's field of view,
    aimed at the target armature, with the live feed as its background"""

    bl_idname = "pose.spawn_camera"
    bl_label = "Spawn Tracking Camera"
    bl_options = {"REGISTER", "UNDO"}

    make_active: BoolProperty(name="Set as Scene Camera", default=True)
    set_resolution: BoolProperty(name="Match Render Resolution", default=True)

    def execute(self, context):
        s = context.scene.bodymocap
        target = s.target
        if target is None and context.active_object and context.active_object.type == "ARMATURE":
            target = context.active_object
        cam = spawn_tracking_camera(context, target, s.tracking_fov, s.capture_width,
                                    s.capture_height, s.camera_distance, self.make_active,
                                    self.set_resolution)
        self.report({"INFO"}, f"{cam.name}: {s.tracking_fov:.0f} deg FOV, "
                              f"{s.capture_width}x{s.capture_height}")
        return {"FINISHED"}


CLASSES = (POSE_OT_spawn_camera,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
