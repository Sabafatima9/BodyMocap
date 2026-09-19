"""BodyMocap: camera pose capture with topology-adaptive retargeting for Blender.

Works as a legacy add-on (``bl_info``) and as a Blender 4.2+ extension
(``blender_manifest.toml``).  Heavy optional dependencies (OpenCV, MediaPipe)
are imported lazily so registration never fails when they are missing.
"""

bl_info = {
    "name": "BodyMocap",
    "author": "BodyMocap Project",
    "version": (2, 1, 0),
    "blender": (4, 2, 0),
    "location": "View3D > Sidebar > Mocap",
    "description": (
        "Webcam / video pose capture, one-click record-to-Action and "
        "topology-adaptive (N:M bone chain) retargeting"
    ),
    "category": "Animation",
    "doc_url": "",
    "tracker_url": "",
}

if "bpy" in locals():  # support F3 > "Reload Scripts"
    import importlib
    import sys

    for _name in sorted([m for m in sys.modules if m.startswith(__name__ + ".")], reverse=True):
        importlib.reload(sys.modules[_name])

try:  # noqa: E402
    import bpy
except ImportError:  # pure-Python test / tooling context (NFR-007)
    bpy = None

_MODULES = (
    "preferences",
    "properties",
    "ui.lists",
    "ui.panels",
    "operators.camera_ops",
    "operators.capture_ops",
    "operators.bake_ops",
    "operators.retarget_ops",
    "operators.deps_ops",
    "operators.rig_ops",
)


def _mods():
    import importlib
    return [importlib.import_module(f"{__name__}.{m}") for m in _MODULES]


def _init_scenes():
    try:
        from .utils import deps
        summary = deps.dependency_summary()
        for scene in bpy.data.scenes:
            scene.bodymocap.deps_status = summary
    except Exception:
        pass
    return None


def register():
    from .utils import deps
    from .utils.logging_util import log_info

    deps.ensure_user_site_on_path()
    for m in _mods():
        m.register()
    bpy.app.timers.register(_init_scenes, first_interval=0.1)
    log_info("BodyMocap registered")


def unregister():
    from .runtime import shutdown_runtime
    from .utils.logging_util import log_info

    try:
        shutdown_runtime()
    except Exception:
        pass
    if bpy.app.timers.is_registered(_init_scenes):
        bpy.app.timers.unregister(_init_scenes)
    for m in reversed(_mods()):
        m.unregister()
    log_info("BodyMocap unregistered")
