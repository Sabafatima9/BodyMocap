"""Dependency install / pose-model download operators (run on a worker thread
in the UI so Blender stays responsive; synchronous in background mode)."""

from __future__ import annotations

import threading

import bpy
from bpy.props import EnumProperty, StringProperty
from bpy.types import Operator

from ..utils import deps


class _ThreadedJob:
    _timer = None
    _thread = None
    _result = None

    def _start(self, context, fn):
        if bpy.app.background or context.window is None:
            self._result = fn()
            return self._done(context)
        self._result = None

        def run():
            self._result = fn()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if self._thread is not None and self._thread.is_alive():
            context.scene.bodymocap.status_text = self.bl_label + "..."
            return {"PASS_THROUGH"}
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        return self._done(context)


class POSE_OT_install_mocap_dependencies(_ThreadedJob, Operator):
    """Install OpenCV + MediaPipe into Blender's Python (downloads from PyPI)"""

    bl_idname = "pose.install_mocap_dependencies"
    bl_label = "Install Dependencies"

    target_dir: StringProperty(name="Target Directory", default="", options={"SKIP_SAVE"},
                               description="Override install directory (default: automatic)")
    packages: StringProperty(name="Packages", default="", options={"SKIP_SAVE"},
                             description="Comma separated pip requirements (default: missing ones)")

    def execute(self, context):
        reqs = [p.strip() for p in self.packages.split(",") if p.strip()] or None
        target = bpy.path.abspath(self.target_dir) if self.target_dir else None
        if target is None:
            from ..runtime import addon_prefs
            prefs = addon_prefs(context)
            if prefs is not None and prefs.install_location == "USER":
                target = deps.user_site_packages()
        self.report({"INFO"}, "Installing Python packages (this can take a few minutes)...")
        return self._start(context, lambda: deps.run_install(reqs, target))

    def _done(self, context):
        ok, out = self._result or (False, "no result")
        s = context.scene.bodymocap
        s.deps_status = deps.dependency_summary()
        s.status_text = "Dependencies installed" if ok else "Dependency install failed"
        if ok:
            self.report({"INFO"}, "Dependencies ready: " + s.deps_status)
            return {"FINISHED"}
        self.report({"ERROR"}, "pip failed: " + out.strip().splitlines()[-1][:200] if out.strip() else "pip failed")
        return {"CANCELLED"}


class POSE_OT_download_pose_model(_ThreadedJob, Operator):
    """Download the MediaPipe pose landmarker model (Google, Apache-2.0)"""

    bl_idname = "pose.download_pose_model"
    bl_label = "Download Pose Model"

    variant: EnumProperty(name="Variant", items=[("CURRENT", "Current", ""), ("LITE", "Lite", ""),
                                                 ("FULL", "Full", ""), ("HEAVY", "Heavy", "")],
                          default="CURRENT", options={"SKIP_SAVE"})

    def execute(self, context):
        from ..runtime import addon_prefs
        s = context.scene.bodymocap
        v = (s.model_variant if self.variant == "CURRENT" else self.variant).lower()
        prefs = addon_prefs(context)
        directory = bpy.path.abspath(prefs.model_directory) if (prefs and prefs.model_directory) else ""
        self.report({"INFO"}, f"Downloading pose_landmarker_{v} (~{deps.MODEL_SIZES_MB[v]:.0f} MB)...")
        return self._start(context, lambda: deps.download_model(v, directory))

    def _done(self, context):
        ok, msg = self._result or (False, "no result")
        context.scene.bodymocap.status_text = "Model ready" if ok else "Model download failed"
        if ok:
            self.report({"INFO"}, f"Model saved: {msg}")
            return {"FINISHED"}
        self.report({"ERROR"}, msg)
        return {"CANCELLED"}


class POSE_OT_refresh_mocap_dependencies(Operator):
    bl_idname = "pose.refresh_mocap_dependencies"
    bl_label = "Refresh"

    def execute(self, context):
        context.scene.bodymocap.deps_status = deps.dependency_summary()
        self.report({"INFO"}, context.scene.bodymocap.deps_status)
        return {"FINISHED"}


CLASSES = (POSE_OT_install_mocap_dependencies, POSE_OT_download_pose_model,
           POSE_OT_refresh_mocap_dependencies)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
