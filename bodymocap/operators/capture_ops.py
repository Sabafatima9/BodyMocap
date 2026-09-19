"""Tracking, one-click recording and calibration operators.

POSE_OT_start_capture runs modally in the UI (capture + inference on a worker
thread, retarget/apply on a 60 Hz timer) and synchronously when Blender runs
in background mode or with ``offline=True`` (video/image/synthetic sources),
which is how the headless test-suite drives the full pipeline.
"""

from __future__ import annotations

import math
import time

import bpy
from bpy.props import BoolProperty, FloatProperty, IntProperty
from bpy.types import Operator

from ..camera.preview import frame_to_image, tag_redraw_all
from ..overlay.draw import blank_canvas, draw_skeleton
from ..runtime import (
    capture_config,
    finish_calibration,
    get_runtime,
    process_sample,
    start_calibration,
)
from ..utils.logging_util import log_info


def _check_backend_ready(op, s, cfg) -> bool:
    if cfg.backend != "MEDIAPIPE":
        return True
    from ..utils import deps
    missing = deps.missing_requirements()
    if missing:
        op.report({"ERROR"}, "Missing Python packages: " + ", ".join(missing) +
                  ". Use 'Install Dependencies' (Tracking > Dependencies) or choose the Synthetic source.")
        return False
    import os
    if not os.path.isfile(cfg.model_path):
        op.report({"ERROR"}, f"Pose model '{s.model_variant.lower()}' not downloaded. "
                             "Use 'Download Pose Model'.")
        return False
    return True


def start_recording(context) -> None:
    s = context.scene.bodymocap
    rt = get_runtime()
    cfg = rt.session.cfg if rt.session else None
    rt.recording.degraded_warn_fraction = s.degraded_warn_fraction
    rt.recording.start(name=s.action_name, meta={
        "source": s.source, "backend": cfg.backend if cfg else "",
        "model": s.model_variant, "fov": s.tracking_fov, "mirror": s.mirror_motion,
        "blender": bpy.app.version_string,
    })
    s.is_recording = True
    s.record_frame_count = 0


def stop_recording(context, op=None, bake: bool = True):
    s = context.scene.bodymocap
    rt = get_runtime()
    if not rt.recording.is_recording:
        return None
    take = rt.recording.stop()
    s.is_recording = False
    rt.last_take = take
    s.take_frame_count = take.frame_count()
    if op is not None:
        if rt.recording.should_warn_tracking():
            op.report({"WARNING"}, rt.recording.tracking_warning_message())
        op.report({"INFO"}, f"Recorded {take.frame_count()} frames ({take.duration:.1f}s)")
    if bake and s.auto_bake and take.frame_count() > 1:
        bpy.ops.pose.bake_animation()
    return take


class _CaptureModal:
    """Shared modal/blocking driver."""

    _timer = None

    def _update_preview(self, context, sample, force=False):
        s = context.scene.bodymocap
        rt = get_runtime()
        if not force and rt.frames_processed % rt.preview_every:
            return
        frame = sample.frame
        if frame is None:
            frame = blank_canvas(s.capture_width, s.capture_height)
        label = f"{sample.pose.tracking_state.name}  {rt.session.fps if rt.session else 0:.0f} fps"
        if s.is_recording:
            label = "REC  " + label
        drawn = draw_skeleton(frame, sample.pose.landmarks, s.min_visibility, s.show_overlay, label)
        frame_to_image(drawn, flip_x=s.mirror_preview and s.source == "WEBCAM")

    def _finish(self, context, cancelled=False):
        s = context.scene.bodymocap
        rt = get_runtime()
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        sess = rt.session
        err = sess.error if sess else ""
        stats = dict(sess.stats) if sess else {}
        if sess is not None:
            sess.close()
        rt.session = None
        rt.last_session_stats = stats
        s.is_capturing = False
        s.tracking_status = "Idle"
        if rt.calib_until > 0.0:
            finish_calibration(s)
        if rt.recording.is_recording:
            stop_recording(context, self, bake=True)
        if err:
            self.report({"WARNING"}, err)
        log_info(f"Capture finished: {stats.get('frames', 0)} frames, "
                 f"{stats.get('detected', 0)} detected, infer {stats.get('infer_ms', 0):.1f} ms")
        tag_redraw_all()
        return {"CANCELLED"} if cancelled else {"FINISHED"}


class POSE_OT_start_capture(_CaptureModal, Operator):
    """Start pose tracking from the selected source and drive the target rig live"""

    bl_idname = "pose.start_capture"
    bl_label = "Start Tracking"
    bl_options = {"REGISTER"}

    record: BoolProperty(name="Record", default=False, options={"SKIP_SAVE"})
    offline: BoolProperty(name="Process Offline", default=False, options={"SKIP_SAVE"},
                          description="Process the whole source synchronously (no UI updates)")
    max_frames: IntProperty(name="Max Frames", default=0, min=0, options={"SKIP_SAVE"})
    calibrate_seconds: FloatProperty(name="Calibrate First", default=0.0, min=0.0,
                                     options={"SKIP_SAVE"},
                                     description="Offline: use the first N seconds as calibration")

    @classmethod
    def poll(cls, context):
        return get_runtime().session is None

    def execute(self, context):
        s = context.scene.bodymocap
        rt = get_runtime()
        if rt.session is not None:
            self.report({"WARNING"}, "Tracking already running")
            return {"CANCELLED"}
        from ..capture.session import CaptureSession
        offline = self.offline or bpy.app.background or context.window is None
        cfg = capture_config(s, context, offline=offline)
        if not _check_backend_ready(self, s, cfg):
            return {"CANCELLED"}
        if offline and s.source == "WEBCAM" and self.max_frames <= 0:
            self.report({"ERROR"}, "Offline webcam capture needs max_frames > 0")
            return {"CANCELLED"}
        if offline:
            cfg.realtime = False
        sess = CaptureSession(cfg)
        if not sess.open():
            self.report({"ERROR"}, sess.error or "Could not start capture")
            return {"CANCELLED"}
        rt.session = sess
        rt.reset_live(s)
        rt.stop_requested = False
        rt.started_at = time.monotonic()
        s.is_capturing = True
        if s.target is None:
            self.report({"WARNING"}, "No target armature set: tracking without live preview")
        if offline:
            return self._run_blocking(context)
        if self.record:
            start_recording(context)
        sess.start_thread()
        wm = context.window_manager
        self._timer = wm.event_timer_add(1.0 / 60.0, window=context.window)
        wm.modal_handler_add(self)
        self.report({"INFO"}, f"Tracking started ({s.source})")
        return {"RUNNING_MODAL"}

    def _run_blocking(self, context):
        s = context.scene.bodymocap
        rt = get_runtime()
        sess = rt.session
        n = 0
        calib_frames = int(self.calibrate_seconds * sess.cfg.fps) if self.calibrate_seconds > 0 else 0
        if calib_frames:
            rt.calib_frames = []
            rt.calib_until = float("inf")
            rt.calib_seconds = 1e9
        recording_started = False
        while True:
            if self.max_frames and n >= self.max_frames:
                break
            sample = sess.step()
            if sample is None:
                break
            if calib_frames and n == calib_frames:
                finish_calibration(s)
                rt.reset_live(s)
            if self.record and not recording_started and n >= calib_frames:
                start_recording(context)
                recording_started = True
            process_sample(context, sample)
            n += 1
        if calib_frames and rt.calib_until > 0.0:
            finish_calibration(s)
        if s.save_debug_video and rt.last_sample is not None:
            self._update_preview(context, rt.last_sample, force=True)
        return self._finish(context)

    def modal(self, context, event):
        rt = get_runtime()
        if rt.session is None:
            return self._finish(context)
        if event.type == "ESC" and event.value == "PRESS":
            return self._finish(context, cancelled=True)
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        if rt.stop_requested:
            return self._finish(context)
        sample = rt.session.latest()
        if sample is not None:
            process_sample(context, sample)
            self._update_preview(context, sample)
            tag_redraw_all()
        if rt.session.exhausted and rt.session.latest(consume=False) is None and not rt.session.running:
            return self._finish(context)
        if rt.session.error and not rt.session.running:
            self.report({"ERROR"}, rt.session.error)
            return self._finish(context)
        return {"PASS_THROUGH"}


class POSE_OT_stop_capture(Operator):
    """Stop tracking (and finish any recording)"""

    bl_idname = "pose.stop_capture"
    bl_label = "Stop Tracking"

    @classmethod
    def poll(cls, context):
        return get_runtime().session is not None

    def execute(self, context):
        get_runtime().stop_requested = True
        return {"FINISHED"}


class POSE_OT_record_toggle(Operator):
    """One click: start tracking + recording; click again to stop, bake to an
    Action and apply it to the target rig(s)"""

    bl_idname = "pose.record_toggle"
    bl_label = "Record"
    bl_options = {"REGISTER"}

    def execute(self, context):
        s = context.scene.bodymocap
        rt = get_runtime()
        if rt.recording.is_recording:
            stop_recording(context, self, bake=True)
            return {"FINISHED"}
        if rt.session is None:
            # start tracking with recording enabled (blocking in background)
            res = bpy.ops.pose.start_capture("EXEC_DEFAULT", record=True)
            return {"FINISHED"} if res & {"FINISHED", "RUNNING_MODAL"} else {"CANCELLED"}
        start_recording(context)
        self.report({"INFO"}, "Recording")
        return {"FINISHED"}


class POSE_OT_calibrate_capture(Operator):
    """Capture the performer's neutral pose (hold the rig's rest pose: T or A,
    palms down, looking ahead) to calibrate hands, feet, head and root motion"""

    bl_idname = "pose.calibrate_capture"
    bl_label = "Calibrate"

    def execute(self, context):
        s = context.scene.bodymocap
        rt = get_runtime()
        if rt.session is not None and not (bpy.app.background or context.window is None):
            start_calibration(s, s.calibration_seconds)
            self.report({"INFO"}, f"Hold the rest pose for {s.calibration_seconds:.1f}s")
            return {"FINISHED"}
        # offline: process N frames of the configured source
        from ..capture.session import CaptureSession
        cfg = capture_config(s, context, offline=True)
        if not _check_backend_ready(self, s, cfg):
            return {"CANCELLED"}
        cfg.realtime = False
        sess = CaptureSession(cfg)
        if not sess.open():
            self.report({"ERROR"}, sess.error)
            return {"CANCELLED"}
        rt.calib_frames = []
        rt.calib_until = float("inf")
        try:
            for _ in range(max(3, int(s.calibration_seconds * cfg.fps))):
                sample = sess.step()
                if sample is None:
                    break
                if sample.skeleton.joints:
                    rt.calib_frames.append(sample.skeleton)
        finally:
            sess.close()
        cal = finish_calibration(s)
        if cal is None:
            self.report({"ERROR"}, "Calibration failed: no body detected")
            return {"CANCELLED"}
        self.report({"INFO"}, s.calibration_status)
        return {"FINISHED"}


class POSE_OT_clear_calibration(Operator):
    bl_idname = "pose.clear_mocap_calibration"
    bl_label = "Clear Calibration"

    def execute(self, context):
        rt = get_runtime()
        rt.calibration = None
        rt.solvers.clear()
        s = context.scene.bodymocap
        s.is_calibrated = False
        s.calibration_status = "Not calibrated"
        return {"FINISHED"}


class POSE_OT_discard_take(Operator):
    bl_idname = "pose.discard_take"
    bl_label = "Discard Take"

    def execute(self, context):
        rt = get_runtime()
        rt.recording.discard()
        rt.last_take = None
        s = context.scene.bodymocap
        s.is_recording = False
        s.record_frame_count = 0
        s.take_frame_count = 0
        self.report({"INFO"}, "Take discarded")
        return {"FINISHED"}


CLASSES = (
    POSE_OT_start_capture,
    POSE_OT_stop_capture,
    POSE_OT_record_toggle,
    POSE_OT_calibrate_capture,
    POSE_OT_clear_calibration,
    POSE_OT_discard_take,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
