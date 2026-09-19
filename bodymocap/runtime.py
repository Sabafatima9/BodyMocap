"""Process-wide runtime state and the per-sample processing path.

The modal capture operator (interactive), the blocking/offline path (background
mode, video files) and the test-suite all call :func:`process_sample`, so the
exact same code filters, retargets, applies and records every frame.
"""

from __future__ import annotations

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import bpy

from .capture.session import CaptureConfig, CaptureSample, CaptureSession
from .core.filters import OneEuroParams
from .core.skeleton import SkeletonFilter, SourceSkeleton, build_calibration
from .core.types import CalibrationData, RestPoseStyle, TrackingState, Vec3
from .recording.session import RecordingSession, Take
from .retarget.rig import RigModel
from .retarget.solver import RetargetSolver, SolverSettings
from .retarget.topology import TopologyProfile, detect_topology
from .utils.logging_util import log_info, log_warning
from .vision.preprocess import ExposureSettings

PROFILE_KEY = "bodymocap_profile"


class Runtime:
    def __init__(self):
        self.session: Optional[CaptureSession] = None
        self.filter: Optional[SkeletonFilter] = None
        self.recording = RecordingSession()
        self.last_take: Optional[Take] = None
        self.calibration: Optional[CalibrationData] = None
        self.calib_frames: List[SourceSkeleton] = []
        self.calib_until: float = 0.0
        self.calib_seconds: float = 0.0
        self.solvers: Dict[str, Tuple[str, RetargetSolver]] = {}
        self.last_sample: Optional[CaptureSample] = None
        self.last_skeleton: Optional[SourceSkeleton] = None
        self.stop_requested = False
        self.preview_every = 2
        self.frames_processed = 0
        self.apply_ms = 0.0
        self.solve_ms = 0.0
        self.started_at = 0.0
        self.last_session_stats: Dict[str, float] = {}
        self.last_bake: Dict[str, object] = {}

    @property
    def capturing(self) -> bool:
        return self.session is not None

    def reset_live(self, settings) -> None:
        self.filter = SkeletonFilter(smoothing_params(settings), min_conf=settings.min_visibility)
        self.solvers.clear()
        self.frames_processed = 0
        self.apply_ms = self.solve_ms = 0.0


_RT: Optional[Runtime] = None


def get_runtime() -> Runtime:
    global _RT
    if _RT is None:
        _RT = Runtime()
    return _RT


def shutdown_runtime() -> None:
    global _RT
    if _RT is not None and _RT.session is not None:
        try:
            _RT.session.close()
        except Exception:
            pass
    _RT = None


# ---------------------------------------------------------------------------
# Settings -> config objects
# ---------------------------------------------------------------------------

def addon_prefs(context=None):
    context = context or bpy.context
    pkg = __package__ or "bodymocap"
    addon = context.preferences.addons.get(pkg)
    return addon.preferences if addon else None


def smoothing_params(s) -> OneEuroParams:
    if not s.smoothing_enabled:
        return OneEuroParams(min_cutoff=0.0)
    return OneEuroParams(min_cutoff=s.smooth_min_cutoff, beta=s.smooth_beta, d_cutoff=3.0)


def exposure_settings(s) -> ExposureSettings:
    return ExposureSettings(mode=s.exposure_mode, gain_ev=s.exposure_ev, gamma=s.exposure_gamma,
                            contrast=s.exposure_contrast, clahe=s.exposure_clahe,
                            clahe_clip=s.exposure_clahe_clip)


def solver_settings(s) -> SolverSettings:
    return SolverSettings(
        min_conf=s.min_visibility,
        knee_forward=s.anti_knee_inversion,
        twist_share=s.twist_share,
        drive_hands=s.drive_hands,
        drive_feet=s.drive_feet,
        drive_head=s.drive_head,
        spine_position_match=s.spine_position_match,
        pelvis_tilt_share=s.pelvis_tilt_share,
        root_motion=s.root_motion,
        root_scale=s.root_scale,
        pole_blend_lo=math.radians(s.pole_blend_lo),
        pole_blend_hi=math.radians(s.pole_blend_hi),
    )


def model_file(s, context=None) -> str:
    from .utils import deps
    prefs = addon_prefs(context)
    directory = bpy.path.abspath(prefs.model_directory) if (prefs and prefs.model_directory) else ""
    return deps.model_path(s.model_variant.lower(), directory)


def capture_config(s, context=None, offline: bool = False) -> CaptureConfig:
    source = s.source
    backend = "SYNTHETIC" if source in ("SYNTHETIC", "TAKE") else "MEDIAPIPE"
    path = ""
    if source == "VIDEO":
        path = bpy.path.abspath(s.video_path)
    elif source == "IMAGES":
        path = bpy.path.abspath(s.image_dir)
    elif source == "TAKE":
        path = bpy.path.abspath(s.take_path)
    return CaptureConfig(
        source=source, device_index=s.device_index, path=path,
        width=s.capture_width, height=s.capture_height, fps=s.capture_fps, loop=s.loop_source,
        mirror=s.mirror_motion, backend=backend, model_path=model_file(s, context),
        running_mode="VIDEO", min_detection=s.min_detection, min_presence=s.min_presence,
        min_tracking=s.min_tracking, min_visibility=s.min_visibility,
        exposure=exposure_settings(s), synthetic_clip=s.synth_clip,
        synthetic_condition=s.synth_condition, synthetic_speed=s.synth_speed,
        synthetic_duration=s.synth_duration if offline else 0.0,
        hfov=math.radians(s.tracking_fov), realtime=True,
    )


# ---------------------------------------------------------------------------
# Targets / profiles / solvers
# ---------------------------------------------------------------------------

def target_objects(s, include_extra: bool = True) -> List[bpy.types.Object]:
    out = []
    if s.target is not None and s.target.type == "ARMATURE":
        out.append(s.target)
    if include_extra:
        for t in s.extra_targets:
            o = t.obj
            if t.enabled and o is not None and o.type == "ARMATURE" and o not in out:
                out.append(o)
    return out


def rig_signature(obj) -> str:
    """Changes whenever the bone set / rest layout changes."""
    parts = [obj.data.name, str(len(obj.data.bones))]
    for b in obj.data.bones:
        h, t = b.head_local, b.tail_local
        parts.append(f"{b.name}:{h.x:.4f},{h.y:.4f},{h.z:.4f},{t.x:.4f},{t.y:.4f},{t.z:.4f}")
    return str(hash("|".join(parts)))


def get_profile(obj, rig: Optional[RigModel] = None, redetect: bool = False) -> TopologyProfile:
    rig = rig or RigModel.from_blender(obj)
    raw = obj.get(PROFILE_KEY)
    if raw and not redetect:
        try:
            prof = TopologyProfile.from_dict(json.loads(raw))
            if not prof.validate(rig):
                return prof
            log_warning(f"Stored profile on {obj.name} no longer matches the armature; re-detecting")
        except Exception as exc:
            log_warning(f"Could not parse stored profile on {obj.name}: {exc}")
    prof = detect_topology(rig, name=obj.name)
    store_profile(obj, prof)
    return prof


def store_profile(obj, prof: TopologyProfile) -> None:
    obj[PROFILE_KEY] = json.dumps(prof.to_dict())


def get_solver(obj, s) -> Optional[RetargetSolver]:
    """Cached solver per target; rebuilt when settings, calibration or the
    armature's bone count change (Detect Topology also clears the cache)."""
    rt = get_runtime()
    sig = f"{obj.data.name}|{len(obj.data.bones)}|{id(rt.calibration)}|" + \
        json.dumps(solver_settings(s).__dict__, sort_keys=True, default=str)
    entry = rt.solvers.get(obj.name)
    if entry and entry[0] == sig:
        return entry[1]
    rig = RigModel.from_blender(obj)
    prof = get_profile(obj, rig)
    if prof.validate(rig):
        return None
    solver = RetargetSolver(rig, prof, solver_settings(s), rt.calibration)
    rt.solvers[obj.name] = (sig, solver)
    return solver


def apply_result(obj, result) -> int:
    """Write solver output onto pose bones (live preview)."""
    pbs = obj.pose.bones
    n = 0
    for name, q in result.local_rot.items():
        pb = pbs.get(name)
        if pb is None:
            continue
        if pb.rotation_mode != "QUATERNION":
            pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = (q.w, q.x, q.y, q.z)
        n += 1
    for name, t in result.root_loc.items():
        pb = pbs.get(name)
        if pb is not None:
            pb.location = (t.x, t.y, t.z)
    return n


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def process_sample(context, sample: CaptureSample, apply: bool = True) -> Optional[SourceSkeleton]:
    s = context.scene.bodymocap
    rt = get_runtime()
    if rt.filter is None:
        rt.reset_live(s)
    rt.last_sample = sample
    raw = sample.skeleton
    state = sample.pose.tracking_state
    filtered = rt.filter.process(raw) if raw.joints else None
    rt.last_skeleton = filtered

    # status
    s.tracking_status = state.name
    if sample.stats is not None:
        s.lighting_status = sample.stats.condition
    if rt.session is not None:
        s.capture_fps_live = rt.session.fps

    # calibration window
    if rt.calib_until > 0.0 and raw.joints:
        rt.calib_frames.append(raw)
        if time.monotonic() >= rt.calib_until or len(rt.calib_frames) >= int(rt.calib_seconds * 120):
            finish_calibration(s)

    # recording: raw landmarks (filtering happens at bake time)
    if rt.recording.is_recording and raw.joints:
        rt.recording.append(raw, state)
        s.record_frame_count = rt.recording.frame_count()

    if apply and s.live_apply and filtered is not None and state != TrackingState.LOST:
        t0 = time.perf_counter()
        for obj in target_objects(s, include_extra=s.live_apply_all_targets):
            solver = get_solver(obj, s)
            if solver is None:
                continue
            t1 = time.perf_counter()
            res = solver.solve(filtered, compute_pose=False)
            t2 = time.perf_counter()
            apply_result(obj, res)
            rt.solve_ms += ((t2 - t1) * 1e3 - rt.solve_ms) * 0.1
        rt.apply_ms += ((time.perf_counter() - t0) * 1e3 - rt.apply_ms) * 0.1
    rt.frames_processed += 1
    return filtered


def start_calibration(s, seconds: float) -> None:
    rt = get_runtime()
    rt.calib_frames = []
    rt.calib_seconds = seconds
    rt.calib_until = time.monotonic() + seconds
    s.calibration_status = "Calibrating... hold your rest pose"


def finish_calibration(s) -> Optional[CalibrationData]:
    rt = get_runtime()
    frames = rt.calib_frames
    rt.calib_until = 0.0
    style = RestPoseStyle.T_POSE if s.rest_pose_style == "T_POSE" else RestPoseStyle.A_POSE
    cal = build_calibration(frames, min_conf=s.min_visibility, rest_style=style)
    if not cal.valid:
        s.calibration_status = "Calibration failed (no body detected)"
        s.is_calibrated = False
        return None
    rt.calibration = cal
    rt.solvers.clear()
    s.is_calibrated = True
    s.calibration_status = f"Calibrated ({cal.samples} frames, leg {cal.scale:.2f} m)"
    log_info(s.calibration_status)
    return cal


def calibration_from_take(take: Take, seconds: float, min_conf: float = 0.5) -> Optional[CalibrationData]:
    if not take.frames:
        return None
    t0 = take.frames[0].timestamp
    frames = [f for f in take.frames if f.timestamp - t0 <= seconds]
    cal = build_calibration(frames, min_conf=min_conf)
    return cal if cal.valid else None
