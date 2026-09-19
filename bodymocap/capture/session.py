"""Capture pipeline: frame source -> exposure compensation -> pose backend ->
source skeleton (+ metric root translation).  bpy-free.

``step()`` runs one frame synchronously (used for offline/background
processing and tests); ``start_thread()`` runs the same loop on a worker thread
so camera reads and inference never block Blender's UI -- the modal operator
just picks up the latest sample on its timer.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from ..core.confidence import ConfidenceConfig
from ..core.root_motion import solve_root
from ..core.skeleton import SourceSkeleton
from ..core.types import PoseFrame, TrackingState
from ..utils.logging_util import log_error, log_info
from ..vision.preprocess import ExposureSettings, FrameStats, compensate
from ..vision.sources import FrameSource, make_source


@dataclass
class CaptureConfig:
    source: str = "WEBCAM"           # WEBCAM | VIDEO | IMAGES | SYNTHETIC | TAKE
    device_index: int = 0
    path: str = ""
    width: int = 640
    height: int = 480
    fps: float = 30.0
    loop: bool = False
    mirror: bool = False
    backend: str = "MEDIAPIPE"       # MEDIAPIPE | SYNTHETIC
    model_path: str = ""
    running_mode: str = "VIDEO"
    min_detection: float = 0.5
    min_presence: float = 0.5
    min_tracking: float = 0.5
    min_visibility: float = 0.5
    exposure: ExposureSettings = field(default_factory=ExposureSettings)
    synthetic_clip: str = "wave"
    synthetic_condition: str = "normal"
    synthetic_speed: float = 1.0
    synthetic_duration: float = 0.0   # >0: finite synthetic source (offline runs)
    seed: int = 0
    hfov: float = math.radians(60.0)
    realtime: bool = True
    keep_frames: bool = True


@dataclass
class CaptureSample:
    index: int
    timestamp: float
    pose: PoseFrame
    skeleton: SourceSkeleton
    frame: Any = None
    stats: Optional[FrameStats] = None
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def detected(self) -> bool:
        return bool(self.pose.landmarks) and self.pose.tracking_state != TrackingState.LOST


class CaptureSession:
    def __init__(self, config: CaptureConfig):
        self.cfg = config
        self.source: Optional[FrameSource] = None
        self.backend = None
        self.error = ""
        self.index = 0
        self.stats = {"frames": 0, "detected": 0, "read_ms": 0.0, "preprocess_ms": 0.0,
                      "infer_ms": 0.0, "post_ms": 0.0, "total_ms": 0.0}
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[CaptureSample] = None
        self._fresh = False
        self._fps_ema = 0.0
        self._last_wall: Optional[float] = None
        self.exhausted = False

    # ------------------------------------------------------------------
    def _make_backend(self):
        c = self.cfg
        conf = ConfidenceConfig(min_confidence=c.min_visibility)
        if c.backend == "SYNTHETIC" or c.source in ("SYNTHETIC", "TAKE"):
            from ..pose.mock_backend import MockBackend
            mode = "fixture" if c.source == "TAKE" else "clip"
            be = MockBackend(mode=mode, clip=c.synthetic_clip, condition=c.synthetic_condition,
                             speed=c.synthetic_speed, seed=c.seed,
                             fixture_path=c.path if c.source == "TAKE" else None,
                             confidence_cfg=conf)
        else:
            from ..pose.mediapipe_backend import MediaPipeBackend
            be = MediaPipeBackend(model_path=c.model_path, running_mode=c.running_mode,
                                  min_detection=c.min_detection, min_presence=c.min_presence,
                                  min_tracking=c.min_tracking, confidence_cfg=conf)
        if not be.initialize():
            self.error = getattr(be, "last_error", "") or "Pose backend failed to initialise"
            return None
        return be

    def open(self) -> bool:
        c = self.cfg
        kind = c.source if c.source in ("WEBCAM", "VIDEO", "IMAGES") else "NONE"
        n = int(round(c.synthetic_duration * c.fps)) if (kind == "NONE" and c.synthetic_duration > 0) else -1
        if c.source == "TAKE" and not c.loop:
            n = self._take_length(c.path)
        self.source = make_source(kind, device_index=c.device_index, width=c.width,
                                  height=c.height, fps=c.fps, path=c.path, loop=c.loop,
                                  frame_count=n)
        if not self.source.open():
            self.error = self.source.last_error or "Could not open frame source"
            self.source = None
            return False
        self.backend = self._make_backend()
        if self.backend is None:
            self.source.close()
            self.source = None
            return False
        self.index = 0
        self.exhausted = False
        log_info(f"Capture opened: source={c.source} backend={getattr(self.backend, 'name', '?')}")
        return True

    @staticmethod
    def _take_length(path: str) -> int:
        try:
            from ..pose.mock_backend import load_landmark_file
            return len(load_landmark_file(path))
        except Exception:
            return -1

    def close(self) -> None:
        self.stop_thread()
        if self.source is not None:
            self.source.close()
            self.source = None
        if self.backend is not None:
            try:
                self.backend.shutdown()
            except Exception:
                pass
            self.backend = None

    # ------------------------------------------------------------------
    def step(self) -> Optional[CaptureSample]:
        """Process one frame. Returns None when the source is exhausted/failed."""
        if self.source is None or self.backend is None:
            return None
        c = self.cfg
        if self.source.exhausted:
            self.exhausted = True
            return None
        t0 = time.perf_counter()
        ok, frame, ts = self.source.read()
        t1 = time.perf_counter()
        if not ok:
            if self.source.exhausted or self.source.kind in ("VIDEO", "IMAGES"):
                self.exhausted = True
            else:
                self.error = self.source.last_error or "Frame read failed"
            return None
        stats = None
        if frame is not None:
            if c.mirror:
                frame = frame[:, ::-1]
            frame, stats = compensate(frame, c.exposure)
        t2 = time.perf_counter()
        pose = self.backend.infer(frame, frame_index=self.index, timestamp=ts)
        t3 = time.perf_counter()
        sk = SourceSkeleton.from_pose_frame(pose)
        if pose.landmarks:
            if pose.image_size:
                w, h = pose.image_size
            elif frame is not None:
                h, w = frame.shape[:2]
            else:
                w, h = c.width, c.height
            root, rconf = solve_root(pose.landmarks, c.hfov, w, h, min_conf=c.min_visibility)
            if root is not None:
                sk.root, sk.root_conf = root, rconf
        if c.mirror and frame is None:
            sk = sk.mirrored()
        t4 = time.perf_counter()
        sample = CaptureSample(
            index=self.index, timestamp=ts, pose=pose, skeleton=sk,
            frame=frame if c.keep_frames else None, stats=stats,
            timings={"read": (t1 - t0) * 1e3, "preprocess": (t2 - t1) * 1e3,
                     "infer": (t3 - t2) * 1e3, "post": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3},
        )
        self.index += 1
        st = self.stats
        st["frames"] += 1
        st["detected"] += 1 if sample.detected else 0
        for k in ("read", "preprocess", "infer", "post", "total"):
            key = f"{k}_ms"
            st[key] += (sample.timings[k] - st[key]) / st["frames"]
        now = time.perf_counter()
        if self._last_wall is not None:
            dt = now - self._last_wall
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
        self._last_wall = now
        return sample

    @property
    def fps(self) -> float:
        return self._fps_ema

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def start_thread(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="BodyMocapCapture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        pace = (self.cfg.realtime and self.source is not None and
                (self.source.kind in ("VIDEO", "IMAGES", "NONE")))
        period = 1.0 / max(self.source.fps if self.source else self.cfg.fps, 1.0)
        next_t = time.perf_counter()
        while not self._stop.is_set():
            try:
                sample = self.step()
            except Exception as exc:  # keep the UI alive, surface the error
                self.error = f"Capture error: {exc}"
                log_error(self.error)
                break
            if sample is None:
                if self.exhausted or self.error:
                    break
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = sample
                self._fresh = True
            if pace:
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_t = time.perf_counter()

    def stop_thread(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest(self, consume: bool = True) -> Optional[CaptureSample]:
        with self._lock:
            if not self._fresh:
                return None
            if consume:
                self._fresh = False
            return self._latest


def run_offline(cfg: CaptureConfig, max_frames: int = 100000) -> Tuple[list, Dict[str, float], str]:
    """Process a finite source synchronously. Returns (samples, stats, error)."""
    cfg.realtime = False
    sess = CaptureSession(cfg)
    if not sess.open():
        return [], sess.stats, sess.error
    out = []
    try:
        while len(out) < max_frames:
            s = sess.step()
            if s is None:
                break
            out.append(s)
    finally:
        sess.close()
    return out, dict(sess.stats), sess.error
