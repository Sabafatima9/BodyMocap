"""MediaPipe Pose backend (Tasks API, with legacy ``solutions`` fallback).

MediaPipe >= 0.10.30 ships only the Tasks API, which needs a ``.task`` model
file (lite / full / heavy).  Models are downloaded on explicit user request
(``utils.deps.download_model``) into the add-on's data directory.  Output world
landmarks are converted to capture space; normalised image landmarks are kept
for overlays and the root-translation solve.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from ..core.confidence import ConfidenceConfig, TrackingHysteresis
from ..core.landmarks import MEDIAPIPE_POSE_NAMES, mp_world_to_capture
from ..core.types import Landmark, PoseFrame, TrackingState, Vec3
from ..utils.logging_util import log_error, log_info
from .backend import PoseBackend


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except Exception:
        return False


def tasks_api_available() -> bool:
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return hasattr(vision, "PoseLandmarker")
    except Exception:
        return False


def legacy_api_available() -> bool:
    try:
        import mediapipe as mp
        return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    except Exception:
        return False


class MediaPipeBackend(PoseBackend):
    name = "mediapipe"

    def __init__(
        self,
        model_path: Optional[str] = None,
        running_mode: str = "VIDEO",
        min_detection: float = 0.5,
        min_presence: float = 0.5,
        min_tracking: float = 0.5,
        confidence_cfg: Optional[ConfidenceConfig] = None,
    ):
        self.model_path = model_path
        self.running_mode = running_mode
        self.min_detection = min_detection
        self.min_presence = min_presence
        self.min_tracking = min_tracking
        self._hyst = TrackingHysteresis(confidence_cfg or ConfidenceConfig())
        self._landmarker = None
        self._legacy = None
        self._mp = None
        self._last_ts_ms = -1
        self.api = ""
        self.last_error = ""

    def is_available(self) -> bool:
        return mediapipe_available()

    def initialize(self, **kwargs: Any) -> bool:
        if not mediapipe_available():
            self.last_error = "MediaPipe is not installed."
            return False
        try:
            import mediapipe as mp
            self._mp = mp
        except Exception as exc:
            self.last_error = f"MediaPipe import failed: {exc}"
            return False
        path = kwargs.get("model_path", self.model_path)
        if tasks_api_available() and path and os.path.isfile(path):
            try:
                from mediapipe.tasks.python import vision
                from mediapipe.tasks.python.core.base_options import BaseOptions
                mode = (vision.RunningMode.VIDEO if self.running_mode == "VIDEO"
                        else vision.RunningMode.IMAGE)
                opts = vision.PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=path),
                    running_mode=mode,
                    num_poses=1,
                    min_pose_detection_confidence=float(self.min_detection),
                    min_pose_presence_confidence=float(self.min_presence),
                    min_tracking_confidence=float(self.min_tracking),
                    output_segmentation_masks=False,
                )
                self._landmarker = vision.PoseLandmarker.create_from_options(opts)
                self.api = "tasks"
                self._last_ts_ms = -1
                log_info(f"MediaPipe PoseLandmarker ready ({os.path.basename(path)}, {self.running_mode})")
                return True
            except Exception as exc:
                self.last_error = f"PoseLandmarker init failed: {exc}"
                log_error(self.last_error)
        if legacy_api_available():
            try:
                self._legacy = self._mp.solutions.pose.Pose(
                    static_image_mode=self.running_mode != "VIDEO",
                    model_complexity=1,
                    min_detection_confidence=float(self.min_detection),
                    min_tracking_confidence=float(self.min_tracking),
                )
                self.api = "solutions"
                return True
            except Exception as exc:
                self.last_error = f"Legacy MediaPipe Pose init failed: {exc}"
                return False
        if not self.last_error:
            self.last_error = ("No pose model file. Use 'Download Pose Model' in the add-on "
                               "panel (MediaPipe >= 0.10.30 requires a .task model).")
        return False

    # ------------------------------------------------------------------
    def infer(self, frame_bgr: Any, frame_index: int = 0, timestamp: float = 0.0) -> PoseFrame:
        pf = PoseFrame(frame_index=frame_index, timestamp=timestamp)
        if frame_bgr is None:
            return pf
        h, w = frame_bgr.shape[:2]
        pf.image_size = (w, h)
        rgb = frame_bgr[:, :, 2::-1]
        try:
            import numpy as np
            rgb = np.ascontiguousarray(rgb)
        except ImportError:  # pragma: no cover
            pass
        world = img = None
        try:
            if self._landmarker is not None:
                mp = self._mp
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                if self.running_mode == "VIDEO":
                    ts = int(round(timestamp * 1000.0))
                    if ts <= self._last_ts_ms:
                        ts = self._last_ts_ms + 1
                    self._last_ts_ms = ts
                    res = self._landmarker.detect_for_video(image, ts)
                else:
                    res = self._landmarker.detect(image)
                if res.pose_world_landmarks:
                    world = res.pose_world_landmarks[0]
                    img = res.pose_landmarks[0] if res.pose_landmarks else None
            elif self._legacy is not None:
                res = self._legacy.process(rgb)
                if res.pose_world_landmarks is not None:
                    world = res.pose_world_landmarks.landmark
                    img = res.pose_landmarks.landmark if res.pose_landmarks else None
        except Exception as exc:
            self.last_error = f"Inference failed: {exc}"
            log_error(self.last_error)
            return pf

        if world is not None:
            for idx, wl in enumerate(world):
                name = MEDIAPIPE_POSE_NAMES.get(idx)
                if name is None:
                    continue
                il = img[idx] if img is not None and idx < len(img) else None
                vis = getattr(il, "visibility", None) if il is not None else None
                if vis is None:
                    vis = getattr(wl, "visibility", None)
                pres = getattr(il, "presence", None) if il is not None else None
                conf = float(vis if vis is not None else 1.0)
                if pres is not None:
                    conf = min(conf, float(pres))
                pf.landmarks[name] = Landmark(
                    name=name,
                    position=mp_world_to_capture(float(wl.x), float(wl.y), float(wl.z)),
                    confidence=conf,
                    valid=True,
                    image_xy=(float(il.x), float(il.y)) if il is not None else None,
                )
        pf.landmarks = self._hyst.filter_landmarks(pf.landmarks)
        pf.tracking_state = self._hyst.update(pf.landmarks) if pf.landmarks else self._hyst.update({})
        return pf

    def shutdown(self) -> None:
        for obj in (self._landmarker, self._legacy):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._landmarker = None
        self._legacy = None

    def info(self) -> Dict[str, str]:
        return {"name": self.name, "api": self.api, "local": "true",
                "model": os.path.basename(self.model_path or "")}
