"""MediaPipe Pose backend — optional dependency (FR-020–024, FR-082)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..core.confidence import ConfidenceConfig, TrackingHysteresis
from ..core.landmarks import MEDIAPIPE_POSE_NAMES
from ..core.types import Landmark, PoseFrame, Vec3
from .backend import PoseBackend


def mediapipe_available() -> bool:
    try:
        import mediapipe  # noqa: F401
        return True
    except ImportError:
        return False


class MediaPipeBackend(PoseBackend):
    name = "mediapipe"

    def __init__(self, confidence_cfg: Optional[ConfidenceConfig] = None):
        self._pose = None
        self._mp = None
        self._hyst = TrackingHysteresis(confidence_cfg or ConfidenceConfig())
        self._ready = False

    def is_available(self) -> bool:
        return mediapipe_available()

    def initialize(self, **kwargs: Any) -> bool:
        if not mediapipe_available():
            return False
        try:
            import mediapipe as mp

            self._mp = mp
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=int(kwargs.get("model_complexity", 1)),
                enable_segmentation=False,
                min_detection_confidence=float(kwargs.get("min_detection_confidence", 0.5)),
                min_tracking_confidence=float(kwargs.get("min_tracking_confidence", 0.5)),
            )
            self._ready = True
            return True
        except Exception:
            self._ready = False
            return False

    def infer(self, frame_bgr: Any, frame_index: int = 0, timestamp: float = 0.0) -> PoseFrame:
        if not self._ready or self._pose is None:
            return PoseFrame(frame_index=frame_index, timestamp=timestamp)

        try:
            import cv2

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._pose.process(rgb)
        except Exception:
            return PoseFrame(frame_index=frame_index, timestamp=timestamp)

        landmarks: Dict[str, Landmark] = {}
        if results.pose_landmarks:
            for idx, lm in enumerate(results.pose_landmarks.landmark):
                name = MEDIAPIPE_POSE_NAMES.get(idx, f"lm_{idx}")
                # MediaPipe: x,y normalized image; z depth-ish. Map to y-up world-ish.
                landmarks[name] = Landmark(
                    name=name,
                    position=Vec3(float(lm.x - 0.5), float(1.0 - lm.y), float(-lm.z)),
                    confidence=float(getattr(lm, "visibility", 1.0)),
                    valid=True,
                )

        filtered = self._hyst.filter_landmarks(landmarks)
        state = self._hyst.update(landmarks)
        return PoseFrame(
            landmarks=filtered,
            tracking_state=state,
            timestamp=timestamp,
            frame_index=frame_index,
        )

    def shutdown(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
        self._pose = None
        self._ready = False

    def info(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "local": "true",
            "available": str(self.is_available()),
        }
