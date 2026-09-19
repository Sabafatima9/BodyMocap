"""Synthetic / replay pose backend for camera-less workflows and tests.

* ``clip`` mode streams the procedural performer (``pose.synthetic``) under a
  chosen lighting-condition model -- no webcam or ML dependency needed.
* ``fixture`` mode replays a recorded take (``*.json`` from Save Take) or a
  legacy landmark fixture (y-up coordinates).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.confidence import ConfidenceConfig, TrackingHysteresis
from ..core.landmarks import MEDIAPIPE_POSE_NAMES
from ..core.skeleton import MP_FROM_JOINT, SourceSkeleton
from ..core.types import Landmark, PoseFrame, TrackingState, Vec3
from .backend import PoseBackend
from .synthetic import CLIPS, CONDITIONS, SyntheticStream


class MockBackend(PoseBackend):
    name = "synthetic"

    def __init__(
        self,
        mode: str = "clip",
        clip: str = "wave",
        condition: str = "normal",
        speed: float = 1.0,
        seed: int = 0,
        fixture_path: Optional[str] = None,
        confidence_cfg: Optional[ConfidenceConfig] = None,
    ):
        self.mode = mode
        self.clip = clip if clip in CLIPS else "wave"
        self.condition = condition if condition in CONDITIONS else "normal"
        self.speed = speed
        self.seed = seed
        self.fixture_path = fixture_path
        self._hyst = TrackingHysteresis(confidence_cfg or ConfidenceConfig())
        self._stream: Optional[SyntheticStream] = None
        self._frames: List[PoseFrame] = []
        self._index = 0
        self._ready = False
        self.last_error = ""

    def initialize(self, **kwargs: Any) -> bool:
        path = kwargs.get("fixture_path", self.fixture_path)
        if self.mode == "fixture" or path:
            if not path or not Path(path).is_file():
                self.last_error = f"Fixture/take file not found: {path}"
                return False
            try:
                self._frames = load_landmark_file(path)
            except Exception as exc:
                self.last_error = f"Could not read {path}: {exc}"
                return False
            if not self._frames:
                self.last_error = f"No frames in {path}"
                return False
            self.mode = "fixture"
        else:
            self._stream = SyntheticStream(self.clip, self.condition, self.speed, self.seed)
        self._index = 0
        self._ready = True
        return True

    def infer(self, frame_bgr: Any = None, frame_index: int = 0, timestamp: float = 0.0) -> PoseFrame:
        if not self._ready and not self.initialize():
            return PoseFrame(frame_index=frame_index, timestamp=timestamp)
        if self.mode == "fixture":
            src = self._frames[self._index % len(self._frames)]
            self._index += 1
            pf = PoseFrame(landmarks=dict(src.landmarks), frame_index=frame_index,
                           timestamp=timestamp, image_size=src.image_size)
        else:
            pf, _, _, _ = self._stream.frame(frame_index, timestamp)
        pf.landmarks = self._hyst.filter_landmarks(pf.landmarks)
        pf.tracking_state = self._hyst.update(pf.landmarks)
        return pf

    def shutdown(self) -> None:
        self._ready = False
        self._stream = None
        self._frames = []

    def info(self) -> Dict[str, str]:
        return {"name": self.name, "mode": self.mode, "clip": self.clip,
                "condition": self.condition, "local": "true"}


def _frame_from_skeleton(sk: SourceSkeleton) -> PoseFrame:
    pf = PoseFrame(timestamp=sk.timestamp, frame_index=sk.frame_index,
                   tracking_state=TrackingState.OK)
    for j, p in sk.joints.items():
        mp_name = MP_FROM_JOINT.get(j)
        if mp_name:
            pf.landmarks[mp_name] = Landmark(mp_name, p.copy(), sk.conf.get(j, 1.0), True)
    return pf


def load_landmark_file(path: str) -> List[PoseFrame]:
    """Load a BodyMocap take (capture space) or a legacy y-up landmark fixture."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and data.get("type") == "bodymocap_take":
        return [_frame_from_skeleton(SourceSkeleton.from_dict(f)) for f in data.get("frames", [])]
    frames_raw = data.get("frames", data) if isinstance(data, dict) else data
    out: List[PoseFrame] = []
    for i, raw in enumerate(frames_raw):
        pf = PoseFrame(timestamp=float(raw.get("timestamp", i / 30.0)),
                       frame_index=int(raw.get("frame_index", i)), tracking_state=TrackingState.OK)
        items = raw.get("landmarks", raw)
        pts: Dict[str, Landmark] = {}
        for name, v in items.items():
            if name not in MEDIAPIPE_POSE_NAMES.values():
                continue
            if isinstance(v, dict):
                pos = v.get("position", v)
                x, y, z = float(pos.get("x", 0)), float(pos.get("y", 0)), float(pos.get("z", 0))
                c = float(v.get("confidence", 1.0))
            else:
                x, y, z = float(v[0]), float(v[1]), float(v[2])
                c = float(v[3]) if len(v) > 3 else 1.0
            # legacy fixtures are y-up with +z towards the camera
            pts[name] = Landmark(name, Vec3(x, -z, y), c, True)
        if "left_hip" in pts and "right_hip" in pts:
            hm = pts["left_hip"].position.lerp(pts["right_hip"].position, 0.5)
            for lm in pts.values():
                lm.position = lm.position - hm
        pf.landmarks = pts
        out.append(pf)
    return out
