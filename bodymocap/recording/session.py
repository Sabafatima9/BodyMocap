"""Recording: takes of *source skeleton* frames (FR-050-053). Pure Python.

A take stores what the camera saw (canonical landmarks + timestamps), not bone
rotations, so a single take can be baked onto any number of armatures with
different topologies, and re-baked later with other smoothing / mapping
settings ("record once, retarget to many").
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.skeleton import SourceSkeleton
from ..core.types import TrackingState

TAKE_TYPE = "bodymocap_take"
TAKE_VERSION = 1


@dataclass
class Take:
    name: str = "Take"
    frames: List[SourceSkeleton] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].timestamp - self.frames[0].timestamp

    def frame_count(self) -> int:
        return len(self.frames)

    def mean_fps(self) -> float:
        d = self.duration
        return (len(self.frames) - 1) / d if d > 1e-6 else 0.0

    def degraded_fraction(self) -> float:
        if not self.states:
            return 0.0
        bad = sum(1 for s in self.states if s in ("DEGRADED", "LOST"))
        return bad / len(self.states)

    def to_dict(self) -> dict:
        return {
            "type": TAKE_TYPE, "version": TAKE_VERSION, "name": self.name,
            "meta": self.meta, "states": list(self.states),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Take":
        if d.get("type") != TAKE_TYPE:
            raise ValueError("Not a BodyMocap take file")
        frames = [SourceSkeleton.from_dict(f) for f in d.get("frames", [])]
        states = list(d.get("states", [])) or ["OK"] * len(frames)
        return cls(name=d.get("name", "Take"), frames=frames, states=states, meta=d.get("meta", {}))

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str) -> "Take":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class RecordingSession:
    """Start / pause / resume / stop / discard around a :class:`Take`."""

    def __init__(self, degraded_warn_fraction: float = 0.15):
        self.take = Take()
        self.is_recording = False
        self.is_paused = False
        self.degraded_warn_fraction = degraded_warn_fraction
        self._t0: Optional[float] = None
        self._pause_started: Optional[float] = None
        self._paused_total = 0.0

    # -- control ---------------------------------------------------------------
    def start(self, name: str = "Take", meta: Optional[Dict[str, Any]] = None) -> None:
        self.take = Take(name=name, meta=dict(meta or {}))
        self.take.meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.is_recording = True
        self.is_paused = False
        self._t0 = None
        self._paused_total = 0.0
        self._pause_started = None

    def pause(self) -> None:
        if self.is_recording and not self.is_paused:
            self.is_paused = True
            self._pause_started = None

    def resume(self) -> None:
        if self.is_recording and self.is_paused:
            self.is_paused = False

    def stop(self) -> Take:
        self.is_recording = False
        self.is_paused = False
        return self.take

    def discard(self) -> None:
        self.take = Take()
        self.is_recording = False
        self.is_paused = False

    # -- data ------------------------------------------------------------------
    def append(self, sk: SourceSkeleton, state: TrackingState = TrackingState.OK) -> bool:
        """Add a frame; timestamps are re-based so the take starts at 0 and
        paused intervals are removed."""
        if not self.is_recording:
            return False
        if self.is_paused:
            if self._pause_started is None:
                self._pause_started = sk.timestamp
            return False
        if self._pause_started is not None:
            self._paused_total += sk.timestamp - self._pause_started
            self._pause_started = None
        if self._t0 is None:
            self._t0 = sk.timestamp
        f = sk.copy()
        f.timestamp = sk.timestamp - self._t0 - self._paused_total
        f.frame_index = len(self.take.frames)
        if self.take.frames and f.timestamp <= self.take.frames[-1].timestamp:
            f.timestamp = self.take.frames[-1].timestamp + 1e-4
        self.take.frames.append(f)
        self.take.states.append(state.name if isinstance(state, TrackingState) else str(state))
        return True

    def frame_count(self) -> int:
        return len(self.take.frames)

    def degraded_or_lost_fraction(self) -> float:
        return self.take.degraded_fraction()

    def should_warn_tracking(self) -> bool:
        return self.degraded_or_lost_fraction() > self.degraded_warn_fraction

    def tracking_warning_message(self) -> str:
        pct = int(round(self.degraded_or_lost_fraction() * 100))
        return (f"Tracking was Degraded/Lost for {pct}% of frames "
                f"(threshold {int(self.degraded_warn_fraction * 100)}%).")


_ACTIVE: Optional[RecordingSession] = None


def get_active_session() -> RecordingSession:
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = RecordingSession()
    return _ACTIVE


def reset_active_session() -> RecordingSession:
    global _ACTIVE
    _ACTIVE = RecordingSession()
    return _ACTIVE
