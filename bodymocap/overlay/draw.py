"""Skeleton overlay drawn onto preview frames (FR-030-032)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..core.landmarks import POSE_CONNECTIONS
from ..core.types import Landmark


def confidence_color(confidence: float, threshold: float = 0.5) -> Tuple[int, int, int]:
    """BGR colour: green high, yellow mid, red low."""
    if confidence >= threshold + 0.2:
        return (0, 220, 0)
    if confidence >= threshold:
        return (0, 200, 220)
    return (0, 0, 220)


def draw_skeleton(frame_bgr: Any, landmarks: Dict[str, Landmark], threshold: float = 0.5,
                  enabled: bool = True, label: str = "") -> Any:
    """Draw joints/bones using normalised image coordinates; returns a copy."""
    if not enabled or frame_bgr is None:
        return frame_bgr
    try:
        import cv2
    except ImportError:
        return frame_bgr
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    pts = {}
    for name, lm in landmarks.items():
        if lm.image_xy is None:
            continue
        u, v = lm.image_xy
        if not (-0.2 <= u <= 1.2 and -0.2 <= v <= 1.2):
            continue
        pts[name] = (int(u * w), int(v * h))
    thick = max(1, w // 320)
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            c = min(landmarks[a].confidence, landmarks[b].confidence)
            cv2.line(out, pts[a], pts[b], confidence_color(c, threshold), 2 * thick, cv2.LINE_AA)
    for name, p in pts.items():
        cv2.circle(out, p, 3 * thick, confidence_color(landmarks[name].confidence, threshold), -1,
                   cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (8, 20 * thick), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thick,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return out


def blank_canvas(width: int = 640, height: int = 480) -> Any:
    try:
        import numpy as np
    except ImportError:
        return None
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (40, 36, 32)
    return img
