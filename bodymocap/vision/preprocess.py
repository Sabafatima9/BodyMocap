"""Lighting / exposure compensation for camera frames (numpy, OpenCV optional).

AUTO mode builds a single 256-entry tone curve per frame (percentile contrast
stretch + gamma towards a mid-grey target) and optionally applies CLAHE on the
luminance channel for back-lit subjects.  The frame is also classified
(UNDEREXPOSED / OVEREXPOSED / LOW_CONTRAST / BACKLIT / OK) so the UI can tell
the user what the camera is seeing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

try:  # numpy is bundled with Blender; guarded for pure-Python test runs
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

MODES = ("OFF", "AUTO", "MANUAL")


@dataclass
class ExposureSettings:
    mode: str = "AUTO"
    gain_ev: float = 0.0        # MANUAL: exposure in stops
    gamma: float = 1.0          # MANUAL: >1 darkens, <1 brightens
    contrast: float = 1.0       # MANUAL: around mid grey
    clahe: bool = True          # local contrast (helps back-light)
    clahe_clip: float = 2.0
    target_mean: float = 0.45   # AUTO: desired mean luminance (0..1)
    stretch_low: float = 1.0    # AUTO: percentiles for contrast stretch
    stretch_high: float = 99.0


@dataclass
class FrameStats:
    mean: float = 0.0
    p_low: float = 0.0
    p_high: float = 1.0
    clipped_high: float = 0.0
    center_mean: float = 0.0
    border_mean: float = 0.0
    condition: str = "OK"
    applied: Dict[str, float] = field(default_factory=dict)

    @property
    def dynamic_range(self) -> float:
        return self.p_high - self.p_low


def _luma(frame: Any) -> Any:
    f = frame[::4, ::4, :3].astype(np.float32)
    return (0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]) / 255.0


def analyze(frame_bgr: Any) -> FrameStats:
    """Luminance statistics + lighting condition classification."""
    st = FrameStats()
    if np is None or frame_bgr is None:
        return st
    y = _luma(frame_bgr)
    h, w = y.shape
    st.mean = float(y.mean())
    st.p_low, st.p_high = (float(v) for v in np.percentile(y, [1.0, 99.0]))
    st.clipped_high = float((y > 0.98).mean())
    ch, cw = h // 4, w // 4
    center = y[ch:h - ch, cw:w - cw]
    st.center_mean = float(center.mean()) if center.size else st.mean
    border_mask = np.ones_like(y, dtype=bool)
    border_mask[ch:h - ch, cw:w - cw] = False
    st.border_mean = float(y[border_mask].mean()) if border_mask.any() else st.mean
    st.condition = classify(st)
    return st


def classify(st: FrameStats) -> str:
    if st.border_mean - st.center_mean > 0.2 and st.border_mean > 0.6:
        return "BACKLIT"
    if st.mean > 0.75 or st.clipped_high > 0.2:
        return "OVEREXPOSED"
    if st.mean < 0.22:
        return "UNDEREXPOSED"
    if st.dynamic_range < 0.35:
        return "LOW_CONTRAST"
    return "OK"


def _auto_lut(st: FrameStats, s: ExposureSettings) -> Tuple[Any, Dict[str, float]]:
    x = np.arange(256, dtype=np.float32) / 255.0
    lo, hi = st.p_low, st.p_high
    applied: Dict[str, float] = {}
    if hi - lo < 0.9 and hi - lo > 1e-3:
        x = np.clip((x - lo) / (hi - lo) * 0.95 + 0.025, 0.0, 1.0)
        mean = float(np.clip((st.mean - lo) / (hi - lo) * 0.95 + 0.025, 1e-3, 0.999))
        applied["stretch"] = 1.0 / (hi - lo)
    else:
        mean = float(min(max(st.mean, 1e-3), 0.999))
    gamma = math.log(s.target_mean) / math.log(mean)
    gamma = min(max(gamma, 0.4), 2.5)
    applied["gamma"] = gamma
    x = np.power(x, gamma)
    return (x * 255.0 + 0.5).clip(0, 255).astype(np.uint8), applied


def _manual_lut(s: ExposureSettings) -> Tuple[Any, Dict[str, float]]:
    x = np.arange(256, dtype=np.float32) / 255.0
    x = x * (2.0 ** s.gain_ev)
    x = (x - 0.5) * s.contrast + 0.5
    x = np.power(np.clip(x, 0.0, 1.0), max(s.gamma, 1e-3))
    return (x * 255.0 + 0.5).clip(0, 255).astype(np.uint8), {
        "gain": 2.0 ** s.gain_ev, "contrast": s.contrast, "gamma": s.gamma}


def _clahe(frame: Any, clip: float) -> Any:
    try:
        import cv2
    except ImportError:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8))
    lab[..., 0] = cl.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def compensate(frame_bgr: Any, settings: Optional[ExposureSettings] = None) -> Tuple[Any, FrameStats]:
    """Return (compensated frame, stats of the *input* frame)."""
    s = settings or ExposureSettings()
    st = analyze(frame_bgr)
    if np is None or frame_bgr is None or s.mode == "OFF":
        return frame_bgr, st
    if s.mode == "MANUAL":
        lut, applied = _manual_lut(s)
    else:
        lut, applied = _auto_lut(st, s)
    out = lut[frame_bgr]
    if s.clahe and (s.mode == "MANUAL" or st.condition in ("BACKLIT", "LOW_CONTRAST", "UNDEREXPOSED")):
        out = _clahe(out, s.clahe_clip)
        applied["clahe"] = s.clahe_clip
    st.applied = applied
    return np.ascontiguousarray(out), st


# ---------------------------------------------------------------------------
# Degradations (used by tests / synthetic video) -- the inverse problem
# ---------------------------------------------------------------------------

def degrade(frame_bgr: Any, kind: str, seed: int = 0, alpha: Any = None) -> Any:
    """Simulate a lighting problem on a well-exposed frame.

    kinds: low_contrast, underexposed, overexposed, backlight (needs the
    subject alpha mask), noise.
    """
    rng = np.random.default_rng(seed)
    f = frame_bgr.astype(np.float32) / 255.0
    if kind == "low_contrast":
        f = 0.5 + (f - 0.5) * 0.3 + 0.1
    elif kind == "underexposed":
        f = np.power(f * 0.25, 1.1)
        f = f + rng.normal(0.0, 0.015, f.shape)
    elif kind == "overexposed":
        f = np.clip(f * 2.6 + 0.15, 0.0, 1.0)
    elif kind == "backlight":
        if alpha is None:
            raise ValueError("backlight needs a subject alpha mask")
        a = alpha.astype(np.float32)[..., None]
        subject = f * 0.18
        bg = np.clip(f * 0.3 + 0.75, 0.0, 1.0)
        f = subject * a + bg * (1.0 - a)
        # veiling glare around the silhouette
        try:
            import cv2
            glow = cv2.GaussianBlur((1.0 - a[..., 0]), (0, 0), 9)[..., None]
            f = np.clip(f + 0.25 * glow * a, 0.0, 1.0)
        except ImportError:
            pass
    elif kind == "noise":
        f = f + rng.normal(0.0, 0.06, f.shape)
    return (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
