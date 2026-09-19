"""Frame sources: webcam, video file, image sequence (OpenCV-backed)."""

from __future__ import annotations

import glob
import os
import time
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from ..utils.logging_util import log_error, log_info

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


class FrameSource(ABC):
    """read() -> (ok, frame_bgr, timestamp_seconds)."""

    kind = "abstract"
    is_live = False

    def __init__(self):
        self.last_error = ""
        self.fps: float = 30.0
        self.size: Tuple[int, int] = (0, 0)
        self.frame_count: int = -1  # unknown / infinite
        self.index = 0

    @abstractmethod
    def open(self) -> bool: ...

    @abstractmethod
    def read(self) -> Tuple[bool, Any, float]: ...

    def close(self) -> None:
        pass

    @property
    def exhausted(self) -> bool:
        return self.frame_count >= 0 and self.index >= self.frame_count


def _cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        return None


class WebcamSource(FrameSource):
    kind = "WEBCAM"
    is_live = True

    def __init__(self, device_index: int = 0, width: int = 640, height: int = 480, fps: float = 30.0):
        super().__init__()
        self.device_index = int(device_index)
        self.req = (int(width), int(height), float(fps))
        self._cap = None
        self._t0 = 0.0

    def open(self) -> bool:
        cv2 = _cv2()
        if cv2 is None:
            self.last_error = "OpenCV is not installed (use 'Install Dependencies')."
            return False
        try:
            cap = cv2.VideoCapture(self.device_index)
            if not cap.isOpened():
                cap.release()
                self.last_error = (f"Could not open camera {self.device_index}. "
                                   "Check that a webcam is connected and not in use.")
                log_error(self.last_error)
                return False
            w, h, fps = self.req
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                self.last_error = f"Camera {self.device_index} opened but returned no frames."
                log_error(self.last_error)
                return False
            self._cap = cap
            self.size = (frame.shape[1], frame.shape[0])
            self.fps = float(cap.get(cv2.CAP_PROP_FPS) or fps) or fps
            self._t0 = time.monotonic()
            log_info(f"Camera {self.device_index} opened at {self.size[0]}x{self.size[1]}")
            return True
        except Exception as exc:  # pragma: no cover - driver specific
            self.last_error = f"Camera open error: {exc}"
            log_error(self.last_error)
            return False

    def read(self) -> Tuple[bool, Any, float]:
        if self._cap is None:
            return False, None, 0.0
        try:
            ok, frame = self._cap.read()
        except Exception as exc:  # pragma: no cover
            self.last_error = str(exc)
            return False, None, 0.0
        self.index += 1
        return bool(ok), frame, time.monotonic() - self._t0

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
            log_info("Camera released")


class VideoFileSource(FrameSource):
    kind = "VIDEO"

    def __init__(self, path: str, loop: bool = False):
        super().__init__()
        self.path = path
        self.loop = loop
        self._cap = None

    def open(self) -> bool:
        cv2 = _cv2()
        if cv2 is None:
            self.last_error = "OpenCV is not installed (use 'Install Dependencies')."
            return False
        if not os.path.isfile(self.path):
            self.last_error = f"Video not found: {self.path}"
            return False
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            self.last_error = f"Could not decode video: {self.path}"
            return False
        self._cap = cap
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or -1)
        self.frame_count = -1 if self.loop else n
        self.size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        return True

    def read(self) -> Tuple[bool, Any, float]:
        if self._cap is None:
            return False, None, 0.0
        ok, frame = self._cap.read()
        if not ok and self.loop:
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        t = self.index / self.fps
        self.index += 1
        return bool(ok), frame, t

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class ImageSequenceSource(FrameSource):
    kind = "IMAGES"

    def __init__(self, directory: str, fps: float = 30.0, loop: bool = False):
        super().__init__()
        self.directory = directory
        self.fps = float(fps)
        self.loop = loop
        self.files: List[str] = []

    def open(self) -> bool:
        cv2 = _cv2()
        if cv2 is None:
            self.last_error = "OpenCV is not installed (use 'Install Dependencies')."
            return False
        pattern = self.directory
        if os.path.isdir(pattern):
            files = [f for f in sorted(glob.glob(os.path.join(pattern, "*")))
                     if f.lower().endswith(IMAGE_EXTS)]
        else:
            files = sorted(glob.glob(pattern))
        if not files:
            self.last_error = f"No images found in {self.directory}"
            return False
        self.files = files
        self.frame_count = -1 if self.loop else len(files)
        first = cv2.imread(files[0], cv2.IMREAD_COLOR)
        if first is None:
            self.last_error = f"Could not read {files[0]}"
            return False
        self.size = (first.shape[1], first.shape[0])
        return True

    def read(self) -> Tuple[bool, Any, float]:
        import cv2
        if not self.files:
            return False, None, 0.0
        if self.index >= len(self.files):
            if not self.loop:
                return False, None, self.index / self.fps
        path = self.files[self.index % len(self.files)]
        frame = cv2.imread(path, cv2.IMREAD_COLOR)
        t = self.index / self.fps
        self.index += 1
        return frame is not None, frame, t


class NullSource(FrameSource):
    """No images (synthetic backend generates landmarks directly)."""

    kind = "NONE"
    is_live = True

    def __init__(self, fps: float = 30.0, frame_count: int = -1):
        super().__init__()
        self.fps = fps
        self.frame_count = frame_count
        self.size = (640, 480)
        self._t0 = time.monotonic()

    def open(self) -> bool:
        self._t0 = time.monotonic()
        return True

    def read(self) -> Tuple[bool, Any, float]:
        t = self.index / self.fps
        self.index += 1
        return True, None, t


def make_source(kind: str, **kw) -> FrameSource:
    kind = kind.upper()
    if kind == "WEBCAM":
        return WebcamSource(kw.get("device_index", 0), kw.get("width", 640), kw.get("height", 480),
                            kw.get("fps", 30.0))
    if kind == "VIDEO":
        return VideoFileSource(kw["path"], loop=kw.get("loop", False))
    if kind == "IMAGES":
        return ImageSequenceSource(kw["path"], fps=kw.get("fps", 30.0), loop=kw.get("loop", False))
    return NullSource(fps=kw.get("fps", 30.0), frame_count=kw.get("frame_count", -1))
