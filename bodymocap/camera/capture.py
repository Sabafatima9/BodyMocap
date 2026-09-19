"""OpenCV camera capture helpers (FR-010–016). bpy-guarded where needed."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from ..utils.logging_util import log_error, log_info


class CameraCapture:
    """Wraps cv2.VideoCapture by device index."""

    def __init__(self):
        self._cap = None
        self.device_index: int = 0
        self.is_open: bool = False
        self.last_error: str = ""

    def open(self, device_index: int = 0, width: int = 640, height: int = 480) -> bool:
        try:
            import cv2
        except ImportError:
            self.last_error = (
                "OpenCV not installed. Install opencv-python-headless into Blender's Python. "
                "See INSTALL.md."
            )
            log_error(self.last_error)
            return False

        self.close()
        self.device_index = device_index
        try:
            cap = cv2.VideoCapture(int(device_index))
            if not cap.isOpened():
                self.last_error = (
                    f"Failed to open camera device index {device_index}. "
                    "Check that a webcam is connected and not in use."
                )
                log_error(self.last_error)
                cap.release()
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Fail-fast read
            ok, _ = cap.read()
            if not ok:
                self.last_error = (
                    f"Camera {device_index} opened but failed to read a frame."
                )
                log_error(self.last_error)
                cap.release()
                return False
            self._cap = cap
            self.is_open = True
            self.last_error = ""
            log_info(f"Camera {device_index} opened")
            return True
        except Exception as exc:
            self.last_error = f"Camera open error: {exc}"
            log_error(self.last_error)
            return False

    def read(self) -> Tuple[bool, Any]:
        if not self.is_open or self._cap is None:
            return False, None
        try:
            return self._cap.read()
        except Exception as exc:
            self.last_error = str(exc)
            return False, None

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.is_open = False
        log_info("Camera released")

    def mirror_frame(self, frame_bgr: Any) -> Any:
        try:
            import cv2

            return cv2.flip(frame_bgr, 1)
        except Exception:
            return frame_bgr


# Process-wide capture used by modal operator
_CAPTURE: Optional[CameraCapture] = None


def get_capture() -> CameraCapture:
    global _CAPTURE
    if _CAPTURE is None:
        _CAPTURE = CameraCapture()
    return _CAPTURE
