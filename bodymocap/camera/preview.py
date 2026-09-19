"""Live preview: push camera frames (with skeleton overlay) into a Blender Image.

The Image is shown in any Image Editor and as the background of the spawned
tracking camera, so looking through that camera overlays the rig on the feed.
"""

from __future__ import annotations

from typing import Any, Optional

PREVIEW_IMAGE_NAME = "BodyMocap_Preview"
MAX_PREVIEW_WIDTH = 480


def ensure_preview_image(width: int = 640, height: int = 480):
    import bpy
    img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    if img is None:
        img = bpy.data.images.new(PREVIEW_IMAGE_NAME, width=width, height=height, alpha=False)
        img.generated_color = (0.05, 0.05, 0.05, 1.0)
    return img


def frame_to_image(frame_bgr: Any, flip_x: bool = False) -> Optional[object]:
    """Upload a BGR uint8 frame (downscaled) to the preview Image."""
    try:
        import numpy as np
    except ImportError:
        return None
    if frame_bgr is None:
        return None
    arr = np.asarray(frame_bgr)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    h, w = arr.shape[:2]
    step = max(1, int(np.ceil(w / MAX_PREVIEW_WIDTH)))
    small = arr[::step, ::step, :3]
    if flip_x:
        small = small[:, ::-1]
    sh, sw = small.shape[:2]
    rgba = np.empty((sh, sw, 4), dtype=np.float32)
    rgba[..., :3] = small[::-1, :, ::-1] / 255.0  # BGR->RGB, flip vertically for Blender
    rgba[..., 3] = 1.0
    import bpy
    img = bpy.data.images.get(PREVIEW_IMAGE_NAME)
    if img is None or img.size[0] != sw or img.size[1] != sh:
        if img is not None:
            bpy.data.images.remove(img)
        img = bpy.data.images.new(PREVIEW_IMAGE_NAME, width=sw, height=sh, alpha=False)
    img.pixels.foreach_set(rgba.ravel())
    img.update()
    return img


def tag_redraw_all() -> None:
    try:
        import bpy
        for window in bpy.context.window_manager.windows:
            for area in window.screen.areas:
                if area.type in ("IMAGE_EDITOR", "VIEW_3D"):
                    area.tag_redraw()
    except Exception:
        pass
