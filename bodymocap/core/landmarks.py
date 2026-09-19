"""MediaPipe Pose 33-landmark naming and coordinate conventions (pure Python)."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import Vec3

# Standard MediaPipe Pose landmark index -> name
MEDIAPIPE_POSE_NAMES: Dict[int, str] = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}

MEDIAPIPE_POSE_INDEX: Dict[str, int] = {v: k for k, v in MEDIAPIPE_POSE_NAMES.items()}

# Skeleton edges for overlays (parent, child)
POSE_CONNECTIONS: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_index"),
    ("left_wrist", "left_pinky"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_index"),
    ("right_wrist", "right_pinky"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
    ("left_ear", "left_eye"),
    ("left_eye", "nose"),
    ("right_ear", "right_eye"),
    ("right_eye", "nose"),
]

# Landmarks that matter for body retargeting (face mesh points excluded).
BODY_LANDMARKS: Tuple[str, ...] = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky", "right_pinky",
    "left_index", "right_index", "left_thumb", "right_thumb",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
)


def mp_world_to_capture(x: float, y: float, z: float) -> Vec3:
    """MediaPipe world axes (x right, y down, z away from camera) -> capture space.

    Capture space uses Blender conventions for a subject facing the camera:
    +X = subject's left (image right), +Y = away from camera, +Z = up.
    The mapping is a proper rotation (det = +1).
    """
    return Vec3(x, z, -y)


def capture_to_mp_world(v: Vec3) -> Tuple[float, float, float]:
    """Inverse of :func:`mp_world_to_capture`."""
    return (v.x, -v.z, v.y)
