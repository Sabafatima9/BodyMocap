"""Blender version checks and dependency status (FR-003, FR-004, FR-082)."""

from __future__ import annotations

from typing import Dict, Tuple


def get_blender_version() -> Tuple[int, int, int]:
    try:
        import bpy

        v = bpy.app.version
        return (int(v[0]), int(v[1]), int(v[2]))
    except Exception:
        return (0, 0, 0)


def is_supported_blender(min_version: Tuple[int, int, int] = (4, 0, 0)) -> bool:
    ver = get_blender_version()
    if ver == (0, 0, 0):
        return False
    return ver >= min_version


def version_warning_message(min_version: Tuple[int, int, int] = (4, 0, 0)) -> str:
    ver = get_blender_version()
    return (
        f"BodyMocap requires Blender {min_version[0]}.{min_version[1]}+; "
        f"detected {ver[0]}.{ver[1]}.{ver[2]}"
    )


def check_opencv() -> Tuple[bool, str]:
    try:
        import cv2

        return True, getattr(cv2, "__version__", "unknown")
    except ImportError:
        return False, "not installed"


def check_mediapipe() -> Tuple[bool, str]:
    try:
        import mediapipe as mp

        return True, getattr(mp, "__version__", "unknown")
    except ImportError:
        return False, "not installed"


def check_numpy() -> Tuple[bool, str]:
    try:
        import numpy as np

        return True, getattr(np, "__version__", "unknown")
    except ImportError:
        return False, "not installed"


def dependency_status() -> Dict[str, Dict[str, str]]:
    status: Dict[str, Dict[str, str]] = {}
    for name, fn in (
        ("numpy", check_numpy),
        ("opencv", check_opencv),
        ("mediapipe", check_mediapipe),
    ):
        ok, ver = fn()
        status[name] = {
            "available": "yes" if ok else "no",
            "version": ver,
        }
    return status


def dependency_panel_text() -> str:
    st = dependency_status()
    lines = []
    for name, info in st.items():
        mark = "OK" if info["available"] == "yes" else "MISSING"
        lines.append(f"{name}: {mark} ({info['version']})")
    return " | ".join(lines)


def install_deps_instructions() -> str:
    return (
        "Install into Blender's Python, e.g.:\n"
        "  blender --python-expr \"import ensurepip; ensurepip.bootstrap()\"\n"
        "  /path/to/blender/python/bin/python -m pip install "
        "opencv-python-headless mediapipe numpy\n"
        "See INSTALL.md for details. Mock/offline backend works without these."
    )
