"""Optional dependency management: detection, pip install, pose model download.

Nothing here runs implicitly: installs and downloads only happen when the user
presses the corresponding button (or a test harness calls them).

Install location: Blender's own ``site-packages`` when writable (portable
installs), otherwise a per-user directory appended to ``sys.path`` so the
packages Blender bundles (numpy, ...) always take precedence.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import sysconfig
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"
MODEL_VARIANTS = ("lite", "full", "heavy")
MODEL_SIZES_MB = {"lite": 5.5, "full": 9.0, "heavy": 29.2}

# (import name, pip requirement)
PACKAGES: Sequence[Tuple[str, str]] = (
    ("numpy", "numpy"),
    ("cv2", "opencv-contrib-python-headless"),
    ("mediapipe", "mediapipe"),
)
# mediapipe's runtime imports; installed explicitly because mediapipe itself is
# installed with --no-deps (it would otherwise pull the non-headless OpenCV
# wheel, which clashes with the headless one).
MEDIAPIPE_RUNTIME_DEPS = ("absl-py", "flatbuffers", "matplotlib")


def _package_name() -> str:
    return __package__.rsplit(".", 1)[0] if __package__ else "bodymocap"


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def user_data_dir(sub: str = "") -> str:
    """Writable per-user data directory for the add-on."""
    path = ""
    try:
        import bpy
        pkg = _package_name()
        if pkg.startswith("bl_ext.") and hasattr(bpy.utils, "extension_path_user"):
            path = bpy.utils.extension_path_user(pkg, path=sub, create=True)
        else:
            path = bpy.utils.user_resource("DATAFILES", path=os.path.join("bodymocap", sub), create=True)
    except Exception:
        path = os.path.join(os.path.expanduser("~"), ".bodymocap", sub)
    os.makedirs(path, exist_ok=True)
    return path


def blender_site_packages() -> str:
    return sysconfig.get_paths()["purelib"]


def user_site_packages() -> str:
    return user_data_dir("site-packages")


def ensure_user_site_on_path() -> None:
    p = user_site_packages()
    if os.path.isdir(p) and p not in sys.path:
        sys.path.append(p)  # append: Blender's bundled packages win


def is_writable_dir(path: str) -> bool:
    """True when ``path`` can be created and written to (os.access lies on Windows)."""
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".bodymocap_write_probe")
        with open(probe, "w", encoding="utf-8") as fh:
            fh.write("x")
        os.remove(probe)
        return True
    except Exception:
        return False


def default_install_target() -> Tuple[str, bool]:
    """(directory, is_blender_site_packages)."""
    sp = blender_site_packages()
    if is_writable_dir(sp):
        return sp, True
    return user_site_packages(), False


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def check_module(name: str) -> Tuple[bool, str]:
    try:
        mod = importlib.import_module(name)
        return True, str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return False, "missing"


def dependency_status() -> Dict[str, Dict[str, str]]:
    ensure_user_site_on_path()
    out: Dict[str, Dict[str, str]] = {}
    for mod, _pip in PACKAGES:
        ok, ver = check_module(mod)
        out[mod] = {"available": "yes" if ok else "no", "version": ver}
    return out


def missing_requirements() -> List[str]:
    return [pip for mod, pip in PACKAGES if dependency_status()[mod]["available"] != "yes"]


def dependency_summary() -> str:
    st = dependency_status()
    return " | ".join(f"{k}: {'OK ' + v['version'] if v['available'] == 'yes' else 'MISSING'}"
                      for k, v in st.items())


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

def _pip_base() -> List[str]:
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            "--no-input", "--prefer-binary"]


def ensure_pip(log: Optional[Callable[[str], None]] = None) -> bool:
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True,
                       capture_output=True, timeout=60)
        return True
    except Exception:
        pass
    try:
        if log:
            log("Bootstrapping pip (ensurepip)...")
        subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True,
                       capture_output=True, timeout=300)
        return True
    except Exception as exc:
        if log:
            log(f"ensurepip failed: {exc}")
        return False


def install_commands(requirements: Sequence[str], target: Optional[str] = None,
                     is_site: bool = True) -> List[List[str]]:
    """pip command lines needed to install ``requirements``."""
    cmds: List[List[str]] = []
    tgt = [] if (target is None or is_site) else ["--target", target, "--upgrade"]
    plain = [r for r in requirements if r != "mediapipe"]
    if "mediapipe" in requirements:
        cmds.append(_pip_base() + tgt + ["--no-deps", "mediapipe"])
        plain = list(dict.fromkeys(plain + list(MEDIAPIPE_RUNTIME_DEPS)))
    if plain:
        cmds.append(_pip_base() + tgt + plain)
    return cmds


def run_install(
    requirements: Optional[Sequence[str]] = None,
    target: Optional[str] = None,
    log: Optional[Callable[[str], None]] = None,
    timeout: float = 1800.0,
) -> Tuple[bool, str]:
    """Blocking install. Returns (ok, combined output tail)."""
    reqs = list(requirements) if requirements is not None else missing_requirements()
    if not reqs:
        return True, "All dependencies already installed."
    if not ensure_pip(log):
        return False, "pip is not available in Blender's Python."
    if target is None:
        target, is_site = default_install_target()
    else:
        is_site = os.path.abspath(target) == os.path.abspath(blender_site_packages())
    os.makedirs(target, exist_ok=True)
    out_tail = []
    for cmd in install_commands(reqs, target, is_site):
        if log:
            log("$ " + " ".join(cmd[2:]))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "pip timed out"
        out_tail.append((proc.stdout or "")[-2000:] + (proc.stderr or "")[-2000:])
        if proc.returncode != 0:
            return False, "\n".join(out_tail)
    if not is_site and target not in sys.path:
        sys.path.append(target)
    importlib.invalidate_caches()
    return True, "\n".join(out_tail)


# ---------------------------------------------------------------------------
# Pose models
# ---------------------------------------------------------------------------

def model_dir(override: str = "") -> str:
    if override:
        os.makedirs(override, exist_ok=True)
        return override
    return user_data_dir("models")


def model_filename(variant: str) -> str:
    return f"pose_landmarker_{variant}.task"


def model_url(variant: str) -> str:
    base = os.environ.get("BODYMOCAP_MODEL_BASE_URL", MODEL_BASE_URL).rstrip("/")
    if base.startswith("file://"):
        return f"{base}/{model_filename(variant)}"
    return f"{base}/pose_landmarker_{variant}/float16/latest/{model_filename(variant)}"


def model_path(variant: str, directory: str = "") -> str:
    return os.path.join(model_dir(directory), model_filename(variant))


def model_available(variant: str, directory: str = "") -> bool:
    p = model_path(variant, directory)
    return os.path.isfile(p) and os.path.getsize(p) > 100_000


def download_model(variant: str, directory: str = "", timeout: float = 120.0) -> Tuple[bool, str]:
    """Download a MediaPipe pose landmarker model. Returns (ok, path_or_error)."""
    if variant not in MODEL_VARIANTS:
        return False, f"Unknown model variant {variant}"
    dest = model_path(variant, directory)
    url = model_url(variant)
    tmp = dest + ".part"
    try:
        ctx = None
        try:
            import ssl

            import certifi
            ctx = ssl.create_default_context(cafile=certifi.where())
        except Exception:
            ctx = None
        req = urllib.request.Request(url, headers={"User-Agent": "BodyMocap-Blender-Addon"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp, open(tmp, "wb") as fh:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                fh.write(chunk)
        if os.path.getsize(tmp) < 100_000:
            os.remove(tmp)
            return False, f"Download from {url} is too small; not a model file"
        os.replace(tmp, dest)
        return True, dest
    except Exception as exc:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False, f"Model download failed ({url}): {exc}"
