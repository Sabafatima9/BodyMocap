# BodyMocap Installation

## 1. Install the add-on zip

From this repository:

```bash
python scripts/make_addon_zip.py
```

Output: `dist/bodymocap.zip`

In Blender 4.x:

1. **Edit → Preferences → Add-ons**
2. **Install…** → select `bodymocap.zip`
3. Enable **BodyMocap**
4. Open **3D Viewport → Sidebar (N) → BodyMocap**

Disable/uninstall via the same Add-ons preferences panel (FR-002).

## 2. Optional Python dependencies (live camera + MediaPipe)

BodyMocap registers without these. Camera + MediaPipe operators report actionable errors if missing. **Mock / Offline** works without them.

Install into **Blender’s bundled Python** (not system Python). Find the interpreter:

| OS | Typical path |
|----|----------------|
| Linux | `/path/to/blender/4.x/python/bin/python3.11` |
| macOS | `Blender.app/Contents/Resources/4.x/python/bin/python3.11` |
| Windows | `Blender\4.x\python\bin\python.exe` |

Example:

```bash
# Ensure pip exists inside Blender’s Python
blender --python-expr "import ensurepip; ensurepip.bootstrap()"

# Then (adjust path to Blender’s python):
/path/to/blender/python/bin/python -m pip install --upgrade pip
/path/to/blender/python/bin/python -m pip install numpy opencv-python-headless mediapipe
```

Alternative one-liner pattern:

```bash
blender --python-expr "import pip; pip.main(['install', 'opencv-python-headless', 'mediapipe', 'numpy'])"
```

(If `pip.main` is unavailable on your Blender build, use the explicit `python -m pip` path above.)

### Packages

| Package | Purpose |
|---------|---------|
| `numpy` | Preview pixel upload, array ops |
| `opencv-python-headless` | Webcam capture + overlay draw |
| `mediapipe` | On-device Pose landmarks |

Pin versions in production deployments after validating against your Blender Python ABI.

## 3. Verify

1. Enable add-on — Blender must not crash
2. Panel → **Refresh** — shows OK/MISSING per dependency
3. Set backend to **Mock / Offline** → **Start** — tracking status updates
4. With OpenCV + camera: backend **MediaPipe** → **Start** — preview Image `BodyMocap_Preview` updates

## 4. Uninstall

Preferences → Add-ons → BodyMocap → uncheck / Remove. Optional: delete installed add-on folder under Blender’s `scripts/addons/bodymocap`.
