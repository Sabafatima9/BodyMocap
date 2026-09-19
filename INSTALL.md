# BodyMocap Installation

Works with **Blender 4.2 – 5.x**. Validated on Blender 5.2.1 LTS
(Windows); the code is OS-agnostic.

## 1. Install the add-on zip

From this repository:

```bash
python scripts/make_addon_zip.py
```

Output: `dist/bodymocap.zip`

In Blender:

1. **Edit → Preferences → Add-ons**
2. **Install…** → select `bodymocap.zip`
3. Enable **BodyMocap** (category: *Animation*, 3D Viewport label *Mocap*)
4. Open **3D Viewport → Sidebar (N) → Mocap**

Disable/uninstall via the same Add-ons preferences panel (FR-002).

## 2. Optional Python dependencies (live camera + MediaPipe)

BodyMocap registers without these; the Synthetic source, recorded takes and all
retargeting work without them.

**Recommended — install from inside Blender:**

1. **Mocap ▸ Setup ▸ Install Dependencies** — installs
   `opencv-contrib-python-headless` + `mediapipe` (+ runtime deps) into a
   writable location (Blender's site-packages when writable, otherwise a
   per-user folder that is added to `sys.path`). Runs on a worker thread;
   Blender stays responsive.
2. **Mocap ▸ Tracking ▸ Download Pose Model** — fetches
   `pose_landmarker_lite.task` (~5 MB) from Google's MediaPipe model storage.

**Manual alternative** (uses Blender's bundled Python; note the 5.x path):

```bash
# Ensure pip exists inside Blender's Python
"<blender>/5.2/python/bin/python.exe" -m ensurepip --upgrade   # Windows
# Linux/macOS: <blender>/5.2/python/bin/python3.13 -m ensurepip --upgrade

# Packages (headless OpenCV avoids clashing with the MediaPipe wheel)
"<blender>/5.2/python/bin/python.exe" -m pip install \
    opencv-contrib-python-headless mediapipe
```

| OS | Typical interpreter path |
|----|--------------------------|
| Windows | `C:\Program Files\Blender Foundation\Blender 5.2\5.2\python\bin\python.exe` |
| Linux | `/path/to/blender/5.2/python/bin/python3.13` |
| macOS | `Blender.app/Contents/Resources/5.2/python/bin/python3.13` |

Packages: `numpy` (bundled with Blender), `opencv-contrib-python-headless`
(camera + overlay), `mediapipe` (pose landmarks). Pin versions for production
deployments after validating against your Blender Python ABI.

## 3. Verify

1. Enable the add-on — Blender must not crash
2. **Setup ▸ Refresh** — shows OK/MISSING per dependency
3. **Source: Synthetic Performer** → **Start Tracking** — status and preview
   update; **Record → Stop & Bake** creates an Action
4. With OpenCV + camera: **Source: Webcam** → **Start Tracking** — preview
   image `BodyMocap_Preview` updates

## 4. Run the test suites (optional)

```bash
python tests/run_tests.py                                    # unit tests
blender -b --factory-startup -P tests/blender_test_suite.py  # acceptance
```

The Blender suite writes `dist/test_report.json` and exits non-zero on failure.

## 5. Uninstall

Preferences → Add-ons → BodyMocap → uncheck / Remove. Optionally delete the
add-on folder under Blender's `scripts/addons/bodymocap` and the data folder
(`%APPDATA%\Blender Foundation\Blender\5.2\datafiles\bodymocap` on Windows).
