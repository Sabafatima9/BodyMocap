# BodyMocap — Autonomous Test Report

**Product:** BodyMocap 2.1.0 (Blender add-on)
**Date:** 2026-09-20
**Method:** fully autonomous — synthetic camera/landmark feeds, headless Blender,
live Blender session via the Blender MCP bridge. No human intervention.
**Machine:** Windows 11, CPU inference, Blender **5.2.1 LTS**, Python 3.13.13.

---

## 1. How to reproduce

```bash
# 1) Pure-Python unit tests (no Blender) — math, solver, topology, presets
python tests/run_tests.py

# 2) Full in-Blender acceptance suite (headless, ~35 s)
blender -b --factory-startup -P tests/blender_test_suite.py

#    fast subset (~10 s)
blender -b --factory-startup -P tests/blender_test_suite.py -- --quick
```

The suite writes a machine-readable report to **`dist/test_report.json`** and
exits non-zero if any check fails. `dist/mediapipe_smoke.png` is the rendered
frame used by the vision smoke test.

---

## 2. Results summary

| Suite | Checks | Result |
|-------|--------|--------|
| `tests/run_tests.py` (unit, pure Python) | 70 | **70 pass / 0 fail** |
| `tests/blender_test_suite.py` (headless Blender) | 47 | **47 pass / 0 fail / 0 skipped** |
| Live Blender session (modal record → bake, retarget op) | 4 | **pass** |

### 2.1 Asset auto-provisioning (T1)

| Rig | Bones | Topology detected |
|-----|-------|-------------------|
| `RigA_Standard` | 21 | 2-segment limbs (`upper=1 lower=1`), 2-bone spine, T-pose |
| `RigB_Segmented` | 31 | 4-segment limbs (`upper=2 lower=2`), 4-bone spine, A-pose, rolled bones |
| `RigC_Continuous` | 24 | generic 3-bone limbs, no anatomical joint → **CONTINUOUS** arc-length solve |

Armatures and skinned mannequin meshes are generated procedurally
(`bodymocap/assets/rig_specs.py`, `build.py`) — no downloads required.

### 2.2 Operator/UI surface (T2)

- `POSE_OT_spawn_camera` (`pose.spawn_camera`) — spawns `BodyMocap_TrackingCam`
  with FOV/`scene.camera`/resolution matched to the tracking settings, aimed at
  the target rig, live feed as camera background. Verified FOV = 55.0°.
- `POSE_OT_start_capture`, `POSE_OT_bake_animation`, `POSE_OT_retarget_action`
  registered and functional.
- Panel `VIEW3D_PT_pose_retargeter` registered in `View3D ▸ Sidebar ▸ Mocap`
  with 8 sub-panels (source, tracking, lighting, topology, record/bake,
  retarget, setup).

### 2.3 Topology-agnostic retargeting accuracy (T3, T4, T5)

Same synthetic capture stream solved on Rig A (2-bone arm) and Rig B (4-bone
arm); error = end-effector distance / limb length (wrists, hands, elbows,
ankles, knees, toes).

| Condition (lighting sim) | Worst Rig A↔B error |
|--------------------------|---------------------|
| clean | 0.000 % |
| normal | 0.100 % |
| low_contrast | 0.228 % |
| extreme_key | 0.414 % |
| backlight | 0.503 % |
| low_light | 0.626 % |
| **all 11 clips × 2 conditions × 2 speeds** | **1.380 %** |
| Baked actions, 3 clips (incl. backlight x2.0, low_light x0.5) | **0.318 %** |
| Armature→armature retarget 2→4 | **0.058 %** |
| Armature→armature retarget 4→2 | **0.058 %** |

**Acceptance budget was < 5 % — worst measured 1.38 % (≈3.6× margin).**

### 2.4 Record → bake quality (T4)

- Keyframe density **1.00** (a key on every channel of every frame); no NaN/Inf
  keys.
- Curve smoothness (mean frame-to-frame step / mean second difference):
  wave 2.2°/1.3°, punch-backlit 14.6°/9.6°, march-low-light 12.2°/8.7° — no
  overshoot or jitter (jitter ratio < 0.7).
- Example one-click live take: **452 frames recorded at 32 FPS → 510-frame
  Action on 3 rigs in one stop** (42 840 + 63 240 + 48 960 keys).

### 2.5 Performance (measured on the test machine, CPU only)

| Metric | Value |
|--------|-------|
| Solver throughput (pure Python) | **1.86 ms/frame** (6624 solves) |
| Offline pipeline (landmark feeds incl. bake) | **375 FPS** |
| Live modal pipeline (synthetic source, GUI) | **32 FPS** sustained |
| MediaPipe Pose-Landmarker Lite (640×480) | **28.2 ms/frame** after warm-up (≈35 FPS), 33 landmarks detected on a rendered mannequin |
| First MediaPipe frame (graph warm-up) | 56 ms |

Targets in the SRS were ≥15 FPS live and ≤150 ms latency; both are met with
margin even with the mock/CPU path.

### 2.6 Robustness (T3, T6, T7)

- **Lighting:** landmark-level conditions (noise, adversarial visibility,
  dropout, per-frame misses) and pixel-level degradations (low contrast,
  under/over-exposure, backlight, noise) are both exercised. The AUTO exposure
  pipeline classifies all four lighting conditions correctly and moves each
  frame's mean luminance toward the 0.45 target (e.g. underexposed 0.09 → 0.49);
  `OFF` mode is bit-exact.
- **Motion speed:** 0.5×, 1×, 2×, 4× playback all within tolerance.
- **Lost landmarks:** chains hold their last valid pose; before the first valid
  frame they hold the *calibrated neutral* pose so different topologies/rest
  poses (T-pose vs A-pose) still agree in world space (see §4).
- **Knee inversion / pole stability:** verified in the unit suite
  (`test_knee_inversion_is_corrected`, `test_straight_arm_pole_stability`).
- **Pose detection from real pixels:** the spawned tracking camera renders the
  mannequin, MediaPipe (Tasks API, `pose_landmarker_lite.task`) detects 33
  landmarks from that render, giving an end-to-end camera→landmarks check
  without a webcam.

---

## 3. Hardware-limited items (honest status)

These are **not** failures; they need hardware/UI the CI box does not have:

| Item | Status |
|------|--------|
| Live webcam device capture (OpenCV `VideoCapture`) | Not exercised (no camera); code path is the same one validated with image-sequence/video sources |
| Usability dry-run with a person (NFR-005) | Not applicable headless |
| Multi-OS matrix (NFR-006) | Only Windows validated here; code is OS-agnostic |

---

## 4. Defects found and fixed during this validation

1. **Pure-Python tests could not import the package** — `bodymocap/__init__.py`
   imported `bpy` unconditionally, so `python tests/run_tests.py` failed with
   7 import errors. `bpy` is now imported behind a guard; the 70-test suite
   runs without Blender again (NFR-007).
2. **Dependency auto-install chose an unwritable directory on Windows** —
   `os.access(path, W_OK)` returns `True` for `C:\Program Files\...` even when
   writes are denied, so `pip --target` failed. `deps.default_install_target()`
   now probes with a real write and falls back to the per-user site-packages,
   which is appended to `sys.path` at registration.
3. **Chains lost on the first frames snapped to each rig's own rest pose** —
   with a T-pose rig and an A-pose rig, the same capture stream started with a
   ~76 %-of-limb-length mismatch when a wrist was occluded at frame 0.
   `RetargetSolver._hold_chain()` now falls back to the calibrated neutral pose
   before the raw rest pose; parity from frame 0 is < 0.4 % with a calibration.
   Regression test: `tests/test_solver.py::test_first_frame_loss_uses_calibrated_neutral`.

---

## 5. Environment used

| Component | Version / path |
|-----------|----------------|
| Blender | 5.2.1 LTS (`C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`) |
| Blender Python | 3.13.13 |
| numpy | 2.3.4 (bundled with Blender) |
| opencv-contrib-python-headless | 5.0.0.93 (installed into the add-on user site) |
| mediapipe | 1.0.1 + `pose_landmarker_lite.task` |
| Add-on install path | `%APPDATA%\Blender Foundation\Blender\5.2\scripts\addons\bodymocap` |
| Add-on data path | `%APPDATA%\Blender Foundation\Blender\5.2\datafiles\bodymocap` |

Dependencies are installed from inside Blender with **Setup ▸ Install
Dependencies** (threaded, non-blocking) and the model with **Download Pose
Model**; both were exercised in the live session.

---

## 6. Verdict

All acceptance criteria that can be validated without a physical camera pass:

- zero-intervention provisioning, capture, record, bake and retarget in headless
  and live Blender 5.2,
- 2-bone ↔ 4-bone (and 3-bone continuous) retargeting within **1.38 %** worst
  case against a 5 % budget,
- clean Blender API lifecycle (`register()`/`unregister()`, AddonPreferences,
  slotted-Action compatibility for 4.4–5.x),
- reproducible test pipeline with machine-readable evidence
  (`dist/test_report.json`).
