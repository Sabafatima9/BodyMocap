# BodyMocap

Blender add-on for **camera-based body motion capture**, **live rig driving**,
**one-click record → Action bake**, and **topology-adaptive retargeting**
(N-bone chain → M-bone chain, e.g. 2-bone arms → 4-bone arms).

Compatible with **Blender 4.2 – 5.x** (slotted Actions on 4.4+, Blender 5 API).
Runs fully locally; no webcam or ML packages are required to use the synthetic
performer / recorded takes.

## Features

- Camera/video/image/synthetic capture with live preview + confidence-colored
  skeleton overlay, exposure compensation (AUTO/MANUAL + CLAHE) and lighting
  classification
- MediaPipe Pose (Tasks API) backend + offline synthetic/replay backend
- **POSE_OT_spawn_camera** — a scene camera matching the tracking FOV, aimed at
  the rig, with the feed as camera background
- One-click **Record / Stop & Bake**: stops the take, bakes every target rig to
  an Action and applies it
- **Topology-adaptive solver**: automatic limb/spine chain detection, arc-length
  proportional rotation distribution, pole-vector fallback, knee-inversion
  guard, lost-chain hold (last pose → calibrated neutral → rest)
- Cross-armature retargeting **2↔4 bones** (and continuous 3-bone chains)
  through pseudo-landmarks; validated to < 1.4 % end-effector error
- Auto-provisioned test rigs: `RigA_Standard` (2-bone limbs, T-pose),
  `RigB_Segmented` (4-bone limbs + 4-bone spine, A-pose), `RigC_Continuous`
  (generic 3-bone limbs) with skinned mannequin meshes
- Missing OpenCV/MediaPipe never blocks registration; install them from the
  panel (threaded, non-blocking) and download the pose model in one click

## Documentation

| Document | Path |
|----------|------|
| Software Requirements Specification | [`docs/SRS.md`](docs/SRS.md) |
| Project Proposal | [`docs/PROPOSAL.md`](docs/PROPOSAL.md) |
| Autonomous test report (2.1.0) | [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) |
| FR verification checklist | [`docs/VERIFICATION.md`](docs/VERIFICATION.md) |
| Install guide | [`INSTALL.md`](INSTALL.md) |

## Quick install

1. Build zip: `python scripts/make_addon_zip.py` → `dist/bodymocap.zip`
2. Blender → Edit → Preferences → Add-ons → Install… → select `bodymocap.zip`
3. Enable **BodyMocap** (category: Animation)
4. **3D Viewport → Sidebar (N) → Mocap** → *Setup ▸ Install Dependencies* and
   *Download Pose Model* for live camera capture (optional)

## Workflow

1. Select a humanoid **Armature** (or *Setup ▸ Create Test Rigs*)
2. Pick a **Source**: Webcam / Video / Image Sequence / Synthetic / Take
3. **Spawn Camera**, then **Start Tracking**; **Calibrate** in the T/A pose
4. **Record** → perform → **Stop & Bake** (one click; auto-bakes to an Action)
5. Optional: **Retarget Action** from any source rig to the target rig

## Tests

```bash
# pure-Python unit tests (no Blender needed)
python tests/run_tests.py

# headless Blender acceptance suite (~35 s), writes dist/test_report.json
blender -b --factory-startup -P tests/blender_test_suite.py
```

See [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) for measured accuracy
(≤ 1.38 % worst-case 2↔4 bone parity, 375 FPS offline, 32 FPS live) and the
list of hardware-limited items.

## Package layout

Installable package is `bodymocap/` (must remain the zip root folder name).

## License

Prototype project code — align redistribution with chosen pose-backend licenses
before public release.
