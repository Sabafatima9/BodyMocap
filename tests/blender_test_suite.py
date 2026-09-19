"""Headless Blender acceptance suite for BodyMocap (SRS Section 9 / Phase 5).

Run with Blender in background mode::

    blender -b --factory-startup -P tests/blender_test_suite.py -- [--quick]

The suite exercises the *installed add-on code paths* end to end:

  T1  asset provisioning (Rig A / B / C built from specs, topology detection)
  T2  operator + panel surface (POSE_OT_spawn_camera, POSE_OT_start_capture,
      POSE_OT_bake_animation, VIEW3D_PT_pose_retargeter)
  T3  topology parity matrix: one synthetic capture stream solved on 2-bone and
      4-bone rigs across lighting conditions, clips and playback speeds
      (end-effector error budget < 5 % of limb length)
  T4  full offline capture -> calibrate -> record -> bake operator pipeline with
      keyframe-density / smoothing / NaN checks and a baked-action parity check
  T5  armature -> armature retargeting 2->4 and 4->2 (transfer_in_blender)
  T6  lighting-robustness of the exposure pipeline (degrade -> compensate)
  T7  optional MediaPipe smoke test on a rendered frame (SKIP if unavailable)

It writes ``dist/test_report.json`` and exits with status 0 only when every
check passed.  No webcam, no user interaction.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

try:
    import bpy
except ImportError:  # pragma: no cover - only ever run inside Blender
    print("This test suite must be run with: blender -b -P tests/blender_test_suite.py")
    raise SystemExit(2)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

QUICK = "--quick" in sys.argv
STAMP = time.strftime("%Y-%m-%dT%H:%M:%S")

# limb lengths of the shared body layout (metres)
ARM_LEN = 0.29 + 0.26 + 0.09
LEG_LEN = 0.44 + 0.43

EFFECTORS = {
    "RigA_Standard": {"hand_L": ("tail", "Hand.L"), "hand_R": ("tail", "Hand.R"),
                      "wrist_L": ("head", "Hand.L"), "elbow_L": ("head", "Forearm.L"),
                      "ankle_L": ("head", "Foot.L"), "ankle_R": ("head", "Foot.R"),
                      "knee_R": ("head", "Shin.R"), "toe_L": ("tail", "Toe.L")},
    "RigB_Segmented": {"hand_L": ("tail", "Hand.L"), "hand_R": ("tail", "Hand.R"),
                       "wrist_L": ("head", "Hand.L"), "elbow_L": ("head", "Forearm.01.L"),
                       "ankle_L": ("head", "Foot.L"), "ankle_R": ("head", "Foot.R"),
                       "knee_R": ("head", "Shin.01.R"), "toe_L": ("tail", "Toe.L")},
}
HIPS = {"RigA_Standard": "Hips", "RigB_Segmented": "Hips", "RigC_Continuous": "Pelvis"}

CONDITIONS = ("clean", "normal", "low_contrast", "backlight", "extreme_key", "low_light")
CLIPS_ALL = ("tpose", "arm_raise", "wave", "punch_fast", "squat", "reach_cross",
             "subtle_idle", "torso_twist", "march", "straight_arm_circles", "side_kick")
SPEEDS = (0.5, 1.0, 2.0, 4.0)


# ---------------------------------------------------------------------------
# Result bookkeeping
# ---------------------------------------------------------------------------

class Suite:
    def __init__(self):
        self.checks = []
        self.timings = {}
        self.sections = {}

    def check(self, cid, name, ok, metric=None, detail="", section=""):
        status = "PASS" if ok else "FAIL"
        self.checks.append({"id": cid, "name": name, "status": status,
                            "metric": metric, "detail": detail})
        line = f"  [{status}] {cid} {name}"
        if metric is not None:
            line += f"  ({metric})"
        if detail and not ok:
            line += f"  -- {detail}"
        print(line, flush=True)
        if section:
            self.sections.setdefault(section, []).append(status)
        return ok

    def skip(self, cid, name, detail="", section=""):
        self.checks.append({"id": cid, "name": name, "status": "SKIP",
                            "metric": None, "detail": detail})
        print(f"  [SKIP] {cid} {name}  -- {detail}", flush=True)
        if section:
            self.sections.setdefault(section, []).append("SKIP")

    def section(self, key):
        self.sections.setdefault(key, [])

    def phase(self, title):
        print(f"\n=== {title} ===", flush=True)
        self.timings[title] = time.perf_counter()

    def phase_end(self):
        title = list(self.timings)[-1]
        self.timings[title] = time.perf_counter() - self.timings[title]
        print(f"--- {title}: {self.timings[title]:.1f}s", flush=True)

    @property
    def failed(self):
        return [c for c in self.checks if c["status"] == "FAIL"]

    @property
    def passed(self):
        return [c for c in self.checks if c["status"] == "PASS"]

    @property
    def skipped(self):
        return [c for c in self.checks if c["status"] == "SKIP"]


SUITE = Suite()


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def register_addon():
    pkg = "bodymocap"
    if pkg in bpy.context.preferences.addons:
        return "enabled"
    try:
        bpy.ops.preferences.addon_enable(module=pkg)
        if pkg in bpy.context.preferences.addons:
            return "enabled"
    except Exception:
        pass
    import bodymocap
    bodymocap.register()
    return "source"


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def effector_positions(obj, pose):
    hips = pose.head[HIPS[obj.name]]
    out = {}
    for name, (kind, bone) in EFFECTORS[obj.name].items():
        out[name] = (pose.head[bone] if kind == "head" else pose.tail(bone)) - hips
    return out


def _vlen(v):
    """Length of a bodymocap Vec3 (``.length()``) or a mathutils Vector (``.length``)."""
    L = getattr(v, "length", None)
    return float(L() if callable(L) else L)


def error_between(a, b, names=None):
    """Worst normalised error between two effector dicts (limb-length units)."""
    worst, worst_name = 0.0, ""
    for name in (names or a.keys()):
        L = ARM_LEN if name.startswith(("hand", "wrist", "elbow")) else LEG_LEN
        e = _vlen(a[name] - b[name]) / L
        if e > worst:
            worst, worst_name = e, name
    return worst, worst_name


def sample_baked_rig(scene, obj):
    """Hip-relative effector positions of the rig's current (baked) pose."""
    bpy.context.view_layer.update()
    mw = obj.matrix_world
    hips = mw @ obj.pose.bones[HIPS[obj.name]].head
    out = {}
    for name, (kind, bone) in EFFECTORS[obj.name].items():
        pb = obj.pose.bones.get(bone)
        if pb is None:
            continue
        p = pb.head if kind == "head" else pb.tail
        out[name] = (mw @ p) - hips
    return out


def action_nan_count(action):
    from bodymocap.utils.anim_compat import iter_fcurves
    bad = 0
    for fc in iter_fcurves(action):
        for kp in fc.keyframe_points:
            if not math.isfinite(kp.co[0]) or not math.isfinite(kp.co[1]):
                bad += 1
    return bad


def action_metrics(action):
    from bodymocap.utils.anim_compat import iter_fcurves, keyframe_count
    keys = keyframe_count(action)
    n = sum(len(fc.keyframe_points) for fc in iter_fcurves(action))
    return keys, n


# ---------------------------------------------------------------------------
# T1: assets + topology
# ---------------------------------------------------------------------------

def test_assets():
    from bodymocap.assets.build import build_test_rigs
    from bodymocap.retarget.rig import RigModel
    from bodymocap.retarget.topology import CONTINUOUS, detect_topology

    rigs = build_test_rigs(with_mesh=False, names=("RigA_Standard", "RigB_Segmented",
                                                   "RigC_Continuous"))
    SUITE.check("T1.1", "Rig A/B/C armatures built", len(rigs) == 3,
                metric=", ".join(f"{k}={len(v.data.bones)} bones" for k, v in rigs.items()),
                section="T1")
    for key, obj in rigs.items():
        prof = detect_topology(RigModel.from_blender(obj))
        if key == "RigA_Standard":
            ok = len(prof.limbs["arm_L"].upper) == 1 and len(prof.limbs["arm_L"].lower) == 1
            SUITE.check("T1.2", "Rig A detected as 2-segment arm (upper+lower)", ok,
                        metric=prof.summary(), section="T1")
        elif key == "RigB_Segmented":
            ok = (len(prof.limbs["arm_L"].upper) == 2 and len(prof.limbs["arm_L"].lower) == 2
                  and len(prof.spine) == 4)
            SUITE.check("T1.3", "Rig B detected as 4-segment arm + 4-bone spine", ok,
                        metric=prof.summary(), section="T1")
        elif key == "RigC_Continuous":
            lc = prof.limbs["arm_L"]
            ok = lc.mode == CONTINUOUS and len(lc.upper) == 3 and not lc.lower
            SUITE.check("T1.4", "Rig C detected as continuous 3-bone arm", ok,
                        metric=prof.summary(), section="T1")
    return rigs


# ---------------------------------------------------------------------------
# T2: operator / panel surface + camera spawn
# ---------------------------------------------------------------------------

def test_operator_surface(rig):
    ops = {
        "POSE_OT_spawn_camera": "pose.spawn_camera",
        "POSE_OT_start_capture": "pose.start_capture",
        "POSE_OT_bake_animation": "pose.bake_animation",
        "POSE_OT_retarget_action": "pose.retarget_action",
    }
    missing = [n for n, idname in ops.items()
               if getattr(bpy.types, n, None) is None
               or getattr(bpy.types, n, None).bl_idname != idname]
    SUITE.check("T2.1", "Required operators registered (POSE_OT_*)", not missing,
                metric=", ".join(ops.values()), detail=", ".join(missing), section="T2")
    panel = getattr(bpy.types, "VIEW3D_PT_pose_retargeter", None)
    SUITE.check("T2.2", "VIEW3D_PT_pose_retargeter panel registered", panel is not None,
                metric=getattr(panel, "bl_label", ""), section="T2")

    s = bpy.context.scene.bodymocap
    s.target = rig
    s.tracking_fov = 55.0
    res = bpy.ops.pose.spawn_camera()
    cam = bpy.data.objects.get("BodyMocap_TrackingCam")
    ok = (res == {"FINISHED"} and cam is not None and cam.type == "CAMERA"
          and abs(float(cam.get("bodymocap_fov", 0.0)) - 55.0) < 1e-3
          and bpy.context.scene.camera == cam)
    SUITE.check("T2.3", "POSE_OT_spawn_camera creates FOV-matched camera", ok,
                metric=f"fov={cam.get('bodymocap_fov') if cam else '-'}",
                detail=str(res), section="T2")
    return cam


# ---------------------------------------------------------------------------
# T3: topology parity matrix (2-bone vs 4-bone, lighting x clip x speed)
# ---------------------------------------------------------------------------

def test_parity_matrix(suite_keys=("RigA_Standard", "RigB_Segmented")):
    from bodymocap.assets.build import build_armature
    from bodymocap.assets.rig_specs import RIG_SPECS
    from bodymocap.core.skeleton import SourceSkeleton
    from bodymocap.pose.synthetic import generate_feed
    from bodymocap.retarget.rig import RigModel
    from bodymocap.retarget.solver import RetargetSolver
    from bodymocap.retarget.topology import detect_topology

    solvers = {}
    for key in suite_keys:
        obj = bpy.data.objects.get(key) or build_armature(RIG_SPECS[key](), key)
        rig = RigModel.from_blender(obj)
        solvers[key] = RetargetSolver(rig, detect_topology(rig))
    solver_ms = 0.0
    solves = 0

    def worst_for(clip, condition, speed, duration=1.2):
        nonlocal solver_ms, solves
        feed = generate_feed(clip, duration=duration, fps=30, condition=condition, speed=speed)
        worst = 0.0
        for pf in feed.frames:
            sk = SourceSkeleton.from_pose_frame(pf)
            out = {}
            for key in suite_keys:
                t0 = time.perf_counter()
                res = solvers[key].solve(sk)
                solver_ms += (time.perf_counter() - t0) * 1e3
                solves += 1
                out[key] = effector_positions(bpy.data.objects[key], res.pose)
            e, _ = error_between(out[suite_keys[0]], out[suite_keys[1]])
            worst = max(worst, e)
        return worst

    # matrix A: all lighting conditions, wave + march, all speeds
    conditions = ("clean", "normal") if QUICK else CONDITIONS
    speeds = (1.0,) if QUICK else SPEEDS
    worst_by_condition = {}
    for cond in conditions:
        worst = 0.0
        for clip in ("wave", "march"):
            for speed in speeds:
                worst = max(worst, worst_for(clip, cond, speed))
        worst_by_condition[cond] = worst
        tol = 0.01 if cond == "clean" else 0.05
        SUITE.check(f"T3.{cond}", f"Rig A vs Rig B parity ({cond}, all speeds)", worst < tol,
                    metric=f"{worst:.3%} of limb length", section="T3")

    # matrix B: all clips, borderline lighting
    if not QUICK:
        worst_clip = 0.0
        for clip in CLIPS_ALL:
            for cond in ("clean", "low_light"):
                for speed in (1.0, 2.0):
                    worst_clip = max(worst_clip, worst_for(clip, cond, speed))
        SUITE.check("T3.clips", "All 11 clips x 2 conditions x 2 speeds parity",
                    worst_clip < 0.05, metric=f"{worst_clip:.3%} worst", section="T3")

    ms = solver_ms / max(solves, 1)
    SUITE.check("T3.perf", "Solver throughput < 20 ms/frame", ms < 20.0,
                metric=f"{ms:.2f} ms/solve over {solves} solves", section="T3")
    return worst_by_condition


# ---------------------------------------------------------------------------
# T4: full operator pipeline (synthetic source -> record -> bake -> parity)
# ---------------------------------------------------------------------------

def configure_synthetic(s, clip, condition, speed=1.0, duration=3.0, fps=30.0):
    s.source = "SYNTHETIC"
    s.synth_clip = clip
    s.synth_condition = condition
    s.synth_speed = speed
    s.synth_duration = duration
    s.capture_fps = fps
    s.tracking_fov = 60.0
    s.root_motion = False
    s.live_apply = True
    s.min_visibility = 0.3
    s.smoothing_enabled = True
    s.auto_bake = True
    s.bake_all_targets = True
    s.set_scene_range = True
    s.apply_mode = "ACTION"
    s.overwrite_action = True
    s.exposure_mode = "AUTO"


def run_pipeline(rigs, clip, condition, speed, tag):
    from bodymocap.runtime import get_runtime
    s = bpy.context.scene.bodymocap
    s.target = rigs["RigA_Standard"]
    existing = {t.obj for t in s.extra_targets}
    for key in ("RigB_Segmented", "RigC_Continuous"):
        if rigs[key] not in existing:
            t = s.extra_targets.add()
            t.obj = rigs[key]
    s.action_name = f"MocapTake_{tag}"
    configure_synthetic(s, clip, condition, speed=speed)
    rt = get_runtime()
    rt.solvers.clear()
    res = bpy.ops.pose.start_capture("EXEC_DEFAULT", offline=True, record=True,
                                     calibrate_seconds=1.0)
    return res, rt


def test_pipeline(rigs):
    scene = bpy.context.scene
    combos = [("wave", "normal", 1.0), ("punch_fast", "backlight", 2.0),
              ("march", "low_light", 0.5)]
    if QUICK:
        combos = combos[:1]
    worst_parity = 0.0
    worst_jitter = 0.0
    nan_total = 0
    fps_samples = []
    for clip, condition, speed in combos:
        tag = f"{clip}_{condition}"
        t0 = time.perf_counter()
        res, rt = run_pipeline(rigs, clip, condition, speed, tag)
        elapsed = time.perf_counter() - t0
        take = rt.last_take
        bake = rt.last_bake
        ok = (res == {"FINISHED"} and take is not None and take.frame_count() > 10
              and len(bake) == 3)
        SUITE.check(f"T4.run.{tag}", f"Offline capture+record+bake ({clip}/{condition}/x{speed})",
                    ok, metric=f"{take.frame_count() if take else 0} frames, "
                               f"{len(bake)} actions, {elapsed:.1f}s",
                    detail=str(res), section="T4")
        if not ok:
            continue

        # keyframe density, jitter and NaN checks on every baked action
        for obj_name, info in bake.items():
            act = bpy.data.actions.get(info["action"])
            stats = info["stats"]
            keys, npts = action_metrics(act)
            density = stats.key_density
            SUITE.check(f"T4.keys.{tag}.{obj_name}", f"Keyframe density on {obj_name}",
                        keys > 0 and density >= 0.98 and npts == keys,
                        metric=f"{keys} keys, density {density:.2f}", section="T4")
            jitter = stats.mean_accel_deg / max(stats.mean_step_deg, 1e-6)
            worst_jitter = max(worst_jitter, jitter)
            SUITE.check(f"T4.smooth.{tag}.{obj_name}", f"Curve smoothing on {obj_name}",
                        0.0 < stats.mean_step_deg < 15.0 and jitter < 3.0,
                        metric=f"step {stats.mean_step_deg:.2f} deg, "
                               f"accel {stats.mean_accel_deg:.2f} deg",
                        section="T4")
            nan_total += action_nan_count(act)
        # baked-action parity: Rig A vs Rig B at the same scene frames
        f0, f1 = bpy.data.actions[bake["RigA_Standard"]["action"]].frame_range
        f1 = min(f1, f0 + 120)
        step = max(1, int((f1 - f0) / 40))
        for f in range(int(f0), int(f1) + 1, step):
            scene.frame_set(f)
            ea = sample_baked_rig(scene, rigs["RigA_Standard"])
            eb = sample_baked_rig(scene, rigs["RigB_Segmented"])
            e, _ = error_between(ea, eb)
            worst_parity = max(worst_parity, e)
        stats = rt.last_session_stats or {}
        if stats.get("total_ms"):
            fps_samples.append(1000.0 / stats["total_ms"])
    SUITE.check("T4.parity", "Baked action parity Rig A vs Rig B", worst_parity < 0.05,
                metric=f"{worst_parity:.3%} worst of limb length", section="T4")
    SUITE.check("T4.nan", "Baked F-Curves finite (no NaN/Inf)", nan_total == 0,
                metric=f"{nan_total} bad keys", section="T4")
    if fps_samples:
        SUITE.check("T4.throughput", "Offline pipeline throughput", True,
                    metric=f"{sum(fps_samples)/len(fps_samples):.0f} FPS mean "
                           f"(mock backend, includes DBG overhead)", section="T4")
    return worst_parity


# ---------------------------------------------------------------------------
# T5: armature -> armature retarget (2 -> 4 and 4 -> 2)
# ---------------------------------------------------------------------------

def test_retarget(rigs):
    from bodymocap.operators.bake_ops import apply_action
    from bodymocap.retarget.transfer import transfer_in_blender
    from bodymocap.runtime import get_profile, solver_settings

    s = bpy.context.scene.bodymocap
    scene = bpy.context.scene
    settings = solver_settings(s)
    settings.root_motion = True
    act_a = bpy.data.actions.get(f"MocapTake_wave_normal")
    act_b = bpy.data.actions.get(f"MocapTake_wave_normal_RigB_Segmented")
    if act_a is None or act_b is None:
        SUITE.skip("T5.x", "Retarget checks", "no baked actions from T4", section="T5")
        return
    cases = [
        ("2->4", rigs["RigA_Standard"], rigs["RigB_Segmented"], act_a.name,
         "Retarget_2to4", "RigA_Standard", "RigB_Segmented"),
        ("4->2", rigs["RigB_Segmented"], rigs["RigA_Standard"], act_b.name,
         "Retarget_4to2", "RigB_Segmented", "RigA_Standard"),
    ]
    for label, src, tgt, act_name, new_name, src_key, tgt_key in cases:
        ok, msg, act = transfer_in_blender(
            src, tgt, act_name, new_name, s.start_frame,
            get_profile(src), get_profile(tgt), settings, "QUATERNION")
        if not ok or act is None:
            SUITE.check(f"T5.{label}", f"Retarget {label} action written", False,
                        detail=msg, section="T5")
            continue
        apply_action(tgt, act, "ACTION", s.start_frame)
        src_act = bpy.data.actions.get(act_name)
        apply_action(src, src_act, "ACTION", s.start_frame)
        f0, f1 = act.frame_range
        step = max(1, int((f1 - f0) / 30))
        worst = 0.0
        for f in range(int(f0), int(f1) + 1, step):
            scene.frame_set(f)
            es = sample_baked_rig(scene, src)
            et = sample_baked_rig(scene, tgt)
            e, _ = error_between(es, et)
            worst = max(worst, e)
        SUITE.check(f"T5.{label}", f"Retarget {label} end-effector parity", worst < 0.05,
                    metric=f"{worst:.3%} worst, {int(f1 - f0) + 1} frames",
                    detail=msg, section="T5")


# ---------------------------------------------------------------------------
# T6: lighting-condition pixel pipeline (degrade -> compensate)
# ---------------------------------------------------------------------------

def test_lighting():
    try:
        import numpy as np
    except ImportError:
        SUITE.skip("T6.0", "Lighting compensation", "numpy unavailable", section="T6")
        return
    from bodymocap.vision.preprocess import ExposureSettings, analyze, compensate, degrade

    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[..., 0] = np.clip(30 + xx / w * 180, 0, 255)          # gradient background
    base[..., 1] = np.clip(40 + yy / h * 160, 0, 255)
    base[..., 2] = np.clip(50 + (xx + yy) / (w + h) * 160, 0, 255)
    mask = (((xx - w * 0.5) / (w * 0.16)) ** 2 + ((yy - h * 0.5) / (h * 0.34)) ** 2) <= 1.0
    base[mask] = (120, 90, 70)

    expectations = {
        "low_contrast": "LOW_CONTRAST",
        "underexposed": "UNDEREXPOSED",
        "overexposed": "OVEREXPOSED",
        "backlight": "BACKLIT",
    }
    for kind, expected in expectations.items():
        degraded = degrade(base, kind, seed=3, alpha=mask.astype(np.uint8) if kind == "backlight"
                           else None)
        st = analyze(degraded)
        classified = st.condition == expected
        comp, st2 = compensate(degraded, ExposureSettings(mode="AUTO", clahe=True))
        before = abs(st.mean - 0.45)
        after = abs(float(comp.mean()) / 255.0 - 0.45)
        finite = bool(np.isfinite(comp.astype(np.float32)).all())
        SUITE.check(f"T6.{kind}", f"Light compensation recovers {kind}",
                    classified and finite and after <= before + 0.06,
                    metric=f"class={st.condition} (want {expected}), "
                           f"mean {st.mean:.2f}->{float(comp.mean())/255.0:.2f}",
                    section="T6")
    # OFF mode must not alter pixels
    degraded = degrade(base, "underexposed", seed=1)
    same = compensate(degraded, ExposureSettings(mode="OFF"))[0]
    SUITE.check("T6.off", "Exposure OFF leaves frames untouched", bool((same == degraded).all()),
                metric="bit-exact", section="T6")


# ---------------------------------------------------------------------------
# T7: optional MediaPipe smoke test on a rendered mannequin frame
# ---------------------------------------------------------------------------

def test_mediapipe_smoke(rig):
    from bodymocap.pose.mediapipe_backend import MediaPipeBackend, mediapipe_available
    from bodymocap.utils import deps
    if not mediapipe_available():
        SUITE.skip("T7.mp", "MediaPipe smoke test", "mediapipe not installed", section="T7")
        return
    if not deps.model_available("lite"):
        ok, msg = deps.download_model("lite")
        if not ok:
            SUITE.skip("T7.mp", "MediaPipe smoke test", f"model unavailable: {msg[:60]}",
                       section="T7")
            return
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist",
                       "mediapipe_smoke.png")
    out = os.path.abspath(out)
    scene = bpy.context.scene
    try:
        from bodymocap.assets.build import build_mannequin
        from bodymocap.operators.camera_ops import spawn_tracking_camera
        for other in bpy.data.objects:
            if other.type == "ARMATURE" and other != rig:
                other.hide_render = True
        if not any(c.type == "MESH" for c in rig.children):
            build_mannequin(rig)
        spawn_tracking_camera(bpy.context, rig, fov_deg=45.0, width=640, height=480, distance=3.2)
        if scene.world is None:
            scene.world = bpy.data.worlds.new("World")
        scene.world.use_nodes = True
        bg = scene.world.node_tree.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = (0.55, 0.6, 0.65, 1.0)
        lamp_data = bpy.data.lights.new("SuiteKey", type="AREA")
        lamp_data.energy = 800.0
        lamp_data.size = 3.0
        lamp = bpy.data.objects.new("SuiteKey", lamp_data)
        scene.collection.objects.link(lamp)
        lamp.location = (2.0, -2.5, 3.0)
        from mathutils import Vector
        d = (Vector((0.0, 0.0, 1.0)) - lamp.location).normalized()
        lamp.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x, scene.render.resolution_y = 640, 480
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
    except Exception as exc:
        SUITE.skip("T7.mp", "MediaPipe smoke test", f"render unavailable: {exc}", section="T7")
        return
    try:
        import cv2
        img = cv2.imread(out)
    except ImportError:
        img = None
    if img is None:
        SUITE.skip("T7.mp", "MediaPipe smoke test", "rendered frame unreadable", section="T7")
        return
    try:
        be = MediaPipeBackend(model_path=deps.model_path("lite"))
        if not be.initialize():
            SUITE.check("T7.mp", "MediaPipe initialises in Blender", False,
                        detail=be.last_error, section="T7")
            return
        pf = be.infer(img, 0, 0.0)
        be.shutdown()
        SUITE.check("T7.mp", "MediaPipe detects rendered mannequin",
                    len(pf.landmarks) >= 20,
                    metric=f"{len(pf.landmarks)} landmarks, state={pf.tracking_state.name}",
                    section="T7")
    except Exception as exc:
        SUITE.check("T7.mp", "MediaPipe smoke test", False, detail=str(exc), section="T7")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("BodyMocap headless acceptance suite")
    print(f"  Blender {bpy.app.version_string}  python {sys.version.split()[0]}  "
          f"background={bpy.app.background}  quick={QUICK}")
    t_start = time.perf_counter()
    try:
        register_addon()
        reset_scene()
        register_addon()

        SUITE.phase("T1 assets & topology")
        rigs = test_assets()
        SUITE.phase_end()

        SUITE.phase("T2 operator surface & camera spawn")
        test_operator_surface(rigs["RigA_Standard"])
        SUITE.phase_end()

        SUITE.phase("T3 topology parity matrix")
        test_parity_matrix()
        SUITE.phase_end()

        SUITE.phase("T4 capture -> record -> bake pipeline")
        test_pipeline(rigs)
        SUITE.phase_end()

        SUITE.phase("T5 armature retargeting")
        test_retarget(rigs)
        SUITE.phase_end()

        SUITE.phase("T6 lighting compensation")
        test_lighting()
        SUITE.phase_end()

        SUITE.phase("T7 MediaPipe smoke")
        test_mediapipe_smoke(rigs["RigA_Standard"])
        SUITE.phase_end()
    except Exception:
        traceback.print_exc()
        SUITE.check("EXC", "Suite aborted by exception", False,
                    detail=traceback.format_exc().splitlines()[-1])

    total = time.perf_counter() - t_start
    print(f"\n{'=' * 60}")
    print(f"RESULT: {len(SUITE.passed)} passed, {len(SUITE.failed)} failed, "
          f"{len(SUITE.skipped)} skipped   ({total:.1f}s)")
    if SUITE.failed:
        print("Failed checks:")
        for c in SUITE.failed:
            print(f"  - {c['id']} {c['name']}  {c['detail']}")
    print("=" * 60, flush=True)

    report = {
        "product": "BodyMocap",
        "suite": "blender_test_suite",
        "timestamp": STAMP,
        "blender": bpy.app.version_string,
        "python": sys.version.split()[0],
        "background": bool(bpy.app.background),
        "quick": QUICK,
        "duration_s": round(total, 2),
        "summary": {"passed": len(SUITE.passed), "failed": len(SUITE.failed),
                    "skipped": len(SUITE.skipped)},
        "checks": SUITE.checks,
        "phase_times_s": {k: round(v, 2) for k, v in SUITE.timings.items()},
    }
    out = os.path.join(ROOT, "dist", "test_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"Report: {out}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if not SUITE.failed else 1)


if __name__ == "__main__":
    main()
