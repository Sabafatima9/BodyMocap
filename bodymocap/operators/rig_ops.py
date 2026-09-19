"""POSE_OT_create_test_rigs: standard 2-segment and 4-segment demo rigs."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty
from bpy.types import Operator


class POSE_OT_create_test_rigs(Operator):
    """Create Rig A (2-bone limbs, T-pose) and Rig B (4-bone limbs, A-pose)
    with mannequin meshes and set them as targets"""

    bl_idname = "pose.create_test_rigs"
    bl_label = "Create Test Rigs"
    bl_options = {"REGISTER", "UNDO"}

    with_mesh: BoolProperty(name="Mannequin Meshes", default=True)
    include_continuous: BoolProperty(name="Add Rig C (3-bone limbs)", default=False)

    def execute(self, context):
        from ..assets.build import build_test_rigs
        from ..runtime import get_runtime
        names = ["RigA_Standard", "RigB_Segmented"] + (["RigC_Continuous"] if self.include_continuous else [])
        rigs = build_test_rigs(with_mesh=self.with_mesh, names=names)
        s = context.scene.bodymocap
        s.target = rigs["RigA_Standard"]
        existing = {t.obj for t in s.extra_targets}
        for key, obj in rigs.items():
            if key != "RigA_Standard" and obj not in existing:
                t = s.extra_targets.add()
                t.obj = obj
        get_runtime().solvers.clear()
        self.report({"INFO"}, "Created " + ", ".join(rigs))
        return {"FINISHED"}


CLASSES = (POSE_OT_create_test_rigs,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
