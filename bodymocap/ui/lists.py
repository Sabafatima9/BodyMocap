"""UILists: topology mapping entries and extra targets."""

from __future__ import annotations

import bpy
from bpy.types import UIList


class BODYMOCAP_UL_profile(UIList):
    bl_idname = "BODYMOCAP_UL_profile"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        s = context.scene.bodymocap
        row = layout.row(align=True)
        split = row.split(factor=0.22, align=True)
        split.prop(item, "chain", text="")
        split2 = split.split(factor=0.3, align=True)
        split2.prop(item, "segment", text="")
        if s.target is not None and s.target.type == "ARMATURE":
            split2.prop_search(item, "bone", s.target.data, "bones", text="")
        else:
            split2.prop(item, "bone", text="")


class BODYMOCAP_UL_targets(UIList):
    bl_idname = "BODYMOCAP_UL_targets"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.prop(item, "enabled", text="")
        row.prop(item, "obj", text="", icon="ARMATURE_DATA")


CLASSES = (BODYMOCAP_UL_profile, BODYMOCAP_UL_targets)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        bpy.utils.unregister_class(c)
