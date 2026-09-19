"""Add-on preferences (bl_idname = package, works as legacy add-on and extension)."""

from __future__ import annotations

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import AddonPreferences

PACKAGE = __package__ or "bodymocap"


class BODYMOCAP_AP_preferences(AddonPreferences):
    bl_idname = PACKAGE

    model_directory: StringProperty(
        name="Model Folder", subtype="DIR_PATH", default="",
        description="Where pose models are stored (empty = add-on user data folder)")
    install_location: EnumProperty(
        name="Install Into",
        items=[("AUTO", "Automatic", "Blender's Python if writable, else a user folder"),
               ("USER", "User Folder", "Always install into the add-on's user folder")],
        default="AUTO")
    log_level: EnumProperty(
        name="Log Level",
        items=[("INFO", "Info", ""), ("DEBUG", "Debug", ""), ("WARNING", "Warning", "")],
        default="INFO")

    def draw(self, context):
        from .utils import deps
        layout = self.layout
        col = layout.column()
        col.prop(self, "model_directory")
        col.prop(self, "install_location")
        col.prop(self, "log_level")
        box = layout.box()
        box.label(text="Dependencies", icon="PREFERENCES")
        for name, info in deps.dependency_status().items():
            ok = info["available"] == "yes"
            box.label(text=f"{name}: {info['version'] if ok else 'missing'}",
                      icon="CHECKMARK" if ok else "ERROR")
        row = box.row()
        row.operator("pose.install_mocap_dependencies", icon="IMPORT")
        row.operator("pose.download_pose_model", icon="URL")


CLASSES = (BODYMOCAP_AP_preferences,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
