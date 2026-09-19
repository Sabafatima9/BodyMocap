"""Action / F-Curve helpers that work on Blender 4.2 LTS through 5.x.

Blender 4.4 introduced layered ("slotted") Actions and Blender 5.0 removed the
legacy ``Action.fcurves`` API, which the MVP relied on.  Everything here
detects the available API at runtime.
"""

from __future__ import annotations

from typing import Iterator, Optional

import bpy


def is_layered(action) -> bool:
    return hasattr(action, "layers") and not hasattr(action, "fcurves")


def new_action(name: str, overwrite: bool = True):
    """Create an Action.  With overwrite, an existing one is emptied and reused
    (keeps NLA strips / users pointing at it valid)."""
    act = bpy.data.actions.get(name)
    if act is not None and overwrite:
        clear_action(act)
        return act
    if act is not None:
        base, i = name, 1
        while f"{base}.{i:03d}" in bpy.data.actions:
            i += 1
        name = f"{base}.{i:03d}"
    act = bpy.data.actions.new(name)
    act.use_fake_user = True
    return act


def iter_fcurves(action) -> Iterator:
    if action is None:
        return
    if hasattr(action, "layers"):
        for layer in action.layers:
            for strip in layer.strips:
                for cb in getattr(strip, "channelbags", []):
                    yield from cb.fcurves
    if hasattr(action, "fcurves"):
        # 4.4-4.5 expose a legacy proxy onto the first slot: skip duplicates
        if not hasattr(action, "layers") or not len(action.layers):
            yield from action.fcurves


def clear_action(action) -> None:
    if hasattr(action, "layers"):
        for layer in list(action.layers):
            for strip in list(layer.strips):
                for cb in list(getattr(strip, "channelbags", [])):
                    fcs = cb.fcurves
                    if hasattr(fcs, "clear"):
                        fcs.clear()
                    else:
                        for fc in list(fcs):
                            fcs.remove(fc)
    if hasattr(action, "fcurves"):
        for fc in list(action.fcurves):
            action.fcurves.remove(fc)


def assign_action(obj, action) -> None:
    if obj.animation_data is None:
        obj.animation_data_create()
    ad = obj.animation_data
    ad.action = action
    # Blender 4.4+: make sure a slot is bound so keys land on this object
    if hasattr(ad, "action_slot") and ad.action_slot is None and hasattr(action, "slots"):
        slot = None
        for s in action.slots:
            if getattr(s, "target_id_type", "OBJECT") in ("OBJECT", "UNSPECIFIED"):
                slot = s
                break
        if slot is None:
            try:
                slot = action.slots.new(id_type="OBJECT", name=obj.name)
            except TypeError:
                slot = action.slots.new("OBJECT", obj.name)
        ad.action_slot = slot


def ensure_fcurve(action, obj, data_path: str, index: int, group: str):
    """F-Curve for (data_path, index) of ``obj`` in ``action`` (assigned first)."""
    if hasattr(action, "fcurve_ensure_for_datablock"):
        return action.fcurve_ensure_for_datablock(obj, data_path, index=index, group_name=group)
    fc = action.fcurves.find(data_path, index=index)
    if fc is None:
        fc = action.fcurves.new(data_path, index=index, action_group=group)
    return fc


def fill_fcurve(fc, frames, values, interpolation: str = "LINEAR") -> None:
    """Replace the keys of an F-Curve with (frame, value) pairs in bulk."""
    kps = fc.keyframe_points
    if len(kps):
        if hasattr(kps, "clear"):
            kps.clear()
        else:
            while len(kps):
                kps.remove(kps[0], fast=True)
    n = len(frames)
    if n == 0:
        return
    kps.add(n)
    co = [0.0] * (2 * n)
    co[0::2] = [float(f) for f in frames]
    co[1::2] = [float(v) for v in values]
    kps.foreach_set("co", co)
    if interpolation != "BEZIER":
        for kp in kps:
            kp.interpolation = interpolation
    else:
        for kp in kps:
            kp.handle_left_type = "AUTO_CLAMPED"
            kp.handle_right_type = "AUTO_CLAMPED"
    fc.update()


def keyframe_count(action) -> int:
    return sum(len(fc.keyframe_points) for fc in iter_fcurves(action))
