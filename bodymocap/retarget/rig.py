"""Pure-Python armature model: rest data, FK and local<->armature conversions.

The model mirrors Blender's pose evaluation for rotation-driven rigs::

    pose(b) = pose(parent) @ rest(parent)^-1 @ rest(b) @ basis(b)

Internally the solver works with *armature-space delta rotations* ``D`` so that
a bone's posed orientation is ``D @ rest``.  Undriven bones inherit their
parent's delta (identity local rotation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..core.math3d import quat_conjugate, quat_identity, quat_normalize, vec_roll_to_quat
from ..core.types import Quat, Vec3


@dataclass
class RigBone:
    name: str
    parent: Optional[str]
    head: Vec3
    tail: Vec3
    rest: Quat
    use_connect: bool = False
    inherit_rotation: bool = True
    use_deform: bool = True
    children: List[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return (self.tail - self.head).length()

    @property
    def y_axis(self) -> Vec3:
        return self.rest.rotate(Vec3(0.0, 1.0, 0.0))


class PoseState:
    """Armature-space pose (per-bone delta rotation and head position)."""

    def __init__(self, rig: "RigModel"):
        self.rig = rig
        self.delta: Dict[str, Quat] = {}
        self.head: Dict[str, Vec3] = {}

    def orientation(self, name: str) -> Quat:
        return self.delta[name] @ self.rig.bones[name].rest

    def tail(self, name: str) -> Vec3:
        b = self.rig.bones[name]
        return self.head[name] + self.delta[name].rotate(b.tail - b.head)

    def point(self, name: str, rest_point: Vec3) -> Vec3:
        """Where a rest-space point rigidly attached to bone ``name`` ends up."""
        b = self.rig.bones[name]
        return self.head[name] + self.delta[name].rotate(rest_point - b.head)


class RigModel:
    def __init__(self, bones: Iterable[RigBone], name: str = ""):
        self.name = name
        self.bones: Dict[str, RigBone] = {}
        for b in bones:
            b.children = []
            self.bones[b.name] = b
        for b in self.bones.values():
            if b.parent is not None and b.parent not in self.bones:
                b.parent = None
        for b in self.bones.values():
            if b.parent is not None:
                self.bones[b.parent].children.append(b.name)
        self.roots: List[str] = [n for n, b in self.bones.items() if b.parent is None]
        self.order: List[str] = []
        stack = list(reversed(self.roots))
        while stack:
            n = stack.pop()
            self.order.append(n)
            stack.extend(reversed(self.bones[n].children))
        self._depth: Dict[str, int] = {}
        for n in self.order:
            p = self.bones[n].parent
            self._depth[n] = 0 if p is None else self._depth[p] + 1

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_spec(cls, spec: Sequence[dict], name: str = "") -> "RigModel":
        """Build from dicts: name, parent, head, tail, [roll], [connect].

        Rest orientations follow Blender's edit-bone convention (vector + roll),
        so the same spec can be used to build a real armature.
        """
        bones = []
        for s in spec:
            head = Vec3.from_seq(s["head"])
            tail = Vec3.from_seq(s["tail"])
            rest = vec_roll_to_quat(tail - head, float(s.get("roll", 0.0)))
            bones.append(RigBone(
                name=s["name"], parent=s.get("parent"), head=head, tail=tail, rest=rest,
                use_connect=bool(s.get("connect", False)),
                inherit_rotation=bool(s.get("inherit_rotation", True)),
                use_deform=bool(s.get("deform", True)),
            ))
        return cls(bones, name=name)

    @classmethod
    def from_blender(cls, arm_obj, deform_only: bool = False) -> "RigModel":
        bones = []
        for b in arm_obj.data.bones:
            if deform_only and not b.use_deform:
                continue
            q = b.matrix_local.to_quaternion()
            bones.append(RigBone(
                name=b.name,
                parent=b.parent.name if b.parent else None,
                head=Vec3(*b.head_local),
                tail=Vec3(*b.tail_local),
                rest=quat_normalize(Quat(q.w, q.x, q.y, q.z)),
                use_connect=bool(b.use_connect),
                inherit_rotation=bool(b.use_inherit_rotation),
                use_deform=bool(b.use_deform),
            ))
        return cls(bones, name=arm_obj.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bones": [
                {
                    "name": b.name, "parent": b.parent,
                    "head": list(b.head.as_tuple()), "tail": list(b.tail.as_tuple()),
                    "rest": list(b.rest.as_tuple()), "connect": b.use_connect,
                    "inherit_rotation": b.inherit_rotation, "deform": b.use_deform,
                }
                for b in (self.bones[n] for n in self.order)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RigModel":
        bones = [
            RigBone(
                name=b["name"], parent=b.get("parent"),
                head=Vec3.from_seq(b["head"]), tail=Vec3.from_seq(b["tail"]),
                rest=Quat.from_seq(b["rest"]), use_connect=bool(b.get("connect", False)),
                inherit_rotation=bool(b.get("inherit_rotation", True)),
                use_deform=bool(b.get("deform", True)),
            )
            for b in d["bones"]
        ]
        return cls(bones, name=d.get("name", ""))

    # ------------------------------------------------------------------
    # Hierarchy helpers
    # ------------------------------------------------------------------
    def depth(self, name: str) -> int:
        return self._depth[name]

    def ancestors(self, name: str) -> List[str]:
        """Parent chain from the bone's parent up to the root."""
        out = []
        p = self.bones[name].parent
        while p is not None:
            out.append(p)
            p = self.bones[p].parent
        return out

    def is_ancestor(self, anc: str, name: str) -> bool:
        return anc in self.ancestors(name)

    def path(self, start: str, end: str) -> Optional[List[str]]:
        """Bones from ``start`` (exclusive) down to ``end`` (inclusive)."""
        chain = []
        n: Optional[str] = end
        while n is not None and n != start:
            chain.append(n)
            n = self.bones[n].parent
        if n != start:
            return None
        chain.reverse()
        return chain

    def lca(self, a: str, b: str) -> Optional[str]:
        anc_a = [a] + self.ancestors(a)
        set_b = set([b] + self.ancestors(b))
        for n in anc_a:
            if n in set_b:
                return n
        return None

    def descendants(self, name: str) -> List[str]:
        out = []
        stack = list(self.bones[name].children)
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(self.bones[n].children)
        return out

    # ------------------------------------------------------------------
    # Local <-> delta conversions
    # ------------------------------------------------------------------
    def local_from_delta(self, name: str, delta: Quat, parent_delta: Optional[Quat]) -> Quat:
        b = self.bones[name]
        rest_c = quat_conjugate(b.rest)
        if b.parent is None or parent_delta is None or not b.inherit_rotation:
            rel = delta
        else:
            rel = quat_conjugate(parent_delta) @ delta
        return quat_normalize(rest_c @ rel @ b.rest)

    def delta_from_local(self, name: str, local: Quat, parent_delta: Optional[Quat]) -> Quat:
        b = self.bones[name]
        rel = b.rest @ local @ quat_conjugate(b.rest)
        if b.parent is None or parent_delta is None or not b.inherit_rotation:
            return quat_normalize(rel)
        return quat_normalize(parent_delta @ rel)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def fk(
        self,
        local_rot: Optional[Dict[str, Quat]] = None,
        local_loc: Optional[Dict[str, Vec3]] = None,
    ) -> PoseState:
        local_rot = local_rot or {}
        local_loc = local_loc or {}
        st = PoseState(self)
        ident = quat_identity()
        for n in self.order:
            b = self.bones[n]
            p = b.parent
            pd = st.delta[p] if p is not None else None
            q = local_rot.get(n, ident)
            d = self.delta_from_local(n, q, pd)
            st.delta[n] = d
            if p is None:
                head = b.head.copy()
                ref = b.rest
            else:
                pb = self.bones[p]
                head = st.head[p] + pd.rotate(b.head - pb.head)
                ref = (pd @ b.rest) if b.inherit_rotation else b.rest
            t = local_loc.get(n)
            if t is not None and not b.use_connect:
                head = head + ref.rotate(t)
            st.head[n] = head
        return st

    def rest_pose(self) -> PoseState:
        return self.fk()


def bone_spec(
    name: str,
    head: Tuple[float, float, float],
    tail: Tuple[float, float, float],
    parent: Optional[str] = None,
    roll: float = 0.0,
    connect: bool = False,
) -> dict:
    return {"name": name, "head": head, "tail": tail, "parent": parent,
            "roll": roll, "connect": connect}
