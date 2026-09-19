"""Topology detection: group an arbitrary humanoid armature into logical chains.

The detector is hierarchy-first: it locates the extremities (hips, head, hands,
feet), derives the spine as the path hips -> LCA(hands), and each limb as the
path from its attachment bone to its extremity.  Bone *names* are only used to
find extremities and to split limbs into anatomical segments; when names are
not informative the split falls back to rest-pose geometry, and when even that
is ambiguous the limb is solved in CONTINUOUS (arc-length) mode.

Output is a :class:`TopologyProfile` that can be edited in the UI and saved as
JSON ("topology mapping profile").
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.math3d import angle_between
from ..core.types import Vec3
from .rig import RigModel

PROFILE_TYPE = "bodymocap_topology_profile"
PROFILE_VERSION = 2

LIMB_NAMES = ("arm_L", "arm_R", "leg_L", "leg_R")
SEGMENTED = "SEGMENTED"
CONTINUOUS = "CONTINUOUS"


# ---------------------------------------------------------------------------
# Name analysis
# ---------------------------------------------------------------------------

_PREFIX_RE = re.compile(
    r"^(?:mixamorig\d*[:_]|def[-_.]|org[-_.]|bip\d*[ _]|cc_base_|"
    r"skeleton_|armature[._:]|rig[._:]|j_bip_c_|j_bip_|j_)",
    re.IGNORECASE,
)

_SIDE_PATTERNS: Sequence[Tuple[re.Pattern, int, int]] = (
    # (pattern, side group, base group(s) are the other groups)
    (re.compile(r"^(left|right)[\s._\-]*(.+)$", re.I), 1, 2),
    (re.compile(r"^(.+?)[\s._\-]*(left|right)((?:[\s._\-]*\d+)*)$", re.I), 2, 1),
    (re.compile(r"^(.+?)[._\-](l|r)((?:[._\-]\d+)*)$", re.I), 2, 1),
    (re.compile(r"^(l|r)[._\-](.+)$", re.I), 1, 2),
    (re.compile(r"^(.+?)[._\-]+(l|r)[._\-]+(.*)$", re.I), 2, 1),
)

_DAZ_SIDE = re.compile(r"^([lr])([A-Z].*)$")

_TAGS: Sequence[Tuple[str, re.Pattern]] = (
    ("ignore", re.compile(r"(?:^|[_.\- ])ik(?:$|[_.\- \d])|^ik|ik$|ctrl|control|pole|target|"
                          r"helper|^mch|socket|attach|weapon|_end$|\.end$|end$|nub$|null|dummy|"
                          r"marker|lookat|^prop", re.I)),
    ("face", re.compile(r"eye|jaw|lip|brow|cheek|nose|tongue|teeth|lid|chin|forehead|temple|"
                        r"mouth|face|^ear|\bear", re.I)),
    ("finger", re.compile(r"thumb|index|middle|ring|pinky|pinkie|finger|digit|metacarpal|"
                          r"^f_|palm|carpal(?!.*hand)", re.I)),
    ("twist", re.compile(r"twist|roll", re.I)),
    ("breast", re.compile(r"breast|chest_?[lr]$|pec", re.I)),
    ("clavicle", re.compile(r"clavicle|collar|shoulder", re.I)),
    ("forearm", re.compile(r"fore_?arm|lower_?arm|lo_?arm|elbow|ulna|radius", re.I)),
    ("upperarm", re.compile(r"upper_?arm|up_?arm|humerus|shldr", re.I)),
    ("arm", re.compile(r"arm", re.I)),
    ("hand", re.compile(r"hand|wrist", re.I)),
    ("thigh", re.compile(r"thigh|up_?leg|upper_?leg|femur", re.I)),
    ("shin", re.compile(r"shin|calf|lower_?leg|lo_?leg|knee|tibia|crus", re.I)),
    ("leg", re.compile(r"leg", re.I)),
    ("toe", re.compile(r"toe|ball|phalan", re.I)),
    ("heel", re.compile(r"heel", re.I)),
    ("foot", re.compile(r"foot|ankle|tarsal", re.I)),
    ("hips", re.compile(r"^hips?$|pelvis|^cog$|^hip_?$|centre_?of|center_?of|^root_?hips", re.I)),
    ("spine", re.compile(r"spine|chest|torso|abdomen|belly|waist|rib|lumbar|thorax|back", re.I)),
    ("neck", re.compile(r"neck", re.I)),
    ("head", re.compile(r"head|skull|cranium", re.I)),
    ("root", re.compile(r"^root$|^armature$|^master$|^world$|^origin$", re.I)),
)


def split_side(name: str) -> Tuple[str, str]:
    """Return (base_name, side) with side in {'L', 'R', ''}."""
    raw = _PREFIX_RE.sub("", name.strip())
    m = _DAZ_SIDE.match(raw)
    if m and len(raw) > 3:
        return m.group(2), m.group(1).upper()
    for pat, sg, bg in _SIDE_PATTERNS:
        m = pat.match(raw)
        if not m:
            continue
        side_tok = m.group(sg).lower()
        side = "L" if side_tok in ("l", "left") else "R"
        base = m.group(bg)
        # keep trailing parts (e.g. "arm_joint_L__4_" -> base "arm_joint" + "4")
        extras = [m.group(i) for i in range(1, (m.lastindex or 0) + 1) if i not in (sg, bg)]
        base = base + "".join(e for e in extras if e)
        return base, side
    return raw, ""


def name_tags(name: str) -> List[str]:
    base, _ = split_side(name)
    norm = base.replace(" ", "_")
    tags = []
    for tag, pat in _TAGS:
        if pat.search(norm):
            tags.append(tag)
    return tags


@dataclass
class _BoneInfo:
    name: str
    base: str
    side: str
    tags: List[str]

    def has(self, *tags: str) -> bool:
        return any(t in self.tags for t in tags)


# ---------------------------------------------------------------------------
# Profile data
# ---------------------------------------------------------------------------

@dataclass
class LimbChain:
    name: str
    kind: str                     # "arm" | "leg"
    side: str                     # "L" | "R"
    root: List[str] = field(default_factory=list)   # clavicle / hip-side bones (kept at rest)
    upper: List[str] = field(default_factory=list)
    lower: List[str] = field(default_factory=list)
    end: Optional[str] = None     # hand / foot
    extra: List[str] = field(default_factory=list)  # toes etc. (kept at rest)
    mode: str = SEGMENTED
    method: str = ""

    @property
    def bones(self) -> List[str]:
        return list(self.upper) + list(self.lower)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "side": self.side, "root": list(self.root),
            "upper": list(self.upper), "lower": list(self.lower), "end": self.end,
            "extra": list(self.extra), "mode": self.mode, "method": self.method,
        }

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "LimbChain":
        return cls(
            name=name, kind=d.get("kind", "arm" if name.startswith("arm") else "leg"),
            side=d.get("side", name[-1]), root=list(d.get("root", [])),
            upper=list(d.get("upper", [])), lower=list(d.get("lower", [])),
            end=d.get("end"), extra=list(d.get("extra", [])),
            mode=d.get("mode", SEGMENTED), method=d.get("method", "preset"),
        )


@dataclass
class TopologyProfile:
    name: str = ""
    hips: Optional[str] = None
    spine: List[str] = field(default_factory=list)   # hips (excl.) -> chest (incl.)
    neck: List[str] = field(default_factory=list)
    head: Optional[str] = None
    limbs: Dict[str, LimbChain] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def chest(self) -> Optional[str]:
        return self.spine[-1] if self.spine else self.hips

    def mapped_bones(self) -> List[str]:
        out: List[str] = []
        if self.hips:
            out.append(self.hips)
        out += self.spine + self.neck
        if self.head:
            out.append(self.head)
        for lc in self.limbs.values():
            out += lc.upper + lc.lower
            if lc.end:
                out.append(lc.end)
        return out

    def chain_counts(self) -> Dict[str, int]:
        d = {"spine": len(self.spine), "neck": len(self.neck)}
        for k, lc in self.limbs.items():
            d[k] = len(lc.upper) + len(lc.lower)
        return d

    def summary(self) -> str:
        parts = [f"hips={self.hips}", f"spine={len(self.spine)}", f"neck={len(self.neck)}",
                 f"head={self.head}"]
        for k in LIMB_NAMES:
            lc = self.limbs.get(k)
            if lc:
                parts.append(f"{k}={len(lc.upper)}+{len(lc.lower)}({lc.mode[0]})")
        return " ".join(parts)

    def validate(self, rig: RigModel) -> List[str]:
        errs: List[str] = []
        for n in self.mapped_bones():
            if n not in rig.bones:
                errs.append(f"Bone '{n}' not in armature")
        seen: Dict[str, int] = {}
        for n in self.mapped_bones():
            seen[n] = seen.get(n, 0) + 1
        errs += [f"Bone '{n}' mapped {c} times" for n, c in seen.items() if c > 1]
        if not self.hips:
            errs.append("No hips/root bone")
        # every mapped chain must be a parent->child sequence
        def check_seq(label: str, seq: List[str]):
            for a, b in zip(seq, seq[1:]):
                if a in rig.bones and b in rig.bones and not rig.is_ancestor(a, b):
                    errs.append(f"{label}: '{a}' is not an ancestor of '{b}'")
        check_seq("spine", ([self.hips] if self.hips else []) + self.spine)
        for k, lc in self.limbs.items():
            seq = lc.upper + lc.lower + ([lc.end] if lc.end else [])
            check_seq(k, seq)
            if not lc.upper or (lc.mode == SEGMENTED and not lc.lower):
                errs.append(f"{k}: needs upper and lower bones")
        return errs

    def to_dict(self) -> dict:
        return {
            "type": PROFILE_TYPE, "version": PROFILE_VERSION, "name": self.name,
            "hips": self.hips, "spine": list(self.spine), "neck": list(self.neck),
            "head": self.head,
            "limbs": {k: v.to_dict() for k, v in self.limbs.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TopologyProfile":
        if d.get("type") not in (None, PROFILE_TYPE):
            raise ValueError(f"Unknown profile type: {d.get('type')}")
        p = cls(name=d.get("name", ""), hips=d.get("hips"), spine=list(d.get("spine", [])),
                neck=list(d.get("neck", [])), head=d.get("head"))
        for k, v in d.get("limbs", {}).items():
            p.limbs[k] = LimbChain.from_dict(k, v)
        return p

    def save(self, path: str) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str) -> "TopologyProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class _Detector:
    def __init__(self, rig: RigModel):
        self.rig = rig
        self.info: Dict[str, _BoneInfo] = {}
        for n in rig.order:
            base, side = split_side(n)
            self.info[n] = _BoneInfo(n, base, side, name_tags(n))
        self.use_deform_filter = any(not b.use_deform for b in rig.bones.values()) and \
            any(b.use_deform for b in rig.bones.values())
        self.warnings: List[str] = []
        self.methods: Dict[str, str] = {}
        self.left_axis = Vec3(1.0, 0.0, 0.0)

    # -- helpers -------------------------------------------------------------
    def _eligible(self, n: str) -> bool:
        i = self.info[n]
        if i.has("ignore", "face", "finger"):
            return False
        if self.use_deform_filter and not self.rig.bones[n].use_deform:
            return False
        return True

    def _pick(self, cands: List[str]) -> Optional[str]:
        if not cands:
            return None
        return min(cands, key=lambda n: (self.rig.depth(n), n))

    def _joint(self, n: str) -> Vec3:
        return self.rig.bones[n].head

    # -- extremities ---------------------------------------------------------
    def find_extremities(self):
        info, rig = self.info, self.rig
        hands, feet = {}, {}
        for side in ("L", "R"):
            c = [n for n in rig.order if self._eligible(n) and info[n].side == side
                 and info[n].has("hand") and not info[n].has("twist")]
            hands[side] = self._pick(c)
            if hands[side]:
                self.methods[f"hand_{side}"] = "name"
            c = [n for n in rig.order if self._eligible(n) and info[n].side == side
                 and info[n].has("foot") and not info[n].has("toe", "heel", "twist")]
            feet[side] = self._pick(c)
            if feet[side]:
                self.methods[f"foot_{side}"] = "name"
        head = self._pick([n for n in rig.order if self._eligible(n) and not info[n].side
                           and info[n].has("head")])
        if head:
            self.methods["head"] = "name"

        # Fallbacks: deepest bone of a generically named arm / leg chain.
        for side in ("L", "R"):
            if hands[side] is None:
                c = [n for n in rig.order if self._eligible(n) and info[n].side == side
                     and info[n].has("arm", "upperarm", "forearm")]
                deepest = self._deepest_on_path(c)
                if deepest:
                    hands[side] = deepest
                    self.methods[f"hand_{side}"] = "deepest-arm"
            if feet[side] is None:
                c = [n for n in rig.order if self._eligible(n) and info[n].side == side
                     and info[n].has("leg", "thigh", "shin", "toe")]
                deepest = self._deepest_on_path(c)
                if deepest:
                    # a >=4 bone leg path ends in a toe: foot is its parent
                    anc_legs = [a for a in rig.ancestors(deepest) if a in c]
                    if len(anc_legs) >= 3 or info[deepest].has("toe"):
                        feet[side] = rig.bones[deepest].parent
                    else:
                        feet[side] = deepest
                    self.methods[f"foot_{side}"] = "deepest-leg"
        if head is None:
            c = [n for n in rig.order if self._eligible(n) and not info[n].side
                 and info[n].has("neck", "head")]
            head = self._deepest_on_path(c)
            if head:
                self.methods["head"] = "deepest-neck"

        hips = None
        if feet["L"] and feet["R"]:
            hips = rig.lca(feet["L"], feet["R"])
            self.methods["hips"] = "lca(feet)"
        if hips is None:
            hips = self._pick([n for n in rig.order if self._eligible(n) and info[n].has("hips")])
            if hips:
                self.methods["hips"] = "name"
        if hips is None and rig.roots:
            hips = rig.roots[0]
            self.methods["hips"] = "root"
        return hips, head, hands, feet

    def _chain_len(self, n: str) -> int:
        kids = [c for c in self.rig.bones[n].children if self._eligible(c)]
        return 1 + (max(self._chain_len(c) for c in kids) if kids else 0)

    def _head_from_chest(self, chest: str, hands: List[str]) -> Optional[str]:
        """Follow the longest central (non-arm) eligible chain above the chest."""
        rig = self.rig
        arm_roots = set()
        for h in hands:
            arm_roots.update([h] + rig.ancestors(h))
        kids = [c for c in rig.bones[chest].children
                if self._eligible(c) and c not in arm_roots and not self.info[c].side]
        if not kids:
            return None
        cx = rig.bones[chest].head
        n = max(kids, key=lambda c: (self._chain_len(c),
                                     -abs((rig.bones[c].head - cx).dot(self.left_axis))))
        while True:
            nxt = [c for c in rig.bones[n].children if self._eligible(c) and not self.info[c].side]
            if not nxt:
                return n
            n = max(nxt, key=lambda c: self._chain_len(c))

    def _deepest_on_path(self, cands: List[str]) -> Optional[str]:
        if not cands:
            return None
        s = set(cands)
        # the deepest candidate whose ancestors include most of the others
        best = max(cands, key=lambda n: (sum(1 for a in self.rig.ancestors(n) if a in s),
                                         self.rig.depth(n)))
        return best

    # -- limb segmentation ---------------------------------------------------
    def _limb_labels(self, bones: List[str], kind: str) -> List[str]:
        labels = []
        seen_upper = False
        for n in bones:
            i = self.info[n]
            if kind == "arm":
                if i.has("clavicle") and not i.has("upperarm", "forearm") and not seen_upper:
                    lab = "root"
                elif i.has("forearm"):
                    lab = "lower"
                elif i.has("upperarm"):
                    lab = "upper"
                elif i.has("twist"):
                    lab = "twist"
                elif i.has("arm"):
                    lab = "upper" if not seen_upper else "generic"
                else:
                    lab = "unknown"
            else:
                if i.has("hips") and not seen_upper:
                    lab = "root"
                elif i.has("shin"):
                    lab = "lower"
                elif i.has("thigh"):
                    lab = "upper"
                elif i.has("twist"):
                    lab = "twist"
                elif i.has("leg"):
                    lab = "lower" if seen_upper else "generic"
                else:
                    lab = "unknown"
            if lab == "upper":
                seen_upper = True
            labels.append(lab)
        return labels

    def segment_limb(self, name: str, kind: str, side: str, path: List[str], end: str) -> LimbChain:
        rig = self.rig
        lc = LimbChain(name=name, kind=kind, side=side, end=end)
        body = path[:-1] if path and path[-1] == end else list(path)
        labels = self._limb_labels(body, kind)

        # Leading root bones (clavicles / hip-side) by name or medial geometry.
        n_root = 0
        while n_root < len(body) and labels[n_root] == "root":
            n_root += 1
        if n_root == 0 and len(body) >= 3 and kind == "arm":
            # geometric clavicle: head near the midline compared to the reach
            attach = rig.bones[body[0]].parent
            wrist = self._joint(end)
            if attach is not None:
                mid = self._joint(attach)
                reach = (wrist - mid).length()
                lat = abs((self._joint(body[0]) - mid).dot(self.left_axis))
                if reach > 1e-6 and lat < 0.1 * reach and labels[0] not in ("upper", "lower"):
                    n_root = 1
        lc.root = body[:n_root]
        rest = body[n_root:]
        labs = labels[n_root:]

        # Name-based split: upper+ (twist) lower+ (twist)
        if rest and "lower" in labs and labs[0] in ("upper", "generic"):
            first_lower = labs.index("lower")
            if all(l in ("upper", "twist", "generic") for l in labs[:first_lower]) and \
               all(l in ("lower", "twist", "generic") for l in labs[first_lower:]) and first_lower > 0:
                lc.upper = rest[:first_lower]
                lc.lower = rest[first_lower:]
                lc.mode = SEGMENTED
                lc.method = "names"
                return lc

        # Geometric split
        if len(rest) == 1:
            lc.upper = list(rest)
            lc.mode = CONTINUOUS
            lc.method = "single-bone"
            self.warnings.append(f"{name}: single bone limb, solved as continuous chain")
            return lc
        if len(rest) == 2:
            lc.upper, lc.lower = [rest[0]], [rest[1]]
            lc.mode = SEGMENTED
            lc.method = "geometry(2 bones)"
            return lc
        joints = [self._joint(n) for n in rest] + [self._joint(end)]
        seg = [(joints[i + 1] - joints[i]).length() for i in range(len(rest))]
        total = sum(seg) or 1.0
        # 1) a clear rest bend at an interior joint
        best_i, best_bend = -1, 0.0
        for i in range(1, len(rest)):
            a = joints[i] - joints[i - 1]
            b = joints[i + 1] - joints[i]
            bend = angle_between(a, b)
            if bend > best_bend:
                best_i, best_bend = i, bend
        if best_bend > math.radians(10.0):
            k = best_i
            method = "geometry(bend)"
        else:
            # 2) interior joint nearest to mid arc length
            acc, k, best = 0.0, -1, 1e9
            for i in range(1, len(rest)):
                acc += seg[i - 1]
                d = abs(acc / total - 0.5)
                if d < best:
                    best, k = d, i
            method = "geometry(arc-mid)"
            if best > 0.12:
                lc.upper = list(rest)
                lc.lower = []
                lc.mode = CONTINUOUS
                lc.method = "arc-length"
                self.warnings.append(
                    f"{name}: no anatomical joint found in {len(rest)} bones; "
                    "using continuous arc-length mapping")
                return lc
        lc.upper, lc.lower = rest[:k], rest[k:]
        lc.mode = SEGMENTED
        lc.method = method
        return lc

    # -- main ----------------------------------------------------------------
    def run(self, name: str = "") -> TopologyProfile:
        rig = self.rig
        prof = TopologyProfile(name=name or rig.name)
        hips, head, hands, feet = self.find_extremities()
        prof.hips = hips
        if hips is None:
            prof.warnings.append("Could not identify hips/root bone")
            return prof
        for a, b in ((hands.get("L"), hands.get("R")), (feet.get("L"), feet.get("R"))):
            if a and b:
                v = self._joint(a) - self._joint(b)
                if v.length() > 1e-6:
                    self.left_axis = v.normalized()
                    break

        chest = None
        if hands["L"] and hands["R"]:
            chest = rig.lca(hands["L"], hands["R"])
        elif head:
            chest = rig.bones[head].parent
        if chest is not None and chest != hips and not rig.is_ancestor(hips, chest):
            prof.warnings.append("Arms do not branch from the spine below the hips")
            chest = None
        if chest is not None and chest != hips:
            prof.spine = rig.path(hips, chest) or []
        spine_top = chest or hips

        if head is None or not (rig.is_ancestor(hips, head)):
            follow = self._head_from_chest(spine_top, [h for h in hands.values() if h])
            if follow:
                head = follow
                self.methods["head"] = "chest-chain"

        if head:
            anchor = spine_top if (rig.is_ancestor(spine_top, head)) else rig.lca(spine_top, head)
            path = rig.path(anchor, head) if anchor else None
            if path:
                prof.head = head
                prof.neck = [n for n in path[:-1] if self._eligible(n)]
        for side in ("L", "R"):
            hand = hands[side]
            if hand and rig.is_ancestor(spine_top, hand):
                path = rig.path(spine_top, hand) or []
                path = [n for n in path if self._eligible(n) or n == hand]
                if len(path) >= 2:
                    prof.limbs[f"arm_{side}"] = self.segment_limb(f"arm_{side}", "arm", side, path, hand)
            elif hand:
                prof.warnings.append(f"hand_{side} '{hand}' is not under the chest")
            foot = feet[side]
            if foot and rig.is_ancestor(hips, foot):
                path = rig.path(hips, foot) or []
                path = [n for n in path if self._eligible(n) or n == foot]
                if len(path) >= 2:
                    lc = self.segment_limb(f"leg_{side}", "leg", side, path, foot)
                    # toes: follow the main child chain below the foot
                    kids = [c for c in rig.bones[foot].children
                            if self._eligible(c) and not self.info[c].has("heel")]
                    while kids:
                        k = max(kids, key=lambda c: (self.info[c].has("toe"), len(rig.descendants(c))))
                        lc.extra.append(k)
                        kids = [c for c in rig.bones[k].children if self._eligible(c)]
                    prof.limbs[f"leg_{side}"] = lc
        for side in ("L", "R"):
            if f"arm_{side}" not in prof.limbs:
                prof.warnings.append(f"arm_{side} not detected")
            if f"leg_{side}" not in prof.limbs:
                prof.warnings.append(f"leg_{side} not detected")
        if not prof.head:
            prof.warnings.append("head not detected")
        prof.warnings += self.warnings
        return prof


def detect_topology(rig: RigModel, name: str = "") -> TopologyProfile:
    """Auto-detect a topology profile for ``rig``."""
    return _Detector(rig).run(name)
