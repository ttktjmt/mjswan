"""A first draft of a skin's maps, for its author to check; never used by a build.

The skin's side is read from the skeleton's shape, its names only breaking ties: the
hips are the joint that two mirrored chains reaching the floor hang from, the spine its
chain toward the head, and the arms the mirrored pair reaching farthest to the side on
the way (on a four-legged animal, the front legs). The robot's side is read from its
bodies' names. ``rest_pose`` is not guessed: a robot's link origins sit off the joints
they look like, so a solved T-pose comes out degrees wrong. The robots checked by hand
are known instead.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, TypeVar

import mujoco
import numpy as np

from ..document.skin import SkinFile
from .humanoid import VRM_PARENTS
from .pose import (
    limb_root,
    pose_at_rest,
    resolve_body,
    resolve_joint,
    standing_height,
)

T = TypeVar("T")

#: Below this share of its height above the lowest joint, a chain reaches the floor.
_FLOOR = 0.15
#: Two nodes this close, as a share of the height, are level or in line.
_NEAR = 0.05
#: A leading segment shorter than this share of its limb is a stub (a shoulder).
_STUB = 0.25


@dataclass(frozen=True)
class SkinGuess:
    """What the skeleton's shape says, in the file's own nodes."""

    kind: Literal["humanoid", "quadruped", "other"]
    humanoid: dict[str, int] = field(default_factory=dict)
    """VRM humanoid bone -> node."""
    facing: str = "+z"
    springs: list[int] = field(default_factory=list)
    """The first node of each chain no bone takes: a tail or hair, to swing."""
    limbs: list[tuple[int, int, int]] = field(default_factory=list)
    """For any other skeleton: chains of three nodes down to the floor."""
    why: dict[str, str] = field(default_factory=dict)
    """Bone -> how it was found, where that is worth saying."""


class _Skeleton:
    """A file's joints at rest, in its own frame and units, y up."""

    def __init__(self, file: SkinFile):
        self.file = file
        self.joints = set(file.joints) or set(range(len(file.names)))
        self.pos = file.rest
        self.kids: dict[int, list[int]] = {i: [] for i in range(len(file.names))}
        for child, parent in enumerate(file.parents):
            if parent is not None and child in self.joints:
                self.kids[parent].append(child)
        heights = [self.pos[j][1] for j in self.joints]
        self.low = min(heights)
        self.height = (max(heights) - self.low) or 1.0
        self._subtrees: dict[int, list[int]] = {}

    def subtree(self, i: int) -> list[int]:
        if i not in self._subtrees:
            self._subtrees[i] = [i] + [n for c in self.kids[i] for n in self.subtree(c)]
        return self._subtrees[i]

    def chain(self, i: int) -> list[int]:
        """*i* and its main line, down to a fork (the fingers) or an end. Where one
        child carries twice the nodes of the next, the rest are twist or cloth bones
        beside the line, and are stepped past."""
        out = [i]
        while kids := self.kids[out[-1]]:
            sizes = sorted((len(self.subtree(c)) for c in kids), reverse=True)
            if len(sizes) > 1 and sizes[0] < 2 * sizes[1]:
                break
            out.append(max(kids, key=lambda c: len(self.subtree(c))))
        return out

    def reaches_floor(self, i: int) -> bool:
        lowest = min(self.pos[j][1] for j in self.subtree(i))
        return lowest < self.low + _FLOOR * self.height

    def mirrored(
        self, a: int, b: int, centre: int, axes: tuple[int, ...] = (0, 2)
    ) -> int | None:
        """The lateral axis (0 for x, 2 for z) of *axes* about which *a* and *b* mirror
        each other across *centre*, each well to its own side; None if they do not."""
        da, db = self.pos[a] - self.pos[centre], self.pos[b] - self.pos[centre]
        near = _NEAR * self.height
        if abs(da[1] - db[1]) > near:
            return None
        for axis in axes:
            other = 2 - axis
            apart = abs(da[other] - db[other])
            if (
                abs(da[axis] + db[axis]) < near
                and apart < near
                and abs(da[axis]) > max(0.005 * self.height, apart)
            ):
                return axis
        return None

    def pairs(
        self, node: int, axes: tuple[int, ...] = (0, 2)
    ) -> list[tuple[int, int, int]]:
        """Mirrored pairs among *node*'s children, with their lateral axis."""
        kids = self.kids[node]
        return [
            (a, b, axis)
            for k, a in enumerate(kids)
            for b in kids[k + 1 :]
            if (axis := self.mirrored(a, b, node, axes)) is not None
        ]


def guess_humanoid(file: SkinFile) -> SkinGuess:
    """The humanoid of a ``.glb``, its front legs as arms if it walks on four."""
    sk = _Skeleton(file)
    by_size = sorted(sk.joints, key=lambda j: len(sk.subtree(j)), reverse=True)
    hips = next(
        (j for j in by_size if any(sk.reaches_floor(a) for a, _, _ in sk.pairs(j))),
        None,
    )
    if hips is None:
        return SkinGuess(kind="other", limbs=_floor_chains(sk))
    legs = [(a, b, axis) for a, b, axis in sk.pairs(hips) if sk.reaches_floor(a)]
    lateral = legs[0][2]
    paired = {n for a, b, _ in legs for n in (a, b)}
    rest = [c for c in sk.kids[hips] if c not in paired]
    if not rest:
        return SkinGuess(kind="other", limbs=_floor_chains(sk))
    up_child = max(rest, key=lambda c: max(sk.pos[j][1] for j in sk.subtree(c)))
    forward = _forward(sk, hips, up_child, legs, lateral)
    left = np.cross([0.0, 1.0, 0.0], forward)

    bones = {"hips": hips}
    why = {"hips": "two mirrored chains reach the floor"}

    def ahead(pair: tuple[int, int, int]) -> float:
        return float((sk.pos[pair[0]] - sk.pos[hips]) @ forward)

    rear = min(legs, key=ahead)
    front = max(legs, key=ahead) if len(legs) > 1 else None
    for leg in rear[:2]:
        side = _side(sk, leg, hips, left)
        found = _limb(sk, leg, ("UpperLeg", "LowerLeg", "Foot", "Toes"))
        bones.update({f"{side}{part}": n for part, n in found.items()})

    # Up the spine toward the head, noting each mirrored fork on the way.
    path, forks, node = [], [], up_child
    while True:
        path.append(node)
        here = sk.pairs(node, (lateral,))
        forks += [(node, (a, b)) for a, b, _ in here]
        onward = [c for c in sk.kids[node] if all(c not in (a, b) for a, b, _ in here)]
        if not onward:
            break
        node = max(
            onward,
            key=lambda c: max(
                sk.pos[j][1] + sk.pos[j] @ forward for j in sk.subtree(c)
            ),
        )
    fork = None
    if front is None and forks:
        # The arms reach farthest to the side; a collar or a bust does not.
        fork, arms = max(
            forks,
            key=lambda f: max(
                abs((sk.pos[j] - sk.pos[hips]) @ left) for j in sk.subtree(f[1][0])
            ),
        )
        front = (*arms, lateral)
    if fork is not None:
        torso, above = path[: path.index(fork) + 1], path[path.index(fork) + 1 :]
        why[_TORSO[min(len(torso), 3) - 1]] = "the arms fork here"
    else:
        torso, above = path[:-2] or path[:1], path[-2:] if len(path) > 2 else path[1:]
    bones.update(zip(_TORSO, torso))
    if len(above) >= 2:
        bones["neck"], bones["head"] = above[0], above[1]
    elif above:
        bones["head"] = above[0]

    quadruped = front is not None and sk.reaches_floor(front[0])
    if front is not None:
        for arm in front[:2]:
            side = _side(sk, arm, hips, left)
            stub = None if quadruped else "Shoulder"
            found = _limb(sk, arm, ("UpperArm", "LowerArm", "Hand"), stub=stub)
            bones.update({f"{side}{part}": n for part, n in found.items()})
        if quadruped:
            why["leftUpperArm"] = "front legs as arms"
    taken = set(bones.values())
    springs = [
        c
        for j in taken
        for c in sk.kids[j]
        if c not in taken and len(sk.chain(c)) >= 2 and not set(sk.subtree(c)) & taken
    ]
    return SkinGuess(
        kind="quadruped" if quadruped else "humanoid",
        humanoid=in_vrm_order(bones),
        facing=_facing(forward),
        springs=sorted(springs),
        why=why,
    )


_TORSO = ("spine", "chest", "upperChest")


def _forward(
    sk: _Skeleton, hips: int, up: int, legs: list[tuple[int, int, int]], lateral: int
) -> np.ndarray:
    """Which way the skeleton faces, along the floor across its lateral axis: toward
    its head where that reaches out ahead, else the way its toes point."""
    along = 2 if lateral == 0 else 0
    tip = max(sk.subtree(up), key=lambda j: sk.pos[j][1])
    lean = float((sk.pos[tip] - sk.pos[hips])[along])
    if abs(lean) <= 0.3 * sk.height:
        leg = sk.chain(legs[0][0])
        lean = (
            float((sk.pos[leg[-1]] - sk.pos[leg[-2]])[along]) if len(leg) > 1 else 0.0
        )
    out = np.zeros(3)
    out[along] = -1.0 if lean < -1e-9 else 1.0
    return out


def _facing(forward: np.ndarray) -> str:
    axis = int(np.argmax(np.abs(forward)))
    return ("+" if forward[axis] > 0 else "-") + "xyz"[axis]


def _side(sk: _Skeleton, node: int, hips: int, left: np.ndarray) -> str:
    return "left" if (sk.pos[node] - sk.pos[hips]) @ left > 0 else "right"


def _limb(
    sk: _Skeleton, start: int, parts: tuple[str, ...], stub: str | None = None
) -> dict[str, int]:
    """The nodes along a limb. A short first segment is a humanoid's shoulder when
    *stub* names it, and stepped past otherwise (the root of a pug's leg)."""
    nodes = sk.chain(start)
    lengths = [
        float(np.linalg.norm(sk.pos[b] - sk.pos[a])) for a, b in zip(nodes, nodes[1:])
    ]
    total = sum(lengths) or 1.0
    skipped = None
    while len(lengths) > 2 and lengths[0] < _STUB * total:
        skipped, nodes, lengths = nodes[0], nodes[1:], lengths[1:]
    found = dict(zip(parts, nodes))
    if stub and skipped is not None:
        found[stub] = skipped
    return found


def _floor_chains(sk: _Skeleton) -> list[tuple[int, int, int]]:
    """Every chain of three nodes ending at a leaf on the floor: a leg's candidates."""
    out = []
    for leaf in sorted(sk.joints):
        if sk.kids[leaf] or not sk.reaches_floor(leaf):
            continue
        lower = sk.file.parents[leaf]
        upper = None if lower is None else sk.file.parents[lower]
        if lower is not None and upper is not None:
            out.append((upper, lower, leaf))
    return out


# The robot's side, by name.

#: The words that name each part of a limb, read along that limb's own bodies only.
_PART_WORDS = {
    "UpperLeg": ("thigh", "upperleg", "upper", "femur", "hip"),
    "LowerLeg": ("calf", "shin", "shank", "knee", "lowerleg", "lower", "tibia"),
    "Foot": ("foot", "ankle"),
    "UpperArm": ("upperarm", "upper", "shoulder", "humerus"),
    "LowerArm": ("elbow", "forearm", "fore", "lowerarm", "lower"),
    "Hand": ("hand", "wrist"),
}
#: On a four-legged robot an arm is a front leg, named as one.
_AS_LEG = {"UpperArm": "UpperLeg", "LowerArm": "LowerLeg", "Hand": "Foot"}


def _tokens(name: str) -> list[str]:
    return [
        t
        for t in re.split(r"[^a-z0-9]+", name.rsplit("/", maxsplit=1)[-1].lower())
        if t
    ]


def _robot_side(name: str) -> tuple[str | None, str | None]:
    """A body's side, and on a four-legged robot whether it is a front or rear leg."""
    tokens = _tokens(name)
    head = tokens[0] if tokens else ""
    if re.fullmatch(r"[fr][lr]", head):
        return ("left" if head[1] == "l" else "right"), (
            "front" if head[0] == "f" else "rear"
        )
    if {"left", "l"} & set(tokens):
        return "left", None
    if {"right", "r"} & set(tokens):
        return "right", None
    return None, None


def guess_bones(model: mujoco.MjModel, bones: Iterable[str]) -> dict[str, str]:
    """Bone -> body for each of *bones* (VRM humanoid bone names) the robot's body names
    point to. Each limb's last body named for a part is that part (a G1's hip pitch,
    roll and yaw links give its yaw); a robot with no foot body stands on its calf."""
    wanted = set(bones)
    names = [model.body(b).name for b in range(model.nbody)]
    root = next(
        (
            b
            for b in range(1, model.nbody)
            if model.body_jntnum[b]
            and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE
        ),
        1 if model.nbody > 1 else 0,
    )

    def down_to(body: int) -> list[int]:
        out = []
        while body not in (0, root):
            out.append(body)
            body = model.body_parentid[body]
        return out[::-1]

    four_legged = any(_robot_side(n)[1] for n in names)
    parents = set(model.body_parentid[1:])
    limbs: dict[tuple[str, str], list[int]] = {}
    for body in range(1, model.nbody):
        side, end = _robot_side(names[body])
        if body in parents or side is None:
            continue
        if four_legged:
            kind = "arm" if end == "front" else "leg"
        else:
            kind = (
                "arm"
                if {"wrist", "hand", "elbow"} & set(_tokens(names[body]))
                else "leg"
            )
        limbs[(side, kind)] = down_to(body)

    found = {"hips": names[root]}
    trunk = [root]
    if not four_legged and ("left", "arm") in limbs:
        trunk = [
            b for b in limbs[("left", "arm")] if _robot_side(names[b])[0] is None
        ] or [root]
    for bone, words in (
        ("spine", {"waist", "spine", "abdomen"}),
        ("chest", {"torso", "chest"}),
    ):
        hits = [b for b in trunk if words & set(_tokens(names[b]))]
        found[bone] = (
            names[hits[-1]] if hits else found["spine" if bone == "chest" else "hips"]
        )
    for bone, word in (("neck", "neck"), ("head", "head")):
        hits = [b for b in range(model.nbody) if word in _tokens(names[b])]
        found[bone] = (
            names[hits[-1]] if hits else found["chest" if bone == "neck" else "neck"]
        )
    for (side, kind), chain in limbs.items():
        parts = (
            ("UpperArm", "LowerArm", "Hand")
            if kind == "arm"
            else ("UpperLeg", "LowerLeg", "Foot")
        )
        for part in parts:
            words = _PART_WORDS[_AS_LEG.get(part, part) if four_legged else part]
            hits = [b for b in chain if set(words) & set(_tokens(names[b]))]
            if hits:
                found[f"{side}{part}"] = names[hits[-1]]
        last = f"{side}{parts[2]}"
        found.setdefault(last, found.get(f"{side}{parts[1]}", ""))
        if kind == "leg":
            found[f"{side}Toes"] = found[last]
    return in_vrm_order(
        {
            bone: body.split("/")[-1]
            for bone, body in found.items()
            if bone in wanted and body
        }
    )


def guess_scale(
    file: SkinFile,
    guess: SkinGuess,
    model: mujoco.MjModel,
    bones: Mapping[str, str],
) -> tuple[float, str] | None:
    """Metres per unit that size a ``.glb`` to the robot, and how that was measured:
    a four-legged one's front and rear limb roots as far apart as the robot's, a
    humanoid's hips as high above its lowest joint as the robot's root stands."""

    def body(bone: str) -> int:
        return _body(model, bones[bone])

    nodes, rest = guess.humanoid, file.rest
    zero = pose_at_rest(model, None)
    if guess.kind == "quadruped" and {"hips", "leftUpperArm", "leftUpperLeg"} <= set(
        bones
    ) & set(nodes):
        root = body("hips")
        front, back = (
            limb_root(model, root, body(b)) for b in ("leftUpperArm", "leftUpperLeg")
        )
        robot = float(np.linalg.norm(zero.xpos[front] - zero.xpos[back]))
        axis = {"+z": 2, "-z": 2, "+x": 0, "-x": 0}[guess.facing]
        skin = abs(
            float(rest[nodes["leftUpperArm"]][axis] - rest[nodes["leftUpperLeg"]][axis])
        )
        if robot > 0 and skin > 0:
            return robot / skin, (
                f"its front and rear limb roots {skin:.3g} units apart; the robot's {robot:.3g} m"
            )
    if guess.kind == "humanoid" and "hips" in bones and "hips" in nodes:
        robot = standing_height(model, body("hips"), zero)
        joints = file.joints or range(len(file.names))
        skin = float(rest[nodes["hips"]][1] - min(rest[j][1] for j in joints))
        if robot > 0 and skin > 0:
            return robot / skin, (
                f"its hips {skin:.3g} units above its lowest joint; the robot's root {robot:.3g} m up"
            )
    return None


#: The robots whose rest pose has been checked by hand, by the joints that hold it: the
#: arms out straight, as a VRM rests.
KNOWN_REST_POSES: dict[str, dict[str, float]] = {
    "mjlab's Unitree G1": {
        "left_shoulder_roll_joint": math.pi / 2,
        "right_shoulder_roll_joint": -math.pi / 2,
        "left_elbow_joint": math.pi / 2,
        "right_elbow_joint": math.pi / 2,
    },
}


def known_rest_pose(model: mujoco.MjModel) -> tuple[str, dict[str, float]] | None:
    """A checked robot's T-pose, if *model* has its joints."""
    for robot, joints in KNOWN_REST_POSES.items():
        if all(resolve_joint(model, name) is not None for name in joints):
            return robot, joints
    return None


def arm_joints(
    model: mujoco.MjModel, bones: Mapping[str, str]
) -> list[tuple[str, tuple[float, float, float]]]:
    """Each hinge from where an arm leaves the trunk to its hand, with its axis in the
    world at the zero pose: the joints a rest pose sets."""
    data = pose_at_rest(model, None)
    trunk = bones.get("chest") or bones.get("spine") or bones.get("hips")
    out = []
    for side in ("left", "right"):
        hand, upper = bones.get(f"{side}Hand"), bones.get(f"{side}UpperArm")
        if not (trunk and hand and upper):
            continue
        top = limb_root(model, _body(model, trunk), _body(model, upper))
        chain, body = [], _body(model, hand)
        while body > 0:
            chain.append(body)
            if body == top:
                break
            body = model.body_parentid[body]
        for b in reversed(chain):
            for j in range(
                model.body_jntadr[b], model.body_jntadr[b] + model.body_jntnum[b]
            ):
                if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                    x, y, z = (round(float(a), 3) + 0.0 for a in data.xaxis[j])
                    out.append((model.joint(j).name.split("/")[-1], (x, y, z)))
    return out


def _body(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, resolve_body(model, name) or ""
    )


def close_bodies(model: mujoco.MjModel, node: str, n: int = 3) -> list[str]:
    """The robot's bodies named most like *node*."""
    names = [model.body(b).name.split("/")[-1] for b in range(1, model.nbody)]
    return difflib.get_close_matches(node, names, n=n, cutoff=0.4)


def draft_source(
    command: str,
    *,
    source: str,
    guess: SkinGuess | None,
    names: tuple[str, ...],
    bones: Mapping[str, str],
    scale: float | None = None,
    rest_pose: str | Mapping[str, float] | None = None,
) -> str:
    """A script defining ``SKIN``, the drafted :class:`~mjswan.SkinConfig`, for its
    author to check and paste. *guess* is None for a VRM, which names its own bones."""

    def table(entries: Mapping[str, object], indent: str = "        ") -> str:
        return "".join(f"{indent}{_py(k)}: {_py(v)},\n" for k, v in entries.items())

    lines = [
        f'"""A skin drafted by `{command}`.\n\nEvery map here is a guess: check each line '
        'against the file and the robot."""\n',
        "import mjswan\n",
        "SKIN = mjswan.SkinConfig(",
        f"    source={_py(source)},",
    ]
    if guess is not None and guess.humanoid:
        humanoid = {b: names[n] for b, n in guess.humanoid.items()}
        lines.append("    humanoid={\n" + table(humanoid) + "    },")
    if guess is not None and guess.kind == "other":
        top = next(
            (
                n
                for n in range(len(names))
                if all(n not in limb for limb in guess.limbs)
            ),
            0,
        )
        lines.append(
            "    bones={\n" + table({names[top]: bones.get("root", "")}) + "    },"
        )
        lines += [
            "    # Chains reaching the floor, each a limb: map its nodes in bones, then",
            "    # list it here.",
            "    limbs=[",
            *(
                f"        # ({', '.join(_py(names[n]) for n in limb)}),"
                for limb in guess.limbs
            ),
            "    ],",
        ]
    else:
        lines.append("    bones={\n" + table(dict(bones)) + "    },")
    if scale is not None:
        lines.append(f"    scale={float(f'{scale:.2g}')!r},")
    if guess is not None and guess.facing != "+z":
        lines.append(f"    facing={_py(guess.facing)},")
    if rest_pose is None:
        lines.append(
            "    # rest_pose: the robot's joints holding the pose the skin rests in."
        )
    elif isinstance(rest_pose, str):
        lines.append(f"    rest_pose={_py(rest_pose)},")
    else:
        lines.append("    rest_pose={\n" + table(dict(rest_pose)) + "    },")
    if guess is not None and guess.springs:
        lines.append("    springs=[")
        lines += [
            f"        mjswan.SpringChain(root={names[n]!r})," for n in guess.springs
        ]
        lines.append("    ],")
    lines.append(")\n")
    return "\n".join(lines)


def _py(value: object) -> str:
    """*value* as Python source, a string double-quoted."""
    return json.dumps(value) if isinstance(value, str) else repr(value)


def in_vrm_order(entries: Mapping[str, T]) -> dict[str, T]:
    """*entries* keyed by VRM bone, as VRM lists the bones: the torso, then each side."""
    order = {bone: i for i, bone in enumerate(VRM_PARENTS)}
    return dict(sorted(entries.items(), key=lambda kv: order.get(kv[0], len(order))))


__all__ = [
    "KNOWN_REST_POSES",
    "SkinGuess",
    "arm_joints",
    "close_bodies",
    "draft_source",
    "guess_bones",
    "guess_humanoid",
    "guess_scale",
    "in_vrm_order",
    "known_rest_pose",
]
