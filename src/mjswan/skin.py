"""Skins: a rigged character posed from a robot's bodies, drawn only, never simulated.

A skin is a ``.vrm``, which names its own bones, or a ``.glb``, whose bones are named by
``humanoid`` (a humanoid, or a four-legged animal with its front legs as arms) or, for
any other skeleton, by node name. Its spring bones swing outside the scene's model, by
three-vrm or in a MuJoCo model of their own, so the scene's physics, collision and policy
never see them.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mujoco

from .document.skin import FACING_TURNS, SkinFile, read_skin_file
from .rig.humanoid import (
    HUMANOID_LIMBS,
    HUMANOID_PRESETS,
    VRM_HUMAN_BONES,
    bone_ancestors,
    resolve_humanoid,
)
from .rig.pose import (
    euler_to_quat,
    hips_offset,
    limb_ratios,
    pose_at_rest,
    quat_mul,
    resolve_body,
    rest_offsets,
    standing_height,
)

if TYPE_CHECKING:
    from .scene import SceneHandle

_URL = re.compile(r"^https?://", re.IGNORECASE)

#: How far a limb may be from the robot's length before ``"position"`` is warned off.
STRETCH_TOLERANCE = 0.1


@dataclass(frozen=True)
class SpringCollider:
    """A sphere on a glTF node, or with ``tail`` a capsule, that springs stay out of.
    Lengths in metres, along the node's own axes."""

    node: str
    radius: float
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tail: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class SpringChain:
    """A glTF node and each first child below it, swinging as a VRM spring does. The
    settings are VRM's: ``stiffness`` pulls back to rest, ``drag`` damps, ``gravity``
    pulls along ``gravity_dir``. ``tip`` (metres) gives the last node something to point
    along, so it swings too."""

    root: str
    stiffness: float = 1.0
    drag: float = 0.4
    gravity: float = 0.0
    gravity_dir: tuple[float, float, float] = (0.0, -1.0, 0.0)
    hit_radius: float = 0.02
    tip: float | None = None
    colliders: tuple[SpringCollider, ...] = ()


@dataclass(frozen=True)
class ToonShading:
    """A glTF's PBR materials drawn as MToon, balanced against the scene's lights as a
    VRM's are: ``shade`` is the shaded side's share of the lit colour, ``toony`` how
    sharply the two meet (0 soft, 1 hard), ``shift`` where they meet."""

    shade: float = 0.6
    toony: float = 0.5
    shift: float = 0.0


@dataclass(frozen=True)
class SkinConfig:
    """Everything one skin needs, so swapping characters swaps only this object.

    Frames: ``offset`` and ``hips_offset`` are MuJoCo's (x forward, y left, z up).
    ``rest_offsets`` are in the character's own frame as VRM 1.0 defines it (y up,
    facing +z, its left toward +x), whatever the file is.
    """

    bones: Mapping[str, str]
    """Bone -> robot body. A bone is a VRM humanoid bone name, for a ``.vrm`` or a
    ``.glb`` given ``humanoid``, else one of the ``.glb``'s node names. Several bones may
    share a body; a body name may leave out its entity prefix (``"pelvis"`` finds
    ``"robot/pelvis"``). A humanoid's ``hips``, or a node map's topmost node, carries
    the position."""

    source: str | None = None
    """Local path or http(s) URL of the ``.vrm`` or ``.glb``. A URL is downloaded by
    :meth:`~mjswan.scene.SceneHandle.add_skin`; either way the build bundles the file
    byte for byte."""

    humanoid: Mapping[str, str] | str | None = None
    """``.glb`` only: VRM humanoid bone -> node name, or a preset in
    :data:`HUMANOID_PRESETS` (``"mixamo"``). A four-legged animal maps its front legs as
    arms. ``None`` keys ``bones`` by node name instead."""

    limbs: Sequence[tuple[str, str, str]] = ()
    """A node map's arms and legs, as ``(upper, lower, end)`` keys of ``bones``:
    what ``retarget="hybrid"`` bends. A humanoid's come from its bones."""

    reparent: Mapping[str, str] = field(default_factory=dict)
    """``.glb`` only: node -> the node to hang it from instead, its rest pose kept, before
    anything else reads the nesting. For a rig whose feet an IK control carries rather
    than its legs, as Quaternius's animals do: posed from joint angles alone, the feet
    would stay behind."""

    scale: float = 1.0
    """``.glb`` only: metres per unit of the file."""

    facing: Literal["+z", "-z", "+x", "-x"] = "+z"
    """``.glb`` only: the axis the model faces (glTF's convention is ``+z``)."""

    springs: Sequence[SpringChain] = ()
    """``.glb`` only: chains that swing, as a VRM's spring bones do."""

    material: ToonShading | None = None
    """``.glb`` only: draw its PBR materials as MToon."""

    rest_pose: str | Mapping[str, float] | None = None
    """The robot's pose while the character holds its own rest pose, as the file has it:
    a keyframe's name, or joint -> value (radians, metres) over the zero pose, joints not
    named left at zero. Every VRM rests in a T-pose, so for a humanoid robot this is its
    joints set to hold the arms out straight. ``None``: the zero pose."""

    rest_offsets: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    """Bone -> the turn, in degrees about the character's x, then y, then z axis, it takes
    from its rest while the robot is at ``rest_pose``: for what the robot's joints cannot
    pose."""

    height_ratio: float | None = None
    """Scale from the robot root's height to the character's root bone height, both above
    the ground under the robot (the first geom in group 0 straight below that is not the
    robot's, as mjlab's height scan sees the terrain), so a hill lifts both alike. Across
    the floor the root bone follows the root exactly. ``None`` takes the root bone's
    height above the character's lowest point at rest, over the robot root's above its
    soles at ``rest_pose``."""

    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Added to the character's position, in metres: it moves where the robot moves,
    this far aside. The default draws it over the robot; ``(0, 1, 0)`` stands it 1 m to
    the robot's left."""

    hips_offset: tuple[float, float, float] | None = None
    """Where the root bone sits on the body it maps to, in that body's own frame,
    metres, so it turns with the robot. ``None`` centres the character's limbs over the
    robot's across the floor: a fox's hips bone is its pelvis, behind a Go1's mid-trunk
    origin."""

    hide_robot: bool = True
    """Stop drawing the robot the skin follows, so the character is drawn in its place.
    ``False`` draws both, as when ``offset`` stands it beside the robot. A hidden robot
    is still simulated, collides and can be grabbed, and Debug Viz's Xray switch draws
    it through the skin."""

    brightness: float = 1.0
    """Scales the character's toon (MToon) colours, which the viewer first divides by the
    scene's summed light when that exceeds 1: a toon shader takes every light it faces in
    full, so MuJoCo's several lights would clip it to white."""

    mujoco_springs: bool = False
    """Simulate the spring bones as MuJoCo bodies, in a model beside the scene's that the
    robot never touches. ``False``, the default, leaves them to three-vrm's springs, which
    cost less per frame."""

    retarget: Literal["rotation", "position", "hybrid"] = "hybrid"
    """How the character follows the robot. ``"hybrid"`` turns each bone the way its body
    has turned since ``rest_pose``, keeping the character's proportions, then bends each
    arm and leg (two-bone IK, each bone keeping its length) so its hand or foot lands on
    the robot's, the whole character lowered as far as the feet it stands on need to
    reach. ``"rotation"`` only turns the bones: the hands and feet land where the
    character's own lengths put them. ``"position"`` fixes every bone to its body, as the
    robot's own meshes are, for a model made to the robot's proportions; on any other,
    the mesh stretches or folds across the joints, and ``add_skin`` warns."""

    license: str | None = None
    """``.glb`` only: a path to its license text, or a generatable SPDX id
    (``"CC-BY-4.0"``, ``"CC0-1.0"``, ...), written to ``LICENSE.skin-<id>`` beside the
    scene, as :meth:`~mjswan.scene.SceneHandle.add_attribution` writes it. A VRM's
    license is its own ``meta``."""

    notice: str | None = None
    """``.glb`` only: its attribution, a path or the text itself, written to
    ``NOTICE.skin-<id>``."""

    name: str = ""
    """Shown in the viewer's Skin menu. Set by :meth:`~mjswan.scene.SceneHandle.add_skin`."""

    id: str = ""
    """``name2id(name)``, unique within the scene; the bundled file's stem."""

    def __post_init__(self) -> None:
        if not self.bones:
            raise ValueError("SkinConfig.bones is empty: map at least the root bone.")
        unmapped = sorted(set(self.rest_offsets) - set(self.bones))
        if unmapped:
            raise ValueError(
                f"rest_offsets names bones that bones does not map: {unmapped}."
            )
        for limb in self.limbs:
            if len(limb) != 3 or len(set(limb)) != 3:
                raise ValueError(
                    f"A limb is (upper, lower, end), three bones; got {limb!r}."
                )
            missing = [bone for bone in limb if bone not in self.bones]
            if missing:
                raise ValueError(
                    f"limbs names bones that bones does not map: {missing}."
                )
        looped = sorted(node for node, to in self.reparent.items() if node == to)
        if looped:
            raise ValueError(f"reparent hangs {looped} from themselves.")
        if self.facing not in FACING_TURNS:
            raise ValueError(
                f"facing must be one of {sorted(FACING_TURNS)}, got {self.facing!r}."
            )
        if self.retarget not in ("rotation", "position", "hybrid"):
            raise ValueError(
                f"retarget must be 'rotation', 'position' or 'hybrid', got {self.retarget!r}."
            )
        for name in ("scale", "brightness", "height_ratio"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")


@dataclass(frozen=True)
class ResolvedSkin:
    """A skin as :meth:`~mjswan.scene.SceneHandle.add_skin` resolved it against its file
    and the scene's model: what the build writes."""

    config: SkinConfig
    """With ``source`` a local file, ``name`` and ``id`` set."""
    file: SkinFile
    rig: Literal["vrm", "humanoid", "nodes"]
    nodes: dict[str, int]
    """A ``.glb``'s bone -> node index; empty for a VRM, which names its own."""
    root: str
    """The bone that carries the position."""
    limbs: list[tuple[str, str, str]]
    """The arms and legs ``retarget="hybrid"`` bends."""
    springs: list[dict[str, Any]]
    rest_offsets: dict[str, list[float]]
    """Bone -> ``[x, y, z, w]``: ``rest_pose`` and ``rest_offsets`` together."""
    root_height: float | None
    hips_offset: tuple[float, float, float]
    reparent: list[tuple[int, int]] = field(default_factory=list)
    """``(node, new parent)`` indices, in order, for the browser to nest as ``file`` is."""

    @property
    def path(self) -> str:
        """Where the build puts the file, relative to the scene directory."""
        return (
            f"assets/{self.config.id}.{'vrm' if self.file.format == 'vrm' else 'glb'}"
        )


def resolve_skin(config: SkinConfig, model: mujoco.MjModel | None) -> ResolvedSkin:
    """*config* resolved: its file read and checked, its bones found, and, given the
    scene's *model*, its rest pose and hips measured against the robot.

    Raises :class:`FileNotFoundError` for a missing file, ``urllib.error.URLError`` for
    a URL that does not download, and :class:`ValueError` for a file or a map that does
    not fit.
    """
    file, reparent = _reparented(config, read_skin_file(_local_source(config)))
    rig, nodes, root = _bones(config, file)
    springs = [_spring(chain, file) for chain in config.springs]
    limbs = _limbs(config, rig)

    at = file.positions(config.scale, config.facing)
    named = file.human_bones if file.format == "vrm" else nodes
    positions = {bone: at[n] for bone, n in named.items()}

    turns: dict[str, list[float]] = {}
    root_height = None
    hips = config.hips_offset or (0.0, 0.0, 0.0)
    if model is not None:
        bodies = _bodies(config, model)
        rest = pose_at_rest(model, config.rest_pose)
        turns = rest_offsets(model, bodies, root, rest)
        root_height = round(standing_height(model, bodies[root], rest), 6)
        # Each limb's first bone -> the mapped bone it hangs from.
        limb_roots = {
            upper: above
            for upper, _, _ in limbs
            if upper in positions and (above := _hangs_from(upper, config, file, nodes))
        }
        if config.hips_offset is None and root in positions:
            hips = hips_offset(
                model, bodies, root, rest, limb_roots=limb_roots, positions=positions
            )
        if config.retarget == "position":
            measured = [
                limb
                for limb in limbs
                if limb[0] in limb_roots and all(bone in positions for bone in limb)
            ]
            _warn_stretch(
                file,
                model,
                bodies,
                measured,
                limb_roots=limb_roots,
                positions=positions,
            )
    for bone, degrees in config.rest_offsets.items():
        turns[bone] = quat_mul(
            turns.get(bone, [0.0, 0.0, 0.0, 1.0]), euler_to_quat(degrees)
        )

    return ResolvedSkin(
        config=replace(config, source=str(file.path)),
        file=file,
        rig=rig,
        nodes=nodes,
        root=root,
        limbs=limbs,
        springs=springs,
        rest_offsets=turns,
        root_height=root_height,
        hips_offset=hips,
        reparent=reparent,
    )


def _reparented(
    config: SkinConfig, file: SkinFile
) -> tuple[SkinFile, list[tuple[int, int]]]:
    """*file* nested as ``reparent`` says, and the moves as node indices.

    Only the nesting changes: ``rest`` holds each node where it stands in the file,
    which a move that keeps the rest pose leaves alone. A VRM is refused by ``_bones``.
    """
    if not config.reparent or file.format == "vrm":
        return file, []
    parents = list(file.parents)
    moves = []
    for name, parent_name in config.reparent.items():
        node, parent = file.node(name), file.node(parent_name)
        above: int | None = parent
        while above is not None:
            if above == node:
                raise ValueError(
                    f"reparent hangs {name!r} from {parent_name!r}, which hangs from it."
                )
            above = parents[above]
        parents[node] = parent
        moves.append((node, parent))
    return replace(file, parents=tuple(parents)), moves


def _local_source(config: SkinConfig) -> Path:
    if config.source is None:
        raise ValueError(
            "SkinConfig.source is not set: give the .vrm's or .glb's path or URL."
        )
    if _URL.match(config.source):
        from .source import url

        return url.fetch_file(config.source)
    path = Path(config.source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Skin file not found: {path}")
    return path


#: Fields a VRM brings itself, so its config may not set them.
_GLTF_ONLY = (
    "humanoid",
    "limbs",
    "reparent",
    "scale",
    "facing",
    "springs",
    "material",
    "license",
    "notice",
)


def _bones(
    config: SkinConfig, file: SkinFile
) -> tuple[Literal["vrm", "humanoid", "nodes"], dict[str, int], str]:
    """Which rig the file is, each bone's node, and the bone that carries the position."""
    if file.format == "vrm":
        defaults = {f.name: f.default for f in fields(SkinConfig)}
        set_here = [
            name
            for name in _GLTF_ONLY
            if getattr(config, name) != defaults[name]
            and getattr(config, name) not in (None, (), [], {})
        ]
        if set_here:
            raise ValueError(
                f"{file.path.name} is a VRM: its humanoid, springs, units, materials and "
                f"license are the file's, bundled unmodified. Remove {', '.join(set_here)}."
            )
        _require_vrm_bones(config)
        absent = sorted(set(config.bones) - set(file.human_bones))
        if absent:
            warnings.warn(
                f"{file.path.name} has no {', '.join(absent)}; it stays at rest.",
                stacklevel=4,
            )
        return "vrm", {}, "hips"

    if config.humanoid is not None:
        if config.limbs:
            raise ValueError(
                "limbs are for a node map; a humanoid's come from its bones."
            )
        _require_vrm_bones(config)
        humanoid = resolve_humanoid(config.humanoid, file)
        unnamed = sorted(set(config.bones) - set(humanoid))
        if unnamed:
            raise ValueError(f"bones maps {unnamed}, which humanoid does not name.")
        rig = "humanoid"
        nodes = {bone: humanoid[bone] for bone in config.bones}
    else:
        rig = "nodes"
        nodes = _named_nodes(config, file)
    # The browser poses a .glb by its own nesting, the topmost mapped node carrying it.
    tops = [bone for bone in nodes if _above(bone, nodes, file) is None]
    if len(tops) != 1 or (rig == "humanoid" and tops != ["hips"]):
        which = "hips, " if rig == "humanoid" else ""
        raise ValueError(
            f"bones must have one topmost node, {which}which carries the position; "
            f"{tops} are."
        )
    for upper, lower, end in _limbs(config, rig):
        if _above(lower, nodes, file) != upper or _above(end, nodes, file) != lower:
            raise ValueError(
                f"limb {(upper, lower, end)!r} is not a chain of the file's nodes, "
                "each the nearest mapped one below the one before."
            )

    swinging = {n for chain in config.springs for n in _chain_joints(chain, file)}
    posed = sorted(bone for bone, n in nodes.items() if n in swinging)
    if posed:
        raise ValueError(f"springs would swing {posed}, which bones already poses.")
    return rig, nodes, tops[0]


def _named_nodes(config: SkinConfig, file: SkinFile) -> dict[str, int]:
    nodes = {}
    for bone in config.bones:
        try:
            nodes[bone] = file.node(bone)
        except ValueError as exc:
            hint = (
                " Give humanoid to key bones by VRM bone name."
                if bone in VRM_HUMAN_BONES
                else ""
            )
            raise ValueError(f"{exc}{hint}") from None
    return nodes


def _above(bone: str, nodes: Mapping[str, int], file: SkinFile) -> str | None:
    """The mapped bone whose node is nearest above *bone*'s."""
    by_node = {n: b for b, n in nodes.items()}
    found = file.nearest_ancestor(nodes[bone], by_node)
    return None if found is None else by_node[found]


def _require_vrm_bones(config: SkinConfig) -> None:
    unknown = sorted(set(config.bones) - VRM_HUMAN_BONES)
    if unknown:
        raise ValueError(f"Not VRM humanoid bones: {unknown}.")
    if "hips" not in config.bones:
        raise ValueError(
            "SkinConfig.bones must map 'hips', which carries the position."
        )


def _limbs(config: SkinConfig, rig: str) -> list[tuple[str, str, str]]:
    if rig == "nodes":
        return list(config.limbs)
    return [
        limb for limb in HUMANOID_LIMBS if all(bone in config.bones for bone in limb)
    ]


def _hangs_from(
    bone: str, config: SkinConfig, file: SkinFile, nodes: Mapping[str, int]
) -> str | None:
    """The nearest mapped bone *bone* hangs from: as VRM nests them in a VRM, as the
    file does in a .glb."""
    if file.format == "gltf":
        return _above(bone, nodes, file)
    return next((b for b in bone_ancestors(bone) if b in config.bones), None)


def _bodies(config: SkinConfig, model: mujoco.MjModel) -> dict[str, int]:
    found = {bone: resolve_body(model, body) for bone, body in config.bones.items()}
    missing = sorted(
        {config.bones[bone] for bone, name in found.items() if name is None}
    )
    if missing:
        raise ValueError(f"The scene's model has no body named {missing}.")
    return {
        bone: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for bone, name in found.items()
        if name is not None
    }


def _warn_stretch(
    file: SkinFile,
    model: mujoco.MjModel,
    bodies: Mapping[str, int],
    limbs: Sequence[tuple[str, str, str]],
    *,
    limb_roots: Mapping[str, str],
    positions: Mapping[str, Any],
) -> None:
    ratios = limb_ratios(
        model, bodies, limbs, limb_roots=limb_roots, positions=positions
    )
    off = sorted(r for r in ratios.values() if abs(r - 1) > STRETCH_TOLERANCE)
    if off:
        span = (
            f"x{off[0]:.2f}"
            if off[0] == off[-1]
            else f"x{off[0]:.2f} to x{off[-1]:.2f}"
        )
        warnings.warn(
            f"{file.path.name}: retarget='position' will stretch it: its arms and legs are "
            f"{span} the robot's lengths. 'hybrid' keeps its own.",
            stacklevel=4,
        )


def _chain_joints(chain: SpringChain, file: SkinFile) -> list[int]:
    """A spring chain's nodes: its root, then each first child down to the last."""
    joints = [file.node(chain.root)]
    while children := file.children(joints[-1]):
        joints.append(children[0])
    return joints


def _spring(chain: SpringChain, file: SkinFile) -> dict[str, Any]:
    joints = _chain_joints(chain, file)
    if len(joints) < 2 and chain.tip is None:
        raise ValueError(
            f"Spring {chain.root!r} has no child to swing toward; give it a tip."
        )
    return {
        "joints": joints,
        "stiffness": chain.stiffness,
        "drag": chain.drag,
        "gravity": chain.gravity,
        "gravity_dir": list(chain.gravity_dir),
        "hit_radius": chain.hit_radius,
        **({"tip": chain.tip} if chain.tip is not None else {}),
        "colliders": [
            {
                "node": file.node(c.node),
                "radius": c.radius,
                "offset": list(c.offset),
                **({"tail": list(c.tail)} if c.tail else {}),
            }
            for c in chain.colliders
        ],
    }


class SkinHandle:
    """A skin added to a scene. Its config is frozen; build a new one to change it."""

    def __init__(self, skin: ResolvedSkin, scene: SceneHandle) -> None:
        self._skin = skin
        self._scene = scene

    @property
    def config(self) -> SkinConfig:
        return self._skin.config

    @property
    def name(self) -> str:
        return self._skin.config.name

    @property
    def id(self) -> str:
        return self._skin.config.id

    @property
    def source(self) -> str | None:
        """The local file the build bundles."""
        return self._skin.config.source

    @property
    def format(self) -> Literal["vrm", "gltf"]:
        return self._skin.file.format


__all__ = [
    "HUMANOID_PRESETS",
    "VRM_HUMAN_BONES",
    "ResolvedSkin",
    "SkinConfig",
    "SkinHandle",
    "SpringChain",
    "SpringCollider",
    "ToonShading",
    "resolve_body",
    "resolve_skin",
]
