"""Skins: VRM avatars posed from a robot's bodies, a render layer outside the simulation.

The browser rotates each mapped VRM bone the way its robot body has rotated since the
robot's zero pose, and moves the avatar's hips with the robot's root. Physics, collision
and the policy never see it. A scene may carry several skins; the viewer shows one at a
time, picked from its Skin menu.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco

if TYPE_CHECKING:
    from .scene import SceneHandle

#: The humanoid bones three-vrm addresses, VRM 1.0's names. A VRM 0.x file names the
#: same bones (its thumb bones aside), and three-vrm maps them onto these.
VRM_HUMAN_BONES = frozenset(
    {
        "hips",
        "spine",
        "chest",
        "upperChest",
        "neck",
        "head",
        "leftEye",
        "rightEye",
        "jaw",
        *(
            f"{side}{bone}"
            for side in ("left", "right")
            for bone in (
                "UpperLeg",
                "LowerLeg",
                "Foot",
                "Toes",
                "Shoulder",
                "UpperArm",
                "LowerArm",
                "Hand",
                "ThumbMetacarpal",
                "ThumbProximal",
                "ThumbDistal",
                *(
                    f"{finger}{joint}"
                    for finger in ("Index", "Middle", "Ring", "Little")
                    for joint in ("Proximal", "Intermediate", "Distal")
                ),
            )
        ),
    }
)

_URL = re.compile(r"^https?://", re.IGNORECASE)


@dataclass(frozen=True)
class SkinConfig:
    """Everything one skin needs, so swapping avatars swaps only this object.

    Frames: ``offset`` is MuJoCo's world frame (x forward, y left, z up).
    ``rest_offsets`` are in the avatar's own frame as VRM 1.0 defines it (y up, facing
    +z, its left hand toward +x), whatever version the file is.
    """

    bones: Mapping[str, str]
    """VRM humanoid bone -> robot body. Several bones may share a body; a body name may
    leave out its entity prefix (``"pelvis"`` finds ``"robot/pelvis"``). ``hips`` is
    required: it carries the avatar's position."""

    source: str | None = None
    """Local path or http(s) URL of the ``.vrm``. A URL is downloaded by
    :meth:`~mjswan.scene.SceneHandle.add_skin`; either way the build bundles the
    file byte for byte."""

    rest_offsets: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    """VRM bone -> the rotation, in degrees about the avatar's x, then y, then z axis,
    that turns the bone from the avatar's T-pose to where it sits while the robot is at
    its zero pose (a G1 hangs its arms down there)."""

    height_ratio: float | None = None
    """Scale from the robot root's height to the avatar's hips height; across the floor
    the hips follow the root exactly. ``None`` takes the avatar's hips height over the
    robot root's height at the model's zero pose."""

    offset: tuple[float, float, float] = (0.0, 1.0, 0.0)
    """Added to the avatar's position, in metres: it moves where the robot moves, this
    far aside (the default stands it 1 m to the robot's left)."""

    hide_robot: bool = False
    """Stop drawing the robot the avatar follows, so with ``offset=(0, 0, 0)`` the
    avatar is its skin. Its bodies are still simulated, collide and can be grabbed."""

    name: str = ""
    """Shown in the viewer's Skin menu. Set by :meth:`~mjswan.scene.SceneHandle.add_skin`."""

    id: str = ""
    """``name2id(name)``, unique within the scene; the bundled file's stem."""

    def __post_init__(self) -> None:
        unknown = sorted(set(self.bones) - VRM_HUMAN_BONES)
        if unknown:
            raise ValueError(f"Not VRM humanoid bones: {unknown}.")
        if "hips" not in self.bones:
            raise ValueError(
                "SkinConfig.bones must map 'hips', which moves the avatar."
            )
        unmapped = sorted(set(self.rest_offsets) - set(self.bones))
        if unmapped:
            raise ValueError(
                f"rest_offsets names bones that bones does not map: {unmapped}."
            )
        if self.height_ratio is not None and self.height_ratio <= 0:
            raise ValueError(f"height_ratio must be positive, got {self.height_ratio}.")

    @property
    def path(self) -> str:
        """Where the build puts the file, relative to the scene directory."""
        return f"assets/{self.id}.vrm"

    def to_dict(self) -> dict[str, Any]:
        """One entry of the scene's manifest ``skins``; offsets go as ``[x, y, z, w]``."""
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "bones": dict(self.bones),
            "rest_offsets": {
                bone: _euler_to_quat(angles)
                for bone, angles in self.rest_offsets.items()
            },
            **({"height_ratio": self.height_ratio} if self.height_ratio else {}),
            "offset": list(self.offset),
            **({"hide_robot": True} if self.hide_robot else {}),
        }


def _euler_to_quat(degrees: tuple[float, float, float]) -> list[float]:
    """Rotations about the fixed x, y, z axes in turn, as ``[x, y, z, w]``."""
    q = [0.0, 0.0, 0.0, 1.0]
    for axis, angle in enumerate(degrees):
        half = math.radians(angle) / 2
        r = [0.0, 0.0, 0.0, math.cos(half)]
        r[axis] = math.sin(half)
        q = _quat_mul(r, q)
    return q


def _quat_mul(a: list[float], b: list[float]) -> list[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ]


def resolve_body(model: mujoco.MjModel, name: str) -> str | None:
    """The model's body *name* names: itself, else the one ``<entity>/<name>``.

    The browser matches the same way, so what passes here is found there.
    """
    names = [model.body(b).name for b in range(model.nbody)]
    if name in names:
        return name
    prefixed = [n for n in names if n.endswith(f"/{name}")]
    return prefixed[0] if len(prefixed) == 1 else None


def resolve_skin(config: SkinConfig, model: mujoco.MjModel | None) -> SkinConfig:
    """*config* with its source made a local file, checked against the scene's model.

    Raises :class:`FileNotFoundError` for a missing file, ``urllib.error.URLError`` for
    a URL that does not download, and :class:`ValueError` for a file that is not a
    binary glTF or a body the model lacks.
    """
    if config.source is None:
        raise ValueError("SkinConfig.source is not set: give the .vrm's path or URL.")
    if _URL.match(config.source):
        from .source import url

        path = url.fetch_file(config.source)
    else:
        path = Path(config.source).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VRM file not found: {path}")
    with path.open("rb") as f:
        if f.read(4) != b"glTF":
            raise ValueError(f"{path} is not a VRM (a binary glTF file).")

    if model is not None:
        missing = sorted(
            body
            for body in set(config.bones.values())
            if resolve_body(model, body) is None
        )
        if missing:
            raise ValueError(f"The scene's model has no body named {missing}.")
    return replace(config, source=str(path))


class SkinHandle:
    """A skin added to a scene. Its config is frozen; build a new one to change it."""

    def __init__(self, skin_config: SkinConfig, scene: SceneHandle) -> None:
        self._config = skin_config
        self._scene = scene

    @property
    def config(self) -> SkinConfig:
        return self._config

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def id(self) -> str:
        return self._config.id

    @property
    def source(self) -> str | None:
        """The local file the build bundles."""
        return self._config.source


__all__ = ["VRM_HUMAN_BONES", "SkinConfig", "SkinHandle", "resolve_skin"]
