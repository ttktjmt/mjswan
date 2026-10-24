"""A skin's file as a document carries it: one binary glTF, everything embedded, nothing
compressed, read for its JSON chunk alone, never its meshes or images. A ``.vrm`` is a
binary glTF too, told apart by its VRM extension, not its name.
"""

from __future__ import annotations

import difflib
import json
import math
import struct
from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

#: Compression the engine has no decoder for. A file that uses it is refused.
COMPRESSION_EXTENSIONS: frozenset[str] = frozenset(
    {
        "KHR_draco_mesh_compression",
        "KHR_texture_basisu",
        "EXT_meshopt_compression",
        "KHR_meshopt_compression",
    }
)

#: A model's front -> the turn about +y that takes VRM 1.0's front (+z) onto it.
FACING_TURNS: dict[str, float] = {
    "+z": 0.0,
    "+x": math.pi / 2,
    "-z": math.pi,
    "-x": -math.pi / 2,
}

_MAGIC = b"glTF"
_JSON_CHUNK = 0x4E4F534A


@dataclass(frozen=True)
class SkinFile:
    """A ``.vrm`` or ``.glb`` as mjswan reads it."""

    path: Path
    format: Literal["vrm", "gltf"]
    vrm_version: Literal["0", "1"] | None
    """``"0"`` for VRM 0.x, which faces -z, ``"1"`` for VRM 1.0; ``None`` for a glTF."""
    names: tuple[str, ...]
    """Each node's name, by index, as the file writes it."""
    parents: tuple[int | None, ...]
    rest: np.ndarray = field(repr=False)
    """Each node's position at rest, ``(nodes, 3)``, in the file's own frame and units."""
    human_bones: dict[str, int] = field(default_factory=dict)
    """A VRM's humanoid: VRM 1.0 bone name -> node. Empty for a glTF."""
    meta: dict[str, Any] = field(default_factory=dict)
    """A VRM's ``meta`` as written. Empty for a glTF."""
    joints: frozenset[int] = frozenset()
    """Its skeleton: the nodes its skins deform the meshes by."""
    json_bytes: int = 0
    """The length of its JSON chunk, which mjswan Cloud bounds."""

    def node(self, name: str) -> int:
        """The one node called *name*; the error names the closest ones."""
        found = [i for i, n in enumerate(self.names) if n == name]
        if len(found) == 1:
            return found[0]
        if found:
            raise ValueError(
                f"{self.path.name} has {len(found)} nodes called {name!r}."
            )
        close = difflib.get_close_matches(name, self.names, n=3)
        hint = f" Similar: {', '.join(close)}." if close else ""
        raise ValueError(f"{self.path.name} has no node {name!r}.{hint}")

    def children(self, index: int) -> list[int]:
        return [i for i, p in enumerate(self.parents) if p == index]

    def ancestors(self, index: int) -> list[int]:
        """*index*'s parents up to a scene root, nearest first."""
        out = []
        parent = self.parents[index]
        while parent is not None:
            out.append(parent)
            parent = self.parents[parent]
        return out

    def nearest_ancestor(self, index: int, among: Collection[int]) -> int | None:
        """The nearest node of *among* above *index*."""
        return next((a for a in self.ancestors(index) if a in among), None)

    def positions(self, scale: float = 1.0, facing: str = "+z") -> np.ndarray:
        """Every node at rest in the avatar's frame (VRM 1.0's: y up, facing +z, its left
        toward +x), in metres: a VRM 0.x turned about, a glTF scaled and turned."""
        if self.vrm_version == "0":
            return self.rest * np.array([-1.0, 1.0, -1.0])
        turn = FACING_TURNS[facing]
        c, s = math.cos(turn), math.sin(turn)
        x, y, z = (self.rest * scale).T
        return np.stack([x * c - z * s, y, x * s + z * c], axis=1)


def read_skin_file(path: Path) -> SkinFile:
    """*path*'s format, nodes and, for a VRM, its humanoid and meta.

    Raises :class:`ValueError` for a file that is not one self-contained binary glTF
    2.0, or that uses compression the engine cannot decode.
    """
    with path.open("rb") as f:
        header = f.read(20)
        if len(header) < 20 or header[:4] != _MAGIC:
            raise ValueError(_not_binary(path, header))
        version, declared, length, kind = struct.unpack_from("<IIII", header, 4)
        if version != 2 or kind != _JSON_CHUNK:
            raise ValueError(f"{path.name} is not a glTF 2.0 binary file.")
        if declared > path.stat().st_size or 20 + length > declared:
            raise ValueError(
                f"{path.name} is truncated: its header describes more bytes than it "
                "holds."
            )
        gltf = json.loads(f.read(length))
    try:
        return _skin_file(path, gltf, length)
    except (AttributeError, IndexError, KeyError, TypeError, RecursionError) as exc:
        # JSON that parses but is not shaped as glTF.
        raise ValueError(f"{path.name} is not a valid glTF ({exc!r}).") from None


def _skin_file(path: Path, gltf: dict[str, Any], json_bytes: int) -> SkinFile:
    outside = [
        f"{kind}[{i}]"
        for kind in ("buffers", "images")
        for i, item in enumerate(gltf.get(kind, []))
        if "uri" in item and not str(item["uri"]).lower().startswith("data:")
    ]
    if outside:
        raise ValueError(
            f"{path.name} points outside itself ({', '.join(outside)} name a file "
            "beside it), so it would not travel as one file. Export it as glTF Binary "
            "with everything embedded."
        )
    compressed = sorted(
        COMPRESSION_EXTENSIONS
        & set(gltf.get("extensionsUsed", []) + gltf.get("extensionsRequired", []))
    )
    if compressed:
        raise ValueError(
            f"{path.name} uses {', '.join(compressed)}, which the viewer cannot decode. "
            "Export it again without compression (in Blender, turn off Compression "
            "under Data and use PNG or JPEG images)."
        )

    nodes = gltf.get("nodes", [])
    parents: list[int | None] = [None] * len(nodes)
    for i, node in enumerate(nodes):
        for child in node.get("children", []):
            parents[child] = i
    world: list[np.ndarray | None] = [None] * len(nodes)

    def matrix(i: int) -> np.ndarray:
        found = world[i]
        if found is None:
            local = _local_matrix(nodes[i])
            parent = parents[i]
            found = local if parent is None else matrix(parent) @ local
            world[i] = found
        return found

    rest = np.array([matrix(i)[:3, 3] for i in range(len(nodes))]).reshape(-1, 3)
    names = tuple(node.get("name", "") for node in nodes)
    joints = frozenset(
        j for skin in gltf.get("skins", []) for j in skin.get("joints", [])
    )

    extensions = gltf.get("extensions", {})
    version: Literal["0", "1"]
    if "VRMC_vrm" in extensions:
        vrm, version = extensions["VRMC_vrm"], "1"
        bones = {
            b: v["node"]
            for b, v in vrm.get("humanoid", {}).get("humanBones", {}).items()
        }
    elif "VRM" in extensions:
        vrm, version = extensions["VRM"], "0"
        bones = {
            _VRM0_THUMBS.get(str(b["bone"]), str(b["bone"])): int(b["node"])
            for b in vrm.get("humanoid", {}).get("humanBones", [])
        }
    else:
        return SkinFile(
            path,
            "gltf",
            None,
            names,
            tuple(parents),
            rest,
            joints=joints,
            json_bytes=json_bytes,
        )
    absent = sorted(bone for bone, node in bones.items() if not 0 <= node < len(nodes))
    if absent:
        raise ValueError(
            f"{path.name}'s humanoid puts {', '.join(absent)} on nodes it does not have."
        )
    meta = vrm.get("meta")
    return SkinFile(
        path,
        "vrm",
        version,
        names,
        tuple(parents),
        rest,
        bones,
        meta if isinstance(meta, dict) else {},
        joints,
        json_bytes=json_bytes,
    )


#: VRM 0.x's thumb bones -> VRM 1.0's, as three-vrm maps them.
_VRM0_THUMBS = {
    f"{side}Thumb{old}": f"{side}Thumb{new}"
    for side in ("left", "right")
    for old, new in (
        ("Proximal", "Metacarpal"),
        ("Intermediate", "Proximal"),
        ("Distal", "Distal"),
    )
}


def _not_binary(path: Path, head: bytes) -> str:
    if path.suffix.lower() == ".gltf" or head.lstrip()[:1] == b"{":
        return (
            f"{path.name} is a .gltf, its JSON and its images and buffers in separate "
            "files. A skin is one file: export it as .glb (in Blender, glTF Binary)."
        )
    return f"{path.name} is not a binary glTF (.vrm or .glb)."


def _local_matrix(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=float).reshape(4, 4).T
    x, y, z, w = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    out = np.eye(4)
    out[:3, :3] = rotation @ np.diag(node.get("scale", [1.0, 1.0, 1.0]))
    out[:3, 3] = node.get("translation", [0.0, 0.0, 0.0])
    return out


__all__ = ["COMPRESSION_EXTENSIONS", "FACING_TURNS", "SkinFile", "read_skin_file"]
