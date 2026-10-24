"""Hand-built skin files and a robot they fit, for the skin tests.

The files are binary glTF with a JSON chunk only: mjswan reads nothing else of them.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

#: Bone, parent, offset from the parent in VRM 1.0's frame (y up, facing +z, left at +x),
#: a T-pose: hips 0.9 m up, shoulders 1.3 m, arms 0.5 m, legs 0.8 m, toes forward.
SKELETON: list[tuple[str, str | None, tuple[float, float, float]]] = [
    ("hips", None, (0.0, 0.9, 0.0)),
    ("spine", "hips", (0.0, 0.1, 0.0)),
    ("chest", "spine", (0.0, 0.15, 0.0)),
    ("neck", "chest", (0.0, 0.2, 0.0)),
    ("head", "neck", (0.0, 0.1, 0.0)),
    *(
        bone
        for side, x in (("left", 1.0), ("right", -1.0))
        for bone in (
            (f"{side}UpperArm", "chest", (0.2 * x, 0.15, 0.0)),
            (f"{side}LowerArm", f"{side}UpperArm", (0.25 * x, 0.0, 0.0)),
            (f"{side}Hand", f"{side}LowerArm", (0.25 * x, 0.0, 0.0)),
            (f"{side}UpperLeg", "hips", (0.1 * x, -0.05, 0.0)),
            (f"{side}LowerLeg", f"{side}UpperLeg", (0.0, -0.4, 0.0)),
            (f"{side}Foot", f"{side}LowerLeg", (0.0, -0.4, 0.0)),
            (f"{side}Toes", f"{side}Foot", (0.0, -0.03, 0.1)),
        )
    ),
]

#: Each bone -> the body of :func:`robot_xml`'s robot it follows.
BONES = {
    "hips": "pelvis",
    "spine": "torso",
    **{
        f"{side}{bone}": f"{s}_{body}"
        for side, s in (("left", "l"), ("right", "r"))
        for bone, body in (
            ("UpperArm", "upper"),
            ("LowerArm", "fore"),
            ("Hand", "hand"),
            ("UpperLeg", "thigh"),
            ("LowerLeg", "shin"),
            ("Foot", "foot"),
        )
    },
}

_SIDE = """
<body name="robot/{s}_upper" pos="0 {y2} 0.15">
  <joint name="{s}_shoulder_roll" axis="1 0 0"/>
  <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03"/>
  <body name="robot/{s}_fore" pos="0 0 -0.25">
    <joint name="{s}_elbow" axis="0 1 0"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.25" size="0.03"/>
    <body name="robot/{s}_hand" pos="0 0 -0.25"/>
  </body>
</body>
"""
_LEG = """
<body name="robot/{s}_thigh" pos="{x} {y1} -0.05">
  <joint name="{s}_hip" axis="0 1 0"/>
  <body name="robot/{s}_shin" pos="0 0 -0.4">
    <joint name="{s}_knee" axis="0 1 0"/>
    <body name="robot/{s}_foot" pos="0 0 -0.4">
      <geom type="box" size="0.1 0.05 0.02" pos="0 0 -0.03"/>
    </body>
  </body>
</body>
"""


def robot_xml(leg_x: float = 0.0) -> str:
    """A humanoid as tall as :data:`SKELETON`, arms hanging, soles on z = 0 at the zero
    pose. *leg_x* moves its hips that far forward of the pelvis origin. Keyframe
    ``lifted`` holds it 0.1 m off the floor; ``bow`` tips the pelvis 30 degrees."""
    sides = [("l", 0.1, 0.2), ("r", -0.1, -0.2)]
    torso = "".join(_SIDE.format(s=s, y2=y2) for s, _, y2 in sides)
    legs = "".join(_LEG.format(s=s, x=leg_x, y1=y1) for s, y1, _ in sides)
    half = math.radians(30) / 2
    return f"""
<mujoco>
  <compiler boundmass="0.01" boundinertia="0.0001"/>
  <default><joint type="hinge"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="robot/pelvis" pos="0 0 0.9">
      <freejoint name="root"/>
      <geom type="sphere" size="0.08" contype="0" conaffinity="0"/>
      <body name="robot/torso" pos="0 0 0.25">
        <joint name="waist" axis="0 0 1"/>
        {torso}
      </body>
      {legs}
    </body>
  </worldbody>
  <keyframe>
    <key name="lifted" qpos="0 0 1.0 1 0 0 0 {"0 " * 9}"/>
    <key name="bow" qpos="0 0 0.9 {math.cos(half)} 0 {math.sin(half)} 0 {"0 " * 9}"/>
  </keyframe>
</mujoco>
"""


#: Holds the arms out straight, as every VRM rests.
T_POSE = {"l_shoulder_roll": math.pi / 2, "r_shoulder_roll": -math.pi / 2}


def write_glb(path: Path, gltf: dict[str, Any]) -> Path:
    text = json.dumps(gltf).encode()
    text += b" " * (-len(text) % 4)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, 20 + len(text))
        + struct.pack("<II", len(text), 0x4E4F534A)
        + text
    )
    return path


def skeleton_gltf(
    *,
    names: dict[str, str] | None = None,
    units: float = 1.0,
    facing: str = "+z",
    arms: float = 1.0,
    mirror: bool = False,
) -> dict[str, Any]:
    """:data:`SKELETON` as glTF nodes: each named by *names* (else as its bone), in
    *units* to the metre, turned to face *facing*, its forearms and hands *arms* times
    as long, *mirror* turned about as VRM 0.x faces."""
    turn = {"+z": 0.0, "+x": math.pi / 2, "-z": math.pi, "-x": -math.pi / 2}[facing]
    c, s = math.cos(turn), math.sin(turn)
    index = {bone: i for i, (bone, _, _) in enumerate(SKELETON)}
    nodes = []
    for bone, _, (x, y, z) in SKELETON:
        if bone.endswith(("LowerArm", "Hand")):
            x *= arms
        if mirror:
            x, z = -x, -z
        nodes.append(
            {
                "name": (names or {}).get(bone, bone),
                "translation": [
                    (x * c + z * s) * units,
                    y * units,
                    (z * c - x * s) * units,
                ],
                "children": [index[b] for b, parent, _ in SKELETON if parent == bone],
            }
        )
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
    }


#: VRM 1.0's meta for a file anyone may use, as the VRM Consortium's samples are.
OPEN_META = {
    "name": "Sample",
    "authors": ["mjswan"],
    "licenseUrl": "https://vrm.dev/licenses/1.0/",
    "avatarPermission": "everyone",
    "commercialUsage": "corporation",
    "creditNotation": "unnecessary",
    "allowRedistribution": True,
}


def write_vrm(
    path: Path,
    *,
    version: str = "1",
    meta: dict[str, Any] | None = None,
    arms: float = 1.0,
) -> Path:
    """:data:`SKELETON` as a VRM 1.0 or 0.x file, its humanoid every bone of it."""
    gltf = skeleton_gltf(arms=arms, mirror=version == "0")
    if version == "1":
        gltf["extensionsUsed"] = ["VRMC_vrm"]
        gltf["extensions"] = {
            "VRMC_vrm": {
                "specVersion": "1.0",
                "meta": OPEN_META if meta is None else meta,
                "humanoid": {
                    "humanBones": {
                        bone: {"node": i} for i, (bone, _, _) in enumerate(SKELETON)
                    }
                },
            }
        }
    else:
        gltf["extensionsUsed"] = ["VRM"]
        gltf["extensions"] = {
            "VRM": {
                "meta": {"title": "Sample", "author": "mjswan", "licenseName": "CC0"}
                if meta is None
                else meta,
                "humanoid": {
                    "humanBones": [
                        {"bone": bone, "node": i}
                        for i, (bone, _, _) in enumerate(SKELETON)
                    ]
                },
            }
        }
    return write_glb(path, gltf)


#: A four-legged animal's skeleton, facing +z: its hind legs and a tail hang from the
#: pelvis, its forelegs from the chest, 0.4 m ahead.
QUADRUPED: list[tuple[str, str | None, tuple[float, float, float]]] = [
    ("Pelvis", None, (0.0, 0.45, -0.2)),
    ("Spine", "Pelvis", (0.0, 0.02, 0.2)),
    ("Chest", "Spine", (0.0, 0.0, 0.2)),
    ("Neck", "Chest", (0.0, 0.1, 0.1)),
    ("Head", "Neck", (0.0, 0.05, 0.1)),
    ("Tail1", "Pelvis", (0.0, 0.05, -0.1)),
    ("Tail2", "Tail1", (0.0, 0.0, -0.1)),
    ("Tail3", "Tail2", (0.0, 0.0, -0.1)),
    *(
        bone
        for side, x in (("L", 1.0), ("R", -1.0))
        for end, parent in (("Hind", "Pelvis"), ("Fore", "Chest"))
        for bone in (
            (f"{end}Thigh.{side}", parent, (0.08 * x, -0.05, 0.0)),
            (f"{end}Shin.{side}", f"{end}Thigh.{side}", (0.0, -0.2, 0.0)),
            (f"{end}Paw.{side}", f"{end}Shin.{side}", (0.0, -0.2, 0.0)),
        )
    ),
]


def quadruped_gltf() -> dict[str, Any]:
    index = {bone: i for i, (bone, _, _) in enumerate(QUADRUPED)}
    nodes = [
        {
            "name": bone,
            "translation": list(offset),
            "children": [index[b] for b, parent, _ in QUADRUPED if parent == bone],
        }
        for bone, _, offset in QUADRUPED
    ]
    return {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "skins": [{"joints": list(range(len(nodes)))}],
    }


_QUAD_LEG = """
<body name="robot/{leg}_hip" pos="{x} {y} 0">
  <joint name="{leg}_hip_joint" axis="1 0 0"/>
  <body name="robot/{leg}_thigh" pos="0 {y} 0">
    <joint name="{leg}_thigh_joint" axis="0 1 0"/>
    <body name="robot/{leg}_calf" pos="0 0 -0.2">
      <joint name="{leg}_calf_joint" axis="0 1 0"/>
      <geom type="sphere" size="0.02" pos="0 0 -0.2"/>
    </body>
  </body>
</body>
"""

#: A Go1-like robot: a trunk, its hips 0.4 m apart front to rear, calves that end in a
#: foot sphere but no foot body; keyframe ``stand``.
QUADRUPED_XML = f"""
<mujoco>
  <compiler boundmass="0.01" boundinertia="0.0001"/>
  <default><joint type="hinge"/></default>
  <worldbody>
    <body name="robot/trunk" pos="0 0 0.42">
      <freejoint/>
      {"".join(_QUAD_LEG.format(leg=leg, x=x, y=y) for leg, x, y in (("FL", 0.2, 0.05), ("FR", 0.2, -0.05), ("RL", -0.2, 0.05), ("RR", -0.2, -0.05)))}
    </body>
  </worldbody>
  <keyframe><key name="stand" qpos="0 0 0.42 1 0 0 0 {"0 " * 12}"/></keyframe>
</mujoco>
"""
