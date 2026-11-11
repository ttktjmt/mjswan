"""MuJoCo Playground

MuJoCo Playground 1.0.1 in the browser, in two parts. `mjswan demo playground` runs it.

- **PickCube**: the SO-101 bringing a cube to a target that moves, driven by policies
  trained on Playground's own `SO101PickCube` and `SO101PickCubeVision`
  (`tasks/manipulation/pick_cube`) with `examples/train/so101_pick_cube.py`: one reads
  the cube's position, the other finds the cube through the wrist camera. Each task is
  stated as an mjlab one, whose terms compute Playground's observations.
- **Tabletop**: the scenes of its agentic manipulation suite (`tasks/agentic/tabletop_v0`),
  nine pick-and-place tasks, each laid out in front of both arms the suite drives, the
  SO-101 and the I2RT YAM. Upstream moves the arm with a vision-language agent and an IK
  controller, neither of which runs in a browser, so these scenes carry no policy: each
  arm holds its home pose, and the objects are there to be pushed and grabbed with the
  mouse.

Playground builds its scenes with MuJoCo 4 and imports PyTorch to do it, while mjswan
runs MuJoCo 3.11, so the recipe is restated here rather than imported: the arms from MuJoCo
Menagerie with the joint dynamics Playground identified on the real ones, the ALOHA
table, and the objects from aloha_sim, each fetched at a pinned commit and cached. The
objects are scattered as Playground's reset scatters them, with a fixed seed per scene.

Adapted from MuJoCo Playground (https://github.com/google-deepmind/mujoco_playground),
Copyright 2026 Google LLC, Apache License 2.0.
"""

from __future__ import annotations

import os
import posixpath
import shutil
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as lift_mdp
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.viewer import ViewerConfig

import mjswan
from mjswan.source import hf

#: Every file comes from one of these, at the commit given.
MENAGERIE = (
    "google-deepmind/mujoco_menagerie",
    "370c43965323c0944b433b202aac5058b5f907a9",
)
ALOHA_SIM = ("google-deepmind/aloha_sim", "d02904607cca1bf6dfb72f30b522506ac7ca0f91")

CACHE_DIR = (
    Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "mjswan" / "playground"
)


# Fetching


def _fetch(repo: tuple[str, str], path: str) -> Path:
    """One file of a GitHub repository at its pinned commit, downloaded once."""
    name, commit = repo
    target = CACHE_DIR / name.replace("/", "_") / commit / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://raw.githubusercontent.com/{name}/{commit}/{path}"
        partial = target.with_name(target.name + ".part")
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            partial.write_bytes(response.read())
        partial.replace(target)
    return target


def _fetch_mjcf(repo: tuple[str, str], path: str) -> Path:
    """An MJCF and every mesh and texture file it names."""
    xml_path = _fetch(repo, path)
    root = ET.parse(xml_path).getroot()
    compiler = {k: v for c in root.iter("compiler") for k, v in c.attrib.items()}
    asset_dir = compiler.get("assetdir", "")
    dirs = {
        "mesh": compiler.get("meshdir", asset_dir),
        "texture": compiler.get("texturedir", asset_dir),
    }
    files = [
        posixpath.normpath(posixpath.join(posixpath.dirname(path), directory, file))
        for tag, directory in dirs.items()
        for element in root.iter(tag)
        if (file := element.get("file"))
    ]
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda file: _fetch(repo, file), files))
    return xml_path


def _pin_asset_paths(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Make every mesh and texture path absolute.

    Once attached to another spec, an asset keeps only its file name, and the build
    could no longer find the file it names.
    """
    base = spec.modelfiledir or ""
    for mesh in spec.meshes:
        if mesh.file:
            mesh.file = os.path.abspath(os.path.join(base, spec.meshdir, mesh.file))
    for texture in spec.textures:
        if texture.file:
            texture.file = os.path.abspath(
                os.path.join(base, spec.texturedir, texture.file)
            )
    return spec


def _load(repo: tuple[str, str], path: str) -> mujoco.MjSpec:
    return _pin_asset_paths(mujoco.MjSpec.from_file(str(_fetch_mjcf(repo, path))))


def _name_unnamed_geoms(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Name every geom, so that attaching gives each the prefix."""
    for body in spec.bodies:
        for i, geom in enumerate(body.geoms):
            if not geom.name:
                geom.name = f"{body.name}_geom_{i}"
    return spec


# Arms: Playground's `robots/*/sim` and `tasks/agentic/tabletop_v0/embodiment.py`

#: Per joint: kp, kd, armature, frictionloss, identified from real teleop data.
SO101_DYNAMICS = {
    "shoulder_pan": (251.5896, 17.5096, 0.1088, 0.0001),
    "shoulder_lift": (265.0958, 28.0620, 0.1472, 0.0),
    "elbow_flex": (243.9844, 25.6669, 0.1224, 0.0),
    "wrist_flex": (223.2804, 25.4543, 0.1133, 0.0),
    "wrist_roll": (218.4066, 23.1058, 0.1146, 0.0),
    "gripper": (223.3508, 34.1633, 0.1985, 0.2517),
}
#: STS3215 stall torque: 30 kg cm at 12 V.
SO101_FORCE_LIMIT = 30 * 9.81 / 100

#: Per actuator: kp, kv, armature, frictionloss, identified on a YAM Pro at 50 Hz. The
#: gripper's apply to both fingers, the right one mirroring the left.
YAM_DYNAMICS = {
    "joint1": (80.0, 6.01, 0.175, 0.567),
    "joint2": (60.9, 3.82, 0.032, 1.26),
    "joint3": (37.4, 2.45, 0.032, 1.07),
    "joint4": (40.0, 2.28, 0.0018, 0.1),
    "joint5": (7.34, 1.21, 0.00959, 0.0805),
    "joint6": (10.0, 1.5, 0.0018, 0.1),
    "gripper": (2000.0, 150.0, 0.1, 0.1),
}
YAM_FINGERS = ("left_finger", "right_finger")


def _so101() -> mujoco.MjSpec:
    spec = _name_unnamed_geoms(_load(MENAGERIE, "robotstudio_so101/so101.xml"))
    for joint, (kp, kd, armature, frictionloss) in SO101_DYNAMICS.items():
        actuator = spec.actuator(joint)
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kd
        actuator.forcelimited = True
        actuator.forcerange = [-SO101_FORCE_LIMIT, SO101_FORCE_LIMIT]
        spec.joint(joint).armature = armature
        spec.joint(joint).frictionloss = frictionloss
    return spec


def _yam() -> mujoco.MjSpec:
    spec = _name_unnamed_geoms(_load(MENAGERIE, "i2rt_yam/yam_linear_4310.xml"))
    for name, (kp, kv, armature, frictionloss) in YAM_DYNAMICS.items():
        actuator = spec.actuator(name)
        actuator.gainprm[0] = kp
        actuator.biasprm[1] = -kp
        actuator.biasprm[2] = -kv
        for joint in YAM_FINGERS if name == "gripper" else (name,):
            spec.joint(joint).armature = armature
            spec.joint(joint).frictionloss = frictionloss
    # `arm` has its origin 1 cm above the bottom of the base plate.
    spec.worldbody.first_body().pos = [0.0, 0.0, 0.01]
    # Menagerie's D405 is 16:9; Playground crops it to the real 4:3 stream.
    camera = spec.camera("wrist_cam")
    camera.sensor_size[0] = camera.sensor_size[1] * 320.0 / 240.0
    while spec.keys:
        spec.delete(spec.keys[0])
    return spec


@dataclass(frozen=True)
class Arm:
    name: str
    spec_fn: Callable[[], mujoco.MjSpec]
    #: Its Menagerie directory, whose license the scene carries.
    menagerie_dir: str
    #: MJCF joint → its position in the arm's home pose.
    home: dict[str, float]
    #: Objects scatter in this rectangle, within `max_reach` of `base_xy`.
    spawn_x: tuple[float, float]
    spawn_y: tuple[float, float]
    base_xy: tuple[float, float]
    max_reach: float
    #: lookat, distance, azimuth, elevation of upstream's overview camera.
    overview: tuple[tuple[float, float, float], float, float, float]
    #: The viewer starts this much further back, so the whole arm is in frame.
    zoom_out: float = 1.0


ARMS = (
    Arm(
        name="SO-101",
        spec_fn=_so101,
        menagerie_dir="robotstudio_so101",
        home={
            "shoulder_pan": -0.192,
            "shoulder_lift": -0.262,
            "elbow_flex": 0.253,
            "wrist_flex": 1.13,
            "wrist_roll": 0.0793,
            "gripper": 0.80,
        },
        spawn_x=(0.16, 0.27),
        spawn_y=(-0.13, 0.13),
        base_xy=(0.04, 0.0),
        max_reach=0.24,
        overview=((0.23, 0.0, 0.05), 0.46, 138.0, -26.0),
        zoom_out=1.3,
    ),
    Arm(
        name="YAM",
        spec_fn=_yam,
        menagerie_dir="i2rt_yam",
        home={
            "joint1": 0.0,
            "joint2": 1.012,
            "joint3": 1.498,
            "joint4": -1.487,
            "joint5": 0.0,
            "joint6": 0.0,
            "left_finger": 0.0475,
            "right_finger": 0.0475,
        },
        spawn_x=(0.24, 0.40),
        spawn_y=(-0.14, 0.14),
        base_xy=(0.0, 0.0),
        max_reach=0.44,
        overview=((0.30, 0.0, 0.08), 0.85, 138.0, -26.0),
    ),
)


# Objects: Playground's `tasks/agentic/tabletop_v0/task.py`

RED = (0.85, 0.15, 0.15, 1.0)
GREEN = (0.15, 0.75, 0.2, 1.0)
BLUE = (0.15, 0.35, 0.9, 1.0)
YELLOW = (0.95, 0.85, 0.15, 1.0)
WHITE = (0.9, 0.9, 0.9, 1.0)
ALOHA_ASSETS = "aloha_sim/assets"
MARKER_XML = f"{ALOHA_ASSETS}/mujoco/Marker/marker.xml"


@dataclass(frozen=True)
class Obj:
    """A free object: a colored box, a mesh model from aloha_sim, or a custom spec."""

    name: str
    rgba: tuple[float, float, float, float] = WHITE
    half_size: tuple[float, float, float] | None = None
    asset: str | None = None
    scale: float = 1.0
    #: Things go into it; never turned at reset.
    container: bool = False
    spec_fn: Callable[[str], mujoco.MjSpec] | None = None

    def spec(self) -> mujoco.MjSpec:
        """The object with its body frame at its bounding-box center."""
        if self.spec_fn is not None:
            return _center(self.spec_fn(self.name))
        if self.asset is not None:
            return _center(_raw_asset_spec(self.name, self.asset, self.scale))
        assert self.half_size is not None
        spec = mujoco.MjSpec()
        spec.modelname = self.name
        body = spec.worldbody.add_body(name=self.name)
        body.add_freejoint(name=f"{self.name}_freejoint")
        body.add_geom(
            name=self.name,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=list(self.half_size),
            density=1000.0,
            condim=3,
            friction=[1.0, 0.005, 0.0001],
            rgba=list(self.rgba),
            solref=[0.01, 1],
        )
        return spec


def _raw_asset_spec(name: str, path: str, scale: float) -> mujoco.MjSpec:
    """A mesh model, scaled, with its body, joint and geoms named."""
    spec = _load(ALOHA_SIM, path)
    spec.modelname = name
    body = spec.worldbody.first_body()
    body.name = name
    body.pos = [0.0, 0.0, 0.0]
    body.quat = [1.0, 0.0, 0.0, 0.0]
    if (joint := body.first_joint()) is not None:
        joint.name = f"{name}_freejoint"
    for mesh in spec.meshes:
        mesh.scale = [s * scale for s in mesh.scale]
    for i, geom in enumerate(body.geoms):
        if not geom.name:
            geom.name = f"{name}_geom_{i}"
        if geom.contype or geom.conaffinity:
            geom.contype = geom.conaffinity = 1
    return spec


def _bounds(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Center and half extents of the visible geoms' bounding box."""
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(model.ngeom):
        if model.geom_group[g] > 3:
            continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[g])
        corners = model.geom_aabb[g, :3] + signs * model.geom_aabb[g, 3:]
        corners = model.geom_pos[g] + corners @ rot.reshape(3, 3).T
        lo, hi = np.minimum(lo, corners.min(0)), np.maximum(hi, corners.max(0))
    return (lo + hi) / 2, (hi - lo) / 2


def _center(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    center, _ = _bounds(spec.compile())
    body = spec.worldbody.first_body()
    for element in (*body.geoms, *body.sites):
        element.pos = list(np.asarray(element.pos) - center)
    return spec


def _socket(name: str, *, vertical: bool = False) -> mujoco.MjSpec:
    """ALOHA's four-wall peg socket: a 36 mm tunnel along x, its white collar at -x."""
    spec = mujoco.MjSpec()
    spec.modelname = name
    body = spec.worldbody.add_body(name=name)
    body.add_freejoint(name=f"{name}_freejoint")
    parts = (  # name, half size, position, color, mass
        ("B", [0.046, 0.022, 0.002], [0.002, 0.0, -0.020], BLUE, 0.25),
        ("T", [0.046, 0.022, 0.002], [0.002, 0.0, 0.020], BLUE, 0.016),
        ("L", [0.046, 0.002, 0.018], [0.002, -0.020, 0.0], BLUE, 0.016),
        ("R", [0.046, 0.002, 0.018], [0.002, 0.020, 0.0], BLUE, 0.016),
        ("rim_B", [0.002, 0.022, 0.002], [-0.046, 0.0, -0.020], WHITE, 0.004),
        ("rim_T", [0.002, 0.022, 0.002], [-0.046, 0.0, 0.020], WHITE, 0.004),
        ("rim_L", [0.002, 0.002, 0.018], [-0.046, -0.020, 0.0], WHITE, 0.004),
        ("rim_R", [0.002, 0.002, 0.018], [-0.046, 0.020, 0.0], WHITE, 0.004),
        (
            "wall",
            [0.004, 0.018, 0.018],
            [0.044, 0.0, 0.0],
            RED,
            0.25 if vertical else 0.016,
        ),
    )
    for suffix, size, pos, rgba, mass in parts:
        if vertical:  # The entrance faces +z, the stop wall becomes the floor.
            size = [size[2], size[1], size[0]]
            pos = [pos[2], pos[1], -pos[0]]
        body.add_geom(
            name=f"{name}_{suffix}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size,
            pos=pos,
            rgba=list(rgba),
            mass=mass,
            condim=4,
            friction=[1.0, 0.05, 0.001],
        )
    return spec


def _marker_stand(name: str) -> mujoco.MjSpec:
    """ALOHA's marker held upright in a base stand."""
    spec = _raw_asset_spec(name, MARKER_XML, scale=0.85)
    spec.worldbody.first_body().add_geom(
        name=f"{name}_base",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.035, 0.005, 0.0],
        pos=[0.0, 0.0, 0.005],
        rgba=list(WHITE),
        mass=0.5,
        friction=[1.0, 0.05, 0.001],
    )
    return spec


def _ycb(name: str, model: str, scale: float, *, container: bool = False) -> Obj:
    path = f"{ALOHA_ASSETS}/ycb/{model}/google_64k/model.xml"
    return Obj(name, asset=path, scale=scale, container=container)


def _cube(name: str, rgba: tuple[float, float, float, float]) -> Obj:
    return Obj(name, rgba, half_size=(0.015,) * 3)


BOWL = _ycb("bowl", "024_bowl", 0.7, container=True)
PLATE = _ycb("plate", "029_plate", 0.6, container=True)
BANANA = _ycb("banana", "011_banana", 0.6)
MARKER = Obj("marker", asset=MARKER_XML, scale=0.8)
CUBES = (_cube("red", RED), _cube("green", GREEN), _cube("blue", BLUE))
PEG = Obj("peg", RED, half_size=(0.010, 0.010, 0.048))
SOCKET = Obj("socket", container=True, spec_fn=lambda n: _socket(n, vertical=True))
CAP = Obj("cap", asset=f"{ALOHA_ASSETS}/mujoco/Marker/cap.xml", scale=0.85)
MARKER_STAND = Obj("marker_stand", container=True, spec_fn=_marker_stand)
USB = Obj(
    "usb",
    asset=f"{ALOHA_ASSETS}/gso/Kingston_DT4000MR_G2_Management_Ready_USB_64GB/model.xml",
    scale=0.9,
)
USB_PORT = Obj("usb_port", container=True, spec_fn=_socket)

#: Scene name, upstream's task name, its prompt, and its objects.
TASKS = (
    (
        "Cube in Bowl",
        "cube_in_bowl",
        "Put the green cube in the bowl.",
        (CUBES[1], BOWL),
    ),
    (
        "Bar in Bowl",
        "bar_in_bowl",
        "Put the yellow bar in the bowl.",
        (Obj("bar", YELLOW, half_size=(0.035, 0.012, 0.014)), BOWL),
    ),
    (
        "Cubes in Bowl",
        "cubes_in_bowl",
        "Put all three cubes in the bowl.",
        (*CUBES, BOWL),
    ),
    ("Banana in Bowl", "banana_in_bowl", "Put the banana in the bowl.", (BANANA, BOWL)),
    (
        "Marker on Plate",
        "marker_on_plate",
        "Put the marker on the plate.",
        (MARKER, PLATE),
    ),
    (
        "Stack Three",
        "stack3",
        "Stack the green cube on the red cube, then the blue cube on top.",
        CUBES,
    ),
    (
        "Peg in Socket",
        "peg_in_socket",
        "Insert the red peg into the blue socket.",
        (PEG, SOCKET),
    ),
    (
        "Cap the Marker",
        "cap_fixed_marker",
        "Place the marker cap onto the upright marker.",
        (CAP, MARKER_STAND),
    ),
    (
        "USB in Port",
        "usb_in_port",
        "Plug the USB thumb drive into the horizontal socket.",
        (USB, USB_PORT),
    ),
)


# Scene: Playground's `_base_scene` and `env_config`

TABLE_HEIGHT = 0.75

#: Playground's `_src/objects/assets/primitive/plane.xml`.
PLANE_XML = """
<mujoco model="plane">
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10">
    <flag eulerdamp="disable"/>
  </option>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
      height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
      reflectance="0.2"/>
  </asset>
  <worldbody>
    <light name="light" pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <light name="front_fill" pos="0 -1.5 1.5" dir="0 1 -1" directional="true"
      diffuse="0.25 0.25 0.25" specular="0 0 0" castshadow="false"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="1"/>
  </worldbody>
</mujoco>
"""

#: Playground's `_src/objects/furniture.py`, with Menagerie's ALOHA table meshes.
ALOHA_TABLE_XML = """
<mujoco model="aloha_table">
  <compiler meshdir="{assets}" texturedir="{assets}"/>
  <asset>
    <mesh name="tabletop" file="tabletop.obj"/>
    <mesh name="tablelegs" file="tablelegs.obj"/>
    <texture name="small_meta_table_diffuse" type="2d" file="small_meta_table_diffuse.png"/>
    <material name="table" texture="small_meta_table_diffuse"/>
  </asset>
  <worldbody>
    <body name="aloha_table" pos="0 0 0.75">
      <geom name="tabletop" type="mesh" mesh="tabletop" material="table" pos="0 0 -0.75"
        quat="1 0 0 1" contype="0" conaffinity="0" group="2"/>
      <geom name="tablelegs" type="mesh" mesh="tablelegs" material="table" pos="0 0 -0.75"
        quat="1 0 0 1" contype="0" conaffinity="0" group="2"/>
      <geom name="table_collision" type="box" size="0.61 0.37 0.1" pos="0 0 -0.1" group="3"
        condim="3" friction="1 0.005 0.0001" solref="0.01 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _aloha_table() -> mujoco.MjSpec:
    files = ("tabletop.obj", "tablelegs.obj", "small_meta_table_diffuse.png")
    paths = [_fetch(MENAGERIE, f"aloha/assets/{file}") for file in files]
    xml = ALOHA_TABLE_XML.format(assets=paths[0].parent)
    return _pin_asset_paths(mujoco.MjSpec.from_string(xml))


def _base_scene(arm: Arm) -> mujoco.MjSpec:
    """The floor, the table with its top at z = 0, and the lights."""
    scene = mujoco.MjSpec.from_string(PLANE_XML)
    scene.option.ccd_iterations = 300
    scene.geom("floor").pos = [0.0, 0.0, -TABLE_HEIGHT]
    table_frame = scene.worldbody.add_frame(
        pos=[0.27, 0.0, -TABLE_HEIGHT], quat=[np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]
    )
    scene.attach(_aloha_table(), prefix="table/", frame=table_frame)
    scene.stat.extent = 4.0
    _workspace_lights(scene, float(np.mean(arm.spawn_x)), room=0.35)
    return scene


def _workspace_lights(scene: mujoco.MjSpec, center_x: float, *, room: float) -> None:
    """The plane's key light aimed at the workspace, casting shadows, and two room
    lights beside it of brightness `room`."""
    center = scene.worldbody.add_body(name="workspace_center", pos=[center_x, 0.0, 0.0])
    key = scene.light("light")
    key.pos = [0.2, 0.0, 1.4]
    key.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
    key.targetbody = center.name
    key.diffuse = [0.6, 0.6, 0.6]
    key.castshadow = True
    for name, x in (("left_room_light", -0.7), ("right_room_light", 0.7)):
        light = scene.worldbody.add_light(name=name, pos=[x, 0.1, 1.2])
        light.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        light.targetbody = center.name
        light.diffuse = [room, room, room]
        light.specular = [0.05, 0.05, 0.05]
        light.castshadow = False


def _scatter(
    arm: Arm, footprints: dict[str, tuple[float, bool]], rng: np.random.Generator
) -> dict[str, tuple[float, float, float]]:
    """x, y and yaw per object, as Playground's `ScatterObjectsResetter` draws them.

    ``footprints`` maps each object to its footprint radius and whether to turn it.
    Objects never overlap, and land within the arm's top-down reach.
    """

    def draw() -> np.ndarray:
        while True:
            xy = rng.uniform(
                (arm.spawn_x[0], arm.spawn_y[0]), (arm.spawn_x[1], arm.spawn_y[1])
            )
            if np.linalg.norm(xy - arm.base_xy) <= arm.max_reach:
                return xy

    placed: dict[str, tuple[float, float, float]] = {}
    margin = 0.02
    # Largest first, so the small ones fit around it.
    for name, (radius, turn) in sorted(footprints.items(), key=lambda o: -o[1][0]):
        for _ in range(100):
            xy = draw()
            if all(
                np.linalg.norm(xy - placed[other][:2])
                >= radius + footprints[other][0] + margin
                for other in placed
            ):
                break
        placed[name] = (
            float(xy[0]),
            float(xy[1]),
            float(rng.uniform(-np.pi, np.pi)) if turn else 0.0,
        )
    return placed


def build_scene(arm: Arm, objects: tuple[Obj, ...], seed: int) -> mujoco.MjSpec:
    """One task's scene, opening on the arm's home pose with the objects scattered."""
    scene = _base_scene(arm)
    scene.attach(arm.spec_fn(), prefix="robot/", frame=scene.worldbody.add_frame())

    specs = {obj.name: obj.spec() for obj in objects}
    halves = {name: _bounds(spec.compile())[1] for name, spec in specs.items()}
    footprints = {
        obj.name: (float(max(halves[obj.name][:2])), not obj.container)
        for obj in objects
    }
    poses = _scatter(arm, footprints, np.random.default_rng(seed))
    for obj in objects:
        x, y, yaw = poses[obj.name]
        frame = scene.worldbody.add_frame(
            pos=[x, y, float(halves[obj.name][2])],
            quat=[np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)],
        )
        scene.attach(specs[obj.name], prefix=f"{obj.name}/", frame=frame)

    # The browser resets to the first keyframe: the arm at home, holding it.
    model = scene.compile()
    data = mujoco.MjData(model)
    for joint, value in arm.home.items():
        data.qpos[model.joint(f"robot/{joint}").qposadr[0]] = value
    for i in range(model.nu):
        joint = model.actuator_trnid[i, 0]
        data.ctrl[i] = data.qpos[model.jnt_qposadr[joint]]
    scene.add_key(name="home", qpos=data.qpos.tolist(), ctrl=data.ctrl.tolist())
    return scene


# PickCube: Playground's `tasks/manipulation/pick_cube`, as an mjlab task

#: The policy `examples/train/so101_pick_cube.py` trained, at the commit this port was
#: checked against.
PICK_CUBE_POLICY = (
    "ttktjmt/so101-pick-cube",
    "7dfa8ca85964949892ef97427f29c5e0dda0735d",
)

#: Playground's `TransformActions` scale: 12.5 times the torque limit over each joint's
#: stiffness, 25 times for the gripper.
SO101_ACTION_SCALE = {
    joint: (25.0 if joint == "gripper" else 12.5) * SO101_FORCE_LIMIT / kp
    for joint, (kp, *_) in SO101_DYNAMICS.items()
}


def _pick_cube_cube() -> mujoco.MjSpec:
    """Playground's `primitive.cube`: 3 cm, 27 g, on a free joint."""
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="cube")
    body.add_freejoint(name="cube_freejoint")
    body.add_geom(
        name="cube",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        density=1000.0,
        condim=3,
        friction=[1.0, 0.005, 0.0001],
        rgba=[0.0, 1.0, 0.0, 1.0],
        contype=2,
        conaffinity=1,
        solref=[0.01, 1.0],
    )
    return spec


def _pick_cube_world(spec: mujoco.MjSpec) -> None:
    """Playground's plane around mjlab's scene."""
    spec.attach(
        mujoco.MjSpec.from_string(PLANE_XML),
        prefix="",
        frame=spec.worldbody.add_frame(),
    )
    spec.stat.center = [0.25, 0.0, 0.15]


def pick_cube_env_cfg() -> ManagerBasedRlEnvCfg:
    """Playground's `SO101PickCube` as mjlab's lift task states it.

    The observations are mjlab's lift-cube terms, which compute Playground's: the arm's
    base is the world frame here. Playground reads the cube and gripper positions from
    `mj_step`'s kinematics, one physics step older than these, up to 4 mm apart on its
    reference rollout, which the policy does not notice. The target moves every 4 to 7 s
    as Playground's resetter moves it, without teleporting the cube.
    """
    so101 = ARMS[0]
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(
            entities={
                "robot": EntityCfg(
                    spec_fn=_so101,
                    init_state=EntityCfg.InitialStateCfg(joint_pos=dict(so101.home)),
                    # `_so101`'s own actuators: Playground's servo model.
                    articulation=EntityArticulationInfoCfg(
                        actuators=(XmlActuatorCfg(target_names_expr=tuple(so101.home)),)
                    ),
                ),
                "cube": EntityCfg(
                    spec_fn=_pick_cube_cube,
                    init_state=EntityCfg.InitialStateCfg(pos=(0.275, 0.0, 0.015)),
                ),
            },
            extent=0.6,
            spec_fn=_pick_cube_world,
        ),
        observations={
            "actor": ObservationGroupCfg(
                {
                    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                    "ee_to_object": ObservationTermCfg(
                        func=lift_mdp.ee_to_object_distance,
                        params={
                            "object_name": "cube",
                            "asset_cfg": SceneEntityCfg(
                                "robot", site_names=("gripperframe",)
                            ),
                        },
                    ),
                    "object_to_goal": ObservationTermCfg(
                        func=lift_mdp.object_to_goal_distance,
                        params={"object_name": "cube", "command_name": "target"},
                    ),
                    "last_action": ObservationTermCfg(func=mdp.last_action),
                }
            )
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=SO101_ACTION_SCALE,
                use_default_offset=True,
            )
        },
        commands={
            "target": LiftingCommandCfg(
                entity_name="cube",
                resampling_time_range=(4.0, 7.0),
                difficulty="dynamic",
                target_position_range=LiftingCommandCfg.TargetPositionRangeCfg(
                    x=(0.20, 0.30), y=(-0.1, 0.1), z=(0.08, 0.2)
                ),
                object_pose_range=None,
                debug_vis=True,
                viz=LiftingCommandCfg.VizCfg(target_color=(1.0, 0.0, 0.0, 0.3)),
            )
        },
        events={
            "reset_robot_joints": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={
                    "position_range": (-0.1, 0.1),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
                },
            ),
            "reset_cube": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.075, 0.075),
                        "y": (-0.1, 0.1),
                        "yaw": (-np.pi, np.pi),
                    },
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("cube"),
                },
            ),
        },
        viewer=ViewerConfig(
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.2, 0.0, 0.1),
            distance=0.8,
            azimuth=138.0,
            elevation=-26.0,
        ),
        sim=SimulationCfg(
            mujoco=MujocoCfg(timestep=0.005, impratio=10, cone="elliptic")
        ),
        decimation=4,
    )


def _target_viz(cfg: LiftingCommandCfg) -> list[dict[str, Any]]:
    """The target, a sphere around where the cube's centre should go."""
    return [
        {
            "shape": "sphere",
            "radius": 0.02,
            "color": list(cfg.viz.target_color),
            "origin": {"state": "target_pos"},
        }
    ]


# mjswan leaves a command class that one task family uses to the port to register.
mjswan.register_command(
    "LiftingCommandCfg",
    mjswan.CommandBinding(
        state_fields=["target_pos"], command_field="target_pos", viz=_target_viz
    ),
)


def _hub_policy(
    source: tuple[str, str], files: tuple[str, ...] = ("policy.onnx",)
) -> onnx.ModelProto:
    """An exported actor, with any weights torch wrote beside it read in.

    Copied out of the Hub's cache first: `onnx.load` will not read weights through its
    symbolic links.
    """
    repo, revision = source
    with tempfile.TemporaryDirectory() as tmp:
        for name in files:
            shutil.copyfile(
                hf.fetch_file(repo, name, revision=revision), Path(tmp, name)
            )
        return onnx.load(str(Path(tmp, files[0])))


# PickCube Vision: Playground's `SO101PickCubeVision`, the wrist camera for the cube

#: The policy `so101_pick_cube.py --task vision` trained: 700 of its 3000 iterations.
PICK_CUBE_VISION_POLICY = ("ttktjmt/so101-pick-cube-vision", "REVISION")

#: Playground's `DeltaActions` for the SO-101: radians per unit of action, and the range
#: each joint's target is held in.
SO101_DELTA_SCALE = {**{joint: 0.05 for joint in SO101_DYNAMICS}, "gripper": 0.25}
SO101_DELTA_LIMITS = {
    "shoulder_pan": (-3.14159, 3.14159),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-0.17453, 2.79253),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.79253, 2.79253),
    "gripper": (-0.174533, 1.7453292),
}


@dataclass(kw_only=True)
class DeltaJointPositionActionCfg(JointPositionActionCfg):
    """Playground's `DeltaActions`: each action, clipped to ±1 and scaled, moves the
    joint targets on from where they were, smoothed with the step before. The targets
    stay within `limits` and start each episode at the default pose."""

    smoothing: float = 0.5
    limits: dict[str, tuple[float, float]] = field(default_factory=dict)

    def build(self, env: Any) -> DeltaJointPositionAction:
        return DeltaJointPositionAction(self, env)


class DeltaJointPositionAction(JointPositionAction):
    cfg: DeltaJointPositionActionCfg

    def __init__(self, cfg: DeltaJointPositionActionCfg, env: Any):
        super().__init__(cfg, env)
        bounds = torch.tensor(
            [cfg.limits[name] for name in self.target_names], device=self.device
        )
        self._low, self._high = bounds[:, 0], bounds[:, 1]
        self._target = self._offset.clone()
        self._step = torch.zeros_like(self._raw_actions)

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        step = actions.clamp(-1.0, 1.0) * self._scale
        smoothing = self.cfg.smoothing
        self._step = (1.0 - smoothing) * step + smoothing * self._step
        self._target = torch.clamp(self._target + self._step, self._low, self._high)
        self._processed_actions = self._target.clone()

    def reset(self, env_ids: Any = None) -> None:
        super().reset(env_ids)
        self._target[env_ids] = self._offset[env_ids]
        self._step[env_ids] = 0.0


def _site_pos(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """A site's position in the world: Playground's `ee/xpos`."""
    return env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)


def _site_xmat(env: Any, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """A site's orientation, its rotation matrix row by row: Playground's `ee/xmat`."""
    quat = env.scene[asset_cfg.name].data.site_quat_w[:, asset_cfg.site_ids].squeeze(1)
    return matrix_from_quat(quat).reshape(-1, 9)


def _wrist_image(env: Any, sensor_name: str, size: int) -> torch.Tensor:
    """Playground's `obs_camera`: the camera's square frame, averaged down to `size`."""
    rgb = lift_mdp.camera_rgb(env, sensor_name)
    return torch.nn.functional.interpolate(rgb, size=(size, size), mode="area")


def _pick_cube_vision_world(spec: mujoco.MjSpec) -> None:
    """Playground's vision scene: a dim headlight, shadows from the key light and the
    front fill, room lights, and the arm, the cube and the floor drawn in flat colour
    (`override_flat_materials`), the floor the SO-101's grey."""
    _pick_cube_world(spec)
    _workspace_lights(spec, 0.25, room=0.25)
    spec.visual.headlight.diffuse = [0.1, 0.1, 0.1]
    spec.visual.headlight.ambient = [0.15, 0.15, 0.15]
    fill = spec.light("front_fill")
    fill.pos = [0.1, -1.0, 1.2]
    fill.diffuse = [0.35, 0.35, 0.35]
    fill.castshadow = True
    spec.geom("floor").rgba = [0.25, 0.25, 0.25, 1.0]
    for geom in spec.geoms:
        if geom.name == "floor" or geom.name.startswith(("robot/", "cube/")):
            geom.material = ""


def pick_cube_vision_env_cfg() -> ManagerBasedRlEnvCfg:
    """Playground's `SO101PickCubeVision`, the task its sim-to-real demo deploys.

    The policy sees the wrist camera's 64 x 64 frame averaged to 16 x 16, beside the
    joints, the gripper's pose and the target in the gripper's frame, and moves the arm
    by deltas at 10 Hz. The rest is `pick_cube_env_cfg`. three.js draws the frame, not
    Playground's ray tracer, so its shading differs from what the policy trained on.
    """
    cfg = pick_cube_env_cfg()
    gripper = SceneEntityCfg("robot", site_names=("gripperframe",))
    cfg.scene.spec_fn = _pick_cube_vision_world
    cfg.scene.sensors = (
        CameraSensorCfg(
            name="wrist_cam",
            camera_name="robot/wrist_cam",
            width=64,
            height=64,
            data_types=("rgb",),
            enabled_geom_groups=(0, 2),
            use_shadows=True,
        ),
    )
    cfg.observations = {
        "actor": ObservationGroupCfg(
            {
                "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
                "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
                "ee_pos": ObservationTermCfg(
                    func=_site_pos, params={"asset_cfg": gripper}
                ),
                "ee_orientation": ObservationTermCfg(
                    func=_site_xmat, params={"asset_cfg": gripper}
                ),
                "goal_position": ObservationTermCfg(
                    func=lift_mdp.target_position,
                    params={"command_name": "target", "asset_cfg": gripper},
                ),
                "last_action": ObservationTermCfg(func=mdp.last_action),
            }
        ),
        "camera": ObservationGroupCfg(
            {
                "camera": ObservationTermCfg(
                    func=_wrist_image, params={"sensor_name": "wrist_cam", "size": 16}
                )
            }
        ),
    }
    cfg.actions = {
        "joint_pos": DeltaJointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=SO101_DELTA_SCALE,
            limits=SO101_DELTA_LIMITS,
        )
    }
    cfg.sim.mujoco.ccd_iterations = 300
    cfg.decimation = 20
    return cfg


# Credits


def _attribute(scene: mjswan.SceneHandle, arm: Arm, objects: tuple[Obj, ...]) -> None:
    """Every component's license file, from the commit the scene was built from.

    A scene composed in code has no model directory for the build to find them beside.
    """
    licenses = [
        ("aloha_table", MENAGERIE, "aloha/LICENSE"),
        (arm.menagerie_dir, MENAGERIE, f"{arm.menagerie_dir}/LICENSE"),
    ]
    if any(obj.asset or obj.spec_fn is _marker_stand for obj in objects):
        licenses.append(("aloha_sim", ALOHA_SIM, "LICENSE"))
    if any(obj.asset and "/ycb/" in obj.asset for obj in objects):
        licenses.append(("ycb", ALOHA_SIM, f"{ALOHA_ASSETS}/ycb/LICENSE"))
    if any(obj.asset and "/gso/" in obj.asset for obj in objects):
        licenses.append(("gso", ALOHA_SIM, f"{ALOHA_ASSETS}/gso/LICENSE"))
    for component, repo, path in licenses:
        scene.add_attribution(component, license=_fetch(repo, path))


def add_pick_cube(project: mjswan.ProjectHandle) -> mjswan.SceneHandle:
    """The SO-101 bringing the cube to a target that moves, driven by its policy."""
    scene = project.add_scene_mjlab(
        "SO101PickCube", name="SO-101", env_cfg=pick_cube_env_cfg()
    )
    # Not an mjlab export, so nothing in the file names the joints: one per actuator, in
    # order, as `_contract` in so101_pick_cube.py records them.
    scene.add_policy(
        name="PickCube",
        # Exported before the script wrote one file.
        policy=_hub_policy(PICK_CUBE_POLICY, ("policy.onnx", "policy.onnx.data")),
        policy_joint_names=scene.actuated_joint_names(),
        default_joint_pos=list(ARMS[0].home.values()),
    )
    scene.add_attribution(
        "robotstudio_so101", license=_fetch(MENAGERIE, "robotstudio_so101/LICENSE")
    )
    return scene


def add_pick_cube_vision(project: mjswan.ProjectHandle) -> mjswan.SceneHandle:
    """The same task, the arm finding the cube through its wrist camera."""
    scene = project.add_scene_mjlab(
        "SO101PickCubeVision", name="SO-101 Vision", env_cfg=pick_cube_vision_env_cfg()
    )
    scene.add_policy(
        name="PickCube Vision",
        policy=_hub_policy(PICK_CUBE_VISION_POLICY),
        in_keys=["actor", "camera"],
        policy_joint_names=scene.actuated_joint_names(),
        default_joint_pos=list(ARMS[0].home.values()),
    )
    scene.add_attribution(
        "robotstudio_so101", license=_fetch(MENAGERIE, "robotstudio_so101/LICENSE")
    )
    return scene


def setup_builder() -> mjswan.Builder:
    """Return the configured builder, not yet built."""
    builder = mjswan.Builder(base_path=os.getenv("MJSWAN_BASE_PATH", "/"))
    project = builder.add_project(name="Playground PickCube")
    add_pick_cube(project)
    add_pick_cube_vision(project)
    for arm in ARMS:
        project = builder.add_project(name=f"Playground Tabletop: {arm.name}")
        lookat, distance, azimuth, elevation = arm.overview
        for seed, (title, _, _, objects) in enumerate(TASKS):
            scene = project.add_scene(
                name=title, spec=build_scene(arm, objects, seed)
            ).set_viewer(
                mjswan.ViewerConfig(
                    lookat=lookat,
                    distance=distance * arm.zoom_out,
                    elevation=elevation,
                    azimuth=azimuth,
                    origin_type=mjswan.ViewerConfig.OriginType.WORLD,
                )
            )
            _attribute(scene, arm, objects)
    return builder


def main():
    """Build every scene and open the app.

    Environment variables:
        MJSWAN_BASE_PATH: Base path for deployment (default: '/')
        MJSWAN_NO_LAUNCH: Set to '1' to skip launching the browser
    """
    app = setup_builder().build()
    if os.getenv("MJSWAN_NO_LAUNCH") != "1":
        app.launch()


if __name__ == "__main__":
    main()
