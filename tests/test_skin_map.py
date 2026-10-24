"""``mjswan skin map``: the drafted maps, checked against skeletons and robots whose
answers are known, and the draft itself, which must resolve as it is written.

Layer: L1 (no Node build, no network).
"""

from __future__ import annotations

import math
import runpy
from pathlib import Path

import mujoco
import pytest
import skin_fixture as fx
from typer.testing import CliRunner

from mjswan.cli import app
from mjswan.document.skin import read_skin_file
from mjswan.rig.guess import (
    arm_joints,
    draft_source,
    guess_bones,
    guess_humanoid,
    guess_scale,
    known_rest_pose,
)
from mjswan.skin import resolve_skin


@pytest.fixture
def humanoid_glb(tmp_path: Path) -> Path:
    return fx.write_glb(tmp_path / "doll.glb", fx.skeleton_gltf())


@pytest.fixture
def fox_glb(tmp_path: Path) -> Path:
    return fx.write_glb(tmp_path / "fox.glb", fx.quadruped_gltf())


@pytest.fixture
def humanoid_xml(tmp_path: Path) -> Path:
    path = tmp_path / "humanoid.xml"
    path.write_text(fx.robot_xml())
    return path


@pytest.fixture
def quadruped_xml(tmp_path: Path) -> Path:
    path = tmp_path / "quadruped.xml"
    path.write_text(fx.QUADRUPED_XML)
    return path


class TestGuessHumanoid:
    @pytest.mark.parametrize("facing", ["+z", "+x", "-z", "-x"])
    def test_a_humanoid_by_its_shape_whichever_way_it_faces(self, tmp_path, facing):
        names = {bone: f"node{i}" for i, (bone, _, _) in enumerate(fx.SKELETON)}
        path = fx.write_glb(
            tmp_path / "a.glb", fx.skeleton_gltf(names=names, facing=facing)
        )
        file = read_skin_file(path)
        guess = guess_humanoid(file)
        assert (guess.kind, guess.facing) == ("humanoid", facing)
        found = {bone: file.names[n] for bone, n in guess.humanoid.items()}
        # The skeleton has no shoulders or upper chest to find.
        expected = {
            bone: names[bone]
            for bone in [*fx.BONES, "neck", "head", "leftToes", "rightToes"]
        }
        assert found == {**expected, "chest": names["chest"]}

    def test_a_four_legged_animal_walks_on_its_arms(self, fox_glb):
        file = read_skin_file(fox_glb)
        guess = guess_humanoid(file)
        assert (guess.kind, guess.facing) == ("quadruped", "+z")
        found = {bone: file.names[n] for bone, n in guess.humanoid.items()}
        assert found["hips"] == "Pelvis" and found["chest"] == "Chest"
        assert (found["leftUpperArm"], found["leftHand"]) == (
            "ForeThigh.L",
            "ForePaw.L",
        )
        assert (found["rightUpperLeg"], found["rightFoot"]) == (
            "HindThigh.R",
            "HindPaw.R",
        )
        assert (found["neck"], found["head"]) == ("Neck", "Head")
        assert [file.names[n] for n in guess.springs] == ["Tail1"]

    def test_anything_else_offers_its_legs(self, tmp_path):
        # A tripod: three legs on the floor, no mirrored pair.
        nodes = [{"name": "body", "translation": [0, 0.5, 0], "children": [1, 4, 7]}]
        for i, x in enumerate((0.3, -0.15, -0.15)):
            base = len(nodes)
            nodes += [
                {
                    "name": f"hip{i}",
                    "translation": [x, 0, 0.26 * (i - 1)],
                    "children": [base + 1],
                },
                {
                    "name": f"knee{i}",
                    "translation": [0, -0.25, 0],
                    "children": [base + 2],
                },
                {"name": f"foot{i}", "translation": [0, -0.25, 0]},
            ]
        gltf = {
            "asset": {"version": "2.0"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": nodes,
        }
        file = read_skin_file(fx.write_glb(tmp_path / "tripod.glb", gltf))
        guess = guess_humanoid(file)
        assert guess.kind == "other" and not guess.humanoid
        assert [tuple(file.names[n] for n in limb) for limb in guess.limbs] == [
            (f"hip{i}", f"knee{i}", f"foot{i}") for i in range(3)
        ]


class TestGuessRobot:
    def test_a_humanoid_robot_by_its_body_names(self):
        robot = mujoco.MjModel.from_xml_string(fx.robot_xml())
        # No waist body: the spine rides the pelvis, the chest the torso.
        assert guess_bones(robot, [*fx.BONES, "chest"]) == {
            **fx.BONES,
            "spine": "pelvis",
            "chest": "torso",
        }

    def test_a_four_legged_robot_stands_its_arms_on_its_calves(self):
        robot = mujoco.MjModel.from_xml_string(fx.QUADRUPED_XML)
        wanted = [
            "hips",
            "spine",
            "leftUpperArm",
            "leftLowerArm",
            "leftHand",
            "rightUpperLeg",
            "rightFoot",
            "rightToes",
        ]
        assert guess_bones(robot, wanted) == {
            "hips": "trunk",
            "spine": "trunk",
            "leftUpperArm": "FL_thigh",
            "leftLowerArm": "FL_calf",
            "leftHand": "FL_calf",
            "rightUpperLeg": "RR_thigh",
            "rightFoot": "RR_calf",
            "rightToes": "RR_calf",
        }

    def test_a_four_legged_skin_is_sized_to_the_robot(self, fox_glb):
        robot = mujoco.MjModel.from_xml_string(fx.QUADRUPED_XML)
        file = read_skin_file(fox_glb)
        guess = guess_humanoid(file)
        scale, how = guess_scale(file, guess, robot, guess_bones(robot, guess.humanoid))
        # Fore and hind thighs 0.4 m apart in the file, as the robot's hips are.
        assert scale == pytest.approx(1.0)
        assert "0.4 m" in how

    def test_only_a_checked_robot_gets_a_rest_pose(self):
        g1_joints = """
        <mujoco><worldbody><body name="b">
          <joint name="left_shoulder_roll_joint"/><joint name="right_shoulder_roll_joint"/>
          <joint name="left_elbow_joint"/><joint name="right_elbow_joint"/>
          <geom size="0.1"/></body></worldbody></mujoco>"""
        robot, pose = known_rest_pose(mujoco.MjModel.from_xml_string(g1_joints))
        assert "G1" in robot and pose["left_shoulder_roll_joint"] == math.pi / 2
        assert known_rest_pose(mujoco.MjModel.from_xml_string(fx.robot_xml())) is None

    def test_lists_the_arm_joints_a_rest_pose_sets(self):
        robot = mujoco.MjModel.from_xml_string(fx.robot_xml())
        joints = dict(arm_joints(robot, fx.BONES))
        assert joints == {
            "l_shoulder_roll": (1.0, 0.0, 0.0),
            "l_elbow": (0.0, 1.0, 0.0),
            "r_shoulder_roll": (1.0, 0.0, 0.0),
            "r_elbow": (0.0, 1.0, 0.0),
        }


class TestDraft:
    def test_the_draft_resolves_as_written(self, tmp_path, fox_glb):
        robot = mujoco.MjModel.from_xml_string(fx.QUADRUPED_XML)
        file = read_skin_file(fox_glb)
        guess = guess_humanoid(file)
        bones = guess_bones(robot, guess.humanoid)
        source = draft_source(
            "mjswan skin map fox.glb",
            source=str(fox_glb),
            guess=guess,
            names=file.names,
            bones=bones,
            scale=1.0,
            rest_pose="stand",
        )
        path = tmp_path / "draft.py"
        path.write_text(source)
        skin = resolve_skin(runpy.run_path(str(path))["SKIN"], robot)
        assert skin.rig == "humanoid" and len(skin.limbs) == 4
        assert [chain["joints"] for chain in skin.springs] == [
            [file.node(n) for n in ("Tail1", "Tail2", "Tail3")]
        ]

    def test_the_cli_prints_the_maps_and_writes_the_draft(
        self, tmp_path, fox_glb, quadruped_xml
    ):
        out = tmp_path / "fox_skin.py"
        result = CliRunner().invoke(
            app,
            [
                "skin",
                "map",
                str(fox_glb),
                "--robot",
                str(quadruped_xml),
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "quadruped" in result.output and "front legs as arms" in result.output
        assert '"stand"' in result.output and "Tail1" in result.output
        config = runpy.run_path(str(out))["SKIN"]
        assert config.rest_pose == "stand" and config.bones["leftHand"] == "FL_calf"

    def test_a_vrm_names_its_own_bones(self, tmp_path, humanoid_xml):
        vrm = fx.write_vrm(tmp_path / "a.vrm")
        out = tmp_path / "a_skin.py"
        result = CliRunner().invoke(
            app, ["skin", "map", str(vrm), "--robot", str(humanoid_xml), "-o", str(out)]
        )
        assert result.exit_code == 0, result.output
        assert "rest_pose" in result.output and "l_shoulder_roll" in result.output
        config = runpy.run_path(str(out))["SKIN"]
        assert config.humanoid is None and config.rest_pose is None
        assert {**fx.BONES, "spine": "pelvis"}.items() <= config.bones.items()
        resolve_skin(config, mujoco.MjModel.from_xml_string(fx.robot_xml()))

    def test_the_cli_refuses_a_file_it_cannot_read(self, tmp_path):
        bad = tmp_path / "a.gltf"
        bad.write_text("{}")
        result = CliRunner().invoke(app, ["skin", "map", str(bad)])
        assert result.exit_code == 1
        assert "glTF Binary" in result.output
