from __future__ import annotations

import mujoco
import pytest

from mjswan import Builder, SliderConfig, ui_command
from mjswan.envs.mdp.actions import (
    CommandJointPositionActionCfg,
    JointPositionActionCfg,
    MuscleActivationActionCfg,
)
from mjswan.policy import action_term_joint_names, drives_joints

# A free base no actuator drives, a hip the policy drives, two elbows a command drives.
ROBOT = """
<mujoco>
  <worldbody>
    <body name="base">
      <freejoint name="root"/>
      <geom size="0.1"/>
      <body name="hip"><joint name="hip"/><geom size="0.05"/></body>
      <body name="left_elbow"><joint name="left_elbow"/><geom size="0.05"/></body>
      <body name="right_elbow"><joint name="right_elbow"/><geom size="0.05"/></body>
    </body>
  </worldbody>
  <actuator>
    <position joint="hip"/>
    <position joint="left_elbow"/>
    <position joint="right_elbow"/>
  </actuator>
</mujoco>
"""


@pytest.fixture
def robot() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(ROBOT)


def _elbows(**kwargs) -> CommandJointPositionActionCfg:
    return CommandJointPositionActionCfg(
        command_name="arm", actuator_names=(".*_elbow",), **kwargs
    )


def _sliders(*names: str):
    return ui_command([SliderConfig(name=name, label=name) for name in names])


def _add(robot, policy, *, commands, arm) -> Builder:
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=robot
    )
    scene.add_policy(
        name="Policy",
        policy=policy,
        policy_joint_names=["hip"],
        commands=commands,
        actions={"legs": JointPositionActionCfg(actuator_names=("hip",)), "arm": arm},
    )
    return builder


def test_to_dict():
    cfg = CommandJointPositionActionCfg(
        command_name="arm",
        actuator_names=("robot/.*_elbow_joint",),
        offset=0.1,
        stiffness=40.0,
        clip={".*": (-1.0, 1.0)},
    )
    assert cfg.unsupported_reason is None
    assert cfg.to_dict() == {
        "type": "joint_position_command",
        "offset": 0.1,
        "actuator_names": ["robot/.*_elbow_joint"],
        "command_name": "arm",
        "stiffness": 40.0,
        "clip": {".*": [-1.0, 1.0]},
    }


def test_reaches_the_mdp_entry_with_its_joints_resolved(
    tmp_path, robot, minimal_onnx, build_manifest
):
    builder = _add(
        robot, minimal_onnx, commands={"arm": _sliders("l", "r")}, arm=_elbows()
    )
    manifest = build_manifest(builder, tmp_path / "out")
    mdp = manifest["projects"][0]["scenes"][0]["mdps"][0]
    assert mdp["actions"]["arm"] == {
        "type": "joint_position_command",
        "actuator_names": ["left_elbow", "right_elbow"],
        "command_name": "arm",
    }


def test_matches_only_actuated_joints(robot, minimal_onnx):
    arm = CommandJointPositionActionCfg(command_name="arm", actuator_names=(".*",))
    with pytest.warns(
        RuntimeWarning, match=r"also drives the policy's joints \['hip'\]"
    ):
        _add(robot, minimal_onnx, commands={"arm": _sliders("a", "b", "c")}, arm=arm)
    assert arm.actuator_names == ("hip", "left_elbow", "right_elbow")


def test_a_pattern_matching_no_actuated_joint_fails(robot, minimal_onnx):
    arm = CommandJointPositionActionCfg(
        command_name="arm", actuator_names=("left_elbow", "root")
    )
    with pytest.raises(ValueError, match=r"matches no actuated joint with \['root'\]"):
        _add(robot, minimal_onnx, commands={"arm": _sliders("root")}, arm=arm)


def test_a_missing_command_fails_the_build(
    tmp_path, robot, minimal_onnx, build_manifest
):
    builder = _add(robot, minimal_onnx, commands={}, arm=_elbows())
    with pytest.raises(ValueError, match="reads command 'arm'"):
        build_manifest(builder, tmp_path / "out")


def test_a_command_of_the_wrong_width_fails_the_build(
    tmp_path, robot, minimal_onnx, build_manifest
):
    builder = _add(robot, minimal_onnx, commands={"arm": _sliders("l")}, arm=_elbows())
    with pytest.raises(
        ValueError, match=r"drives 2 joint\(s\), but command 'arm' is 1"
    ):
        build_manifest(builder, tmp_path / "out")


def test_a_pattern_keyed_offset_resolves_against_its_own_joints(robot, minimal_onnx):
    arm = _elbows(offset={".*_elbow": 0.3})
    _add(robot, minimal_onnx, commands={"arm": _sliders("l", "r")}, arm=arm)
    assert arm.offset == {"left_elbow": 0.3, "right_elbow": 0.3}


def test_policy_joint_names_leave_out_a_command_term(robot):
    legs = JointPositionActionCfg(actuator_names=("hip",))
    assert action_term_joint_names({"legs": legs, "arm": _elbows()}, robot) == ["hip"]
    assert action_term_joint_names({"arm": _elbows()}, robot) is None


def test_a_command_term_needs_no_policy_joint_names():
    muscle = MuscleActivationActionCfg(actuator_names=("m",))
    assert not drives_joints({"muscle": muscle, "arm": _elbows()})
    assert drives_joints({"legs": JointPositionActionCfg(), "arm": _elbows()})
