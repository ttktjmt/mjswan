"""``JointVelocityActionCfg``: serialization, and what the mjlab adapter fills in."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from mjswan.envs.mdp.actions import JointVelocityActionCfg
from mjswan.mjlab import resolve_default_joint_vel

JOINTS = ["robot/left_hip", "robot/left_wheel", "robot/right_wheel"]


def _env_cfg(joint_vel: dict[str, float], actuators: tuple = ()) -> SimpleNamespace:
    robot = SimpleNamespace(
        init_state=SimpleNamespace(joint_vel=joint_vel),
        articulation=SimpleNamespace(actuators=actuators),
    )
    return SimpleNamespace(scene=SimpleNamespace(entities={"robot": robot}))


def test_to_dict():
    cfg = JointVelocityActionCfg(
        actuator_names=("robot/.*_wheel",), scale=100.0, clip={".*": (-5.0, 5.0)}
    )
    assert cfg.unsupported_reason is None
    assert cfg.to_dict() == {
        "type": "joint_velocity",
        "scale": 100.0,
        "actuator_names": ["robot/.*_wheel"],
        "clip": {".*": [-5.0, 5.0]},
    }


def test_default_offset_is_the_default_joint_velocity():
    cfg = JointVelocityActionCfg(actuator_names=("robot/.*_wheel",), offset=9.0)
    resolve_default_joint_vel(
        {"vel": cfg}, JOINTS, _env_cfg({"left_wheel": 1.5, ".*": 0.0})
    )
    # Replaced, not added to, as in mjlab.
    assert cfg.offset == {"robot/left_wheel": 1.5}


def test_zero_default_velocity_leaves_no_offset():
    cfg = JointVelocityActionCfg(actuator_names=("robot/.*_wheel",), offset=9.0)
    resolve_default_joint_vel({"vel": cfg}, JOINTS, _env_cfg({".*": 0.0}))
    assert cfg.offset == 0.0
    assert "offset" not in cfg.to_dict()


def test_offset_kept_without_use_default_offset():
    cfg = JointVelocityActionCfg(
        actuator_names=("robot/.*_wheel",), offset=0.5, use_default_offset=False
    )
    resolve_default_joint_vel({"vel": cfg}, JOINTS, _env_cfg({".*": 1.0}))
    assert cfg.offset == 0.5


@pytest.mark.mjlab
def test_adapted_from_mjlab():
    pytest.importorskip("mjlab")
    from mjlab.envs.mdp.actions import JointVelocityActionCfg as MjlabCfg

    from mjswan.mjlab import adapt_actions

    adapted = adapt_actions(
        {
            "vel": MjlabCfg(
                entity_name="robot", actuator_names=(".*_wheel",), scale=100.0
            )
        }
    )
    assert adapted is not None
    term = adapted["vel"]
    assert isinstance(term, JointVelocityActionCfg)
    assert term.unsupported_reason is None
    assert term.to_dict()["actuator_names"] == ["robot/.*_wheel"]
