"""An actuator's torque law, traced to a graph the browser's evaluator runs every
physics substep: what traces, what is refused and why, and the MDP's ``actuators``."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("mjlab")
torch = pytest.importorskip("torch")

import mujoco  # noqa: E402
import onnx  # noqa: E402
from onnx.reference import ReferenceEvaluator  # noqa: E402

from mjswan.build.mdp.actuator import serialize_actuators  # noqa: E402
from mjswan.compile.actuator import (  # noqa: E402
    INPUTS,
    OUTPUT,
    SUPPORTED_OPS,
    UnsupportedActuator,
    trace_actuator,
)
from mjswan.compile.export import DivergentGraph  # noqa: E402
from mjswan.managers import action_manager  # noqa: E402
from mjswan.mdp import MdpConfig  # noqa: E402
from mjswan.mjlab.actuator import build_actuator  # noqa: E402
from mjswan.mjlab.env import build_single_entity_trace_env  # noqa: E402

XML = """<mujoco><worldbody><body name="base"><freejoint/><geom size="0.1"/>
<body name="thigh"><joint name="hip" type="hinge"/><geom size="0.05"/>
<body name="shin"><joint name="knee" type="hinge"/><geom size="0.05"/>
</body></body></body></worldbody></mujoco>"""


@pytest.fixture(scope="module")
def env():
    return build_single_entity_trace_env(lambda: mujoco.MjSpec.from_string(XML))


@pytest.fixture
def actuator():
    return pytest.importorskip("mjlab.actuator")


def _ideal_pd(actuator, **fields):
    return actuator.IdealPdActuatorCfg(
        target_names_expr=(".*",), stiffness=40.0, damping=2.0, **fields
    )


def _dc_motor(actuator):
    return actuator.DcMotorActuatorCfg(
        target_names_expr=(".*",),
        stiffness=40.0,
        damping=2.0,
        effort_limit=20.0,
        saturation_effort=30.0,
        velocity_limit=10.0,
    )


def _agrees(built, onnx_bytes: bytes, width: int) -> None:
    """The graph gives what the live ``compute`` gives, joint speeds well past a
    torque-speed curve's corner included."""
    from mjlab.actuator.actuator import ActuatorCmd

    run = ReferenceEvaluator(onnx.load_from_string(onnx_bytes))
    rng = np.random.default_rng(0)
    for _ in range(32):
        feeds = {
            name: (rng.uniform(-1, 1, (1, width)) * spread).astype(np.float32)
            for name, spread in zip(INPUTS, (3.0, 40.0, 3.0, 10.0, 30.0))
        }
        want = built.compute(
            ActuatorCmd(
                position_target=torch.from_numpy(feeds["pos_target"]),
                velocity_target=torch.from_numpy(feeds["vel_target"]),
                effort_target=torch.from_numpy(feeds["effort_target"]),
                pos=torch.from_numpy(feeds["pos"]),
                vel=torch.from_numpy(feeds["vel"]),
            )
        ).numpy()
        (got,) = run.run(None, {k: v for k, v in feeds.items() if k in run.input_names})
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("make", [_ideal_pd, _dc_motor], ids=["ideal-pd", "dc-motor"])
def test_an_ideal_pd_family_law_traces_to_elementwise_operators(env, actuator, make):
    built, targets = build_actuator(env, "robot", make(actuator))
    export = trace_actuator(built, len(targets), what="Actuator")

    model = onnx.load_from_string(export.onnx_bytes)
    assert [o.name for o in model.graph.output] == [OUTPUT]
    assert {i.name for i in model.graph.input} <= set(INPUTS)
    assert set(export.ops) <= SUPPORTED_OPS
    assert targets == ["hip", "knee"]
    _agrees(built, export.onnx_bytes, len(targets))


def test_a_law_outside_the_evaluator_is_refused_naming_its_operator(env, actuator):
    built, targets = build_actuator(env, "robot", _ideal_pd(actuator))
    mix = torch.ones(len(targets), len(targets))

    with pytest.raises(
        UnsupportedActuator, match=r"traces to MatMul.*register_actuator"
    ):
        trace_actuator(
            built, len(targets), what="Actuator", compute=lambda a, cmd: cmd.vel @ mix
        )


def test_a_law_keeping_state_between_calls_is_refused(env, actuator):
    built, targets = build_actuator(env, "robot", _ideal_pd(actuator))
    held = {"vel": torch.zeros(1, len(targets))}

    def filtered(a, cmd):
        held["vel"] = 0.5 * held["vel"] + 0.5 * cmd.vel
        return a.stiffness * (cmd.position_target - cmd.pos) - a.damping * held["vel"]

    with pytest.raises(DivergentGraph, match="state kept between calls"):
        trace_actuator(built, len(targets), what="Actuator", compute=filtered)


def test_a_law_that_does_not_export_is_refused(env, actuator):
    built, targets = build_actuator(env, "robot", _ideal_pd(actuator))
    net = torch.jit.script(torch.nn.Linear(len(targets), len(targets)))

    with pytest.raises(UnsupportedActuator, match="does not trace to a graph"):
        trace_actuator(
            built, len(targets), what="Actuator", compute=lambda a, cmd: net(cmd.vel)
        )


def test_the_mdp_entry_lists_each_graph_and_its_joints(env, actuator, tmp_path):
    entries = serialize_actuators(
        [("robot", _ideal_pd(actuator)), ("robot", _dc_motor(actuator))],
        ["robot/hip", "robot/knee"],
        env=env,
        out_dir=tmp_path,
        scope="mdp/walk",
    )

    assert list(entries) == ["IdealPdActuator_0", "DcMotorActuator_1"]
    entry = entries["DcMotorActuator_1"]
    assert entry["joints"] == ["robot/hip", "robot/knee"]
    assert entry["onnx"] == "mdp/walk/actuator/DcMotorActuator_1.onnx"
    assert entry["func"] == "mjlab.actuator.dc_actuator:DcMotorActuator"
    written = onnx.load(str(tmp_path / entry["onnx"]))
    meta = {p.key: p.value for p in written.metadata_props}
    assert meta["mjswan.kind"] == "actuator"
    json.dumps(entries)


def test_a_bare_policy_names_its_joints_bare(env, actuator, tmp_path):
    entries = serialize_actuators(
        [("robot", _ideal_pd(actuator))],
        ["hip", "knee"],
        env=env,
        out_dir=tmp_path,
        scope="mdp/walk",
    )

    assert entries["IdealPdActuator_0"]["joints"] == ["hip", "knee"]


def test_a_policy_naming_no_joints_gets_the_mjlab_scenes_names(env, actuator, tmp_path):
    # A policy of command terms alone: its actuators are still an mjlab scene's.
    entries = serialize_actuators(
        [("robot", _ideal_pd(actuator))], [], env=env, out_dir=tmp_path
    )

    assert entries["IdealPdActuator_0"]["joints"] == ["robot/hip", "robot/knee"]
    assert entries["IdealPdActuator_0"]["onnx"] == "actuator/IdealPdActuator_0.onnx"


def test_a_registered_law_is_traced_in_place_of_the_actuators_own(
    env, actuator, tmp_path, monkeypatch
):
    def half_torque(a, cmd):
        return 0.5 * a.compute(cmd)

    monkeypatch.setitem(
        action_manager._actuator_registry, "IdealPdActuatorCfg", half_torque
    )
    entries = serialize_actuators(
        [("robot", _ideal_pd(actuator))],
        ["robot/hip", "robot/knee"],
        env=env,
        out_dir=tmp_path,
        scope="mdp/walk",
    )

    entry = entries["IdealPdActuator_0"]
    assert entry["func"].endswith("half_torque")
    run = ReferenceEvaluator(onnx.load(str(tmp_path / entry["onnx"])))
    feeds = {name: np.zeros((1, 2), np.float32) for name in run.input_names}
    feeds["pos_target"] = np.ones((1, 2), np.float32)
    (torque,) = run.run(None, feeds)
    np.testing.assert_allclose(torque, [[20.0, 20.0]])


def _mdp_entry(actuator, env, tmp_path):
    from types import SimpleNamespace

    from mjswan.build.manifest import mdp_entry

    mdp = MdpConfig()
    mdp._actuators = (("robot", _ideal_pd(actuator)),)
    owner = SimpleNamespace(
        id="walk",
        name="Walk",
        policy_joint_names=["robot/hip", "robot/knee"],
        policy_num_actions=None,
    )
    return mdp_entry(
        mdp,
        "walk",
        [owner],
        sidecars={"walk": {}},
        env=env,
        scene_dir=tmp_path,
        on_term=lambda kind: None,
    )


def test_the_built_mdp_entry_carries_its_actuators(env, actuator, tmp_path):
    entry = _mdp_entry(actuator, env, tmp_path)

    law = entry["actuators"]["IdealPdActuator_0"]
    assert law["onnx"] == "mdp/walk/actuator/IdealPdActuator_0.onnx"
    assert law["joints"] == ["robot/hip", "robot/knee"]
    assert (tmp_path / law["onnx"]).is_file()


def test_actuators_on_a_scene_without_a_trace_env_are_refused(actuator, tmp_path):
    with pytest.raises(ValueError, match="has no trace env to build them on"):
        _mdp_entry(actuator, None, tmp_path)
