"""A task's own action term traced to ONNX (ADR 0009).

Layer: L3 (builds a small mjlab env on the CPU).

The trace has to reproduce the term's own ``process_actions`` and ``reset``: the state
it promotes to graph I/O, the env reads it threads as slots, and the branches it bakes.
The build runs each graph against the body before shipping it; these pin what that
check is for, and what the tracer refuses instead of shipping a frozen copy.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("mjlab")

import action_term_fixture as fx  # noqa: E402

from mjswan.build.mdp.action import (  # noqa: E402
    inherited_apply,
    serialize_actions,
)
from mjswan.compile.action import (  # noqa: E402
    UntraceableAction,
    _freeze,
    _same,
    compare_body,
    trace_action_term,
)
from mjswan.envs.mdp.actions import TracedActionCfg  # noqa: E402
from mjswan.managers.action_manager import ActionBinding  # noqa: E402


@pytest.fixture(scope="module")
def env():
    """One env for every term: each test builds its own term on it."""
    env = fx.arm_env({"joint_pos": fx.on_arm(fx.DeltaActionCfg)})
    yield env
    env.close()


def _ort(graph):
    import onnxruntime as ort

    session = ort.InferenceSession(graph.onnx_bytes)
    names = {i.name for i in session.get_inputs()}

    def run(feeds):
        outs = session.run(
            graph.output_names, {k: v for k, v in feeds.items() if k in names}
        )
        return dict(zip(graph.output_names, outs))

    return run


def _state_names(export) -> list[str]:
    return [spec["name"] for spec in export.state_fields]


def test_a_delta_term_carries_its_target_and_restarts_it_at_the_measured_pose(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos", ramp_from=True)

    assert "_target" in _state_names(export)
    assert export.target_names == ["robot/j1", "robot/j2", "robot/j3"]
    # The soft limits are constants; only the reset reads the live joint positions.
    assert export.process.input_names == []
    assert export.reset is not None
    assert export.reset.input_names == ["robot__joint_pos"]
    assert export.process.output_names[-2:] == ["processed", "processed_from"]


def test_a_floating_base_without_mocap_bodies_traces():
    env = fx.arm_env({"joint_pos": fx.on_arm(fx.DeltaActionCfg)}, floating=True)
    try:
        assert env.sim.mj_model.nmocap == 0
        term = fx.on_arm(fx.DeltaActionCfg).build(env)
        export = trace_action_term(term, name="joint_pos", ramp_from=True)
        assert "_target" in _state_names(export)
    finally:
        env.close()


def test_the_delta_graph_matches_the_body_in_onnx_runtime(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos", ramp_from=True)
    run_process, run_reset = _ort(export.process), _ort(export.reset)
    generator = torch.Generator().manual_seed(3)

    assert compare_body(term, export, "reset", run_reset) is None
    for _ in range(12):
        raw = torch.rand((1, 3), generator=generator) * 4 - 2
        assert compare_body(term, export, "process", run_process, raw=raw) is None


def test_the_ramp_starts_where_the_term_stood_before_the_step(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos", ramp_from=True)
    run = _ort(export.process)
    feeds = {
        f"prev_{s['name']}": np.asarray(s["init"], np.float32).reshape(s["shape"])
        for s in export.state_fields
    }
    feeds["raw"] = np.ones((1, 3), np.float32)
    outputs = run(feeds)
    np.testing.assert_allclose(
        outputs["processed_from"], feeds["prev__processed_actions"]
    )
    # The home pose is inside the soft limits, so a full action is one whole delta.
    np.testing.assert_allclose(
        outputs["processed"] - outputs["processed_from"], 1 / 24, rtol=1e-5
    )


def test_a_held_term_reads_the_episode_clock_and_zeroes_the_action_while_holding(env):
    term = fx.on_arm(fx.HeldActionCfg, scale=0.5).build(env)
    export = trace_action_term(term, name="joint_pos")
    assert export.process.input_names == ["env__episode_length_buf"]
    run = _ort(export.process)

    def processed(step: int) -> np.ndarray:
        feeds = {
            f"prev_{s['name']}": np.asarray(s["init"], np.float32).reshape(s["shape"])
            for s in export.state_fields
        }
        feeds["raw"] = np.ones((1, 3), np.float32)
        feeds["env__episode_length_buf"] = np.asarray([step], np.float32)
        return run(feeds)["processed"]

    default = term._offset.numpy()
    # 50 steps of 20 ms is the 1 s hold: the trace kept both sides of mjlab's branch.
    np.testing.assert_allclose(processed(49), default, atol=1e-6)
    np.testing.assert_allclose(processed(50), default + 0.5, atol=1e-6)


def test_an_ema_term_gets_mjlabs_apply_without_a_binding(env):
    term = fx.on_arm(fx.EmaActionCfg).build(env)
    export = trace_action_term(term, name="joint_pos")
    assert "_ema" in _state_names(export)
    assert inherited_apply(term, unchanged=True) == "joint_position"


def test_a_term_overriding_apply_needs_a_binding_to_say_what_it_does(env):
    held = fx.on_arm(fx.HeldActionCfg).build(env)
    assert inherited_apply(held, unchanged=True) is None
    # With a binding, the mjlab base it subclasses names the apply.
    assert inherited_apply(held, unchanged=False) == "joint_position"
    delta = fx.on_arm(fx.DeltaActionCfg).build(env)
    assert inherited_apply(delta, unchanged=False) is None


@pytest.mark.parametrize(
    ("cfg", "match"),
    [
        (fx.on_arm(fx.CountingActionCfg), r"changes \['_steps'\]"),
        (fx.on_arm(fx.LazyCounterActionCfg), r"creates \['_calls'\]"),
        (fx.on_arm(fx.QueueLagActionCfg), r"changes \['_queue'\]"),
        (fx.on_arm(fx.SlotsCounterActionCfg), r"changes \['_box'\]"),
        (fx.on_arm(fx.DeepCounterActionCfg), r"changes \['_deep'\]"),
        (fx.on_arm(fx.WritingActionCfg), r"set_joint_position_target"),
        (fx.on_arm(fx.SceneWritingActionCfg), r"write_joint_state_to_sim"),
        (fx.on_arm(fx.BufferWritingActionCfg), r"entity\.data\.joint_pos_target"),
        (fx.on_arm(fx.AliasActionCfg), r"views of one tensor"),
        (fx.on_arm(fx.CachedDataActionCfg), r"keeps \['_data'\]"),
        (fx.on_arm(fx.NestedDataActionCfg), r"keeps \['_parts'\]"),
        (fx.on_arm(fx.ClockResetActionCfg), r"episode_length_buf in reset"),
        (fx.on_arm(fx.BiasReadingActionCfg), r"robot\.encoder_bias"),
    ],
    ids=[
        "python-counter",
        "attribute-made-in-the-body",
        "state-in-a-deque",
        "state-in-a-slotted-dataclass",
        "state-five-containers-down",
        "writes-the-sim",
        "writes-the-sim-through-the-scene",
        "writes-an-entity-buffer",
        "aliased-state",
        "cached-entity-data",
        "entity-data-inside-an-object",
        "reset-reads-the-episode-clock",
        "reads-a-field-the-browser-does-not-serve",
    ],
)
def test_what_a_graph_cannot_carry_is_refused(env, cfg, match):
    with pytest.raises(UntraceableAction, match=match):
        trace_action_term(cfg.build(env), name="joint_pos")


def test_a_scene_write_leaves_the_trace_env_alone(env):
    qpos = env.sim.data.qpos.clone()
    with pytest.raises(UntraceableAction):
        trace_action_term(fx.on_arm(fx.SceneWritingActionCfg).build(env), name="a")
    assert torch.equal(env.sim.data.qpos, qpos)


def test_an_untouched_nan_is_not_state_the_body_changed(env):
    term = fx.on_arm(fx.NanFieldActionCfg).build(env)
    trace_action_term(term, name="joint_pos")


@dataclass
class _Box:
    item: Any


@dataclass(slots=True)
class _SlotBox:
    item: Any


def _looped():
    node = SimpleNamespace(item=[0])
    node.self = node
    return node


@pytest.mark.parametrize(
    ("make", "change"),
    [
        (lambda: torch.zeros(2), lambda v: v.add_(1)),
        (lambda: np.zeros(2), lambda v: v.__iadd__(1)),
        (lambda: {"k": [0]}, lambda v: v["k"].append(1)),
        (lambda: SimpleNamespace(item=[0]), lambda v: v.item.append(1)),
        (lambda: {0}, lambda v: v.add(1)),
        (lambda: _Box([0]), lambda v: v.item.append(1)),
        (lambda: _SlotBox([0]), lambda v: v.item.append(1)),
        (lambda: [[[[[[0]]]]]], lambda v: v[0][0][0][0][0].append(1)),
        (_looped, lambda v: v.item.append(1)),
    ],
    ids=[
        "tensor",
        "array",
        "container",
        "plain-object",
        "set",
        "dataclass",
        "slotted-dataclass",
        "six-deep",
        "cycle",
    ],
)
def test_a_change_anywhere_inside_kept_state_is_seen(make, change):
    value = make()
    before = _freeze(value)
    assert _same(before, _freeze(value))
    change(value)
    assert not _same(before, _freeze(value))


def test_values_compare_by_value_and_code_by_identity():
    assert _same(_freeze(slice(None)), _freeze(slice(None)))
    assert not _same(_freeze(slice(0)), _freeze(slice(1)))
    assert not _same(_freeze(lambda: 0), _freeze(lambda: 0))


def test_a_clipped_term_rebinding_twice_keeps_its_processed_action_as_state(env):
    # mjlab's own process_actions rebinds `_processed_actions` twice under `clip`.
    from mjlab.envs.mdp.actions import JointPositionActionCfg

    cfg = fx.on_arm(JointPositionActionCfg, clip={".*": (-0.5, 0.5)})
    for _ in range(30):
        export = trace_action_term(cfg.build(env), name="joint_pos", ramp_from=True)
        assert "_processed_actions" in _state_names(export)


def test_a_per_joint_mask_is_not_an_env_row_mask():
    from mjswan.compile.gate import MaskedRows

    gains = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    with MaskedRows(1):
        assert torch.equal(gains[mask], torch.tensor([1.0, 3.0]))
        rows = torch.ones(1, 3)
        rows[torch.tensor([False])] = 0.0
    assert torch.equal(rows, torch.ones(1, 3))


def test_a_term_that_names_no_joints_is_refused():
    from types import SimpleNamespace

    from mjswan.compile.action import joint_targets

    ik = SimpleNamespace(cfg=SimpleNamespace(entity_name="robot"))
    with pytest.raises(UntraceableAction, match="names no joints"):
        joint_targets(ik, "ik")


def test_a_ramp_on_a_velocity_apply_is_refused(env, tmp_path):
    cfg = TracedActionCfg(
        mjlab_cfg=fx.on_arm(fx.EmaActionCfg),
        binding=ActionBinding(apply="joint_velocity", interpolate=True),
    )
    with pytest.raises(ValueError, match="ramp or hold a position target"):
        serialize_actions({"joint_pos": cfg}, env=env, out_dir=tmp_path)


def test_a_trace_leaves_the_term_as_it_found_it(env):
    term = fx.on_arm(fx.DeltaActionCfg).build(env)
    before = {
        k: v.clone() for k, v in vars(term).items() if isinstance(v, torch.Tensor)
    }
    trace_action_term(term, name="joint_pos", ramp_from=True)
    for name, value in before.items():
        assert torch.equal(getattr(term, name), value), name


def test_the_build_writes_both_graphs_and_names_the_apply(env, tmp_path):
    entry = serialize_actions(
        {
            "joint_pos": TracedActionCfg(
                mjlab_cfg=fx.on_arm(fx.DeltaActionCfg),
                binding=ActionBinding(
                    apply="joint_position",
                    interpolate=lambda cfg: cfg.interpolate_decimation,
                ),
            )
        },
        env=env,
        out_dir=tmp_path,
    )["joint_pos"]

    assert entry["type"] == "joint_position"
    assert entry["interpolate"] is True
    assert entry["actuator_names"] == ["robot/j1", "robot/j2", "robot/j3"]
    # The ramp is the entry's `interpolate`; the graph's `processed_from` feeds it.
    assert "ramp_from" not in entry["process"]
    assert (tmp_path / entry["process"]["onnx"]).is_file()
    assert (tmp_path / entry["reset"]["onnx"]).is_file()
    assert entry["reset"]["input_slots"] == [
        {
            "entity": "robot",
            "field": "joint_pos",
            "input": "robot__joint_pos",
            "shape": [1, 3],
        }
    ]
    init = {f["name"]: f["init"] for f in entry["process"]["state_fields"]}
    np.testing.assert_allclose(init["_target"], list(fx.HOME.values()), atol=1e-6)


def test_the_build_refuses_an_overridden_apply_without_a_binding(env, tmp_path):
    cfg = TracedActionCfg(mjlab_cfg=fx.on_arm(fx.HeldActionCfg))
    with pytest.raises(ValueError, match="overrides apply_actions"):
        serialize_actions({"joint_pos": cfg}, env=env, out_dir=tmp_path)


def test_without_a_trace_env_the_build_says_how_to_supply_one(tmp_path):
    cfg = TracedActionCfg(mjlab_cfg=fx.on_arm(fx.DeltaActionCfg))
    with pytest.raises(ValueError, match="set_trace_env"):
        serialize_actions({"joint_pos": cfg}, out_dir=tmp_path)


def test_an_observation_reads_a_traced_terms_state_as_a_slot(env, tmp_path):
    from mjswan.build.mdp import serialize_observation_group
    from mjswan.compile.slot import traced_action_state
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    def target(env, action_name="joint_pos"):
        return env.action_manager.get_term(action_name)._target

    def processed(env, action_name="joint_pos"):
        return env.action_manager.get_term(action_name)._processed_actions

    group = ObservationGroupCfg(
        terms={
            "target": ObservationTermCfg(func=target),
            "processed": ObservationTermCfg(func=processed),
        }
    )
    with traced_action_state({"joint_pos": ["_target", "_processed_actions"]}):
        entry = serialize_observation_group(group, env, tmp_path, "policy")

    assert [(s["action_term"], s["field"]) for s in entry["input_slots"]] == [
        ("joint_pos", "_processed_actions"),
        ("joint_pos", "_target"),
    ]


def test_an_observation_reads_a_long_state_field_as_float32(tmp_path):
    import onnx

    from mjswan.build.mdp import serialize_observation_group
    from mjswan.compile.slot import traced_action_state
    from mjswan.managers.observation_manager import (
        ObservationGroupCfg,
        ObservationTermCfg,
    )

    def steps(env, action_name="joint_pos"):
        return env.action_manager.get_term(action_name)._steps.float().unsqueeze(-1)

    counting = fx.arm_env({"joint_pos": fx.on_arm(fx.StepCountActionCfg)})
    group = ObservationGroupCfg(terms={"steps": ObservationTermCfg(func=steps)})
    try:
        with traced_action_state({"joint_pos": ["_steps"]}):
            entry = serialize_observation_group(group, counting, tmp_path, "policy")
    finally:
        counting.close()

    (slot,) = entry["input_slots"]
    model = onnx.load(str(next(tmp_path.rglob("*.onnx"))))
    inputs = {i.name: i.type.tensor_type.elem_type for i in model.graph.input}
    # The browser feeds every action_term slot as float32.
    assert inputs[slot["input"]] == onnx.TensorProto.FLOAT


def test_an_observation_reads_only_a_traced_terms_state_and_processed_action(env):
    from mjswan.compile.record import _RecordingEnv
    from mjswan.compile.slot import UnsupportedEnvRead, traced_action_state

    with traced_action_state({"joint_pos": ["_target", "_raw_actions"]}):
        term = _RecordingEnv(env).action_manager.get_term("joint_pos")
        term._target
        term._processed_actions
        term.raw_action
        assert term.num_envs == 1
        # Neither the browser holds: their trace-time values would bake.
        with pytest.raises(UnsupportedEnvRead):
            term._target_ids
        with pytest.raises(UnsupportedEnvRead):
            term._entity


def test_an_observation_cannot_reach_the_entity_through_a_holder(env):
    from mjswan.compile.record import _RecordingEnv
    from mjswan.compile.slot import UnsupportedEnvRead, traced_action_state

    real = env.action_manager.get_term("joint_pos")
    real._parts = SimpleNamespace(entity=real._entity)
    try:
        with traced_action_state({"joint_pos": ["_target"]}):
            term = _RecordingEnv(env).action_manager.get_term("joint_pos")
            with pytest.raises(UnsupportedEnvRead):
                term._parts
    finally:
        del real._parts


def _find(tree, key):
    """Every value under *key*, anywhere in a JSON tree."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if k == key:
                yield v
            yield from _find(v, key)
    elif isinstance(tree, list):
        for v in tree:
            yield from _find(v, key)


def test_a_scene_builds_a_task_action_into_its_manifest(
    tmp_path, build_manifest, minimal_onnx
):
    from mjswan import Builder, register_action
    from mjswan.managers.action_manager import _action_registry

    env = fx.arm_env({"joint_pos": fx.on_arm(fx.DeltaActionCfg)})
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=env.sim.mj_model
    )
    scene.set_trace_env(env)
    scene.add_policy(
        name="Policy",
        policy=minimal_onnx,
        actions={"joint_pos": fx.on_arm(fx.DeltaActionCfg)},
        policy_joint_names=["robot/j1", "robot/j2", "robot/j3"],
    )
    # After `add_policy`: the build looks the binding up.
    register_action(
        "DeltaActionCfg", ActionBinding(apply="joint_position", interpolate=True)
    )
    try:
        manifest = build_manifest(builder, tmp_path / "dist")
    finally:
        _action_registry.pop("DeltaActionCfg", None)
        env.close()

    (actions,) = list(_find(manifest, "actions"))
    entry = actions["joint_pos"]
    assert entry["type"] == "joint_position"
    assert entry["interpolate"] is True
    assert {"process", "reset"} <= set(entry)
    graphs = {p.name for p in (tmp_path / "dist").rglob("*.onnx")}
    assert {"joint_pos.onnx", "joint_pos_reset.onnx"} <= graphs


def test_a_binding_registered_again_after_add_policy_wins(
    tmp_path, build_manifest, minimal_onnx
):
    from mjswan import Builder, register_action
    from mjswan.managers.action_manager import _action_registry

    env = fx.arm_env({"joint_pos": fx.on_arm(fx.DeltaActionCfg)})
    register_action("DeltaActionCfg", ActionBinding(apply="joint_position"))
    builder = Builder()
    scene = builder.add_project(name="P").add_scene(
        control_dt=0.02, name="S", model=env.sim.mj_model
    )
    scene.set_trace_env(env)
    scene.add_policy(
        name="Policy",
        policy=minimal_onnx,
        actions={"joint_pos": fx.on_arm(fx.DeltaActionCfg)},
        policy_joint_names=["robot/j1", "robot/j2", "robot/j3"],
    )
    # A notebook cell run again, with the ramp added.
    register_action(
        "DeltaActionCfg", ActionBinding(apply="joint_position", interpolate=True)
    )
    try:
        manifest = build_manifest(builder, tmp_path / "dist")
    finally:
        _action_registry.pop("DeltaActionCfg", None)
        env.close()

    (actions,) = list(_find(manifest, "actions"))
    assert actions["joint_pos"]["interpolate"] is True


def test_run_parity_traces_a_task_action_and_what_the_build_registers():
    from mjlab.managers.observation_manager import (
        ObservationGroupCfg as MjlabGroupCfg,
    )
    from mjlab.managers.observation_manager import (
        ObservationTermCfg as MjlabTermCfg,
    )

    from mjswan import register_observation
    from mjswan.compile.parity import run_parity
    from mjswan.envs.mdp.observations import _custom_registry

    def target(env, action_name="joint_pos"):
        return env.action_manager.get_term(action_name)._target

    def joint_pos_via_model(env):
        env.sim.mj_model  # A read the tracer refuses.
        return env.scene["robot"].data.joint_pos

    def joint_pos(env):
        return env.scene["robot"].data.joint_pos

    register_observation("joint_pos_via_model", joint_pos)
    parity_env = fx.arm_env(
        {"joint_pos": fx.on_arm(fx.DeltaActionCfg)},
        observations={
            "actor": MjlabGroupCfg(
                terms={
                    "target": MjlabTermCfg(func=target),
                    "joints": MjlabTermCfg(func=joint_pos_via_model),
                }
            )
        },
    )
    try:
        report = run_parity(parity_env, n_steps=4, event_modes=())
    finally:
        _custom_registry.pop("joint_pos_via_model", None)
        parity_env.close()

    terms = {t.name: t for t in report.terms}
    assert terms["joint_pos"].kind == "action"
    assert terms["target"].representation == "onnx"
    # The build ships the registered replacement; it is checked against mjlab's own.
    assert terms["joints"].representation == "onnx"
    assert report.passed, report.summary()
