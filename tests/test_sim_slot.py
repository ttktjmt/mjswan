"""Raw ``SimData`` fields as traced-graph slots, and ``EntityData`` traced through them.

Layer: L3 (builds a real mjlab env from a spec).

Raw ``mjData`` is what a slot is. A term reading ``entity.data.data.<field>`` directly
(myosuite's mimic terms read ``act`` and ``time`` this way) gets a ``sim`` slot; a term
reading an ``EntityData`` property gets either a value slot — when the browser has a
reader for it, the shortcut — or, for any other property, the raw fields the property
reads, with mjlab's math traced into the graph.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "disable")

pytest.importorskip("mjlab")
pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

from mjswan.compile.tracer import (  # noqa: E402
    READER_FIELDS,
    TermExport,
    read_slot,
    slots_json,
    trace_term,
)
from mjswan.trace_env import build_single_entity_trace_env  # noqa: E402

MUSCLE_MODEL = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <site name="s_a" pos="0 0 0.1"/>
      <body name="link" pos="0 0 0.3">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <site name="s_b" pos="0 0 -0.1"/>
      </body>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t"><site site="s_a"/><site site="s_b"/></spatial>
  </tendon>
  <actuator>
    <muscle name="m" tendon="t" lengthrange="0 1"/>
  </actuator>
</mujoco>
"""


def _spec():
    return mujoco.MjSpec.from_string(MUSCLE_MODEL)


@pytest.fixture(scope="module")
def env():
    return build_single_entity_trace_env(_spec)


def _run(export: TermExport, env) -> np.ndarray:
    """The graph's output on the env's current state, fed through ``read_slot``."""
    session = ort.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    declared = {i.name for i in session.get_inputs()}
    feeds = {
        name: read_slot(env, slot).detach().cpu().numpy().astype(np.float32)
        for name, slot in zip(export.input_names, export.input_slots)
        if name in declared
    }
    (out,) = session.run([export.output_name], feeds)
    return out


def _move(env, seed: int) -> None:
    """Put the sim somewhere a value baked at trace time would not follow."""
    gen = torch.Generator().manual_seed(seed)
    data = env.scene["robot"].data
    quat = torch.rand(1, 4, generator=gen) - 0.5
    quat = quat / quat.norm()
    data.write_root_pose(torch.cat([torch.rand(1, 3, generator=gen), quat], dim=-1))
    data.write_root_velocity(torch.rand(1, 6, generator=gen) - 0.5)
    data.write_joint_position(torch.rand(1, 1, generator=gen) - 0.5)
    data.write_joint_velocity(torch.rand(1, 1, generator=gen) - 0.5)
    env.sim.forward()


def _act_scaled_by_step(env):
    """A myosuite-shaped term: raw ``act`` scaled by the control rate."""
    data = env.scene["robot"].data.data
    return data.act * (env.physics_dt * env.cfg.decimation)


def _time(env):
    return torch.as_tensor(env.scene["robot"].data.data.time).reshape(-1, 1)


def _joint_pos(env):
    return env.scene["robot"].data.joint_pos


def _body_link_vel(env):
    return env.scene["robot"].data.body_link_vel_w.reshape(1, -1)


def _joint_torques(env):
    return env.scene["robot"].data.joint_torques


def test_raw_sim_fields_become_sim_slots(env):
    export = trace_term(_act_scaled_by_step, {}, env, name="act")
    assert export.input_slots == [("__sim__", "act")]
    assert slots_json(export) == [
        {"sim": "act", "input": "sim__act", "shape": [1, 1]},
    ]


def test_graph_matches_the_live_term_on_a_fresh_value(env):
    export = trace_term(_act_scaled_by_step, {}, env, name="act")
    session = ort.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    # Move the sim off the trace-time value: a baked constant would not follow.
    env.scene["robot"].data.data.act[:] = 0.7
    feeds = {export.input_names[0]: read_slot(env, export.input_slots[0]).numpy()}
    (out,) = session.run([export.output_name], feeds)
    np.testing.assert_allclose(out, _act_scaled_by_step(env).numpy(), rtol=1e-6)
    assert out[0, 0] == pytest.approx(0.7 * env.physics_dt * env.cfg.decimation)


def test_sim_time_is_a_dynamic_slot(env):
    export = trace_term(_time, {}, env, name="time")
    assert export.input_slots == [("__sim__", "time")]


class TestTraceThrough:
    """An ``EntityData`` property the browser has no reader for.

    Every property has one now, so the tests force the path with an empty
    ``reader_fields`` — what a property mjlab adds tomorrow gets by default.
    """

    def test_reads_become_the_raw_fields_the_property_reads(self, env):
        export = trace_term(_body_link_vel, {}, env, name="body_vel", reader_fields=())
        # mjlab's `compute_velocity_from_cvel(xpos, subtree_com, cvel)`, nothing else.
        assert export.input_slots == [
            ("__sim__", "cvel"),
            ("__sim__", "subtree_com"),
            ("__sim__", "xpos"),
        ]
        assert all(entry["sim"] for entry in slots_json(export))

    def test_graph_reproduces_the_property_over_moved_state(self, env):
        export = trace_term(_body_link_vel, {}, env, name="body_vel", reader_fields=())
        for seed in (1, 2):
            _move(env, seed)
            live = _body_link_vel(env).detach().numpy()
            assert np.abs(live).max() > 0.05, "state did not move"
            np.testing.assert_allclose(_run(export, env), live, rtol=1e-5, atol=1e-6)

    def test_a_property_that_raises_fails_the_build(self, env):
        # `joint_torques` raises in mjlab itself; used to build fine and freeze.
        with pytest.raises(NotImplementedError):
            trace_term(_joint_torques, {}, env, name="torques")


class TestShortcut:
    """A property the browser reads natively stays one value slot."""

    def test_reader_field_is_a_value_slot(self, env):
        export = trace_term(_joint_pos, {}, env, name="jp")
        assert export.input_slots == [("robot", "joint_pos")]

    def test_shortcut_and_traced_path_agree(self, env):
        # Both paths for one property: the value slot the reader fills, and the
        # raw `qpos` slot with mjlab's indexing traced in. Same numbers either way.
        shortcut = trace_term(_joint_pos, {}, env, name="jp")
        traced = trace_term(_joint_pos, {}, env, name="jp", reader_fields=())
        assert shortcut.input_slots == [("robot", "joint_pos")]
        assert traced.input_slots == [("__sim__", "qpos")]
        _move(env, 3)
        live = _joint_pos(env).detach().numpy()
        np.testing.assert_allclose(_run(shortcut, env), live, rtol=1e-6)
        np.testing.assert_allclose(_run(traced, env), live, rtol=1e-6)

    def test_reader_fields_are_every_property_but_joint_torques(self):
        # A name here that mjlab dropped would be a slot the build emits for nothing; a
        # property mjlab added and this list lacks is traced through, which works but
        # deserves a reader.
        from mjlab.entity.data import EntityData

        properties = {n for n, v in vars(EntityData).items() if isinstance(v, property)}
        assert READER_FIELDS == (properties - {"joint_torques"}) | {"gravity_vec_w"}
