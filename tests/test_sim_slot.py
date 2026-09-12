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
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from mjswan.compile.tracer import (  # noqa: E402
    READER_FIELDS,
    GroupTermSpec,
    read_slot,
    slots_json,
    trace_observation_group,
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


def _run(export, env) -> np.ndarray:
    """The graph's output on the env's current state, fed through ``read_slot``."""
    session = ort.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    declared = {i.name for i in session.get_inputs()}
    rows = export.input_rows or [None] * len(export.input_slots)
    feeds = {
        name: read_slot(env, slot, row).detach().cpu().numpy().astype(np.float32)
        for name, slot, row in zip(export.input_names, export.input_slots, rows)
        if name in declared
    }
    (out,) = session.run([export.output_name], feeds)
    return out


def _ops(export) -> list[str]:
    return [n.op_type for n in onnx.load_from_string(export.onnx_bytes).graph.node]


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


def _body_pos(env):
    return env.scene["robot"].data.body_link_pos_w.reshape(1, -1)


def _root_ang_vel(env):
    return env.scene["robot"].data.root_link_ang_vel_w


def _body_ang_vel(env):
    return env.scene["robot"].data.body_link_ang_vel_w.reshape(1, -1)


def _cvel_twice(env):
    """One field, two index sets: the root body alone, then bodies 1 and 2."""
    cvel = env.scene["robot"].data.data.cvel
    return torch.cat([cvel[:, 1].reshape(1, -1), cvel[:, [1, 2]].reshape(1, -1)], -1)


def _qvel_whole(env):
    return (env.scene["robot"].data.data.qvel * 2.0)[:, 0:1]


def _qpos_masked(env):
    qpos = env.scene["robot"].data.data.qpos
    mask = torch.zeros(qpos.shape[1], dtype=torch.bool)
    mask[0] = True
    return qpos[:, mask]


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


class TestNarrowing:
    """A sim slot carries only the rows the term indexes (§3): the manifest says which,
    the graph gathers nothing, and the browser reads those rows."""

    def test_indexed_reads_narrow_the_slot_to_their_rows(self, env):
        indexing = env.scene["robot"].indexing
        export = trace_term(_body_link_vel, {}, env, name="v", reader_fields=())
        entries = {e["sim"]: e for e in slots_json(export)}
        body_ids = [int(b) for b in indexing.body_ids.tolist()]
        assert entries["xpos"]["rows"] == body_ids
        assert entries["xpos"]["shape"] == [1, len(body_ids), 3]
        assert entries["cvel"]["rows"] == body_ids
        assert entries["subtree_com"]["rows"] == [indexing.root_body_id]

    def test_a_read_of_exactly_the_rows_gathers_nothing(self, env):
        # `body_link_pos_w` indexes xpos and xquat by every body: the input already is
        # those rows, in order, so the graph is a Concat and a Slice, no Gather.
        export = trace_term(_body_pos, {}, env, name="p", reader_fields=())
        assert "Gather" not in _ops(export)
        _move(env, 4)
        np.testing.assert_allclose(
            _run(export, env), _body_pos(env).detach().numpy(), rtol=1e-6
        )

    def test_a_narrowed_slot_feeds_the_same_numbers_as_the_whole_field(self, env):
        export = trace_term(_body_link_vel, {}, env, name="v", reader_fields=())
        _move(env, 5)
        for slot, rows in zip(export.input_slots, export.input_rows):
            assert rows is not None
            whole = read_slot(env, slot)
            torch.testing.assert_close(read_slot(env, slot, rows), whole[:, rows])
        np.testing.assert_allclose(
            _run(export, env),
            _body_link_vel(env).detach().numpy(),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_reads_by_different_index_sets_take_the_union(self, env):
        export = trace_term(_cvel_twice, {}, env, name="c")
        assert export.input_rows == [[1, 2]]
        _move(env, 6)
        np.testing.assert_allclose(
            _run(export, env), _cvel_twice(env).detach().numpy(), rtol=1e-6
        )

    def test_a_group_unions_across_its_terms(self, env):
        indexing = env.scene["robot"].indexing
        export = trace_observation_group(
            [
                GroupTermSpec(name="root", func=_root_ang_vel, params={}),
                GroupTermSpec(name="bodies", func=_body_ang_vel, params={}),
            ],
            env,
            name="g",
            reader_fields=(),
        )
        assert export.input_slots == [("__sim__", "cvel")]
        assert export.input_rows == [[int(b) for b in indexing.body_ids.tolist()]]

    def test_a_field_used_whole_or_by_a_mask_ships_whole(self, env):
        for term in (_qvel_whole, _qpos_masked):
            export = trace_term(term, {}, env, name=term.__name__)
            assert export.input_rows == [None]
            assert "rows" not in slots_json(export)[0]
            _move(env, 7)
            np.testing.assert_allclose(
                _run(export, env), term(env).detach().numpy(), rtol=1e-6
            )
