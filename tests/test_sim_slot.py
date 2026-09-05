"""Raw ``SimData`` fields as traced-graph slots.

Layer: L3 (builds a real mjlab env from a spec).

mjlab's ``EntityData`` wraps most of the sim, but not a muscle model's activation state
(``act``) or the sim ``time``; a task that needs them reads ``entity.data.data.<field>``
(myosuite's mimic terms do exactly this). Those fields must become graph inputs in
their own ``sim`` namespace, served whole by the browser from ``mjData``.
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


def _act_scaled_by_step(env):
    """A myosuite-shaped term: raw ``act`` scaled by the control rate."""
    data = env.scene["robot"].data.data
    return data.act * (env.physics_dt * env.cfg.decimation)


def _time(env):
    return torch.as_tensor(env.scene["robot"].data.data.time).reshape(-1, 1)


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
