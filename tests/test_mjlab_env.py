"""The trace env's defaults, which every `*_rel` observation bakes in.

Layer: L3 (builds a real mjlab env from a spec).

`build_single_entity_trace_env` exists so a plain `add_scene()` scene can be
traced at all. What it hands the tracer is not just kinematics: `joint_pos_rel`
and friends bake `default_joint_pos` into the graph as a constant, so a wrong
default is not a wrong *reading* — it is a permanent offset in what the policy
sees, invisible in the artifact and fatal in the browser (a balance policy reads
its whole stand pose as error and falls over).
"""

from __future__ import annotations

import pytest

pytest.importorskip("mjlab")

import mujoco  # noqa: E402

from mjswan.mjlab.env import (  # noqa: E402
    _quiet_warp_module_loads,
    build_mjlab_env,
    build_single_entity_trace_env,
)

STAND = {"hinge_a": 0.25, "hinge_b": -0.4}

MODEL = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link_a" pos="0 0 0.2">
        <joint name="hinge_a" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.1"/>
        <body name="link_b" pos="0 0 0.2">
          <joint name="hinge_b" type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.05 0.05 0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  {keyframe}
</mujoco>
"""

KEYFRAME = f"""
  <keyframe>
    <key name="stand" qpos="0 0 1 1 0 0 0 {STAND["hinge_a"]} {STAND["hinge_b"]}"/>
  </keyframe>
"""


def _spec_fn(keyframe: str):
    return lambda: mujoco.MjSpec.from_string(MODEL.format(keyframe=keyframe))


def test_default_joint_pos_comes_from_the_models_keyframe():
    """The browser resets to keyframe 0, so the traced default has to be that pose."""
    env = build_single_entity_trace_env(_spec_fn(KEYFRAME))
    defaults = env.scene["robot"].data.default_joint_pos.reshape(-1).tolist()
    names = env.scene["robot"].joint_names
    assert dict(zip(names, defaults)) == pytest.approx(STAND)


def test_the_trace_env_computes_no_contacts():
    """Nothing traced reads a contact; too many can crash mujoco_warp in `reset()`."""
    resting = """
    <mujoco>
      <worldbody>
        <geom type="plane" size="1 1 0.1"/>
        <body name="base" pos="0 0 0.05">
          <freejoint/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    env = build_single_entity_trace_env(lambda: mujoco.MjSpec.from_string(resting))
    assert int(env.sim.data.nacon[0]) == 0


def test_a_model_without_a_keyframe_keeps_mjlabs_zero_default():
    """No keyframe means the zero pose *is* the rest pose — mjlab's own default."""
    env = build_single_entity_trace_env(_spec_fn(""))
    defaults = env.scene["robot"].data.default_joint_pos.reshape(-1).tolist()
    assert defaults == [0.0, 0.0]


def test_a_time_out_traced_against_the_default_env_never_fires():
    """Its episodes have no time limit, so the graph has no horizon to reach."""
    onnxruntime = pytest.importorskip("onnxruntime")
    import numpy as np
    from mjlab.envs.mdp import time_out

    from mjswan.compile import trace_term

    export = trace_term(
        time_out, {}, build_single_entity_trace_env(_spec_fn(KEYFRAME)), name="time_out"
    )
    session = onnxruntime.InferenceSession(
        export.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    # The last count float32 holds exactly: about 93 hours at 50 Hz.
    feed = {export.input_names[0]: np.array([2.0**24], dtype=np.float32)}
    assert not session.run(None, feed)[0].any()


def test_control_dt_and_episode_length_set_the_horizon_in_control_steps():
    env = build_single_entity_trace_env(
        _spec_fn(KEYFRAME), control_dt=0.02, episode_length_s=10.0
    )
    assert env.step_dt == 0.02
    assert env.max_episode_length == 500


def test_an_episode_length_without_control_dt_is_refused():
    """The env would count it in its own 2 ms steps rather than the scene's."""
    with pytest.raises(ValueError, match="control_dt"):
        build_single_entity_trace_env(_spec_fn(KEYFRAME), episode_length_s=10.0)


def test_the_envs_manager_tables_stay_out_of_the_build_log(capsys):
    """mjlab prints ~120 lines of tables per env, with no flag to turn them off."""
    build_single_entity_trace_env(_spec_fn(KEYFRAME))
    out = capsys.readouterr().out
    assert "Base Environment" not in out
    assert "ObservationManager" not in out


def test_a_failed_env_build_replays_the_tables(monkeypatch, capsys):
    """Held, not dropped: the widths they list are what one debugs a failure with."""
    import mjlab.envs

    def _explode(*_args, **_kwargs):
        print("| Active Observation Terms |")
        raise RuntimeError("boom")

    monkeypatch.setattr(mjlab.envs, "ManagerBasedRlEnv", _explode)
    with pytest.raises(RuntimeError, match="boom"):
        build_mjlab_env(object())
    assert "| Active Observation Terms |" in capsys.readouterr().out


def test_warp_module_load_lines_are_quieted_without_burying_warnings():
    """INFO is what prints them; WARNING still lets warnings through."""
    import warp

    original = warp.config.log_level
    try:
        warp.config.log_level = warp.LOG_INFO
        _quiet_warp_module_loads()
        assert warp.config.log_level == warp.LOG_WARNING

        # The escape hatch for a verbose build must survive.
        warp.config.log_level = warp.LOG_DEBUG
        _quiet_warp_module_loads()
        assert warp.config.log_level == warp.LOG_DEBUG
    finally:
        warp.config.log_level = original
