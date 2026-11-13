"""Class events trace through mjlab's own instance, or one built the way mjlab builds
it, with the env and entities it keeps routed through the tracer."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")
ManagerTermBase = pytest.importorskip("mjlab.managers.manager_base").ManagerTermBase

from mjswan.build.mdp.event import serialize_event  # noqa: E402
from mjswan.compile.export import DivergentGraph  # noqa: E402
from mjswan.managers.event_manager import EventTermCfg  # noqa: E402


class _Scene(dict):
    @property
    def entities(self):
        return self


class _Entity:
    def __init__(self):
        self.data = SimpleNamespace(joint_pos=torch.tensor([[0.1, 0.2, 0.3]]))

    def write_joint_state_to_sim(
        self, position, velocity, joint_ids=None, env_ids=None
    ):
        self.data.joint_pos[:] = position


class _Selector:
    name = "robot"
    joint_ids = slice(None)

    def resolve(self, scene):
        self.joint_ids = [2]


class _CachedReset(ManagerTermBase):
    initialized = 0

    def __init__(self, cfg, env):
        super().__init__(env)
        type(self).initialized += 1
        self._entity = env.scene[cfg.params["asset_cfg"].name]
        self._indices = cfg.params["asset_cfg"].joint_ids

    def __call__(self, env, env_ids, asset_cfg):
        position = self._entity.data.joint_pos[:, self._indices] + self._env.step_dt
        self._entity.write_joint_state_to_sim(
            position, torch.zeros_like(position), env_ids=env_ids
        )


def test_class_event_resolves_before_init_and_keeps_the_live_env_unchanged(tmp_path):
    entity = _Entity()
    before = entity.data.joint_pos.clone()
    env = SimpleNamespace(
        scene=_Scene(robot=entity), num_envs=1, device="cpu", step_dt=0.02
    )
    cfg = EventTermCfg(
        func=_CachedReset, mode="reset", params={"asset_cfg": _Selector()}
    )
    _CachedReset.initialized = 0
    entry = serialize_event("reset", cfg, env, tmp_path)
    assert _CachedReset.initialized == 1
    assert cfg.params["asset_cfg"].joint_ids == slice(None)
    assert entry["write_targets"][0]["joint_ids"] == [2]
    torch.testing.assert_close(entity.data.joint_pos, before)
    session = ort.InferenceSession(
        str(tmp_path / entry["onnx"]), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        None, {"robot__joint_pos": torch.tensor([[1.0, 2.0, 3.0]]).numpy()}
    )
    torch.testing.assert_close(torch.from_numpy(outputs[0]), torch.tensor([[3.02]]))
    torch.testing.assert_close(torch.from_numpy(outputs[1]), torch.zeros(1, 1))


@pytest.mark.parametrize("change_output", [True, False])
@pytest.mark.parametrize("tensor_counter", [True, False])
def test_no_input_class_event_cannot_hide_mutating_history(
    tmp_path, change_output, tensor_counter
):
    class Counter(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(env)
            self.calls = torch.zeros(()) if tensor_counter else 0

        def __call__(self, env, env_ids):
            self.calls += 1
            position = torch.full((1, 1), float(self.calls) if change_output else 0.0)
            env.scene["robot"].write_joint_state_to_sim(
                position, torch.zeros_like(position)
            )

    env = SimpleNamespace(scene=_Scene(robot=_Entity()), num_envs=1, device="cpu")
    with pytest.raises(DivergentGraph, match="stateful event history"):
        serialize_event(
            "counter", EventTermCfg(func=Counter, mode="reset"), env, tmp_path
        )
    assert not list(tmp_path.rglob("*.onnx"))


class _PlainReset:
    def __init__(self, cfg, env):
        self._entity = env.scene[cfg.params["asset_cfg"].name]
        self._indices = cfg.params["asset_cfg"].joint_ids

    def __call__(self, env, env_ids, asset_cfg):
        position = self._entity.data.joint_pos[:, self._indices] + 1.0
        self._entity.write_joint_state_to_sim(
            position, torch.zeros_like(position), env_ids=env_ids
        )


def test_a_class_without_a_mjlab_base_traces_and_is_named_after_itself(tmp_path):
    env = SimpleNamespace(scene=_Scene(robot=_Entity()), num_envs=1, device="cpu")
    cfg = EventTermCfg(
        func=_PlainReset, mode="reset", params={"asset_cfg": _Selector()}
    )
    entry = serialize_event("plain", cfg, env, tmp_path)
    assert entry["func"].endswith(":_PlainReset")
    session = ort.InferenceSession(
        str(tmp_path / entry["onnx"]), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        None, {"robot__joint_pos": torch.tensor([[1.0, 2.0, 3.0]]).numpy()}
    )
    torch.testing.assert_close(torch.from_numpy(outputs[0]), torch.tensor([[4.0]]))


class _Offset(ManagerTermBase):
    built = 0

    def __init__(self, cfg, env):
        super().__init__(env)
        type(self).built += 1
        self._entity = env.scene["robot"]
        self._offset = 0.0

    def __call__(self, env, env_ids):
        position = self._entity.data.joint_pos[:, [0]] + self._offset
        self._entity.write_joint_state_to_sim(
            position, torch.zeros_like(position), env_ids=env_ids
        )


def test_the_instance_the_env_built_is_traced_and_no_second_one_is_made(tmp_path):
    env = SimpleNamespace(scene=_Scene(robot=_Entity()), num_envs=1, device="cpu")
    cfg = EventTermCfg(func=_Offset, mode="reset")
    _Offset.built = 0
    built = _Offset(cfg, env)
    built._offset = 5.0
    env.event_manager = SimpleNamespace(
        get_term_cfg=lambda name: SimpleNamespace(func=built)
    )
    entry = serialize_event("offset", cfg, env, tmp_path)
    assert _Offset.built == 1
    session = ort.InferenceSession(
        str(tmp_path / entry["onnx"]), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(
        None, {"robot__joint_pos": torch.tensor([[1.0, 2.0, 3.0]]).numpy()}
    )
    torch.testing.assert_close(torch.from_numpy(outputs[0]), torch.tensor([[6.0]]))


@pytest.mark.parametrize("bridged", [False, True])
def test_a_cached_sim_array_is_refused(tmp_path, bridged):
    wp = pytest.importorskip("warp")
    from mjlab.sim.sim_data import TorchArray, WarpBridge

    qpos = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
    cached = WarpBridge(SimpleNamespace(qpos=qpos)) if bridged else TorchArray(qpos)

    class KeepsSim(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(env)
            self._entity = env.scene["robot"]
            self._sim = cached

        def __call__(self, env, env_ids):
            qpos = self._sim.qpos if bridged else self._sim
            position = qpos[:, [0]] + 1.0
            self._entity.write_joint_state_to_sim(
                position, torch.zeros_like(position), env_ids=env_ids
            )

    env = SimpleNamespace(scene=_Scene(robot=_Entity()), num_envs=1, device="cpu")
    with pytest.raises(ValueError, match="unsupported live state in '_sim'"):
        serialize_event(
            "keeps_sim", EventTermCfg(func=KeepsSim, mode="reset"), env, tmp_path
        )


_MODEL = """
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="link" pos="0 0 0.2">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size="0.05 0.05 0.1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def test_on_a_real_env_entity_reads_stay_live_and_cached_sim_data_is_refused(
    tmp_path,
):
    mujoco = pytest.importorskip("mujoco")
    from mjswan.mjlab.env import build_single_entity_trace_env

    env = build_single_entity_trace_env(lambda: mujoco.MjSpec.from_string(_MODEL))

    class KeepsQpos(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(env)
            self._asset = env.scene["robot"]
            self._qpos = env.sim.data.qpos

        def __call__(self, env, env_ids):
            position = self._qpos[:, self._asset.indexing.joint_q_adr] + 0.1
            self._asset.write_joint_state_to_sim(
                position, torch.zeros_like(position), env_ids=env_ids
            )

    class ReadsEntity(ManagerTermBase):
        def __init__(self, cfg, env):
            super().__init__(env)
            self._asset = env.scene["robot"]

        def __call__(self, env, env_ids):
            position = self._asset.data.joint_pos + 0.1
            self._asset.write_joint_state_to_sim(
                position, torch.zeros_like(position), env_ids=env_ids
            )

    with pytest.raises(ValueError, match="unsupported live state in '_qpos'"):
        serialize_event(
            "keeps_qpos", EventTermCfg(func=KeepsQpos, mode="reset"), env, tmp_path
        )
    entry = serialize_event(
        "reads_entity", EventTermCfg(func=ReadsEntity, mode="reset"), env, tmp_path
    )
    assert [slot["input"] for slot in entry["input_slots"]] == ["robot__joint_pos"]
    session = ort.InferenceSession(
        str(tmp_path / entry["onnx"]), providers=["CPUExecutionProvider"]
    )
    outputs = session.run(None, {"robot__joint_pos": torch.tensor([[0.25]]).numpy()})
    torch.testing.assert_close(torch.from_numpy(outputs[0]), torch.tensor([[0.35]]))
