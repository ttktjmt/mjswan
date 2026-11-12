"""Class events initialize once with resolved params and capture cached entities."""

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
