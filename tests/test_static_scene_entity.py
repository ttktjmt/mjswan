"""Value terms can use scene origins and static entity layout during both passes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

from mjswan.compile.term import trace_term  # noqa: E402


class _Scene(dict):
    env_origins = torch.tensor([[10.0, 20.0, 30.0]])


def _relative_position(env):
    entity = env.scene["robot"]
    joints = entity.data.joint_pos[:, entity.indexing.joint_ids]
    if entity.is_fixed_base:
        return joints
    return torch.cat(
        (joints, entity.data.root_link_pos_w - env.scene.env_origins), dim=-1
    )


@pytest.mark.parametrize("fixed", [False, True])
def test_static_layout_and_origins_keep_live_observations_live(fixed):
    entity = SimpleNamespace(
        is_fixed_base=fixed,
        indexing=SimpleNamespace(joint_ids=torch.tensor([2, 0])),
        data=SimpleNamespace(
            joint_pos=torch.tensor([[0.1, 0.2, 0.3]]),
            root_link_pos_w=torch.tensor([[11.0, 22.0, 33.0]]),
        ),
    )
    env = SimpleNamespace(scene=_Scene(robot=entity), num_envs=1, device="cpu")
    exported = trace_term(_relative_position, {}, env, name="relative_position")
    session = ort.InferenceSession(
        exported.onnx_bytes, providers=["CPUExecutionProvider"]
    )
    # Replay new state, not just the trace inputs, so an accidentally baked live
    # position or incorrect joint order cannot pass.
    entity.data.joint_pos = torch.tensor([[0.4, 0.5, 0.6]])
    entity.data.root_link_pos_w = torch.tensor([[-2.0, -3.0, -4.0]])
    feeds = {
        name: getattr(entity.data, field).numpy()
        for name, (_entity, field) in zip(exported.input_names, exported.input_slots)
    }
    (actual,) = session.run([exported.output_name], feeds)
    torch.testing.assert_close(torch.from_numpy(actual), _relative_position(env))
