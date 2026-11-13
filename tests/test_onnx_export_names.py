"""Every traced graph output keeps the name it was exported under.

Layer: L2 (torch + onnx + onnxruntime, no mjlab or MuJoCo).

The TorchScript exporter names an output that constant-folds to an initializer after
the initializer (``"8"``), while `run_parity` and the browser look outputs up by the
requested name.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")

import onnx  # noqa: E402
from torch import nn  # noqa: E402

from mjswan.compile.export import (  # noqa: E402
    DivergentGraph,
    _export_onnx,
    check_against_module,
)

OUTPUTS = ["e__joint_state__position", "e__joint_state__velocity"]


class _Park(nn.Module):
    """A fixed position from a baked buffer and zero velocity: no input is read."""

    def __init__(self):
        super().__init__()
        self.register_buffer("default", torch.tensor([[0.0, 0.3]]))

    def forward(self, rand):
        pos = self.default[:, [0]].clone() + 0.05
        return pos, torch.zeros_like(pos)


class _Hold(nn.Module):
    """Reads a live input, so nothing folds."""

    def forward(self, joint_pos, rand):
        pos = joint_pos[:, [0]].clone()
        return pos, torch.zeros_like(pos)


def _export(module, example, inputs):
    return _export_onnx(
        module,
        example,
        input_names=inputs,
        output_names=OUTPUTS,
        batch_axis=[*inputs[:-1], *OUTPUTS],
        opset=17,
    )


def test_a_constant_folded_output_keeps_its_name():
    onnx_bytes = _export(_Park(), (torch.zeros(0),), ["rand"])
    graph = onnx.load_from_string(onnx_bytes).graph
    assert [o.name for o in graph.output] == OUTPUTS

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    position, velocity = session.run(OUTPUTS, {})
    assert position.reshape(-1).tolist() == pytest.approx([0.05])
    assert velocity.tolist() == [[0.0]]


def test_an_output_computed_from_an_input_keeps_its_name():
    module, example = _Hold(), (torch.zeros(1, 2), torch.zeros(0))
    onnx_bytes = _export(module, example, ["sim__qpos", "rand"])
    graph = onnx.load_from_string(onnx_bytes).graph
    assert [o.name for o in graph.output] == OUTPUTS


class _EmptyLocks(nn.Module):
    """A keyframe projection with no locked joints, a valid empty index write."""

    def __init__(self):
        super().__init__()
        self.qpos = torch.tensor([0.3, 0.5, 0.1])
        self.locked = torch.tensor([], dtype=torch.long)
        self.values = torch.tensor([])

    def forward(self, rand):
        position = self.qpos.repeat(1, 1)
        velocity = torch.zeros_like(position)
        position[..., self.locked] = self.values
        velocity[..., self.locked] = 0.0
        return position, velocity


def test_empty_index_writes_on_a_constant_reset_make_a_valid_graph():
    module = _EmptyLocks()
    example = (torch.zeros(0),)
    onnx_bytes = _export(module, example, ["rand"])
    model = onnx.load_from_string(onnx_bytes)
    onnx.checker.check_model(model, full_check=True)
    assert not model.graph.input
    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    actual = session.run(OUTPUTS, {})
    for got, expected in zip(actual, module(*example)):
        torch.testing.assert_close(torch.from_numpy(got), expected)


def test_structure_is_checked_before_skipping_random_graph_parity():
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("RandomUniform", [], ["random"], shape=[1]),
            onnx.helper.make_node("Add", ["", "random"], ["output"]),
        ],
        "invalid",
        [],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1])],
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 17)]
    )
    with pytest.raises(DivergentGraph, match="ONNX structural validation"):
        check_against_module(
            nn.Identity(),
            model.SerializeToString(),
            (),
            input_names=[],
            output_names=["output"],
            what="broken reset",
        )


def test_no_input_random_graph_is_not_frozen_to_one_draw():
    class Draw(nn.Module):
        def forward(self, rand):
            return torch.rand(1, 2), torch.zeros(1, 2)

    model = onnx.load_from_string(_export(Draw(), (torch.zeros(0),), ["rand"]))
    assert not model.graph.input
    assert any(node.op_type == "RandomUniform" for node in model.graph.node)


def test_no_input_changing_outputs_fail_numeric_parity():
    class Counter(nn.Module):
        calls = 0

        def forward(self, rand):
            self.calls += 1
            position = torch.full((1, 1), float(self.calls))
            return position, torch.zeros_like(position)

    module, example = Counter(), (torch.zeros(0),)
    onnx_bytes = _export(module, example, ["rand"])
    assert not onnx.load_from_string(onnx_bytes).graph.input
    with pytest.raises(DivergentGraph, match="disagrees with its own body"):
        check_against_module(
            module,
            onnx_bytes,
            example,
            input_names=["rand"],
            output_names=OUTPUTS,
            what="counter reset",
        )
