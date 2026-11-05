"""``export_checkpoint`` reads a checkpoint without running code from it."""

from __future__ import annotations

import pickle

import onnx
import pytest

from mjswan.mjlab.runner import PtOnnxExportContext, export_checkpoint

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

unpickled: list[str] = []


def _record(tag: str) -> str:
    unpickled.append(tag)
    return tag


class _Payload:
    """Unpickles by calling ``_record``, as a hostile checkpoint calls any callable."""

    def __reduce__(self):
        return _record, ("payload",)


class _Policy:
    obs_normalization = False


class _Runner:
    def __init__(self, model: onnx.ModelProto) -> None:
        self.alg = self
        self.model = model
        self.loaded: list[str] = []

    def get_policy(self) -> _Policy:
        return _Policy()

    def load(self, path: str, **_) -> None:
        self.loaded.append(path)

    def export_policy_to_onnx(self, out_dir: str, name: str) -> None:
        onnx.save(self.model, f"{out_dir}/{name}")


def _context(runner: _Runner) -> PtOnnxExportContext:
    return PtOnnxExportContext(
        env=None, runner=runner, joint_names=[], default_joint_pos=[], encoder_bias=[]
    )


def test_an_rsl_rl_checkpoint_exports(tmp_path, minimal_onnx):
    path = tmp_path / "model_100.pt"
    torch.save(
        {
            "actor_state_dict": {"mlp.0.weight": torch.zeros(2, 4)},
            "optimizer_state_dict": {"state": {}, "param_groups": [{"lr": 1e-3}]},
            "iter": 100,
            "infos": None,
        },
        path,
    )
    runner = _Runner(minimal_onnx)

    model = export_checkpoint(_context(runner), path)

    assert runner.loaded == [str(path)]
    assert model.graph.node[0].op_type == "Identity"


def test_a_checkpoint_that_calls_out_is_refused_before_the_runner_loads_it(
    tmp_path, minimal_onnx
):
    path = tmp_path / "model_100.pt"
    torch.save({"actor_state_dict": {}, "infos": _Payload()}, path)
    runner = _Runner(minimal_onnx)
    unpickled.clear()

    with pytest.raises(pickle.UnpicklingError):
        export_checkpoint(_context(runner), path)

    assert unpickled == []
    assert runner.loaded == []
